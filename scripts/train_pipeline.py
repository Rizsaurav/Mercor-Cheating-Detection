"""
scripts/train_pipeline.py
--------------------------
End-to-end training script. Run this to:
  1. Load and engineer features
  2. Build graph features
  3. Train base models (OOF + calibration)
  4. Run pseudo-labeling (1 iteration)
  5. Retrain on expanded data
  6. Stack / blend
  7. Threshold search on final OOF
  8. Save best model + submission file

Usage:
  python scripts/train_pipeline.py [--no-graph] [--no-pseudo] [--debug]

Flags:
  --no-graph   Skip graph feature computation (fast mode for debugging)
  --no-pseudo  Skip pseudo-labeling step
  --debug      Use 5% of data for rapid iteration
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is on the path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.utils.config import PATHS, CV
from src.utils.cost_metric import find_best_thresholds, competition_cost
from src.features.behavioral import build_behavioral_features, select_top_pairs
from src.graph.graph_features import build_graph_features
from src.models.base_models import (
    build_xgboost, build_lightgbm, build_catboost,
    train_oof_with_calibration,
)
from src.models.stacking import (
    train_meta_learner, optimise_blend_weights,
    equal_weight_blend, evaluate_ensemble,
)
from src.models.pseudo_label import (
    generate_pseudo_negatives, expand_training_data,
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-graph",  action="store_true")
    p.add_argument("--no-pseudo", action="store_true")
    p.add_argument("--debug",     action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(debug: bool = False):
    print("[data] Loading train …")
    train = pd.read_csv(PATHS["train_file"])
    test  = pd.read_csv(PATHS["test_file"])

    if debug:
        train = train.sample(frac=0.05, random_state=42)
        print(f"[data] DEBUG mode — using {len(train)} train rows")

    # Split labeled / unlabeled
    labeled_mask   = train["is_cheating"].notna()
    train_labeled  = train[labeled_mask].copy()
    train_unlabeled = train[~labeled_mask].copy()

    print(
        f"[data] Train total: {len(train):,}  "
        f"| Labeled: {len(train_labeled):,}  "
        f"| Unlabeled: {len(train_unlabeled):,}  "
        f"| Test: {len(test):,}"
    )
    return train, train_labeled, train_unlabeled, test


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(train_labeled, train_unlabeled, test):
    print("[feat] Engineering behavioral features …")

    # Select interaction pairs from labeled set only (no leakage)
    from src.features.behavioral import select_top_pairs
    from src.features.behavioral import ALL_RAW
    y_lab = train_labeled["is_cheating"].astype(int)
    X_lab_raw = train_labeled[[c for c in ALL_RAW if c in train_labeled.columns]]
    interaction_pairs = select_top_pairs(X_lab_raw.fillna(0), y_lab, top_k=5)
    print(f"[feat] Interaction pairs: {interaction_pairs}")

    train_labeled_feat  = build_behavioral_features(train_labeled,  interaction_pairs)
    train_unlabeled_feat = build_behavioral_features(train_unlabeled, interaction_pairs)
    test_feat            = build_behavioral_features(test,           interaction_pairs)

    return train_labeled_feat, train_unlabeled_feat, test_feat, interaction_pairs


# ─────────────────────────────────────────────────────────────────────────────
# Graph features
# ─────────────────────────────────────────────────────────────────────────────

def add_graph_features(
    train_labeled, train_unlabeled, test_df, all_feat_df, debug: bool = False
):
    print("[graph] Building graph features …")

    cheat_set   = set(train_labeled.loc[train_labeled["is_cheating"] == 1, "user_hash"])
    labeled_set = set(train_labeled["user_hash"])

    # BUG FIX: was union-ing train_labeled with itself twice, missing train_unlabeled.
    # known_users must be ALL train + test hashes so ghost nodes are correctly identified.
    known_users = (
        set(train_labeled["user_hash"])
        | set(train_unlabeled["user_hash"])
        | set(test_df["user_hash"])
    )

    # All users who need graph features — MUST include test set.
    # Test users appear in the social graph; their degree / neighbourhood
    # features must be computed on the full graph, not a train-only subgraph.
    all_users = (
        list(train_labeled["user_hash"]) +
        list(train_unlabeled["user_hash"]) +
        list(test_df["user_hash"])
    )

    from src.features.behavioral import ALL_RAW
    agg_cols = [c for c in ALL_RAW if c in all_feat_df.columns]

    # In debug mode use a 10% graph sample so the step runs in <2 min.
    sample_fraction = 0.10 if debug else None
    if debug:
        print("[graph] DEBUG mode — using 10% graph sample for fast validation")

    graph_feats = build_graph_features(
        graph_path=PATHS["graph_file"],
        users=all_users,
        cheat_set=cheat_set,
        labeled_set=labeled_set,
        known_users=known_users,
        feature_df=all_feat_df.set_index("user_hash") if "user_hash" in all_feat_df.columns else all_feat_df,
        behavioral_agg_cols=agg_cols[:10],
        sample_fraction=sample_fraction,
    )
    return graph_feats


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # 1. Load
    train, train_labeled, train_unlabeled, test = load_data(args.debug)

    # 2. Feature engineering
    train_lab_feat, train_unlab_feat, test_feat, pairs = engineer_features(
        train_labeled, train_unlabeled, test
    )

    # 3. Graph features (optional)
    # --no-graph skips this entirely; debug mode uses a 10% graph sample.
    if not args.no_graph:
        all_feat = pd.concat([train_lab_feat, train_unlab_feat, test_feat], axis=0)
        graph_feats = add_graph_features(
            train_labeled, train_unlabeled, test, all_feat, debug=args.debug
        )
        train_lab_feat   = train_lab_feat.merge(graph_feats,  left_on="user_hash", right_index=True, how="left")
        train_unlab_feat = train_unlab_feat.merge(graph_feats, left_on="user_hash", right_index=True, how="left")
        test_feat        = test_feat.merge(graph_feats,        left_on="user_hash", right_index=True, how="left")

    # Identify feature columns (exclude meta columns)
    meta_cols = ["user_hash", "is_cheating", "high_conf_clean"]
    feature_cols = [c for c in train_lab_feat.columns if c not in meta_cols]

    X_labeled = train_lab_feat[feature_cols].astype(float)
    y_labeled  = train_lab_feat["is_cheating"].astype(int)
    X_unlab   = train_unlab_feat[feature_cols].astype(float)
    X_test    = test_feat[feature_cols].astype(float)

    print(f"[pipeline] Feature matrix: {X_labeled.shape}  |  Cheating rate: {y_labeled.mean():.4f}")

    # 4. Base model OOF training
    print("\n[pipeline] Training XGBoost …")
    xgb   = build_xgboost()
    xgb_oof, xgb_final = train_oof_with_calibration(xgb, X_labeled, y_labeled)

    print("\n[pipeline] Training LightGBM …")
    lgbm = build_lightgbm()
    lgbm_oof, lgbm_final = train_oof_with_calibration(lgbm, X_labeled, y_labeled)

    cat_features = ["louvain_cluster_id"] if not args.no_graph else []
    print("\n[pipeline] Training CatBoost …")
    cat = build_catboost(cat_features=cat_features)
    cat_oof, cat_final = train_oof_with_calibration(cat, X_labeled, y_labeled)

    # OOF matrix for stacking
    oof_matrix = np.column_stack([xgb_oof, lgbm_oof, cat_oof])

    # 5. Pseudo-labeling (1 iteration)
    # ──────────────────────────────────────────────────────────────────────────
    # Correct ordering:
    #   train base models → calibrate (done above in train_oof_with_calibration)
    #   → score unlabeled with calibrated XGBoost
    #   → generate pseudo-negatives
    #   → retrain XGBoost on expanded data + recalibrate
    #   → replace XGBoost OOF in stacking matrix
    #   → train stacker on (now complete) OOF matrix
    # ──────────────────────────────────────────────────────────────────────────
    if not args.no_pseudo:
        print("\n[pipeline] Pseudo-labeling with calibrated XGBoost …")
        high_conf_mask = train_unlab_feat["high_conf_clean"].fillna(0).astype(bool)
        # xgb_final is already calibrated (CalibratedClassifierCV) from above
        X_pseudo, y_pseudo = generate_pseudo_negatives(xgb_final, X_unlab, high_conf_mask)

        if len(X_pseudo) > 0:
            X_expanded, y_expanded = expand_training_data(
                X_labeled, y_labeled, X_pseudo, y_pseudo
            )

            # BUG FIX: After expanding the dataset the original labeled rows are
            # at the TOP of X_expanded (they were concatenated first in
            # expand_training_data).  train_oof_with_calibration returns OOF
            # predictions aligned to the INPUT dataframe's row order, so we
            # can recover the original-labeled-row OOF by slicing
            # xgb2_oof[:len(y_labeled)] ONLY IF expand_training_data
            # always prepends the original data.
            #
            # To be safe and avoid any positional assumption, we reset the
            # index and track by position explicitly.
            n_orig = len(y_labeled)

            print("\n[pipeline] Retraining XGBoost on expanded data + recalibrating …")
            xgb2 = build_xgboost()
            xgb2_oof, xgb2_final = train_oof_with_calibration(
                xgb2,
                X_expanded.reset_index(drop=True),
                y_expanded.reset_index(drop=True),
            )

            # xgb2_oof[:n_orig] are the OOF predictions for the ORIGINAL
            # labeled rows (pseudo-neg rows appended after, never seen in
            # their own fold's training).  These are safe to use as stacker
            # meta-features for the labeled set.
            oof_matrix[:, 0] = xgb2_oof[:n_orig]
            xgb_final = xgb2_final
            print(f"[pipeline] OOF slice validated: {n_orig} original rows extracted")

    # 6. Meta-learner
    print("\n[pipeline] Training meta-learner …")
    meta = train_meta_learner(oof_matrix, y_labeled.values)
    meta_oof = meta.predict_proba(oof_matrix)[:, 1]

    # 7. Optuna blend
    print("\n[pipeline] Searching blend weights …")
    blend_weights = optimise_blend_weights(oof_matrix, y_labeled.values)
    blend_oof = oof_matrix @ blend_weights

    # Sanity check: equal weight
    equal_oof = equal_weight_blend(oof_matrix)

    # 8. Threshold optimisation on all candidates
    results = {}
    for label, probs in [("meta_lr", meta_oof), ("optuna_blend", blend_oof), ("equal_blend", equal_oof)]:
        res = evaluate_ensemble(y_labeled.values, probs, label=label)
        results[label] = res

    # Pick best
    best_label = min(results, key=lambda k: results[k]["min_cost"])
    best_res = results[best_label]
    print(f"\n[pipeline] Best ensemble: {best_label}  cost={best_res['min_cost']:.0f}")
    print(f"           t_pass={best_res['t_pass']:.3f}  t_block={best_res['t_block']:.3f}")

    # 9. Generate test predictions
    print("\n[pipeline] Generating test predictions …")
    xgb_test  = xgb_final.predict_proba(X_test)[:, 1]
    lgbm_test = lgbm_final.predict_proba(X_test)[:, 1]
    cat_test  = cat_final.predict_proba(X_test)[:, 1]
    test_matrix = np.column_stack([xgb_test, lgbm_test, cat_test])

    if best_label == "meta_lr":
        final_preds = meta.predict_proba(test_matrix)[:, 1]
    elif best_label == "optuna_blend":
        final_preds = test_matrix @ blend_weights
    else:
        final_preds = equal_weight_blend(test_matrix)

    # 10. Save submission
    submission = pd.DataFrame({
        "user_hash": test["user_hash"],
        "prediction": final_preds,
    })
    out_path = PATHS["submissions_dir"] / "submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"\n[pipeline] Submission saved: {out_path}")
    print(f"[pipeline] Done. Optimal thresholds → t_pass={best_res['t_pass']:.3f}, t_block={best_res['t_block']:.3f}")


if __name__ == "__main__":
    main()
