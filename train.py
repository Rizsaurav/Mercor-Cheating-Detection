"""
train.py
---------
Main training pipeline. Runs end-to-end:
  1. Load and preprocess data
  2. Engineer behavioral + graph features
  3. Train model progression: LogReg → DT → RF → XGBoost → LightGBM
  4. Calibrate probabilities
  5. Blend / stack predictions
  6. Threshold optimization
  7. (Optional) Pseudo-labeling
  8. Save submission

Usage:
  python train.py                        # full run with graph features
  python train.py --no-graph             # skip graph (fast)
  python train.py --no-graph --debug     # 5% sample, no graph (fastest)
  python train.py --no-pseudo            # skip pseudo-labeling
"""
import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path

from preprocessing import (
    load_data, build_behavioral_features, select_top_pairs,
    build_graph_features, scale_for_logistic, ALL_RAW, DATA_DIR,
)
from models import (
    build_logistic_regression, build_decision_tree, build_random_forest,
    build_xgboost, build_lightgbm, train_oof,
    generate_pseudo_negatives, expand_training_data,
)
from evaluation import (
    evaluate_ensemble, find_best_thresholds, compute_metrics,
    print_classification_report, three_zone_report,
)


def parse_args():
    p = argparse.ArgumentParser(description="Mercor Cheating Detection Pipeline")
    p.add_argument("--no-graph",  action="store_true", help="Skip graph features")
    p.add_argument("--no-pseudo", action="store_true", help="Skip pseudo-labeling")
    p.add_argument("--debug",     action="store_true", help="Use 5%% data sample")
    p.add_argument("--data-dir",  type=str, default="data", help="Path to data directory")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    os.makedirs("outputs/submissions", exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # 1. LOAD DATA
    # ──────────────────────────────────────────────────────────────────────────
    train, train_labeled, train_unlabeled, test = load_data(data_dir, debug=args.debug)

    # ──────────────────────────────────────────────────────────────────────────
    # 2. FEATURE ENGINEERING
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[feat] Engineering behavioral features …")
    y_lab = train_labeled["is_cheating"].astype(int)
    X_lab_raw = train_labeled[[c for c in ALL_RAW if c in train_labeled.columns]]
    interaction_pairs = select_top_pairs(X_lab_raw.fillna(0), y_lab, top_k=5)
    print(f"[feat] Interaction pairs: {interaction_pairs}")

    train_lab_feat   = build_behavioral_features(train_labeled, interaction_pairs)
    train_unlab_feat = build_behavioral_features(train_unlabeled, interaction_pairs)
    test_feat        = build_behavioral_features(test, interaction_pairs)

    # ──────────────────────────────────────────────────────────────────────────
    # 3. GRAPH FEATURES (optional)
    # ──────────────────────────────────────────────────────────────────────────
    if not args.no_graph:
        sample_frac = 0.10 if args.debug else None
        graph_feats = build_graph_features(
            data_dir, train_labeled, train_unlabeled, test, sample_fraction=sample_frac
        )
        if graph_feats is not None:
            train_lab_feat   = train_lab_feat.merge(graph_feats, left_on="user_hash", right_index=True, how="left")
            train_unlab_feat = train_unlab_feat.merge(graph_feats, left_on="user_hash", right_index=True, how="left")
            test_feat        = test_feat.merge(graph_feats, left_on="user_hash", right_index=True, how="left")

    # Identify feature columns
    meta_cols = ["user_hash", "is_cheating", "high_conf_clean"]
    feature_cols = [c for c in train_lab_feat.columns if c not in meta_cols]

    X_labeled = train_lab_feat[feature_cols].astype(float)
    y_labeled = train_lab_feat["is_cheating"].astype(int)
    X_unlab   = train_unlab_feat[feature_cols].astype(float)
    X_test    = test_feat[feature_cols].astype(float)

    print(f"\n[pipeline] Feature matrix: {X_labeled.shape}  |  Cheating rate: {y_labeled.mean():.4f}")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. MODEL PROGRESSION
    # ──────────────────────────────────────────────────────────────────────────
    results = {}

    # 4a. Logistic Regression (baseline)
    print("\n" + "="*60)
    print("[pipeline] 1/5  Logistic Regression (baseline)")
    print("="*60)
    X_scaled, _, _ = scale_for_logistic(X_labeled.fillna(0), X_test.fillna(0))
    lr_oof, lr_final = train_oof(build_logistic_regression(), X_scaled, y_labeled)
    results["LogReg"] = evaluate_ensemble(y_labeled.values, lr_oof, "LogReg")

    # 4b. Decision Tree
    print("\n" + "="*60)
    print("[pipeline] 2/5  Decision Tree")
    print("="*60)
    dt_oof, dt_final = train_oof(build_decision_tree(), X_labeled, y_labeled)
    results["DecisionTree"] = evaluate_ensemble(y_labeled.values, dt_oof, "DecisionTree")

    # 4c. Random Forest
    print("\n" + "="*60)
    print("[pipeline] 3/5  Random Forest")
    print("="*60)
    rf_oof, rf_final = train_oof(build_random_forest(), X_labeled, y_labeled)
    results["RandomForest"] = evaluate_ensemble(y_labeled.values, rf_oof, "RandomForest")

    # 4d. XGBoost
    print("\n" + "="*60)
    print("[pipeline] 4/5  XGBoost")
    print("="*60)
    xgb_oof, xgb_final = train_oof(build_xgboost(), X_labeled, y_labeled)
    results["XGBoost"] = evaluate_ensemble(y_labeled.values, xgb_oof, "XGBoost")

    # 4e. LightGBM
    print("\n" + "="*60)
    print("[pipeline] 5/5  LightGBM")
    print("="*60)
    lgbm_oof, lgbm_final = train_oof(build_lightgbm(), X_labeled, y_labeled)
    results["LightGBM"] = evaluate_ensemble(y_labeled.values, lgbm_oof, "LightGBM")

    # ──────────────────────────────────────────────────────────────────────────
    # 5. ENSEMBLE: EQUAL-WEIGHT BLEND OF XGBoost + LightGBM
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("[pipeline] Ensemble: XGBoost + LightGBM blend")
    print("="*60)
    oof_blend = (xgb_oof + lgbm_oof) / 2
    results["Ensemble"] = evaluate_ensemble(y_labeled.values, oof_blend, "Ensemble")

    # ──────────────────────────────────────────────────────────────────────────
    # 6. PSEUDO-LABELING (optional)
    # ──────────────────────────────────────────────────────────────────────────
    if not args.no_pseudo and "high_conf_clean" in train_unlab_feat.columns:
        print("\n" + "="*60)
        print("[pipeline] Semi-supervised pseudo-labeling")
        print("="*60)
        high_conf_mask = train_unlab_feat["high_conf_clean"].fillna(0).astype(bool)
        X_pseudo, y_pseudo = generate_pseudo_negatives(xgb_final, X_unlab, high_conf_mask)

        if len(X_pseudo) > 0:
            X_expanded, y_expanded = expand_training_data(X_labeled, y_labeled, X_pseudo, y_pseudo)
            n_orig = len(y_labeled)

            xgb2_oof, xgb2_final = train_oof(
                build_xgboost(),
                X_expanded.reset_index(drop=True),
                y_expanded.reset_index(drop=True),
            )
            # Replace XGBoost OOF with retrained version
            xgb_oof_new = xgb2_oof[:n_orig]
            oof_blend = (xgb_oof_new + lgbm_oof) / 2
            xgb_final = xgb2_final
            results["Ensemble+Pseudo"] = evaluate_ensemble(y_labeled.values, oof_blend, "Ensemble+Pseudo")

    # ──────────────────────────────────────────────────────────────────────────
    # 7. SELECT BEST & GENERATE SUBMISSION
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("[pipeline] Model Comparison")
    print("="*60)
    print(f"{'Model':<20} {'t_pass':>8} {'t_block':>8} {'Cost':>12}")
    print("-" * 50)
    for name, res in results.items():
        print(f"{name:<20} {res['t_pass']:>8.3f} {res['t_block']:>8.3f} ${res['min_cost']:>10,.0f}")

    best_name = min(results, key=lambda k: results[k]["min_cost"])
    best = results[best_name]
    print(f"\n✓ Best model: {best_name}  |  Cost: ${best['min_cost']:,.0f}")

    # Three-zone breakdown for best model
    if best_name == "Ensemble" or best_name == "Ensemble+Pseudo":
        best_oof = oof_blend
    elif best_name == "XGBoost":
        best_oof = xgb_oof
    elif best_name == "LightGBM":
        best_oof = lgbm_oof
    else:
        best_oof = oof_blend  # fallback to ensemble

    three_zone_report(y_labeled.values, best_oof, best["t_pass"], best["t_block"])

    # Secondary metrics
    metrics = compute_metrics(y_labeled.values, best_oof, threshold=0.5)
    print(f"\n[metrics] AUC-ROC: {metrics['auc_roc']:.4f}  |  F1: {metrics['f1']:.4f}  "
          f"|  Precision: {metrics['precision']:.4f}  |  Recall: {metrics['recall']:.4f}")

    # ──────────────────────────────────────────────────────────────────────────
    # 8. GENERATE TEST PREDICTIONS
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[pipeline] Generating test predictions …")
    xgb_test  = xgb_final.predict_proba(X_test.fillna(0))[:, 1]
    lgbm_test = lgbm_final.predict_proba(X_test.fillna(0))[:, 1]
    final_preds = (xgb_test + lgbm_test) / 2

    submission = pd.DataFrame({
        "user_hash": test["user_hash"],
        "prediction": final_preds,
    })
    out_path = "outputs/submissions/submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"[pipeline] Submission saved: {out_path}")
    print(f"[pipeline] Done. Best: {best_name}  t_pass={best['t_pass']:.3f}  t_block={best['t_block']:.3f}")


if __name__ == "__main__":
    main()
