"""
src/models/pseudo_label.py
---------------------------
Conservative semi-supervised pseudo-labeling.

Strategy (safe version):
  1. Train strong base model on labeled data only
  2. Score high_conf_clean=1 unlabeled users
  3. Users with prob < NEG_THRESHOLD are pseudo-labeled as negatives
  4. Re-train with expanded dataset
  5. Repeat once (single iteration to prevent error accumulation)

We ONLY pseudo-label negatives:
  - FP pseudo-positives cascading into training inflate the cheater class
    with noisy signal, hurting precision and costing $300+ per mistake
  - Pseudo-negatives are safe: P(y=1 | prob<0.05) is extremely low

Label propagation on graph (optional):
  - Users whose 1-hop cheating density > 0.5 and are currently unlabeled
    can be soft-labeled. Use with caution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import PSEUDO_CFG


NEG_THRESHOLD = PSEUDO_CFG["neg_prob_threshold"]   # 0.05
MIN_CONF = PSEUDO_CFG["min_confidence"]            # 0.95 (high_conf_clean flag)


def generate_pseudo_negatives(
    model,
    X_unlabeled: pd.DataFrame,
    unlabeled_high_conf_mask: pd.Series,
    threshold: float = NEG_THRESHOLD,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Score unlabeled samples and return those confidently predicted as negative.

    Parameters
    ----------
    model                  : fitted classifier with predict_proba
    X_unlabeled            : features for unlabeled users
    unlabeled_high_conf_mask: boolean Series, True where high_conf_clean==1
    threshold              : max probability to accept as pseudo-negative

    Returns
    -------
    X_pseudo_neg : feature rows selected as pseudo-negatives
    y_pseudo_neg : all-zero Series (pseudo-label = not cheating)
    """
    # Only consider high_conf_clean users
    X_candidates = X_unlabeled[unlabeled_high_conf_mask]
    if len(X_candidates) == 0:
        print("[pseudo] No high_conf_clean unlabeled candidates — skipping.")
        return X_unlabeled.iloc[:0], pd.Series(dtype=float)

    probs = model.predict_proba(X_candidates)[:, 1]
    mask = probs < threshold
    X_pseudo = X_candidates[mask]
    y_pseudo = pd.Series(0, index=X_pseudo.index, dtype=int)

    print(
        f"[pseudo] Candidates: {len(X_candidates):,}  "
        f"| Pseudo-negatives (prob<{threshold}): {mask.sum():,}"
    )
    return X_pseudo, y_pseudo


def expand_training_data(
    X_labeled: pd.DataFrame,
    y_labeled: pd.Series,
    X_pseudo: pd.DataFrame,
    y_pseudo: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Concatenate real labels with pseudo-negative labels."""
    X_expanded = pd.concat([X_labeled, X_pseudo], axis=0)
    y_expanded = pd.concat([y_labeled, y_pseudo], axis=0)
    print(
        f"[pseudo] Expanded training: {len(X_labeled):,} → {len(X_expanded):,} rows  "
        f"| Cheating rate: {y_expanded.mean():.4f}"
    )
    return X_expanded, y_expanded


def graph_label_propagation(
    df: pd.DataFrame,
    cheat_rate_col: str = "neigh_cheat_rate_1hop",
    density_threshold: float = 0.6,
) -> pd.Series:
    """
    Soft graph-based propagation: mark unlabeled users as pseudo-positive
    candidates IF their 1-hop cheating density is extremely high.

    Returns a boolean Series (True = candidate for pseudo-positive).
    Use this only as additional evidence, NOT as a hard pseudo-label.
    """
    is_unlabeled = df["is_cheating"].isna()
    high_density = df[cheat_rate_col] > density_threshold
    candidates = is_unlabeled & high_density
    print(
        f"[graph-prop] Pseudo-positive candidates (density>{density_threshold}): "
        f"{candidates.sum():,}"
    )
    return candidates
