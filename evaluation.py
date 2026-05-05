"""
evaluation.py
--------------
Evaluation methods (from project proposal):
  1. Custom cost function ($600 FN, $300 FP-block, $150 FP-review, $5 TP-review)
  2. Secondary metrics: AUC-ROC, Precision, Recall, F1-Score
  3. Confusion matrix across all three decision zones
  4. Stratified K-Fold Cross-Validation
  5. Threshold sweep: find optimal t_pass / t_block to minimize total cost
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

# ─── Cost Constants ───────────────────────────────────────────────────────────
COST_FN        = 600   # Missed cheater (auto-passed)
COST_FP_BLOCK  = 300   # Legit user auto-blocked
COST_FP_REVIEW = 150   # Legit user sent to manual review
COST_TP_REVIEW = 5     # Cheater correctly flagged for review


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CUSTOM COST FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def competition_cost(y_true, y_prob, t_pass, t_block):
    """
    Compute total operational cost given two decision thresholds.

    Decision rule:
      prob < t_pass    → auto-pass  (risk: missed cheaters → $600 each)
      t_pass ≤ prob < t_block  → manual review
      prob ≥ t_block   → auto-block (risk: blocking legit → $300 each)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    auto_pass  = y_prob < t_pass
    auto_block = y_prob >= t_block
    review     = ~auto_pass & ~auto_block

    cost = (
        COST_FN        * ((y_true == 1) & auto_pass).sum()
      + COST_FP_BLOCK  * ((y_true == 0) & auto_block).sum()
      + COST_FP_REVIEW * ((y_true == 0) & review).sum()
      + COST_TP_REVIEW * ((y_true == 1) & review).sum()
    )
    return float(cost)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. THRESHOLD SWEEP
# ═══════════════════════════════════════════════════════════════════════════════

def find_best_thresholds(y_true, y_prob, step=0.01):
    """
    Grid-search over (t_pass, t_block) pairs to minimize competition cost.
    Returns: (best_t_pass, best_t_block, best_cost)
    """
    t_pass_vals  = np.arange(0.01, 0.50, step)
    t_block_vals = np.arange(0.50, 0.99, step)

    best_cost, best_t1, best_t2 = float("inf"), None, None
    for t1 in t_pass_vals:
        for t2 in t_block_vals:
            if t1 >= t2:
                continue
            c = competition_cost(y_true, y_prob, t1, t2)
            if c < best_cost:
                best_cost, best_t1, best_t2 = c, t1, t2

    return best_t1, best_t2, best_cost


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SECONDARY METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_prob, threshold=0.5):
    """Compute AUC-ROC, Precision, Recall, F1 at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc_roc":   roc_auc_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }


def print_classification_report(y_true, y_prob, threshold=0.5):
    """Print sklearn classification report + confusion matrix."""
    y_pred = (y_prob >= threshold).astype(int)
    print("\n" + classification_report(y_true, y_pred, target_names=["Legit", "Cheater"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. THREE-ZONE CONFUSION MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

def three_zone_report(y_true, y_prob, t_pass, t_block):
    """
    Breakdown of predictions across the three decision zones.
    Shows count and cost contribution for each cell.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    auto_pass  = y_prob < t_pass
    auto_block = y_prob >= t_block
    review     = ~auto_pass & ~auto_block

    zones = {"auto_pass": auto_pass, "review": review, "auto_block": auto_block}
    costs = {"auto_pass": {0: 0, 1: COST_FN},
             "review":    {0: COST_FP_REVIEW, 1: COST_TP_REVIEW},
             "auto_block":{0: COST_FP_BLOCK, 1: 0}}

    print(f"\n{'Zone':<15} {'Legit':>8} {'Cheater':>8} {'Total':>8} {'Cost':>10}")
    print("-" * 55)
    total_cost = 0
    for zone_name, mask in zones.items():
        n_legit  = ((y_true == 0) & mask).sum()
        n_cheat  = ((y_true == 1) & mask).sum()
        zone_cost = costs[zone_name][0] * n_legit + costs[zone_name][1] * n_cheat
        total_cost += zone_cost
        print(f"{zone_name:<15} {n_legit:>8,} {n_cheat:>8,} {mask.sum():>8,} ${zone_cost:>9,}")
    print("-" * 55)
    print(f"{'TOTAL':<15} {(y_true==0).sum():>8,} {(y_true==1).sum():>8,} {len(y_true):>8,} ${total_cost:>9,}")
    return total_cost


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ENSEMBLE EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_ensemble(y_true, y_prob, label="ensemble"):
    """Run threshold search and report minimum achievable cost."""
    t_pass, t_block, cost = find_best_thresholds(y_true, y_prob)
    print(f"[{label}] t_pass={t_pass:.3f}  t_block={t_block:.3f}  min_cost=${cost:,.0f}")
    return {"t_pass": t_pass, "t_block": t_block, "min_cost": cost}
