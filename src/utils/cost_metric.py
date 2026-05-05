"""
src/utils/cost_metric.py
------------------------
Competition cost function + threshold grid search.
This is the ONLY evaluation metric that matters for the leaderboard.

Cost table:
  False Negative (missed cheater)   $600
  FP auto-block                     $300
  FP manual review                  $150
  TP requiring review               $5
  Correct auto-pass / auto-block    $0
"""
from __future__ import annotations

import numpy as np
from typing import Tuple

from src.utils.config import (
    COST_FN, COST_FP_BLOCK, COST_FP_REVIEW,
    COST_TP_REVIEW, THRESHOLD_GRID,
)


# ─────────────────────────────────────────────────────────────────────────────
# Core metric
# ─────────────────────────────────────────────────────────────────────────────

def competition_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    t_pass: float,
    t_block: float,
) -> float:
    """
    Compute the total operational cost given two decision thresholds.

    Decision rule:
        prob < t_pass   → auto-pass  (risky zone for FN)
        t_pass ≤ prob < t_block → manual review
        prob ≥ t_block  → auto-block (risky zone for FP-block)

    Parameters
    ----------
    y_true  : binary ground truth (0 = legit, 1 = cheater)
    y_prob  : model predicted probability of cheating
    t_pass  : lower threshold; below this → auto-pass
    t_block : upper threshold; at or above → auto-block

    Returns
    -------
    float : total cost (lower is better)
    """
    assert t_pass < t_block, "t_pass must be strictly less than t_block"

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    auto_pass  = y_prob < t_pass
    auto_block = y_prob >= t_block
    review     = ~auto_pass & ~auto_block

    fn_cost        = COST_FN        * ((y_true == 1) & auto_pass).sum()
    fp_block_cost  = COST_FP_BLOCK  * ((y_true == 0) & auto_block).sum()
    fp_review_cost = COST_FP_REVIEW * ((y_true == 0) & review).sum()
    tp_review_cost = COST_TP_REVIEW * ((y_true == 1) & review).sum()

    return float(fn_cost + fp_block_cost + fp_review_cost + tp_review_cost)


# ─────────────────────────────────────────────────────────────────────────────
# Threshold optimiser
# ─────────────────────────────────────────────────────────────────────────────

def find_best_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    step: float | None = None,
) -> Tuple[float, float, float]:
    """
    Grid-search over (t_pass, t_block) pairs to minimise competition cost.

    Returns
    -------
    (best_t_pass, best_t_block, best_cost)
    """
    cfg = THRESHOLD_GRID
    step = step or cfg["step"]

    t_pass_vals  = np.arange(cfg["t_pass_min"],  cfg["t_pass_max"],  step)
    t_block_vals = np.arange(cfg["t_block_min"], cfg["t_block_max"], step)

    best_cost, best_t1, best_t2 = float("inf"), None, None

    for t1 in t_pass_vals:
        for t2 in t_block_vals:
            if t1 >= t2:
                continue
            c = competition_cost(y_true, y_prob, t1, t2)
            if c < best_cost:
                best_cost, best_t1, best_t2 = c, t1, t2

    return best_t1, best_t2, best_cost


def normalised_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    t_pass: float = 0.3,
    t_block: float = 0.7,
) -> float:
    """
    Cost normalised by N so it's comparable across fold sizes.
    Useful as a per-fold CV metric.
    """
    raw = competition_cost(y_true, y_prob, t_pass, t_block)
    return raw / len(y_true)
