"""
src/models/stacking.py
-----------------------
Layer-2: Logistic Regression meta-learner over OOF predictions from base models.
Layer-3: Optuna-optimised blend weights with equal-weight baseline sanity check.

Why logistic regression for the meta-learner?
  - OOF predictions are already well-calibrated probabilities
  - LR stays well-calibrated itself (no further calibration needed)
  - Low variance; avoids overfitting the small meta-feature space
  - Coefficients are interpretable (which base model contributes most)

⚠  LEAKAGE RULE — strictly enforced by assertion in train_meta_learner:
  oof_matrix[i, :] must be the prediction for row i made by a model
  that was trained WITHOUT row i in its training fold.

  If any column of oof_matrix contains in-sample predictions (i.e. the
  base model saw row i during fit and is predicting row i), the meta-
  learner will learn to trust that model far too much, the stacker will
  dramatically overfit on the labeled set, and leaderboard performance
  will crater vs. local CV — the canonical sign of stacking leakage.

  The assertion `len(oof_matrix) == len(y)` is a necessary (not
  sufficient) check. Correctness depends on train_oof_with_calibration
  in base_models.py using StratifiedKFold and storing val predictions
  only from the fold where each row was held out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.utils.cost_metric import competition_cost, find_best_thresholds


# ─────────────────────────────────────────────────────────────────────────────
# Meta-learner
# ─────────────────────────────────────────────────────────────────────────────

def train_meta_learner(
    oof_matrix: np.ndarray,  # shape (N, n_base_models) — OOF probabilities ONLY
    y: np.ndarray,
    C: float = 1.0,
) -> LogisticRegression:
    """
    Fit a logistic regression meta-learner on OOF predictions.
    oof_matrix columns must be in the same order as test predictions later.

    Leakage guard
    -------------
    This function asserts oof_matrix.shape[0] == len(y).  That is a
    necessary condition for OOF correctness but not sufficient — caller
    is responsible for ensuring each oof_matrix[i, k] was produced by a
    fold that did NOT include row i in its training data.
    """
    # ── Hard guard: shape must match ─────────────────────────────────────────
    assert oof_matrix.shape[0] == len(y), (
        f"Leakage guard failed: oof_matrix has {oof_matrix.shape[0]} rows "
        f"but y has {len(y)} rows.  These must be identical — the stacker "
        f"requires one OOF prediction per labeled sample."
    )
    assert oof_matrix.ndim == 2, "oof_matrix must be 2-D (N_samples, N_models)"
    assert np.all((oof_matrix >= 0) & (oof_matrix <= 1)), (
        "oof_matrix values should be probabilities in [0, 1]. "
        "Did you pass raw scores instead of predict_proba output?"
    )
    # ─────────────────────────────────────────────────────────────────────────

    meta = LogisticRegression(C=C, solver="lbfgs", max_iter=1000)
    meta.fit(oof_matrix, y)
    print(f"[meta] LR coefficients: {meta.coef_[0]}")
    print(f"[meta] Model weights (softmax): {np.exp(meta.coef_[0]) / np.exp(meta.coef_[0]).sum()}")
    return meta


def predict_meta(
    meta: LogisticRegression,
    test_preds: np.ndarray,  # shape (N_test, n_base_models)
) -> np.ndarray:
    """Return meta-learner probability predictions on test set."""
    return meta.predict_proba(test_preds)[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Optuna blend weight search
# ─────────────────────────────────────────────────────────────────────────────

def optimise_blend_weights(
    oof_matrix: np.ndarray,
    y: np.ndarray,
    n_trials: int = 200,
    t_pass: float = 0.3,
    t_block: float = 0.7,
) -> np.ndarray:
    """
    Use Optuna to find blend weights that minimise competition cost on OOF.
    Weights are constrained to [0, 1] and normalised to sum to 1.

    Returns
    -------
    weights : 1-D array of shape (n_base_models,)
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("optuna not installed: pip install optuna")

    n_models = oof_matrix.shape[1]

    def objective(trial):
        raw_weights = np.array(
            [trial.suggest_float(f"w{i}", 0.0, 1.0) for i in range(n_models)]
        )
        weights = raw_weights / (raw_weights.sum() + 1e-9)
        blend = oof_matrix @ weights
        # Use fixed thresholds during optimisation for speed;
        # final threshold search runs separately on the best blend
        cost = competition_cost(y, blend, t_pass, t_block)
        return cost

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    raw = np.array([best[f"w{i}"] for i in range(n_models)])
    weights = raw / raw.sum()
    print(f"[blend] Best weights: {weights}")
    print(f"[blend] Best trial cost: {study.best_value:.2f}")
    return weights


def equal_weight_blend(oof_matrix: np.ndarray) -> np.ndarray:
    """Sanity-check baseline: simple mean of all base-model OOF predictions."""
    return oof_matrix.mean(axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_ensemble(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label: str = "ensemble",
) -> dict:
    """
    Run threshold grid search and report the minimum achievable cost.
    Returns a dict suitable for logging to W&B / MLflow.
    """
    t_pass, t_block, cost = find_best_thresholds(y_true, y_prob)
    print(
        f"[{label}] Best t_pass={t_pass:.3f}  t_block={t_block:.3f}  "
        f"min_cost={cost:.0f}"
    )
    return {"t_pass": t_pass, "t_block": t_block, "min_cost": cost, "label": label}
