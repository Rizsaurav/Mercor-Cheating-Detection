"""
src/models/base_models.py
--------------------------
Layer-1 base learners.

Each model is wrapped in a lightweight class that:
  - exposes fit(X, y) / predict_proba(X)
  - is calibrated via isotonic regression on a held-out set
  - stores OOF (out-of-fold) predictions for stacking

Cost-aware design decisions:
  XGBoost  → scale_pos_weight = COST_RATIO (2.0) mirrors FN vs FP-block cost
  LightGBM → is_unbalance=True + cost-ratio as sample_weight alternative
  CatBoost → handles cluster IDs as categoricals natively

Calibration is CRITICAL: the leaderboard's threshold optimiser rewards
sharp, calibrated probabilities. Raw tree output probabilities are often
not well-calibrated; we always apply isotonic calibration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

from src.utils.config import COST_RATIO, CV
from src.utils.calibration import select_and_fit_calibrator, calibration_quality_report


class CalibratedModel:
    """
    Thin wrapper pairing a fitted base model with a separate calibrator.
    Avoids sklearn's clone() requirement (which breaks CatBoost).
    Exposes predict_proba(X) for downstream compatibility.
    """
    def __init__(self, model, calibrator):
        self.model      = model
        self.calibrator = calibrator

    def predict_proba(self, X) -> np.ndarray:
        raw = self.model.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw)
        return np.column_stack([1 - cal, cal])


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost
# ─────────────────────────────────────────────────────────────────────────────

def build_xgboost(scale_pos_weight: float = COST_RATIO, **kwargs):
    """
    XGBoost with cost-aware positive class weighting.
    scale_pos_weight = 2.0 reflects FN($600) / FP-block($300).
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("xgboost not installed: pip install xgboost")

    defaults = dict(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=CV["random_state"],
        n_jobs=-1,
        tree_method="hist",   # fast on CPU; switch to "gpu_hist" with CUDA
    )
    defaults.update(kwargs)
    return XGBClassifier(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM
# ─────────────────────────────────────────────────────────────────────────────

def build_lightgbm(**kwargs):
    """
    LightGBM with is_unbalance=True to handle class imbalance.
    Often faster than XGBoost for large feature sets.
    """
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        raise ImportError("lightgbm not installed: pip install lightgbm")

    defaults = dict(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        is_unbalance=True,
        random_state=CV["random_state"],
        n_jobs=-1,
        verbose=-1,
    )
    defaults.update(kwargs)
    return LGBMClassifier(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# CatBoost
# ─────────────────────────────────────────────────────────────────────────────

def build_catboost(cat_features: list[str] | None = None, **kwargs):
    """
    CatBoost handles categorical cluster IDs natively.
    Pass cat_features=['louvain_cluster_id'] when graph features are included.
    """
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        raise ImportError("catboost not installed: pip install catboost")

    defaults = dict(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        auto_class_weights="Balanced",
        eval_metric="Logloss",
        random_state=CV["random_state"],
        verbose=0,
        cat_features=cat_features or [],
    )
    defaults.update(kwargs)
    return CatBoostClassifier(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# OOF training + isotonic calibration
# ─────────────────────────────────────────────────────────────────────────────

def train_oof_with_calibration(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = CV["n_splits"],
    calibration_fraction: float = 0.1,
    random_state: int = CV["random_state"],
) -> tuple[np.ndarray, object]:
    """
    Stratified K-fold OOF training with per-fold isotonic calibration.

    Workflow per fold:
      1. Split train fold into fit_set (90%) and calibration_set (10%)
      2. Fit base model on fit_set
      3. Calibrate with isotonic regression on calibration_set
      4. Store OOF predictions on the val fold using calibrated model

    Parameters
    ----------
    model       : unfitted sklearn-compatible classifier
    X           : feature DataFrame
    y           : binary target Series
    n_splits    : number of CV folds
    calibration_fraction : fraction of train fold held for calibration
    random_state: for reproducibility

    Returns
    -------
    oof_probs   : out-of-fold probability array (len == len(X))
    fitted_model: model fitted on ALL training data with isotonic calibration
    """
    from sklearn.model_selection import train_test_split

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_probs_raw = np.zeros(len(X))   # raw (uncalibrated) for quality report
    oof_probs     = np.zeros(len(X))   # calibrated, used for stacking

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Hold back calibration set from train fold (10%)
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X_tr, y_tr,
            test_size=calibration_fraction,
            stratify=y_tr,
            random_state=random_state,
        )

        import copy
        fold_model = copy.deepcopy(model)

        # Fill NaN in categorical columns for CatBoost
        if hasattr(fold_model, 'get_params'):
            cat_cols = fold_model.get_params().get('cat_features') or []
            if cat_cols:
                X_fit = X_fit.copy()
                X_cal = X_cal.copy()
                X_val = X_val.copy()
                for col in cat_cols:
                    if col in X_fit.columns:
                        X_fit[col] = X_fit[col].fillna('missing').astype(str)
                        X_cal[col] = X_cal[col].fillna('missing').astype(str)
                        X_val[col] = X_val[col].fillna('missing').astype(str)

        fold_model.fit(X_fit, y_fit)

        raw_val  = fold_model.predict_proba(X_val)[:, 1]
        raw_cal  = fold_model.predict_proba(X_cal)[:, 1]
        y_cal_np = np.asarray(y_cal)

        # Auto-select calibrator based on positive count in calibration set
        cal = select_and_fit_calibrator(y_cal_np, raw_cal)

        oof_probs_raw[val_idx] = raw_val
        oof_probs[val_idx]     = cal.predict(raw_val)
        print(
            f"  [fold {fold+1}/{n_splits}] OOF stored for {len(val_idx)} samples "
            f"| cal={cal.__class__.__name__}"
        )

    # Calibration quality report on full OOF
    calibration_quality_report(y.values, oof_probs_raw, oof_probs)

    # ── Final model on ALL data ───────────────────────────────────────────────
    # Train on 85%, calibrate on 15% holdout.
    # Uses same auto-selector — with full dataset our positive count is high
    # enough for isotonic.
    X_final_fit, X_final_cal, y_final_fit, y_final_cal = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=random_state,
    )
    import copy
    final_model = copy.deepcopy(model)

    # Fill NaN in categorical columns for CatBoost (final model)
    if hasattr(final_model, 'get_params'):
        cat_cols = final_model.get_params().get('cat_features') or []
        if cat_cols:
            X_final_fit = X_final_fit.copy()
            X_final_cal = X_final_cal.copy()
            for col in cat_cols:
                if col in X_final_fit.columns:
                    X_final_fit[col] = X_final_fit[col].fillna('missing').astype(str)
                    X_final_cal[col] = X_final_cal[col].fillna('missing').astype(str)

    final_model.fit(X_final_fit, y_final_fit)

    raw_final_cal = final_model.predict_proba(X_final_cal)[:, 1]
    final_cal = select_and_fit_calibrator(
        np.asarray(y_final_cal), raw_final_cal
    )

    return oof_probs, CalibratedModel(final_model, final_cal)
