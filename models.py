"""
models.py
----------
Model progression (from project proposal):
  1. Logistic Regression (baseline, interpretable)
  2. Decision Tree
  3. Random Forest
  4. XGBoost / LightGBM (final model, handles missing data + class imbalance)
  5. Semi-supervised pseudo-labeling layer

Also includes:
  - OOF (Out-of-Fold) training with per-fold calibration
  - Calibration: Temperature → Platt → Isotonic (auto-selected)
  - CalibratedModel wrapper for inference
"""

import copy
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

# ─── Constants ────────────────────────────────────────────────────────────────
COST_RATIO = 2.0    # FN($600) / FP-block($300) → scale_pos_weight
N_SPLITS   = 5
RANDOM_STATE = 42


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_logistic_regression():
    """Baseline: Logistic Regression with class_weight='balanced'."""
    return LogisticRegression(
        C=1.0, class_weight="balanced", solver="lbfgs",
        max_iter=1000, random_state=RANDOM_STATE,
    )


def build_decision_tree():
    """Decision Tree with class_weight='balanced'."""
    return DecisionTreeClassifier(
        max_depth=8, min_samples_leaf=20,
        class_weight="balanced", random_state=RANDOM_STATE,
    )


def build_random_forest():
    """Random Forest with class_weight='balanced'."""
    return RandomForestClassifier(
        n_estimators=500, max_depth=10, min_samples_leaf=10,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )


def build_xgboost(scale_pos_weight=COST_RATIO):
    """XGBoost with cost-aware scale_pos_weight."""
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        gamma=0.1, scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", random_state=RANDOM_STATE,
        n_jobs=-1, tree_method="hist",
    )


def build_lightgbm():
    """LightGBM with is_unbalance=True."""
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=1000, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        is_unbalance=True, random_state=RANDOM_STATE,
        n_jobs=-1, verbose=-1,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

class TemperatureScaler:
    """Single-parameter calibration: divides log-odds by temperature T."""
    def __init__(self):
        self.T = 1.0

    def fit(self, y, probs):
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        logits = logit(probs)
        def nll(T):
            if T <= 0: return 1e9
            scaled = np.clip(expit(logits / T), 1e-7, 1 - 1e-7)
            return -np.mean(y * np.log(scaled) + (1 - y) * np.log(1 - scaled))
        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.T = result.x
        return self

    def predict(self, probs):
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        return expit(logit(probs) / self.T)


class PlattScaler:
    """Logistic regression on raw model scores (2-parameter calibration)."""
    def __init__(self):
        self._lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)

    def fit(self, y, probs):
        X = probs.reshape(-1, 1)
        self._lr.fit(X, y.astype(int))
        return self

    def predict(self, probs):
        return self._lr.predict_proba(probs.reshape(-1, 1))[:, 1]


class IsotonicScaler:
    """Non-parametric monotone calibration. Needs >200 positives."""
    def __init__(self):
        self._iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, y, probs):
        self._iso.fit(probs, y)
        return self

    def predict(self, probs):
        return self._iso.predict(probs)


def select_calibrator(y_cal, probs_cal):
    """Auto-select calibrator based on positive count."""
    n_pos = int(y_cal.sum())
    if n_pos < 100:
        cal = TemperatureScaler().fit(y_cal, probs_cal)
        method = "temperature"
    elif n_pos < 500:
        cal = PlattScaler().fit(y_cal, probs_cal)
        method = "platt"
    else:
        cal = IsotonicScaler().fit(y_cal, probs_cal)
        method = "isotonic"
    print(f"  [cal] {method} ({n_pos} positives)")
    return cal


# ═══════════════════════════════════════════════════════════════════════════════
# CALIBRATED MODEL WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class CalibratedModel:
    """Pairs a fitted model with a calibrator for inference."""
    def __init__(self, model, calibrator):
        self.model = model
        self.calibrator = calibrator

    def predict_proba(self, X):
        raw = self.model.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw)
        return np.column_stack([1 - cal, cal])


# ═══════════════════════════════════════════════════════════════════════════════
# OOF TRAINING WITH CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

def train_oof(model, X, y, n_splits=N_SPLITS, cal_fraction=0.1):
    """
    Stratified K-Fold OOF training with per-fold calibration.

    Returns:
      oof_probs   : calibrated out-of-fold predictions
      final_model : CalibratedModel fitted on all data
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_probs = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val = X.iloc[val_idx]

        # Hold back calibration set
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X_tr, y_tr, test_size=cal_fraction, stratify=y_tr, random_state=RANDOM_STATE,
        )

        fold_model = copy.deepcopy(model)
        X_fit_clean = X_fit.fillna(0)
        X_cal_clean = X_cal.fillna(0)
        X_val_clean = X_val.fillna(0)

        fold_model.fit(X_fit_clean, y_fit)

        raw_val = fold_model.predict_proba(X_val_clean)[:, 1]
        raw_cal = fold_model.predict_proba(X_cal_clean)[:, 1]

        cal = select_calibrator(np.asarray(y_cal), raw_cal)
        oof_probs[val_idx] = np.clip(cal.predict(raw_val), 0.0, 1.0)
        print(f"  [fold {fold+1}/{n_splits}] OOF stored for {len(val_idx)} samples")

    # Final model on all data
    X_final_fit, X_final_cal, y_final_fit, y_final_cal = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE,
    )
    final_model = copy.deepcopy(model)
    final_model.fit(X_final_fit.fillna(0), y_final_fit)
    raw_final = final_model.predict_proba(X_final_cal.fillna(0))[:, 1]
    final_cal = select_calibrator(np.asarray(y_final_cal), raw_final)

    return oof_probs, CalibratedModel(final_model, final_cal)


# ═══════════════════════════════════════════════════════════════════════════════
# PSEUDO-LABELING (SEMI-SUPERVISED)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_pseudo_negatives(model, X_unlabeled, high_conf_mask, threshold=0.05):
    """
    Score unlabeled high-confidence-clean users.
    Those with prob < threshold become pseudo-negatives.
    Only pseudo-label negatives — pseudo-positives are too risky ($600 per FN).
    """
    X_candidates = X_unlabeled[high_conf_mask]
    if len(X_candidates) == 0:
        print("[pseudo] No high_conf_clean candidates — skipping.")
        return X_unlabeled.iloc[:0], pd.Series(dtype=float)

    probs = model.predict_proba(X_candidates.fillna(0))[:, 1]
    mask = probs < threshold
    X_pseudo = X_candidates[mask]
    y_pseudo = pd.Series(0, index=X_pseudo.index, dtype=int)

    print(f"[pseudo] Candidates: {len(X_candidates):,}  →  Pseudo-negatives: {mask.sum():,}")
    return X_pseudo, y_pseudo


def expand_training_data(X_labeled, y_labeled, X_pseudo, y_pseudo):
    """Concatenate real labels with pseudo-negatives."""
    X_expanded = pd.concat([X_labeled, X_pseudo], axis=0)
    y_expanded = pd.concat([y_labeled, y_pseudo], axis=0)
    print(f"[pseudo] Expanded: {len(X_labeled):,} → {len(X_expanded):,}  |  Cheat rate: {y_expanded.mean():.4f}")
    return X_expanded, y_expanded
