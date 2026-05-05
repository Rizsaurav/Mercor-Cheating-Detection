"""
src/utils/calibration.py
-------------------------
Calibration methods ordered by stability with sparse positive classes:

  1. Temperature scaling  — 1 param, works even with few positives per fold
  2. Platt scaling        — 2 params (logistic on scores), stable & parametric
  3. Isotonic regression  — non-parametric, needs >200 positives to avoid overfit

Selection rule (applied automatically by `select_and_fit_calibrator`):
  positives < 100  → temperature scaling only
  100–500          → Platt scaling
  > 500            → isotonic (our case: ~2,755 per fold ✓)

BUT: even with enough positives, if t_block stays ≥ 0.95 after isotonic,
it means the discrimination problem is pre-calibration (model doesn't
separate 0.5–0.9 region). The diagnostic is in `calibration_quality_report`.

Why temperature scaling fixes tree overconfidence
--------------------------------------------------
Trees output leaf purity as probability. A leaf trained on 8/10 cheaters
outputs 0.8 regardless of leaf size or regularisation. Temperature T > 1
compresses all probabilities toward 0.5; T < 1 sharpens them. We want
T slightly > 1 to de-overconfide the high-probability tail.

Usage
-----
  cal = select_and_fit_calibrator(y_cal, raw_probs_cal)
  calibrated_test_probs = cal.predict(raw_probs_test)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


# ─────────────────────────────────────────────────────────────────────────────
# Temperature Scaling
# ─────────────────────────────────────────────────────────────────────────────

class TemperatureScaler:
    """
    Single-parameter calibration: divides log-odds by temperature T.

    T > 1 → softens probabilities (fixes overconfident high-prob predictions)
    T < 1 → sharpens probabilities (rarely needed with trees)
    T = 1 → identity (no calibration)

    T is fit by minimising NLL on the calibration set.
    """

    def __init__(self):
        self.T: float = 1.0

    def fit(self, y: np.ndarray, probs: np.ndarray) -> TemperatureScaler:
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        logits = logit(probs)   # log(p / (1-p))

        def nll(T):
            if T <= 0:
                return 1e9
            scaled = expit(logits / T)
            scaled = np.clip(scaled, 1e-7, 1 - 1e-7)
            return -np.mean(y * np.log(scaled) + (1 - y) * np.log(1 - scaled))

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.T = result.x
        print(f"[cal] TemperatureScaler: T={self.T:.4f}  NLL={result.fun:.6f}")
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        return expit(logit(probs) / self.T)

    def __repr__(self):
        return f"TemperatureScaler(T={self.T:.4f})"


# ─────────────────────────────────────────────────────────────────────────────
# Platt Scaling
# ─────────────────────────────────────────────────────────────────────────────

class PlattScaler:
    """
    Logistic regression on raw model scores (probabilities).
    More flexible than temperature (2 params: intercept + slope).
    Stable with as few as ~100 positives. Uses Platt's original formulation
    with label smoothing to avoid overconfidence on calibration set.
    """

    def __init__(self):
        self._lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)

    def fit(self, y: np.ndarray, probs: np.ndarray) -> PlattScaler:
        n_pos = y.sum()
        n_neg = (1 - y).sum()
        # Platt label smoothing: avoids fitting 0/1 hard targets
        y_smooth = np.where(y == 1,
                            (n_pos + 1) / (n_pos + 2),
                            1 / (n_neg + 2))
        X = probs.reshape(-1, 1)
        self._lr.fit(X, (y_smooth > 0.5).astype(int))
        print(
            f"[cal] PlattScaler: coef={self._lr.coef_[0][0]:.4f}  "
            f"intercept={self._lr.intercept_[0]:.4f}"
        )
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(probs.reshape(-1, 1))[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Isotonic Calibration (thin wrapper for consistency)
# ─────────────────────────────────────────────────────────────────────────────

class IsotonicScaler:
    """
    Non-parametric monotone calibration. Needs >200 positives to avoid
    overfitting the calibration curve in the tails.
    """

    def __init__(self):
        self._iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, y: np.ndarray, probs: np.ndarray) -> IsotonicScaler:
        self._iso.fit(probs, y)
        print(f"[cal] IsotonicScaler: fit on {y.sum():.0f} positives")
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        return self._iso.predict(probs)


# ─────────────────────────────────────────────────────────────────────────────
# Auto-selector
# ─────────────────────────────────────────────────────────────────────────────

def select_and_fit_calibrator(
    y_cal: np.ndarray,
    probs_cal: np.ndarray,
    min_pos_for_platt: int = 100,
    min_pos_for_isotonic: int = 500,
    force: str | None = None,
) -> TemperatureScaler | PlattScaler | IsotonicScaler:
    """
    Select and fit the appropriate calibrator based on positive count.

    Parameters
    ----------
    y_cal               : binary labels on calibration set
    probs_cal           : raw model probabilities on calibration set
    min_pos_for_platt   : minimum positives to use Platt (default 100)
    min_pos_for_isotonic: minimum positives to use isotonic (default 500)
    force               : 'temperature' | 'platt' | 'isotonic' to override

    Returns
    -------
    Fitted calibrator with a .predict(probs) method
    """
    n_pos = int(y_cal.sum())
    print(f"[cal] Calibration set: {len(y_cal):,} samples, {n_pos:,} positives")

    if force == "temperature" or n_pos < min_pos_for_platt:
        method = "temperature"
    elif force == "platt" or n_pos < min_pos_for_isotonic:
        method = "platt"
    else:
        method = "isotonic"

    if force:
        method = force

    print(f"[cal] Selected method: {method}")

    if method == "temperature":
        return TemperatureScaler().fit(y_cal, probs_cal)
    elif method == "platt":
        return PlattScaler().fit(y_cal, probs_cal)
    else:
        return IsotonicScaler().fit(y_cal, probs_cal)


# ─────────────────────────────────────────────────────────────────────────────
# Quality diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def calibration_quality_report(
    y_true: np.ndarray,
    probs_before: np.ndarray,
    probs_after: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compare calibration error (ECE) before and after.
    Also prints the 90th-percentile probability for each class —
    if cheat p90 < 0.90 after calibration, the discrimination
    problem is pre-calibration (need better features, not better calibration).

    Returns dict with ece_before, ece_after, cheat_p90_before, cheat_p90_after.
    """
    def ece(y, p, n_bins):
        bins = np.linspace(0, 1, n_bins + 1)
        total = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (p >= lo) & (p < hi)
            if mask.sum() == 0:
                continue
            acc = y[mask].mean()
            conf = p[mask].mean()
            total += mask.sum() * abs(acc - conf)
        return total / len(y)

    ece_b = ece(y_true, probs_before, n_bins)
    ece_a = ece(y_true, probs_after,  n_bins)

    cheat_mask = y_true == 1
    cp90_b = np.percentile(probs_before[cheat_mask], 90)
    cp90_a = np.percentile(probs_after[cheat_mask],  90)

    print(f"\n[cal] ECE before: {ece_b:.4f}  →  after: {ece_a:.4f}")
    print(f"[cal] Cheater p90 before: {cp90_b:.3f}  →  after: {cp90_a:.3f}")

    if cp90_a < 0.85:
        print("[cal] ⚠ Cheater p90 still below 0.85 after calibration.")
        print("[cal]   This is a DISCRIMINATION problem — need better features,")
        print("[cal]   not better calibration. Graph features are the fix.")
    else:
        print("[cal] ✓ Cheater p90 > 0.85 — calibration is working correctly.")

    return {
        "ece_before": ece_b, "ece_after": ece_a,
        "cheat_p90_before": cp90_b, "cheat_p90_after": cp90_a,
    }
