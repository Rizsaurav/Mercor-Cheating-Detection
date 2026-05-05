"""
src/features/behavioral.py
---------------------------
All feature engineering on the 18 raw behavioral columns.

Engineering checklist:
  [x] Imputation (median + binary missingness flag)
  [x] Log-transform feature_010 (range 0–39M)
  [x] Z-score outlier binary flags for skewed numerics
  [x] Top-5 pairwise interaction terms (MI-selected)
  [x] Ratio features capturing behavioral rates
  [x] feature_015 tail flag (>80) + 5-bin ordinal — EDA secondary-bump signal
  [x] feature_013 × feature_015 interaction — highest-precision cheater profile

Note on feature_013 (from EDA leakage check):
  cheat_rate when f013=0: 35.7%  (flag absent → higher risk)
  cheat_rate when f013=1: 16.4%  (flag present → lower risk; protective/exemption flag)
  NOT leakage — the flag REDUCES cheating probability, consistent with a legitimate
  exemption or verified-identity marker. Safe to use in training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import warnings

warnings.filterwarnings("ignore")

# Features with high missing % — impute with median + add flag
HIGH_MISSING = [
    "feature_001", "feature_002", "feature_003",
    "feature_007", "feature_008", "feature_009",
    "feature_010", "feature_011", "feature_012",
    "feature_013", "feature_014", "feature_017", "feature_018",
]

# Binary features (no log transform / Z-score)
BINARY_FEATURES = ["feature_007", "feature_011", "feature_013", "feature_014"]

# Highly skewed: needs log1p
LOG_FEATURES = ["feature_010", "feature_016", "feature_015"]

ALL_RAW = [f"feature_{i:03d}" for i in range(1, 19)]
NUMERIC_RAW = [f for f in ALL_RAW if f not in BINARY_FEATURES]


# ─────────────────────────────────────────────────────────────────────────────

def impute_and_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Median-impute HIGH_MISSING features and add binary flag columns
    indicating whether the original value was missing.
    Operates in-place on a copy.
    """
    df = df.copy()
    for col in HIGH_MISSING:
        if col not in df.columns:
            continue
        flag_col = f"{col}_was_missing"
        df[flag_col] = df[col].isna().astype(np.int8)
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    return df


def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log1p to heavily skewed numeric features.
    feature_015 can be negative; we shift by abs(min)+1 before log.
    """
    df = df.copy()
    for col in LOG_FEATURES:
        if col not in df.columns:
            continue
        shift = 0.0
        if df[col].min() <= 0:
            shift = abs(df[col].min()) + 1.0
        df[f"{col}_log"] = np.log1p(df[col] + shift)
    return df


def add_zscore_flags(df: pd.DataFrame, threshold: float = 3.5) -> pd.DataFrame:
    """
    For each numeric feature, add a binary flag = 1 if the value is
    more than `threshold` standard deviations from the mean.
    Extreme values may signal automated / bot behaviour.
    """
    df = df.copy()
    for col in NUMERIC_RAW:
        if col not in df.columns:
            continue
        mu, sigma = df[col].mean(), df[col].std()
        if sigma < 1e-9:
            continue
        z = (df[col] - mu) / sigma
        df[f"{col}_is_extreme"] = (z.abs() > threshold).astype(np.int8)
    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Domain-motivated ratio features:
      - Efficiency: score per unit of time/resource
      - Consistency: ratio of two score-like features
    Adjust denominator guard (+1 or +eps) to avoid div-by-zero.
    """
    df = df.copy()
    eps = 1e-6

    # Score / difficulty proxies
    df["f003_f001_ratio"] = df["feature_003"] / (df["feature_001"] + eps)
    df["f002_f005_ratio"] = df["feature_002"] / (df["feature_005"] + eps)
    df["f008_f009_ratio"] = df["feature_008"] / (df["feature_009"] + eps)

    # Session intensity: high time (f015) but low score (f004) = suspicious
    df["f004_per_f015"] = df["feature_004"] / (df["feature_015"].abs() + eps)

    # Composite: combined binary flag count (cheaters may have specific combos)
    binary_cols = [c for c in BINARY_FEATURES if c in df.columns]
    df["binary_flag_sum"] = df[binary_cols].sum(axis=1)

    # ── feature_015 tail signals (EDA: secondary cheater bump ~100) ──────────
    if "feature_015" in df.columns:
        # Binary flag: cheater secondary mode visible above ~80
        df["feature_015_gt80"] = (df["feature_015"] > 80).astype(np.int8)

        # 5-bin ordinal — lets the tree find the true KDE boundary
        # rather than hard-coding the visual estimate at 80
        df["feature_015_bin"] = pd.cut(
            df["feature_015"],
            bins=[-np.inf, 0, 20, 80, 200, np.inf],
            labels=[0, 1, 2, 3, 4],
        ).astype(float)  # float so XGBoost/LGBM treat as numeric

    return df


def add_eda_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hard-coded interactions identified by EDA analysis.

    feature_013 interpretation (from leakage check):
      flag=0 → 35.7% cheat rate  (HIGHER risk — flag absent)
      flag=1 → 16.4% cheat rate  (LOWER risk — exemption/verified flag)

    So (1 - feature_013) × feature_015 captures: "no exemption flag AND long session"
    which is the highest-precision cheater sub-profile from the EDA.
    """
    df = df.copy()
    if "feature_013" in df.columns and "feature_015" in df.columns:
        # Inverted flag: 1 = at-risk (no exemption)
        at_risk = 1 - df["feature_013"].fillna(0)
        df["f013inv_x_f015"]      = at_risk * df["feature_015"]
        df["f013inv_x_f015_gt80"] = at_risk * df.get("feature_015_gt80", 0)
        df["f013inv_x_f015_bin"]  = at_risk * df.get("feature_015_bin", 0)

    if "feature_013" in df.columns and "feature_004" in df.columns:
        # No-exemption AND high score: legitimate high performers who aren't flagged
        # tend to have consistent scores. This captures the opposite pattern.
        at_risk = 1 - df["feature_013"].fillna(0)
        df["f013inv_x_f004"] = at_risk * df["feature_004"]

    return df


def add_interaction_terms(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """
    Multiply top feature pairs. If `pairs` is None, uses pre-selected pairs
    from offline MI analysis. Can be replaced by calling `select_top_pairs`
    on the training set first.
    """
    DEFAULT_PAIRS = [
        ("feature_001", "feature_003"),
        ("feature_004", "feature_005"),
        ("feature_002", "feature_006"),
        ("feature_008", "feature_009"),
        ("feature_015", "feature_016"),
    ]
    pairs = pairs or DEFAULT_PAIRS
    df = df.copy()
    for a, b in pairs:
        if a in df.columns and b in df.columns:
            df[f"{a}_x_{b}"] = df[a] * df[b]
    return df


def select_top_pairs(
    df: pd.DataFrame,
    y: pd.Series,
    top_k: int = 5,
) -> list[tuple[str, str]]:
    """
    Compute pairwise interaction MI scores against y and return the top_k pairs.
    Call once on the training set; pass result to add_interaction_terms.
    """
    from itertools import combinations
    cols = [c for c in NUMERIC_RAW if c in df.columns]
    scores = {}
    for a, b in combinations(cols, 2):
        interaction = (df[a] * df[b]).values.reshape(-1, 1)
        mi = mutual_info_classif(interaction, y, discrete_features=False)[0]
        scores[(a, b)] = mi
    return sorted(scores, key=lambda k: scores[k], reverse=True)[:top_k]


# ─────────────────────────────────────────────────────────────────────────────

def build_behavioral_features(
    df: pd.DataFrame,
    interaction_pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """
    Master pipeline: run all behavioral feature engineering steps in order.

    Parameters
    ----------
    df : raw dataframe (must contain feature_001 … feature_018 columns)
    interaction_pairs : MI-selected pairs from training set; if None, uses defaults

    Returns
    -------
    Engineered dataframe (does not include target columns)
    """
    df = impute_and_flag(df)
    df = log_transform(df)
    df = add_zscore_flags(df)
    df = add_ratio_features(df)      # includes feature_015 tail flags
    df = add_eda_interactions(df)    # f013-inv × f015 profile interactions
    df = add_interaction_terms(df, interaction_pairs)  # MI-selected pairs
    return df
