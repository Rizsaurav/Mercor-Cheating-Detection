"""
preprocessing.py
-----------------
All data loading, cleaning, and feature engineering.

Steps (from project proposal):
  1. Load train.csv, test.csv, social_graph.csv
  2. Handle missing values (median imputation + missingness flags)
  3. Check for and remove duplicate records
  4. Feature scaling using StandardScaler (for LogReg baseline)
  5. Log-transform heavily skewed features
  6. Z-score outlier flags
  7. Interaction terms (MI-selected)
  8. Engineer social graph features (degree centrality, neighborhood risk score)
  9. Merge graph features onto train/test by user_hash
"""

import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import time
import warnings
from pathlib import Path
from itertools import combinations
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

warnings.filterwarnings("ignore")

# ─── Constants ────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
ALL_RAW = [f"feature_{i:03d}" for i in range(1, 19)]
BINARY_FEATURES = ["feature_007", "feature_011", "feature_013", "feature_014"]
NUMERIC_RAW = [f for f in ALL_RAW if f not in BINARY_FEATURES]
HIGH_MISSING = [
    "feature_001", "feature_002", "feature_003",
    "feature_007", "feature_008", "feature_009",
    "feature_010", "feature_011", "feature_012",
    "feature_013", "feature_014", "feature_017", "feature_018",
]
LOG_FEATURES = ["feature_010", "feature_016", "feature_015"]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(data_dir=DATA_DIR, debug=False):
    """Load train, test, and split labeled/unlabeled."""
    print("[data] Loading data …")
    train = pd.read_csv(data_dir / "train.csv")
    test  = pd.read_csv(data_dir / "test.csv")

    if debug:
        train = train.sample(frac=0.05, random_state=42)
        print(f"[data] DEBUG mode — using {len(train)} train rows")

    # Remove duplicates
    n_before = len(train)
    train = train.drop_duplicates(subset=["user_hash"])
    if len(train) < n_before:
        print(f"[data] Removed {n_before - len(train)} duplicate rows")

    labeled_mask   = train["is_cheating"].notna()
    train_labeled  = train[labeled_mask].copy()
    train_unlabeled = train[~labeled_mask].copy()

    print(
        f"[data] Train: {len(train):,}  |  Labeled: {len(train_labeled):,}  "
        f"|  Unlabeled: {len(train_unlabeled):,}  |  Test: {len(test):,}"
    )
    return train, train_labeled, train_unlabeled, test


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BEHAVIORAL FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def impute_and_flag(df):
    """Median-impute high-missing features + add binary missingness flags."""
    df = df.copy()
    for col in HIGH_MISSING:
        if col not in df.columns:
            continue
        df[f"{col}_was_missing"] = df[col].isna().astype(np.int8)
        df[col] = df[col].fillna(df[col].median())
    return df


def log_transform(df):
    """Log1p transform for heavily skewed features."""
    df = df.copy()
    for col in LOG_FEATURES:
        if col not in df.columns:
            continue
        shift = 0.0
        if df[col].min() <= 0:
            shift = abs(df[col].min()) + 1.0
        df[f"{col}_log"] = np.log1p(df[col] + shift)
    return df


def add_zscore_flags(df, threshold=3.5):
    """Binary flag for extreme values (>3.5 std from mean)."""
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


def add_ratio_features(df):
    """Domain-motivated ratio and interaction features."""
    df = df.copy()
    eps = 1e-6

    df["f003_f001_ratio"] = df["feature_003"] / (df["feature_001"] + eps)
    df["f002_f005_ratio"] = df["feature_002"] / (df["feature_005"] + eps)
    df["f008_f009_ratio"] = df["feature_008"] / (df["feature_009"] + eps)
    df["f004_per_f015"]   = df["feature_004"] / (df["feature_015"].abs() + eps)

    binary_cols = [c for c in BINARY_FEATURES if c in df.columns]
    df["binary_flag_sum"] = df[binary_cols].sum(axis=1)

    # feature_015 tail signals (EDA-identified secondary cheater bump ~100)
    if "feature_015" in df.columns:
        df["feature_015_gt80"] = (df["feature_015"] > 80).astype(np.int8)
        df["feature_015_bin"] = pd.cut(
            df["feature_015"],
            bins=[-np.inf, 0, 20, 80, 200, np.inf],
            labels=[0, 1, 2, 3, 4],
        ).astype(float)

    # feature_013 EDA interactions (f013=0 → higher risk, inverted flag)
    if "feature_013" in df.columns and "feature_015" in df.columns:
        at_risk = 1 - df["feature_013"].fillna(0)
        df["f013inv_x_f015"]      = at_risk * df["feature_015"]
        df["f013inv_x_f015_gt80"] = at_risk * df.get("feature_015_gt80", 0)

    if "feature_013" in df.columns and "feature_004" in df.columns:
        at_risk = 1 - df["feature_013"].fillna(0)
        df["f013inv_x_f004"] = at_risk * df["feature_004"]

    return df


def select_top_pairs(df, y, top_k=5):
    """Compute pairwise interaction MI scores and return top-k pairs."""
    cols = [c for c in NUMERIC_RAW if c in df.columns]
    scores = {}
    for a, b in combinations(cols, 2):
        interaction = (df[a] * df[b]).values.reshape(-1, 1)
        mi = mutual_info_classif(interaction, y, discrete_features=False)[0]
        scores[(a, b)] = mi
    return sorted(scores, key=lambda k: scores[k], reverse=True)[:top_k]


def add_interaction_terms(df, pairs=None):
    """Multiply top feature pairs (MI-selected)."""
    DEFAULT_PAIRS = [
        ("feature_001", "feature_003"), ("feature_004", "feature_005"),
        ("feature_002", "feature_006"), ("feature_008", "feature_009"),
        ("feature_015", "feature_016"),
    ]
    pairs = pairs or DEFAULT_PAIRS
    df = df.copy()
    for a, b in pairs:
        if a in df.columns and b in df.columns:
            df[f"{a}_x_{b}"] = df[a] * df[b]
    return df


def build_behavioral_features(df, interaction_pairs=None):
    """Master pipeline: all behavioral feature engineering in order."""
    df = impute_and_flag(df)
    df = log_transform(df)
    df = add_zscore_flags(df)
    df = add_ratio_features(df)
    df = add_interaction_terms(df, interaction_pairs)
    return df


def scale_for_logistic(X_train, X_test):
    """StandardScaler for logistic regression baseline (trees don't need this)."""
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train_scaled, X_test_scaled, scaler


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GRAPH FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def load_graph(graph_path):
    """Load social_graph.csv into NetworkX + sparse adjacency matrix."""
    t0 = time.time()
    print(f"[graph] Reading {graph_path} …")
    edges = pd.read_csv(graph_path)
    G = nx.from_pandas_edgelist(edges, source="user_a", target="user_b")
    node_list = np.array(list(G.nodes()))
    node2idx  = {n: i for i, n in enumerate(node_list)}
    print(f"[graph] Loaded  nodes={G.number_of_nodes():,}  edges={G.number_of_edges():,}  ({time.time()-t0:.1f}s)")
    return G, node_list, node2idx


def _build_sparse_adj(G, node2idx):
    """Build scipy CSR adjacency matrix from NetworkX graph."""
    n = len(node2idx)
    rows, cols = [], []
    for u, v in G.edges():
        i, j = node2idx[u], node2idx[v]
        rows += [i, j]
        cols += [j, i]
    data = np.ones(len(rows), dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def compute_graph_features(G, node_list, node2idx, A, users, cheat_set, labeled_set, known_users):
    """
    Compute all graph features:
      - Degree centrality
      - Neighbourhood cheating density (1-hop + 2-hop)
      - Louvain community cheat rate
      - Ghost-user ratio
    """
    n = len(node_list)
    t0 = time.time()
    print(f"[graph] Computing features for {len(users):,} users …")

    # ── Degree ────────────────────────────────────────────────────────────────
    deg = dict(G.degree())
    degree_vals = [deg.get(u, 0) for u in users]

    # ── Cheating density vectors ──────────────────────────────────────────────
    cheat_vec   = np.zeros(n, dtype=np.float32)
    labeled_vec = np.zeros(n, dtype=np.float32)
    for node in cheat_set:
        if node in node2idx:
            cheat_vec[node2idx[node]] = 1.0
    for node in labeled_set:
        if node in node2idx:
            labeled_vec[node2idx[node]] = 1.0

    # 1-hop
    cheat_1hop   = np.array(A @ cheat_vec).ravel()
    labeled_1hop = np.array(A @ labeled_vec).ravel()

    # 2-hop: A²v = A(Av) — two sparse mat-vec multiplies, no A² materialization
    A2_cheat   = np.array(A @ (A @ cheat_vec)).ravel()
    A2_labeled = np.array(A @ (A @ labeled_vec)).ravel()

    # ── Ghost-user features ───────────────────────────────────────────────────
    ghost_vec = np.zeros(n, dtype=np.float32)
    for i, node in enumerate(node_list):
        if node not in known_users:
            ghost_vec[i] = 1.0
    ghost_1hop = np.array(A @ ghost_vec).ravel()
    degree_arr = np.array(A.sum(axis=1)).ravel()

    # ── Louvain communities ───────────────────────────────────────────────────
    print("[graph] Running Louvain community detection …")
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G, resolution=1.0)
    except ImportError:
        print("[graph] python-louvain not found — falling back to connected components")
        partition = {}
        for i, comp in enumerate(nx.connected_components(G)):
            for node in comp:
                partition[node] = i

    community_members = defaultdict(list)
    for node, cid in partition.items():
        community_members[cid].append(node)

    cluster_size = {cid: len(m) for cid, m in community_members.items()}
    cluster_cheat_rate = {}
    for cid, members in community_members.items():
        lab = [m for m in members if m in labeled_set]
        cluster_cheat_rate[cid] = sum(1 for m in lab if m in cheat_set) / len(lab) if lab else 0.0

    # ── Build result per user ─────────────────────────────────────────────────
    user_idxs = np.array([node2idx.get(u, -1) for u in users])
    in_graph  = user_idxs >= 0
    safe_idxs = np.where(in_graph, user_idxs, 0)

    c1   = np.where(in_graph, cheat_1hop[safe_idxs], 0.0)
    l1   = np.where(in_graph, labeled_1hop[safe_idxs], 0.0)
    c2   = np.where(in_graph, A2_cheat[safe_idxs], 0.0)
    l2   = np.where(in_graph, A2_labeled[safe_idxs], 0.0)
    gc   = np.where(in_graph, ghost_1hop[safe_idxs], 0.0).astype(int)
    du   = np.where(in_graph, degree_arr[safe_idxs], 1.0)

    result = pd.DataFrame({
        "user_hash":               users,
        "graph_degree":            degree_vals,
        "neigh_cheat_rate_1hop":   np.where(l1 > 0, c1 / l1, 0.0),
        "cheat_neigh_count_1hop":  c1.astype(int),
        "neigh_cheat_rate_2hop":   np.where(l2 > 0, c2 / l2, 0.0),
        "cheat_neigh_count_2hop":  c2.astype(int),
        "ghost_neigh_count":       gc,
        "ghost_neigh_ratio":       np.where(du > 0, gc / du, 0.0),
        "louvain_cluster_id":      [partition.get(u, -1) for u in users],
        "cluster_size":            [cluster_size.get(partition.get(u, -1), 1) for u in users],
        "cluster_cheat_rate":      [cluster_cheat_rate.get(partition.get(u, -1), 0.0) for u in users],
    }).set_index("user_hash")

    print(f"[graph] Features done ({time.time()-t0:.1f}s)")
    return result


def build_graph_features(data_dir, train_labeled, train_unlabeled, test, sample_fraction=None):
    """Full graph feature pipeline: load graph → compute features → return DataFrame."""
    graph_path = data_dir / "social_graph.csv"
    if not graph_path.exists():
        print("[graph] social_graph.csv not found — skipping graph features")
        return None

    cheat_set   = set(train_labeled.loc[train_labeled["is_cheating"] == 1, "user_hash"])
    labeled_set = set(train_labeled["user_hash"])
    known_users = set(train_labeled["user_hash"]) | set(train_unlabeled["user_hash"]) | set(test["user_hash"])
    all_users   = list(train_labeled["user_hash"]) + list(train_unlabeled["user_hash"]) + list(test["user_hash"])

    print(f"\n[graph] Starting graph pipeline  |  users={len(all_users):,}  cheaters={len(cheat_set):,}")
    G, node_list, node2idx = load_graph(graph_path)

    if sample_fraction:
        import random
        random.seed(42)
        nodes = list(G.nodes())
        sample_nodes = set(random.sample(nodes, int(len(nodes) * sample_fraction)))
        G = G.subgraph(sample_nodes).copy()
        node_list = np.array(list(G.nodes()))
        node2idx  = {n: i for i, n in enumerate(node_list)}
        all_users = [u for u in all_users if u in node2idx]
        print(f"[graph] SAMPLE MODE: {sample_fraction:.0%} of graph")

    A = _build_sparse_adj(G, node2idx)
    return compute_graph_features(G, node_list, node2idx, A, all_users, cheat_set, labeled_set, known_users)
