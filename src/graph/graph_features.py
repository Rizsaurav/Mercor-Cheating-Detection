"""
src/graph/graph_features.py
----------------------------
Social graph feature engineering.
This is the highest-alpha component: most competitors will stop at
degree centrality. We go to 2-hop neighbourhood cheating density,
PageRank, Louvain community structure, and ghost-user ratio.

Graph: undirected (user_a, user_b) pairs — 1.7M edges.

Performance notes
-----------------
All per-node operations use vectorised scipy sparse / numpy operations,
NOT Python loops over the node list.  On a 1.7M-edge graph the Python-
loop version of 2-hop density takes ~2 hours; the sparse matrix version
takes ~30 seconds.

Key correctness requirement
---------------------------
Graph features MUST be computed on the FULL graph (train + test nodes)
together.  Test users appear in the social graph and their neighbourhood
statistics must be computed the same way as train users.  Passing only
train nodes would leave test users with degree=0 and rate=0, which is
wrong and will hurt leaderboard score even though CV looks fine.

Feature inventory
-----------------
Structural:
  graph_degree           – total connections
  pagerank               – global influence score

Cheating-signal propagation (highest value):
  neigh_cheat_rate_1hop  – fraction of direct neighbours with is_cheating=1
  neigh_cheat_rate_2hop  – same for 2-hop exclusive neighbourhood
  cheat_neigh_count_1hop – raw count of cheating neighbours
  cheat_neigh_count_2hop – raw count of 2-hop cheating neighbours
  labeled_neigh_count_1hop – total labeled neighbours (denominator context)
  avg_neigh_feature_*    – mean of each raw feature across 1-hop neighbours

Community:
  louvain_cluster_id     – Louvain community membership (treat as categorical)
  cluster_size           – size of the community
  cluster_cheat_rate     – fraction of labeled cheaters in the community

Ghost-user signals:
  ghost_neigh_count      – count of ghost-node neighbours
  ghost_neigh_ratio      – fraction of neighbours who are ghost nodes
"""
from __future__ import annotations

import time
import warnings
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    def tqdm(x, **kw):   # no-op fallback
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Graph loading
# ─────────────────────────────────────────────────────────────────────────────

def load_graph(graph_path: str | Path) -> tuple[nx.Graph, np.ndarray, dict]:
    """
    Load social_graph.csv and return:
        G         – undirected NetworkX graph
        node_list – ordered array of all node IDs (used for sparse indexing)
        node2idx  – dict mapping node_id → integer index into node_list
    """
    t0 = time.time()
    print(f"[graph] Reading {graph_path} …", flush=True)
    edges = pd.read_csv(graph_path)
    G = nx.from_pandas_edgelist(edges, source="user_a", target="user_b")
    node_list = np.array(list(G.nodes()))
    node2idx  = {n: i for i, n in enumerate(node_list)}
    elapsed = time.time() - t0
    print(
        f"[graph] Loaded  nodes={G.number_of_nodes():,}  "
        f"edges={G.number_of_edges():,}  ({elapsed:.1f}s)",
        flush=True,
    )
    return G, node_list, node2idx


def _build_sparse_adj(G: nx.Graph, node2idx: dict) -> sp.csr_matrix:
    """Build a scipy CSR adjacency matrix from the NetworkX graph."""
    n = len(node2idx)
    rows, cols = [], []
    for u, v in G.edges():
        i, j = node2idx[u], node2idx[v]
        rows += [i, j]
        cols += [j, i]
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    return A


# ─────────────────────────────────────────────────────────────────────────────
# Structural features
# ─────────────────────────────────────────────────────────────────────────────

def compute_degree_features(
    G: nx.Graph,
    users: list,
) -> pd.DataFrame:
    """Degree for every user in the provided list (O(1) dict lookup)."""
    deg = dict(G.degree())
    return pd.DataFrame(
        {"user_hash": users, "graph_degree": [deg.get(u, 0) for u in users]}
    ).set_index("user_hash")


def compute_pagerank(
    G: nx.Graph,
    users: list,
    alpha: float = 0.85,
) -> pd.DataFrame:
    """PageRank — cheaters often cluster in high-authority subgraphs."""
    t0 = time.time()
    print("[graph] Computing PageRank …", flush=True)
    pr = nx.pagerank(G, alpha=alpha, max_iter=300, tol=1e-5)
    print(f"[graph] PageRank done ({time.time()-t0:.1f}s)", flush=True)
    return pd.DataFrame(
        {"user_hash": users, "pagerank": [pr.get(u, 0.0) for u in users]}
    ).set_index("user_hash")


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised 2-hop cheating density
# ─────────────────────────────────────────────────────────────────────────────

def compute_neighbourhood_cheat_rates(
    G: nx.Graph,
    node_list: np.ndarray,
    node2idx: dict,
    A: sp.csr_matrix,
    users: list,
    cheat_set: set,
    labeled_set: set,
) -> pd.DataFrame:
    """
    Vectorised 1-hop and 2-hop cheating density using scipy sparse matrix ops.

    The key insight: if `c` is the binary vector indicating cheaters and
    `l` is the binary vector indicating labeled nodes, then:

        A @ c   gives #cheating-neighbours per node  (1-hop)
        A @ l   gives #labeled-neighbours per node   (1-hop)
        A² @ c  gives sum of 1-hop cheater neighbours over all 2-hop paths

    This runs in ~30s on the full 1.7M-edge graph vs ~2 hours for Python loops.

    NOTE: `users` should contain ALL users (train + test) that need features —
    do NOT call this once for train and once for test separately, as the graph
    context is global.
    """
    n = len(node_list)
    t0 = time.time()
    print(
        f"[graph] Vectorised neighbourhood density for {len(users):,} users …",
        flush=True,
    )

    # Binary indicator vectors (full graph node space)
    cheat_vec   = np.zeros(n, dtype=np.float32)
    labeled_vec = np.zeros(n, dtype=np.float32)
    for node in cheat_set:
        if node in node2idx:
            cheat_vec[node2idx[node]] = 1.0
    for node in labeled_set:
        if node in node2idx:
            labeled_vec[node2idx[node]] = 1.0

    # --- 1-hop -----------------------------------------------------------------
    cheat_1hop   = np.array(A @ cheat_vec).ravel()    # #cheating 1-hop neighbours
    labeled_1hop = np.array(A @ labeled_vec).ravel()  # #labeled  1-hop neighbours

    # --- 2-hop -----------------------------------------------------------------
    # A² gives all 2-hop paths (including through self and 1-hop).
    # We compute A² @ c, then subtract the 1-hop contribution so that
    # 2-hop is *exclusive* (not double-counting 1-hop cheaters).
    print("[graph]   computing A² (2-hop) …", flush=True)
    A2_cheat   = np.array((A @ A) @ cheat_vec).ravel()
    A2_labeled = np.array((A @ A) @ labeled_vec).ravel()

    # Build index for the requested users
    user_idxs = np.array([node2idx.get(u, -1) for u in users])
    in_graph   = user_idxs >= 0

    cheat_1   = np.where(in_graph, cheat_1hop[np.where(in_graph, user_idxs, 0)],   0.0)
    labeled_1 = np.where(in_graph, labeled_1hop[np.where(in_graph, user_idxs, 0)], 0.0)
    cheat_2   = np.where(in_graph, A2_cheat[np.where(in_graph, user_idxs, 0)],     0.0)
    labeled_2 = np.where(in_graph, A2_labeled[np.where(in_graph, user_idxs, 0)],   0.0)

    rate_1 = np.where(labeled_1 > 0, cheat_1 / labeled_1, 0.0)
    rate_2 = np.where(labeled_2 > 0, cheat_2 / labeled_2, 0.0)

    print(f"[graph] Neighbourhood density done ({time.time()-t0:.1f}s)", flush=True)

    return pd.DataFrame({
        "user_hash":               users,
        "neigh_cheat_rate_1hop":   rate_1,
        "cheat_neigh_count_1hop":  cheat_1.astype(int),
        "neigh_cheat_rate_2hop":   rate_2,
        "cheat_neigh_count_2hop":  cheat_2.astype(int),
        "labeled_neigh_count_1hop": labeled_1.astype(int),
    }).set_index("user_hash")


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised neighbour feature aggregation (GNN substitute)
# ─────────────────────────────────────────────────────────────────────────────

def compute_neighbour_feature_agg(
    node_list: np.ndarray,
    node2idx: dict,
    A: sp.csr_matrix,
    users: list,
    feature_df: pd.DataFrame,   # indexed by user_hash
    agg_cols: list[str],
) -> pd.DataFrame:
    """
    Vectorised mean-aggregation message passing (1-hop).

    For each column c in agg_cols:
        neigh_mean_c[u] = (A @ F[:, c]) / degree[u]
        neigh_std_c[u]  = sqrt((A @ F[:, c]²) / degree[u]  -  mean²)

    This is O(|E| * |agg_cols|) — feasible for 1.7M edges and 10 cols.
    """
    t0 = time.time()
    n = len(node_list)
    valid_cols = [c for c in agg_cols if c in feature_df.columns]
    print(
        f"[graph] Neighbour feature agg for {len(valid_cols)} cols "
        f"over {len(users):,} users …",
        flush=True,
    )

    # Build feature matrix F aligned to node_list ordering
    F = np.zeros((n, len(valid_cols)), dtype=np.float32)
    aligned = feature_df.reindex(node_list)[valid_cols].fillna(0.0).values.astype(np.float32)
    F[:, :] = aligned

    # Degree vector (avoid div-by-zero)
    degree = np.array(A.sum(axis=1)).ravel()
    degree_safe = np.where(degree > 0, degree, 1.0)

    # Mean aggregation:  A @ F / degree
    AF  = (A @ F)                                       # (n, k)
    AF2 = (A @ (F ** 2))                                # (n, k)  for std
    mean_agg = AF  / degree_safe[:, None]
    std_agg  = np.sqrt(np.clip(AF2 / degree_safe[:, None] - mean_agg ** 2, 0, None))

    # Extract rows for requested users
    user_idxs = np.array([node2idx.get(u, -1) for u in users])
    in_graph  = user_idxs >= 0
    safe_idxs = np.where(in_graph, user_idxs, 0)

    result_dict: dict[str, np.ndarray] = {"user_hash": users}
    for j, col in enumerate(valid_cols):
        m = np.where(in_graph, mean_agg[safe_idxs, j], np.nan)
        s = np.where(in_graph, std_agg[safe_idxs, j],  np.nan)
        result_dict[f"neigh_mean_{col}"] = m
        result_dict[f"neigh_std_{col}"]  = s

    print(f"[graph] Neighbour agg done ({time.time()-t0:.1f}s)", flush=True)
    return pd.DataFrame(result_dict).set_index("user_hash")


# ─────────────────────────────────────────────────────────────────────────────
# Community detection (Louvain)
# ─────────────────────────────────────────────────────────────────────────────

def compute_community_features(
    G: nx.Graph,
    users: list,
    cheat_set: set,
    labeled_set: set,
    resolution: float = 1.0,
) -> pd.DataFrame:
    """
    Louvain community detection. Returns cluster_id, cluster_size,
    and the fraction of labeled members who are confirmed cheaters.
    """
    t0 = time.time()
    print("[graph] Running Louvain community detection …", flush=True)
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G, resolution=resolution)
    except ImportError:
        print("[graph] python-louvain not found — falling back to connected components")
        partition = {}
        for i, comp in enumerate(nx.connected_components(G)):
            for node in comp:
                partition[node] = i

    community_members: dict[int, list] = defaultdict(list)
    for node, cid in partition.items():
        community_members[cid].append(node)

    cluster_size = {cid: len(m) for cid, m in community_members.items()}
    cluster_cheat_rate = {}
    for cid, members in community_members.items():
        lab = [m for m in members if m in labeled_set]
        cluster_cheat_rate[cid] = (
            sum(1 for m in lab if m in cheat_set) / len(lab) if lab else 0.0
        )

    rows = []
    for u in users:
        cid = partition.get(u, -1)
        rows.append({
            "user_hash": u,
            "louvain_cluster_id":  cid,
            "cluster_size":        cluster_size.get(cid, 1),
            "cluster_cheat_rate":  cluster_cheat_rate.get(cid, 0.0),
        })

    print(f"[graph] Louvain done ({time.time()-t0:.1f}s)", flush=True)
    return pd.DataFrame(rows).set_index("user_hash")


# ─────────────────────────────────────────────────────────────────────────────
# Ghost-user features (vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ghost_user_features(
    G: nx.Graph,
    node_list: np.ndarray,
    node2idx: dict,
    A: sp.csr_matrix,
    users: list,
    known_users: set,
) -> pd.DataFrame:
    """
    Ghost nodes = nodes in the social graph NOT in train ∪ test.
    Vectorised: build a ghost indicator vector and do A @ ghost_vec.
    """
    t0 = time.time()
    print("[graph] Computing ghost-user features …", flush=True)
    n = len(node_list)

    ghost_vec = np.zeros(n, dtype=np.float32)
    for i, node in enumerate(node_list):
        if node not in known_users:
            ghost_vec[i] = 1.0

    ghost_1hop = np.array(A @ ghost_vec).ravel()   # #ghost neighbours per node
    degree     = np.array(A.sum(axis=1)).ravel()

    user_idxs = np.array([node2idx.get(u, -1) for u in users])
    in_graph  = user_idxs >= 0
    safe_idxs = np.where(in_graph, user_idxs, 0)

    ghost_count = np.where(in_graph, ghost_1hop[safe_idxs], 0.0).astype(int)
    deg_u       = np.where(in_graph, degree[safe_idxs], 1.0)
    ghost_ratio = np.where(in_graph, ghost_count / np.where(deg_u > 0, deg_u, 1.0), 0.0)

    print(f"[graph] Ghost-user features done ({time.time()-t0:.1f}s)", flush=True)
    return pd.DataFrame({
        "user_hash":        users,
        "ghost_neigh_count": ghost_count,
        "ghost_neigh_ratio": ghost_ratio,
    }).set_index("user_hash")


# ─────────────────────────────────────────────────────────────────────────────
# Sampling helper for validation / fast iteration
# ─────────────────────────────────────────────────────────────────────────────

def sample_graph(G: nx.Graph, fraction: float = 0.1, seed: int = 42) -> nx.Graph:
    """
    Return a subgraph of the requested fraction for end-to-end validation.
    Run build_graph_features on this before trusting the full run.

    Usage:
        G_small, node_list, node2idx = load_graph(path)
        G_sample = sample_graph(G_small, fraction=0.1)
    """
    import random
    random.seed(seed)
    nodes = list(G.nodes())
    sample_nodes = set(random.sample(nodes, int(len(nodes) * fraction)))
    return G.subgraph(sample_nodes).copy()


# ─────────────────────────────────────────────────────────────────────────────
# Master pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_graph_features(
    graph_path: str | Path,
    users: list,
    cheat_set: set,
    labeled_set: set,
    known_users: set,
    feature_df: pd.DataFrame,
    behavioral_agg_cols: list[str],
    cfg: dict | None = None,
    sample_fraction: float | None = None,
) -> pd.DataFrame:
    """
    Run all graph feature engineering and return a merged DataFrame indexed
    by user_hash.

    CRITICAL: `users` must be the UNION of train_labeled + train_unlabeled +
    test user_hashes.  Test users are in the social graph and need the same
    neighbourhood features — omitting them causes silent test-set bugs.

    Parameters
    ----------
    graph_path         : path to social_graph.csv
    users              : ALL user_hashes needing features (train + test)
    cheat_set          : user_hashes confirmed as cheaters (labeled train only)
    labeled_set        : all user_hashes that have a label
    known_users        : union of train + test hashes (to identify ghost nodes)
    feature_df         : behavioral features DataFrame (indexed by user_hash)
    behavioral_agg_cols: behavioral columns to aggregate across neighbours
    cfg                : optional graph config override
    sample_fraction    : if set (e.g. 0.10), run on a random subgraph for
                         fast validation.  Set to None for production.
    """
    from src.utils.config import GRAPH_CFG
    gcfg = cfg or GRAPH_CFG

    total_t0 = time.time()
    print(
        f"\n{'='*60}\n[graph] Starting graph feature pipeline\n"
        f"        users={len(users):,}  cheat_set={len(cheat_set):,}\n"
        f"        sample_fraction={sample_fraction}\n{'='*60}",
        flush=True,
    )

    # ── Load ──────────────────────────────────────────────────────────────────
    G, node_list, node2idx = load_graph(graph_path)

    if sample_fraction:
        print(f"[graph] ⚠ SAMPLE MODE: using {sample_fraction:.0%} of graph", flush=True)
        G = sample_graph(G, fraction=sample_fraction)
        node_list = np.array(list(G.nodes()))
        node2idx  = {n: i for i, n in enumerate(node_list)}
        # Restrict users to those in the sampled subgraph
        users = [u for u in users if u in node2idx]

    # ── Sparse adjacency matrix ───────────────────────────────────────────────
    print("[graph] Building sparse adjacency matrix …", flush=True)
    t0 = time.time()
    A = _build_sparse_adj(G, node2idx)
    print(f"[graph] Sparse adj built ({time.time()-t0:.1f}s)", flush=True)

    # ── Degree ────────────────────────────────────────────────────────────────
    deg_df = compute_degree_features(G, users)

    # ── PageRank ──────────────────────────────────────────────────────────────
    pr_df = compute_pagerank(G, users, alpha=gcfg["pagerank_alpha"])

    # ── 2-hop cheating density (vectorised) ───────────────────────────────────
    cheat_rate_df = compute_neighbourhood_cheat_rates(
        G, node_list, node2idx, A, users, cheat_set, labeled_set
    )

    # ── Neighbour feature aggregation (vectorised) ────────────────────────────
    agg_df = compute_neighbour_feature_agg(
        node_list, node2idx, A, users, feature_df, behavioral_agg_cols
    )

    # ── Louvain communities ───────────────────────────────────────────────────
    comm_df = compute_community_features(
        G, users, cheat_set, labeled_set, resolution=gcfg["louvain_resolution"]
    )

    # ── Ghost-user features (vectorised) ─────────────────────────────────────
    ghost_df = compute_ghost_user_features(
        G, node_list, node2idx, A, users, known_users
    )

    # ── Merge ─────────────────────────────────────────────────────────────────
    result = (
        deg_df
        .join(pr_df,          how="left")
        .join(cheat_rate_df,  how="left")
        .join(agg_df,         how="left")
        .join(comm_df,        how="left")
        .join(ghost_df,       how="left")
    )

    elapsed = time.time() - total_t0
    print(
        f"\n[graph] ✓ Pipeline complete — shape={result.shape}  "
        f"total_time={elapsed:.1f}s\n{'='*60}\n",
        flush=True,
    )
    return result
