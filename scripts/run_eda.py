"""
scripts/run_eda.py
------------------
Focused EDA covering four diagnostics:
  1. feature_015 / feature_016 class-conditional KDEs (bimodality check)
  2. MI scores + concentration (top-3 share of total MI)
  3. Behavioral vs graph-feature correlations (collinearity / validation)
  4. Threshold cost heatmap on baseline XGBoost OOF predictions

Run: python3 scripts/run_eda.py
Outputs saved to: outputs/plots/
OOF probs saved to: outputs/submissions/baseline_oof.csv
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde

PLOTS = Path("outputs/plots")
PLOTS.mkdir(parents=True, exist_ok=True)
SUBS  = Path("outputs/submissions")
SUBS.mkdir(parents=True, exist_ok=True)

# ── palette ──────────────────────────────────────────────────────────────────
LEGIT_C  = "#4C9BE8"   # blue
CHEAT_C  = "#E8614C"   # red
BG       = "#1C1C2E"
GRID_C   = "#2E2E45"

def dark_fig(w=12, h=5):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)
    ax.grid(color=GRID_C, linewidth=0.5)
    return fig, ax

def dark_figs(nrows, ncols, w=16, h=5):
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), facecolor=BG)
    for ax in np.array(axes).ravel():
        ax.set_facecolor(BG)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_C)
        ax.grid(color=GRID_C, linewidth=0.5)
    return fig, axes

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
print("[eda] Loading data …")
train = pd.read_csv("data/train.csv")
labeled = train[train["is_cheating"].notna()].copy()
labeled["is_cheating"] = labeled["is_cheating"].astype(int)

legit  = labeled[labeled["is_cheating"] == 0]
cheat  = labeled[labeled["is_cheating"] == 1]
print(f"[eda] Labeled: {len(labeled):,}  |  Cheaters: {len(cheat):,}  |  Legit: {len(legit):,}")

feat_cols = [f"feature_{i:03d}" for i in range(1, 19)]

# ─────────────────────────────────────────────────────────────────────────────
# 1. feature_015 and feature_016 — KDE + histogram overlay
# ─────────────────────────────────────────────────────────────────────────────
print("[eda] Plot 1: feature_015 / feature_016 KDE …")

fig, axes = dark_figs(1, 2, w=16, h=6)
fig.suptitle("feature_015 & feature_016 — Class-Conditional Distributions",
             color="white", fontsize=14, fontweight="bold")

for ax, col in zip(axes, ["feature_015", "feature_016"]):
    l_vals = legit[col].dropna().values
    c_vals = cheat[col].dropna().values

    # Clip to 1st–99th pct for KDE readability
    lo = np.percentile(np.concatenate([l_vals, c_vals]), 1)
    hi = np.percentile(np.concatenate([l_vals, c_vals]), 99)
    xs = np.linspace(lo, hi, 500)

    for vals, color, label in [(l_vals, LEGIT_C, "Legit"), (c_vals, CHEAT_C, "Cheater")]:
        clipped = vals[(vals >= lo) & (vals <= hi)]
        ax.hist(clipped, bins=80, density=True, alpha=0.25, color=color)
        kde = gaussian_kde(clipped, bw_method=0.15)
        ax.plot(xs, kde(xs), color=color, linewidth=2, label=label)

    # Mark mean / median per class
    for vals, color in [(l_vals, LEGIT_C), (c_vals, CHEAT_C)]:
        ax.axvline(np.median(vals), color=color, linestyle="--", linewidth=1, alpha=0.7)

    ax.set_title(col, color="white", fontsize=12)
    ax.set_xlabel("Value", color="white")
    ax.set_ylabel("Density", color="white")
    ax.legend(facecolor=BG, labelcolor="white", fontsize=9)

    # Annotate stats
    stats = (f"Legit  μ={l_vals.mean():.1f} σ={l_vals.std():.1f}\n"
             f"Cheat  μ={c_vals.mean():.1f} σ={c_vals.std():.1f}")
    ax.text(0.97, 0.95, stats, transform=ax.transAxes,
            color="white", fontsize=8, va="top", ha="right",
            bbox=dict(facecolor=GRID_C, alpha=0.7, boxstyle="round"))

plt.tight_layout()
out = PLOTS / "kde_015_016.png"
plt.savefig(out, dpi=160, facecolor=BG)
plt.close()
print(f"[eda]  → {out}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. MI scores — concentration analysis
# ─────────────────────────────────────────────────────────────────────────────
print("[eda] Plot 2: Mutual information scores …")
from sklearn.feature_selection import mutual_info_classif

X_mi = labeled[feat_cols].fillna(labeled[feat_cols].median())
y_mi = labeled["is_cheating"]
mi   = mutual_info_classif(X_mi, y_mi, discrete_features=False, random_state=42)
mi_s = pd.Series(mi, index=feat_cols).sort_values(ascending=False)

top3_share  = mi_s.iloc[:3].sum() / mi_s.sum()
top5_share  = mi_s.iloc[:5].sum() / mi_s.sum()
print(f"[eda]  Top-3 share of total MI: {top3_share:.1%}")
print(f"[eda]  Top-5 share of total MI: {top5_share:.1%}")
print("[eda]  Top features:", mi_s.head(6).to_dict())

fig, ax = dark_fig(w=13, h=5)
colors = [CHEAT_C if i < 3 else LEGIT_C if i < 6 else "#888" for i in range(len(mi_s))]
bars = ax.bar(mi_s.index, mi_s.values, color=colors)
ax.set_title(
    f"Mutual Information vs is_cheating  |  "
    f"Top-3 = {top3_share:.0%} of total MI  |  Top-5 = {top5_share:.0%}",
    color="white", fontsize=11
)
ax.set_ylabel("MI Score", color="white")
ax.set_xticklabels(mi_s.index, rotation=45, ha="right", fontsize=8, color="white")
# Annotate values
for bar, val in zip(bars, mi_s.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{val:.3f}", ha="center", va="bottom", fontsize=7, color="white")

from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor=CHEAT_C, label="Top-3"),
    Patch(facecolor=LEGIT_C, label="Top 4-6"),
    Patch(facecolor="#888",  label="Rest"),
], facecolor=BG, labelcolor="white", fontsize=9)

plt.tight_layout()
out = PLOTS / "mi_scores.png"
plt.savefig(out, dpi=160, facecolor=BG)
plt.close()
print(f"[eda]  → {out}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. ALL 18 feature KDEs in one grid
# ─────────────────────────────────────────────────────────────────────────────
print("[eda] Plot 3: All 18 feature KDEs …")
fig = plt.figure(figsize=(22, 14), facecolor=BG)
fig.suptitle("All Features — Cheater vs Legit Distributions",
             color="white", fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(3, 6, figure=fig, hspace=0.45, wspace=0.35)

for idx, col in enumerate(feat_cols):
    ax = fig.add_subplot(gs[idx // 6, idx % 6])
    ax.set_facecolor(BG)
    ax.tick_params(colors="white", labelsize=6)
    ax.set_title(col, color="white", fontsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)
    ax.grid(color=GRID_C, linewidth=0.4)

    l_vals = legit[col].dropna().values
    c_vals = cheat[col].dropna().values
    if len(l_vals) < 10 or len(c_vals) < 10:
        continue

    lo = np.percentile(np.concatenate([l_vals, c_vals]), 1)
    hi = np.percentile(np.concatenate([l_vals, c_vals]), 99)
    if hi <= lo:
        continue
    xs = np.linspace(lo, hi, 300)
    for vals, color in [(l_vals, LEGIT_C), (c_vals, CHEAT_C)]:
        clipped = vals[(vals >= lo) & (vals <= hi)]
        if len(np.unique(clipped)) < 5:
            continue
        try:
            kde = gaussian_kde(clipped, bw_method=0.2)
            ax.plot(xs, kde(xs), color=color, linewidth=1.5)
            ax.fill_between(xs, kde(xs), alpha=0.15, color=color)
        except Exception:
            pass

out = PLOTS / "all_features_kde.png"
plt.savefig(out, dpi=140, facecolor=BG, bbox_inches="tight")
plt.close()
print(f"[eda]  → {out}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Baseline OOF + threshold heatmap
# ─────────────────────────────────────────────────────────────────────────────
print("[eda] Training baseline XGBoost (5-fold OOF) …")
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from src.features.behavioral import build_behavioral_features
from src.utils.cost_metric import competition_cost, find_best_thresholds

lab_feat = build_behavioral_features(labeled)
meta_cols = {"user_hash", "is_cheating", "high_conf_clean"}
feat_eng  = [c for c in lab_feat.columns if c not in meta_cols]

X = lab_feat[feat_eng].fillna(0).astype(float)
y = lab_feat["is_cheating"].astype(int)

xgb = XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=2.0,
    eval_metric="logloss", n_jobs=-1, random_state=42, tree_method="hist",
    verbosity=0,
)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("[eda]  Running cross_val_predict …")
oof_probs = cross_val_predict(xgb, X, y, cv=skf, method="predict_proba")[:, 1]

auc = roc_auc_score(y, oof_probs)
t_pass, t_block, best_cost = find_best_thresholds(y.values, oof_probs)
print(f"[eda]  OOF AUC        : {auc:.4f}")
print(f"[eda]  Optimal t_pass : {t_pass:.3f}")
print(f"[eda]  Optimal t_block: {t_block:.3f}")
print(f"[eda]  Min cost       : ${best_cost:,.0f}")

# Save OOF for tune_thresholds.py
oof_df = pd.DataFrame({"user_hash": labeled["user_hash"].values, "prediction": oof_probs})
oof_path = SUBS / "baseline_oof.csv"
oof_df.to_csv(oof_path, index=False)
print(f"[eda]  OOF saved → {oof_path}")

# ── OOF probability distribution ─────────────────────────────────────────────
fig, axes = dark_figs(1, 2, w=16, h=5)
fig.suptitle(f"Baseline XGBoost OOF  |  AUC={auc:.4f}  |  Min Cost=${best_cost:,.0f}",
             color="white", fontsize=13, fontweight="bold")

ax = axes[0]
xs = np.linspace(0, 1, 300)
for grp_y, color, label in [(0, LEGIT_C, "Legit"), (1, CHEAT_C, "Cheater")]:
    vals = oof_probs[y.values == grp_y]
    ax.hist(vals, bins=60, density=True, alpha=0.3, color=color)
    kde = gaussian_kde(vals, bw_method=0.05)
    ax.plot(xs, kde(xs), color=color, linewidth=2, label=label)
ax.axvline(t_pass,  color="#FFD700", linestyle="--", linewidth=1.5, label=f"t_pass={t_pass:.2f}")
ax.axvline(t_block, color="#FF6B6B", linestyle="--", linewidth=1.5, label=f"t_block={t_block:.2f}")
ax.set_title("OOF Probability Distribution", color="white")
ax.set_xlabel("Predicted P(cheating)", color="white")
ax.set_ylabel("Density", color="white")
ax.legend(facecolor=BG, labelcolor="white", fontsize=9)

# Annotate zone areas
ax.axvspan(0,       t_pass,  alpha=0.07, color=LEGIT_C, label="auto-pass zone")
ax.axvspan(t_pass,  t_block, alpha=0.07, color="#FFD700")
ax.axvspan(t_block, 1,       alpha=0.07, color=CHEAT_C)

# ── Cost heatmap ─────────────────────────────────────────────────────────────
ax2 = axes[1]
t1_vals = np.arange(0.01, 0.50, 0.01)
t2_vals = np.arange(0.50, 0.99, 0.01)
Z = np.full((len(t1_vals), len(t2_vals)), np.nan)
for i, t1 in enumerate(t1_vals):
    for j, t2 in enumerate(t2_vals):
        Z[i, j] = competition_cost(y.values, oof_probs, t1, t2)

im = ax2.imshow(
    Z, aspect="auto", origin="lower",
    extent=[t2_vals[0], t2_vals[-1], t1_vals[0], t1_vals[-1]],
    cmap="RdYlGn_r", vmin=np.nanpercentile(Z, 5), vmax=np.nanpercentile(Z, 95),
)
plt.colorbar(im, ax=ax2, label="Total Cost ($)").ax.yaxis.label.set_color("white")
ax2.scatter([t_block], [t_pass], color="white", s=80, zorder=5,
            marker="*", label=f"Optimal ({t_pass:.2f}, {t_block:.2f})")
ax2.set_xlabel("t_block", color="white")
ax2.set_ylabel("t_pass",  color="white")
ax2.set_title("Cost Heatmap — Threshold Grid Search", color="white")
ax2.legend(facecolor=BG, labelcolor="white", fontsize=9)

plt.tight_layout()
out = PLOTS / "oof_and_heatmap.png"
plt.savefig(out, dpi=160, facecolor=BG)
plt.close()
print(f"[eda]  → {out}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Behavioral vs graph feature correlations
#    (compute graph features on 20% sample for speed)
# ─────────────────────────────────────────────────────────────────────────────
print("[eda] Plot 4: Behavioral–graph correlation matrix …")
import networkx as nx
import scipy.sparse as sp

GRAPH_SAMPLE_FRAC = 0.20
print(f"[eda]  Loading graph (random {GRAPH_SAMPLE_FRAC:.0%} sample) …")
edges = pd.read_csv("data/social_graph.csv")
# Sample edges
rng = np.random.default_rng(42)
edges = edges.iloc[rng.choice(len(edges), int(len(edges)*GRAPH_SAMPLE_FRAC), replace=False)]
G = nx.from_pandas_edgelist(edges, source="user_a", target="user_b")
print(f"[eda]  Sampled graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

deg = dict(G.degree())
labeled["graph_degree"] = labeled["user_hash"].map(deg).fillna(0)

# 1-hop cheating density (vectorised)
node_list = np.array(list(G.nodes()))
node2idx  = {n: i for i, n in enumerate(node_list)}
n = len(node_list)
cheat_set = set(labeled.loc[labeled["is_cheating"]==1, "user_hash"])
labeled_set = set(labeled["user_hash"])
cheat_vec   = np.array([1.0 if nd in cheat_set else 0.0 for nd in node_list], dtype=np.float32)
labeled_vec = np.array([1.0 if nd in labeled_set else 0.0 for nd in node_list], dtype=np.float32)
rows, cols, data = [], [], []
for u, v in G.edges():
    i, j = node2idx[u], node2idx[v]
    rows += [i, j]; cols += [j, i]; data += [1.0, 1.0]
A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
cheat1 = np.array(A @ cheat_vec).ravel()
lab1   = np.array(A @ labeled_vec).ravel()
uidxs  = np.array([node2idx.get(u, -1) for u in labeled["user_hash"]])
in_g   = uidxs >= 0
safe   = np.where(in_g, uidxs, 0)
labeled["neigh_cheat_rate_1hop"] = np.where(in_g, np.where(lab1[safe]>0, cheat1[safe]/lab1[safe], 0), 0)

# Louvain community cheat rate (connected-component fallback for speed)
comp_map = {}
for cid, comp in enumerate(nx.connected_components(G)):
    for nd in comp:
        comp_map[nd] = cid
labeled["louvain_cluster_id"] = labeled["user_hash"].map(comp_map).fillna(-1).astype(int)
cluster_cr = labeled.groupby("louvain_cluster_id")["is_cheating"].mean()
labeled["cluster_cheat_rate"] = labeled["louvain_cluster_id"].map(cluster_cr).fillna(0)

graph_feat_cols = ["graph_degree", "neigh_cheat_rate_1hop", "cluster_cheat_rate"]
beh_cols_for_corr = [c for c in feat_cols if c in labeled.columns]
corr_df = labeled[beh_cols_for_corr + graph_feat_cols].corr()
cross_corr = corr_df.loc[beh_cols_for_corr, graph_feat_cols]

fig, ax = dark_fig(w=10, h=7)
import matplotlib.colors as mcolors
cmap = plt.cm.RdBu_r
im = ax.imshow(cross_corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(graph_feat_cols)))
ax.set_yticks(range(len(beh_cols_for_corr)))
ax.set_xticklabels(graph_feat_cols, color="white", fontsize=9, rotation=20, ha="right")
ax.set_yticklabels(beh_cols_for_corr, color="white", fontsize=8)
plt.colorbar(im, ax=ax, label="Pearson r").ax.yaxis.label.set_color("white")
ax.set_title("Behavioral Feature × Graph Feature Correlations\n(Collinearity / Signal Validation)",
             color="white", fontsize=11)
# Annotate cells
for i in range(len(beh_cols_for_corr)):
    for j in range(len(graph_feat_cols)):
        val = cross_corr.values[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                color="white" if abs(val) < 0.5 else "black", fontsize=7)
plt.tight_layout()
out = PLOTS / "behavioral_graph_correlation.png"
plt.savefig(out, dpi=160, facecolor=BG)
plt.close()
print(f"[eda]  → {out}")

# Warn on high cross-correlations
high_cross = [(beh_cols_for_corr[i], graph_feat_cols[j], cross_corr.values[i,j])
              for i in range(len(beh_cols_for_corr))
              for j in range(len(graph_feat_cols))
              if abs(cross_corr.values[i,j]) > 0.35]
if high_cross:
    print("[eda]  ⚠ High behavioral–graph correlations (|r|>0.35):")
    for a, b, v in sorted(high_cross, key=lambda x: -abs(x[2])):
        print(f"       {a} × {b}: {v:+.3f}")
else:
    print("[eda]  ✓ No high collinearity detected (all |r| < 0.35)")

# ─────────────────────────────────────────────────────────────────────────────
# Summary report
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EDA SUMMARY")
print("="*60)
print(f"  OOF AUC              : {auc:.4f}")
print(f"  Optimal t_pass       : {t_pass:.3f}")
print(f"  Optimal t_block      : {t_block:.3f}")
print(f"  Min achievable cost  : ${best_cost:,.0f}")
print(f"  Top-3 MI share       : {top3_share:.1%}")
print(f"  Top-5 MI share       : {top5_share:.1%}")
print(f"  Top MI features      : {', '.join(mi_s.head(5).index.tolist())}")
print(f"\n  f015  legit  μ={legit['feature_015'].mean():.2f} σ={legit['feature_015'].std():.2f}")
print(f"  f015  cheat  μ={cheat['feature_015'].mean():.2f} σ={cheat['feature_015'].std():.2f}")
print(f"  f016  legit  μ={legit['feature_016'].mean():.2f} σ={legit['feature_016'].std():.2f}")
print(f"  f016  cheat  μ={cheat['feature_016'].mean():.2f} σ={cheat['feature_016'].std():.2f}")
print(f"\n  Plots → outputs/plots/")
print(f"  OOF   → outputs/submissions/baseline_oof.csv")
print("="*60)
