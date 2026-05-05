"""
scripts/tune_thresholds.py
---------------------------
Standalone threshold optimiser.
Load any saved prediction file and run the full (t_pass, t_block) grid search.

Usage:
  python scripts/tune_thresholds.py \
      --preds outputs/submissions/oof_probs.csv \
      --labels data/train.csv
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.cost_metric import competition_cost, find_best_thresholds


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preds",  required=True, help="CSV with user_hash,prediction")
    p.add_argument("--labels", required=True, help="Train CSV with user_hash,is_cheating")
    p.add_argument("--plot",   action="store_true", help="Save cost heatmap")
    return p.parse_args()


def main():
    args = parse_args()

    preds_df  = pd.read_csv(args.preds)
    labels_df = pd.read_csv(args.labels)

    merged = preds_df.merge(labels_df[["user_hash", "is_cheating"]], on="user_hash")
    merged = merged[merged["is_cheating"].notna()]
    y_true = merged["is_cheating"].astype(int).values
    y_prob = merged["prediction"].values

    print(f"Evaluating {len(y_true):,} labeled samples …")
    t_pass, t_block, best_cost = find_best_thresholds(y_true, y_prob)

    print(f"\nOptimal t_pass  : {t_pass:.3f}")
    print(f"Optimal t_block : {t_block:.3f}")
    print(f"Minimum cost    : ${best_cost:,.0f}")

    # Decision breakdown at optimal thresholds
    auto_pass  = y_prob < t_pass
    auto_block = y_prob >= t_block
    review     = ~auto_pass & ~auto_block

    print("\n── Decision breakdown ──")
    print(f"Auto-pass  : {auto_pass.sum():,}  (FN: {((y_true==1) & auto_pass).sum()})")
    print(f"Manual rev : {review.sum():,}  (FP: {((y_true==0) & review).sum()})")
    print(f"Auto-block : {auto_block.sum():,}  (FP: {((y_true==0) & auto_block).sum()})")

    if args.plot:
        _plot_cost_heatmap(y_true, y_prob)


def _plot_cost_heatmap(y_true, y_prob):
    t_pass_vals  = np.arange(0.01, 0.50, 0.01)
    t_block_vals = np.arange(0.50, 0.99, 0.01)
    Z = np.zeros((len(t_pass_vals), len(t_block_vals)))

    for i, t1 in enumerate(t_pass_vals):
        for j, t2 in enumerate(t_block_vals):
            Z[i, j] = competition_cost(y_true, y_prob, t1, t2)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(Z, aspect="auto", origin="lower",
                   extent=[0.50, 0.99, 0.01, 0.50], cmap="RdYlGn_r")
    plt.colorbar(im, ax=ax, label="Total Cost ($)")
    ax.set_xlabel("t_block")
    ax.set_ylabel("t_pass")
    ax.set_title("Competition Cost Heatmap — Lower is Better")
    out = ROOT / "outputs" / "plots" / "threshold_heatmap.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Heatmap saved: {out}")


if __name__ == "__main__":
    main()
