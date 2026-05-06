"""
visualizations.py
-----------------
Run this cell in a NEW Kaggle notebook (after the main pipeline has run
and the variables below are in memory), or paste it as the final cell.

Requires in memory:
  results, best_oof, best, best_name, oof_blend,
  xgb_oof, lgbm_oof, xgb_final, lgbm_final,
  y_labeled, X_labeled, feature_cols, competition_cost
"""

from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    average_precision_score, roc_auc_score,
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

TEAL   = '#2ec4b6'
ORANGE = '#e76f51'
BLUE   = '#264653'
GREY   = '#dee2e6'
GREEN  = '#52b788'
RED    = '#e63946'

OUT = '/kaggle/working/'

# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Model Progression: Cost Reduction Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
model_order = ['LogReg', 'DecisionTree', 'RandomForest', 'XGBoost', 'LightGBM',
               'Ensemble', 'Ensemble+Pseudo']
model_labels = ['Logistic\nRegression', 'Decision\nTree', 'Random\nForest',
                'XGBoost', 'LightGBM', 'Ensemble\n(XGB+LGBM)', 'Ensemble\n+Pseudo']
model_costs  = [results[m]['min_cost'] for m in model_order if m in results]
labels_used  = [model_labels[i] for i, m in enumerate(model_order) if m in results]
bar_colors   = [GREY] * (len(model_costs) - 1) + [TEAL]

fig, ax = plt.subplots(figsize=(13, 5))
bars = ax.bar(labels_used, [c / 1e6 for c in model_costs],
              color=bar_colors, edgecolor='white', linewidth=0.8, width=0.6)

for bar, cost in zip(bars, model_costs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f'${cost / 1e6:.2f}M', ha='center', va='bottom', fontsize=10, fontweight='bold')

for i in range(len(model_costs) - 1):
    reduction = model_costs[i] - model_costs[i + 1]
    if reduction > 0:
        ax.annotate(f'-${reduction / 1e3:.0f}K',
                    xy=(i + 0.5, max(model_costs[i], model_costs[i + 1]) / 1e6 + 0.25),
                    ha='center', fontsize=8, color='#555', style='italic')

ax.set_ylabel('Total Cost (Millions USD)', fontweight='bold')
ax.set_title('Model Progression — Cost Reduction from Baseline to Best')
ax.set_ylim(0, max(model_costs) / 1e6 * 1.18)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:.0f}M'))
ax.axhline(model_costs[-1] / 1e6, color=TEAL, linestyle=':', alpha=0.5, lw=1.2)
fig.tight_layout()
fig.savefig(f'{OUT}fig1_model_progression.png', dpi=150, bbox_inches='tight')
plt.show()
print('Fig 1 saved')

# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Three-Zone: User Distribution + Cost Breakdown
# ══════════════════════════════════════════════════════════════════════════════
y_arr  = y_labeled.values
y_prob = best_oof
tp, tb = best['t_pass'], best['t_block']

ap = y_prob < tp
ab = y_prob >= tb
rv = ~ap & ~ab

zone_labels = [f'Auto-Pass\n(prob < {tp:.2f})',
               f'Manual Review\n({tp:.2f} – {tb:.2f})',
               f'Auto-Block\n(prob ≥ {tb:.2f})']
legit_vals  = [((y_arr == 0) & ap).sum(), ((y_arr == 0) & rv).sum(), ((y_arr == 0) & ab).sum()]
cheat_vals  = [((y_arr == 1) & ap).sum(), ((y_arr == 1) & rv).sum(), ((y_arr == 1) & ab).sum()]
zone_costs  = [
    600 * cheat_vals[0],
    150 * legit_vals[1] + 5 * cheat_vals[1],
    300 * legit_vals[2],
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x = np.arange(len(zone_labels))
w = 0.5
axes[0].bar(x, legit_vals, w, label='Legitimate', color=GREEN, edgecolor='white')
axes[0].bar(x, cheat_vals, w, bottom=legit_vals, label='Cheater', color=ORANGE, edgecolor='white')
axes[0].set_xticks(x); axes[0].set_xticklabels(zone_labels)
axes[0].set_ylabel('Number of Candidates')
axes[0].set_title('Three-Zone — User Distribution')
axes[0].legend()
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v/1e3:.0f}K'))
for i, (lv, cv) in enumerate(zip(legit_vals, cheat_vals)):
    axes[0].text(i, lv + cv + 300, f'{lv+cv:,}', ha='center', va='bottom', fontsize=9)

axes[1].bar(zone_labels, [c / 1e6 for c in zone_costs],
            color=[RED, ORANGE, GREEN], edgecolor='white', width=0.5)
for i, cost in enumerate(zone_costs):
    axes[1].text(i, cost / 1e6 + 0.02, f'${cost/1e6:.2f}M',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[1].set_ylabel('Zone Cost (Millions USD)')
axes[1].set_title('Three-Zone — Cost Breakdown')
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:.1f}M'))
axes[1].set_xticks(range(len(zone_labels))); axes[1].set_xticklabels(zone_labels)

total_cost = results[best_name]['min_cost']
fig.suptitle(f'{best_name}  |  Total Cost: ${total_cost:,.0f}',
             fontsize=13, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(f'{OUT}fig2_three_zone.png', dpi=150, bbox_inches='tight')
plt.show()
print('Fig 2 saved')

# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — ROC Curve
# ══════════════════════════════════════════════════════════════════════════════
fpr, tpr, _ = roc_curve(y_labeled, best_oof)
auc_val      = roc_auc_score(y_labeled, best_oof)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr, tpr, color=TEAL, lw=2.5, label=f'{best_name}  (AUC = {auc_val:.4f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='Random classifier')
ax.fill_between(fpr, tpr, alpha=0.08, color=TEAL)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend(loc='lower right')
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
fig.tight_layout()
fig.savefig(f'{OUT}fig3_roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()
print('Fig 3 saved')

# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Precision-Recall Curve
# ══════════════════════════════════════════════════════════════════════════════
prec, rec, _ = precision_recall_curve(y_labeled, best_oof)
ap_score      = average_precision_score(y_labeled, best_oof)
baseline_pr   = y_labeled.mean()

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(rec, prec, color=ORANGE, lw=2.5, label=f'{best_name}  (AP = {ap_score:.4f})')
ax.axhline(baseline_pr, color='grey', lw=1, linestyle='--',
           label=f'Baseline (cheating rate = {baseline_pr:.2f})')
ax.fill_between(rec, prec, alpha=0.08, color=ORANGE)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.legend(loc='upper right')
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
fig.tight_layout()
fig.savefig(f'{OUT}fig4_pr_curve.png', dpi=150, bbox_inches='tight')
plt.show()
print('Fig 4 saved')

# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Top 20 Feature Importance (XGBoost)
# ══════════════════════════════════════════════════════════════════════════════
xgb_model   = xgb_final.model
importances = xgb_model.feature_importances_
feat_df     = (pd.DataFrame({'feature': feature_cols, 'importance': importances})
               .sort_values('importance', ascending=True)
               .tail(20))

colors_imp = [TEAL if i >= 15 else BLUE for i in range(len(feat_df))]

fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(feat_df['feature'], feat_df['importance'], color=colors_imp, edgecolor='white')
ax.set_xlabel('Feature Importance (XGBoost gain)')
ax.set_title('Top 20 Most Important Features')
ax.tick_params(axis='y', labelsize=9)
fig.tight_layout()
fig.savefig(f'{OUT}fig5_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print('Fig 5 saved')

# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — OOF Score Distribution by Class
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(best_oof[y_arr == 0], bins=60, alpha=0.6, color=GREEN,
        label='Legitimate', density=True)
ax.hist(best_oof[y_arr == 1], bins=60, alpha=0.6, color=ORANGE,
        label='Cheater', density=True)
ax.axvline(tp, color='black', lw=1.5, linestyle='--', label=f't_pass = {tp:.2f}')
ax.axvline(tb, color='red',   lw=1.5, linestyle='--', label=f't_block = {tb:.2f}')
ymax = ax.get_ylim()[1]
ax.fill_betweenx([0, ymax], tp, tb, alpha=0.07, color='yellow', label='Review zone')
ax.set_ylim(0, ymax)
ax.set_xlabel('Predicted Cheating Probability')
ax.set_ylabel('Density')
ax.set_title('OOF Score Distribution — Cheaters vs Legitimate Users')
ax.legend()
fig.tight_layout()
fig.savefig(f'{OUT}fig6_score_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print('Fig 6 saved')

# ══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Cost Sensitivity Heatmap
# ══════════════════════════════════════════════════════════════════════════════
t_pass_vals  = np.arange(0.05, 0.45, 0.05)
t_block_vals = np.arange(0.55, 1.00, 0.05)
cost_grid    = np.zeros((len(t_pass_vals), len(t_block_vals)))

for i, tp_val in enumerate(t_pass_vals):
    for j, tb_val in enumerate(t_block_vals):
        cost_grid[i, j] = competition_cost(y_labeled.values, best_oof, tp_val, tb_val)

opt_ij  = np.unravel_index(np.argmin(cost_grid), cost_grid.shape)

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(cost_grid / 1e6, aspect='auto', cmap='RdYlGn_r', origin='lower')
ax.set_xticks(range(len(t_block_vals)))
ax.set_xticklabels([f'{v:.2f}' for v in t_block_vals], rotation=45)
ax.set_yticks(range(len(t_pass_vals)))
ax.set_yticklabels([f'{v:.2f}' for v in t_pass_vals])
ax.set_xlabel('t_block (auto-block threshold)')
ax.set_ylabel('t_pass (auto-pass threshold)')
ax.set_title('Cost Sensitivity Heatmap — Total Cost ($M) by Threshold Pair')
ax.plot(opt_ij[1], opt_ij[0], 'w*', markersize=14, label='Optimal')
ax.legend()
plt.colorbar(im, ax=ax, label='Total Cost ($M)')
fig.tight_layout()
fig.savefig(f'{OUT}fig7_cost_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print('Fig 7 saved')

# ══════════════════════════════════════════════════════════════════════════════
# FIG 8 — Ensemble & Pseudo-labeling Impact (zoomed)
# ══════════════════════════════════════════════════════════════════════════════
stage_keys   = ['XGBoost', 'LightGBM', 'Ensemble', 'Ensemble+Pseudo']
stage_labels = ['XGBoost\n(baseline)', 'LightGBM\n(baseline)',
                'Ensemble\n(blend)', 'Ensemble\n+Pseudo']
stage_costs  = [results[k]['min_cost'] for k in stage_keys if k in results]
s_labels     = [stage_labels[i] for i, k in enumerate(stage_keys) if k in results]
s_colors     = [BLUE, BLUE, ORANGE, TEAL][:len(stage_costs)]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(s_labels, [c / 1e6 for c in stage_costs],
              color=s_colors, edgecolor='white', width=0.5)
for bar, cost in zip(bars, stage_costs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
            f'${cost / 1e6:.3f}M', ha='center', va='bottom',
            fontsize=10, fontweight='bold')

margin = (max(stage_costs) - min(stage_costs)) / 1e6
ax.set_ylim(min(stage_costs) / 1e6 - margin, max(stage_costs) / 1e6 + margin * 3)
ax.set_ylabel('Total Cost (Millions USD)')
ax.set_title('Ensemble & Pseudo-labeling Impact on Cost')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'${v:.2f}M'))

legend_handles = [
    mpatches.Patch(color=BLUE,   label='Individual models'),
    mpatches.Patch(color=ORANGE, label='Ensemble blend'),
    mpatches.Patch(color=TEAL,   label='Best: Ensemble + Pseudo-labels'),
]
ax.legend(handles=legend_handles, loc='upper right', fontsize=9)
fig.tight_layout()
fig.savefig(f'{OUT}fig8_pseudo_impact.png', dpi=150, bbox_inches='tight')
plt.show()
print('Fig 8 saved')

print('\nAll 8 figures saved to', OUT)
