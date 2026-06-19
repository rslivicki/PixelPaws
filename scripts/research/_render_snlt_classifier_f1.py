"""Single-graph F1 summary for all classifiers in for_claude.json
(SNLT Baseline). Mean CV-F1 bars with ±std, per-fold scores as dots,
OOF best F1 (where available) as a star marker."""

import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt

JSON_PATH = r'E:/RSVIDS/Blackbox/2603_SNLT_JG/Baseline/transitions/for_claude.json'
OUT_PATH  = r'E:/RSVIDS/Blackbox/2603_SNLT_JG/Baseline/transitions/_classifier_f1.png'

cfg = json.load(open(JSON_PATH))
clfs = cfg['classifiers']

names, mean_f1s, std_f1s, fold_lists, oof_f1s = [], [], [], [], []
for c in clfs:
    p = c['path']
    base = os.path.basename(p).replace('.pkl', '').replace('PixelPaws_', '')
    base = base.replace('_AllFeatures', '')
    d = joblib.load(p)
    mf = float(d.get('mean_cv_f1', np.nan))
    sf = float(d.get('std_cv_f1', 0.0))
    folds = list(d.get('cv_f1_scores', []) or [])
    oof = d.get('oof_best_f1', None)
    names.append(base)
    mean_f1s.append(mf)
    std_f1s.append(sf)
    fold_lists.append(folds)
    oof_f1s.append(float(oof) if oof is not None else np.nan)
    print(f'{base:25s}  mean_cv_f1={mf:.3f}±{sf:.3f}  '
          f'folds={folds}  oof={oof}')

# ── inferno-palette colors per classifier ────────────────────────────
INFERNO = plt.cm.inferno
n = len(names)
colors = [INFERNO(0.20 + 0.55 * (i / max(n - 1, 1))) for i in range(n)]

fig, ax = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
x = np.arange(n)
bar_w = 0.62

bars = ax.bar(x, mean_f1s, bar_w,
              yerr=std_f1s, capsize=5,
              color=colors, edgecolor='black', linewidth=0.6,
              error_kw={'ecolor': '#222', 'elinewidth': 1.2,
                        'capthick': 1.2}, zorder=2,
              label='Mean CV-F1 (±std)')

# Per-fold dots
for xi, folds in zip(x, fold_lists):
    if folds:
        jitter = (np.random.RandomState(0).rand(len(folds)) - 0.5) * 0.18
        ax.scatter(np.full(len(folds), xi) + jitter, folds,
                   s=28, color='white', edgecolor='black',
                   linewidth=0.8, zorder=3, label='Per-fold' if xi == 0 else None)

# Numeric labels above each bar (mean F1)
for xi, mf in zip(x, mean_f1s):
    ax.text(xi, mf + 0.025, f'{mf:.3f}',
            ha='center', va='bottom', fontsize=9.5,
            color='#222', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=10)
ax.set_ylim(0, 1.05)
ax.set_yticks(np.linspace(0, 1, 11))
ax.set_ylabel('F1', fontsize=11)
ax.set_title('SNLT Baseline classifiers — CV-F1 summary',
             fontsize=12, fontweight='bold')
ax.grid(axis='y', linewidth=0.5, color='#bbb', zorder=1)
ax.set_axisbelow(True)

# Dedupe legend entries
handles, labels = ax.get_legend_handles_labels()
seen = set(); h2, l2 = [], []
for h, l in zip(handles, labels):
    if l not in seen:
        seen.add(l); h2.append(h); l2.append(l)
ax.legend(h2, l2, frameon=False, loc='lower center',
          bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)

fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'\nwrote {OUT_PATH}')
