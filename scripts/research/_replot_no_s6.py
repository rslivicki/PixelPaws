"""Quick re-plot from existing _occupancy_fractions.csv, dropping S6."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = r'E:/RSVIDS/Blackbox/2512_Blackbox_Formalin_Oxy/Left_paws/transitions'
df = pd.read_csv(os.path.join(OUT_DIR, '_occupancy_fractions.csv'))
df = df[df['session'] != '2512_FormOxy_S6']
df = df[df['behavior'] != 'Other']

TREAT_ORDER = ['Veh', 'Oxy1', 'Oxy3', 'Oxy10']
behaviors = ['Facial_grooming', 'Left_licking', 'rearing', 'still', 'walking']
INFERNO = plt.cm.inferno
treat_colors = {t: INFERNO(0.20 + 0.62 * (i / 3))
                for i, t in enumerate(TREAT_ORDER)}

fig, axes = plt.subplots(1, len(behaviors), figsize=(3.4 * len(behaviors), 4.2),
                         sharey=True, constrained_layout=True)
for ax, bname in zip(axes, behaviors):
    sub = df[df['behavior'] == bname]
    means, stds, ns = [], [], []
    for t in TREAT_ORDER:
        v = sub[sub['treatment'] == t]['fraction'].values
        means.append(v.mean() if len(v) else 0.0)
        stds.append(v.std() if len(v) > 1 else 0.0)
        ns.append(len(v))
    x = np.arange(len(TREAT_ORDER))
    ax.bar(x, means, 0.65, yerr=stds, capsize=4,
           color=[treat_colors[t] for t in TREAT_ORDER],
           edgecolor='black', linewidth=0.6,
           error_kw={'ecolor': '#222', 'elinewidth': 1.2,
                     'capthick': 1.2}, zorder=2)
    for ti, t in enumerate(TREAT_ORDER):
        v = sub[sub['treatment'] == t]['fraction'].values
        jit = (np.random.RandomState(ti).rand(len(v)) - 0.5) * 0.20
        ax.scatter(np.full(len(v), ti) + jit, v, s=22, color='white',
                   edgecolor='black', linewidth=0.6, zorder=3)
    for xi, mv in zip(x, means):
        ax.text(xi, mv + 0.012, f'{mv:.2f}', ha='center', va='bottom',
                fontsize=8.5, color='#222', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t}\n(n={n})' for t, n in zip(TREAT_ORDER, ns)],
                       fontsize=9)
    ax.set_title(bname, fontsize=11, fontweight='bold')
    ax.grid(axis='y', linewidth=0.4, color='#bbb', zorder=1)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(0.05, df['fraction'].max() * 1.15))
axes[0].set_ylabel('Fractional occupancy', fontsize=11)
fig.suptitle('FormOxy / Left_paws — behavior occupancy by treatment '
             '(S6 excluded)', fontsize=12, fontweight='bold')

out = os.path.join(OUT_DIR, '_occupancy_fractions_no_S6.png')
fig.savefig(out, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'wrote {out}')
