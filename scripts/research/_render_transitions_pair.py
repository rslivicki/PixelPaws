"""Quick render: two-panel transition matrix (Naive + SNTX) with
identical styling on both panels, matching the GUI's inferno theme."""

import numpy as np
import matplotlib.pyplot as plt

LABELS = ['Facial_grooming', 'Left_licking', 'rearing', 'still', 'walking']

NAIVE = np.array([
    [0.00, 0.03, 0.85, 0.00, 0.12],
    [0.09, 0.00, 0.23, 0.00, 0.28],
    [0.24, 0.01, 0.00, 0.00, 0.75],
    [0.04, 0.00, 0.06, 0.00, 0.90],
    [0.08, 0.01, 0.76, 0.16, 0.00],
])
SNTX = np.array([
    [0.00, 0.14, 0.71, 0.00, 0.16],
    [0.44, 0.00, 0.24, 0.00, 0.32],
    [0.22, 0.04, 0.00, 0.00, 0.74],
    [0.02, 0.01, 0.04, 0.00, 0.94],
    [0.11, 0.00, 0.57, 0.31, 0.00],
])

PALETTE = 'inferno'
VMIN, VMAX = 0.0, 1.0


def render_panel(ax, mat, title, labels):
    n = mat.shape[0]
    im = ax.imshow(mat, cmap=PALETTE, vmin=VMIN, vmax=VMAX, aspect='equal')
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            # text colour: white on dark cells, black on bright cells
            txt_col = 'white' if v < 0.55 else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    color=txt_col, fontsize=10)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
    ax.set_yticklabels(labels, rotation=0, fontsize=9)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
    ax.set_xlabel('To', fontsize=11)
    ax.set_ylabel('From', fontsize=11)
    ax.tick_params(length=0)
    return im


fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8),
                         gridspec_kw={'wspace': 0.40})
ims = []
for ax, mat, title in zip(axes, [NAIVE, SNTX], ['Naive', 'SNTX']):
    ims.append(render_panel(ax, mat, title, LABELS))

# matched colorbars (identical extent + style)
for ax, im in zip(axes, ims):
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03, aspect=18)
    cbar.outline.set_linewidth(0.5)

out = r'E:/PixelPaws/analysis_output/_transitions_pair_test.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'wrote {out}')
