"""Three-panel transition matrix preview: Naive | SNTX | (SNTX - Naive)
all rendered in inferno. Diff panel encodes magnitude via inferno
brightness; sign is shown in the cell text."""

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
DIFF = SNTX - NAIVE

PALETTE = 'inferno'


def render_panel(ax, mat_color, mat_text, title, *, vmin, vmax,
                 fmt='{:+.2f}', signed_text=False):
    n = mat_color.shape[0]
    im = ax.imshow(mat_color, cmap=PALETTE, vmin=vmin, vmax=vmax,
                   aspect='equal')
    rng = vmax - vmin
    for i in range(n):
        for j in range(n):
            v_color = mat_color[i, j]
            v_text  = mat_text[i, j]
            # text colour: white on dark cells, black on bright cells
            txt_col = 'white' if (v_color - vmin) / rng < 0.55 else 'black'
            ax.text(j, i, fmt.format(v_text) if signed_text else f'{v_text:.2f}',
                    ha='center', va='center', color=txt_col, fontsize=10)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(LABELS, rotation=35, ha='right', fontsize=9)
    ax.set_yticklabels(LABELS, rotation=0, fontsize=9)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
    ax.set_xlabel('To', fontsize=11)
    ax.set_ylabel('From', fontsize=11)
    ax.tick_params(length=0)
    return im


fig, axes = plt.subplots(1, 3, figsize=(15.8, 4.8),
                         gridspec_kw={'wspace': 0.42})

# Panel 1 + 2: Naive, SNTX (raw probs, vmin=0, vmax=1)
im0 = render_panel(axes[0], NAIVE, NAIVE, 'Naive', vmin=0.0, vmax=1.0)
im1 = render_panel(axes[1], SNTX,  SNTX,  'SNTX',  vmin=0.0, vmax=1.0)

# Panel 3: diff — inferno mapped to |diff|, text shows signed value
diff_mag = np.abs(DIFF)
diff_max = max(0.05, np.ceil(diff_mag.max() * 20) / 20)  # round up to nearest 0.05
im2 = render_panel(axes[2], diff_mag, DIFF, 'SNTX − Naive',
                   vmin=0.0, vmax=diff_max,
                   fmt='{:+.2f}', signed_text=True)

# Matched colorbars
for ax, im in zip(axes, [im0, im1, im2]):
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03, aspect=18)
    cbar.outline.set_linewidth(0.5)

# Diff colorbar label clarifies the magnitude semantics
axes[2].images[0].colorbar.set_label('|SNTX − Naive|', fontsize=9)

out = r'E:/PixelPaws/analysis_output/_transitions_diff_inferno_test.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'wrote {out}')
