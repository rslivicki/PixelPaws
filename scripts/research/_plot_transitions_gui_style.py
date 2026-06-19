"""GUI-style transitions + occupancy plots for the FormOxy/Left_paws
project. Reads the per-session state sequences saved by
_compute_transitions_formoxy.py and renders:

  1) Transition matrix per treatment group + pairwise diff vs Vehicle
     (inferno palette; diff uses |Δ| brightness with signed text).
  2) Fractional occupancy grouped bar chart (one bar per treatment per
     behavior) — matches the Transitions tab `_plot_state_occupancy`
     style, but in the inferno palette.
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT = r'E:/RSVIDS/Blackbox/2512_Blackbox_Formalin_Oxy/Left_paws'
OUT_DIR = os.path.join(PROJECT, 'transitions')
STATES_DIR = os.path.join(OUT_DIR, '_states')
JSON_PATH = os.path.join(OUT_DIR, 'Config1.json')

TREAT_ORDER = ['Veh', 'Oxy1', 'Oxy3', 'Oxy10']
PALETTE = 'inferno'
EXCLUDE_SESSIONS = {'2512_FormOxy_S6'}  # zero-occupancy outlier


def transition_matrix(seq: np.ndarray, n_states: int) -> np.ndarray:
    """First-order transition matrix. Rows = from-state, cols = to-state.
    Row-normalized to conditional probabilities P(j | i)."""
    mat = np.zeros((n_states, n_states), dtype=float)
    if len(seq) < 2:
        return mat
    # Count transitions
    for a, b in zip(seq[:-1], seq[1:]):
        if a != b:  # skip self-loops to make off-diagonal structure visible
            mat[a, b] += 1
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return mat / row_sums


def render_panel(ax, mat_color, mat_text, title, *, vmin, vmax, labels,
                 fmt='{:.2f}', signed_text=False):
    n = mat_color.shape[0]
    im = ax.imshow(mat_color, cmap=PALETTE, vmin=vmin, vmax=vmax, aspect='equal')
    rng = max(vmax - vmin, 1e-9)
    for i in range(n):
        for j in range(n):
            v_color = mat_color[i, j]
            v_text = mat_text[i, j]
            txt_col = 'white' if (v_color - vmin) / rng < 0.55 else 'black'
            ax.text(j, i, fmt.format(v_text) if signed_text else f'{v_text:.2f}',
                    ha='center', va='center', color=txt_col, fontsize=9)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
    ax.set_yticklabels(labels, rotation=0, fontsize=9)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel('To', fontsize=10)
    ax.set_ylabel('From', fontsize=10)
    ax.tick_params(length=0)
    return im


def main():
    cfg = json.load(open(JSON_PATH))
    sessions = cfg['session_selection']
    classifiers = cfg['classifiers']
    behavior_names = []
    for c in classifiers:
        n = os.path.basename(c['path']).replace('.pkl', '').replace(
            'PixelPaws_', '').replace('_AllFeatures', '')
        behavior_names.append(n)
    state_labels = ['Other'] + behavior_names
    n_states = len(state_labels)
    print(f'[setup] {len(sessions)} sessions, {n_states} states '
          f'({state_labels})')

    key_df = pd.read_excel(cfg['key_file'])
    treat_map = dict(zip(key_df['Subject'], key_df['Treatment']))

    # Load each session's state sequence (skipping any in EXCLUDE_SESSIONS)
    seqs = {}
    for s in sessions:
        if s in EXCLUDE_SESSIONS:
            print(f'  [skip] {s} (excluded)')
            continue
        npy = os.path.join(STATES_DIR, f'{s}_states.npy')
        if not os.path.isfile(npy):
            print(f'  [warn] missing {npy}')
            continue
        seqs[s] = np.load(npy)
    print(f'[load ] {len(seqs)} state sequences')

    # ── per-session transition matrices, then averaged per group ────
    by_group_mats = {t: [] for t in TREAT_ORDER}
    by_group_occ  = {t: [] for t in TREAT_ORDER}
    for sess, seq in seqs.items():
        m = transition_matrix(seq, n_states)
        treat = treat_map.get(sess, 'Unknown')
        if treat not in by_group_mats:
            by_group_mats[treat] = []
            by_group_occ[treat]  = []
        by_group_mats[treat].append(m)
        # Drop the "Other" state from occupancy; renormalize? GUI keeps raw.
        occ = np.array([float(np.mean(seq == sid)) for sid in range(n_states)])
        by_group_occ[treat].append(occ)

    group_mats = {g: (np.mean(np.stack(ms), axis=0) if ms else None)
                  for g, ms in by_group_mats.items()}
    # Drop "Other" row + col so the heatmap shows behavior-to-behavior only
    keep = list(range(1, n_states))  # indices 1..N-1
    keep_labels = [state_labels[i] for i in keep]

    # Collect groups that have data, in TREAT_ORDER
    available = [g for g in TREAT_ORDER if group_mats.get(g) is not None]
    print(f'[grps ] available={available}')

    # ── Figure 1: transition-matrix grid (2 rows) ───────────────────
    # Row 1: K group panels.  Row 2: (K-1) diff-vs-Vehicle panels.
    K = len(available)
    n_diff = max(K - 1, 0)
    n_cols = max(K, n_diff)
    n_rows = 1 + (1 if n_diff > 0 else 0)
    panel_w, panel_h = 4.0, 4.4
    fig_w = max(12.0, panel_w * n_cols + 0.9 * n_cols)
    fig_h = panel_h * n_rows + 1.0
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                             gridspec_kw={'wspace': 0.45, 'hspace': 0.55})
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    # Row 0 — group panels
    for col in range(n_cols):
        ax = axes[0, col]
        if col >= K:
            ax.axis('off')
            continue
        grp = available[col]
        m = group_mats[grp][np.ix_(keep, keep)]
        rs = m.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
        m = m / rs
        im = render_panel(ax, m, m, grp,
                          vmin=0.0, vmax=1.0, labels=keep_labels)
        cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03, aspect=18)
        cbar.outline.set_linewidth(0.5)

    # Row 1 — diff vs Veh panels
    if n_diff > 0 and 'Veh' in available:
        veh_mat = group_mats['Veh'][np.ix_(keep, keep)]
        rs = veh_mat.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
        veh_mat = veh_mat / rs
        diff_groups = [g for g in available if g != 'Veh']
        for col in range(n_cols):
            ax = axes[1, col]
            if col >= len(diff_groups):
                ax.axis('off')
                continue
            grp = diff_groups[col]
            other = group_mats[grp][np.ix_(keep, keep)]
            rs = other.sum(axis=1, keepdims=True); rs[rs == 0] = 1.0
            other = other / rs
            diff = other - veh_mat
            mag = np.abs(diff)
            vmax = max(0.05, np.ceil(mag.max() * 20) / 20)
            im = render_panel(ax, mag, diff,
                              f'{grp} − Veh',
                              vmin=0.0, vmax=vmax,
                              labels=keep_labels,
                              fmt='{:+.2f}', signed_text=True)
            cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03, aspect=18)
            cbar.set_label(f'|{grp} − Veh|', fontsize=8)
            cbar.outline.set_linewidth(0.5)

    fig.suptitle('FormOxy / Left_paws — first-order transition matrices '
                 '(off-diagonal P(j|i))', fontsize=13, fontweight='bold',
                 y=0.995)
    out_png = os.path.join(OUT_DIR, '_transitions_matrix_by_group.png')
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_png}')

    # ── Figure 2: GUI-style fractional occupancy bars ───────────────
    # Drop "Other" from the x-axis (matches GUI behaviour for behavior-only
    # classifiers) and show one bar per treatment per behavior.
    INFERNO = plt.cm.inferno
    n_b = len(behavior_names)
    x = np.arange(n_b)
    n_g = len(available)
    width = 0.78 / n_g
    grp_colors = {t: INFERNO(0.20 + 0.62 * (i / max(len(TREAT_ORDER) - 1, 1)))
                  for i, t in enumerate(TREAT_ORDER)}

    fig2, ax = plt.subplots(figsize=(max(8.0, 1.3 * n_b * n_g), 4.6),
                            constrained_layout=True)
    for gi, grp in enumerate(available):
        offset = (gi - n_g / 2 + 0.5) * width
        occs = np.stack(by_group_occ[grp])  # (n_subj, n_states)
        # drop "Other" col
        occs_b = occs[:, 1:]
        means = occs_b.mean(axis=0)
        sems  = occs_b.std(axis=0) / np.sqrt(max(len(occs_b), 1))
        ax.bar(x + offset, means, width, yerr=sems, capsize=4,
               color=grp_colors[grp], edgecolor='black', linewidth=0.6,
               error_kw={'ecolor': '#222', 'elinewidth': 1.0,
                         'capthick': 1.0}, zorder=2,
               label=f'{grp} (n={len(occs_b)})')
        # Per-subject dots
        rng = np.random.RandomState(gi)
        jitter = (rng.rand(len(occs_b), n_b) - 0.5) * (width * 0.6)
        for si in range(len(occs_b)):
            ax.scatter(x + offset + jitter[si], occs_b[si],
                       s=22, color='white', edgecolor='black',
                       linewidth=0.5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(behavior_names, fontsize=10)
    ax.set_ylabel('Fraction of time', fontsize=11)
    ax.set_title('Fractional occupancy by treatment '
                 '(FormOxy / Left_paws)',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linewidth=0.5, color='#bbb', zorder=1)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=n_g, loc='upper right', fontsize=9)

    out_png2 = os.path.join(OUT_DIR, '_occupancy_grouped_bars.png')
    fig2.savefig(out_png2, dpi=180, bbox_inches='tight')
    plt.close(fig2)
    print(f'wrote {out_png2}')


if __name__ == '__main__':
    main()
