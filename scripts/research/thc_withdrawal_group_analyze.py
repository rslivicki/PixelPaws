"""
Post-predict: per-group transition + immobility analysis for THC
withdrawal cohort. Reads <PROJECT>/results/<session>_predictions.csv
files, applies argmax assignment per the SNLT for_claude.json spec,
and emits:

  - per-session state-occupancy CSV (% time per state)
  - per-group/condition transition matrices (avg over sessions)
  - PNG plots: time-immobile by group/condition, transition matrices
  - Discord file uploads via webhook

Group mapping (per user): odd-numbered mouse = THC, even = vehicle.
Each mouse has Baseline + Postdrug recordings.

Run:
  PYTHONIOENCODING=utf-8 py -X utf8 scripts/research/thc_withdrawal_group_analyze.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Headless backend for plot generation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(r'E:\RSVIDS\Blackbox\260506_RS_THC_Withdrawal')
RESULTS_DIR = PROJECT_ROOT / 'results'
ANALYSIS_DIR = PROJECT_ROOT / 'analysis'
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
JSON_PATH = Path(r'E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\transitions\for_claude.json')

WEBHOOK = (
    ""
)


# Behaviors in argmax priority order (matches for_claude.json)
PRIORITY = ['Facial_grooming', 'Left_licking', 'rearing', 'still', 'walking']
NOISE_LABEL = 'noise'


def parse_session(name: str):
    """THC<n>_<Baseline|Postdrug> -> (mouse_num:int, condition:str, group:str)"""
    m = re.match(r'^THC(\d+)_(Baseline|Postdrug)$', name)
    if not m:
        return None
    n = int(m.group(1))
    cond = m.group(2)
    group = 'THC' if (n % 2 == 1) else 'Vehicle'
    return n, cond, group


def assign_states(df: pd.DataFrame, priority: list[str]) -> np.ndarray:
    """Argmax assignment honouring priority. Higher priority wins ties.

    For each frame, the assigned state is the highest-priority behavior
    whose binary column == 1. Frames with no behavior on get NOISE_LABEL.
    """
    n = len(df)
    states = np.full(n, NOISE_LABEL, dtype=object)
    # Walk priority list in reverse so earlier (higher priority) overwrites later
    for behavior in reversed(priority):
        if behavior in df.columns:
            mask = df[behavior].astype(int).to_numpy() == 1
            states[mask] = behavior
    return states


EXCLUDE_NOISE = True  # Set False to keep 'other' frames in analysis.


def state_occupancy(states: np.ndarray, all_states: list[str]) -> dict:
    """Return {state: fraction_of_frames}.

    When EXCLUDE_NOISE, the denominator is restricted to non-noise
    frames so percentages reflect time spent in *labeled* behaviors
    only. Noise fraction is still reported for transparency.
    """
    n_total = len(states)
    out = {s: 0.0 for s in all_states}
    out[NOISE_LABEL] = 0.0
    vals, counts = np.unique(states, return_counts=True)
    raw = dict(zip(vals, counts))
    n_noise = int(raw.get(NOISE_LABEL, 0))
    n_labeled = n_total - n_noise

    out[NOISE_LABEL] = n_noise / max(1, n_total)
    denom = n_labeled if (EXCLUDE_NOISE and n_labeled > 0) else max(1, n_total)
    for s in all_states:
        out[s] = raw.get(s, 0) / denom
    return out


def transition_matrix(states: np.ndarray, all_states: list[str]) -> pd.DataFrame:
    """Bout-mode transition counts (collapse runs to single states).

    With EXCLUDE_NOISE, transitions involving the NOISE_LABEL state are
    omitted: noise runs are skipped (we record A->B if A and B are
    adjacent labeled bouts even with noise gaps between).

    Returns a (state, state) DataFrame of transition counts (not
    normalised). Rows are 'from', columns are 'to'.
    """
    if len(states) == 0:
        idx = all_states + ([] if EXCLUDE_NOISE else [NOISE_LABEL])
        return pd.DataFrame(0, index=idx, columns=idx)

    # Collapse to bouts
    bouts = [states[0]]
    for s in states[1:]:
        if s != bouts[-1]:
            bouts.append(s)

    if EXCLUDE_NOISE:
        # Drop noise bouts entirely so transitions go labeled -> labeled
        bouts = [b for b in bouts if b != NOISE_LABEL]
        idx = all_states
    else:
        idx = all_states + [NOISE_LABEL]

    M = pd.DataFrame(0, index=idx, columns=idx)
    for a, b in zip(bouts[:-1], bouts[1:]):
        if a in M.index and b in M.columns:
            M.loc[a, b] += 1
    return M


def normalise_rows(M: pd.DataFrame) -> pd.DataFrame:
    rs = M.sum(axis=1).replace(0, 1)
    return M.div(rs, axis=0)


def plot_immobility(occ_df: pd.DataFrame, png: Path) -> None:
    """Three-panel: Postdrug THC vs Vehicle (primary), Baseline THC vs
    Vehicle, paired within-subject Baseline vs Postdrug per group."""
    palette = {'THC': '#d62728', 'Vehicle': '#1f77b4'}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def _box(ax, sub_df, group_col, title):
        groups = sorted(sub_df[group_col].unique())
        data = [(sub_df[sub_df[group_col] == g]['still'].to_numpy() * 100).tolist()
                for g in groups]
        bp = ax.boxplot(data, labels=groups, patch_artist=True, widths=0.55,
                        medianprops=dict(color='black'))
        for patch, g in zip(bp['boxes'], groups):
            patch.set_facecolor(palette.get(g, '#888'))
            patch.set_alpha(0.6)
        for i, (g, ys) in enumerate(zip(groups, data), 1):
            if ys:
                xs = np.random.normal(i, 0.05, len(ys))
                ax.scatter(xs, ys, color='black', s=20, zorder=3)
                ax.text(i, max(ys) + 1, f'n={len(ys)}',
                        ha='center', fontsize=9, color='gray')
        ax.set_ylabel('% time immobile (still)')
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)

    # Panel 1 (PRIMARY): Postdrug THC vs Vehicle
    pd_sub = occ_df[occ_df['condition'] == 'Postdrug']
    _box(axes[0], pd_sub, 'group', 'POSTDRUG: THC vs Vehicle (primary)')

    # Panel 2: Baseline THC vs Vehicle
    bl_sub = occ_df[occ_df['condition'] == 'Baseline']
    _box(axes[1], bl_sub, 'group', 'Baseline: THC vs Vehicle')

    # Panel 3: paired within-subject (Baseline -> Postdrug per mouse)
    ax = axes[2]
    for grp in ['THC', 'Vehicle']:
        sub = occ_df[occ_df['group'] == grp]
        for mouse, mdf in sub.groupby('mouse'):
            if len(mdf) >= 2:
                bl = mdf[mdf['condition'] == 'Baseline']['still']
                po = mdf[mdf['condition'] == 'Postdrug']['still']
                if len(bl) and len(po):
                    ax.plot([0, 1], [bl.iloc[0] * 100, po.iloc[0] * 100],
                            'o-', color=palette[grp], alpha=0.7,
                            label=f'mouse {mouse} ({grp})')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Baseline', 'Postdrug'])
    ax.set_ylabel('% time immobile (still)')
    ax.set_title('Within-subject: Baseline -> Postdrug')
    ax.grid(axis='y', alpha=0.3)
    # Legend by group only (not per mouse)
    handles = [plt.Line2D([0], [0], color=palette[g], marker='o',
                          linestyle='-', label=g) for g in ['THC', 'Vehicle']]
    ax.legend(handles=handles, loc='best')

    fig.suptitle('Time spent immobile - THC withdrawal cohort',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    plt.close(fig)


def plot_state_occupancy(occ_df: pd.DataFrame, png: Path) -> None:
    """Bar chart of mean % time per state, grouped by condition*group."""
    states_to_plot = PRIORITY if EXCLUDE_NOISE else PRIORITY + [NOISE_LABEL]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, cond in zip(axes, ['Baseline', 'Postdrug']):
        sub = occ_df[occ_df['condition'] == cond]
        x = np.arange(len(states_to_plot))
        w = 0.35
        for i, grp in enumerate(['THC', 'Vehicle']):
            grp_sub = sub[sub['group'] == grp]
            means = [grp_sub[s].mean() * 100 if s in grp_sub.columns else 0.0
                     for s in states_to_plot]
            sems = [grp_sub[s].sem() * 100 if s in grp_sub.columns else 0.0
                    for s in states_to_plot]
            color = '#d62728' if grp == 'THC' else '#1f77b4'
            ax.bar(x + (i - 0.5) * w, means, w, yerr=sems, label=grp,
                   color=color, alpha=0.7, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(states_to_plot, rotation=30, ha='right')
        ax.set_ylabel('% time')
        ax.set_title(cond)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Per-state occupancy — THC withdrawal cohort')
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    plt.close(fig)


def plot_transition_heatmaps(group_mats: dict, png: Path) -> None:
    """4-panel heatmap: (group x condition) row-normalised transition probs."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for r, cond in enumerate(['Baseline', 'Postdrug']):
        for c, grp in enumerate(['THC', 'Vehicle']):
            ax = axes[r, c]
            M = group_mats.get((grp, cond))
            if M is None or M.sum().sum() == 0:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center')
                ax.set_title(f'{grp} / {cond}')
                continue
            P = normalise_rows(M)
            im = ax.imshow(P.to_numpy(), cmap='inferno', vmin=0, vmax=1)
            ax.set_xticks(range(len(P.columns)))
            ax.set_xticklabels(P.columns, rotation=45, ha='right')
            ax.set_yticks(range(len(P.index)))
            ax.set_yticklabels(P.index)
            ax.set_title(f'{grp} / {cond}')
            ax.set_xlabel('to')
            ax.set_ylabel('from')
            for i in range(P.shape[0]):
                for j in range(P.shape[1]):
                    v = P.iloc[i, j]
                    if v > 0.01:
                        ax.text(j, i, f'{v:.2f}',
                                ha='center', va='center',
                                color='white' if v < 0.5 else 'black',
                                fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle('Transition probabilities (row-normalised) - THC withdrawal')
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    plt.close(fig)


def _draw_diff(ax, A, B, title, vlim=0.4):
    """Heatmap of (A - B) row-normalised transition probabilities."""
    if A is None or B is None or A.sum().sum() == 0 or B.sum().sum() == 0:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center')
        ax.set_title(title)
        return
    PA = normalise_rows(A)
    PB = normalise_rows(B)
    idx = PA.index.union(PB.index)
    cols = PA.columns.union(PB.columns)
    PA = PA.reindex(index=idx, columns=cols, fill_value=0)
    PB = PB.reindex(index=idx, columns=cols, fill_value=0)
    D = (PA - PB).to_numpy()
    im = ax.imshow(D, cmap='RdBu_r', vmin=-vlim, vmax=vlim)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha='right')
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(idx)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('to')
    ax.set_ylabel('from')
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            v = D[i, j]
            if abs(v) > 0.03:
                ax.text(j, i, f'{v:+.2f}', ha='center', va='center',
                        color='black', fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_transition_diffs(group_mats: dict, png: Path) -> None:
    """Difference-matrix grid for the key contrasts: between-group at
    each condition, within-subject per group, and the drug x group
    interaction."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    _draw_diff(axes[0, 0],
               group_mats.get(('THC', 'Postdrug')),
               group_mats.get(('Vehicle', 'Postdrug')),
               'POSTDRUG: THC - Vehicle (primary between)')
    _draw_diff(axes[0, 1],
               group_mats.get(('THC', 'Baseline')),
               group_mats.get(('Vehicle', 'Baseline')),
               'Baseline: THC - Vehicle (between, control)')

    M_thc_p = group_mats.get(('THC', 'Postdrug'))
    M_thc_b = group_mats.get(('THC', 'Baseline'))
    M_veh_p = group_mats.get(('Vehicle', 'Postdrug'))
    M_veh_b = group_mats.get(('Vehicle', 'Baseline'))
    if all(m is not None and m.sum().sum() > 0
           for m in (M_thc_p, M_thc_b, M_veh_p, M_veh_b)):
        idx = normalise_rows(M_thc_p).index.union(normalise_rows(M_veh_p).index)
        cols = normalise_rows(M_thc_p).columns.union(normalise_rows(M_veh_p).columns)
        thc_delta = (normalise_rows(M_thc_p).reindex(index=idx, columns=cols, fill_value=0)
                     - normalise_rows(M_thc_b).reindex(index=idx, columns=cols, fill_value=0))
        veh_delta = (normalise_rows(M_veh_p).reindex(index=idx, columns=cols, fill_value=0)
                     - normalise_rows(M_veh_b).reindex(index=idx, columns=cols, fill_value=0))
        D = (thc_delta - veh_delta).to_numpy()
        ax = axes[0, 2]
        im = ax.imshow(D, cmap='RdBu_r', vmin=-0.3, vmax=0.3)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha='right')
        ax.set_yticks(range(len(idx))); ax.set_yticklabels(idx)
        ax.set_title('Drug x Group interaction:\n(THC: P-B) - (Veh: P-B)',
                     fontsize=10)
        ax.set_xlabel('to'); ax.set_ylabel('from')
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                v = D[i, j]
                if abs(v) > 0.03:
                    ax.text(j, i, f'{v:+.2f}', ha='center', va='center',
                            color='black', fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        axes[0, 2].text(0.5, 0.5, 'incomplete data', ha='center', va='center')
        axes[0, 2].set_title('Drug x Group interaction')

    _draw_diff(axes[1, 0],
               group_mats.get(('THC', 'Postdrug')),
               group_mats.get(('THC', 'Baseline')),
               'THC: Postdrug - Baseline (within)')
    _draw_diff(axes[1, 1],
               group_mats.get(('Vehicle', 'Postdrug')),
               group_mats.get(('Vehicle', 'Baseline')),
               'Vehicle: Postdrug - Baseline (within)')

    ax = axes[1, 2]
    ax.axis('off')
    ax.text(
        0.05, 0.7,
        'Reading the diff heatmaps:\n\n'
        '  red  = first cell > second cell\n'
        '  blue = second cell > first cell\n'
        '  rows = "from" state\n'
        '  cols = "to"   state\n\n'
        'Cell value = diff in row-normalised\n'
        'transition probability.',
        fontsize=10, va='top',
    )
    fig.suptitle('Transition probability differences - THC withdrawal contrasts',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    plt.close(fig)


def collect_bout_stats(per_session_states: dict, all_states: list) -> pd.DataFrame:
    """Per session: bout count and mean bout length (frames) per state."""
    rows = []
    for sess, info in per_session_states.items():
        states = info['states']
        bouts = []
        if len(states):
            cur_state = states[0]; cur_start = 0
            for i in range(1, len(states)):
                if states[i] != cur_state:
                    bouts.append((cur_state, cur_start, i - 1))
                    cur_state = states[i]; cur_start = i
            bouts.append((cur_state, cur_start, len(states) - 1))
        bout_lens = {s: [] for s in all_states + [NOISE_LABEL]}
        for s, a, b in bouts:
            if s in bout_lens:
                bout_lens[s].append(b - a + 1)
        row = {'session': sess, 'mouse': info['mouse'],
               'condition': info['condition'], 'group': info['group']}
        for s in all_states + [NOISE_LABEL]:
            row[f'{s}_bouts'] = len(bout_lens[s])
            row[f'{s}_mean_len'] = float(np.mean(bout_lens[s])) if bout_lens[s] else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def plot_bout_structure(bout_df: pd.DataFrame, png: Path) -> None:
    """Two figures stacked: bout COUNTS and mean bout LENGTH per state, by
    (group, condition)."""
    states = PRIORITY  # skip noise for cleanliness
    palette = {'THC': '#d62728', 'Vehicle': '#1f77b4'}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex='col')
    for col_i, cond in enumerate(['Baseline', 'Postdrug']):
        sub = bout_df[bout_df['condition'] == cond]
        x = np.arange(len(states))
        w = 0.35
        # Row 0: counts
        ax_c = axes[0, col_i]
        for i, grp in enumerate(['THC', 'Vehicle']):
            grp_sub = sub[sub['group'] == grp]
            means = [grp_sub[f'{s}_bouts'].mean() if len(grp_sub) else 0
                     for s in states]
            sems = [grp_sub[f'{s}_bouts'].sem() if len(grp_sub) > 1 else 0
                    for s in states]
            ax_c.bar(x + (i - 0.5) * w, means, w, yerr=sems, label=grp,
                     color=palette[grp], alpha=0.7, capsize=3)
        ax_c.set_xticks(x); ax_c.set_xticklabels(states, rotation=30, ha='right')
        ax_c.set_ylabel('bout count')
        ax_c.set_title(f'{cond} - bout count')
        ax_c.legend()
        ax_c.grid(axis='y', alpha=0.3)
        # Row 1: mean length
        ax_l = axes[1, col_i]
        for i, grp in enumerate(['THC', 'Vehicle']):
            grp_sub = sub[sub['group'] == grp]
            # mean length in seconds (60 fps)
            means = [grp_sub[f'{s}_mean_len'].mean() / 60.0 if len(grp_sub) else 0
                     for s in states]
            sems = [grp_sub[f'{s}_mean_len'].sem() / 60.0 if len(grp_sub) > 1 else 0
                    for s in states]
            ax_l.bar(x + (i - 0.5) * w, means, w, yerr=sems, label=grp,
                     color=palette[grp], alpha=0.7, capsize=3)
        ax_l.set_xticks(x); ax_l.set_xticklabels(states, rotation=30, ha='right')
        ax_l.set_ylabel('mean bout length (s)')
        ax_l.set_title(f'{cond} - mean bout length')
        ax_l.legend()
        ax_l.grid(axis='y', alpha=0.3)
    fig.suptitle('Bout structure - THC withdrawal cohort',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    plt.close(fig)


def plot_ethograms(per_session_states: dict, all_states: list, png: Path,
                   fps: int = 60) -> None:
    """Per-session ethogram: rows = sessions (ordered by group/condition),
    columns = time, cell color = state."""
    # Order sessions
    items = sorted(
        per_session_states.items(),
        key=lambda kv: (kv[1]['group'], kv[1]['condition'], kv[1]['mouse'])
    )
    if not items:
        return
    n_sess = len(items)
    # Encode states to ints for imshow
    state_to_int = {s: i for i, s in enumerate(all_states + [NOISE_LABEL])}
    max_len = max(len(info['states']) for _, info in items)
    grid = np.full((n_sess, max_len), state_to_int[NOISE_LABEL], dtype=int)
    labels = []
    for r, (sess, info) in enumerate(items):
        s = info['states']
        ints = np.array([state_to_int.get(x, state_to_int[NOISE_LABEL])
                         for x in s])
        grid[r, : len(ints)] = ints
        labels.append(f'{info["group"][:3]} {info["condition"][:4]}  m{info["mouse"]}')

    n_states = len(all_states) + 1
    cmap = plt.get_cmap('tab10', n_states)

    fig, ax = plt.subplots(figsize=(15, max(4, n_sess * 0.45)))
    im = ax.imshow(grid, aspect='auto', interpolation='nearest', cmap=cmap,
                   vmin=-0.5, vmax=n_states - 0.5)
    ax.set_yticks(range(n_sess))
    ax.set_yticklabels(labels, fontsize=9)
    # X axis in minutes
    n_min = max_len / fps / 60.0
    xt = np.linspace(0, max_len, min(11, int(n_min) + 1))
    ax.set_xticks(xt)
    ax.set_xticklabels([f'{int(t / fps / 60)}' for t in xt])
    ax.set_xlabel('Time (min)')
    ax.set_title('Per-session ethograms (sorted by group, condition, mouse)')
    cbar = plt.colorbar(im, ax=ax, ticks=range(n_states), fraction=0.02, pad=0.01)
    cbar.ax.set_yticklabels(all_states + [NOISE_LABEL], fontsize=9)
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    plt.close(fig)


def post_to_discord(text: str, png_paths: list[Path]) -> None:
    """Multipart upload with files + content message."""
    import subprocess
    cmd = ['curl', '-s', '-o', os.devnull, '-w', '%{http_code}', '-X', 'POST',
           '-F', f'payload_json={{"content":"{text}"}}']
    for i, p in enumerate(png_paths):
        cmd += ['-F', f'file{i}=@{p}']
    cmd.append(WEBHOOK)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f'Discord upload: HTTP {r.stdout}')


def main() -> int:
    csvs = sorted(RESULTS_DIR.glob('*_predictions.csv'))
    if not csvs:
        print(f'No prediction CSVs in {RESULTS_DIR}')
        return 1

    rows = []
    group_mats = {}  # (group, condition) -> sum-of-matrices across mice
    per_session_states = {}  # for ethograms + bout stats
    for csv in csvs:
        base = csv.name.replace('_predictions.csv', '')
        parsed = parse_session(base)
        if not parsed:
            print(f'  skip (cannot parse): {base}')
            continue
        n, cond, grp = parsed
        df = pd.read_csv(csv)
        states = assign_states(df, PRIORITY)
        occ = state_occupancy(states, PRIORITY)
        occ.update({'session': base, 'mouse': n, 'condition': cond, 'group': grp})
        rows.append(occ)

        per_session_states[base] = {
            'states': states, 'mouse': n, 'condition': cond, 'group': grp,
        }

        M = transition_matrix(states, PRIORITY)
        key = (grp, cond)
        if key in group_mats:
            group_mats[key] = group_mats[key].add(M, fill_value=0)
        else:
            group_mats[key] = M

    occ_df = pd.DataFrame(rows)
    occ_csv = ANALYSIS_DIR / 'state_occupancy.csv'
    occ_df.to_csv(occ_csv, index=False)
    print(f'Wrote: {occ_csv}')

    # Save raw transition matrices (per group/condition)
    for (grp, cond), M in group_mats.items():
        out = ANALYSIS_DIR / f'transitions_{grp}_{cond}.csv'
        M.to_csv(out)

    # Plots
    p_immob = ANALYSIS_DIR / 'immobility.png'
    p_states = ANALYSIS_DIR / 'state_occupancy.png'
    p_trans = ANALYSIS_DIR / 'transitions.png'
    p_diffs = ANALYSIS_DIR / 'transitions_diffs.png'
    p_bouts = ANALYSIS_DIR / 'bout_structure.png'
    p_etho = ANALYSIS_DIR / 'ethograms.png'

    plot_immobility(occ_df, p_immob)
    plot_state_occupancy(occ_df, p_states)
    plot_transition_heatmaps(group_mats, p_trans)
    plot_transition_diffs(group_mats, p_diffs)

    bout_df = collect_bout_stats(per_session_states, PRIORITY)
    bout_df.to_csv(ANALYSIS_DIR / 'bout_stats.csv', index=False)
    plot_bout_structure(bout_df, p_bouts)
    plot_ethograms(per_session_states, PRIORITY, p_etho)

    print(f'Plots: {p_immob}, {p_states}, {p_trans}, {p_diffs}, {p_bouts}, {p_etho}')

    # Quick paired contrasts: Postdrug between groups + within-group B->P
    def _mean_sd(arr):
        a = np.asarray(arr, dtype=float)
        if len(a) == 0:
            return 'n/a'
        return f'{a.mean()*100:.1f} +/- {a.std(ddof=1)*100:.1f}'

    pd_thc = occ_df[(occ_df['condition'] == 'Postdrug') & (occ_df['group'] == 'THC')]['still']
    pd_veh = occ_df[(occ_df['condition'] == 'Postdrug') & (occ_df['group'] == 'Vehicle')]['still']
    bl_thc = occ_df[(occ_df['condition'] == 'Baseline') & (occ_df['group'] == 'THC')]['still']
    bl_veh = occ_df[(occ_df['condition'] == 'Baseline') & (occ_df['group'] == 'Vehicle')]['still']

    # Paired delta per mouse
    deltas = {'THC': [], 'Vehicle': []}
    for grp in ['THC', 'Vehicle']:
        sub = occ_df[occ_df['group'] == grp]
        for mouse, mdf in sub.groupby('mouse'):
            bl = mdf[mdf['condition'] == 'Baseline']['still']
            po = mdf[mdf['condition'] == 'Postdrug']['still']
            if len(bl) and len(po):
                deltas[grp].append(po.iloc[0] - bl.iloc[0])

    lines = ['THC withdrawal analysis complete.']
    lines.append('% time immobile (mean +/- SD):')
    lines.append(f'  POSTDRUG  THC: {_mean_sd(pd_thc)}  (n={len(pd_thc)})')
    lines.append(f'  POSTDRUG  Veh: {_mean_sd(pd_veh)}  (n={len(pd_veh)})')
    lines.append(f'  Baseline  THC: {_mean_sd(bl_thc)}')
    lines.append(f'  Baseline  Veh: {_mean_sd(bl_veh)}')
    lines.append(f'Within-subject delta (Postdrug - Baseline, % time):')
    lines.append(f'  THC:     {_mean_sd(deltas["THC"])}')
    lines.append(f'  Vehicle: {_mean_sd(deltas["Vehicle"])}')
    summary = '\\n'.join(lines).replace('"', '')

    # Discord caps a single message at 10 attachments / 8MB per file. Six
    # PNGs is fine; but the ethogram can get big at high resolution, so
    # split into two posts if needed.
    post_to_discord(summary, [p_immob, p_states, p_trans, p_diffs])
    post_to_discord('Additional figures: bout structure + ethograms.',
                    [p_bouts, p_etho])
    return 0


if __name__ == '__main__':
    sys.exit(main())
