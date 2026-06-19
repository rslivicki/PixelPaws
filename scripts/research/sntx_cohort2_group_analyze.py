"""
Group analysis for SNTX cohort 2: CCIX vs SilkX vs Sham at BL and 1h.

Reads:
  - <project>/results/<session>_predictions.csv (per-frame 5-behavior preds)
  - sntx_cohort2_blinding_key.json (mouse_id -> group)

Outputs:
  - <project>/analysis/state_occupancy.csv
  - <project>/analysis/transitions_<group>_<condition>.csv (6 files)
  - <project>/analysis/{immobility,state_occupancy,transitions,transitions_diffs,bout_structure,ethograms}.png
  - Discord push.

Run:
  PYTHONIOENCODING=utf-8 py -X utf8 scripts/research/sntx_cohort2_group_analyze.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(r'E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\videos\2605_Cohort2')

KEY_PATH = Path(r'E:\PixelPaws\scripts\research\sntx_cohort2_blinding_key.json')
WEBHOOK = (
    ""
)

# Same priority + noise label as THC
PRIORITY = ['Facial_grooming', 'Left_licking', 'rearing', 'still', 'walking']
NOISE_LABEL = 'noise'
EXCLUDE_NOISE = True

# Fixed analysis windows so different-length recordings compare cleanly.
# BL recordings are 30-min sessions; 1h recordings are 60-min sessions.
FPS = 60
WINDOW_MINUTES = {'BL': 30, '1h': 60}

GROUP_ORDER = ['Sham', 'CCIX', 'SilkX']
CONDITION_ORDER = ['BL', '1h']
PALETTE = {'Sham': '#7f7f7f', 'CCIX': '#d62728', 'SilkX': '#2ca02c'}


def max_frames_for(condition: str) -> int:
    return WINDOW_MINUTES[condition] * 60 * FPS

# Filename pattern: 050126_JG_SNTX_<id>_<BL|1h>_cropped or 05026_JG_SNTX_<id>_<BL|1h>_cropped
SESSION_RE = re.compile(r'_SNTX_(?P<mouse>\d+)_(?P<cond>BL|1h)(?:_cropped)?$')


def load_key():
    with open(KEY_PATH, 'r') as f:
        spec = json.load(f)
    return {mid: meta['group'] for mid, meta in spec['mice'].items()}


def parse_session(name: str, key: dict):
    m = SESSION_RE.search(name)
    if not m:
        return None
    mouse = m.group('mouse')
    cond = m.group('cond')
    group = key.get(mouse)
    if group is None:
        return None
    return mouse, cond, group


def assign_states(df: pd.DataFrame, priority: list) -> np.ndarray:
    n = len(df)
    states = np.full(n, NOISE_LABEL, dtype=object)
    for behavior in reversed(priority):
        if behavior in df.columns:
            mask = df[behavior].astype(int).to_numpy() == 1
            states[mask] = behavior
    return states


def state_occupancy(states: np.ndarray, all_states: list) -> dict:
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


def transition_matrix(states: np.ndarray, all_states: list) -> pd.DataFrame:
    """Build a bout-mode transition count matrix.

    With EXCLUDE_NOISE, we drop noise bouts from the sequence (treat them
    as "didn't happen") and then RE-COLLAPSE: two bouts of the same state
    separated by a noise gap would otherwise appear as a self-transition
    on the diagonal, which is misleading — those are just continuous bouts
    interrupted by classifier uncertainty, not real state changes.
    """
    if len(states) == 0:
        idx = all_states + ([] if EXCLUDE_NOISE else [NOISE_LABEL])
        return pd.DataFrame(0, index=idx, columns=idx)
    bouts = [states[0]]
    for s in states[1:]:
        if s != bouts[-1]:
            bouts.append(s)
    if EXCLUDE_NOISE:
        bouts = [b for b in bouts if b != NOISE_LABEL]
        # Re-collapse adjacent same-state bouts so the diagonal reflects
        # ONLY true within-state revisits across at least one different
        # state, not noise-gap fragmentation. After this pass the diagonal
        # of the resulting matrix is zero.
        collapsed = []
        for b in bouts:
            if not collapsed or b != collapsed[-1]:
                collapsed.append(b)
        bouts = collapsed
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
    """3-group, 2-condition box plots + within-subject paired."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    def _box(ax, sub_df, title):
        groups = [g for g in GROUP_ORDER if g in sub_df['group'].values]
        data = [(sub_df[sub_df['group'] == g]['still'].to_numpy() * 100).tolist()
                for g in groups]
        bp = ax.boxplot(data, tick_labels=groups, patch_artist=True, widths=0.55,
                        medianprops=dict(color='black'))
        for patch, g in zip(bp['boxes'], groups):
            patch.set_facecolor(PALETTE.get(g, '#888'))
            patch.set_alpha(0.6)
        for i, (g, ys) in enumerate(zip(groups, data), 1):
            if ys:
                xs = np.random.normal(i, 0.07, len(ys))
                ax.scatter(xs, ys, color='black', s=22, zorder=3)
                ax.text(i, max(ys) + 1, f'n={len(ys)}', ha='center', fontsize=9, color='gray')
        ax.set_ylabel('% time immobile (still)')
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)

    _box(axes[0], occ_df[occ_df['condition'] == '1h'], '1h post: 3-group')
    _box(axes[1], occ_df[occ_df['condition'] == 'BL'], 'Baseline: 3-group')

    ax = axes[2]
    for grp in GROUP_ORDER:
        sub = occ_df[occ_df['group'] == grp]
        for mouse, mdf in sub.groupby('mouse'):
            bl = mdf[mdf['condition'] == 'BL']['still']
            po = mdf[mdf['condition'] == '1h']['still']
            if len(bl) and len(po):
                ax.plot([0, 1], [bl.iloc[0] * 100, po.iloc[0] * 100],
                        'o-', color=PALETTE[grp], alpha=0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['BL', '1h'])
    ax.set_ylabel('% time immobile (still)')
    ax.set_title('Within-subject: BL -> 1h')
    ax.grid(axis='y', alpha=0.3)
    handles = [plt.Line2D([0], [0], color=PALETTE[g], marker='o',
                          linestyle='-', label=g) for g in GROUP_ORDER]
    ax.legend(handles=handles, loc='best')

    fig.suptitle('Time spent immobile -- SNTX cohort 2 (CCIX vs SilkX vs Sham)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    plt.close(fig)


def plot_state_occupancy(occ_df: pd.DataFrame, png: Path) -> None:
    states_to_plot = PRIORITY if EXCLUDE_NOISE else PRIORITY + [NOISE_LABEL]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, cond in zip(axes, CONDITION_ORDER):
        sub = occ_df[occ_df['condition'] == cond]
        x = np.arange(len(states_to_plot))
        groups = [g for g in GROUP_ORDER if g in sub['group'].values]
        w = 0.85 / max(len(groups), 1)
        for i, grp in enumerate(groups):
            grp_sub = sub[sub['group'] == grp]
            means = [grp_sub[s].mean() * 100 if s in grp_sub.columns else 0.0
                     for s in states_to_plot]
            sems = [grp_sub[s].sem() * 100 if s in grp_sub.columns else 0.0
                    for s in states_to_plot]
            ax.bar(x + (i - (len(groups) - 1) / 2) * w, means, w, yerr=sems,
                   label=grp, color=PALETTE[grp], alpha=0.75, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(states_to_plot, rotation=30, ha='right')
        ax.set_ylabel('% time')
        ax.set_title(cond)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Per-state occupancy -- SNTX cohort 2')
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    plt.close(fig)


def plot_transition_heatmaps(group_mats: dict, png: Path) -> None:
    """3 (groups) x 2 (conditions) grid of row-normalised transition probs."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for r, cond in enumerate(CONDITION_ORDER):
        for c, grp in enumerate(GROUP_ORDER):
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
            ax.set_xlabel('to'); ax.set_ylabel('from')
            for i in range(P.shape[0]):
                for j in range(P.shape[1]):
                    v = P.iloc[i, j]
                    if v > 0.01:
                        ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                                color='white' if v < 0.5 else 'black', fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle('Transition probabilities -- SNTX cohort 2 (CCIX vs SilkX vs Sham, BL vs 1h)')
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    plt.close(fig)


def _draw_diff(ax, A, B, title, vlim=0.4):
    if A is None or B is None or A.sum().sum() == 0 or B.sum().sum() == 0:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center')
        ax.set_title(title, fontsize=10)
        return
    PA = normalise_rows(A); PB = normalise_rows(B)
    idx = PA.index.union(PB.index); cols = PA.columns.union(PB.columns)
    PA = PA.reindex(index=idx, columns=cols, fill_value=0)
    PB = PB.reindex(index=idx, columns=cols, fill_value=0)
    D = (PA - PB).to_numpy()
    im = ax.imshow(D, cmap='RdBu_r', vmin=-vlim, vmax=vlim)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha='right')
    ax.set_yticks(range(len(idx))); ax.set_yticklabels(idx)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('to'); ax.set_ylabel('from')
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            v = D[i, j]
            if abs(v) > 0.03:
                ax.text(j, i, f'{v:+.2f}', ha='center', va='center',
                        color='black', fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_transition_diffs(group_mats: dict, png: Path) -> None:
    """Key contrasts: pairwise group at 1h, pairwise group at BL,
    within-group 1h-BL."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    # Row 0: between-group at 1h (pairwise)
    _draw_diff(axes[0, 0],
               group_mats.get(('CCIX', '1h')), group_mats.get(('Sham', '1h')),
               '1h: CCIX - Sham (primary between)')
    _draw_diff(axes[0, 1],
               group_mats.get(('SilkX', '1h')), group_mats.get(('Sham', '1h')),
               '1h: SilkX - Sham')
    _draw_diff(axes[0, 2],
               group_mats.get(('CCIX', '1h')), group_mats.get(('SilkX', '1h')),
               '1h: CCIX - SilkX')
    # Row 1: within-group BL→1h
    _draw_diff(axes[1, 0],
               group_mats.get(('CCIX', '1h')), group_mats.get(('CCIX', 'BL')),
               'CCIX: 1h - BL (within)')
    _draw_diff(axes[1, 1],
               group_mats.get(('Sham', '1h')), group_mats.get(('Sham', 'BL')),
               'Sham: 1h - BL (within)')
    _draw_diff(axes[1, 2],
               group_mats.get(('SilkX', '1h')), group_mats.get(('SilkX', 'BL')),
               'SilkX: 1h - BL (within)')
    fig.suptitle('Transition probability differences -- SNTX cohort 2',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    plt.close(fig)


def collect_bout_stats(per_session_states: dict, all_states: list) -> pd.DataFrame:
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
    states = PRIORITY
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex='col')
    for col_i, cond in enumerate(CONDITION_ORDER):
        sub = bout_df[bout_df['condition'] == cond]
        x = np.arange(len(states))
        groups = [g for g in GROUP_ORDER if g in sub['group'].values]
        w = 0.85 / max(len(groups), 1)
        ax_c = axes[0, col_i]
        for i, grp in enumerate(groups):
            grp_sub = sub[sub['group'] == grp]
            means = [grp_sub[f'{s}_bouts'].mean() if len(grp_sub) else 0 for s in states]
            sems = [grp_sub[f'{s}_bouts'].sem() if len(grp_sub) > 1 else 0 for s in states]
            ax_c.bar(x + (i - (len(groups) - 1) / 2) * w, means, w, yerr=sems,
                     label=grp, color=PALETTE[grp], alpha=0.75, capsize=3)
        ax_c.set_xticks(x); ax_c.set_xticklabels(states, rotation=30, ha='right')
        ax_c.set_ylabel('bout count'); ax_c.set_title(f'{cond} - bout count')
        ax_c.legend(); ax_c.grid(axis='y', alpha=0.3)
        ax_l = axes[1, col_i]
        for i, grp in enumerate(groups):
            grp_sub = sub[sub['group'] == grp]
            means = [grp_sub[f'{s}_mean_len'].mean() / 60.0 if len(grp_sub) else 0
                     for s in states]
            sems = [grp_sub[f'{s}_mean_len'].sem() / 60.0 if len(grp_sub) > 1 else 0
                    for s in states]
            ax_l.bar(x + (i - (len(groups) - 1) / 2) * w, means, w, yerr=sems,
                     label=grp, color=PALETTE[grp], alpha=0.75, capsize=3)
        ax_l.set_xticks(x); ax_l.set_xticklabels(states, rotation=30, ha='right')
        ax_l.set_ylabel('mean bout length (s)')
        ax_l.set_title(f'{cond} - mean bout length')
        ax_l.legend(); ax_l.grid(axis='y', alpha=0.3)
    fig.suptitle('Bout structure -- SNTX cohort 2', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    plt.close(fig)


def plot_ethograms(per_session_states: dict, all_states: list, png: Path,
                   fps: int = 60) -> None:
    items = sorted(
        per_session_states.items(),
        key=lambda kv: (kv[1]['group'], kv[1]['condition'], kv[1]['mouse'])
    )
    if not items:
        return
    n_sess = len(items)
    state_to_int = {s: i for i, s in enumerate(all_states + [NOISE_LABEL])}
    max_len = max(len(info['states']) for _, info in items)
    grid = np.full((n_sess, max_len), state_to_int[NOISE_LABEL], dtype=int)
    labels = []
    for r, (sess, info) in enumerate(items):
        s = info['states']
        ints = np.array([state_to_int.get(x, state_to_int[NOISE_LABEL]) for x in s])
        grid[r, : len(ints)] = ints
        labels.append(f'{info["group"][:4]:<4} {info["condition"]:<2}  m{info["mouse"]}')
    n_states = len(all_states) + 1
    cmap = plt.get_cmap('tab10', n_states)
    fig, ax = plt.subplots(figsize=(15, max(4, n_sess * 0.45)))
    im = ax.imshow(grid, aspect='auto', interpolation='nearest', cmap=cmap,
                   vmin=-0.5, vmax=n_states - 0.5)
    ax.set_yticks(range(n_sess))
    ax.set_yticklabels(labels, fontsize=9)
    n_min = max_len / fps / 60.0
    xt = np.linspace(0, max_len, min(11, int(n_min) + 1))
    ax.set_xticks(xt)
    ax.set_xticklabels([f'{int(t / fps / 60)}' for t in xt])
    ax.set_xlabel('Time (min)')
    ax.set_title('Per-session ethograms (SNTX cohort 2, sorted by group/condition/mouse)')
    cbar = plt.colorbar(im, ax=ax, ticks=range(n_states), fraction=0.02, pad=0.01)
    cbar.ax.set_yticklabels(all_states + [NOISE_LABEL], fontsize=9)
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    plt.close(fig)


def post_to_discord(text: str, png_paths: list) -> None:
    import subprocess
    cmd = ['curl', '-s', '-o', os.devnull, '-w', '%{http_code}', '-X', 'POST',
           '-F', f'payload_json={{"content":"{text}"}}']
    for i, p in enumerate(png_paths):
        cmd += ['-F', f'file{i}=@{p}']
    cmd.append(WEBHOOK)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f'Discord upload: HTTP {r.stdout}')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', default='',
                        help='Suffix on results/ and analysis/ folders. '
                             'Empty (default) = raw h5 mode. "_filtered" = '
                             'filtered h5 mode.')
    args = parser.parse_args()
    suffix = args.suffix

    results_dir = PROJECT_ROOT / f'results{suffix}'
    analysis_dir = PROJECT_ROOT / f'analysis{suffix}'
    analysis_dir.mkdir(parents=True, exist_ok=True)

    key = load_key()
    print(f'Loaded blinding key: {len(key)} mice')
    print(f'Reading from {results_dir}, writing to {analysis_dir}')

    csvs = sorted(results_dir.glob('*_predictions.csv'))
    if not csvs:
        print(f'No prediction CSVs in {results_dir}')
        return 1

    rows = []
    group_mats = {}
    per_session_states = {}
    skipped = []
    truncation_log = []
    for csv in csvs:
        base = csv.name.replace('_predictions.csv', '')
        parsed = parse_session(base, key)
        if parsed is None:
            skipped.append(base)
            print(f'  skip (not in key or unparseable): {base}')
            continue
        mouse, cond, grp = parsed
        df = pd.read_csv(csv)
        # Cap to fixed analysis window per condition (BL=30 min, 1h=60 min)
        max_f = max_frames_for(cond)
        original_len = len(df)
        df = df.head(max_f)
        truncated_len = len(df)
        truncation_log.append((base, cond, original_len, truncated_len, max_f))
        states = assign_states(df, PRIORITY)
        occ = state_occupancy(states, PRIORITY)
        occ.update({'session': base, 'mouse': mouse, 'condition': cond, 'group': grp})
        rows.append(occ)
        per_session_states[base] = {
            'states': states, 'mouse': mouse, 'condition': cond, 'group': grp,
        }
        M = transition_matrix(states, PRIORITY)
        key_t = (grp, cond)
        if key_t in group_mats:
            group_mats[key_t] = group_mats[key_t].add(M, fill_value=0)
        else:
            group_mats[key_t] = M

    # Truncation report
    n_short = sum(1 for _, _, o, t, _ in truncation_log if t == o)
    n_capped = sum(1 for _, _, o, t, _ in truncation_log if t < o)
    print(f'\nWindow caps applied: BL={WINDOW_MINUTES["BL"]}min, '
          f'1h={WINDOW_MINUTES["1h"]}min @ {FPS} fps.')
    print(f'  {n_capped}/{len(truncation_log)} session(s) capped; '
          f'{n_short} were already shorter than the cap.')

    occ_df = pd.DataFrame(rows)
    occ_csv = analysis_dir / 'state_occupancy.csv'
    occ_df.to_csv(occ_csv, index=False)
    print(f'Wrote: {occ_csv}')

    for (grp, cond), M in group_mats.items():
        out = analysis_dir / f'transitions_{grp}_{cond}.csv'
        M.to_csv(out)

    p_immob = analysis_dir / 'immobility.png'
    p_states = analysis_dir / 'state_occupancy.png'
    p_trans = analysis_dir / 'transitions.png'
    p_diffs = analysis_dir / 'transitions_diffs.png'
    p_bouts = analysis_dir / 'bout_structure.png'
    p_etho = analysis_dir / 'ethograms.png'

    plot_immobility(occ_df, p_immob)
    plot_state_occupancy(occ_df, p_states)
    plot_transition_heatmaps(group_mats, p_trans)
    plot_transition_diffs(group_mats, p_diffs)

    bout_df = collect_bout_stats(per_session_states, PRIORITY)
    bout_df.to_csv(analysis_dir / 'bout_stats.csv', index=False)
    plot_bout_structure(bout_df, p_bouts)
    plot_ethograms(per_session_states, PRIORITY, p_etho)

    print(f'Plots: {p_immob}, {p_states}, {p_trans}, {p_diffs}, {p_bouts}, {p_etho}')

    # Stats summary text
    def _ms(arr):
        a = np.asarray(arr, dtype=float)
        if len(a) == 0:
            return 'n/a'
        return f'{a.mean()*100:.1f} +/- {a.std(ddof=1)*100:.1f}'

    h5_label = 'filtered h5' if suffix == '_filtered' else 'raw h5 (unfiltered)'
    lines = [f'SNTX cohort 2 -- group analysis ({h5_label}). CCIX vs SilkX vs Sham.']
    lines.append(f'Windows capped: BL={WINDOW_MINUTES["BL"]} min, '
                 f'1h={WINDOW_MINUTES["1h"]} min. Self-transitions collapsed '
                 f'(diagonal of transition matrix is 0 by design — '
                 f'high diagonal previously came from noise-gap fragmentation, '
                 f'not real state revisits).')
    if skipped:
        lines.append(f'  Skipped (not in blinding key): {len(skipped)} session(s): {", ".join(skipped)}')
    lines.append('')
    lines.append('% time immobile (mean +/- SD):')
    for grp in GROUP_ORDER:
        for cond in CONDITION_ORDER:
            sub = occ_df[(occ_df['group'] == grp) & (occ_df['condition'] == cond)]['still']
            lines.append(f'  {grp:>5} {cond:>2}: {_ms(sub)}  (n={len(sub)})')
    summary = '\\n'.join(lines).replace('"', '')

    post_to_discord(summary, [p_immob, p_states, p_trans, p_diffs])
    post_to_discord(f'SNTX cohort 2 ({h5_label}): bout structure + ethograms.',
                    [p_bouts, p_etho])
    return 0


if __name__ == '__main__':
    sys.exit(main())
