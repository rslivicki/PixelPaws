"""
Re-run the THC withdrawal group analysis using only the FIRST 25 MINUTES of
each session (90 000 frames at 60 fps). Useful when sessions vary slightly
in length and you want a fixed-window comparison.

Reuses every analysis function from thc_withdrawal_group_analyze.py — only
the CSV-reading step is changed (df.head(MAX_FRAMES)) and the output dir is
swapped to analysis_25min/ so the full-duration results remain untouched.

Run:
  PYTHONIOENCODING=utf-8 py -X utf8 scripts/research/thc_25min_analyze.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r'E:\PixelPaws')
sys.path.insert(0, str(REPO / 'scripts' / 'research'))

from thc_withdrawal_group_analyze import (  # noqa: E402
    parse_session, assign_states, state_occupancy, transition_matrix,
    plot_immobility, plot_state_occupancy, plot_transition_heatmaps,
    plot_transition_diffs, plot_bout_structure, plot_ethograms,
    collect_bout_stats, post_to_discord,
    PRIORITY, NOISE_LABEL,
)

PROJECT_ROOT = Path(r'E:\RSVIDS\Blackbox\260506_RS_THC_Withdrawal')
RESULTS_DIR = PROJECT_ROOT / 'results'
ANALYSIS_DIR = PROJECT_ROOT / 'analysis_25min'
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

FPS = 60
MAX_MINUTES = 25
MAX_FRAMES = MAX_MINUTES * 60 * FPS  # 90000


def main() -> int:
    csvs = sorted(RESULTS_DIR.glob('*_predictions.csv'))
    if not csvs:
        print(f'No prediction CSVs in {RESULTS_DIR}')
        return 1

    rows = []
    group_mats: dict = {}
    per_session_states: dict = {}
    truncation_log = []
    for csv in csvs:
        base = csv.name.replace('_predictions.csv', '')
        parsed = parse_session(base)
        if not parsed:
            print(f'  skip (cannot parse): {base}')
            continue
        n, cond, grp = parsed
        df = pd.read_csv(csv)
        original_len = len(df)
        df = df.head(MAX_FRAMES)
        truncated_len = len(df)
        truncation_log.append((base, original_len, truncated_len))

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

    for (grp, cond), M in group_mats.items():
        out = ANALYSIS_DIR / f'transitions_{grp}_{cond}.csv'
        M.to_csv(out)

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

    # Truncation summary
    print()
    print(f'Truncation (first {MAX_MINUTES} min = {MAX_FRAMES} frames):')
    n_full = sum(1 for _, o, t in truncation_log if t == o)
    n_cut = sum(1 for _, o, t in truncation_log if t < o)
    print(f'  {n_full} session(s) shorter than {MAX_MINUTES} min (no truncation)')
    print(f'  {n_cut} session(s) truncated')

    # Stats summary
    def _mean_sd(arr):
        a = np.asarray(arr, dtype=float)
        if len(a) == 0:
            return 'n/a'
        return f'{a.mean()*100:.1f} +/- {a.std(ddof=1)*100:.1f}'

    pd_thc = occ_df[(occ_df['condition'] == 'Postdrug') & (occ_df['group'] == 'THC')]['still']
    pd_veh = occ_df[(occ_df['condition'] == 'Postdrug') & (occ_df['group'] == 'Vehicle')]['still']
    bl_thc = occ_df[(occ_df['condition'] == 'Baseline') & (occ_df['group'] == 'THC')]['still']
    bl_veh = occ_df[(occ_df['condition'] == 'Baseline') & (occ_df['group'] == 'Vehicle')]['still']

    deltas = {'THC': [], 'Vehicle': []}
    for grp in ['THC', 'Vehicle']:
        sub = occ_df[occ_df['group'] == grp]
        for mouse, mdf in sub.groupby('mouse'):
            bl = mdf[mdf['condition'] == 'Baseline']['still']
            po = mdf[mdf['condition'] == 'Postdrug']['still']
            if len(bl) and len(po):
                deltas[grp].append(po.iloc[0] - bl.iloc[0])

    lines = [f'THC withdrawal -- 25-minute window analysis ({len(csvs)} sessions).']
    lines.append(f'Frames per session capped at {MAX_FRAMES} ({MAX_MINUTES} min @ {FPS} fps).')
    lines.append(f'  {n_cut}/{len(truncation_log)} sessions actually truncated; {n_full} were already shorter.')
    lines.append('% time immobile (mean +/- SD):')
    lines.append(f'  POSTDRUG  THC: {_mean_sd(pd_thc)}  (n={len(pd_thc)})')
    lines.append(f'  POSTDRUG  Veh: {_mean_sd(pd_veh)}  (n={len(pd_veh)})')
    lines.append(f'  Baseline  THC: {_mean_sd(bl_thc)}')
    lines.append(f'  Baseline  Veh: {_mean_sd(bl_veh)}')
    lines.append(f'Within-subject delta (Postdrug - Baseline, % time):')
    lines.append(f'  THC:     {_mean_sd(deltas["THC"])}')
    lines.append(f'  Vehicle: {_mean_sd(deltas["Vehicle"])}')
    summary = '\\n'.join(lines).replace('"', '')

    post_to_discord(summary, [p_immob, p_states, p_trans, p_diffs])
    post_to_discord('Additional figures (25-min): bout structure + ethograms.',
                    [p_bouts, p_etho])
    return 0


if __name__ == '__main__':
    sys.exit(main())
