"""
Statistical comparison of immobility (% time `still`) across THC vs Vehicle,
Baseline vs Postdrug, on the 25-minute window. Non-parametric tests (small N,
no normality assumption).

Tests:
  1. Postdrug between-group (THC vs Vehicle): Mann-Whitney U  -- PRIMARY
  2. Baseline between-group (control): Mann-Whitney U
  3. Within-subject (Baseline -> Postdrug) per group: Wilcoxon signed-rank
  4. Drug x Time interaction: Mann-Whitney U on per-mouse deltas
     (THC delta vs Vehicle delta)

Effect sizes: rank-biserial r for Mann-Whitney, matched-pairs r for Wilcoxon.

Outputs:
  - <project>/analysis_25min/immobility_stats.txt   (full report)
  - <project>/analysis_25min/immobility_stats.png   (annotated boxplot)
  - Discord push with both attached.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

PROJECT_ROOT = Path(r'E:\RSVIDS\Blackbox\260506_RS_THC_Withdrawal')
ANALYSIS_DIR = PROJECT_ROOT / 'analysis_25min'
OCC_CSV = ANALYSIS_DIR / 'state_occupancy.csv'
OUT_TXT = ANALYSIS_DIR / 'immobility_stats.txt'
OUT_PNG = ANALYSIS_DIR / 'immobility_stats.png'

WEBHOOK = (
    ""
)


def desc(arr):
    a = np.asarray(arr, dtype=float)
    if len(a) == 0:
        return 'n/a'
    return (f'n={len(a)}, mean={a.mean()*100:.1f}%, sd={a.std(ddof=1)*100:.1f}%, '
            f'median={np.median(a)*100:.1f}%, '
            f'IQR=[{np.percentile(a, 25)*100:.1f}, {np.percentile(a, 75)*100:.1f}]')


def mwu(a, b, label_a='A', label_b='B'):
    """Mann-Whitney U with rank-biserial effect size."""
    a = np.asarray(a); b = np.asarray(b)
    if len(a) < 2 or len(b) < 2:
        return f'  {label_a} vs {label_b}: insufficient data (n_a={len(a)}, n_b={len(b)})'
    u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    n1, n2 = len(a), len(b)
    # Rank-biserial effect size: r = 1 - 2U / (n1 * n2)
    # (positive r means a > b)
    rb = 1 - 2 * u / (n1 * n2)
    return (f'  {label_a} (n={n1}, median={np.median(a)*100:.1f}%) vs '
            f'{label_b} (n={n2}, median={np.median(b)*100:.1f}%): '
            f'U={u:.0f}, p={p:.4f}, rank-biserial r={rb:+.3f}')


def wsr(a, b, label_pair):
    """Wilcoxon signed-rank (paired). a, b must be same length."""
    a = np.asarray(a); b = np.asarray(b)
    if len(a) != len(b) or len(a) < 3:
        return f'  {label_pair}: insufficient pairs (n={len(a)})'
    diff = a - b
    if np.all(diff == 0):
        return f'  {label_pair}: all zero differences, no test'
    try:
        w, p = stats.wilcoxon(a, b, alternative='two-sided', zero_method='wilcox')
    except ValueError as e:
        return f'  {label_pair}: wilcoxon failed ({e})'
    n = len(a)
    # Matched-pairs r approx: z / sqrt(n) — scipy returns W not z, so we use a
    # rough r = W / (n*(n+1)/2) - 1, which is r-corrected for direction
    max_w = n * (n + 1) / 2
    r = 2 * w / max_w - 1  # matches sign convention (b > a → positive)
    return (f'  {label_pair} (n={n}): W={w:.0f}, p={p:.4f}, '
            f'median delta={np.median(diff)*100:+.1f}%, r={r:+.3f}')


def post_to_discord(text: str, paths: list) -> None:
    cmd = ['curl', '-s', '-o', os.devnull, '-w', '%{http_code}', '-X', 'POST',
           '-F', f'payload_json={{"content":"{text}"}}']
    for i, p in enumerate(paths):
        cmd += ['-F', f'file{i}=@{p}']
    cmd.append(WEBHOOK)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f'Discord upload: HTTP {r.stdout}')


def main() -> int:
    if not OCC_CSV.is_file():
        print(f'Missing {OCC_CSV} -- run thc_25min_analyze.py first')
        return 1

    df = pd.read_csv(OCC_CSV)
    print(f'Loaded {len(df)} rows from {OCC_CSV.name}')

    # Slice into the 4 groups
    pd_thc = df[(df['condition'] == 'Postdrug') & (df['group'] == 'THC')]['still'].to_numpy()
    pd_veh = df[(df['condition'] == 'Postdrug') & (df['group'] == 'Vehicle')]['still'].to_numpy()
    bl_thc = df[(df['condition'] == 'Baseline') & (df['group'] == 'THC')]['still'].to_numpy()
    bl_veh = df[(df['condition'] == 'Baseline') & (df['group'] == 'Vehicle')]['still'].to_numpy()

    # Per-mouse paired arrays (in matching order)
    paired = {'THC': [], 'Vehicle': []}
    delta = {'THC': [], 'Vehicle': []}
    for grp in ['THC', 'Vehicle']:
        sub = df[df['group'] == grp]
        for mouse, mdf in sub.groupby('mouse'):
            bl_row = mdf[mdf['condition'] == 'Baseline']
            po_row = mdf[mdf['condition'] == 'Postdrug']
            if len(bl_row) and len(po_row):
                bl_v = float(bl_row.iloc[0]['still'])
                po_v = float(po_row.iloc[0]['still'])
                paired[grp].append((bl_v, po_v))
                delta[grp].append(po_v - bl_v)
    thc_bl = np.array([p[0] for p in paired['THC']])
    thc_po = np.array([p[1] for p in paired['THC']])
    veh_bl = np.array([p[0] for p in paired['Vehicle']])
    veh_po = np.array([p[1] for p in paired['Vehicle']])

    lines = []
    lines.append('THC Withdrawal -- Immobility statistics (% time still, 25-minute window)')
    lines.append('=' * 76)
    lines.append('')
    lines.append('Descriptive:')
    lines.append(f'  Postdrug THC:    {desc(pd_thc)}')
    lines.append(f'  Postdrug Veh:    {desc(pd_veh)}')
    lines.append(f'  Baseline THC:    {desc(bl_thc)}')
    lines.append(f'  Baseline Veh:    {desc(bl_veh)}')
    lines.append('')
    lines.append('PRIMARY: Postdrug, THC vs Vehicle (Mann-Whitney U, two-sided):')
    lines.append(mwu(pd_thc, pd_veh, 'THC', 'Vehicle'))
    lines.append('')
    lines.append('Baseline, THC vs Vehicle (Mann-Whitney U, two-sided):')
    lines.append(mwu(bl_thc, bl_veh, 'THC', 'Vehicle'))
    lines.append('')
    lines.append('Within-subject Baseline -> Postdrug (Wilcoxon signed-rank, two-sided):')
    lines.append(wsr(thc_po, thc_bl, 'THC: Postdrug vs Baseline'))
    lines.append(wsr(veh_po, veh_bl, 'Vehicle: Postdrug vs Baseline'))
    lines.append('')
    lines.append('Drug x Time interaction:')
    lines.append('  Compares per-mouse delta (Postdrug minus Baseline).')
    lines.append(mwu(delta['THC'], delta['Vehicle'],
                     'THC delta', 'Vehicle delta'))
    lines.append('')
    lines.append('Notes:')
    lines.append('  - Non-parametric tests appropriate for small N (n_THC=8, n_Veh=7).')
    lines.append('  - rank-biserial r:  +/-0.1 small, +/-0.3 medium, +/-0.5 large.')
    lines.append('  - Two-sided p; halve for one-sided if directional hypothesis.')

    report = '\n'.join(lines)
    OUT_TXT.write_text(report, encoding='utf-8')
    print(report)
    print(f'\nWrote: {OUT_TXT}')

    # Plot: 4 boxplots side-by-side with individual points + significance
    fig, ax = plt.subplots(figsize=(9, 6))
    groups = ['Baseline\nVehicle', 'Baseline\nTHC', 'Postdrug\nVehicle', 'Postdrug\nTHC']
    data = [bl_veh * 100, bl_thc * 100, pd_veh * 100, pd_thc * 100]
    palette_arr = ['#1f77b4', '#d62728', '#1f77b4', '#d62728']
    bp = ax.boxplot(data, tick_labels=groups, patch_artist=True, widths=0.55,
                    medianprops=dict(color='black'))
    for patch, c in zip(bp['boxes'], palette_arr):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    for i, ys in enumerate(data, 1):
        if len(ys):
            xs = np.random.normal(i, 0.06, len(ys))
            ax.scatter(xs, ys, color='black', s=24, zorder=3)

    ax.set_ylabel('% time immobile (still)')
    ax.set_title('Immobility -- THC withdrawal (25-min window)\nMann-Whitney U + Wilcoxon signed-rank',
                 fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Annotate primary contrast (Postdrug THC vs Veh)
    u_pd, p_pd = stats.mannwhitneyu(pd_thc, pd_veh, alternative='two-sided')
    y_max = max(max(d) for d in data) if all(len(d) for d in data) else 100
    y_bar = y_max + 6
    ax.plot([3, 4], [y_bar, y_bar], 'k-', lw=1.2)
    sig = ('***' if p_pd < 0.001 else '**' if p_pd < 0.01 else
           '*' if p_pd < 0.05 else 'ns')
    ax.text(3.5, y_bar + 1, f'{sig}  p={p_pd:.4f}',
            ha='center', fontsize=10)
    ax.set_ylim(top=y_bar + 8)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f'Wrote: {OUT_PNG}')

    # Discord post: file the txt + png. Discord webhooks accept .txt files.
    sig_label = 'sig.' if p_pd < 0.05 else 'ns'
    msg = ('THC immobility statistics (25-min window). PRIMARY contrast: '
           f'Postdrug THC vs Vehicle Mann-Whitney p={p_pd:.4f} ({sig_label}).')
    post_to_discord(msg, [OUT_PNG, OUT_TXT])
    return 0


if __name__ == '__main__':
    sys.exit(main())
