"""
silhouette_normalized_compare.py — same as silhouette_cross_project_compare.py
but with per-session normalization to strip out rig-level scale differences.
Three normalizations are tried and compared:

  raw                  no normalization (direct silhouette_frac)
  per-session zscore   (sf - sf.median()) / sf.iqr()  — robust z-score
  fraction-of-max      sf / np.percentile(sf, 99)     — fraction of typical max

The idea is that fraction-of-max anchors each session to its own ceiling
("lying fully flat" = 1.0 by definition), which should be similar across rigs
because the same posture maxes out the silhouette regardless of framing.
"""

from __future__ import annotations
import glob, os, sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats as _scipy_stats
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


GROUPS = [
    {"name": "2604_DV_DSS_BL", "dir": r"E:\RSVIDS\Blackbox\2604_DV_DSS\features",
     "filter": "_BL_", "exclude": ("_STIM",)},
    {"name": "SNLT_Baseline", "dir": r"E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\features",
     "filter": "_Baseline_", "exclude": ()},
]
ART_FRAC_THRESH = 0.5
TAIL_TRIM_FRAC  = 0.05


def discover_pkls(g):
    out = {}
    for p in sorted(glob.glob(os.path.join(g["dir"], "*.pkl"))):
        f = os.path.basename(p)
        if g["filter"] not in f: continue
        if any(x in f for x in g["exclude"]): continue
        stem = f.split("_features_")[0]
        if stem not in out or os.path.getmtime(p) > os.path.getmtime(out[stem]):
            out[stem] = p
    return [out[k] for k in sorted(out)]


def stats_for_pkl(p):
    X = pd.read_pickle(p)
    if 'silhouette_frac' not in X.columns: return None
    sf = X['silhouette_frac'].astype(float)
    n = len(X); cutoff = int(n * (1.0 - TAIL_TRIM_FRAC))
    mask = (sf <= ART_FRAC_THRESH) & (sf.index < cutoff)
    sf_k = sf[mask]
    if len(sf_k) < 1000: return None

    # Three views:
    raw_med   = float(sf_k.median())
    raw_p95   = float(np.nanpercentile(sf_k.values, 95))
    raw_p99   = float(np.nanpercentile(sf_k.values, 99))

    # Robust z-score: (x - median) / IQR — IQR = p75 - p25
    q25, q75 = np.nanpercentile(sf_k.values, [25, 75])
    iqr = q75 - q25 if (q75 - q25) > 1e-9 else 1e-9
    sf_z = (sf_k - raw_med) / iqr
    z_p95 = float(np.nanpercentile(sf_z.values, 95))   # how many IQRs above median is the top 5%
    # (z_median is 0 by construction; not informative)

    # Fraction-of-max: sf / sf.p99   — anchors max ~1.0 per session
    sf_fm = sf_k / raw_p99
    fm_med = float(sf_fm.median())             # typical posture as fraction of max
    fm_p95 = float(np.nanpercentile(sf_fm.values, 95))

    return {
        'session': os.path.basename(p).split('_features_')[0],
        'pkl': p,
        'n_kept': int(mask.sum()),
        'raw_median':       raw_med,
        'raw_p95':          raw_p95,
        'raw_p99':          raw_p99,
        'iqr':              float(iqr),
        'zscore_p95':       z_p95,           # robust upper-tail width
        'fracmax_median':   fm_med,
        'fracmax_p95':      fm_p95,
    }


def main():
    rows = []
    for g in GROUPS:
        for p in discover_pkls(g):
            s = stats_for_pkl(p)
            if s is None: continue
            s['group'] = g['name']
            rows.append(s)
    if not rows:
        sys.exit("No usable pkls found")
    df = pd.DataFrame(rows)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "analysis_output",
                            f"silhouette_normalized_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "per_session_stats.csv"), index=False)

    metrics = [
        ('raw_median',       'RAW median silhouette_frac\n(rig-dependent)'),
        ('raw_p95',          'RAW p95 silhouette_frac'),
        ('zscore_p95',       'Robust z-score, p95\n(median-anchored, IQR-scaled)'),
        ('fracmax_median',   'Fraction-of-max median\n(sf / sf.p99 per session)'),
    ]
    group_names = [g['name'] for g in GROUPS]
    palette = ['#2266aa', '#cc6622']

    fig, axes = plt.subplots(1, len(metrics),
                              figsize=(2.6*len(metrics), 4.2),
                              constrained_layout=True)
    rng = np.random.default_rng(42)
    for ax, (col, title) in zip(axes, metrics):
        positions = []
        for gi, gname in enumerate(group_names):
            sub = df[df['group']==gname][col].dropna().values
            positions.append(sub)
            xs = gi + rng.uniform(-0.10, 0.10, len(sub))
            ax.scatter(xs, sub, color=palette[gi], s=44, alpha=0.85,
                        edgecolor='black', linewidth=0.4, zorder=3)
        bp = ax.boxplot(positions, positions=range(len(group_names)),
                          widths=0.55, showfliers=False, patch_artist=True, zorder=1)
        for box, c in zip(bp['boxes'], palette):
            box.set(facecolor=c+'22', edgecolor=c, linewidth=1.2)
        for med in bp['medians']:
            med.set(color='black', linewidth=1.4)
        ax.set_xticks(range(len(group_names)))
        ax.set_xticklabels([f'{g}\n(n={(df["group"]==g).sum()})' for g in group_names], fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Cross-project silhouette: raw vs normalized', fontsize=12)
    fig.savefig(os.path.join(out_dir, 'distributions_normalized.png'), dpi=120)
    plt.close(fig)

    # Stats table with effect sizes for each normalization
    lines = ["Cross-project silhouette comparison — raw vs normalized",
             f"  Out: {out_dir}", ""]
    for col, label in metrics:
        lines.append(f"  {label.splitlines()[0]} ({col}):")
        for gname in group_names:
            v = df[df['group']==gname][col].dropna().values
            if len(v):
                lines.append(f"    {gname:<22} mean={v.mean():.4f}  med={np.median(v):.4f}  "
                             f"min={v.min():.4f}  max={v.max():.4f}  n={len(v)}")
        if _SCIPY_OK:
            a = df[df['group']==group_names[0]][col].dropna().values
            b = df[df['group']==group_names[1]][col].dropna().values
            if len(a) >= 2 and len(b) >= 2:
                u, p = _scipy_stats.mannwhitneyu(a, b, alternative='two-sided')
                gt = sum(1 for x in a for y in b if x > y)
                lt = sum(1 for x in a for y in b if x < y)
                cd = (gt - lt) / (len(a) * len(b))
                lines.append(f"    Mann-Whitney U={u:.0f}  p={p:.4f}  cliffs_d={cd:+.2f}")
        lines.append("")
    summary = "\n".join(lines)
    with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary + "\n")
    print(summary)
    print(f"\nOutputs -> {out_dir}")


if __name__ == "__main__":
    main()
