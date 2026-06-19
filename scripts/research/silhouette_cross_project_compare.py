"""
silhouette_cross_project_compare.py
====================================
Compare silhouette feature distributions between two project cohorts.
Default: 2604_DV_DSS *_BL* sessions  vs  2603_SNLT_JG/Baseline *_Baseline*
sessions. Both groups represent "naive / pre-treatment" animals, so the
expectation is that silhouette features should be similar — large
discrepancies would indicate rig differences, animal-population biases,
or extraction artifacts.

Per-session summary stats (each becomes one data point per group):
    median(silhouette_frac)         — typical contact extent
    p95(silhouette_frac)            — peak contact (excluding outliers)
    median(silhouette_aspect)       — typical posture compactness
    median(silhouette_blob_frac)    — typical animal-only contact area

End-of-session retrieval artifacts (e.g. experimenter hand reaching in)
are filtered out two ways before computing stats:
    1. drop frames where silhouette_frac > 0.5
    2. drop the last 5% of frames per session as a defensive trim

Outputs:
    <out_dir>/per_session_stats.csv     one row per session, all groups
    <out_dir>/distributions.png         box + strip plots, group-by-metric
    <out_dir>/timecourses.png           median silhouette_frac vs frame, per session
    <out_dir>/summary.txt               group-level stats + Mann-Whitney
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats as _scipy_stats
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


# Default cohorts.
GROUPS_DEFAULT = [
    {
        "name": "2604_DV_DSS_BL",
        "features_dir": r"E:\RSVIDS\Blackbox\2604_DV_DSS\features",
        "name_filter":  "_BL_",
        "exclude":      ("_STIM",),
    },
    {
        "name": "SNLT_Baseline",
        "features_dir": r"E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\features",
        "name_filter":  "_Baseline_",
        "exclude":      (),
    },
]

# Filter / trim parameters.
ART_FRAC_THRESH    = 0.5      # drop frames > this — hand-in-frame artifacts
TAIL_TRIM_FRAC     = 0.05     # also drop the last 5% of each session


def discover_pkls(group: dict) -> list[str]:
    """Return all pkls under features_dir whose filename contains
    group['name_filter'] and none of group['exclude']. Newest one per
    session_name (in case multiple hashes coexist) wins."""
    candidates = sorted(glob.glob(os.path.join(group["features_dir"], "*.pkl")))
    out: dict[str, str] = {}  # session_name -> path (newest wins)
    for p in candidates:
        fname = os.path.basename(p)
        if group["name_filter"] not in fname:
            continue
        if any(x in fname for x in group["exclude"]):
            continue
        # Strip the trailing _features_<hash>.pkl to recover session name
        stem = fname.split("_features_")[0]
        if stem in out:
            # Prefer the file with the newer mtime
            if os.path.getmtime(p) > os.path.getmtime(out[stem]):
                out[stem] = p
        else:
            out[stem] = p
    return [out[k] for k in sorted(out.keys())]


def silhouette_stats_for_session(pkl_path: str) -> dict:
    """Compute per-session stats; skip if the pkl lacks silhouette
    columns (caller should filter out None returns)."""
    X = pd.read_pickle(pkl_path)
    cols = ('silhouette_frac', 'silhouette_blob_frac', 'silhouette_aspect')
    if not all(c in X.columns for c in cols):
        return None  # type: ignore

    sf = X['silhouette_frac'].astype(float)
    sb = X['silhouette_blob_frac'].astype(float)
    sa = X['silhouette_aspect'].astype(float)

    # Filter retrieval artifacts + trim last 5%
    n = len(X)
    cutoff = int(n * (1.0 - TAIL_TRIM_FRAC))
    mask = (sf <= ART_FRAC_THRESH) & (sf.index < cutoff)
    n_dropped_artifact = int((sf > ART_FRAC_THRESH).sum())
    n_kept = int(mask.sum())

    sf_k = sf[mask]; sb_k = sb[mask]; sa_k = sa[mask]
    return {
        'session':          os.path.basename(pkl_path).split('_features_')[0],
        'pkl':              pkl_path,
        'n_frames':         n,
        'n_kept':           n_kept,
        'n_dropped_artifact': n_dropped_artifact,
        'sf_min':           float(sf_k.min())    if n_kept else float('nan'),
        'sf_median':        float(sf_k.median()) if n_kept else float('nan'),
        'sf_p95':           float(np.nanpercentile(sf_k.values, 95)) if n_kept else float('nan'),
        'sf_max':           float(sf_k.max())    if n_kept else float('nan'),
        'sb_median':        float(sb_k.median()) if n_kept else float('nan'),
        'sa_median':        float(sa_k.median()) if n_kept else float('nan'),
        # keep the trimmed series for the time-course plot below
        '_series_sf':       sf_k,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None,
                    help="Output dir (default: timestamped under "
                          "E:\\PixelPaws\\analysis_output)")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "analysis_output",
        f"silhouette_cross_project_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    series_by_group: dict[str, list] = {}
    for group in GROUPS_DEFAULT:
        pkls = discover_pkls(group)
        print(f"\n[{group['name']}] {len(pkls)} pkls found in "
              f"{group['features_dir']}")
        for p in pkls:
            print(f"  - {os.path.basename(p)}")
        kept = []
        for p in pkls:
            s = silhouette_stats_for_session(p)
            if s is None:
                print(f"  ! {os.path.basename(p)}: missing silhouette cols, skipping")
                continue
            s['group'] = group['name']
            kept.append(s)
            rows.append({k: v for k, v in s.items() if k != '_series_sf'})
        series_by_group[group['name']] = kept
        print(f"  -> {len(kept)} usable sessions")

    if not rows:
        sys.exit("No usable pkls in any group. Check feature dirs / "
                 "verify silhouette extraction completed.")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'per_session_stats.csv'), index=False)

    # Distribution comparison: 4 metrics × 2 groups = 4 panels
    metrics = [
        ('sf_median', 'silhouette_frac (median)\nTypical contact extent'),
        ('sf_p95',    'silhouette_frac (p95)\nPeak contact'),
        ('sa_median', 'silhouette_aspect (median)\n1=square, →0=elongated'),
        ('sb_median', 'silhouette_blob_frac (median)\nAnimal-only contact area'),
    ]
    group_names = [g['name'] for g in GROUPS_DEFAULT]
    palette = ['#2266aa', '#cc6622']

    fig, axes = plt.subplots(1, len(metrics),
                              figsize=(2.6 * len(metrics), 4.2),
                              constrained_layout=True)
    rng = np.random.default_rng(42)
    for ax, (col, title) in zip(axes, metrics):
        positions = []
        for gi, gname in enumerate(group_names):
            sub = df[df['group'] == gname][col].dropna().values
            positions.append(sub)
            xs = gi + rng.uniform(-0.10, 0.10, len(sub))
            ax.scatter(xs, sub, color=palette[gi], s=42, alpha=0.85,
                       edgecolor='black', linewidth=0.4, zorder=3)
        # Boxplot under the strip
        bp = ax.boxplot(positions, positions=range(len(group_names)),
                          widths=0.55, showfliers=False, patch_artist=True,
                          zorder=1)
        for box, c in zip(bp['boxes'], palette):
            box.set(facecolor=c + '22', edgecolor=c, linewidth=1.2)
        for med in bp['medians']:
            med.set(color='black', linewidth=1.4)
        ax.set_xticks(range(len(group_names)))
        ax.set_xticklabels([f'{g}\n(n={(df["group"]==g).sum()})'
                              for g in group_names], fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Cross-project silhouette comparison '
                  '(per-session summary stats)', fontsize=12)
    fig.savefig(os.path.join(out_dir, 'distributions.png'), dpi=120)
    plt.close(fig)

    # Time-course: median silhouette_frac vs frame, per session, colored by group
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for gi, gname in enumerate(group_names):
        for s in series_by_group[gname]:
            srs = s['_series_sf']
            # Bin by 1000 frames for legibility
            if len(srs) < 1000:
                continue
            binned = srs.groupby(srs.index // 1000).median()
            ax.plot(binned.index * 1000, binned.values,
                     color=palette[gi], alpha=0.5, linewidth=0.9,
                     label=gname if s == series_by_group[gname][0] else None)
    ax.set_xlabel('frame')
    ax.set_ylabel('median silhouette_frac (per 1000-frame bin)')
    ax.set_title('Per-session silhouette_frac time-course '
                  '(retrieval artifacts removed)')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')
    fig.savefig(os.path.join(out_dir, 'timecourses.png'), dpi=120)
    plt.close(fig)

    # Stats summary
    lines = [
        f"Cross-project silhouette comparison",
        f"  Out dir   : {out_dir}",
        f"  Generated : {datetime.now().isoformat(timespec='seconds')}",
        f"  Filter    : drop frames with silhouette_frac > {ART_FRAC_THRESH} "
        f"(retrieval artifacts) AND drop last {TAIL_TRIM_FRAC*100:.0f}% of frames",
        ""]
    for gname in group_names:
        sub = df[df['group'] == gname]
        lines.append(f"  [{gname}] n={len(sub)} sessions")
        for col, _label in metrics:
            vals = sub[col].dropna().values
            if len(vals) == 0:
                continue
            lines.append(
                f"    {col:<14} mean={vals.mean():.4f}  "
                f"sd={vals.std(ddof=1) if len(vals)>1 else 0:.4f}  "
                f"med={np.median(vals):.4f}  "
                f"min={vals.min():.4f}  max={vals.max():.4f}")
        lines.append("")

    if _SCIPY_OK and len(group_names) == 2:
        a = df[df['group'] == group_names[0]]
        b = df[df['group'] == group_names[1]]
        lines.append(f"  Mann-Whitney U  ({group_names[0]} vs {group_names[1]}):")
        for col, label in metrics:
            av = a[col].dropna().values
            bv = b[col].dropna().values
            if len(av) >= 2 and len(bv) >= 2:
                u, pval = _scipy_stats.mannwhitneyu(av, bv, alternative='two-sided')
                # Cliff's delta as a non-parametric effect size
                gt = sum(1 for x in av for y in bv if x > y)
                lt = sum(1 for x in av for y in bv if x < y)
                cd = (gt - lt) / (len(av) * len(bv))
                lines.append(
                    f"    {col:<14}  U={u:>7.1f}  p={pval:.4f}  "
                    f"cliffs_d={cd:+.2f}")

    summary = "\n".join(lines)
    with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(summary + "\n")
    print()
    print(summary)
    print()
    print(f"Outputs → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
