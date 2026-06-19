"""
silhouette_bouts.py — Detect "lying down" bouts from silhouette_frac.

Pipeline (per session):
    1. Load silhouette_frac from the cache pkl
    2. Drop end-of-session retrieval artifacts (frac > 0.5; last 5%)
    3. Smooth with a centered rolling MEDIAN (robust to single-frame spikes)
    4. Threshold per-session at fraction-of-max (default: smoothed > 0.85 * p99
       of the smoothed signal). Rig-invariant: the threshold scales with the
       animal's own max contact, so cross-rig comparisons make sense.
    5. Apply minimum-bout-duration filter (default 5 s) to drop transient peaks.
    6. Emit per-bout records and per-session aggregates.

Outputs (in E:\\PixelPaws\\analysis_output\\silhouette_bouts_<ts>\\):
    bouts.csv               one row per detected bout (group, session, ...)
    session_summary.csv     one row per session: bout count, total time, etc.
    group_summary.txt       group-level aggregates + Mann-Whitney
    bout_examples_<group>_<session>.png   2 example bouts visualized

CLI:
    py silhouette_bouts.py
        [--smooth-s 1.0]      smoothing window in seconds (default 1.0)
        [--threshold 0.85]    fraction-of-max threshold (default 0.85)
        [--min-bout-s 5.0]    min bout duration in seconds (default 5)
"""

from __future__ import annotations
import argparse
import glob
import os
import sys
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats as _scipy_stats
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


GROUPS = [
    {
        "name": "2604_DV_DSS_BL",
        "features_dir": r"E:\RSVIDS\Blackbox\2604_DV_DSS\features",
        "videos_dir":   r"E:\RSVIDS\Blackbox\2604_DV_DSS\videos",
        "filter":       "_BL_",
        "exclude":      ("_STIM",),
        "fps_default":  60.0,
    },
    {
        "name": "SNLT_Baseline",
        "features_dir": r"E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\features",
        "videos_dir":   r"E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\videos",
        "filter":       "_Baseline_",
        "exclude":      (),
        "fps_default":  60.0,
    },
]
ART_FRAC_THRESH = 0.5
TAIL_TRIM_FRAC  = 0.05


# ─────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────

def discover_pkls(g):
    out = {}
    for p in sorted(glob.glob(os.path.join(g["features_dir"], "*.pkl"))):
        f = os.path.basename(p)
        if g["filter"] not in f: continue
        if any(x in f for x in g["exclude"]): continue
        stem = f.split("_features_")[0]
        if stem not in out or os.path.getmtime(p) > os.path.getmtime(out[stem]):
            out[stem] = p
    return [out[k] for k in sorted(out)]


def find_video(videos_dir: str, session: str) -> str | None:
    for ext in (".mp4", ".avi", ".mov", ".mkv"):
        p = os.path.join(videos_dir, session + ext)
        if os.path.isfile(p):
            return p
    return None


# ─────────────────────────────────────────────────────────────────────
# Bout detection
# ─────────────────────────────────────────────────────────────────────

def detect_bouts(sf: pd.Series, fps: float,
                 smooth_s: float, frac_of_max: float, min_bout_s: float
                 ) -> tuple[pd.Series, pd.Series, list[dict]]:
    """
    Returns:
        smoothed:  pd.Series of rolling-median silhouette_frac
        in_bout:   pd.Series of bool, post min-bout filter
        bouts:     list of dicts with start_frame, end_frame, duration_s, mean_smoothed
    """
    n = len(sf)
    # Mask out artifacts and tail
    cutoff = int(n * (1 - TAIL_TRIM_FRAC))
    valid = (sf <= ART_FRAC_THRESH) & (sf.index < cutoff)

    # Smooth with rolling median, centered
    win = max(1, int(round(smooth_s * fps)))
    if win % 2 == 0: win += 1   # centered window must be odd
    smoothed = sf.rolling(win, center=True, min_periods=1).median()

    # Per-session threshold on smoothed signal: frac_of_max * p99 of smoothed
    # Use the SMOOTHED signal's p99 so the threshold is on the same scale as
    # what we're thresholding (post-smoothing peaks, not raw spikes).
    p99 = float(np.nanpercentile(smoothed[valid].values, 99))
    thresh = frac_of_max * p99

    in_bout_raw = (smoothed > thresh) & valid
    arr = in_bout_raw.values.astype(np.uint8)

    # Run-length: find True runs
    diff = np.diff(np.concatenate(([0], arr, [0])))
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]   # exclusive
    min_bout_frames = max(1, int(round(min_bout_s * fps)))

    bouts = []
    keep = np.zeros(len(arr), dtype=bool)
    for s, e in zip(starts, ends):
        dur_frames = e - s
        if dur_frames < min_bout_frames:
            continue
        keep[s:e] = True
        bouts.append({
            'start_frame': int(s),
            'end_frame':   int(e),
            'duration_s':  round(dur_frames / fps, 3),
            'mean_smoothed': round(float(smoothed.iloc[s:e].mean()), 4),
            'peak_smoothed': round(float(smoothed.iloc[s:e].max()),  4),
        })

    in_bout = pd.Series(keep, index=sf.index)
    return smoothed, in_bout, bouts


# ─────────────────────────────────────────────────────────────────────
# Visualization: bout examples
# ─────────────────────────────────────────────────────────────────────

def render_bout_examples(group_name: str, session: str, video_path: str,
                          sf: pd.Series, smoothed: pd.Series,
                          bouts: list[dict], thresh: float, fps: float,
                          out_png: str, n_examples: int = 2):
    """For up to n_examples bouts, draw: a slice of the smoothed signal
    around the bout, plus 4 thumbnails sampled uniformly inside the bout.
    Picks the LONGEST bouts as examples — most informative."""
    if not bouts:
        # Render a "no bouts found" placeholder for inventory completeness
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, f'{session} — no bouts detected',
                 ha='center', va='center', transform=ax.transAxes,
                 fontsize=11, color='gray')
        ax.axis('off')
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        return

    pick = sorted(bouts, key=lambda b: -b['duration_s'])[:n_examples]

    fig, axes = plt.subplots(len(pick), 5,
                              figsize=(13, 2.6 * len(pick)),
                              constrained_layout=True, squeeze=False,
                              gridspec_kw={'width_ratios': [3, 1, 1, 1, 1]})

    cap = cv2.VideoCapture(video_path) if video_path else None
    for ri, b in enumerate(pick):
        s, e = b['start_frame'], b['end_frame']
        # Time-course panel (left): show ±50% of bout duration on each side
        pad = max(60, (e - s) // 2)
        lo = max(0, s - pad); hi = min(len(sf), e + pad)
        ax_tc = axes[ri, 0]
        ax_tc.plot(sf.index[lo:hi], sf.values[lo:hi],
                    color='#bbbbbb', linewidth=0.6, label='raw')
        ax_tc.plot(smoothed.index[lo:hi], smoothed.values[lo:hi],
                    color='#222222', linewidth=1.4, label='smoothed')
        ax_tc.axhline(thresh, color='crimson', linestyle='--', linewidth=1.0,
                       label=f'thresh={thresh:.3f}')
        ax_tc.axvspan(s, e, color='gold', alpha=0.4, label='bout')
        ax_tc.set_xlabel('frame')
        ax_tc.set_ylabel('silhouette_frac')
        ax_tc.set_title(f'bout {ri+1}/{len(pick)}: f{s}-{e}, '
                         f'dur={b["duration_s"]:.1f}s, '
                         f'peak={b["peak_smoothed"]:.3f}', fontsize=9)
        ax_tc.legend(loc='upper right', fontsize=7, framealpha=0.85)
        ax_tc.grid(alpha=0.3)

        # 4 thumbnails sampled uniformly inside the bout
        sample_frames = np.linspace(s, e - 1, 4).astype(int)
        for ci, f in enumerate(sample_frames):
            ax_p = axes[ri, ci + 1]
            ax_p.set_xticks([]); ax_p.set_yticks([])
            if cap is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
                ok, fr = cap.read()
                if ok and fr is not None:
                    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                    ax_p.imshow(g, cmap='gray', vmin=0, vmax=255)
            ax_p.set_title(f'f{f}\nfrac={sf.iloc[int(f)]:.3f}', fontsize=8)

    if cap is not None:
        cap.release()
    fig.suptitle(f'{group_name}  |  {session}  |  longest {len(pick)} bouts',
                  fontsize=11)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smooth-s", type=float, default=1.0,
                     help="Rolling-median window in seconds (default 1.0)")
    ap.add_argument("--threshold", type=float, default=0.85,
                     help="Fraction-of-max threshold (default 0.85)")
    ap.add_argument("--min-bout-s", type=float, default=5.0,
                     help="Minimum bout duration in seconds (default 5.0)")
    ap.add_argument("--examples-per-session", type=int, default=2)
    args = ap.parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "analysis_output",
                            f"silhouette_bouts_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)

    bout_rows = []
    sess_rows = []

    for g in GROUPS:
        pkls = discover_pkls(g)
        print(f"\n[{g['name']}] {len(pkls)} sessions")
        for p in pkls:
            X = pd.read_pickle(p)
            if 'silhouette_frac' not in X.columns:
                print(f"  skip {os.path.basename(p)} (no silhouette)")
                continue
            sf = X['silhouette_frac'].astype(float)
            session = os.path.basename(p).split("_features_")[0]
            video = find_video(g["videos_dir"], session)
            fps = g["fps_default"]   # could read from video; default 60 is correct here

            smoothed, in_bout, bouts = detect_bouts(
                sf, fps, args.smooth_s, args.threshold, args.min_bout_s)
            thresh = args.threshold * float(np.nanpercentile(
                smoothed[(sf <= ART_FRAC_THRESH) & (sf.index < int(len(sf)*0.95))].values,
                99))

            n_bouts = len(bouts)
            total_bout_frames = sum(b['end_frame'] - b['start_frame']
                                      for b in bouts)
            session_n = int(((sf <= ART_FRAC_THRESH) &
                              (sf.index < int(len(sf)*(1-TAIL_TRIM_FRAC)))).sum())
            frac_in_bout = (total_bout_frames / session_n) if session_n else 0.0
            durs = [b['duration_s'] for b in bouts]

            sess_rows.append({
                'group':            g['name'],
                'session':          session,
                'fps':              fps,
                'n_frames_used':    session_n,
                'threshold':        round(thresh, 4),
                'n_bouts':          n_bouts,
                'total_bout_s':     round(total_bout_frames / fps, 1),
                'frac_in_bout':     round(frac_in_bout, 4),
                'mean_bout_s':      round(float(np.mean(durs)), 2) if durs else 0.0,
                'median_bout_s':    round(float(np.median(durs)), 2) if durs else 0.0,
                'max_bout_s':       round(float(np.max(durs)), 2) if durs else 0.0,
            })
            for b in bouts:
                bout_rows.append({
                    'group':         g['name'],
                    'session':       session,
                    **b,
                })

            # Visual verification: pick longest N bouts
            png = os.path.join(out_dir,
                                f"bout_examples_{g['name']}_{session}.png")
            render_bout_examples(g['name'], session, video,
                                  sf, smoothed, bouts, thresh, fps, png,
                                  n_examples=args.examples_per_session)
            print(f"  {session:<40} bouts={n_bouts:>3}  "
                  f"frac_in_bout={frac_in_bout:.3f}  "
                  f"mean_dur={np.mean(durs):.1f}s" if durs
                  else f"  {session:<40} bouts=  0")

    bouts_df = pd.DataFrame(bout_rows)
    sess_df  = pd.DataFrame(sess_rows)
    bouts_df.to_csv(os.path.join(out_dir, "bouts.csv"), index=False)
    sess_df.to_csv (os.path.join(out_dir, "session_summary.csv"), index=False)

    # Group summary + Mann-Whitney
    lines = [f"Silhouette lying-bout detection",
             f"  Smooth window: {args.smooth_s}s  Threshold: {args.threshold} of session p99  "
             f"Min bout: {args.min_bout_s}s",
             f"  Out: {out_dir}",
             ""]
    metrics = [
        ('n_bouts',      'Number of bouts'),
        ('frac_in_bout', 'Fraction of session in bout'),
        ('mean_bout_s',  'Mean bout duration (s)'),
        ('median_bout_s','Median bout duration (s)'),
        ('total_bout_s', 'Total time in bout (s)'),
    ]
    group_names = [g['name'] for g in GROUPS]
    for col, label in metrics:
        lines.append(f"  {label} ({col}):")
        for gname in group_names:
            v = sess_df[sess_df['group']==gname][col].dropna().values
            if len(v):
                lines.append(f"    {gname:<22} mean={v.mean():.2f}  "
                             f"med={np.median(v):.2f}  "
                             f"min={v.min():.2f}  max={v.max():.2f}  n={len(v)}")
        if _SCIPY_OK and len(group_names) == 2:
            a = sess_df[sess_df['group']==group_names[0]][col].dropna().values
            b = sess_df[sess_df['group']==group_names[1]][col].dropna().values
            if len(a) >= 2 and len(b) >= 2:
                u, p = _scipy_stats.mannwhitneyu(a, b, alternative='two-sided')
                gt = sum(1 for x in a for y in b if x > y)
                lt = sum(1 for x in a for y in b if x < y)
                cd = (gt - lt) / (len(a) * len(b))
                lines.append(f"    Mann-Whitney U={u:.0f}  p={p:.4f}  "
                             f"cliffs_d={cd:+.2f}")
        lines.append("")
    summary = "\n".join(lines)
    with open(os.path.join(out_dir, "group_summary.txt"), "w",
               encoding="utf-8") as f:
        f.write(summary + "\n")
    print()
    print(summary)
    print(f"Outputs -> {out_dir}")


if __name__ == "__main__":
    main()
