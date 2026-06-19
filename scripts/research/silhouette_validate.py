"""
silhouette_validate.py — sanity-check the new silhouette columns in a
freshly extracted feature pkl.

Given a project + session, the script:
  1. Loads the cache pkl, confirms the three silhouette columns exist
  2. Reports descriptive stats: range, NaN%, distribution percentiles
  3. Plots silhouette_frac over time (line plot, full session)
  4. Dumps the LOWEST-silhouette and HIGHEST-silhouette frames to disk
     so the user can eyeball whether the feature really tracks
     belly-on-floor extent (lo = standing/rearing; hi = pressed flat)
  5. Writes a one-page summary PNG with the time-course and the patch grid

Outputs land in <project>/analysis/silhouette_validate_<session>_<ts>/

Usage:
    python silhouette_validate.py <project> <session>
        [--n-examples 6]   number of low + number of high frames to dump
        [--floor 35]       reuse floor for re-rendering masks (info only)
"""

from __future__ import annotations
import argparse
import os
import sys
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from feature_cache import FeatureCacheManager
    _CACHE_OK = True
except ImportError:
    FeatureCacheManager = None
    _CACHE_OK = False


REQUIRED = ('silhouette_frac', 'silhouette_blob_frac', 'silhouette_aspect')


def load_features(project: str, session: str) -> tuple[pd.DataFrame, str]:
    feat_dir = os.path.join(project, 'features')
    if _CACHE_OK:
        cp = FeatureCacheManager.find_any_cache(
            session, feat_dir, '', project_root=project)
        if cp is not None:
            return pd.read_pickle(cp), cp
    cands = [f for f in os.listdir(feat_dir)
              if f.startswith(session) and f.endswith('.pkl')]
    if not cands:
        sys.exit(f"No cached pkl for '{session}' in {feat_dir}")
    cands.sort(key=lambda f: os.path.getmtime(os.path.join(feat_dir, f)),
                reverse=True)
    p = os.path.join(feat_dir, cands[0])
    return pd.read_pickle(p), p


def find_video(project: str, session: str) -> str:
    vid_dir = os.path.join(project, 'videos')
    for ext in ('.mp4', '.avi', '.mov', '.mkv'):
        p = os.path.join(vid_dir, session + ext)
        if os.path.isfile(p):
            return p
    sys.exit(f"No video '{session}.<mp4|avi|mov|mkv>' in {vid_dir}")


def grab_frames(video_path: str, frame_idxs: list[int]) -> dict:
    """Read specific frames from a video file. Returns {idx: gray_image}."""
    cap = cv2.VideoCapture(video_path)
    out = {}
    for f in sorted(set(frame_idxs)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        out[f] = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                   if ok and frame is not None else None)
    cap.release()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('project')
    ap.add_argument('session')
    ap.add_argument('--n-examples', type=int, default=6)
    ap.add_argument('--floor', type=int, default=35)
    args = ap.parse_args()

    print(f"[1/4] Loading features for '{args.session}'…")
    X, pkl_path = load_features(args.project, args.session)
    print(f"      {pkl_path}")
    print(f"      shape={X.shape}")

    missing = [c for c in REQUIRED if c not in X.columns]
    if missing:
        sys.exit(f"FAIL: cache lacks columns {missing}. "
                 f"Re-extract with compute_silhouette=True.")
    print(f"      ✓ All three silhouette columns present")

    sf = X['silhouette_frac'].astype(float)
    sb = X['silhouette_blob_frac'].astype(float)
    sa = X['silhouette_aspect'].astype(float)

    print(f"\n[2/4] Descriptive stats:")
    print(f"      silhouette_frac      "
          f"min={sf.min():.4f} med={sf.median():.4f} max={sf.max():.4f} "
          f"NaN%={sf.isna().mean()*100:.1f}")
    print(f"      silhouette_blob_frac "
          f"min={sb.min():.4f} med={sb.median():.4f} max={sb.max():.4f} "
          f"NaN%={sb.isna().mean()*100:.1f}")
    print(f"      silhouette_aspect    "
          f"min={sa.min():.4f} med={sa.median():.4f} max={sa.max():.4f} "
          f"NaN%={sa.isna().mean()*100:.1f}")

    # Pick representative low/high frames spread ACROSS the session, not
    # clustered around the single global extremum. Approach: split the
    # session into N equal time chunks; within each chunk, pick the
    # frame with the lowest (resp. highest) silhouette_frac. Guarantees
    # temporal diversity — we see N different events, not N adjacent
    # frames of one event.
    sf_clean = sf.where(sb.notna())
    n = args.n_examples
    n_frames = len(X)
    chunk_size = max(1, n_frames // n)
    lo_idx, hi_idx = [], []
    for i in range(n):
        s = i * chunk_size
        e = (i + 1) * chunk_size if i < n - 1 else n_frames
        chunk_lo = sf_clean.iloc[s:e]
        chunk_hi = sf.iloc[s:e]
        if chunk_lo.notna().any():
            lo_idx.append(int(chunk_lo.idxmin()))
        if chunk_hi.notna().any():
            hi_idx.append(int(chunk_hi.idxmax()))

    print(f"\n[3/4] Grabbing {2*n} example frames from video…")
    video_path = find_video(args.project, args.session)
    frames = grab_frames(video_path, lo_idx + hi_idx)

    out_dir = os.path.join(
        args.project, 'analysis',
        f'silhouette_validate_{args.session}_{datetime.now():%Y%m%d_%H%M%S}')
    os.makedirs(out_dir, exist_ok=True)

    # Time-course plot
    print(f"[4/4] Rendering plots → {out_dir}")
    fig, ax = plt.subplots(2, 1, figsize=(12, 6),
                            constrained_layout=True, sharex=True)
    ax[0].plot(sf.index, sf.values, color='#222', linewidth=0.4)
    ax[0].plot(sf.index, sb.values, color='#cc2222', linewidth=0.4,
                alpha=0.7)
    ax[0].set_ylabel('silhouette_frac\n(black) / blob_frac (red)')
    ax[0].grid(alpha=0.3)
    ax[1].plot(sa.index, sa.values, color='#226688', linewidth=0.4)
    ax[1].set_ylabel('silhouette_aspect')
    ax[1].set_xlabel('frame')
    ax[1].grid(alpha=0.3)
    fig.suptitle(f'{args.session} — silhouette feature time-course',
                  fontsize=11)
    fig.savefig(os.path.join(out_dir, 'silhouette_timecourse.png'), dpi=120)
    plt.close(fig)

    # Patch grid: low (top row) vs high (bottom row)
    fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 4.5),
                              squeeze=False, constrained_layout=True)
    for col_i in range(n):
        for row_i, (idxs, lbl) in enumerate(
                [(lo_idx, 'LOW'), (hi_idx, 'HIGH')]):
            ax_p = axes[row_i, col_i]
            ax_p.set_xticks([]); ax_p.set_yticks([])
            f = idxs[col_i]
            img = frames.get(f)
            if img is not None:
                ax_p.imshow(img, cmap='gray', vmin=0, vmax=255)
                v_sf = sf.iloc[f]
                v_sa = sa.iloc[f] if not np.isnan(sa.iloc[f]) else float('nan')
                ax_p.set_title(
                    f'f{f}\nfrac={v_sf:.3f}\nasp={v_sa:.2f}',
                    fontsize=8)
            else:
                ax_p.text(0.5, 0.5, '(no frame)', ha='center', va='center',
                            transform=ax_p.transAxes, color='gray')
    axes[0, 0].set_ylabel('LOW silhouette_frac\n(standing / rearing?)',
                            fontsize=10)
    axes[1, 0].set_ylabel('HIGH silhouette_frac\n(belly down?)',
                            fontsize=10)
    fig.suptitle(f'{args.session} — silhouette extremes (floor={args.floor})',
                  fontsize=11)
    fig.savefig(os.path.join(out_dir, 'silhouette_extremes.png'), dpi=120)
    plt.close(fig)

    # Summary text
    summary_lines = [
        f"Silhouette validation — {args.session}",
        f"  Cache pkl : {pkl_path}",
        f"  Frames    : {len(X)}",
        f"  silhouette_frac      min={sf.min():.4f}  med={sf.median():.4f}  max={sf.max():.4f}  NaN%={sf.isna().mean()*100:.1f}",
        f"  silhouette_blob_frac min={sb.min():.4f}  med={sb.median():.4f}  max={sb.max():.4f}  NaN%={sb.isna().mean()*100:.1f}",
        f"  silhouette_aspect    min={sa.min():.4f}  med={sa.median():.4f}  max={sa.max():.4f}  NaN%={sa.isna().mean()*100:.1f}",
        "",
        f"Low-silhouette frame indices : {lo_idx}",
        f"High-silhouette frame indices: {hi_idx}",
        "",
        "Validation criteria:",
        "  PASS if low frames show standing / rearing / off-floor postures",
        "       AND high frames show belly-pressed / lying-flat postures",
        "       AND silhouette_frac range is at least ~5x (e.g. 0.02 → 0.20)",
        "  FAIL if low and high frames look similar — feature isn't",
        "       discriminating posture, threshold may need tuning",
    ]
    summary = "\n".join(summary_lines)
    with open(os.path.join(out_dir, 'summary.txt'), 'w',
               encoding='utf-8') as f:
        f.write(summary + "\n")
    print()
    print(summary)
    print()
    print(f"Outputs → {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
