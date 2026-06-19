"""
belly_groundtruth_check.py
==========================
Quick separability check for belly-down detection signals.

Workflow
--------
1. Open the session in the GUI's Live Preview, scrub through, and write
   down 5-10 frame indices that are *unambiguously* belly-down on the
   floor, plus 5-10 that are *unambiguously* not (standing, walking,
   rearing, sleeping curled — varied is good).
2. Run this script with those frames as ranges:

       python belly_groundtruth_check.py <project> <session> \
           --bd  "1450-1480,3210-3240,5800,9120-9135" \
           --nbd "200-220,4500-4520,7800-7810"

   Single integers are treated as single-frame events.

3. Open the output folder. The patch grid + signal distributions tell
   you whether brightness alone separates the classes, or whether
   texture / posture is needed.

CLI
---
    python belly_groundtruth_check.py PROJECT SESSION
        [--bd "<ranges>"] [--nbd "<ranges>"]
        [--labels labels.csv]
        [--bp centroid|tailbase]   default: centroid
        [--roi 30]                 ROI half-size in px (default 30)

Either supply --bd/--nbd OR --labels labels.csv with columns:
    frame_start, frame_end, belly_down       (0 or 1)
        OR
    frame, belly_down

Output (in <project>/analysis/belly_check_<session>_<ts>/)
    per_event_signals.csv     — one row per labeled event
    signal_distributions.png  — strip plot per signal × class
    roi_patches.png           — patch grid: belly-down (top) vs not (bot)
    summary.txt               — Cohen's d per signal
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


# ─────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────

def load_features(project: str, session: str) -> pd.DataFrame:
    """Try FeatureCacheManager first; fall back to newest matching pkl."""
    feat_dir = os.path.join(project, 'features')
    if _CACHE_OK:
        try:
            cp = FeatureCacheManager.find_any_cache(
                session, feat_dir, '', project_root=project)
            if cp is not None:
                return pd.read_pickle(cp)
        except Exception:
            pass
    if not os.path.isdir(feat_dir):
        raise FileNotFoundError(f"No features dir at {feat_dir}")
    cands = [f for f in os.listdir(feat_dir)
             if f.startswith(session) and f.endswith('.pkl')]
    if not cands:
        raise FileNotFoundError(
            f"No cached pkl for '{session}' in {feat_dir}")
    cands.sort(key=lambda f: os.path.getmtime(os.path.join(feat_dir, f)),
               reverse=True)
    return pd.read_pickle(os.path.join(feat_dir, cands[0]))


def find_video(project: str, session: str) -> str:
    vid_dir = os.path.join(project, 'videos')
    for ext in ('.mp4', '.avi', '.mov', '.mkv'):
        p = os.path.join(vid_dir, session + ext)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"No video '{session}.{{mp4,avi,mov,mkv}}' in {vid_dir}")


def parse_ranges(spec: str) -> list[tuple[int, int]]:
    """'1500-1520,3210,4400-4410' → [(1500,1520),(3210,3210),(4400,4410)]"""
    out: list[tuple[int, int]] = []
    if not spec:
        return out
    for tok in spec.split(','):
        tok = tok.strip()
        if not tok:
            continue
        if '-' in tok:
            a, b = tok.split('-', 1)
            out.append((int(a), int(b)))
        else:
            out.append((int(tok), int(tok)))
    return out


def labels_from_args(args) -> pd.DataFrame:
    if args.labels:
        df = pd.read_csv(args.labels)
        if 'frame_start' not in df.columns:
            if 'frame' in df.columns:
                df = df.rename(columns={'frame': 'frame_start'})
                df['frame_end'] = df['frame_start']
            else:
                raise ValueError(
                    "labels CSV must have frame_start (or frame) column")
        if 'frame_end' not in df.columns:
            df['frame_end'] = df['frame_start']
        if 'belly_down' not in df.columns:
            raise ValueError("labels CSV must have 'belly_down' column (0/1)")
        return df[['frame_start', 'frame_end', 'belly_down']].astype(int)

    rows = []
    for fs, fe in parse_ranges(args.bd or ''):
        rows.append({'frame_start': fs, 'frame_end': fe, 'belly_down': 1})
    for fs, fe in parse_ranges(args.nbd or ''):
        rows.append({'frame_start': fs, 'frame_end': fe, 'belly_down': 0})
    if not rows:
        raise ValueError(
            "No labels supplied. Use --bd / --nbd or --labels.")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Patch + texture
# ─────────────────────────────────────────────────────────────────────

def extract_patches(video_path: str, frames: list[int],
                     x_arr: np.ndarray, y_arr: np.ndarray,
                     half: int) -> dict[int, np.ndarray | None]:
    cap = cv2.VideoCapture(video_path)
    out: dict[int, np.ndarray | None] = {}
    for f in sorted(set(frames)):
        if f < 0 or f >= len(x_arr):
            out[f] = None
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok or frame is None:
            out[f] = None
            continue
        x, y = x_arr[f], y_arr[f]
        if not (np.isfinite(x) and np.isfinite(y)):
            out[f] = None
            continue
        x, y = int(round(x)), int(round(y))
        h, w = frame.shape[:2]
        x0, x1 = max(0, x - half), min(w, x + half)
        y0, y1 = max(0, y - half), min(h, y + half)
        if x1 - x0 < 4 or y1 - y0 < 4:
            out[f] = None
            continue
        out[f] = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    cap.release()
    return out


def texture_stats(patch: np.ndarray | None,
                   bright_floor: int = 180) -> dict[str, float]:
    """Per-patch: pixel std, mean gradient mag, fraction above bright floor.
    Pressed-belly hypothesis: tex_std + tex_grad drop, frac_bright rises.
    """
    if patch is None or patch.size == 0:
        return {'tex_std': np.nan, 'tex_grad': np.nan,
                f'frac_bright_{bright_floor}': np.nan}
    p = patch.astype(np.float32)
    gx, gy = np.gradient(p)
    return {
        'tex_std':  float(p.std()),
        'tex_grad': float(np.mean(np.sqrt(gx * gx + gy * gy))),
        f'frac_bright_{bright_floor}': float((patch > bright_floor).mean()),
    }


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float('nan')
    pooled = np.sqrt(
        ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
        / (len(a) + len(b) - 2))
    if pooled < 1e-12:
        return float('nan')
    return float((a.mean() - b.mean()) / pooled)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=("Belly-down separability check. Compares brightness "
                     "and texture signals on hand-labeled frames."))
    ap.add_argument('project')
    ap.add_argument('session')
    ap.add_argument('--bd', help='belly-down frames (e.g. "1500-1520,3210")')
    ap.add_argument('--nbd', help='not-belly-down frames')
    ap.add_argument('--labels', help='alternative: CSV with frame_start, '
                                       'frame_end, belly_down')
    ap.add_argument('--bp', default='centroid',
                     choices=['centroid', 'tailbase'])
    ap.add_argument('--roi', type=int, default=30,
                     help='ROI half-size in px (default 30)')
    ap.add_argument('--bright-floor', type=int, default=180,
                     help='abs brightness floor for frac_bright (default 180)')
    args = ap.parse_args()

    bp = args.bp
    half = args.roi

    print(f"[1/5] Loading features for '{args.session}'…")
    X = load_features(args.project, args.session)

    pix_col  = f'Pix_{bp}'
    dbrt_col = f'Pix_baseline_sub_{bp}'
    h_col    = f'{bp}_Height'
    x_col, y_col = f'{bp}_x', f'{bp}_y'

    if pix_col not in X.columns:
        sys.exit(f"ERROR: '{pix_col}' missing from cache. "
                 f"Re-extract features with '{bp}' in bp_pixbrt_list.")
    if x_col not in X.columns or y_col not in X.columns:
        sys.exit(f"ERROR: '{x_col}'/'{y_col}' missing from cache.")

    print(f"[2/5] Locating video…")
    video_path = find_video(args.project, args.session)

    print(f"[3/5] Parsing labels…")
    labels = labels_from_args(args)
    n_bd = int((labels['belly_down'] == 1).sum())
    n_nb = int((labels['belly_down'] == 0).sum())
    if n_bd == 0 or n_nb == 0:
        sys.exit(f"ERROR: need at least one belly-down AND one "
                 f"not-belly-down event (got bd={n_bd}, not={n_nb}).")
    print(f"      {n_bd} belly-down + {n_nb} not = {len(labels)} events")

    # Per-event signal: averaged over the labeled range from cache
    pix_mean = float(X[pix_col].mean())
    pix_std  = float(X[pix_col].std())
    rows: list[dict] = []
    for _, lab in labels.iterrows():
        fs, fe = int(lab['frame_start']), int(lab['frame_end'])
        sl = X.iloc[fs:fe + 1]
        r: dict[str, float | int] = {
            'frame_start': fs, 'frame_end': fe,
            'belly_down': int(lab['belly_down']),
            'Pix_raw':          float(sl[pix_col].mean()),
        }
        if dbrt_col in X.columns:
            r['Pix_baseline_sub'] = float(sl[dbrt_col].mean())
        if h_col in X.columns:
            r['Height'] = float(sl[h_col].mean())
        if pix_std > 1e-12:
            r['Pix_z'] = float(((sl[pix_col] - pix_mean) / pix_std).mean())
        rows.append(r)

    # Texture from video on the start frame of each event
    print(f"[4/5] Extracting {len(rows)} ROI patches from video "
          f"(this is the slow step)…")
    starts = [int(r['frame_start']) for r in rows]
    patches = extract_patches(video_path, starts,
                                X[x_col].values, X[y_col].values, half)
    for r in rows:
        r.update(texture_stats(patches.get(int(r['frame_start'])),
                                bright_floor=args.bright_floor))
    df = pd.DataFrame(rows)

    # Outputs
    print(f"[5/5] Rendering plots + summary…")
    out_dir = os.path.join(
        args.project, 'analysis',
        f'belly_check_{args.session}_{datetime.now():%Y%m%d_%H%M%S}')
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'per_event_signals.csv'), index=False)

    signal_cols = [c for c in df.columns
                    if c not in ('frame_start', 'frame_end', 'belly_down')
                    and df[c].notna().any()]

    # Strip plot — distribution per signal × class
    n = len(signal_cols)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 4),
                              constrained_layout=True)
    if n == 1:
        axes = [axes]
    rng = np.random.default_rng(42)
    for ax, sig in zip(axes, signal_cols):
        for cls, color in [(0, '#888888'), (1, '#cc2222')]:
            vals = df.loc[df['belly_down'] == cls, sig].dropna().values
            xs = cls + rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(xs, vals, color=color, s=44, alpha=0.75,
                        edgecolor='black', linewidth=0.4,
                        label=('belly-down' if cls else 'not'))
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['not', 'belly'])
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(sig, fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle(f'{args.session} — {bp} (n={len(df)})', fontsize=11)
    fig.savefig(os.path.join(out_dir, 'signal_distributions.png'), dpi=120)
    plt.close(fig)

    # Patch grid — belly-down on top, not on bottom
    bd_df  = df[df['belly_down'] == 1].reset_index(drop=True)
    nbd_df = df[df['belly_down'] == 0].reset_index(drop=True)
    n_cols = max(len(bd_df), len(nbd_df))
    fig, axes = plt.subplots(2, n_cols,
                              figsize=(1.8 * n_cols, 3.8),
                              squeeze=False, constrained_layout=True)
    for col_i in range(n_cols):
        for row_i, src_df in enumerate([bd_df, nbd_df]):
            ax = axes[row_i, col_i]
            ax.set_xticks([]); ax.set_yticks([])
            if col_i < len(src_df):
                f = int(src_df.iloc[col_i]['frame_start'])
                p = patches.get(f)
                if p is not None:
                    ax.imshow(p, cmap='gray', vmin=0, vmax=255)
                    ax.set_title(f'f{f}', fontsize=8)
                else:
                    ax.text(0.5, 0.5, '(no patch)', ha='center',
                             va='center', transform=ax.transAxes,
                             color='gray', fontsize=8)
            else:
                for s in ax.spines.values():
                    s.set_visible(False)
    axes[0, 0].set_ylabel('belly-down', fontsize=10)
    axes[1, 0].set_ylabel('not', fontsize=10)
    fig.suptitle(f'{args.session} — {bp} ROI patches (±{half} px)',
                  fontsize=11)
    fig.savefig(os.path.join(out_dir, 'roi_patches.png'), dpi=120)
    plt.close(fig)

    # Cohen's d summary
    bd_mask = df['belly_down'] == 1
    nb_mask = df['belly_down'] == 0
    summary_rows = []
    for sig in signal_cols:
        d = cohens_d(df.loc[bd_mask, sig].values,
                      df.loc[nb_mask, sig].values)
        summary_rows.append({
            'signal':       sig,
            'belly_mean':   float(df.loc[bd_mask, sig].mean()),
            'not_mean':     float(df.loc[nb_mask, sig].mean()),
            'cohens_d':     d,
            'interpret':    ('large'  if abs(d) >= 0.8 else
                              'medium' if abs(d) >= 0.5 else
                              'small'  if abs(d) >= 0.2 else
                              'tiny'),
        })
    sum_df = pd.DataFrame(summary_rows).sort_values(
        by='cohens_d', key=lambda s: s.abs(), ascending=False)
    sum_df.to_csv(os.path.join(out_dir, 'separability.csv'), index=False)

    lines = [
        f"Belly-down separability check",
        f"  Session : {args.session}",
        f"  Bodypart: {bp}    ROI half: {half} px",
        f"  N : {len(df)} events ({n_bd} belly-down + {n_nb} not)",
        "",
        f"  {'Signal':<22} {'belly':>10} {'not':>10} "
        f"{'Cohens d':>11}  size",
        "  " + "-" * 64,
    ]
    for r in sum_df.to_dict('records'):
        lines.append(
            f"  {r['signal']:<22} "
            f"{r['belly_mean']:>10.3f} "
            f"{r['not_mean']:>10.3f} "
            f"{r['cohens_d']:>11.2f}  {r['interpret']}")
    lines += [
        "",
        "  Cohen's d guide: |d| >= 0.8 large, 0.5-0.8 medium, "
        "0.2-0.5 small.",
        "  Signals with |d| >= 0.8 are candidates for threshold-based",
        "  detection. If no signal reaches that bar, a multi-feature",
        "  classifier (Train tab) is the better path.",
    ]
    summary = "\n".join(lines)
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
