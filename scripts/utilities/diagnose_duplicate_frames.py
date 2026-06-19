"""
diagnose_duplicate_frames.py — detect duplicated-frame stutter in videos.

Some PixelPaws captures were saved at a stored fps (e.g. 60) while the
sensor / pipeline was actually delivering unique frames at a lower (and
sometimes variable) rate, with each frame held for multiple ticks.
Classifiers trained on this content learn a dt=1 velocity that is ~0
every other frame; running them on true-rate video then mis-scales
motion features.

THRESHOLD CALIBRATION (important — read before trusting the output)
-------------------------------------------------------------------
Frame-to-frame mean-abs-pixel-difference splits into three regimes
in a behavior video:

    diff ≈ 0          — true held / duplicate frame (after re-encode)
    diff ≈ 0.05–0.5   — STILL animal: sensor + encoder noise floor
    diff > ~0.5       — animal in motion

A naive threshold around 0.5 will incorrectly count "still mouse"
frames as duplicates. The default eps is therefore 0.05 (above
re-encode noise, below idle-frame noise) — but this varies with
sensor + encoder + bit-rate, so each run prints the diff distribution
(p10/p25/p50/p75/p90) and a "bit-exact duplicates" floor count so the
threshold can be sanity-checked from the same output that uses it.

This script reports:

  - duplicate fraction at chosen eps
  - bit-exact (diff < 1e-6) and near-exact (diff < 0.01) duplicate counts
  - frame-diff distribution (percentiles)
  - run-length histogram of duplicate streaks
  - inferred true fps  =  stored_fps × unique_fraction
  - cv2.CAP_PROP_FPS for cross-check
  - bimodality score (high = clear duplicate signal, low = ambiguous)

Read-only. Writes a JSON report under <project>/diagnostics/ when a
project root is supplied; otherwise prints to stdout only.

Usage
-----
  python diagnose_duplicate_frames.py VIDEO [VIDEO ...]
  python diagnose_duplicate_frames.py --project PROJECT_DIR
  python diagnose_duplicate_frames.py --project PROJECT_DIR --eps 0.05

Exit codes: 0 = ran cleanly (regardless of findings), 1 = error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


# Default: above re-encode noise (~0.01) but below idle-frame noise
# (~0.1+ for live behavior video). See module docstring for calibration.
DUP_EPS_DEFAULT = 0.05
# Below this we're certain the frame is a re-encode of a duplicate, not just still.
BIT_EXACT_EPS = 1e-6
NEAR_EXACT_EPS = 0.01
DOWNSCALE_DEFAULT = 4  # speed: compare frames at 1/4 resolution


def _streak_lengths(is_dup: np.ndarray) -> List[int]:
    """Return run-lengths of *True* streaks in a bool array."""
    if is_dup.size == 0:
        return []
    out: List[int] = []
    run = 0
    for v in is_dup:
        if v:
            run += 1
        elif run:
            out.append(run)
            run = 0
    if run:
        out.append(run)
    return out


def diagnose_video(
    video_path: Path,
    eps: float = DUP_EPS_DEFAULT,
    downscale: int = DOWNSCALE_DEFAULT,
    max_frames: Optional[int] = None,
) -> Dict:
    """Scan one video, return a dict with stutter stats."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            'video': str(video_path),
            'error': 'could not open',
        }

    stored_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    n_pairs = 0
    n_dup = 0
    n_bit_exact = 0
    n_near_exact = 0
    is_dup_stream: List[bool] = []
    diffs_all: List[float] = []  # keep all diffs for distribution + bimodality

    ok, prev = cap.read()
    if not ok:
        cap.release()
        return {
            'video': str(video_path),
            'error': 'no frames',
            'stored_fps': stored_fps,
        }

    if downscale > 1:
        prev = cv2.resize(prev, (prev.shape[1] // downscale, prev.shape[0] // downscale),
                          interpolation=cv2.INTER_AREA)
    prev_grey = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY).astype(np.int16)

    t0 = time.time()
    frame_idx = 1
    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break
        ok, cur = cap.read()
        if not ok:
            break
        if downscale > 1:
            cur = cv2.resize(cur, (prev.shape[1], prev.shape[0]),
                             interpolation=cv2.INTER_AREA)
        cur_grey = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY).astype(np.int16)
        diff = float(np.mean(np.abs(cur_grey - prev_grey)))
        n_pairs += 1
        is_dup = diff < eps
        if is_dup:
            n_dup += 1
        if diff < BIT_EXACT_EPS:
            n_bit_exact += 1
        if diff < NEAR_EXACT_EPS:
            n_near_exact += 1
        is_dup_stream.append(is_dup)
        diffs_all.append(diff)
        prev_grey = cur_grey
        frame_idx += 1

    cap.release()
    elapsed = time.time() - t0

    if n_pairs == 0:
        return {
            'video': str(video_path),
            'error': 'only one frame',
            'stored_fps': stored_fps,
        }

    is_dup_arr = np.asarray(is_dup_stream, dtype=bool)
    diffs_arr = np.asarray(diffs_all, dtype=float)
    dup_fraction = n_dup / n_pairs
    unique_fraction = 1.0 - dup_fraction
    streaks = _streak_lengths(is_dup_arr)
    streak_hist = dict(Counter(streaks))

    inferred_true_fps = stored_fps * unique_fraction if stored_fps > 0 else None

    # Bimodality: a real "duplicated frames" video shows two clusters of
    # diffs — a tight cluster near 0 (held frames) and a wider one above
    # the noise floor (motion frames). Approximate by comparing the gap
    # between p25 and p75 to the absolute p25. If p25 << p75, there's a
    # heavy zero-cluster + a separate motion cluster (suspicious).
    # If p25 ≈ p75, the distribution is unimodal noise-floor + motion
    # gradient (normal).
    p25, p50, p75 = (float(np.percentile(diffs_arr, q)) for q in (25, 50, 75))
    p10, p90 = float(np.percentile(diffs_arr, 10)), float(np.percentile(diffs_arr, 90))
    near_exact_fraction = n_near_exact / n_pairs
    bit_exact_fraction = n_bit_exact / n_pairs

    # Pattern classification — only flag as duplicated when:
    #   (a) the *near-exact* (re-encode-noise-tolerant) fraction is
    #       substantial, not just the loose-eps fraction; AND
    #   (b) the diff distribution is bimodal (held frames cluster near 0
    #       distinctly below the motion cluster).
    # Otherwise we trust the file even if the loose dup_fraction is high.
    pattern = 'unique'
    if near_exact_fraction > 0.05:
        # Bimodal if there's a clear gap: p25 must be << p75 by an order
        # of magnitude and at least 0.1 absolute.
        bimodal = (p75 - p25) > max(0.1, 5 * p25)
        if bimodal:
            if 0.40 < near_exact_fraction < 0.60 and streaks and abs(np.median(streaks) - 1) < 0.5:
                pattern = 'uniform_2x'
            else:
                pattern = 'variable_stutter'
        else:
            pattern = 'ambiguous_high_dup_low_bimodality'

    return {
        'video': str(video_path),
        'stored_fps': stored_fps,
        'total_frames_meta': total_frames_meta,
        'pairs_scanned': n_pairs,
        'duplicate_pairs_at_eps': n_dup,
        'duplicate_fraction_at_eps': round(dup_fraction, 4),
        'bit_exact_duplicate_pairs': n_bit_exact,
        'bit_exact_fraction': round(bit_exact_fraction, 4),
        'near_exact_duplicate_pairs': n_near_exact,
        'near_exact_fraction': round(near_exact_fraction, 4),
        'unique_fraction_at_eps': round(unique_fraction, 4),
        'inferred_true_fps_at_eps': round(inferred_true_fps, 3) if inferred_true_fps is not None else None,
        'streak_length_histogram': {int(k): int(v) for k, v in streak_hist.items()},
        'streak_median': float(np.median(streaks)) if streaks else 0.0,
        'streak_max': int(max(streaks)) if streaks else 0,
        'pattern': pattern,
        'eps': eps,
        'downscale': downscale,
        'elapsed_sec': round(elapsed, 2),
        'diff_p10': round(p10, 4),
        'diff_p25': round(p25, 4),
        'diff_p50': round(p50, 4),
        'diff_p75': round(p75, 4),
        'diff_p90': round(p90, 4),
        'diff_max': round(float(diffs_arr.max()), 4),
    }


def _gather_project_videos(project_dir: Path) -> List[Path]:
    videos_dir = project_dir / 'videos'
    if not videos_dir.is_dir():
        return []
    seen: Dict[str, Path] = {}
    # Windows is case-insensitive; iterate once and dedupe by resolved path.
    for ext in ('.mp4', '.avi', '.mov'):
        for p in videos_dir.glob(f'*{ext}'):
            key = str(p.resolve()).lower()
            seen.setdefault(key, p)
    return sorted(seen.values())


def _print_report(rec: Dict) -> None:
    if 'error' in rec:
        print(f"  ! {rec['video']}  -- {rec['error']}")
        return
    print(f"  - {os.path.basename(rec['video'])}")
    inf = rec.get('inferred_true_fps_at_eps')
    inf_str = f'{inf:.2f}' if inf is not None else 'n/a'
    print(f"      stored_fps = {rec['stored_fps']:.2f}    "
          f"pattern = {rec['pattern']}")
    print(f"      bit_exact = {rec['bit_exact_fraction']:.4f} ({rec['bit_exact_duplicate_pairs']})    "
          f"near_exact = {rec['near_exact_fraction']:.4f} ({rec['near_exact_duplicate_pairs']})")
    print(f"      eps={rec['eps']} -> dup_frac = {rec['duplicate_fraction_at_eps']:.3f}    "
          f"inferred_true_fps_at_eps = {inf_str}")
    print(f"      diff p10/p25/p50/p75/p90 = "
          f"{rec['diff_p10']:.3f}/{rec['diff_p25']:.3f}/{rec['diff_p50']:.3f}/"
          f"{rec['diff_p75']:.3f}/{rec['diff_p90']:.3f}    "
          f"max = {rec['diff_max']:.2f}")
    if rec['pattern'] != 'unique':
        print(f"      streak histogram (length: count) = {rec['streak_length_histogram']}")
    print(f"      scanned {rec['pairs_scanned']} pairs in {rec['elapsed_sec']}s")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('videos', nargs='*', help='Video file paths')
    p.add_argument('--project', help='PixelPaws project root '
                                     '(scans <project>/videos/ and writes report '
                                     'under <project>/diagnostics/)')
    p.add_argument('--eps', type=float, default=DUP_EPS_DEFAULT,
                   help=f'Duplicate threshold (mean |Δ| in 8-bit). '
                        f'Default {DUP_EPS_DEFAULT}.')
    p.add_argument('--downscale', type=int, default=DOWNSCALE_DEFAULT,
                   help=f'Downscale factor for speed. Default {DOWNSCALE_DEFAULT}.')
    p.add_argument('--max-frames', type=int, default=None,
                   help='Cap frames per video (full scan if omitted).')
    args = p.parse_args(argv)

    videos: List[Path] = [Path(v) for v in args.videos]
    project_dir: Optional[Path] = None
    if args.project:
        project_dir = Path(args.project)
        videos.extend(_gather_project_videos(project_dir))
    videos = [v for v in videos if v.is_file()]
    if not videos:
        print('No videos found. Pass paths or --project <dir>.', file=sys.stderr)
        return 1

    print('=' * 68)
    print('DUPLICATE-FRAME DIAGNOSTIC')
    print('=' * 68)
    print(f'Videos: {len(videos)}    eps={args.eps}    downscale={args.downscale}')
    if args.max_frames:
        print(f'Max frames per video: {args.max_frames}')
    print()

    records: List[Dict] = []
    for v in videos:
        rec = diagnose_video(v, eps=args.eps, downscale=args.downscale,
                             max_frames=args.max_frames)
        records.append(rec)
        _print_report(rec)
        print()

    # Aggregate
    valid = [r for r in records if 'error' not in r]
    if valid:
        patterns = Counter(r['pattern'] for r in valid)
        median_near = float(np.median([r['near_exact_fraction'] for r in valid]))
        median_bit = float(np.median([r['bit_exact_fraction'] for r in valid]))
        print('-' * 68)
        print(f'Summary across {len(valid)} videos:')
        print(f'  pattern counts: {dict(patterns)}')
        print(f'  median bit_exact_fraction:  {median_bit:.4f}')
        print(f'  median near_exact_fraction: {median_near:.4f}')
        print('  (near_exact uses diff < 0.01; only this (or bit_exact) reliably')
        print('   indicates a held/duplicated frame. eps-based dup_fraction')
        print('   above can include still-but-unique frames.)')
        print('-' * 68)

    if project_dir is not None:
        out_dir = project_dir / 'diagnostics'
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = out_dir / f'dup_frames_{ts}.json'
        out_path.write_text(json.dumps({
            'generated_at': ts,
            'project': str(project_dir),
            'eps': args.eps,
            'downscale': args.downscale,
            'max_frames': args.max_frames,
            'videos': records,
        }, indent=2), encoding='utf-8')
        print(f'Report written: {out_path}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
