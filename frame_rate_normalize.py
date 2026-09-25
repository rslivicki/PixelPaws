"""
frame_rate_normalize.py — frame-rate cleanup for projects with
duplicated/held-frame videos.

Background
----------
Some PixelPaws captures stored videos at e.g. 60 fps while the sensor
delivered unique frames at a lower (and sometimes variable) rate, with
each frame held for one or more ticks. Classifiers trained on this
content learn motion features whose `dt=1` velocity is ~0 every other
frame; new true-rate captures then misalign with those features.

This module trims duplicate frames out of a video, then re-mirrors the
trim into the matching DLC `.h5`, the per-frame label CSV, and the
sparse-DB / dense-region label files that older labeling sessions left. It
backs up originals before overwriting and is dry-run by default.

Use via Tools > Normalize Project Frame Rate, or programmatically:

    from frame_rate_normalize import normalize_project
    report = normalize_project('/path/to/project',
                               target_fps=None,   # None = per-video content rate
                               dry_run=True)

The diagnostic at `scripts/utilities/diagnose_duplicate_frames.py`
should be run first — its output (or the same content-diff scan
implemented here) decides which frames to keep per video.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from io_utils import atomic_dataframe_to_csv

# Same defaults as diagnose_duplicate_frames.py — keep in sync.
DUP_EPS_DEFAULT = 0.5
DOWNSCALE_DEFAULT = 4

# An explicit target fps further than this from a video's content rate
# (stored fps x unique-frame fraction) changes the output's duration.
FPS_MISMATCH_TOL = 0.05


def content_fps(stored_fps: float, duplicate_fraction: float) -> Optional[float]:
    """Rate of a video's unique frames: round(stored_fps * (1 - dup_frac)).

    Writing the kept frames at this rate keeps the video's duration (60 fps
    with 33% duplicates written at 30 fps would run 1.33x long). None when
    the container reports no fps.
    """
    if not stored_fps or stored_fps <= 0:
        return None
    return float(max(1, round(stored_fps * (1.0 - float(duplicate_fraction)))))


# ============================================================================
# Frame-keep computation
# ============================================================================

def compute_keep_indices(
    video_path: Path,
    eps: float = DUP_EPS_DEFAULT,
    downscale: int = DOWNSCALE_DEFAULT,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[np.ndarray, Dict]:
    """Scan a video and return the indices of *unique* (non-duplicate) frames.

    Frame 0 is always kept. Frame i (i >= 1) is kept iff its mean abs
    pixel difference (greyscale, optionally downscaled) versus frame i-1
    is >= ``eps``. The result is the kept indices into the *original*
    video's frame stream.

    Returns
    -------
    keep_indices : np.ndarray, shape (M,) int
    stats : dict — for logging / sidecar JSON
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'cv2 could not open {video_path}')
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stored_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    keep: List[int] = []
    ok, prev = cap.read()
    if not ok:
        cap.release()
        return np.empty(0, dtype=np.int64), {'video': str(video_path), 'error': 'no frames'}
    keep.append(0)
    if downscale > 1:
        prev = cv2.resize(prev, (prev.shape[1] // downscale, prev.shape[0] // downscale),
                          interpolation=cv2.INTER_AREA)
    prev_grey = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY).astype(np.int16)

    idx = 1
    n_dup = 0
    while True:
        ok, cur = cap.read()
        if not ok:
            break
        if downscale > 1:
            cur = cv2.resize(cur, (prev_grey.shape[1], prev_grey.shape[0]),
                             interpolation=cv2.INTER_AREA)
        cur_grey = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY).astype(np.int16)
        diff = float(np.mean(np.abs(cur_grey - prev_grey)))
        if diff >= eps:
            keep.append(idx)
            prev_grey = cur_grey
        else:
            n_dup += 1
            # NB: we keep *prev_grey* unchanged so a long held streak
            # is collapsed against the originator, not the most recent
            # near-equal frame.
        idx += 1
        if progress and (idx % 200 == 0):
            progress(idx, n_total)

    cap.release()
    keep_arr = np.asarray(keep, dtype=np.int64)
    stats = {
        'video': str(video_path),
        'n_total': idx,
        'n_kept': len(keep_arr),
        'n_dropped': int(idx - len(keep_arr)),
        'duplicate_fraction': round(n_dup / max(1, idx - 1), 4),
        'stored_fps': stored_fps,
        'eps': eps,
        'downscale': downscale,
    }
    return keep_arr, stats


def build_inverse_map(keep_indices: np.ndarray, n_total: int) -> np.ndarray:
    """Map old frame index -> new frame index.

    For dropped frames, the value points to the *previous* kept frame
    (since a duplicate was held over from there). The first kept index
    is mapped to itself (slot 0). Returns an int array of length
    ``n_total``; values >= 0.
    """
    inv = np.full(n_total, -1, dtype=np.int64)
    inv[keep_indices] = np.arange(len(keep_indices))
    # Fill dropped frames with the previous kept slot
    last = -1
    for i in range(n_total):
        if inv[i] == -1:
            inv[i] = last
        else:
            last = int(inv[i])
    if (inv < 0).any():
        # Shouldn't happen if frame 0 was kept; clamp to 0.
        inv[inv < 0] = 0
    return inv


# ============================================================================
# Asset rewrites
# ============================================================================

def downsample_video(
    src_mp4: Path,
    dst_mp4: Path,
    keep_indices: np.ndarray,
    output_fps: float,
    fourcc: str = 'mp4v',
) -> Dict:
    """Re-encode `src_mp4` keeping only frames listed in `keep_indices`.

    Writes to `dst_mp4` at constant frame interval = 1/output_fps. Uses
    `cv2.VideoWriter` with the given fourcc (default 'mp4v', matching
    `render_skeleton_video.py`).
    """
    cap = cv2.VideoCapture(str(src_mp4))
    if not cap.isOpened():
        raise RuntimeError(f'cv2 could not open {src_mp4}')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst_mp4),
                             cv2.VideoWriter_fourcc(*fourcc),
                             output_fps,
                             (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f'cv2 could not open VideoWriter for {dst_mp4}')

    keep_set = set(int(k) for k in keep_indices)
    n_written = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in keep_set:
            writer.write(frame)
            n_written += 1
        idx += 1
    cap.release()
    writer.release()
    return {
        'src': str(src_mp4),
        'dst': str(dst_mp4),
        'n_total_in': idx,
        'n_written': n_written,
        'output_fps': output_fps,
    }


def remux_with_source_metadata(encoded: Path, original: Path, dst: Path) -> Tuple[bool, str]:
    """Stream-copy `encoded` into `dst` with `original`'s container tags.

    cv2.VideoWriter writes no custom tags, so a re-encode drops PawCapture's
    calibration (mm_per_pixel, pixelpaws_calibrated, pixelpaws_ref_*).
    Returns (ok, note); on failure `dst` is not written and the caller keeps
    the cv2 output.
    """
    if shutil.which('ffmpeg') is None:
        return False, 'cv2 only: ffmpeg not found, container tags (mm_per_pixel etc.) dropped'
    cmd = ['ffmpeg', '-y', '-v', 'error',
           '-i', str(encoded), '-i', str(original),
           '-map', '0', '-map_metadata', '1',
           '-movflags', 'use_metadata_tags',
           '-c', 'copy', str(dst)]
    try:
        r = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, text=True)
    except (OSError, subprocess.SubprocessError) as e:
        return False, f'cv2 only: ffmpeg remux could not run ({e}), container tags dropped'
    if r.returncode != 0 or not dst.is_file() or dst.stat().st_size == 0:
        try:
            dst.unlink()
        except OSError:
            pass
        err = (r.stderr or '').strip().splitlines()
        return False, ('cv2 only: ffmpeg remux failed'
                       + (f' ({err[-1]})' if err else '')
                       + ', container tags dropped')
    return True, 'ffmpeg remux: container tags copied from the original'


def write_normalized_video(
    src: Path,
    dst: Path,
    keep_indices: np.ndarray,
    output_fps: float,
) -> Dict:
    """`downsample_video` + `remux_with_source_metadata`: re-encode the kept
    frames, then carry the source's container tags over when ffmpeg is
    available (falling back to the bare cv2 output). The returned dict's
    'metadata' entry says which happened."""
    raw = dst.with_name(dst.stem + '.cv2' + dst.suffix)
    info = downsample_video(src, raw, keep_indices, output_fps=output_fps)
    ok, note = remux_with_source_metadata(raw, src, dst)
    if ok:
        raw.unlink()
    else:
        os.replace(str(raw), str(dst))
    info['dst'] = str(dst)
    info['metadata'] = note
    return info


def downsample_dlc_h5(
    src_h5: Path,
    dst_h5: Path,
    keep_indices: np.ndarray,
) -> Dict:
    """Select rows `keep_indices` from a DLC `.h5` and write to `dst_h5`.

    Preserves the MultiIndex columns DLC uses. Loads with `pd.read_hdf`
    (mirrors `pose_features.py:106`).
    """
    df = pd.read_hdf(src_h5)
    keep_indices = np.asarray(keep_indices, dtype=np.int64)
    if keep_indices.size and keep_indices.max() >= len(df):
        raise ValueError(
            f'keep_indices max {keep_indices.max()} >= h5 rows {len(df)} '
            f'({src_h5})'
        )
    out = df.iloc[keep_indices].reset_index(drop=True)
    dst_h5.parent.mkdir(parents=True, exist_ok=True)
    # DLC convention: key 'df_with_missing'. Try to preserve original key if any.
    try:
        # Detect original key
        with pd.HDFStore(str(src_h5), 'r') as store:
            keys = store.keys()
        key = keys[0].lstrip('/') if keys else 'df_with_missing'
    except Exception:
        key = 'df_with_missing'
    out.to_hdf(str(dst_h5), key=key, mode='w', format='fixed')
    return {
        'src': str(src_h5),
        'dst': str(dst_h5),
        'n_in': len(df),
        'n_out': len(out),
        'key': key,
    }


def remap_labels_csv(
    src_csv: Path,
    dst_csv: Path,
    keep_indices: np.ndarray,
    inverse_map: Optional[np.ndarray] = None,
    collapse: str = 'or',
) -> Dict:
    """Remap a per-frame label CSV (rows positional, row N = video frame N).

    For each *kept* frame K, the new row absorbs labels from all old
    frames that map to K (i.e. all old frames in the duplicate streak
    ending at K). With ``collapse='or'`` (default), binary labels are
    OR'd across the streak (NaN ignored unless all are NaN). With
    ``collapse='last_wins'``, only the kept frame's value is used.

    All non-frame columns (every column whose dtype is numeric or bool)
    are processed identically. A dedicated 'frame' / 'Frame' index
    column, if present, is rewritten with the new row indices.
    """
    df = pd.read_csv(src_csv)
    n_old = len(df)
    keep_indices = np.asarray(keep_indices, dtype=np.int64)
    if keep_indices.size == 0:
        out = df.iloc[:0].copy()
    else:
        if inverse_map is None:
            inverse_map = build_inverse_map(keep_indices, n_old)
        # Map every old row to a new-row index; unknown rows beyond the
        # video length get clamped to the last new-row.
        if len(inverse_map) < n_old:
            # CSV has more rows than video — pad inverse_map by repeating last
            pad = np.full(n_old - len(inverse_map),
                          inverse_map[-1] if len(inverse_map) else 0,
                          dtype=np.int64)
            inverse_map = np.concatenate([inverse_map, pad])
        elif len(inverse_map) > n_old:
            inverse_map = inverse_map[:n_old]

        n_new = len(keep_indices)

        # Identify "frame index" columns (case-insensitive 'frame')
        frame_cols = [c for c in df.columns if c.lower() == 'frame']
        data_cols = [c for c in df.columns if c.lower() != 'frame']

        new_rows: Dict[str, np.ndarray] = {}
        for fc in frame_cols:
            new_rows[fc] = np.arange(n_new, dtype=np.int64)

        for col in data_cols:
            ser = df[col]
            arr = ser.to_numpy()
            try:
                farr = arr.astype(float)
            except (TypeError, ValueError):
                # Non-numeric — fall back to last_wins (string-safe).
                kept = arr[keep_indices] if collapse == 'last_wins' else arr[keep_indices]
                new_rows[col] = kept
                continue

            if collapse == 'or':
                # Group old rows by their target new-row index, then take
                # the nan-aware max (binary OR for {0,1,nan}).
                acc = np.full(n_new, np.nan, dtype=float)
                # np.maximum.at with nans needs custom handling — use a
                # group-by via np.bincount on non-nan rows:
                isval = ~np.isnan(farr)
                if isval.any():
                    groups = inverse_map[isval]
                    vals = farr[isval]
                    # np.maximum.at: accumulator stays at -inf for unseen,
                    # so seed with -inf and post-mask.
                    seed = np.full(n_new, -np.inf, dtype=float)
                    np.maximum.at(seed, groups, vals)
                    seen = np.zeros(n_new, dtype=bool)
                    seen[groups] = True
                    acc = np.where(seen, seed, np.nan)
                new_rows[col] = acc
            elif collapse == 'last_wins':
                new_rows[col] = farr[keep_indices]
            else:
                raise ValueError(f'Unknown collapse mode: {collapse!r}')

        out = pd.DataFrame(new_rows)
        # Preserve original column order
        out = out[[c for c in df.columns]]

    atomic_dataframe_to_csv(out, str(dst_csv), index=False)
    return {
        'src': str(src_csv),
        'dst': str(dst_csv),
        'n_in': n_old,
        'n_out': len(out),
        'collapse': collapse,
    }


def remap_sparse_db(
    src_csv: Path,
    dst_csv: Path,
    inverse_map: np.ndarray,
    keep_indices: np.ndarray,
    collapse: str = 'or',
) -> Dict:
    """Remap `<video>_labels.db.csv` (sparse label database).

    Schema: frame_index, label, source, timestamp,
    confidence. Frames whose old index maps to a slot also occupied by
    a kept frame's prior label are OR'd (when collapse='or').
    """
    df = pd.read_csv(src_csv)
    if 'frame_index' not in df.columns:
        # Not the expected schema; pass through untouched.
        atomic_dataframe_to_csv(df, str(dst_csv), index=False)
        return {'src': str(src_csv), 'dst': str(dst_csv),
                'n_in': len(df), 'n_out': len(df),
                'note': 'no frame_index column — passed through'}

    n_inv = len(inverse_map)
    keep_set = set(int(k) for k in keep_indices)

    new_indices: List[int] = []
    for fi in df['frame_index'].astype(int):
        if 0 <= fi < n_inv:
            new_indices.append(int(inverse_map[fi]))
        else:
            new_indices.append(-1)
    df = df.assign(frame_index=new_indices)
    df = df[df['frame_index'] >= 0]

    if collapse == 'or':
        # Drop duplicates on (frame_index), keep the max label
        if 'label' in df.columns:
            df = (df.sort_values('label', ascending=False, kind='stable')
                    .drop_duplicates(subset=['frame_index'], keep='first')
                    .sort_values('frame_index')
                    .reset_index(drop=True))
        else:
            df = df.drop_duplicates(subset=['frame_index']).reset_index(drop=True)
    elif collapse == 'last_wins':
        # Keep only rows whose old index was itself kept
        df = df.reset_index(drop=True)
        # Filter by membership of original frame in keep_set was lost above;
        # for last_wins, recompute from the original CSV.
        orig = pd.read_csv(src_csv)
        if 'frame_index' in orig.columns:
            mask = orig['frame_index'].astype(int).isin(keep_set)
            df = (orig[mask]
                  .assign(frame_index=lambda x: [int(inverse_map[i])
                                                 for i in x['frame_index'].astype(int)])
                  .reset_index(drop=True))

    atomic_dataframe_to_csv(df, str(dst_csv), index=False)
    return {
        'src': str(src_csv),
        'dst': str(dst_csv),
        'n_out': len(df),
        'collapse': collapse,
    }


def remap_dense_regions(
    src_json: Path,
    dst_json: Path,
    inverse_map: np.ndarray,
) -> Dict:
    """Remap `<video>_metadata.json` dense_regions list.

    Each region is `{'start': int, 'end': int}`.
    `(start, end)` -> `(inverse_map[start], inverse_map[end])`. Empty
    regions (start > end after remap) are dropped.
    """
    with open(src_json, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    regions = meta.get('dense_regions', [])
    n_inv = len(inverse_map)

    new_regions = []
    for r in regions:
        try:
            s = int(r.get('start', -1))
            e = int(r.get('end', -1))
        except (TypeError, ValueError):
            continue
        if s < 0 or e < 0:
            continue
        s = int(inverse_map[min(s, n_inv - 1)])
        e = int(inverse_map[min(e, n_inv - 1)])
        if s > e:
            continue
        new_regions.append({'start': s, 'end': e})

    meta['dense_regions'] = new_regions
    if 'dense_region_count' in meta:
        meta['dense_region_count'] = len(new_regions)

    dst_json.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_json, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    return {
        'src': str(src_json),
        'dst': str(dst_json),
        'n_in': len(regions),
        'n_out': len(new_regions),
    }


# ============================================================================
# Project orchestrator
# ============================================================================

@dataclass
class VideoNormalizePlan:
    video: str
    dlc_h5: Optional[str] = None
    labels_csv: Optional[str] = None
    sparse_db_csv: Optional[str] = None
    dense_meta_json: Optional[str] = None
    n_frames_in: int = 0
    n_frames_out: int = 0
    duplicate_fraction: float = 0.0
    stored_fps: float = 0.0
    content_fps: Optional[float] = None    # round(stored_fps * (1 - dup_frac))
    output_fps: float = 0.0                # what the video is (or would be) written at
    fps_warning: Optional[str] = None      # explicit target_fps off content_fps by > 5%
    video_metadata: Optional[str] = None   # ffmpeg remux vs cv2-only (tags dropped)
    status: str = 'planned'                # planned | applied | failed
    error: Optional[str] = None
    actions: List[str] = field(default_factory=list)


@dataclass
class NormalizeReport:
    project: str
    target_fps: Optional[float]            # None = each video at its content_fps
    eps: float
    dry_run: bool
    backup_dir: str
    plans: List[VideoNormalizePlan] = field(default_factory=list)
    cache_dir_invalidated: Optional[str] = None
    process_fps_written: Optional[float] = None
    elapsed_sec: float = 0.0


def _find_session_assets(video_path: Path, project_dir: Path) -> Dict[str, Optional[Path]]:
    """Locate DLC h5 + labels + sparse DB + dense meta for one video.

    Mirrors the resolution logic in `evaluation_tab.py:380`
    (`find_session_triplets`). Kept lightweight to avoid an import cycle
    — when behavior changes, sync with that function.
    """
    base = video_path.stem
    video_dir = video_path.parent

    # DLC h5 — prefer filtered (DLC convention)
    dlc_h5 = None
    for cand in (
        video_dir / f'{base}_filtered.h5',
        video_dir / f'{base}.h5',
    ):
        if cand.is_file():
            dlc_h5 = cand
            break
    # Glob fallback for DLC's <stem>DLC_*.h5 naming. Nothing may sit between
    # the stem and "DLC": the old `{base}*DLC*` let S1 pick up S10's pose file
    # ('0' sorts before 'D'), so a wet run trimmed and overwrote the wrong h5.
    # Active network's file first (h5_resolve), filtered preferred.
    if dlc_h5 is None:
        cands = [str(p) for p in video_dir.glob(f'{glob.escape(base)}DLC*.h5')]
        try:
            from h5_resolve import rank_h5
            cands = rank_h5(cands)
        except Exception:
            cands = sorted(cands)
        filtered = [c for c in cands if 'filtered' in Path(c).name.lower()]
        if filtered or cands:
            dlc_h5 = Path((filtered or cands)[0])

    # Labels CSV — search order mirrors evaluation_tab.py:380 find_session_triplets
    labels_csv = None
    candidate_dirs = [
        project_dir / 'behavior_labels',
        video_dir,
        project_dir / 'labels',
        project_dir / 'Labels',
        project_dir / 'targets',
        project_dir / 'Targets',
    ]
    for d in candidate_dirs:
        for cand in (d / f'{base}_labels.csv',
                     d / f'{base}_Labels.csv',
                     d / f'{base}.csv'):
            if cand.is_file():
                labels_csv = cand
                break
        if labels_csv:
            break

    # Sparse DB / dense metadata
    sparse_db = None
    dense_meta = None
    for cand_dir in candidate_dirs:
        c1 = cand_dir / f'{base}_labels.db.csv'
        c2 = cand_dir / f'{base}_metadata.json'
        if sparse_db is None and c1.is_file():
            sparse_db = c1
        if dense_meta is None and c2.is_file():
            dense_meta = c2

    return {
        'dlc_h5': dlc_h5,
        'labels_csv': labels_csv,
        'sparse_db': sparse_db,
        'dense_meta': dense_meta,
    }


def _backup_in_place(src: Path, backup_dir: Path) -> Path:
    """Move `src` into `backup_dir` (flat); returns the backup path. A name
    already taken there gets its parent folder prefixed instead of being
    overwritten."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    dst = backup_dir / src.name
    n = 0
    while dst.exists():
        n += 1
        dst = backup_dir / f'{src.parent.name}__{n}__{src.name}'
    shutil.move(str(src), str(dst))
    return dst


def _swap_in(pairs: List[Tuple[Path, Path]], backup_dir: Path) -> None:
    """Back up each original and move its staged replacement into place.

    All or nothing: if a move fails (e.g. the video is open elsewhere on
    Windows), the originals already swapped are restored before the error
    propagates, so the video is left as it was and a rerun can redo it.
    """
    done: List[Tuple[Path, Path]] = []
    try:
        for orig, staged in pairs:
            bak = _backup_in_place(orig, backup_dir)
            try:
                shutil.move(str(staged), str(orig))
            except Exception:
                shutil.move(str(bak), str(orig))
                raise
            done.append((orig, bak))
    except Exception:
        for orig, bak in reversed(done):
            try:
                os.replace(str(bak), str(orig))
            except OSError:
                pass                        # the backup stays in backup_dir
        raise


def _plan_video(
    v: Path,
    project_dir: Path,
    target_fps: Optional[float],
    eps: float,
    downscale: int,
    ffmpeg_ok: bool,
    plan: VideoNormalizePlan,
) -> Tuple[np.ndarray, Dict[str, Optional[Path]]]:
    """Scan one video and fill `plan`; returns (keep_indices, assets)."""
    keep, stats = compute_keep_indices(v, eps=eps, downscale=downscale)
    if 'error' in stats:
        raise RuntimeError(stats['error'])

    plan.n_frames_in = int(stats['n_total'])
    plan.n_frames_out = int(stats['n_kept'])
    plan.duplicate_fraction = float(stats['duplicate_fraction'])
    plan.stored_fps = float(stats['stored_fps'])
    plan.content_fps = content_fps(plan.stored_fps, plan.duplicate_fraction)

    # Output rate: the content rate unless the caller overrode it. Any other
    # rate stretches or squeezes the video in time.
    if target_fps is not None:
        plan.output_fps = float(target_fps)
        auto = plan.content_fps
        if auto and abs(plan.output_fps - auto) / auto > FPS_MISMATCH_TOL:
            plan.fps_warning = (
                f'target {plan.output_fps:g} fps is '
                f'{abs(plan.output_fps - auto) / auto:.0%} off the content rate '
                f'{auto:g} fps (stored {plan.stored_fps:g} x '
                f'{1 - plan.duplicate_fraction:.3f} unique); the output runs '
                f'{auto / plan.output_fps:.2f}x the original duration')
    elif plan.content_fps is None:
        plan.output_fps = 30.0
        plan.fps_warning = 'video reports no fps; writing at 30 fps (duration may change)'
    else:
        plan.output_fps = plan.content_fps

    plan.actions.append(
        f'video: keep {plan.n_frames_out}/{plan.n_frames_in} frames '
        f'(dup_frac={plan.duplicate_fraction:.3f}, '
        f'output @ {plan.output_fps:g} fps'
        f'{"" if target_fps is not None else " = content rate"})'
    )
    plan.video_metadata = ('ffmpeg remux planned: container tags kept' if ffmpeg_ok else
                           'cv2 only: ffmpeg not found, container tags (mm_per_pixel etc.) '
                           'would be dropped')

    assets = _find_session_assets(v, project_dir)
    plan.dlc_h5 = str(assets['dlc_h5']) if assets['dlc_h5'] else None
    plan.labels_csv = str(assets['labels_csv']) if assets['labels_csv'] else None
    plan.sparse_db_csv = str(assets['sparse_db']) if assets['sparse_db'] else None
    plan.dense_meta_json = str(assets['dense_meta']) if assets['dense_meta'] else None
    if plan.dlc_h5:
        plan.actions.append(f'dlc: select {plan.n_frames_out} rows from {Path(plan.dlc_h5).name}')
    if plan.labels_csv:
        plan.actions.append(f'labels: OR-collapse to {plan.n_frames_out} rows '
                            f'in {Path(plan.labels_csv).name}')
    if plan.sparse_db_csv:
        plan.actions.append(f'sparse_db: remap frame_index in {Path(plan.sparse_db_csv).name}')
    if plan.dense_meta_json:
        plan.actions.append(f'dense_meta: remap regions in {Path(plan.dense_meta_json).name}')
    return keep, assets


def _apply_video(
    v: Path,
    plan: VideoNormalizePlan,
    keep: np.ndarray,
    assets: Dict[str, Optional[Path]],
    project_dir: Path,
    backup_dir: Path,
) -> None:
    """Write every output for one video into a staging folder, then swap
    them all in. Nothing in the project changes unless every output was
    written; the staging folder is removed either way."""
    inverse_map = build_inverse_map(keep, plan.n_frames_in)
    stage = Path(tempfile.mkdtemp(prefix='_fps_normalize_stage_', dir=str(project_dir)))
    try:
        def staged(kind: str, src: Path) -> Path:
            d = stage / kind
            d.mkdir(parents=True, exist_ok=True)
            return d / src.name

        pairs: List[Tuple[Path, Path]] = []

        dst = staged('video', v)
        info = write_normalized_video(v, dst, keep, output_fps=plan.output_fps)
        plan.video_metadata = info['metadata']
        pairs.append((v, dst))

        if assets['dlc_h5']:
            dst = staged('dlc', assets['dlc_h5'])
            downsample_dlc_h5(assets['dlc_h5'], dst, keep)
            pairs.append((assets['dlc_h5'], dst))

        if assets['labels_csv']:
            dst = staged('labels', assets['labels_csv'])
            remap_labels_csv(assets['labels_csv'], dst, keep, inverse_map)
            pairs.append((assets['labels_csv'], dst))

        if assets['sparse_db']:
            dst = staged('sparse_db', assets['sparse_db'])
            remap_sparse_db(assets['sparse_db'], dst, inverse_map, keep)
            pairs.append((assets['sparse_db'], dst))

        if assets['dense_meta']:
            dst = staged('dense_meta', assets['dense_meta'])
            remap_dense_regions(assets['dense_meta'], dst, inverse_map)
            pairs.append((assets['dense_meta'], dst))

        _swap_in(pairs, backup_dir)
    finally:
        shutil.rmtree(str(stage), ignore_errors=True)


def normalize_project(
    project_dir: str,
    target_fps: Optional[float] = None,
    eps: float = DUP_EPS_DEFAULT,
    downscale: int = DOWNSCALE_DEFAULT,
    dry_run: bool = True,
    invalidate_features: bool = True,
    progress: Optional[Callable[[str], None]] = None,
) -> NormalizeReport:
    """Trim duplicate frames from every video in `<project>/videos/` and
    re-mirror the trim into DLC h5, label CSV, sparse DB, dense regions.

    Each video is written at its content rate, round(stored_fps x
    (1 - duplicate_fraction)), so its duration is unchanged; pass
    `target_fps` to force one rate for all (videos it is more than 5% off
    get `fps_warning` in the report).

    Originals are moved to `<project>/_pre_fps_normalize_<TS>/` before
    being overwritten. A video whose outputs fail to write is left
    untouched and marked failed; the rest still run, and the report JSON
    lists both. With `dry_run=True` (default) nothing is written — the
    report tells you what *would* change.
    """
    project_dir = Path(project_dir).resolve()
    if not project_dir.is_dir():
        raise NotADirectoryError(project_dir)

    videos_dir = project_dir / 'videos'
    if not videos_dir.is_dir():
        raise FileNotFoundError(videos_dir)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = project_dir / f'_pre_fps_normalize_{ts}'

    # Collect videos (case-insensitive on Windows)
    seen: Dict[str, Path] = {}
    for ext in ('.mp4', '.avi', '.mov'):
        for p in videos_dir.glob(f'*{ext}'):
            seen.setdefault(str(p.resolve()).lower(), p)
    videos = sorted(seen.values())
    if not videos:
        raise FileNotFoundError(f'No videos under {videos_dir}')

    log = (lambda m: progress(m)) if progress else (lambda m: print(m))

    report = NormalizeReport(
        project=str(project_dir),
        target_fps=float(target_fps) if target_fps is not None else None,
        eps=float(eps),
        dry_run=bool(dry_run),
        backup_dir=str(backup_dir),
    )
    ffmpeg_ok = shutil.which('ffmpeg') is not None

    t0 = time.time()
    for v in videos:
        log(f'[{v.name}] scanning for duplicates...')
        plan = VideoNormalizePlan(video=str(v))
        report.plans.append(plan)
        try:
            keep, assets = _plan_video(v, project_dir, target_fps, eps, downscale,
                                       ffmpeg_ok, plan)
            for a in plan.actions:
                log(f'  - {a}')
            if plan.fps_warning:
                log(f'  ! {plan.fps_warning}')
            if not ffmpeg_ok:
                log(f'  ! {plan.video_metadata}')
            if dry_run:
                continue
            # Wet run: stage every output, then back up + swap them together.
            _apply_video(v, plan, keep, assets, project_dir, backup_dir)
            plan.status = 'applied'
            log(f'  applied ({plan.video_metadata})')
        except Exception as e:
            plan.status = 'failed'
            plan.error = f'{type(e).__name__}: {e}'
            log(f'  ! failed, video left unchanged: {plan.error}')

    applied = [p for p in report.plans if p.status == 'applied']
    failed = [p for p in report.plans if p.status == 'failed']
    if not dry_run and applied:
        # Invalidate the project feature cache so post-normalize features
        # are extracted fresh.
        if invalidate_features:
            cache_dir = project_dir / 'features'
            if cache_dir.is_dir():
                try:
                    # Move (not delete) so the user can recover if they realise
                    # the normalize was a mistake.
                    shutil.move(str(cache_dir), str(backup_dir / 'features'))
                    report.cache_dir_invalidated = str(cache_dir)
                except Exception as e:
                    log(f'  ! could not move the feature cache aside: {e}')

        # Persist process_fps to project config: the rate most videos were
        # written at (they differ only with per-video content rates).
        fps_counts = Counter(p.output_fps for p in applied)
        proc_fps = float(fps_counts.most_common(1)[0][0])
        if len(fps_counts) > 1:
            log(f'  ! videos were written at different rates '
                f'({", ".join(f"{f:g}" for f in sorted(fps_counts))} fps); '
                f'process_fps set to the most common, {proc_fps:g}')
        try:
            from project_config import ProjectConfig
            cfg = ProjectConfig.load(str(project_dir))
            cfg.process_fps = proc_fps
            cfg.source_fps_note = (
                cfg.source_fps_note or
                f'normalized {ts}: trimmed duplicates @ eps={eps}'
            )
            cfg.save(str(project_dir))
            report.process_fps_written = proc_fps
        except Exception as e:
            log(f'  ! could not write process_fps to project config: {e}')

    if dry_run:
        log(f'{len(report.plans) - len(failed)} video(s) planned, {len(failed)} failed')
    else:
        log(f'{len(applied)} video(s) applied, {len(failed)} failed')
    report.elapsed_sec = round(time.time() - t0, 2)

    # Write a sidecar JSON of the report
    report_dir = project_dir / 'diagnostics'
    report_dir.mkdir(parents=True, exist_ok=True)
    suffix = 'plan' if dry_run else 'applied'
    out_path = report_dir / f'fps_normalize_{suffix}_{ts}.json'
    out_path.write_text(json.dumps(asdict(report), indent=2), encoding='utf-8')
    log(f'Report written: {out_path}')

    return report


# ============================================================================
# CLI
# ============================================================================

def _cli(argv: Optional[List[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('project', help='PixelPaws project root')
    p.add_argument('--target-fps', type=float, default=None,
                   help='Output fps for every video (default: each video\'s '
                        'content rate, stored fps x unique-frame fraction).')
    p.add_argument('--eps', type=float, default=DUP_EPS_DEFAULT)
    p.add_argument('--apply', action='store_true',
                   help='Actually rewrite files (default is dry-run).')
    args = p.parse_args(argv)

    normalize_project(
        args.project,
        target_fps=args.target_fps,
        eps=args.eps,
        dry_run=not args.apply,
    )
    return 0


if __name__ == '__main__':
    sys.exit(_cli())
