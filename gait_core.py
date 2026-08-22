"""
gait_core.py — headless Gait & Limb Use analysis compute
=========================================================

Validated compute extracted VERBATIM from ``gait_limb_tab.py`` (the GUI tab),
with only the plumbing changed:

  * ``self._log(msg)``            → ``log(msg)`` (no-op-defaulted callback)
  * progress / ``after(...)``     → optional ``progress_cb(done, total, text)``
  * ``self._cancel_flag``         → optional ``cancel`` (threading.Event)
  * reads of self/app state       → explicit parameters (see ``GaitContext``)

Zero tkinter imports. The GUI tab(s) are built on top of this module.

Architecture (the point of the extraction): ``analyze_session`` is three stages —

  1. ``load_session_data``    — DLC load, heights, fps, confidence/locomotion
                                masks, contact masks, brightness + contour
                                extraction (incl. byte-compatible caches),
                                lick mask.  Produces the ``data`` dict (the
                                same keys the old tab stashed per session in
                                ``_session_intermediates``).
  2. ``compute_selection_masks`` — licking-excluded base mask, 4-paw mask,
                                analyzed mask.
  3. ``compute_all_metrics``  — the ONE metrics implementation (the old
                                ``_metrics`` / ``_gait_block`` closures +
                                summary/bins assembly).  The Adjust-Contact
                                path (``recompute_with_contact``) rebuilds
                                contact masks from cached arrays, rebuilds
                                the selection masks (INCLUDING lick + 4-paw
                                — fixing the drifted ``_recompute_contact``
                                duplicate, which silently dropped them), and
                                calls this same function.

``params`` contract — the exact 33-key dict the old ``_start_analysis`` builds:
contact_threshold, height_window, bin_seconds, bin_unit, fallback_fps,
use_brightness, brt_threshold, brt_weight, roi_sizes, crop_offset_x,
crop_offset_y, extraction_stride, contact_method, speed_threshold,
median_filter_ms, min_bout_ms, min_stance_ms, use_likelihood,
likelihood_threshold, loco_filter, loco_threshold, paw_contour,
contour_roi_sizes, contour_forelimbs, contour_area_threshold,
contour_area_max, exclude_licking, lick_behavior, lick_threshold, gate_4paw.
(``speed_threshold`` is a float or the string 'auto'.)
"""

import os
import re
import pickle
import threading
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    from scipy.ndimage import median_filter as _median_filter
    _SCIPY_NDIMAGE_OK = True
except ImportError:
    _SCIPY_NDIMAGE_OK = False

try:
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False

from pose_features import PoseFeatureExtractor
from brightness_features import PixelBrightnessExtractorOptimized

try:
    from evaluation_tab import find_session_triplets
except ImportError as _fst_err:
    # If evaluation_tab fails to import, callers get an empty session
    # list. Pre-2026-05-01 this happened silently; now we log it once
    # at module load so the failure is at least traceable.
    print(f'[gait_core] WARNING: could not import '
          f'find_session_triplets from evaluation_tab: {_fst_err}. '
          f'Session scans will return an empty list.')
    def find_session_triplets(folder, **kw):
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Presets (verbatim copy of GaitLimbTab.GAIT_PRESETS — the "manuscript" preset
# is the deployed default profile)
# ─────────────────────────────────────────────────────────────────────────────

ROLES = ('HL', 'HR', 'FL', 'FR')

# Quick-setup presets — each maps to the interdependent toggle vars so the
# user never has to know that contour needs brightness needs video.
GAIT_PRESETS = {
    # Default profile — the manuscript's deployed paw-contact gate, as used for the
    # formalin/oxycodone dose analyses: contour ROI 50 half-size (100x100 px box),
    # Otsu per ROI per frame, contour-area band 1,500-5,000 px^2 on both hind paws,
    # licking excluded, 5-min bins. The DLC-likelihood frame filter is OFF: its
    # retention tracks the manipulation (69% at vehicle vs 23% at oxycodone 10 mg/kg),
    # so it cannot arbitrate a treatment effect; the area band catches mislocated
    # boxes instead. Sessions without pose files are still skipped upstream.
    'Paw contact (manuscript gate)': dict(
        brightness=True, contour=True, fore=False, contour_roi=50,
        likelihood=False, bin_seconds=5, bin_unit='minutes',
        contact_method='contour_area', contour_area_threshold=1500,
        contour_area_max=5000, exclude_licking=True),
    'Hindpaw gait (minimal)':        dict(brightness=False, contour=False, fore=False,
                                          contact_method='height'),
    'Gait + forepaws':               dict(brightness=False, contour=False, fore=True,
                                          contact_method='height'),
    'Gait + brightness':             dict(brightness=True,  contour=False, fore=False,
                                          contact_method='height'),
    'Gait + brightness + contour':   dict(brightness=True,  contour=True,  fore=False,
                                          contact_method='height'),
}

# Default paw-like contour filter thresholds (old tab: self._pawlike_thresholds)
DEFAULT_PAWLIKE_THRESHOLDS = {'solidity': 1.00, 'aspect_ratio': 1.6,
                              'circularity': 0.10}


def _default_pawlike():
    return dict(DEFAULT_PAWLIKE_THRESHOLDS)


@dataclass
class GaitContext:
    """Everything the compute path used to read off self/app.

    Fields
    ------
    project_folder : str
        The project root (old: ``app.current_project_folder.get()``).  Roots
        the cache dirs (``gait_limb_analysis/`` writes, legacy
        ``weight_bearing_analysis/`` read-compat), the ``results/`` folder for
        licking-behavior predictions, and ``PixelPaws_project.json`` for
        mm-per-pixel calibration.
    pawlike_thresholds : dict
        Paw-like contour filter thresholds — solidity / aspect_ratio /
        circularity (old: ``self._pawlike_thresholds``; editable via the
        Filter Preview dialog).
    """
    project_folder: str = ''
    pawlike_thresholds: dict = field(default_factory=_default_pawlike)


def _noop_log(msg):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Small pure helpers (verbatim moves)
# ─────────────────────────────────────────────────────────────────────────────

def robust_unpickle(path):
    """Load a .pkl that may be joblib+LZ4 OR plain pickle (see PixelPaws_GUI)."""
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with open(path, 'rb') as f:
            return pickle.load(f)


def extract_behavior_name(filename):
    """Behavior name from a prediction filename: strip the pred suffix, then
    take the tokens after 'PixelPaws' (mirrors the Transitions loader)."""
    base = os.path.basename(filename)
    for suf in ('_predictions.csv', '_prediction.csv', '_pred.csv'):
        if base.endswith(suf):
            base = base[:-len(suf)]
            break
    else:
        base = os.path.splitext(base)[0]
    if 'PixelPaws' in base:
        beh = base.split('PixelPaws', 1)[1].strip('_')
        if beh:
            return beh
    return base


def scan_lick_behaviors(folder, sessions=()):
    """Behavior names available as predictions in the project's results/ folder.
    Extracted from prediction filenames (robust to per-session subfolders),
    with a consolidated per_frame header fallback.

    ``folder`` is the PROJECT folder (results/ is appended, as in the old tab).
    ``sessions`` is the known session list (dicts with 'session_name', or
    plain names) used to drop session-named noise.
    """
    names = set()
    if not folder:
        return []
    results = os.path.join(folder, 'results')
    if not os.path.isdir(results):
        return []
    import glob as _glob
    sessions = {s['session_name'] if isinstance(s, dict) else str(s)
                for s in (sessions or [])}

    def _is_session_named(name):
        return any(name == sn or name.startswith(sn + '_') or sn in name
                   for sn in sessions)

    # Preferred: immediate behavior subfolders (results/{behavior}/…) that
    # actually hold prediction files — this is the canonical batch layout.
    for entry in os.listdir(results):
        p = os.path.join(results, entry)
        if not os.path.isdir(p) or entry.lower() == 'per_frame':
            continue
        if _is_session_named(entry):
            continue
        if _glob.glob(os.path.join(p, '*_predictions.csv')) or \
           _glob.glob(os.path.join(p, '**', '*_predictions.csv'), recursive=True):
            names.add(entry)
    # Fallback: parse behavior from filenames, dropping session-named noise.
    if not names:
        for path in _glob.glob(os.path.join(results, '**', '*_predictions.csv'),
                               recursive=True):
            beh = extract_behavior_name(path)
            if beh and not _is_session_named(beh):
                names.add(beh)
    # Last resort: consolidated per_frame headers.
    if not names:
        pf = os.path.join(results, 'per_frame')
        if os.path.isdir(pf):
            for fn in os.listdir(pf):
                if fn.endswith('_frames.csv'):
                    try:
                        cols = pd.read_csv(os.path.join(pf, fn), nrows=0).columns
                        for c in cols:
                            if c.endswith('_pred'):
                                names.add(c[:-5])
                    except Exception:
                        pass
                    break
    return sorted(names)


def load_lick_mask(folder, session_name, behavior, thr, n_frames):
    """Boolean array (len n_frames, True = licking) for a session, from the
    project's behavior predictions. Returns all-False when unavailable.

    ``folder`` is the PROJECT folder (results/ is appended, as in the old tab).
    """
    out = np.zeros(int(n_frames), dtype=bool)
    if not behavior:
        return out
    if not folder:
        return out
    results = os.path.join(folder, 'results')
    vals = None
    # 1) Consolidated per_frame sheet: {session}_frames.csv with {behavior}_pred.
    pf = os.path.join(results, 'per_frame', f'{session_name}_frames.csv')
    if os.path.isfile(pf):
        try:
            df = pd.read_csv(pf)
            pred_cols = [c for c in df.columns
                         if c == f'{behavior}_pred'
                         or (c.startswith(f'{behavior}_') and c.endswith('_pred'))]
            if pred_cols:
                vals = df[pred_cols[0]].values.astype(float)
            elif f'{behavior}_prob' in df.columns:
                vals = (df[f'{behavior}_prob'].values.astype(float) >= float(thr))
        except Exception:
            vals = None
    # 2) Per-behavior predictions CSV — match files for this session whose
    #    extracted behavior name equals the chosen behavior.
    if vals is None:
        import glob as _glob
        cands = _glob.glob(os.path.join(
            results, behavior, f'{session_name}*_predictions.csv'))
        cands += [p for p in _glob.glob(os.path.join(
            results, '**', f'{session_name}*_predictions.csv'), recursive=True)
            if extract_behavior_name(p) == behavior]
        for path in cands:
            try:
                df = pd.read_csv(path)
                if behavior in df.columns:
                    vals = df[behavior].values.astype(float)
                elif 'probability' in df.columns:
                    vals = (df['probability'].values.astype(float) >= float(thr))
                if vals is not None:
                    break
            except Exception:
                continue
    if vals is None:
        return out
    lick = np.asarray(vals).astype(bool)
    k = min(len(lick), len(out))
    out[:k] = lick[:k]
    return out


def compute_selection_masks(contact_masks, lick_mask, gate_4paw, n):
    """Return (analyzed_mask, base_mask, four_mask) as full-length bool arrays.

    base_mask  = licking-excluded frames (the metric denominator when not gating);
    four_mask  = frames with all four paws in contact (None if <4 paws present);
    analyzed_mask = base_mask & (four_mask when the 4-paw gate is on).
    """
    base = np.ones(int(n), dtype=bool)
    if lick_mask is not None:
        lm = np.asarray(lick_mask, dtype=bool)
        base[:len(lm)] &= ~lm[:n]
    four = None
    if all(r in contact_masks for r in ('HL', 'HR', 'FL', 'FR')):
        four = np.ones(int(n), dtype=bool)
        for r in ('HL', 'HR', 'FL', 'FR'):
            mk = contact_masks[r]
            arr = (mk.values.astype(bool) if hasattr(mk, 'values')
                   else np.asarray(mk, dtype=bool))
            four[:len(arr)] &= arr[:n]
    analyzed = base.copy()
    if gate_4paw and four is not None:
        analyzed &= four
    return analyzed, base, four


# ─────────────────────────────────────────────────────────────────────────────
# Key file handling (verbatim, minus messageboxes/UI refresh)
# ─────────────────────────────────────────────────────────────────────────────

def scan_key_files(folder):
    """Walk project folder for CSV/XLSX files with Subject+Treatment cols.
    Returns the list of candidate paths."""
    _SKIP    = {'__pycache__', '.git', '.claude', 'node_modules', '.idea'}
    _PRED_KW = ('prediction', 'predictions', 'pred', 'bout', 'bouts')
    candidates = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in sorted(dirs)
                   if d not in _SKIP and not d.startswith('.')]
        for fname in files:
            fl = fname.lower()
            if not fl.endswith(('.csv', '.xlsx')):
                continue
            if any(kw in fl for kw in _PRED_KW):
                continue
            full = os.path.join(root, fname)
            try:
                if full.endswith('.xlsx'):
                    cols = pd.read_excel(full, nrows=0).columns.tolist()
                else:
                    cols = pd.read_csv(full, nrows=0).columns.tolist()
                if 'Subject' in cols and 'Treatment' in cols:
                    candidates.append(full)
            except Exception:
                pass
    return candidates


def load_key_file(path):
    """Read a key file (CSV/XLSX). Raises ValueError when the required
    Subject/Treatment columns are missing (the old tab showed a messagebox)."""
    df = pd.read_excel(path) if path.endswith('.xlsx') else pd.read_csv(path)
    missing = [c for c in ('Subject', 'Treatment') if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    df['Subject'] = df['Subject'].astype(str)
    return df


_extract_sid = None
_extract_sid_loaded = False


def _get_extract_sid():
    """Lazy import of the PixelPaws_GUI legacy subject-ID helper (rung 3 of the
    resolution ladder). Lazy so importing gait_core stays light and headless."""
    global _extract_sid, _extract_sid_loaded
    if not _extract_sid_loaded:
        _extract_sid_loaded = True
        try:
            from PixelPaws_GUI import extract_subject_id_from_filename as _f
            _extract_sid = _f
        except Exception:
            _extract_sid = None
    return _extract_sid


def resolve_subject(session_name, key_df=None, strip_prefix=''):
    """Extract subject ID via 4-strategy fallback (mirrors analysis_tab)."""
    stem = session_name

    # 1. Key-file token match
    if key_df is not None:
        tokens = stem.split('_')
        for subj in key_df['Subject']:
            if str(subj) in tokens:
                return str(subj)
        for subj in key_df['Subject']:
            if f'_{subj}_' in f'_{stem}_':
                return str(subj)

    # 2. Prefix strip
    pfx = (strip_prefix or '').strip()
    if pfx and stem.startswith(pfx):
        remainder = stem[len(pfx):]
        token = remainder.split('_')[0] if remainder else ''
        if token:
            return token

    # 3. PixelPaws_GUI legacy helper
    _sid_fn = _get_extract_sid()
    if _sid_fn is not None:
        sid = _sid_fn(session_name)
        if sid:
            return str(sid)

    # 4. First 4-digit token heuristic
    for token in stem.split('_'):
        if re.match(r'^\d{4}$', token):
            return token

    return stem


def get_treatment(subject, key_df):
    if key_df is None:
        return ''
    row = key_df[key_df['Subject'] == str(subject)]
    return str(row.iloc[0]['Treatment']) if not row.empty else ''


# ─────────────────────────────────────────────────────────────────────────────
# Calibration + cache paths (verbatim, folder from ctx)
# ─────────────────────────────────────────────────────────────────────────────

def resolve_mm_per_pixel(ctx, session):
    """Look up the effective mm_per_pixel for this session, honouring
    the project's ``calibration_mode``. Returns None when calibration
    is off or unavailable so callers fall back to legacy pixel
    behaviour transparently.
    """
    try:
        from project_config import ProjectConfig
        cfg = ProjectConfig.load(ctx.project_folder)
        return cfg.resolve_mm_per_pixel(session)
    except Exception:
        return None


def brt_cache_path(ctx, session_name, cache_key):
    """Return path for brightness series cache CSV, or None if unavailable."""
    import hashlib, json
    folder = ctx.project_folder
    if not folder:
        return None
    h = hashlib.md5(json.dumps(cache_key, sort_keys=True).encode()).hexdigest()[:8]
    fname = f'{session_name}_brt_{h}.csv'
    # Check legacy directory first
    legacy = os.path.join(folder, 'weight_bearing_analysis', fname)
    if os.path.isfile(legacy):
        return legacy
    feat_dir = os.path.join(folder, 'gait_limb_analysis')
    os.makedirs(feat_dir, exist_ok=True)
    return os.path.join(feat_dir, fname)


def contour_cache_path(ctx, session_name, cache_key):
    """Return path for paw contour cache CSV, or None if unavailable."""
    import hashlib, json
    folder = ctx.project_folder
    if not folder:
        return None
    h = hashlib.md5(json.dumps(cache_key, sort_keys=True).encode()).hexdigest()[:8]
    fname = f'{session_name}_contour_{h}.csv'
    # Check legacy directory first
    legacy = os.path.join(folder, 'weight_bearing_analysis', fname)
    if os.path.isfile(legacy):
        return legacy
    feat_dir = os.path.join(folder, 'gait_limb_analysis')
    os.makedirs(feat_dir, exist_ok=True)
    return os.path.join(feat_dir, fname)


# ─────────────────────────────────────────────────────────────────────────────
# Speed-based contact detection  (Kumar Lab, Cell Reports 2022)
# ─────────────────────────────────────────────────────────────────────────────

def compute_speed_contact(paw_x, paw_y, fps,
                          threshold='auto',
                          median_ms=50,
                          min_bout_ms=30):
    """Return boolean stance mask using paw speed thresholding.

    Parameters
    ----------
    paw_x, paw_y : array-like  — DLC x,y coordinates for one paw
    fps           : float       — video frame rate
    threshold     : float|'auto' — speed cutoff (px/s); 'auto' = 20th pctile
    median_ms     : int          — median filter window in ms
    min_bout_ms   : int          — debounce: remove bouts shorter than this

    Returns
    -------
    np.ndarray[bool] — True = stance (contact), False = swing
    """
    x = np.asarray(paw_x, dtype=float)
    y = np.asarray(paw_y, dtype=float)
    n = len(x)
    if n < 2:
        return np.ones(n, dtype=bool)

    # Frame-to-frame speed in px/s
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    speed = np.sqrt(dx**2 + dy**2) * fps

    # Median filter (smooth jitter)
    if _SCIPY_NDIMAGE_OK and median_ms > 0:
        win = max(1, round(median_ms / 1000.0 * fps))
        if win % 2 == 0:
            win += 1  # median_filter needs odd window
        speed = _median_filter(speed, size=win)

    # Threshold
    if threshold == 'auto' or threshold is None:
        threshold = float(np.percentile(speed, 20))
    else:
        threshold = float(threshold)

    stance = speed < threshold

    # Debounce: remove stance/swing bouts shorter than min_bout_ms
    if min_bout_ms > 0:
        min_frames = max(1, round(min_bout_ms / 1000.0 * fps))
        stance = debounce(stance, min_frames)

    return stance


def debounce(mask, min_frames):
    """Remove boolean runs shorter than min_frames."""
    out = mask.copy()
    changes = np.diff(out.astype(int), prepend=int(out[0]) ^ 1)
    starts = np.where(changes != 0)[0]
    ends = np.append(starts[1:], len(out))
    for s, e in zip(starts, ends):
        if (e - s) < min_frames:
            out[s:e] = not out[s]  # flip short bout to surrounding state
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Gait bout extraction helper
# ─────────────────────────────────────────────────────────────────────────────

def gait_bouts(mask, fps, min_stride_ms=0):
    """Find onset/offset of stance (True) and swing (False) runs.

    Parameters
    ----------
    mask : array-like of bool — per-frame contact mask
    fps : float — frames per second
    min_stride_ms : float — minimum stance bout duration in ms;
        bouts shorter than this are discarded as noise (default 0 = no filter)

    Returns
    -------
    stance_durs : list[float] — durations of stance bouts in seconds
    swing_durs  : list[float] — durations of swing bouts in seconds
    stance_onsets : list[int] — frame indices where stance begins
    """
    n = len(mask)
    if n == 0:
        return [], [], []
    m = np.asarray(mask, dtype=bool)
    d = np.diff(m.astype(int), prepend=int(~m[0]))
    raw_stance_onsets = np.where(d == 1)[0]   # swing→stance transitions
    stance_offsets = np.where(d == -1)[0]      # stance→swing transitions

    # Ensure balanced pairs
    if len(raw_stance_onsets) == 0:
        # All stance or all swing
        if m[0]:
            return [n / fps], [], [0]
        else:
            return [], [n / fps], []

    min_frames = max(1, round(min_stride_ms / 1000.0 * fps)) if min_stride_ms > 0 else 0

    # Build stance bouts and filter by minimum duration
    stance_onsets = []
    stance_durs = []
    for i, on in enumerate(raw_stance_onsets):
        idx = np.searchsorted(stance_offsets, on, side='right')
        off = int(stance_offsets[idx]) if idx < len(stance_offsets) else n
        bout_len = off - on
        if min_frames > 0 and bout_len < min_frames:
            continue
        stance_onsets.append(int(on))
        stance_durs.append(bout_len / fps)

    if not stance_onsets:
        return [], [], []

    # Swing durations: gap between end of one stance and start of the next
    swing_durs = []
    stance_onsets_arr = np.array(stance_onsets)
    for i in range(len(stance_onsets) - 1):
        # Find the offset of the current stance bout
        idx = np.searchsorted(stance_offsets, stance_onsets[i], side='right')
        off = int(stance_offsets[idx]) if idx < len(stance_offsets) else n
        swing_dur = (stance_onsets[i + 1] - off) / fps
        if swing_dur > 0:
            swing_durs.append(swing_dur)

    return stance_durs, swing_durs, stance_onsets


def regularity_index(masks, fps, frame_slice, loco_filter_mask,
                     confidence_mask, min_stance_ms=0):
    """Compute regularity index (RI) — percentage of normal step sequences.

    RI = (NSSP × 4 / total_paw_placements) × 100
    where NSSP = number of normal step-sequence patterns.
    Normal patterns: any cycle of 4 consecutive paw placements where
    all 4 paws appear exactly once.

    Returns float or None if insufficient data.
    """
    if not all(r in masks for r in ('HL', 'HR', 'FL', 'FR')):
        return None

    # Get stance onsets for all 4 paws
    all_onsets = []
    for role in ('HL', 'HR', 'FL', 'FR'):
        mask_arr = (masks[role].values.astype(bool) if hasattr(masks[role], 'values')
                    else np.asarray(masks[role], dtype=bool))
        if loco_filter_mask is not None:
            lm = loco_filter_mask[frame_slice] if frame_slice is not None else loco_filter_mask
            mask_arr = mask_arr & lm[:len(mask_arr)]
        if confidence_mask is not None:
            cm = confidence_mask[frame_slice] if frame_slice is not None else confidence_mask
            mask_arr = mask_arr & cm[:len(mask_arr)]
        _, _, onsets = gait_bouts(mask_arr, fps, min_stance_ms)
        for o in onsets:
            all_onsets.append((o, role))

    if len(all_onsets) < 8:
        return None

    # Sort by frame
    all_onsets.sort(key=lambda x: x[0])
    total_placements = len(all_onsets)

    # Slide a window of 4 and check if all 4 paws are represented
    nssp = 0
    for i in range(0, len(all_onsets) - 3, 4):
        group_roles = {all_onsets[i + k][1] for k in range(4)}
        if len(group_roles) == 4:
            nssp += 1

    if total_placements < 4:
        return None

    ri = round(nssp * 4 / total_placements * 100, 2)
    return ri


def print_position(masks, paw_xy, hind_role, fore_role, fps, frame_slice,
                   loco_filter_mask, confidence_mask, min_stance_ms=0):
    """Compute print position — distance between hind paw strike and
    most recent ipsilateral fore paw strike position (px).

    Returns float (mean distance in px) or NaN.
    """
    # Get stance onsets for hind and fore paw
    for role in (hind_role, fore_role):
        if role not in masks or role not in paw_xy:
            return float('nan')

    def _get_onsets(role):
        mask_arr = (masks[role].values.astype(bool) if hasattr(masks[role], 'values')
                    else np.asarray(masks[role], dtype=bool))
        if loco_filter_mask is not None:
            lm = loco_filter_mask[frame_slice] if frame_slice is not None else loco_filter_mask
            mask_arr = mask_arr & lm[:len(mask_arr)]
        if confidence_mask is not None:
            cm = confidence_mask[frame_slice] if frame_slice is not None else confidence_mask
            mask_arr = mask_arr & cm[:len(mask_arr)]
        _, _, onsets = gait_bouts(mask_arr, fps, min_stance_ms)
        return onsets

    hind_onsets = _get_onsets(hind_role)
    fore_onsets = _get_onsets(fore_role)

    if len(fore_onsets) < 2 or len(hind_onsets) < 1:
        return float('nan')

    hpx, hpy = paw_xy[hind_role]
    fpx, fpy = paw_xy[fore_role]
    sl_start = (frame_slice.start or 0) if frame_slice is not None else 0

    fore_arr = np.array(fore_onsets)
    dists = []
    for ho in hind_onsets:
        abs_ho = sl_start + ho
        # Find the most recent fore onset before this hind onset
        idx = np.searchsorted(fore_arr, ho, side='left') - 1
        if idx < 0:
            continue
        fo = fore_onsets[idx]
        abs_fo = sl_start + fo
        if abs_ho < len(hpx) and abs_fo < len(fpx):
            d = np.sqrt((hpx[abs_ho] - fpx[abs_fo])**2 +
                        (hpy[abs_ho] - fpy[abs_fo])**2)
            dists.append(d)

    return round(float(np.mean(dists)), 2) if dists else float('nan')


# ─────────────────────────────────────────────────────────────────────────────
# Contour shape helpers (verbatim staticmethod moves)
# ─────────────────────────────────────────────────────────────────────────────

def resample_contour(pts, n_points=64):
    """Resample a contour to a fixed number of evenly spaced points.

    pts: (M, 2) array of contour coordinates.
    Returns: (n_points, 2) array.
    """
    if len(pts) < 3:
        return None
    # Close the contour
    closed = np.vstack([pts, pts[0:1]])
    # Cumulative arc length
    diffs = np.diff(closed, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cum_len = np.concatenate([[0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]
    if total_len <= 0:
        return None
    # Evenly spaced parameter values (exclude endpoint to avoid duplicate)
    t_new = np.linspace(0, total_len, n_points, endpoint=False)
    # Interpolate x and y
    x_new = np.interp(t_new, cum_len, closed[:, 0])
    y_new = np.interp(t_new, cum_len, closed[:, 1])
    return np.column_stack([x_new, y_new])


def normalize_contour(pts, area):
    """Center contour at origin and normalize by sqrt(area)."""
    if pts is None or area <= 0:
        return None
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    scale = np.sqrt(area)
    if scale > 0:
        centered = centered / scale
    return centered


def shape_metrics(pts):
    """Compute aspect_ratio and circularity from (N,2) contour points."""
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    w, h = x_max - x_min, y_max - y_min
    dim_max, dim_min = max(w, h), min(w, h)
    ar = dim_max / dim_min if dim_min > 0 else 999.0
    # Perimeter (sum of segment lengths)
    diffs = np.diff(np.vstack([pts, pts[0:1]]), axis=0)
    perimeter = np.sum(np.sqrt((diffs ** 2).sum(axis=1)))
    # Area (shoelace)
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    circ = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
    return ar, circ


def shape_metrics_batch(stacked):
    """Vectorized aspect_ratio + circularity for (N, 64, 2) contour array."""
    mins = stacked.min(axis=1)
    maxs = stacked.max(axis=1)
    wh = maxs - mins
    dim_max = wh.max(axis=1)
    dim_min = wh.min(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        ar = np.where(dim_min > 0, dim_max / dim_min, 999.0)
    closed = np.concatenate([stacked, stacked[:, 0:1, :]], axis=1)
    diffs = np.diff(closed, axis=1)
    perimeter = np.sqrt((diffs ** 2).sum(axis=2)).sum(axis=1)
    x, y = stacked[:, :, 0], stacked[:, :, 1]
    area = 0.5 * np.abs(
        (x * np.roll(y, -1, axis=1)).sum(axis=1) -
        (y * np.roll(x, -1, axis=1)).sum(axis=1))
    with np.errstate(divide='ignore', invalid='ignore'):
        circ = np.where(perimeter > 0,
                        (4 * np.pi * area) / (perimeter ** 2), 0.0)
    return ar, circ


def rebin_timecourse(xs, means, errs, rebin_min):
    """Aggregate timecourse data into larger bins."""
    if not xs or rebin_min <= 0:
        return xs, means, errs
    new_xs, new_means, new_errs = [], [], []
    i = 0
    while i < len(xs):
        bin_start = xs[i]
        bin_end = bin_start + rebin_min
        group_m, group_e = [], []
        while i < len(xs) and xs[i] < bin_end:
            group_m.append(means[i])
            group_e.append(errs[i])
            i += 1
        if group_m:
            new_xs.append(bin_start)
            new_means.append(float(np.mean(group_m)))
            # Propagate error: average of errors (simple approach)
            new_errs.append(float(np.mean(group_e)))
    return new_xs, new_means, new_errs


# ─────────────────────────────────────────────────────────────────────────────
# Contact-mask construction (single implementation, shared by the analyze
# path and the Adjust-Contact recompute path)
# ─────────────────────────────────────────────────────────────────────────────

def build_contact_masks(active_paws, height_df, paw_xy, n_frames, fps,
                        params, mm_per_px=None):
    """Height / speed / combined per-paw contact masks (verbatim block from
    ``_analyze_session``). The contour-area override and brightness-weight
    refinement are applied separately (``apply_contour_area_contact`` /
    ``apply_brightness_weight``)."""
    contact_masks = {}
    contact_method = params.get('contact_method', 'height')
    thresh = params['contact_threshold']
    # User-tuned threshold is in px (matches GUI label "Contact thresh (px):").
    # Height column is now in mm when calibration is on, so scale
    # the threshold to keep the user's intent intact.
    if mm_per_px is not None:
        thresh = float(thresh) * float(mm_per_px)

    for role, bp in active_paws.items():
        px, py = paw_xy.get(role, (None, None))

        # Height-based contact
        h_col = f'{bp}_Height'
        if h_col not in height_df.columns:
            matches = [c for c in height_df.columns if bp.lower() in c.lower()]
            h_col = matches[0] if matches else None

        height_mask = None
        if h_col:
            height_mask = (height_df[h_col].values[:n_frames] < thresh)

        # Speed-based contact
        speed_mask = None
        if contact_method in ('speed', 'combined') and px is not None:
            speed_mask = compute_speed_contact(
                px, py, fps,
                threshold=params.get('speed_threshold', 'auto'),
                median_ms=params.get('median_filter_ms', 50),
                min_bout_ms=params.get('min_bout_ms', 30))

        # Combine according to method
        if contact_method == 'height':
            mask = height_mask
        elif contact_method == 'speed':
            mask = speed_mask if speed_mask is not None else height_mask
        elif contact_method == 'combined':
            if height_mask is not None and speed_mask is not None:
                mask = height_mask & speed_mask
            else:
                mask = height_mask if height_mask is not None else speed_mask
        else:
            mask = height_mask

        if mask is not None:
            contact_masks[role] = pd.Series(mask, dtype=bool).reset_index(drop=True)

    return contact_masks


def apply_brightness_weight(contact_masks, active_paws, height_df,
                            brightness_series, n_frames, params,
                            log=_noop_log):
    """Brightness-weighted contact refinement (verbatim block). Mutates and
    returns ``contact_masks``."""
    brt_weight = params.get('brt_weight', 0.0)
    if brt_weight > 0 and brightness_series and params.get('contact_method') != 'contour_area':
        thresh = params['contact_threshold']
        for role, bp in active_paws.items():
            if role not in brightness_series or role not in contact_masks:
                continue

            # Retrieve raw height values for this paw
            h_col = f'{bp}_Height'
            if h_col not in height_df.columns:
                matches = [c for c in height_df.columns if bp.lower() in c.lower()]
                h_col = matches[0] if matches else None
            if h_col is None:
                continue

            h_vals = height_df[h_col].values[:n_frames].astype(float)
            b_vals = brightness_series[role].values.astype(float)

            # Height score: 1 at floor level, 0 at threshold or above
            h_score = np.clip(1.0 - h_vals / max(float(thresh), 1.0), 0.0, 1.0)

            # Brightness score: normalise to 90th percentile of session brightness
            valid_b = b_vals[np.isfinite(b_vals) & (b_vals > 0)]
            brt_90 = float(np.percentile(valid_b, 90)) if len(valid_b) > 0 else 1.0
            b_score = np.clip(b_vals / max(brt_90, 1.0), 0.0, 1.0)

            combined = (1.0 - brt_weight) * h_score + brt_weight * b_score
            contact_masks[role] = pd.Series(combined > 0.5, dtype=bool)
            log(f"  {role}: brt_weight={brt_weight:.2f}, brt_90th={brt_90:.1f}")
    return contact_masks


def apply_contour_area_contact(contact_masks, paw_contour_data, n_frames,
                               params, log=_noop_log):
    """Contour-area contact (formalin-style), verbatim block.

    A hind paw counts as in-contact when its contour area exceeds the
    threshold; both hind paws are gated together (matches the formalin
    figures). Applied after contour extraction — replaces the
    height/brightness contact masks. The DLC-likelihood filter is applied
    downstream in the metrics stage, exactly as for every other contact
    method. Mutates and returns ``contact_masks``."""
    if params.get('contact_method') == 'contour_area' and paw_contour_data:
        # Band, not floor: the upper bound excludes contours that spilled off the paw
        # onto the body or background, which bias intensity down asymmetrically.
        _area_thr = float(params.get('contour_area_threshold', 1500))
        _area_max = float(params.get('contour_area_max', 5000)) or float('inf')
        _in_band = lambda a: (a > _area_thr) & (a <= _area_max)
        _hind = [r for r in ('HL', 'HR')
                 if r in paw_contour_data and 'areas' in paw_contour_data[r]]
        if len(_hind) == 2:
            _aHL = np.asarray(paw_contour_data['HL']['areas'])[:n_frames]
            _aHR = np.asarray(paw_contour_data['HR']['areas'])[:n_frames]
            _both = pd.Series(_in_band(_aHL) & _in_band(_aHR),
                              dtype=bool).reset_index(drop=True)
            contact_masks['HL'] = _both
            contact_masks['HR'] = _both.copy()
            log(f"  Contact: contour-area band ({_area_thr:.0f}-{_area_max:.0f} "
                f"px^2, both hind).")
        elif len(_hind) == 1:
            _r = _hind[0]
            _a = np.asarray(paw_contour_data[_r]['areas'])[:n_frames]
            contact_masks[_r] = pd.Series(_in_band(_a), dtype=bool).reset_index(drop=True)
            log(f"  Contact: contour-area band ({_area_thr:.0f}-{_area_max:.0f} "
                f"px^2, {_r} only).")
        else:
            log("  Contour-area contact selected but no hind contour data was "
                "extracted; keeping the existing contact masks.")
    return contact_masks


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: session data loading (DLC, fps, masks, brightness/contour + caches)
# ─────────────────────────────────────────────────────────────────────────────

def load_session_data(sess, paw_map, params, ctx,
                      log=None, progress_cb=None, cancel=None):
    """Load & extract everything ``compute_all_metrics`` needs for one session.

    Returns the ``data`` dict (same keys the old tab stashed per session in
    ``_session_intermediates``, plus ``session_name`` and ``_mm_per_px``),
    or None on failure. Verbatim port of the first ~2/3 of
    ``GaitLimbTab._analyze_session``.
    """
    log = log or _noop_log
    if cancel is None:
        cancel = threading.Event()

    def _sub_progress(done, total, text):
        if progress_cb is not None:
            try:
                progress_cb(done, total, text)
            except Exception:
                pass

    dlc_file   = sess.get('dlc')
    video_file = sess.get('video')

    if not dlc_file or not os.path.isfile(dlc_file):
        log("  Skipped (no DLC file)")
        return None

    active_paws = {role: bp for role, bp in paw_map.items() if bp}
    active_bps  = list(set(active_paws.values()))

    # Resolve mm_per_pixel for this session via project's
    # calibration_mode. When set, pose extractor scales coords to
    # mm so output gait metrics (stride length, step length, body
    # speed) come out in physical units. Falls back to pixels
    # transparently when calibration is off.
    _mm_per_px = resolve_mm_per_pixel(
        ctx, {'video': video_file, 'mm_per_pixel': None})
    # When the session has its own embedded mm_per_pixel via
    # PawCapture metadata, prefer that (auto mode).
    if _mm_per_px is None:
        try:
            from pawcapture_meta import read_calibration as _pc_read
            _cal = _pc_read(video_file) if video_file else None
            if _cal and _cal.get('mm_per_pixel'):
                _mm_per_px = resolve_mm_per_pixel(
                    ctx, {'video': video_file,
                          'mm_per_pixel': float(_cal['mm_per_pixel'])})
        except Exception:
            pass

    # ── Paw heights + coordinates ───────────────────────────────────────
    extractor = PoseFeatureExtractor(active_bps, mm_per_pixel=_mm_per_px)
    if _mm_per_px is not None:
        log(f"  ✓ Calibration on: gait metrics in mm "
            f"(mm_per_pixel={_mm_per_px:.5f})")
    try:
        dlc_df = extractor.load_dlc_data(dlc_file)
        bp_xcord, bp_ycord, bp_prob = extractor.get_bodypart_coords(dlc_df)
        height_df = extractor.calculate_paw_height(
            bp_xcord, bp_ycord, window=params['height_window'])
    except Exception as e:
        log(f"  DLC error: {e}")
        return None

    n_frames = len(height_df)

    # ── FPS ──────────────────────────────────────────────────────────────
    fps = params['fallback_fps']
    _used_fallback_fps = True
    if video_file and os.path.isfile(video_file) and _CV2_OK:
        try:
            cap = cv2.VideoCapture(video_file)
            fps_v = cap.get(cv2.CAP_PROP_FPS)
            if fps_v > 0:
                fps = fps_v
                _used_fallback_fps = False
            cap.release()
        except Exception as _vid_err:
            print(f"Warning: could not read FPS from {video_file}: {_vid_err}")

    # ── DLC confidence mask (Step 6) ─────────────────────────────────────
    confidence_mask = None
    if params.get('use_likelihood') and bp_prob is not None:
        lk_thresh = params.get('likelihood_threshold', 0.6)
        # Per-paw confidence: mark frame as low-confidence if ANY
        # active paw is below threshold
        paw_ok = np.ones(n_frames, dtype=bool)
        for role, bp in active_paws.items():
            prob_col = next((c for c in bp_prob.columns
                             if bp.lower() in c.lower()), None)
            if prob_col is not None:
                paw_ok &= (bp_prob[prob_col].values[:n_frames] >= lk_thresh)
        confidence_mask = paw_ok
        n_low = int((~paw_ok).sum())
        if n_low > 0:
            log(f"  DLC filter: {n_low} frames ({100*n_low/n_frames:.1f}%) below likelihood {lk_thresh}")

    # ── Locomotion filter mask & body speed (always computed) ─────────
    loco_mask = None
    body_speed = None
    frame_displacements = None
    loco_thresh = params.get('loco_threshold', 20.0)
    # User-tuned threshold is in px/s. When calibration is on, the
    # body_speed below comes out in mm/s, so the threshold needs
    # the same px→mm conversion to remain semantically equivalent
    # to what the user set in the GUI.
    if _mm_per_px is not None:
        loco_thresh = float(loco_thresh) * float(_mm_per_px)
    # Try to find tailbase coordinates
    tb_bp = None
    tb_x_col = None
    for candidate in ['tailbase', 'tail_base', 'tb']:
        tb_x_col = next((c for c in bp_xcord.columns
                         if candidate in c.lower()), None)
        if tb_x_col:
            tb_bp = candidate
            break
    if tb_x_col is None:
        # Fallback: use the midpoint of hind paws as body center
        hl_bp = active_paws.get('HL', '')
        hr_bp = active_paws.get('HR', '')
        hl_x_col = next((c for c in bp_xcord.columns if hl_bp.lower() in c.lower()), None)
        hr_x_col = next((c for c in bp_xcord.columns if hr_bp.lower() in c.lower()), None)
        hl_y_col = next((c for c in bp_ycord.columns if hl_bp.lower() in c.lower()), None)
        hr_y_col = next((c for c in bp_ycord.columns if hr_bp.lower() in c.lower()), None)
        if hl_x_col and hr_x_col:
            cx = (bp_xcord[hl_x_col].values + bp_xcord[hr_x_col].values) / 2.0
            cy = (bp_ycord[hl_y_col].values + bp_ycord[hr_y_col].values) / 2.0
        else:
            cx = cy = None
    else:
        # NOTE: ``candidate`` is the loop variable above — this only executes
        # after ``break``, so it necessarily holds the name that matched
        # tb_x_col. Verified semantically sound; kept verbatim.
        tb_y_col = next((c for c in bp_ycord.columns
                         if candidate in c.lower()), None)
        cx = bp_xcord[tb_x_col].values[:n_frames].astype(float)
        cy = bp_ycord[tb_y_col].values[:n_frames].astype(float) if tb_y_col else None

    if cx is not None and cy is not None:
        dx = np.diff(cx, prepend=cx[0])
        dy = np.diff(cy, prepend=cy[0])
        frame_displacements = np.sqrt(dx**2 + dy**2)
        body_speed = frame_displacements * fps
        loco_mask = body_speed > loco_thresh
        n_loco = int(loco_mask.sum())
        _unit = 'mm/s' if _mm_per_px is not None else 'px/s'
        log(f"  Body speed computed: {n_loco}/{n_frames} frames above "
            f"{loco_thresh:.1f} {_unit}")
    else:
        log("  Body speed: no tailbase/body center found, skipping")

    # ── Contact masks ────────────────────────────────────────────────────
    # Helper to get x,y arrays for a body part
    def _get_xy(bp):
        x_col = next((c for c in bp_xcord.columns if bp.lower() in c.lower()), None)
        y_col = next((c for c in bp_ycord.columns if bp.lower() in c.lower()), None)
        if x_col and y_col:
            return (bp_xcord[x_col].values[:n_frames].astype(float),
                    bp_ycord[y_col].values[:n_frames].astype(float))
        return None, None

    # Store per-paw x,y for gait spatial metrics later
    paw_xy = {}
    for role, bp in active_paws.items():
        px, py = _get_xy(bp)
        if px is not None:
            paw_xy[role] = (px, py)

    contact_masks = build_contact_masks(
        active_paws, height_df, paw_xy, n_frames, fps, params,
        mm_per_px=_mm_per_px)

    # ── Brightness (optional) ────────────────────────────────────────────
    brightness_series = {}
    paw_contour_data = {}  # may be populated during brightness pass or standalone
    _contour_cache_path = None
    contour_cache_key = None
    # Build contour paw subset (hind-only by default)
    contour_paws = {}
    if params.get('paw_contour'):
        _hind_roles = ('HL', 'HR')
        contour_paws = {r: bp for r, bp in active_paws.items()
                        if r in _hind_roles or params.get('contour_forelimbs')}
    if contour_paws and video_file and os.path.isfile(video_file) and _CV2_OK:
        contour_roi_sizes = params.get('contour_roi_sizes', params.get('roi_sizes', {}))
        contour_cache_key = {
            'video_mtime': round(os.path.getmtime(video_file), 2),
            'dlc_mtime':   round(os.path.getmtime(dlc_file), 2),
            'contour_roi_sizes': sorted(
                {contour_paws[r]: contour_roi_sizes.get(r, 20) for r in contour_paws}.items()),
            'crop_x': params.get('crop_offset_x', 0),
            'crop_y': params.get('crop_offset_y', 0),
            'extraction_stride': params.get('extraction_stride', 1),
        }
        _contour_cache_path = contour_cache_path(ctx, sess['session_name'], contour_cache_key)
        if _contour_cache_path and not os.path.isfile(_contour_cache_path):
            log("  Contour cache miss (video/pose files or extraction "
                "settings changed since the last run) — extracting from "
                "video (this may take a while).")

        # --- try contour cache load ---
        if _contour_cache_path and os.path.isfile(_contour_cache_path):
            try:
                cached_cdf = pd.read_csv(_contour_cache_path)
                metric_names = ['areas', 'spreads', 'intensities', 'widths',
                                'solidities', 'aspect_ratios', 'circularities']
                for role in contour_paws:
                    role_data = {}
                    for mn in metric_names:
                        col = f'{mn}_{role}'
                        if col in cached_cdf.columns:
                            arr = cached_cdf[col].values
                            if len(arr) > n_frames:
                                arr = arr[:n_frames]
                            elif len(arr) < n_frames:
                                arr = np.pad(arr, (0, n_frames - len(arr)))
                            role_data[mn] = arr.astype(float)
                    if role_data:
                        paw_contour_data[role] = role_data
                # Load cached contour shapes (.npz) if available
                shapes_path = _contour_cache_path.replace('.csv', '_shapes.npz')
                if os.path.isfile(shapes_path):
                    try:
                        _npz = np.load(shapes_path, allow_pickle=False)
                        for role in contour_paws:
                            if role in _npz and role in paw_contour_data:
                                arr = _npz[role]
                                if arr.ndim == 3 and arr.shape[1:] == (64, 2):
                                    paw_contour_data[role]['contour_shapes'] = list(arr)
                        log("  Contour shapes loaded from cache.")
                    except Exception:
                        pass
                if paw_contour_data:
                    log("  Contour data loaded from cache.")
            except Exception as e:
                log(f"  Contour cache load failed ({e}), will re-extract.")
                paw_contour_data = {}

    if params['use_brightness'] and video_file and os.path.isfile(video_file):
        # --- build cache key (depends only on inputs, not analysis params) ---
        roi_sizes = params.get('roi_sizes', {})
        cache_key = {
            'video_mtime': round(os.path.getmtime(video_file), 2),
            'dlc_mtime':   round(os.path.getmtime(dlc_file),   2),
            'roi_sizes':   sorted(
                {active_paws[r]: roi_sizes.get(r, 50) for r in active_paws}.items()),
            'crop_x':           params.get('crop_offset_x', 0),
            'crop_y':           params.get('crop_offset_y', 0),
            'brt_thresh':       params.get('brt_threshold', 0),
            'extraction_stride': params.get('extraction_stride', 1),
        }
        cache_path = brt_cache_path(ctx, sess['session_name'], cache_key)

        # --- try cache load ---
        if cache_path and os.path.isfile(cache_path):
            try:
                cached_df = pd.read_csv(cache_path)
                for role, bp in active_paws.items():
                    col = f'Pix_{bp}'
                    if col in cached_df.columns:
                        s = cached_df[col].reset_index(drop=True)
                        if len(s) > n_frames:
                            s = s.iloc[:n_frames].reset_index(drop=True)
                        elif len(s) < n_frames:
                            s = s.reindex(range(n_frames))
                        brightness_series[role] = s
                log(f"  Brightness loaded from cache.")
            except Exception as e:
                log(f"  Cache load failed ({e}), re-extracting.")
                brightness_series = {}

        # --- extract fresh if cache miss ---
        if not brightness_series:
            log(f"  Brightness cache miss (video/pose files or extraction "
                f"settings changed since the last run) — extracting from "
                f"video (this may take a while).")
            try:
                thresh_val  = params.get('brt_threshold', 0)
                square_size = {active_paws[role]: roi_sizes.get(role, 50)
                               for role in active_paws}

                # When brightness doesn't affect contact detection (brt_weight==0),
                # only decode frames where at least one paw is in contact — much faster.
                hint_mask = None
                if abs(params.get('brt_weight', 1.0)) < 1e-6 and contact_masks:
                    hint_mask = np.zeros(n_frames, dtype=bool)
                    for mask_arr in contact_masks.values():
                        hint_mask |= mask_arr.values

                # ── Contour callback (piggyback on brightness video pass) ──
                contour_callback = None
                if (contour_paws and _CV2_OK
                        and not paw_contour_data
                        and video_file and os.path.isfile(video_file)):
                    # Pre-allocate contour arrays (including toe-spreading)
                    for role in contour_paws:
                        paw_contour_data[role] = {
                            'areas': np.zeros(n_frames, dtype=float),
                            'spreads': np.zeros(n_frames, dtype=float),
                            'intensities': np.zeros(n_frames, dtype=float),
                            'widths': np.zeros(n_frames, dtype=float),
                            'solidities': np.zeros(n_frames, dtype=float),
                            'aspect_ratios': np.zeros(n_frames, dtype=float),
                            'circularities': np.zeros(n_frames, dtype=float),
                            'contour_shapes': [],   # normalized (64,2) arrays
                            'contour_solidities': [],  # extraction-time solidity per shape
                        }
                    contour_roi_sizes = params.get('contour_roi_sizes',
                                                   params.get('roi_sizes', {}))
                    _max_shapes = 500  # limit stored shapes per paw
                    _stride_val_shapes = max(1, params.get('extraction_stride', 1))
                    _shape_every = max(1, n_frames // (_max_shapes * _stride_val_shapes))
                    contour_crop_x = params.get('crop_offset_x', 0)
                    contour_crop_y = params.get('crop_offset_y', 0)

                    # Sub-progress setup
                    _stride_val = max(1, params.get('extraction_stride', 1))
                    total_contour_frames = len(range(0, n_frames, _stride_val))
                    _sub_progress(0, 100, 'Contour extraction: 0%')
                    _contour_update_interval = max(1, total_contour_frames // 20)

                    def contour_callback(i_frame, gray_u8, frame):
                        """Called by brightness extractor for each decoded frame."""
                        fh, fw = gray_u8.shape[:2]
                        for role, bp in contour_paws.items():
                            if role not in paw_xy:
                                continue
                            px_arr, py_arr = paw_xy[role]
                            if i_frame >= len(px_arr):
                                continue
                            bx = int(px_arr[i_frame]) + contour_crop_x
                            by = int(py_arr[i_frame]) + contour_crop_y
                            rh = contour_roi_sizes.get(role, 20)
                            x1 = max(0, bx - rh); x2 = min(fw, bx + rh)
                            y1 = max(0, by - rh); y2 = min(fh, by + rh)
                            if x2 <= x1 or y2 <= y1:
                                continue
                            roi = gray_u8[y1:y2, x1:x2]
                            if roi.size == 0:
                                continue
                            blurred = cv2.GaussianBlur(roi, (3, 3), 0)
                            _, thresh_img = cv2.threshold(blurred, 0, 255,
                                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                            contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL,
                                                            cv2.CHAIN_APPROX_SIMPLE)
                            if not contours:
                                continue
                            best = max(contours, key=lambda c: cv2.contourArea(c))
                            area = cv2.contourArea(best)
                            if area < 4:
                                continue
                            if area > 0:
                                paw_contour_data[role]['areas'][i_frame] = area
                                x_b, y_b, w_b, h_b = cv2.boundingRect(best)
                                paw_contour_data[role]['spreads'][i_frame] = max(w_b, h_b)
                                mask_c = np.zeros(roi.shape, dtype=np.uint8)
                                cv2.drawContours(mask_c, [best], -1, 255, -1)
                                paw_contour_data[role]['intensities'][i_frame] = cv2.mean(roi, mask=mask_c)[0]
                                # Toe-spreading metrics
                                paw_contour_data[role]['widths'][i_frame] = min(w_b, h_b)
                                hull = cv2.convexHull(best)
                                hull_area = cv2.contourArea(hull)
                                paw_contour_data[role]['solidities'][i_frame] = area / hull_area if hull_area > 0 else 0.0
                                dim_max = max(w_b, h_b)
                                dim_min = min(w_b, h_b)
                                paw_contour_data[role]['aspect_ratios'][i_frame] = dim_max / dim_min if dim_min > 0 else 0.0
                                perimeter = cv2.arcLength(best, True)
                                paw_contour_data[role]['circularities'][i_frame] = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
                                # Store normalized contour shape (subsampled)
                                if (i_frame % _shape_every == 0
                                        and len(paw_contour_data[role]['contour_shapes']) < _max_shapes):
                                    pts = best.squeeze()
                                    if pts.ndim == 2 and len(pts) >= 3:
                                        resampled = resample_contour(pts, 64)
                                        normed = normalize_contour(resampled, area)
                                        if normed is not None:
                                            paw_contour_data[role]['contour_shapes'].append(normed)
                                            paw_contour_data[role]['contour_solidities'].append(
                                                paw_contour_data[role]['solidities'][i_frame])

                        # Progress update
                        frame_idx = i_frame // _stride_val
                        if frame_idx % _contour_update_interval == 0:
                            pct = int(100 * frame_idx / total_contour_frames)
                            _sub_progress(pct, 100, f'Contour extraction: {pct}%')

                brt_ex = PixelBrightnessExtractorOptimized(
                    active_bps,
                    square_size=square_size,
                    pixel_threshold=float(thresh_val) if thresh_val > 0 else None,
                    crop_offset_x=params.get('crop_offset_x', 0),
                    crop_offset_y=params.get('crop_offset_y', 0),
                )
                brt_df = brt_ex.extract_brightness_features(
                    dlc_file, video_file,
                    stride=params.get('extraction_stride', 1),
                    frame_mask=hint_mask,
                    cancel_flag=cancel,
                    frame_callback=contour_callback,
                )
                for role, bp in active_paws.items():
                    col = f'Pix_{bp}'
                    if col not in brt_df.columns:
                        matches = [c for c in brt_df.columns
                                   if bp.lower() in c.lower()
                                   and c.startswith('Pix_')]
                        col = matches[0] if matches else None
                    if col:
                        s = brt_df[col].reset_index(drop=True)
                        if len(s) > n_frames:
                            s = s.iloc[:n_frames].reset_index(drop=True)
                        elif len(s) < n_frames:
                            s = s.reindex(range(n_frames))
                        brightness_series[role] = s
                # --- save to cache ---
                if cache_path and brightness_series:
                    try:
                        pd.DataFrame({f'Pix_{active_paws[r]}': brightness_series[r]
                                      for r in brightness_series}).to_csv(cache_path, index=False)
                        log(f"  Brightness cached.")
                        import json as _json
                        sidecar = cache_path.replace('.csv', '.json')
                        with open(sidecar, 'w') as _f:
                            _json.dump(cache_key, _f, indent=2)
                    except Exception:
                        pass
                # --- save contour to cache ---
                if _contour_cache_path and paw_contour_data:
                    try:
                        contour_cols = {}
                        for role, arrays in paw_contour_data.items():
                            for metric_name, arr in arrays.items():
                                if metric_name in ('contour_shapes', 'contour_solidities'):
                                    continue  # not cacheable as CSV column
                                contour_cols[f'{metric_name}_{role}'] = arr
                        pd.DataFrame(contour_cols).to_csv(_contour_cache_path, index=False)
                        log("  Contour data cached.")
                        import json as _json2
                        sidecar = _contour_cache_path.replace('.csv', '.json')
                        with open(sidecar, 'w') as _f:
                            _json2.dump(contour_cache_key, _f, indent=2)
                        # Save contour shapes as .npz
                        _shape_arrays = {}
                        for _sr, _sd in paw_contour_data.items():
                            _sl = _sd.get('contour_shapes', [])
                            if _sl:
                                _shape_arrays[_sr] = np.array(_sl)
                        if _shape_arrays:
                            np.savez_compressed(
                                _contour_cache_path.replace('.csv', '_shapes.npz'),
                                **_shape_arrays)
                            log("  Contour shapes cached.")
                    except Exception:
                        pass
                # Clean up contour progress if callback was used
                if contour_callback is not None:
                    _sub_progress(None, None, '')
                    log("  Paw contour extraction complete (during brightness pass).")
            except Exception as e:
                log(f"  Brightness skipped: {e}")

    # ── Brightness-weighted contact refinement ────────────────────────────
    contact_masks = apply_brightness_weight(
        contact_masks, active_paws, height_df, brightness_series,
        n_frames, params, log=log)

    # ── Paw contour area extraction (standalone fallback) ─────────────
    # Only runs when contour wasn't already done during brightness pass
    # (i.e. brightness cached, brightness disabled, or contour-only mode)
    if (contour_paws and not paw_contour_data
            and video_file and os.path.isfile(video_file) and _CV2_OK):
        log("  Extracting paw contour areas (standalone)…")
        try:
            cap = cv2.VideoCapture(video_file)
            if cap.isOpened():
                roi_sizes = params.get('contour_roi_sizes', params.get('roi_sizes', {}))
                crop_x = params.get('crop_offset_x', 0)
                crop_y = params.get('crop_offset_y', 0)
                stride = max(1, params.get('extraction_stride', 1))

                # Pre-allocate arrays (including toe-spreading)
                _max_shapes_b = 500
                _shape_every_b = max(1, n_frames // (_max_shapes_b * stride))
                for role in contour_paws:
                    paw_contour_data[role] = {
                        'areas': np.zeros(n_frames, dtype=float),
                        'spreads': np.zeros(n_frames, dtype=float),
                        'intensities': np.zeros(n_frames, dtype=float),
                        'widths': np.zeros(n_frames, dtype=float),
                        'solidities': np.zeros(n_frames, dtype=float),
                        'aspect_ratios': np.zeros(n_frames, dtype=float),
                        'circularities': np.zeros(n_frames, dtype=float),
                        'contour_shapes': [],
                        'contour_solidities': [],  # extraction-time solidity per shape
                    }

                total_contour_frames = len(range(0, n_frames, stride))
                _sub_progress(0, 100, 'Contour extraction: 0%')
                _update_interval = max(1, total_contour_frames // 20)

                frame_idx = 0
                for fi in range(n_frames):
                    if cancel.is_set():
                        break

                    if fi % stride != 0:
                        cap.grab()   # advance without decoding — fast
                        continue

                    ret, frame = cap.read()
                    if not ret:
                        frame_idx += 1
                        continue

                    # Progress update
                    if frame_idx % _update_interval == 0:
                        pct = int(100 * frame_idx / total_contour_frames)
                        _sub_progress(pct, 100, f'Contour extraction: {pct}%')

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    fh, fw = frame.shape[:2]

                    for role, bp in contour_paws.items():
                        if role not in paw_xy:
                            continue
                        px_arr, py_arr = paw_xy[role]
                        if fi >= len(px_arr):
                            continue
                        bx = int(px_arr[fi]) + crop_x
                        by = int(py_arr[fi]) + crop_y
                        rh = roi_sizes.get(role, 20)
                        x1 = max(0, bx - rh); x2 = min(fw, bx + rh)
                        y1 = max(0, by - rh); y2 = min(fh, by + rh)
                        if x2 <= x1 or y2 <= y1:
                            continue

                        roi = gray[y1:y2, x1:x2]
                        if roi.size == 0:
                            continue

                        blurred = cv2.GaussianBlur(roi, (3, 3), 0)
                        _, thresh_img = cv2.threshold(
                            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                        contours, _ = cv2.findContours(
                            thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        if not contours:
                            continue

                        best = max(contours, key=lambda c: cv2.contourArea(c))
                        area = cv2.contourArea(best)
                        if area > 0:
                            paw_contour_data[role]['areas'][fi] = area
                            x_b, y_b, w_b, h_b = cv2.boundingRect(best)
                            paw_contour_data[role]['spreads'][fi] = max(w_b, h_b)
                            mask_c = np.zeros(roi.shape, dtype=np.uint8)
                            cv2.drawContours(mask_c, [best], -1, 255, -1)
                            paw_contour_data[role]['intensities'][fi] = cv2.mean(roi, mask=mask_c)[0]
                            # Toe-spreading metrics
                            paw_contour_data[role]['widths'][fi] = min(w_b, h_b)
                            hull = cv2.convexHull(best)
                            hull_area = cv2.contourArea(hull)
                            paw_contour_data[role]['solidities'][fi] = area / hull_area if hull_area > 0 else 0.0
                            dim_max = max(w_b, h_b)
                            dim_min = min(w_b, h_b)
                            paw_contour_data[role]['aspect_ratios'][fi] = dim_max / dim_min if dim_min > 0 else 0.0
                            perimeter = cv2.arcLength(best, True)
                            paw_contour_data[role]['circularities'][fi] = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
                            # Store normalized contour shape (subsampled)
                            if (fi % _shape_every_b == 0
                                    and len(paw_contour_data[role]['contour_shapes']) < _max_shapes_b):
                                pts = best.squeeze()
                                if pts.ndim == 2 and len(pts) >= 3:
                                    resampled = resample_contour(pts, 64)
                                    normed = normalize_contour(resampled, area)
                                    if normed is not None:
                                        paw_contour_data[role]['contour_shapes'].append(normed)
                                        paw_contour_data[role]['contour_solidities'].append(
                                            paw_contour_data[role]['solidities'][fi])

                    frame_idx += 1

                # Reset sub-progress
                _sub_progress(None, None, '')

                cap.release()
                log("  Paw contour extraction complete (standalone).")
                # --- save contour to cache ---
                if _contour_cache_path and paw_contour_data:
                    try:
                        contour_cols = {}
                        for role, arrays in paw_contour_data.items():
                            for metric_name, arr in arrays.items():
                                if metric_name in ('contour_shapes', 'contour_solidities'):
                                    continue  # not cacheable as CSV column
                                contour_cols[f'{metric_name}_{role}'] = arr
                        pd.DataFrame(contour_cols).to_csv(_contour_cache_path, index=False)
                        log("  Contour data cached.")
                        import json as _json3
                        sidecar = _contour_cache_path.replace('.csv', '.json')
                        with open(sidecar, 'w') as _f:
                            _json3.dump(contour_cache_key, _f, indent=2)
                        # Save contour shapes as .npz
                        _shape_arrays_b = {}
                        for _sr, _sd in paw_contour_data.items():
                            _sl = _sd.get('contour_shapes', [])
                            if _sl:
                                _shape_arrays_b[_sr] = np.array(_sl)
                        if _shape_arrays_b:
                            np.savez_compressed(
                                _contour_cache_path.replace('.csv', '_shapes.npz'),
                                **_shape_arrays_b)
                            log("  Contour shapes cached.")
                    except Exception:
                        pass
        except Exception as e:
            log(f"  Paw contour extraction failed: {e}")
            paw_contour_data = {}

    # ── Contour-area contact (formalin-style) ────────────────────────────
    contact_masks = apply_contour_area_contact(
        contact_masks, paw_contour_data, n_frames, params, log=log)

    # ── Licking-frame mask ────────────────────────────────────────────────
    lick_mask = np.zeros(n_frames, dtype=bool)
    if params.get('exclude_licking') and params.get('lick_behavior'):
        lick_mask = load_lick_mask(
            ctx.project_folder, sess['session_name'], params['lick_behavior'],
            params.get('lick_threshold', 0.5), n_frames)
        log(f"  Excluding {int(lick_mask.sum())} licking frame(s) "
            f"('{params['lick_behavior']}').")

    return {
        'session_name': sess['session_name'],
        'height_df': height_df,
        'bp_xcord': bp_xcord,
        'bp_ycord': bp_ycord,
        'bp_prob': bp_prob,
        'fps': fps,
        '_used_fallback_fps': _used_fallback_fps,
        'n_frames': n_frames,
        'active_paws': active_paws,
        'contact_masks': contact_masks,
        'paw_xy': paw_xy,
        'brightness_series': brightness_series,
        'paw_contour_data': paw_contour_data,
        'confidence_mask': confidence_mask,
        'loco_mask': loco_mask,
        'body_speed': body_speed,
        'frame_displacements': frame_displacements,
        'lick_mask': lick_mask,
        'params': params,
        '_mm_per_px': _mm_per_px,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: THE metrics implementation (one copy; verbatim ``_metrics`` +
# ``_gait_block`` closures + summary/bins assembly)
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(data, selection_masks, params, frame_slice=None,
                        log=None, pawlike_thresholds=None):
    """Compute {'summary': dict, 'bins': [dicts]} from loaded session data and
    precomputed selection masks.

    Parameters
    ----------
    data : dict — output of ``load_session_data`` (or the cached
        intermediates dict for the Adjust-Contact recompute path).
    selection_masks : (analyzed_mask, base_mask, four_mask) — output of
        ``compute_selection_masks``.
    params : dict — the 33-key params contract (bin_seconds/bin_unit drive
        the per-bin rows).
    frame_slice : slice or None — when given, only the summary over that
        slice is computed ('bins' comes back empty).
    """
    log = log or _noop_log
    if pawlike_thresholds is None:
        pawlike_thresholds = DEFAULT_PAWLIKE_THRESHOLDS

    contact_masks       = data['contact_masks']
    n_frames            = data['n_frames']
    fps                 = data['fps']
    _used_fallback_fps  = data['_used_fallback_fps']
    brightness_series   = data['brightness_series']
    paw_contour_data    = data['paw_contour_data']
    paw_xy              = data['paw_xy']
    confidence_mask     = data['confidence_mask']
    loco_mask           = data['loco_mask']
    body_speed          = data['body_speed']
    frame_displacements = data['frame_displacements']
    analyzed_mask, base_mask, four_mask = selection_masks

    # ── Metric computation helper ────────────────────────────────────────
    def _metrics(frame_slice=None):
        if frame_slice is not None:
            masks = {r: m.iloc[frame_slice].reset_index(drop=True)
                     for r, m in contact_masks.items()}
            n = len(next(iter(masks.values()))) if masks else 0
            _am = np.asarray(analyzed_mask[frame_slice], bool)
            _bm = np.asarray(base_mask[frame_slice], bool)
            _fm = (np.asarray(four_mask[frame_slice], bool)
                   if four_mask is not None else None)
        else:
            masks = contact_masks
            n = n_frames
            _am, _bm = analyzed_mask, base_mask
            _fm = four_mask
        _am = _am[:n]; _bm = _bm[:n]
        n_analyzed = int(_am.sum())

        m = {'n_frames': n, 'n_analyzed': n_analyzed,
             'analyzed_pct': round(n_analyzed / n * 100, 2) if n else float('nan'),
             'fps': round(fps, 2), 'fallback_fps_used': _used_fallback_fps}

        for role, mask in masks.items():
            marr = (mask.values.astype(bool) if hasattr(mask, 'values')
                    else np.asarray(mask, bool))[:n]
            inb = marr & _am
            m[f'contact_pct_{role}'] = (
                round(float(inb.sum()) / n_analyzed * 100, 2)
                if n_analyzed > 0 else float('nan'))
            m[f'n_contact_{role}'] = int(inb.sum())

        # Quadrupedal-stance fraction (over licking-excluded frames).
        if _fm is not None:
            _fm = _fm[:n]
            _bden = int(_bm.sum())
            m['quad_stance_pct'] = (
                round(float((_fm & _bm).sum()) / _bden * 100, 2)
                if _bden > 0 else float('nan'))

        # Hind WBI / SI
        if 'HL' in masks and 'HR' in masks:
            cHL, cHR = m['contact_pct_HL'], m['contact_pct_HR']
            tot = cHL + cHR
            m['WBI_hind'] = round(cHL / tot * 100, 2) if tot > 0 else float('nan')
            m['SI_hind']  = round((cHL - cHR) / tot * 100, 2) if tot > 0 else float('nan')

        # Fore WBI / SI
        if 'FL' in masks and 'FR' in masks:
            cFL, cFR = m['contact_pct_FL'], m['contact_pct_FR']
            tot = cFL + cFR
            m['WBI_fore'] = round(cFL / tot * 100, 2) if tot > 0 else float('nan')
            m['SI_fore']  = round((cFL - cFR) / tot * 100, 2) if tot > 0 else float('nan')

        # Per-paw WBI — each paw's fraction of total contact time
        total_contact = sum(m[f'contact_pct_{r}'] for r in masks)
        if total_contact > 0:
            for role in masks:
                m[f'WBI_{role}'] = round(
                    m[f'contact_pct_{role}'] / total_contact * 100, 2)

        # SBI (absolute symmetry index)
        if 'HL' in masks and 'HR' in masks:
            cHL, cHR = m['contact_pct_HL'], m['contact_pct_HR']
            tot = cHL + cHR
            m['SBI_hind'] = round(
                2 * abs(cHL - cHR) / tot * 100, 2) if tot > 0 else float('nan')
        if 'FL' in masks and 'FR' in masks:
            cFL, cFR = m['contact_pct_FL'], m['contact_pct_FR']
            tot = cFL + cFR
            m['SBI_fore'] = round(
                2 * abs(cFL - cFR) / tot * 100, 2) if tot > 0 else float('nan')

        # Hind / fore ratio
        if all(r in masks for r in ('HL', 'HR', 'FL', 'FR')):
            hind = (m['contact_pct_HL'] + m['contact_pct_HR']) / 2
            fore = (m['contact_pct_FL'] + m['contact_pct_FR']) / 2
            m['hind_fore_ratio'] = round(hind / fore, 4) if fore > 0 else float('nan')

        # Brightness during contact
        for role, brt_full in brightness_series.items():
            mask = masks.get(role)
            if mask is None:
                continue
            if frame_slice is not None:
                brt_slice = brt_full.iloc[frame_slice].reset_index(drop=True)
            else:
                brt_slice = brt_full
            _mb = mask.values.astype(bool)
            _L = min(len(_mb), len(brt_slice), len(_am))
            _sel = _mb[:_L] & _am[:_L]          # contact ∧ analyzed frames
            contact_brt = brt_slice.values[:_L][_sel]
            if len(contact_brt) > 0:
                m[f'brightness_{role}'] = round(float(np.nanmean(contact_brt)), 4)
            else:
                m[f'brightness_{role}'] = float('nan')

        if 'brightness_HL' in m and 'brightness_HR' in m:
            bHL = m['brightness_HL']
            bHR = m['brightness_HR']
            if (not (np.isnan(bHL) or np.isnan(bHR))) and bHR > 0:
                m['brightness_ratio_HL_HR'] = round(bHL / bHR, 4)

        # ── Distance & movement time ──────────────────────────────────
        if frame_displacements is not None:
            disp_sl = frame_displacements[frame_slice] if frame_slice is not None else frame_displacements
            m['total_distance'] = round(float(np.nansum(disp_sl)), 2)
            if loco_mask is not None:
                lm_sl = loco_mask[frame_slice] if frame_slice is not None else loco_mask
                lm_sl = lm_sl[:len(disp_sl)]
                m['loco_total_distance'] = round(float(np.nansum(disp_sl[lm_sl])), 2)
                m['time_moving_s'] = round(float(lm_sl.sum()) / fps, 2) if fps > 0 else 0.0
                m['time_moving_pct'] = round(float(lm_sl.mean()) * 100, 2)

        # ── Body speed mean ───────────────────────────────────────────
        if body_speed is not None:
            bs_sl = body_speed[frame_slice] if frame_slice is not None else body_speed
            m['body_speed_mean'] = round(float(np.nanmean(bs_sl)), 2)
            if loco_mask is not None:
                lm_sl = loco_mask[frame_slice] if frame_slice is not None else loco_mask
                lm_sl = lm_sl[:len(bs_sl)]
                loco_bs = bs_sl[lm_sl]
                m['body_speed_loco'] = round(float(np.nanmean(loco_bs)), 2) if len(loco_bs) > 0 else float('nan')

        # ── Support patterns (paw count per frame during locomotion) ──
        if len(masks) >= 4 and loco_mask is not None:
            lm_sl = loco_mask[frame_slice] if frame_slice is not None else loco_mask
            all_contact = np.zeros(n, dtype=int)
            for role_sp, mask_sp in masks.items():
                ma = mask_sp.values.astype(bool) if hasattr(mask_sp, 'values') else np.asarray(mask_sp, dtype=bool)
                all_contact[:len(ma)] += ma[:n].astype(int)
            lm_frames = lm_sl[:n]
            n_loco_frames = int(lm_frames.sum())
            if n_loco_frames > 0:
                for npaws in range(5):
                    cnt = int(((all_contact == npaws) & lm_frames).sum())
                    m[f'support_{npaws}paw_pct'] = round(cnt / n_loco_frames * 100, 2)

        # ── Gait metrics (dual: all frames + locomotion only) ─────────
        def _gait_block(m, masks, loco_filter_mask, prefix=''):
            """Compute gait timing, spatial, phase, symmetry metrics with optional prefix."""
            for role, mask in masks.items():
                mask_arr = mask.values.astype(bool) if hasattr(mask, 'values') else np.asarray(mask, dtype=bool)

                # Apply confidence and locomotion filters to gait computation
                gait_valid = np.ones(len(mask_arr), dtype=bool)
                if confidence_mask is not None:
                    cm_slice = confidence_mask[frame_slice] if frame_slice is not None else confidence_mask
                    if len(cm_slice) == len(mask_arr):
                        gait_valid &= cm_slice
                if loco_filter_mask is not None:
                    lm_slice = loco_filter_mask[frame_slice] if frame_slice is not None else loco_filter_mask
                    if len(lm_slice) == len(mask_arr):
                        gait_valid &= lm_slice

                # Mask out invalid frames for gait analysis
                gait_contact = mask_arr & gait_valid

                _min_stance = params.get('min_stance_ms', 0)
                stance_durs, swing_durs, stance_onsets = gait_bouts(
                    gait_contact, fps, min_stride_ms=_min_stance)

                if stance_durs:
                    m[f'{prefix}stance_dur_{role}'] = round(float(np.mean(stance_durs)), 4)
                    m[f'{prefix}n_strides_{role}'] = len(stance_durs)
                else:
                    m[f'{prefix}stance_dur_{role}'] = float('nan')
                    m[f'{prefix}n_strides_{role}'] = 0

                if swing_durs:
                    m[f'{prefix}swing_dur_{role}'] = round(float(np.mean(swing_durs)), 4)
                else:
                    m[f'{prefix}swing_dur_{role}'] = float('nan')

                # Stride = stance + swing
                if stance_durs and swing_durs:
                    stride_dur = m[f'{prefix}stance_dur_{role}'] + m[f'{prefix}swing_dur_{role}']
                    m[f'{prefix}stride_dur_{role}'] = round(stride_dur, 4)
                    m[f'{prefix}duty_cycle_{role}'] = round(
                        m[f'{prefix}stance_dur_{role}'] / stride_dur * 100, 2) if stride_dur > 0 else float('nan')
                    m[f'{prefix}cadence_{role}'] = round(
                        60.0 / stride_dur, 2) if stride_dur > 0 else float('nan')
                else:
                    m[f'{prefix}stride_dur_{role}'] = float('nan')
                    m[f'{prefix}duty_cycle_{role}'] = float('nan')
                    m[f'{prefix}cadence_{role}'] = float('nan')

                # Stride length (distance between consecutive foot-strikes)
                if role in paw_xy and len(stance_onsets) >= 2:
                    px_arr, py_arr = paw_xy[role]
                    if frame_slice is not None:
                        sl_start = frame_slice.start or 0
                        abs_onsets = [sl_start + o for o in stance_onsets
                                      if sl_start + o < len(px_arr)]
                    else:
                        abs_onsets = [o for o in stance_onsets if o < len(px_arr)]
                    if len(abs_onsets) >= 2:
                        dists = []
                        for j in range(1, len(abs_onsets)):
                            i0, i1 = abs_onsets[j - 1], abs_onsets[j]
                            d = np.sqrt((px_arr[i1] - px_arr[i0])**2 +
                                        (py_arr[i1] - py_arr[i0])**2)
                            dists.append(d)
                        m[f'{prefix}stride_len_{role}'] = round(float(np.mean(dists)), 2)
                    else:
                        m[f'{prefix}stride_len_{role}'] = float('nan')
                else:
                    m[f'{prefix}stride_len_{role}'] = float('nan')

                # Swing speed = stride_length / swing_duration
                sl_key = f'{prefix}stride_len_{role}'
                sw_key = f'{prefix}swing_dur_{role}'
                if (sl_key in m and sw_key in m
                        and not np.isnan(m[sl_key]) and not np.isnan(m[sw_key])
                        and m[sw_key] > 0):
                    m[f'{prefix}swing_speed_{role}'] = round(m[sl_key] / m[sw_key], 2)
                else:
                    m[f'{prefix}swing_speed_{role}'] = float('nan')

                # Stride-to-stride CV (coefficient of variation of stance durations)
                if len(stance_durs) >= 3:
                    sd_arr = np.array(stance_durs)
                    sd_mean = np.mean(sd_arr)
                    if sd_mean > 0:
                        m[f'{prefix}stride_cv_{role}'] = round(
                            float(np.std(sd_arr, ddof=1) / sd_mean * 100), 2)
                    else:
                        m[f'{prefix}stride_cv_{role}'] = float('nan')
                else:
                    m[f'{prefix}stride_cv_{role}'] = float('nan')

            # ── Step length / width (contralateral) ──────────────────
            for pair_name, left_role, right_role in [('hind', 'HL', 'HR'), ('fore', 'FL', 'FR')]:
                if left_role not in masks or right_role not in masks:
                    continue
                # Get stance onsets for both sides
                l_mask = masks[left_role].values.astype(bool) if hasattr(masks[left_role], 'values') else np.asarray(masks[left_role], dtype=bool)
                r_mask = masks[right_role].values.astype(bool) if hasattr(masks[right_role], 'values') else np.asarray(masks[right_role], dtype=bool)

                # Apply filters for loco variant
                if loco_filter_mask is not None:
                    lm_slice = loco_filter_mask[frame_slice] if frame_slice is not None else loco_filter_mask
                    l_mask = l_mask & lm_slice[:len(l_mask)]
                    r_mask = r_mask & lm_slice[:len(r_mask)]
                if confidence_mask is not None:
                    cm_slice = confidence_mask[frame_slice] if frame_slice is not None else confidence_mask
                    l_mask = l_mask & cm_slice[:len(l_mask)]
                    r_mask = r_mask & cm_slice[:len(r_mask)]

                _, _, l_onsets = gait_bouts(l_mask, fps, _min_stance)
                _, _, r_onsets = gait_bouts(r_mask, fps, _min_stance)

                if (left_role in paw_xy and right_role in paw_xy
                        and l_onsets and r_onsets):
                    lpx, lpy = paw_xy[left_role]
                    rpx, rpy = paw_xy[right_role]
                    sl_start = (frame_slice.start or 0) if frame_slice is not None else 0

                    # Step length: distance between contralateral strikes
                    all_strikes = sorted(
                        [('L', sl_start + o) for o in l_onsets] +
                        [('R', sl_start + o) for o in r_onsets],
                        key=lambda x: x[1])
                    step_lens = []
                    for j in range(1, len(all_strikes)):
                        if all_strikes[j][0] != all_strikes[j-1][0]:
                            i0 = all_strikes[j-1][1]
                            i1 = all_strikes[j][1]
                            if i0 < len(lpx) and i1 < len(lpx):
                                # Use position of the striking paw at its onset
                                if all_strikes[j-1][0] == 'L':
                                    x0, y0 = lpx[i0], lpy[i0]
                                else:
                                    x0, y0 = rpx[i0], rpy[i0]
                                if all_strikes[j][0] == 'L':
                                    x1, y1 = lpx[i1], lpy[i1]
                                else:
                                    x1, y1 = rpx[i1], rpy[i1]
                                step_lens.append(np.sqrt((x1 - x0)**2 + (y1 - y0)**2))
                    m[f'{prefix}step_len_{pair_name}'] = round(float(np.mean(step_lens)), 2) if step_lens else float('nan')

                    # Step width: lateral distance at mid-stance
                    widths = []
                    r_onsets_arr = np.array(r_onsets)
                    for lo in l_onsets:
                        abs_lo = sl_start + lo
                        # Find nearest right onset using searchsorted
                        idx = np.searchsorted(r_onsets_arr, lo)
                        candidates = []
                        if idx < len(r_onsets_arr):
                            candidates.append(r_onsets_arr[idx])
                        if idx > 0:
                            candidates.append(r_onsets_arr[idx - 1])
                        if candidates:
                            nearest_r = min(candidates, key=lambda ro: abs(ro - lo))
                            abs_r = sl_start + nearest_r
                            if abs_lo < len(lpx) and abs_r < len(rpx):
                                widths.append(abs(lpy[abs_lo] - rpy[abs_r]))
                    m[f'{prefix}step_width_{pair_name}'] = round(float(np.mean(widths)), 2) if widths else float('nan')
                else:
                    m[f'{prefix}step_len_{pair_name}'] = float('nan')
                    m[f'{prefix}step_width_{pair_name}'] = float('nan')

            # ── Interlimb coordination (phase) ───────────────────────
            if all(r in masks for r in ('HL', 'HR', 'FL', 'FR')):
                def _phase(ref_role, test_role):
                    ref_mask_arr = masks[ref_role].values.astype(bool) if hasattr(masks[ref_role], 'values') else np.asarray(masks[ref_role], dtype=bool)
                    tst_mask_arr = masks[test_role].values.astype(bool) if hasattr(masks[test_role], 'values') else np.asarray(masks[test_role], dtype=bool)

                    if loco_filter_mask is not None:
                        lm_slice = loco_filter_mask[frame_slice] if frame_slice is not None else loco_filter_mask
                        ref_mask_arr = ref_mask_arr & lm_slice[:len(ref_mask_arr)]
                        tst_mask_arr = tst_mask_arr & lm_slice[:len(tst_mask_arr)]
                    if confidence_mask is not None:
                        cm_slice = confidence_mask[frame_slice] if frame_slice is not None else confidence_mask
                        ref_mask_arr = ref_mask_arr & cm_slice[:len(ref_mask_arr)]
                        tst_mask_arr = tst_mask_arr & cm_slice[:len(tst_mask_arr)]

                    _, _, ref_on = gait_bouts(ref_mask_arr, fps, _min_stance)
                    _, _, tst_on = gait_bouts(tst_mask_arr, fps, _min_stance)
                    if len(ref_on) < 3 or not tst_on:
                        return float('nan')
                    phases = []
                    tst_arr = np.array(tst_on)
                    for i in range(len(ref_on) - 1):
                        cycle_len = ref_on[i+1] - ref_on[i]
                        if cycle_len <= 0:
                            continue
                        # Find test onsets within this cycle using searchsorted
                        lo = np.searchsorted(tst_arr, ref_on[i], side='left')
                        hi = np.searchsorted(tst_arr, ref_on[i+1], side='left')
                        for j in range(lo, hi):
                            phases.append((tst_arr[j] - ref_on[i]) / cycle_len)
                    return round(float(np.mean(phases)), 3) if phases else float('nan')

                m[f'{prefix}phase_HL_HR'] = _phase('HR', 'HL')
                m[f'{prefix}phase_diagonal'] = _phase('HR', 'FL')  # HR-FL diagonal pair
                m[f'{prefix}phase_FL_FR'] = _phase('FR', 'FL')
                m[f'{prefix}phase_HL_FL'] = _phase('FL', 'HL')
                m[f'{prefix}phase_HR_FR'] = _phase('FR', 'HR')
                m[f'{prefix}phase_HL_FR'] = _phase('FR', 'HL')

            # ── Regularity index ──────────────────────────────────────
            if all(r in masks for r in ('HL', 'HR', 'FL', 'FR')):
                ri = regularity_index(masks, fps, frame_slice,
                                      loco_filter_mask, confidence_mask,
                                      params.get('min_stance_ms', 0))
                if ri is not None:
                    m[f'{prefix}regularity_index'] = ri

            # ── Print position (hind-fore overlap) ────────────────────
            if all(r in masks for r in ('HL', 'HR', 'FL', 'FR')):
                for side_lbl, hind_r, fore_r in [('L', 'HL', 'FL'), ('R', 'HR', 'FR')]:
                    if hind_r in paw_xy and fore_r in paw_xy:
                        pp = print_position(
                            masks, paw_xy, hind_r, fore_r, fps, frame_slice,
                            loco_filter_mask, confidence_mask,
                            params.get('min_stance_ms', 0))
                        m[f'{prefix}print_position_{side_lbl}'] = pp

            # ── Gait symmetry indices ────────────────────────────────
            if f'{prefix}stance_dur_HL' in m and f'{prefix}stance_dur_HR' in m:
                sHL = m[f'{prefix}stance_dur_HL']
                sHR = m[f'{prefix}stance_dur_HR']
                tot = sHL + sHR
                m[f'{prefix}stance_SI_hind'] = round((sHL - sHR) / tot * 100, 2) if tot > 0 and not (np.isnan(sHL) or np.isnan(sHR)) else float('nan')
            if f'{prefix}stride_len_HL' in m and f'{prefix}stride_len_HR' in m:
                lHL = m[f'{prefix}stride_len_HL']
                lHR = m[f'{prefix}stride_len_HR']
                tot = lHL + lHR
                m[f'{prefix}stride_len_SI_hind'] = round((lHL - lHR) / tot * 100, 2) if tot > 0 and not (np.isnan(lHL) or np.isnan(lHR)) else float('nan')

        _gait_block(m, masks, None, prefix='')           # All frames
        if loco_mask is not None:
            _gait_block(m, masks, loco_mask, prefix='loco_')  # Locomotion only

        # ── Paw contour area (Step 5) ────────────────────────────────────
        if paw_contour_data:
            # Build full-stance mask (all active paws in contact)
            stance_mask_all = None
            if masks:
                _contour_roles = [r for r in paw_contour_data if r in masks]
                stance_arrays = [masks[r].values.astype(bool) if hasattr(masks[r], 'values')
                                 else np.asarray(masks[r], dtype=bool) for r in _contour_roles]
                if stance_arrays:
                    stance_mask_all = stance_arrays[0].copy()
                    for _sm in stance_arrays[1:]:
                        _sml = min(len(stance_mask_all), len(_sm))
                        stance_mask_all = stance_mask_all[:_sml] & _sm[:_sml]

            for role in list(paw_contour_data.keys()):
                areas_full = paw_contour_data[role]['areas']
                spreads_full = paw_contour_data[role]['spreads']
                intensities_full = paw_contour_data[role]['intensities']
                mask_arr = masks[role].values.astype(bool) if role in masks else np.ones(n_frames, dtype=bool)
                sl = frame_slice
                areas_sl = areas_full[sl] if sl is not None else areas_full
                spreads_sl = spreads_full[sl] if sl is not None else spreads_full
                ints_sl = intensities_full[sl] if sl is not None else intensities_full
                mask_sl = mask_arr

                widths_full = paw_contour_data[role].get('widths')
                solidities_full = paw_contour_data[role].get('solidities')
                aspect_ratios_full = paw_contour_data[role].get('aspect_ratios')
                circularities_full = paw_contour_data[role].get('circularities')
                widths_sl = (widths_full[sl] if sl is not None else widths_full) if widths_full is not None else None
                solidities_sl = (solidities_full[sl] if sl is not None else solidities_full) if solidities_full is not None else None
                ar_sl = (aspect_ratios_full[sl] if sl is not None else aspect_ratios_full) if aspect_ratios_full is not None else None
                circ_sl = (circularities_full[sl] if sl is not None else circularities_full) if circularities_full is not None else None

                # ── Regular (per-paw contact mask) ──
                valid = mask_sl & (areas_sl[:len(mask_sl)] > 0)
                _ca = areas_sl[:len(mask_sl)][valid]
                m[f'paw_area_{role}'] = round(float(np.nanmean(_ca)), 2) if len(_ca) > 0 else float('nan')
                _cs = spreads_sl[:len(mask_sl)][valid]
                m[f'paw_spread_{role}'] = round(float(np.nanmean(_cs)), 2) if len(_cs) > 0 else float('nan')
                _ci = ints_sl[:len(mask_sl)][valid]
                m[f'contact_intensity_{role}'] = round(float(np.nanmean(_ci)), 2) if len(_ci) > 0 else float('nan')
                if widths_sl is not None:
                    m[f'paw_width_{role}'] = round(float(np.nanmean(widths_sl[:len(mask_sl)][valid])), 2) if valid.any() else float('nan')
                    m[f'paw_solidity_{role}'] = round(float(np.nanmean(solidities_sl[:len(mask_sl)][valid])), 4) if valid.any() else float('nan')
                    m[f'paw_aspect_ratio_{role}'] = round(float(np.nanmean(ar_sl[:len(mask_sl)][valid])), 4) if valid.any() else float('nan')
                    m[f'paw_circularity_{role}'] = round(float(np.nanmean(circ_sl[:len(mask_sl)][valid])), 4) if valid.any() else float('nan')

                # ── Paw-like filtered (solidity + AR + circularity) ──
                PAWLIKE_SOL = pawlike_thresholds.get('solidity', 1.00)
                PAWLIKE_AR = pawlike_thresholds.get('aspect_ratio', 1.6)
                PAWLIKE_CIRC = pawlike_thresholds.get('circularity', 0.10)
                if solidities_sl is not None:
                    sol_arr = solidities_sl[:len(mask_sl)]
                    valid_paw = valid & (sol_arr <= PAWLIKE_SOL)
                    if ar_sl is not None:
                        valid_paw = valid_paw & (ar_sl[:len(mask_sl)] <= PAWLIKE_AR)
                    if circ_sl is not None:
                        valid_paw = valid_paw & (circ_sl[:len(mask_sl)] >= PAWLIKE_CIRC)
                    _pca = areas_sl[:len(mask_sl)][valid_paw]
                    m[f'pawlike_area_{role}'] = round(float(np.nanmean(_pca)), 2) if len(_pca) > 0 else float('nan')
                    _pcs = spreads_sl[:len(mask_sl)][valid_paw]
                    m[f'pawlike_spread_{role}'] = round(float(np.nanmean(_pcs)), 2) if len(_pcs) > 0 else float('nan')
                    _pci = ints_sl[:len(mask_sl)][valid_paw]
                    m[f'pawlike_intensity_{role}'] = round(float(np.nanmean(_pci)), 2) if len(_pci) > 0 else float('nan')
                    if widths_sl is not None:
                        m[f'pawlike_width_{role}'] = round(float(np.nanmean(widths_sl[:len(mask_sl)][valid_paw])), 2) if valid_paw.any() else float('nan')
                        m[f'pawlike_solidity_{role}'] = round(float(np.nanmean(sol_arr[valid_paw])), 4) if valid_paw.any() else float('nan')
                        m[f'pawlike_aspect_ratio_{role}'] = round(float(np.nanmean(ar_sl[:len(mask_sl)][valid_paw])), 4) if valid_paw.any() else float('nan')
                        m[f'pawlike_circularity_{role}'] = round(float(np.nanmean(circ_sl[:len(mask_sl)][valid_paw])), 4) if valid_paw.any() else float('nan')

                # ── Full-stance variant (all contour paws in contact) ──
                if stance_mask_all is not None:
                    stance_sl = stance_mask_all[sl] if sl is not None else stance_mask_all
                    _ml = min(len(mask_sl), len(stance_sl))
                    stance_valid = mask_sl[:_ml] & stance_sl[:_ml] & (areas_sl[:_ml] > 0)
                    _sca = areas_sl[:_ml][stance_valid]
                    m[f'paw_area_stance_{role}'] = round(float(np.nanmean(_sca)), 2) if len(_sca) > 0 else float('nan')
                    _scs = spreads_sl[:_ml][stance_valid]
                    m[f'paw_spread_stance_{role}'] = round(float(np.nanmean(_scs)), 2) if len(_scs) > 0 else float('nan')
                    _sci = ints_sl[:_ml][stance_valid]
                    m[f'contact_intensity_stance_{role}'] = round(float(np.nanmean(_sci)), 2) if len(_sci) > 0 else float('nan')
                    if widths_sl is not None:
                        m[f'paw_width_stance_{role}'] = round(float(np.nanmean(widths_sl[:_ml][stance_valid])), 2) if stance_valid.any() else float('nan')
                        m[f'paw_solidity_stance_{role}'] = round(float(np.nanmean(solidities_sl[:_ml][stance_valid])), 4) if stance_valid.any() else float('nan')
                        m[f'paw_aspect_ratio_stance_{role}'] = round(float(np.nanmean(ar_sl[:_ml][stance_valid])), 4) if stance_valid.any() else float('nan')
                        m[f'paw_circularity_stance_{role}'] = round(float(np.nanmean(circ_sl[:_ml][stance_valid])), 4) if stance_valid.any() else float('nan')

            # Area ratios (regular)
            if 'paw_area_HL' in m and 'paw_area_HR' in m:
                aHL, aHR = m['paw_area_HL'], m['paw_area_HR']
                if not (np.isnan(aHL) or np.isnan(aHR)) and aHR > 0:
                    m['paw_area_ratio_hind'] = round(aHL / aHR, 4)
            # Area ratios (stance)
            if 'paw_area_stance_HL' in m and 'paw_area_stance_HR' in m:
                aHL_s, aHR_s = m['paw_area_stance_HL'], m['paw_area_stance_HR']
                if not (np.isnan(aHL_s) or np.isnan(aHR_s)) and aHR_s > 0:
                    m['paw_area_ratio_stance_hind'] = round(aHL_s / aHR_s, 4)
            # Intensity ratios (regular)
            if 'contact_intensity_HL' in m and 'contact_intensity_HR' in m:
                iHL, iHR = m['contact_intensity_HL'], m['contact_intensity_HR']
                if not (np.isnan(iHL) or np.isnan(iHR)) and iHR > 0:
                    m['contact_intensity_ratio_hind'] = round(iHL / iHR, 4)
            # Intensity ratios (stance)
            if 'contact_intensity_stance_HL' in m and 'contact_intensity_stance_HR' in m:
                iHL_s, iHR_s = m['contact_intensity_stance_HL'], m['contact_intensity_stance_HR']
                if not (np.isnan(iHL_s) or np.isnan(iHR_s)) and iHR_s > 0:
                    m['contact_intensity_ratio_stance_hind'] = round(iHL_s / iHR_s, 4)
            # Area ratios (paw-like)
            if 'pawlike_area_HL' in m and 'pawlike_area_HR' in m:
                aHL_p, aHR_p = m['pawlike_area_HL'], m['pawlike_area_HR']
                if not (np.isnan(aHL_p) or np.isnan(aHR_p)) and aHR_p > 0:
                    m['pawlike_area_ratio_hind'] = round(aHL_p / aHR_p, 4)
            # Intensity ratios (paw-like)
            if 'pawlike_intensity_HL' in m and 'pawlike_intensity_HR' in m:
                iHL_p, iHR_p = m['pawlike_intensity_HL'], m['pawlike_intensity_HR']
                if not (np.isnan(iHL_p) or np.isnan(iHR_p)) and iHR_p > 0:
                    m['pawlike_intensity_ratio_hind'] = round(iHL_p / iHR_p, 4)

        return m

    if frame_slice is not None:
        return {'summary': _metrics(frame_slice), 'bins': []}

    # ── Overall summary ───────────────────────────────────────────────────
    log("  Computing metrics…")
    summary = _metrics()
    log("  Summary metrics done.")

    # ── Per-bin ───────────────────────────────────────────────────────────
    bin_rows = []
    bin_val  = params['bin_seconds']
    bin_unit = params.get('bin_unit', 'seconds')
    bin_sec  = bin_val * 60 if bin_unit == 'minutes' else bin_val
    if bin_sec > 0:
        bin_frames = max(1, round(bin_sec * fps))
        n_bins = n_frames // bin_frames
        for i in range(n_bins):
            start = i * bin_frames
            end   = min(start + bin_frames, n_frames)
            row = _metrics(slice(start, end))
            row['bin_index']   = i
            row['bin_start_s'] = round(start / fps, 2) if fps > 0 else start
            row['bin_end_s']   = round(end   / fps, 2) if fps > 0 else end
            bin_rows.append(row)
        log(f"  {n_bins} time bins computed.")

    return {'summary': summary, 'bins': bin_rows}


# ─────────────────────────────────────────────────────────────────────────────
# Per-session analysis (stage 1 → 2 → 3) and the run loop
# ─────────────────────────────────────────────────────────────────────────────

def analyze_session(sess, paw_map, params, ctx,
                    log=None, progress_cb=None, cancel=None):
    """
    Returns {'summary': dict, 'bins': list_of_dicts, 'intermediates': dict},
    or None on failure.

    'intermediates' is the per-session data dict the old tab kept in
    ``_session_intermediates`` (height_df, bp_xcord/ycord/prob, fps,
    _used_fallback_fps, n_frames, active_paws, contact_masks, paw_xy,
    brightness_series, paw_contour_data — incl. contour_shapes /
    contour_solidities for the contour graph tabs —, confidence_mask,
    loco_mask, body_speed, frame_displacements, lick_mask, params), plus
    'session_name' and '_mm_per_px'. The graph window's contour tabs and the
    Adjust-Contact dialog (``recompute_with_contact``) consume it.
    """
    log = log or _noop_log
    data = load_session_data(sess, paw_map, params, ctx,
                             log=log, progress_cb=progress_cb, cancel=cancel)
    if data is None:
        return None

    n_frames = data['n_frames']
    analyzed_mask, base_mask, four_mask = compute_selection_masks(
        data['contact_masks'], data['lick_mask'],
        bool(params.get('gate_4paw')), n_frames)
    if params.get('gate_4paw'):
        if four_mask is not None:
            log(f"  4-paw gate: {int(four_mask.sum())}/{n_frames} "
                f"quadrupedal-stance frames (contact-% WBI ~50; use "
                f"brightness / contour-area asymmetry).")
        else:
            log("  4-paw gate requested but fore paws are unavailable; "
                "gate skipped.")

    res = compute_all_metrics(data, (analyzed_mask, base_mask, four_mask),
                              params, log=log,
                              pawlike_thresholds=ctx.pawlike_thresholds)
    return {'summary': res['summary'], 'bins': res['bins'],
            'intermediates': data}


def recompute_with_contact(cached_data, new_params, ctx, log=None):
    """Adjust-Contact path: rebuild contact masks from cached arrays with new
    contact parameters, rebuild the selection masks (INCLUDING the licking
    exclusion and the 4-paw gate — the old ``_recompute_contact`` duplicate
    silently dropped both, producing wrong denominators), then run the SAME
    ``compute_all_metrics``.

    ``cached_data`` is the 'intermediates' dict from ``analyze_session``.
    ``new_params`` typically carries contact_method / contact_threshold /
    speed_threshold / median_filter_ms / min_bout_ms / brt_weight (the
    Adjust-Contact dialog's fields); it is merged over the run's params.

    Returns {'summary': dict, 'bins': list_of_dicts}. ``cached_data`` is
    updated in place with the new 'contact_masks' and merged 'params'
    (mirroring the old dialog, which updated the stored intermediates).
    """
    log = log or _noop_log

    # Parse speed threshold (verbatim tolerance for str/float input)
    if 'speed_threshold' in new_params:
        speed_thresh_raw = new_params.get('speed_threshold', 'auto')
        if isinstance(speed_thresh_raw, str) and speed_thresh_raw.lower() != 'auto':
            try:
                speed_thresh = float(speed_thresh_raw)
            except ValueError:
                speed_thresh = 'auto'
        else:
            speed_thresh = 'auto' if speed_thresh_raw == 'auto' else speed_thresh_raw
        new_params = {**new_params, 'speed_threshold': speed_thresh}

    params = {**cached_data['params'], **new_params}

    n_frames    = cached_data['n_frames']
    fps         = cached_data['fps']
    active_paws = cached_data['active_paws']
    height_df   = cached_data['height_df']
    paw_xy      = cached_data['paw_xy']

    contact_masks = build_contact_masks(
        active_paws, height_df, paw_xy, n_frames, fps, params,
        mm_per_px=cached_data.get('_mm_per_px'))
    contact_masks = apply_brightness_weight(
        contact_masks, active_paws, height_df,
        cached_data['brightness_series'], n_frames, params, log=log)
    contact_masks = apply_contour_area_contact(
        contact_masks, cached_data['paw_contour_data'], n_frames, params,
        log=log)

    # Update stored intermediates with new contact masks (as the old dialog did)
    cached_data['contact_masks'] = contact_masks
    cached_data['params'] = params

    selection_masks = compute_selection_masks(
        contact_masks, cached_data.get('lick_mask'),
        bool(params.get('gate_4paw')), n_frames)

    return compute_all_metrics(cached_data, selection_masks, params, log=log,
                               pawlike_thresholds=ctx.pawlike_thresholds)


def run_sessions(sessions, paw_map, params, key_df, ctx,
                 log=None, progress_cb=None, cancel=None, strip_prefix=''):
    """Port of ``_analysis_thread``'s loop (minus all widget calls).

    Returns (summary_rows, bin_rows, not_done) where each summary/bin row is
    the session's metrics dict merged over
    dict(session=…, subject=…, treatment=…).
    """
    log = log or _noop_log
    summary_rows = []
    bin_rows = []
    not_done = []   # sessions that raised or produced no result (skipped/failed)
    for i_sess, sess in enumerate(sessions):
        if cancel is not None and cancel.is_set():
            log("Cancelled.")
            break
        name = sess['session_name']
        log(f"Processing: {name}")
        try:
            result = analyze_session(sess, paw_map, params, ctx,
                                     log=log, progress_cb=progress_cb,
                                     cancel=cancel)
            if result:
                subj      = resolve_subject(name, key_df, strip_prefix)
                treatment = get_treatment(subj, key_df)
                base = dict(session=name, subject=subj, treatment=treatment)
                srow = {**base, **result['summary']}
                summary_rows.append(srow)
                for brow in result['bins']:
                    bin_rows.append({**base, **brow})
            else:
                not_done.append(name)
                log(f"  Skipped: {name} (no result)")
        except Exception as e:
            log(f"  ERROR: {e}")
            not_done.append(name)
        if progress_cb is not None:
            try:
                progress_cb(i_sess + 1, len(sessions), name)
            except Exception:
                pass

    log(f"Analysis loop finished: {len(summary_rows)}/{len(sessions)} sessions produced results.")
    return summary_rows, bin_rows, not_done
