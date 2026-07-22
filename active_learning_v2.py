"""
active_learning_v2.py — PixelPaws Active Learning v2
=====================================================
Inspired by A-SOiD (Tillmann et al., Nature Methods 2024).

New features vs v1:
- Confidence histogram inspection with adjustable threshold
- Learning curve tracking (train F1 + CV F1) with JSON persistence
- Auto-convergence detection
- Temporal label propagation (cosine similarity)
- Post-AL sub-behavior discovery (UMAP + HDBSCAN)
"""

import os
import json
import pickle
import threading
import traceback
import time
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import messagebox, scrolledtext
import cv2

try:
    import ttkbootstrap as ttk
    _TTKBOOTSTRAP = True
except ImportError:
    from tkinter import ttk
    _TTKBOOTSTRAP = False

# Optional matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# Canonical feature-schema hash for this pipeline (the 635-col 8aed1c22 set).
# When a session has several `*_features_<hash>.pkl` variants, the cache
# loader prefers this one so every session contributes the SAME column set —
# mixing schemas silently breaks the cross-session concat. Projects whose
# canonical schema differs simply won't have this file and fall back to the
# first available variant (no regression).
PREFERRED_FEATURE_HASH = '8aed1c22'


def _trim_to_positive_span(labels: np.ndarray) -> np.ndarray:
    """Boolean keep-mask that drops frames BEFORE the first positive and AFTER
    the last positive — mirrors the Train tab's trim_to_first/last_positive
    (PixelPaws_GUI.py:6600-6626), which keeps BORIS leading/trailing zeros out of
    the TRAINING set. Operates on the full label array (values -1/0/1); frames
    inside the span keep their own label. No positives → keep everything (the
    caller raises a clear 'no positives' error downstream).

    NOTE for AL: unlike the Train tab (where trailing 0s are *unreviewed*), an AL
    `0` is a *reviewed* negative, so this can drop deliberately-labeled negatives
    outside the positive span. Callers log how many labeled frames are dropped.
    """
    keep = np.ones(len(labels), dtype=bool)
    pos = np.where(labels == 1)[0]
    if len(pos) > 0:
        keep[:pos[0]] = False
        keep[pos[-1] + 1:] = False
    return keep


def _robust_unpickle(path):
    """Load a .pkl that may be joblib+LZ4 OR plain pickle.

    The canonical 8aed1c22 feature caches and the encyclopedia classifiers are
    written with joblib + LZ4 compression; project-local GUI artifacts are plain
    pickle. ``joblib.load`` transparently reads both, so it is the universal
    loader. Falls back to ``pickle`` only if joblib is unavailable/erroring.
    """
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with open(path, 'rb') as f:
            return pickle.load(f)


def _align_features(model, features: np.ndarray, feature_cols) -> np.ndarray:
    """Reindex numpy feature array to match model.feature_names_in_.
    - If model has no feature_names_in_ or feature_cols is None: return as-is.
    - Already-aligned: return as-is (fast path).
    - Missing columns filled with 0.0; extra columns dropped.
    """
    if not hasattr(model, 'feature_names_in_') or feature_cols is None:
        return features
    model_cols = list(model.feature_names_in_)
    fc = list(feature_cols)
    if fc == model_cols:
        return features
    missing = [c for c in model_cols if c not in set(fc)]
    if missing:
        msg = (f"_align_features: {len(missing)}/{len(model_cols)} model features missing from the "
               f"provided columns (e.g. {missing[:5]}) — 0-filling. This usually means a feature-schema "
               f"mismatch (wrong cache/body parts/versions).")
        print("[_align_features] WARNING: " + msg, flush=True)
        import warnings as _w
        _w.warn(msg)
        # Hard-fail when the gap is large: predicting on a mostly-zero feature set is garbage.
        if len(missing) > max(5, int(0.10 * len(model_cols))):
            raise ValueError(msg + " Refusing to predict on a badly misaligned feature set.")
    df = pd.DataFrame(features, columns=fc)
    return df.reindex(columns=model_cols, fill_value=0.0).values.astype(np.float32)


from ui_utils import ToolTip, _bind_tight_layout_on_resize, FONT_FAMILY, METRICS_HELP
from dialogs import ConfidenceHistogramDialog
from io_utils import atomic_pickle_save, atomic_dataframe_to_csv
# Optional UMAP + HDBSCAN
try:
    import umap
    import hdbscan
    from sklearn.preprocessing import StandardScaler
    UMAP_HDBSCAN_AVAILABLE = True
except ImportError:
    UMAP_HDBSCAN_AVAILABLE = False

# Optional sklearn metrics
try:
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

def _make_bout_groups(frame_indices, min_bout=1, session_ids=None):
    """
    Assign group IDs to labeled frame indices so that a new group
    starts whenever consecutive frames are more than min_bout apart,
    OR whenever the session changes (when session_ids is provided).
    Forcing a break at session boundaries prevents cross-session bouts
    from being merged into one GroupKFold group (label leakage).
    Returns int array same length as frame_indices.
    """
    if len(frame_indices) == 0:
        return np.array([], dtype=int)
    groups = np.zeros(len(frame_indices), dtype=int)
    gid = 0
    for i in range(1, len(frame_indices)):
        new_session = session_ids is not None and session_ids[i] != session_ids[i - 1]
        if new_session or frame_indices[i] - frame_indices[i - 1] > min_bout:
            gid += 1
        groups[i] = gid
    return groups


def _cv_oof(X, y, n_folds=5, session_ids=None, bout_groups=None,
           bout_aware=True, n_estimators=200, seed=42, progress_cb=None,
           eligible_sessions=None):
    """Shared CV-OOF for AL parity with the Train tab.

    Returns ``(oof_proba, fold_f1s, mode, fold_of)`` where ``oof_proba`` is full
    length (``len(y)``) with each frame's prediction scattered back to its position
    (so downstream sweeps see honest ordered probs), and ``fold_of[i]`` is the
    validation-fold index that produced frame ``i``'s OOF prediction (-1 if none).
    ``fold_of`` enables leave-one-fold-out (nested) threshold selection downstream.

    Grouping priority (matches the agreed AL↔Train-tab policy):
      1. **session-level** GroupKFold when ``session_ids`` has ≥ ``n_folds`` unique
         sessions (same as the Train tab's session-level CV),
      2. **bout-level** GroupKFold when ``bout_aware`` and ≥ ``n_folds`` bout groups,
      3. **frame-level** StratifiedKFold fallback (few sessions/bouts — common early
         in AL). No 500-frame subsample (the Train tab does not subsample).
    """
    from sklearn.model_selection import GroupKFold, StratifiedKFold
    from xgboost import XGBClassifier
    y = np.asarray(y)
    X = np.asarray(X)
    n = len(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    spw = float(n_neg / max(n_pos, 1))
    oof = np.full(n, 0.5, dtype=float)
    fold_of = np.full(n, -1, dtype=int)

    groups, mode = None, 'frame'
    if session_ids is not None and len(np.unique(session_ids)) >= n_folds:
        groups, mode = np.asarray(session_ids), 'session'
    elif (bout_aware and bout_groups is not None and len(bout_groups) > 0
          and int(np.max(bout_groups)) + 1 >= n_folds):
        groups, mode = np.asarray(bout_groups), 'bout'

    n_splits = min(n_folds, n_pos, n_neg)
    if groups is not None:
        n_splits = min(n_splits, len(np.unique(groups)))

    def _mk():
        return XGBClassifier(n_estimators=n_estimators, max_depth=6, learning_rate=0.1,
                             scale_pos_weight=spw, random_state=seed, verbosity=0)

    from sklearn.metrics import precision_score, recall_score

    if n_splits < 2:
        clf = _mk(); clf.fit(X, y)
        oof = clf.predict_proba(X)[:, 1]
        fold_of[:] = 0   # single resubstitution "fold" (no honest CV possible)
        _pred = (oof >= 0.5).astype(int)
        _det = [{'f1': float(f1_score(y, _pred, zero_division=0)),
                 'precision': float(precision_score(y, _pred, zero_division=0)),
                 'recall': float(recall_score(y, _pred, zero_division=0)),
                 'test_groups': [], 'n_test': int(len(y)), 'n_pos': int(np.sum(y))}]
        return oof, [_det[0]['f1']], 'no-CV', fold_of, _det

    # Sparse sessions → training-only: fold ONLY over eligible sessions; each fold
    # still trains on everything except its held-out sessions (so train-only sessions
    # are in every fold's training set, never validated). Session-level mode only.
    _use_elig = (mode == 'session' and eligible_sessions is not None
                 and not set(np.unique(session_ids).tolist()).issubset(set(eligible_sessions)))
    if _use_elig:
        sid_arr = np.asarray(session_ids)
        elig_mask = np.isin(sid_arr, list(eligible_sessions))
        elig_idx = np.where(elig_mask)[0]
        n_splits = max(2, min(n_splits, len(np.unique(sid_arr[elig_idx]))))
        splits = []
        for (_tr, _va) in GroupKFold(n_splits=n_splits).split(
                elig_idx, y[elig_idx], groups=sid_arr[elig_idx]):
            vidx = elig_idx[_va]
            held = set(np.unique(sid_arr[vidx]).tolist())
            tidx = np.where(~np.isin(sid_arr, list(held)))[0]
            splits.append((tidx, vidx))
    elif groups is not None:
        splits = list(GroupKFold(n_splits=n_splits).split(X, y, groups=groups))
    else:
        splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True,
                                      random_state=seed).split(X, y))
    fold_f1s = []
    fold_details = []
    for fi, (tr, val) in enumerate(splits):
        clf = _mk(); clf.fit(X[tr], y[tr])
        pv = clf.predict_proba(X[val])[:, 1]
        oof[val] = pv
        fold_of[val] = fi
        pred = (pv >= 0.5).astype(int)
        f1 = float(f1_score(y[val], pred, zero_division=0))
        fold_f1s.append(f1)
        tg = sorted(set(int(g) for g in groups[val])) if groups is not None else []
        fold_details.append({
            'f1': f1,
            'precision': float(precision_score(y[val], pred, zero_division=0)),
            'recall': float(recall_score(y[val], pred, zero_division=0)),
            'test_groups': tg,
            'n_test': int(len(val)),
            'n_pos': int(np.sum(y[val])),
        })
        if progress_cb:
            try:
                # 4-arg form streams the just-finished fold's detail + CV mode so the
                # GUI can log/plot per-fold metrics live; tolerate a 2-arg callback.
                try:
                    progress_cb(fi + 1, len(splits), fold_details[-1], mode)
                except TypeError:
                    progress_cb(fi + 1, len(splits))
            except Exception:
                pass
    return oof, fold_f1s, mode, fold_of, fold_details


def _best_global_threshold(oof_proba, y, thresholds=None):
    """Threshold maximizing F1 on the pooled OOF (optimistic — for auxiliary
    precision/recall display and the shipped operating point, NOT the honest F1)."""
    y = np.asarray(y).astype(int)
    oof_proba = np.asarray(oof_proba, dtype=float)
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)
    best_t, best_f = 0.5, -1.0
    for t in thresholds:
        f = f1_score(y, (oof_proba >= float(t)).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t


def _honest_oof_f1(oof_proba, y, fold_of, thresholds=None):
    """Leave-one-fold-out (nested) threshold selection → leak-free CV F1.

    For each fold *f*, pick the F1-maximizing threshold on the OOF predictions of
    ALL OTHER folds, then apply it to fold *f*'s own frames. A fold never informs
    its own operating point, so the pooled F1 is free of the threshold-selection-
    on-the-test-set leakage that ``_sweep_postprocessing`` (global sweep on the
    same OOF) carries. Returns ``(honest_f1, per_fold_f1s, mean, std)``.
    """
    y = np.asarray(y).astype(int)
    oof_proba = np.asarray(oof_proba, dtype=float)
    fold_of = np.asarray(fold_of)
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)
    valid = fold_of >= 0
    folds = list(np.unique(fold_of[valid])) if valid.any() else []

    def _best_t(mask):
        if not mask.any():
            return 0.5
        yy, pp = y[mask], oof_proba[mask]
        best_t, best_f = 0.5, -1.0
        for t in thresholds:
            f = f1_score(yy, (pp >= float(t)).astype(int), zero_division=0)
            if f > best_f:
                best_f, best_t = f, float(t)
        return best_t

    if len(folds) < 2:
        # Can't leave a fold out — honest predictions, single global threshold.
        t = _best_t(valid)
        f1 = float(f1_score(y[valid], (oof_proba[valid] >= t).astype(int), zero_division=0)) \
            if valid.any() else 0.0
        return f1, [f1], f1, 0.0

    preds = np.zeros(len(y), dtype=int)
    per_fold = []
    for f in folds:
        this = (fold_of == f)
        others = valid & (fold_of != f)
        t = _best_t(others)                        # threshold from the OTHER folds only
        preds[this] = (oof_proba[this] >= t).astype(int)
        per_fold.append(float(f1_score(y[this], preds[this], zero_division=0)))
    honest = float(f1_score(y[valid], preds[valid], zero_division=0))
    per_fold = np.asarray(per_fold, dtype=float)
    return honest, [float(v) for v in per_fold], float(per_fold.mean()), float(per_fold.std())


def _fallback_sweep(oof_proba, y):
    """Threshold F1 sweep — fallback when the GUI's _sweep_postprocessing is
    unavailable (e.g. headless engine). Returns the same keys the caller expects."""
    import numpy as _np
    from sklearn.metrics import f1_score as _f1s
    y = _np.asarray(y).astype(int)
    best = {'thresh': 0.5, 'min_bout': 1, 'min_after_bout': 1, 'max_gap': 0, 'f1': 0.0}
    for t in _np.linspace(0.05, 0.95, 19):
        f = float(_f1s(y, (oof_proba >= float(t)).astype(int), zero_division=0))
        if f > best['f1']:
            best = {'thresh': float(t), 'min_bout': 1, 'min_after_bout': 1, 'max_gap': 0, 'f1': f}
    return best


def _fast_f1_binary(y_true, y_pred):
    """Inline binary F1 (pos_label=1, zero_division=0) — numerically identical to
    sklearn.metrics.f1_score on int arrays but ~10-50× faster per call (no input
    validation / label-binarization). Used to make the 8,288-combo sweep fast."""
    yt = y_true.astype(bool, copy=False)
    yp = y_pred.astype(bool, copy=False)
    tp = int(np.count_nonzero(yp & yt))
    fp = int(np.count_nonzero(yp & ~yt))
    fn = int(np.count_nonzero(~yp & yt))
    denom = 2 * tp + fp + fn
    return (2.0 * tp / denom) if denom > 0 else 0.0


def _session_slices(session_ids):
    """Contiguous ``[start, end)`` index runs of equal session id. Assumes each
    session's labeled frames form one contiguous, time-ordered block (as the retrain
    builds them); a session split into non-adjacent blocks is treated as separate runs,
    which is the correct unit for temporal bout filtering. ``None`` → ``None``."""
    if session_ids is None:
        return None
    sid = np.asarray(session_ids)
    if len(sid) == 0:
        return []
    change = np.where(sid[1:] != sid[:-1])[0] + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(sid)]])
    return list(zip(starts.tolist(), ends.tolist()))


def _apply_bouts_per_session(y_raw, session_ids, min_bout, min_after_bout, max_gap):
    """Apply ``_apply_bout_filtering`` WITHIN each session's contiguous block so bouts
    never bridge / get removed across session boundaries. ``session_ids=None`` falls back
    to whole-array filtering (Train-tab parity)."""
    from evaluation_tab import _apply_bout_filtering
    yr = np.asarray(y_raw)
    if session_ids is None:
        return _apply_bout_filtering(yr.copy(), min_bout=min_bout,
                                     min_after_bout=min_after_bout, max_gap=max_gap)
    out = yr.copy()
    for (s, e) in _session_slices(session_ids):
        out[s:e] = _apply_bout_filtering(out[s:e].copy(), min_bout=min_bout,
                                         min_after_bout=min_after_bout, max_gap=max_gap)
    return out


def exclude_bouts_by_length(labels, min_len=0, max_len=0):
    """Exclude *labeled positive* bouts whose length is outside [min_len, max_len].

    Finds contiguous runs of ``labels == 1`` and sets any run shorter than ``min_len``
    or (when ``max_len > 0``) longer than ``max_len`` to ``-1`` (unobserved → dropped
    from training). Returns ``(new_labels, n_bouts_excluded, n_frames_excluded)``.
    A 0 threshold disables that bound. Operates on a copy; never mutates the input or
    any CSV. Used by BOTH the Train tab and Active Learning so labels are cleaned
    identically.
    """
    y = np.asarray(labels).copy()
    min_len = int(min_len or 0)
    max_len = int(max_len or 0)
    if min_len <= 0 and max_len <= 0:
        return y, 0, 0
    pos = (y == 1).astype(np.int8)
    if pos.sum() == 0:
        return y, 0, 0
    # Run boundaries via diff on a zero-padded positive mask.
    d = np.diff(np.concatenate([[0], pos, [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]   # half-open [start, end)
    n_bouts = n_frames = 0
    for s, e in zip(starts, ends):
        length = e - s
        too_short = min_len > 0 and length < min_len
        too_long = max_len > 0 and length > max_len
        if too_short or too_long:
            y[s:e] = -1
            n_bouts += 1
            n_frames += int(length)
    return y, n_bouts, n_frames


def _effective_train_labels(labels, obj):
    """Training labels for *obj* with short/long bouts excluded per its
    ``_label_bout_min`` / ``_label_bout_max`` attrs (0 = off). Returns a copy when
    filtering applies, else the array unchanged. Used so AL trains/evaluates on the
    SAME cleaned labels as the Train tab."""
    mn = int(getattr(obj, '_label_bout_min', 0) or 0)
    mx = int(getattr(obj, '_label_bout_max', 0) or 0)
    if mn <= 0 and mx <= 0:
        return np.asarray(labels)
    return exclude_bouts_by_length(labels, mn, mx)[0]


def session_positive_counts(labels, min_bout_len=1):
    """(n_positive_frames, n_positive_bouts) for a 0/1/-1 label array.

    Only positive runs of length >= ``min_bout_len`` count as a "bout", so stray
    single-frame label speckle doesn't inflate the event count. Call on the EFFECTIVE
    labels (after the label-bout filter); eligibility passes a small floor (e.g. 3)."""
    y = np.asarray(labels)
    pos = (y == 1).astype(np.int8)
    n_frames = int(pos.sum())
    if n_frames == 0:
        return 0, 0
    d = np.diff(np.concatenate([[0], pos, [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    if min_bout_len <= 1:
        n_bouts = len(starts)
    else:
        n_bouts = int(np.sum((ends - starts) >= int(min_bout_len)))
    return n_frames, n_bouts


def select_cv_eligible(per_session_counts, mode='auto', min_frames=0, min_bouts=0):
    """Decide which sessions may be HELD OUT in CV (eligible) vs used training-only.

    ``per_session_counts``: dict ``{session_key: (n_pos_frames, n_pos_bouts)}``.
    Returns ``(eligible_set, train_only_set, info)`` where ``info`` records the rule
    used (for transparent logging). A session is NEVER dropped from training — being
    train-only only excludes it from the held-out evaluation folds.

    - ``mode='off'``  → every session with ≥1 positive is eligible (legacy behavior).
    - ``mode='manual'`` → eligible iff ``pos_bouts ≥ min_bouts`` AND ``pos_frames ≥
      min_frames`` (a 0 threshold disables that bound).
    - ``mode='auto'`` → adaptive cutoff from the cohort's own bout distribution:
      ``adaptive = max(3, ceil(0.25 × median(pos_bouts over positive sessions)))``;
      eligible iff ``pos_bouts ≥ adaptive``.

    Safeguards (all modes): sessions with 0 positives are always train-only; never leave
    fewer than ``min(3, n_positive_sessions)`` eligible (if a rule would, keep the top-N
    by pos_bouts); if <2 would remain eligible, make ALL positive sessions eligible
    (pruning disabled) so there is always something to evaluate.
    """
    import math
    pos_sessions = {k: v for k, v in per_session_counts.items() if v[1] > 0}
    n_pos_sess = len(pos_sessions)
    info = {'mode': mode}

    if n_pos_sess == 0:
        return set(), set(per_session_counts), {'mode': mode, 'reason': 'no positives'}

    if mode == 'off':
        elig = set(pos_sessions)
    elif mode == 'manual':
        elig = {k for k, (pf, pb) in pos_sessions.items()
                if (min_bouts <= 0 or pb >= min_bouts)
                and (min_frames <= 0 or pf >= min_frames)}
        info.update(min_frames=min_frames, min_bouts=min_bouts)
    else:  # auto
        bouts = sorted(pb for (_pf, pb) in pos_sessions.values())
        med = bouts[len(bouts) // 2] if len(bouts) % 2 else \
            (bouts[len(bouts) // 2 - 1] + bouts[len(bouts) // 2]) / 2.0
        # A held-out fold needs ≥3 events to give a non-degenerate operating point;
        # the ¼-of-median term raises the bar further for dense behaviors.
        adaptive = max(3, int(math.ceil(0.25 * med)))
        elig = {k for k, (_pf, pb) in pos_sessions.items() if pb >= adaptive}
        info.update(adaptive_min_bouts=adaptive, median_bouts=med)

    # Safeguard: keep at least min(3, n_pos_sess) eligible (top-N by bouts).
    keep_min = min(3, n_pos_sess)
    if len(elig) < keep_min:
        ranked = sorted(pos_sessions, key=lambda k: pos_sessions[k][1], reverse=True)
        elig = set(ranked[:keep_min])
        info['safeguard'] = f'kept top {keep_min} by bouts'
    if len(elig) < 2:
        elig = set(pos_sessions)   # nothing reliable to compare → evaluate on all
        info['safeguard'] = 'pruning disabled (<2 eligible)'

    train_only = set(per_session_counts) - elig
    info['n_eligible'] = len(elig)
    info['n_train_only'] = len(train_only)
    return elig, train_only, info


# Sweep grid — MUST mirror PixelPaws_GUI._sweep_postprocessing exactly so the AL
# fast path produces bit-identical (threshold, min_bout, min_after_bout, max_gap, f1).
_SWEEP_THRESHOLDS      = None  # built lazily (needs np.arange)
# Comprehensive @60fps: min_bout up to 150 fr (~2.5 s), max_gap up to 90 fr (~1.5 s).
_SWEEP_MIN_BOUTS       = [1, 2, 3, 5, 8, 12, 18, 25, 35, 50, 70, 100, 150]
_SWEEP_MIN_AFTER_BOUTS = [0, 1, 3, 5]
_SWEEP_MAX_GAPS        = [0, 2, 4, 6, 10, 15, 25, 40, 60, 90]


def _sweep_one_threshold(t, oof_proba, y, min_bouts, min_after_bouts, max_gaps,
                         sess_slices=None):
    """Best (f1, mb, ma, mg) for a single threshold, scanning bout params in the
    SAME order as the serial grid (mb→ma→mg ascending), keeping the FIRST max.
    When ``sess_slices`` is given, bout filtering is applied within each session block
    (no cross-session bridging); otherwise whole-array (Train-tab parity)."""
    from evaluation_tab import _apply_bout_filtering
    y_raw = (oof_proba >= float(t)).astype(np.int8)
    best_f1, best_mb, best_ma, best_mg = -1.0, 1, 0, 0
    for mb in min_bouts:
        for ma in min_after_bouts:
            for mg in max_gaps:
                if mb == 1 and ma == 0 and mg == 0:
                    y_filt = y_raw
                elif sess_slices is None:
                    y_filt = _apply_bout_filtering(y_raw.copy(), min_bout=mb,
                                                   min_after_bout=ma, max_gap=mg)
                else:
                    y_filt = y_raw.copy()
                    for (s, e) in sess_slices:
                        y_filt[s:e] = _apply_bout_filtering(
                            y_filt[s:e].copy(), min_bout=mb, min_after_bout=ma, max_gap=mg)
                score = _fast_f1_binary(y, y_filt)
                if score > best_f1:
                    best_f1, best_mb, best_ma, best_mg = score, mb, ma, mg
    return (float(t), best_f1, best_mb, best_ma, best_mg)


def _sweep_postprocessing_fast(oof_proba, y, progress_cb=None, n_jobs=-1, session_ids=None):
    """Parallel replacement for PixelPaws_GUI._sweep_postprocessing.

    The 37 thresholds are independent, so we fan them out across processes; within
    each threshold the 224 bout-param combos are scanned serially in the original
    order. Global winner = max f1, ties broken by the SMALLEST threshold (and, within
    a threshold, the first bout-param combo) — exactly replicating the serial loop's
    'keep first strict-max' behavior. Falls back to serial on any failure.

    ``session_ids=None`` → whole-array bout filtering, **bit-identical** to the Train
    tab's serial sweep. When ``session_ids`` is provided, bout filtering is applied
    per-session (no cross-boundary bridging) — this **intentionally diverges** from the
    Train-tab pooled filtering because it is the correct unit for temporal post-proc."""
    global _SWEEP_THRESHOLDS
    if _SWEEP_THRESHOLDS is None:
        _SWEEP_THRESHOLDS = list(np.arange(0.05, 0.96, 0.025))   # 37 values
    thresholds = _SWEEP_THRESHOLDS
    y = np.asarray(y).astype(np.int8)
    oof_proba = np.asarray(oof_proba, dtype=np.float32)
    _slices = _session_slices(session_ids) if session_ids is not None else None
    args = (_SWEEP_MIN_BOUTS, _SWEEP_MIN_AFTER_BOUTS, _SWEEP_MAX_GAPS, _slices)

    results = []
    try:
        from joblib import Parallel, delayed
        tasks = (delayed(_sweep_one_threshold)(t, oof_proba, y, *args) for t in thresholds)
        try:
            # joblib >= 1.3: stream results so we can report progress as folds land.
            gen = Parallel(n_jobs=n_jobs, return_as='generator')(tasks)
            for i, r in enumerate(gen):
                results.append(r)
                if progress_cb:
                    try: progress_cb(i + 1, len(thresholds))
                    except Exception: pass
        except TypeError:
            results = Parallel(n_jobs=n_jobs)(tasks)
            if progress_cb:
                try: progress_cb(len(thresholds), len(thresholds))
                except Exception: pass
    except Exception:
        # Serial fallback (still uses the fast inline F1).
        results = []
        for i, t in enumerate(thresholds):
            results.append(_sweep_one_threshold(t, oof_proba, y, *args))
            if progress_cb:
                try: progress_cb(i + 1, len(thresholds))
                except Exception: pass

    # Global pick: highest f1; tie → smallest threshold (thresholds are ascending,
    # so the first strict-max in this ordering matches the serial loop's winner).
    results.sort(key=lambda r: r[0])   # ascending threshold
    best = (-1.0, 0.5, 1, 0, 0)        # (f1, t, mb, ma, mg)
    for (t, f1, mb, ma, mg) in results:
        if f1 > best[0]:
            best = (f1, t, mb, ma, mg)
    return {
        'thresh':         float(round(best[1], 2)),
        'min_bout':       int(best[2]),
        'min_after_bout': int(best[3]),
        'max_gap':        int(best[4]),
        'f1':             float(best[0]),
    }


def _honest_pipeline_oof_f1(oof_proba, y, fold_of, session_ids=None,
                            progress_cb=None, n_jobs=-1, bout_tol=6):
    """Leak-free F1 of the FULL deployed pipeline (threshold + bout post-processing).

    Nested leave-one-fold-out: for each fold *f*, the session-aware grid sweep
    (threshold + min_bout + min_after_bout + max_gap) is run on ALL OTHER folds' OOF to
    choose the operating point, which is then applied (per-session) to fold *f*'s own
    frames. A fold never informs its own operating point, so — unlike the global
    `_sweep_postprocessing_fast` on all data — there is no params-on-the-test-set leakage.
    Extends `_honest_oof_f1` (which only LOFO-selected the threshold and ignored bouts).

    Returns ``(honest_f1, per_fold_detail, mean, std, bout)`` where ``per_fold_detail`` is
    a list of dicts ``{fold, f1, precision, recall, n_test, n_pos, bout_f1}`` at that fold's
    swept operating point, and ``bout`` is the session-aware **event/bout-level** P/R/F1
    (with ±``bout_tol``-frame tolerance) computed on the same leak-free honest predictions."""
    from evaluation_tab import bout_level_prf as _bout_prf
    y = np.asarray(y).astype(np.int8)
    oof_proba = np.asarray(oof_proba, dtype=np.float32)
    fold_of = np.asarray(fold_of)
    sid = np.asarray(session_ids) if session_ids is not None else None
    valid = fold_of >= 0
    folds = list(np.unique(fold_of[valid])) if valid.any() else []

    def _sid(mask):
        return sid[mask] if sid is not None else None

    def _prf(yt_true, yp_pred):
        """(f1, precision, recall) from inline TP/FP/FN (binary, pos=1, zero_div=0)."""
        a = yt_true.astype(bool, copy=False); b = yp_pred.astype(bool, copy=False)
        tp = int(np.count_nonzero(b & a)); fp = int(np.count_nonzero(b & ~a))
        fn = int(np.count_nonzero(~b & a))
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2.0 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        return f, p, r

    if len(folds) < 2:
        # Can't leave a fold out — honest predictions, single global operating point.
        bp = _sweep_postprocessing_fast(oof_proba[valid], y[valid],
                                        session_ids=_sid(valid), n_jobs=n_jobs)
        yt = (oof_proba[valid] >= bp['thresh']).astype(np.int8)
        yf = _apply_bouts_per_session(yt, _sid(valid),
                                      bp['min_bout'], bp['min_after_bout'], bp['max_gap'])
        f1, pr, rc = _prf(y[valid], yf)
        bout = _bout_prf(y[valid], yf, _sid(valid), tol=bout_tol)
        det = [{'fold': 0, 'f1': f1, 'precision': pr, 'recall': rc,
                'n_test': int(valid.sum()), 'n_pos': int(y[valid].sum()),
                'bout_f1': bout['f1']}]
        return f1, det, f1, 0.0, bout

    preds = np.zeros(len(y), dtype=np.int8)
    per_fold = []
    for k, f in enumerate(folds):
        this = (fold_of == f)
        others = valid & (fold_of != f)
        bp = _sweep_postprocessing_fast(oof_proba[others], y[others],
                                        session_ids=_sid(others), n_jobs=n_jobs)
        yt = (oof_proba[this] >= bp['thresh']).astype(np.int8)
        preds[this] = _apply_bouts_per_session(yt, _sid(this),
                                               bp['min_bout'], bp['min_after_bout'], bp['max_gap'])
        f1, pr, rc = _prf(y[this], preds[this])
        _bf = _bout_prf(y[this], preds[this], _sid(this), tol=bout_tol)['f1']
        per_fold.append({'fold': int(f), 'f1': f1, 'precision': pr, 'recall': rc,
                         'n_test': int(this.sum()), 'n_pos': int(y[this].sum()),
                         'bout_f1': _bf})
        if progress_cb:
            try: progress_cb(k + 1, len(folds))
            except Exception: pass
    honest = float(_fast_f1_binary(y[valid], preds[valid]))
    f1s = np.asarray([d['f1'] for d in per_fold], dtype=float)
    # Leak-free event/bout-level F1 on the assembled nested honest predictions.
    bout = _bout_prf(y[valid], preds[valid], _sid(valid), tol=bout_tol)
    return honest, per_fold, float(f1s.mean()), float(f1s.std()), bout


def _honest_hmm_oof_f1(oof_proba, y, fold_of, session_ids=None, max_frames=3_000_000):
    """Leak-free F1 using HMM/Viterbi smoothing instead of morphological bout filtering
    — a COMPARISON point for `_honest_pipeline_oof_f1` (NOT deployed). Nested-LOFO: for
    each fold, fit the 2-state transition prior on the OTHER folds' labels and Viterbi-
    decode each held-out session's OOF probability stream (no hard threshold). Returns
    the honest pooled F1, or None if it can't run / the set exceeds ``max_frames`` (the
    plain-Python Viterbi is ~1-2 s per ~700k frames, so the cap is generous now)."""
    try:
        from evaluation_tab import fit_hmm_transitions, viterbi_smooth
    except Exception:
        return None
    y = np.asarray(y).astype(np.int8)
    if len(y) > int(max_frames):
        return None
    oof_proba = np.asarray(oof_proba, dtype=np.float32)
    fold_of = np.asarray(fold_of)
    sid = np.asarray(session_ids) if session_ids is not None else None
    valid = fold_of >= 0
    folds = list(np.unique(fold_of[valid])) if valid.any() else []
    if len(folds) < 1:
        return None
    preds = np.zeros(len(y), dtype=np.int8)
    for f in folds:
        this = (fold_of == f)
        others = valid & (fold_of != f)
        if not others.any() or int(y[others].sum()) == 0:
            preds[this] = (oof_proba[this] >= 0.5).astype(np.int8)
            continue
        log_trans, log_prior = fit_hmm_transitions(y[others])
        this_idx = np.where(this)[0]
        if sid is not None:
            for (s, e) in _session_slices(sid[this]):
                seg = this_idx[s:e]
                preds[seg] = viterbi_smooth(oof_proba[seg], log_trans, log_prior).astype(np.int8)
        else:
            preds[this_idx] = viterbi_smooth(oof_proba[this_idx], log_trans, log_prior).astype(np.int8)
    return float(_fast_f1_binary(y[valid], preds[valid]))


def _engine_select_bouts(all_bouts, subs_features, n_bouts, min_gap=30, pos_quota=0.4, seed=0):
    """Pick a global batch from a pooled multi-session BoutCandidate list using the
    headless engine's strategy (uncertainty x feature-space diversity x class-balance
    quota x temporal min-gap). Returns the chosen BoutCandidates (subset of all_bouts).
    Raises on any error so callers can fall back to the legacy round-robin."""
    from active_learning_engine import EngineBout, select_batch
    ebs = []
    for b in all_bouts:
        feats = subs_features[b.session_idx]
        s = max(0, int(b.start_frame)); e = min(len(feats) - 1, int(b.end_frame))
        fm = feats[s:e + 1].mean(axis=0) if e >= s else None
        eb = EngineBout(session_idx=int(b.session_idx), start=int(b.start_frame),
                        end=int(b.end_frame), score=float(b.mean_uncertainty),
                        peak_score=float(b.mean_uncertainty),
                        pred_pos=bool(b.mean_proba >= 0.5), feat_mean=fm)
        eb._src = b
        ebs.append(eb)
    chosen = select_batch(ebs, batch_size=n_bouts, min_frame_gap=min_gap,
                          pos_quota_frac=pos_quota, seed=seed)
    return [eb._src for eb in chosen]


# Session discovery
try:
    from evaluation_tab import find_session_triplets
    _FIND_SESSIONS_AVAILABLE = True
except ImportError:
    find_session_triplets = None
    _FIND_SESSIONS_AVAILABLE = False


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class ALIterationRecord:
    iteration: int
    n_labeled_total: int
    n_positive: int
    train_f1: float
    oof_f1: Optional[float]
    n_below_threshold: int
    timestamp: str
    oof_precision: Optional[float] = None
    oof_recall:    Optional[float] = None
    held_out_f1:   Optional[float] = None   # unbiased F1 on a fixed held-out test set
    cv_mode:       Optional[str]   = None   # how oof_f1 was estimated:
    #   'session' (leave-animals-out, honest) | 'bout' | 'frame' (leaky) |
    #   'baseline' (seeded from a warm-start classifier's stored honest F1) | None (legacy)


@dataclass
class BoutCandidate:
    """A contiguous run of uncertain frames presented as a single labeling unit."""
    start_frame: int        # core uncertain region start
    end_frame: int          # core uncertain region end (inclusive)
    clip_start: int         # clip start shown to user (with context padding)
    clip_end: int           # clip end shown to user (with context padding)
    mean_proba: float
    mean_uncertainty: float
    duration_frames: int    # length of core bout
    session_idx: int = 0
    video_path: str = ""


# ============================================================================
# UncertaintyEngineV2
# ============================================================================

class UncertaintyEngineV2:
    def __init__(self, min_frame_spacing=30):
        self.min_frame_spacing = min_frame_spacing
        self._last_probas = None

    def score_all_frames(self, model, features) -> np.ndarray:
        probas = model.predict_proba(features)[:, 1]
        self._last_probas = probas
        return probas

    def find_uncertain_frames(self, probas, current_labels, n_suggestions,
                               confidence_threshold=0.30,
                               avoid_labeled=True) -> Tuple[np.ndarray, np.ndarray]:
        # Eligible: within threshold of 0.5
        uncertainty = 1.0 - np.abs(probas - 0.5) * 2
        eligible_mask = np.abs(probas - 0.5) * 2 < confidence_threshold

        scores = uncertainty.copy()

        if avoid_labeled:
            labeled_mask = current_labels >= 0
            scores[labeled_mask] *= 0.1

        # Only consider eligible frames
        scores[~eligible_mask] = 0.0

        selected = []
        suppressed = np.zeros(len(probas), dtype=bool)

        for _ in range(n_suggestions):
            if scores.max() <= 0:
                break
            idx = int(np.argmax(scores))
            selected.append(idx)
            # Suppress neighbors
            lo = max(0, idx - self.min_frame_spacing)
            hi = min(len(scores), idx + self.min_frame_spacing + 1)
            scores[lo:hi] = 0.0

        selected = np.array(selected, dtype=int)
        if len(selected) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        return selected, probas[selected]

    def find_uncertain_bouts(self, probas, current_labels, n_bouts=10,
                              confidence_threshold=0.30,
                              min_bout_frames=5, context_frames=30,
                              max_bout_frames=300,
                              class_balanced: bool = True,
                              diversity_radius: int = 0,
                              _stats=None) -> List[BoutCandidate]:
        """Find contiguous unlabeled segments, ranked by classifier uncertainty.

        Long runs are subdivided into windows of at most max_bout_frames so the
        user is never shown a clip that spans the entire video.
        """
        # If labels extend beyond features cache, fill with 0.5 (max uncertainty)
        n_labels = len(current_labels)
        if n_labels > len(probas):
            probas = np.concatenate([probas, np.full(n_labels - len(probas), 0.5)])

        # Base: unlabeled frames (labels == -1)
        base_mask = current_labels < 0
        n = len(probas)

        # Group consecutive unlabeled frames into runs
        runs = []
        in_run = False
        run_start = 0
        for i in range(n):
            if base_mask[i] and not in_run:
                run_start = i
                in_run = True
            elif not base_mask[i] and in_run:
                runs.append((run_start, i - 1))
                in_run = False
        if in_run:
            runs.append((run_start, n - 1))

        bouts = []
        n_too_short = 0
        for (run_start, run_end) in runs:
            dur = run_end - run_start + 1
            if dur < min_bout_frames:
                n_too_short += 1
                continue

            if dur <= max_bout_frames:
                # Short enough: use as a single bout
                windows = [(run_start, run_end)]
            else:
                # Subdivide into non-overlapping windows of max_bout_frames
                windows = []
                pos = run_start
                while pos <= run_end:
                    w_end = min(pos + max_bout_frames - 1, run_end)
                    if w_end - pos + 1 >= min_bout_frames:
                        windows.append((pos, w_end))
                    pos += max_bout_frames

            for (start, end) in windows:
                clip_start = max(0, start - context_frames)
                clip_end = min(n - 1, end + context_frames)
                seg_probas = probas[start:end + 1]
                mean_proba = float(np.mean(seg_probas))
                mean_uncertainty = float(np.mean(1.0 - np.abs(seg_probas - 0.5) * 2))
                bouts.append(BoutCandidate(
                    start_frame=start, end_frame=end,
                    clip_start=clip_start, clip_end=clip_end,
                    mean_proba=mean_proba, mean_uncertainty=mean_uncertainty,
                    duration_frames=end - start + 1,
                ))

        if _stats is not None:
            _stats['n_runs'] = len(runs)
            _stats['n_too_short'] = n_too_short

        bouts.sort(key=lambda b: -b.mean_uncertainty)

        def _apply_diversity(bucket, radius):
            if radius <= 0:
                return bucket
            kept = []
            for b in bucket:  # already sorted by descending uncertainty
                if not any(abs(b.start_frame - s.start_frame) < radius for s in kept):
                    kept.append(b)
            return kept

        if not class_balanced:
            return bouts[:n_bouts]

        # Round-robin: alternate positive-predicted and negative-predicted bouts
        pos_bouts = _apply_diversity([b for b in bouts if b.mean_proba >= 0.5], diversity_radius)
        neg_bouts = _apply_diversity([b for b in bouts if b.mean_proba < 0.5],  diversity_radius)
        selected, i, j = [], 0, 0
        while len(selected) < n_bouts and (i < len(pos_bouts) or j < len(neg_bouts)):
            if i < len(pos_bouts):
                selected.append(pos_bouts[i]); i += 1
            if j < len(neg_bouts) and len(selected) < n_bouts:
                selected.append(neg_bouts[j]); j += 1
        return selected


# ============================================================================
# LabelPropagator
# ============================================================================

class LabelPropagator:
    def __init__(self, n_neighbors=5, max_frame_spread=30, similarity_threshold=0.92):
        self.n_neighbors = n_neighbors
        self.max_frame_spread = max_frame_spread
        self.similarity_threshold = similarity_threshold

    def propagate(self, labeled_frame: int, label: int,
                  features: np.ndarray, current_labels: np.ndarray) -> dict:
        result = {}
        v = features[labeled_frame].astype(float)
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            return result

        lo = max(0, labeled_frame - self.max_frame_spread)
        hi = min(len(features), labeled_frame + self.max_frame_spread + 1)

        candidates = []
        for f in range(lo, hi):
            if f == labeled_frame:
                continue
            if current_labels[f] != -1:
                continue
            u = features[f].astype(float)
            u_norm = np.linalg.norm(u)
            if u_norm == 0:
                continue
            sim = np.dot(v, u) / (v_norm * u_norm)
            if sim >= self.similarity_threshold:
                candidates.append((sim, f))

        # Sort descending by similarity, take top n_neighbors
        candidates.sort(key=lambda x: -x[0])
        for _, f in candidates[:self.n_neighbors]:
            result[f] = label

        return result


# ============================================================================
# LearningCurveTracker
# ============================================================================

class LearningCurveTracker:
    def __init__(self):
        self.records: List[ALIterationRecord] = []

    def record(self, model, X_train, y_train, n_below_threshold,
               labels_array=None, min_bout=1, session_ids=None, n_folds=5,
               oof_f1=None, oof_precision=None, oof_recall=None,
               held_out_f1=None, cv_mode=None) -> ALIterationRecord:
        n_labeled = len(y_train)
        n_positive = int((y_train == 1).sum())

        # Train F1
        if SKLEARN_AVAILABLE and n_labeled > 0:
            preds = model.predict(X_train)
            try:
                train_f1 = float(f1_score(y_train, preds, zero_division=0))
            except Exception:
                train_f1 = 0.0
        else:
            train_f1 = 0.0

        # OOF F1 — use the caller's precomputed value when given (keeps the plotted
        # number identical to the saved classifier's), else compute via the shared
        # _cv_oof + leave-one-fold-out (nested) honest threshold (no leakage).
        if oof_f1 is None and SKLEARN_AVAILABLE and n_labeled >= 30:
            n_neg = n_labeled - n_positive
            if min(n_positive, n_neg) >= 3:
                try:
                    bgroups = None
                    if labels_array is not None:
                        _lab_idx = np.where(labels_array >= 0)[0]
                        bgroups = _make_bout_groups(_lab_idx, min_bout, session_ids=session_ids)
                    oof_proba_arr, _foldf1, _mode, _fold_of, _folddet = _cv_oof(
                        X_train, y_train, n_folds=n_folds,
                        session_ids=session_ids, bout_groups=bgroups,
                        bout_aware=True, n_estimators=100)
                    if cv_mode is None:
                        cv_mode = _mode   # label this point with how it was estimated
                    oof_true_arr = np.asarray(y_train)
                    oof_f1, _per_fold, _mean, _std = _honest_oof_f1(
                        oof_proba_arr, oof_true_arr, _fold_of)
                    # precision/recall at the honest (pooled LOFO) operating point
                    from sklearn.metrics import precision_score, recall_score
                    _gt = _best_global_threshold(oof_proba_arr, oof_true_arr)
                    best_preds = (oof_proba_arr >= _gt).astype(int)
                    oof_precision = float(precision_score(oof_true_arr, best_preds, zero_division=0))
                    oof_recall    = float(recall_score(oof_true_arr, best_preds, zero_division=0))
                except Exception:
                    oof_f1 = None
                    oof_precision = None
                    oof_recall = None

        rec = ALIterationRecord(
            iteration=len(self.records),
            n_labeled_total=n_labeled,
            n_positive=n_positive,
            train_f1=train_f1,
            oof_f1=oof_f1,
            n_below_threshold=n_below_threshold,
            timestamp=datetime.now().isoformat(),
            oof_precision=oof_precision if oof_f1 is not None else None,
            oof_recall=oof_recall    if oof_f1 is not None else None,
            held_out_f1=held_out_f1,
            cv_mode=cv_mode,
        )
        self.records.append(rec)
        return rec

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump([asdict(r) for r in self.records], f, indent=2)

    def load(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        self.records = []
        for d in data:
            if 'cv_f1' in d and 'oof_f1' not in d:
                d['oof_f1'] = d.pop('cv_f1')
            d.setdefault('oof_precision', None)
            d.setdefault('oof_recall', None)
            d.setdefault('held_out_f1', None)
            d.setdefault('cv_mode', None)
            self.records.append(ALIterationRecord(**d))

    def to_dataframe(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(columns=[
                'iteration', 'n_labeled_total', 'n_positive',
                'train_f1', 'oof_f1', 'n_below_threshold', 'timestamp'
            ])
        return pd.DataFrame([asdict(r) for r in self.records])


# ============================================================================
# ALSessionV2
# ============================================================================

def _select_behavior_column(df, behavior_name):
    """Choose which label-CSV column to read for the target behavior.

    The target behavior is authoritative — a label CSV may hold many behaviors
    (BORIS exports write all of them, with 'rearing' first), so the first column
    is NOT a safe default. Returns the column name to read, or None when the
    requested behavior is absent (caller treats the session as unlabeled for it).

    - behavior_name given and present  → that column.
    - behavior_name not given, single-column CSV → that lone column (legacy).
    - otherwise (behavior absent / can't disambiguate) → None.
    """
    if behavior_name and behavior_name in df.columns:
        return behavior_name
    if not behavior_name and len(df.columns) == 1:
        return df.columns[0]
    return None


class ALSessionV2:
    def __init__(self, labels_csv: str, video_path: str, features_cache: str,
                 min_frame_spacing: int = 30, dlc_path: str = None,
                 behavior_name: str = None):
        self.labels_csv = labels_csv
        self.video_path = video_path
        self.features_cache = features_cache
        # DLC .h5 pose file — needed to compute egocentric features when a
        # warm-start classifier requires them (augment_features_post_cache).
        self._dlc_path = dlc_path or ''

        # Load features first (joblib+LZ4 for canonical 8aed1c22 caches, else
        # pickle) so we know the frame count even when there is no labels CSV.
        feats = _robust_unpickle(features_cache)
        if isinstance(feats, pd.DataFrame):
            self._feature_cols = list(feats.columns)
            feats = feats.values
        else:
            self._feature_cols = None
        self._features = feats.astype(np.float32)

        # Load labels from CSV, OR start a fully-unlabeled pool session when no
        # CSV exists yet (label-from-scratch on new data — needs a warm-start
        # classifier to score, and writes a fresh CSV on save).
        self._behavior_missing = False
        if labels_csv and os.path.isfile(labels_csv):
            df = pd.read_csv(labels_csv)
            # Pick the TARGET behavior column. Label CSVs may hold many behaviors
            # (e.g. BORIS exports with 'rearing' first), so df.columns[0] is NOT a
            # safe proxy — read the requested behavior or, if it's absent, treat the
            # session as unlabeled rather than silently reading a different behavior.
            col = _select_behavior_column(df, behavior_name)
            self.behavior_name = behavior_name or col or 'behavior'
            if col is None:
                self._labels = np.full(len(self._features), -1, dtype=int)
                self._behavior_missing = True
            else:
                raw = df[col].values
                self._labels = np.where(np.isnan(raw.astype(float)), -1, raw.astype(int))
        else:
            self.behavior_name = behavior_name or 'behavior'
            self._labels = np.full(len(self._features), -1, dtype=int)

        # Sync length: extend labels with -1 if features cover more frames than CSV
        n_feat = len(self._features)
        n_csv  = len(self._labels)
        if n_feat > n_csv:
            # Features cover more frames than CSV: extend labels with -1 (unlabeled)
            self._labels = np.concatenate([
                self._labels,
                np.full(n_feat - n_csv, -1, dtype=int)
            ])
            self._n_csv_rows = n_csv   # remember original CSV length for save logic
            self._truncation_warning = None
        elif n_feat < n_csv:
            # Features shorter than CSV: can only score up to features length
            truncated_tail = self._labels[n_feat:]
            n_truncated_labeled = int(np.sum(truncated_tail >= 0))
            self._labels = self._labels[:n_feat]
            self._n_csv_rows = n_feat
            self._truncation_warning = (n_csv - n_feat, n_truncated_labeled)
        else:
            self._n_csv_rows = n_csv
            self._truncation_warning = None
        # Now len(_labels) == len(_features) always

        self._engine = UncertaintyEngineV2(min_frame_spacing)
        self._propagator = LabelPropagator()
        self.tracker = LearningCurveTracker()
        self._iteration = 0
        self._seen_bouts: set = set()   # (start_frame, end_frame) of shown bouts

    def get_labels_and_features(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lab = _effective_train_labels(self._labels, self)
        labeled_mask = lab >= 0
        X_labeled = self._features[labeled_mask]
        y_labeled = lab[labeled_mask]
        return labeled_mask, X_labeled, y_labeled, self._features

    def train_model(self):
        lab = _effective_train_labels(self._labels, self)
        labeled_mask = lab >= 0
        # Trim-to-positive-span parity with the Train tab (training set only).
        if getattr(self, '_trim_to_positive', True):
            keep = _trim_to_positive_span(lab)
            dropped = int(np.sum(labeled_mask & ~keep))
            self._trim_dropped = dropped
            labeled_mask = labeled_mask & keep
        X_labeled = self._features[labeled_mask]
        y_labeled = lab[labeled_mask]
        if len(X_labeled) == 0:
            raise ValueError("No labeled frames available for training.")
        n_positive = int((y_labeled == 1).sum())
        if n_positive == 0:
            raise ValueError("No positive-labeled frames. Label at least one positive example.")
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                              random_state=42, verbosity=0)
        if self._feature_cols:
            model.fit(pd.DataFrame(X_labeled, columns=self._feature_cols), y_labeled)
        else:
            model.fit(X_labeled, y_labeled)
        return model

    def get_full_probas(self, model) -> np.ndarray:
        feats = _align_features(model, self._features, self._feature_cols)
        return self._engine.score_all_frames(model, feats)

    def _adaptive_max_bout_frames(self, default: int = 300) -> int:
        """90th-percentile duration of labeled positive runs, or default if too few samples."""
        positive = (self._labels == 1)
        if not positive.any():
            return default
        durations = []
        run_len = 0
        for v in positive:
            if v:
                run_len += 1
            elif run_len:
                durations.append(run_len)
                run_len = 0
        if run_len:
            durations.append(run_len)
        if len(durations) < 3:
            return default
        p90 = int(np.percentile(durations, 90))
        return max(30, min(p90, 1000))

    def retrain_and_snapshot(self, confidence_threshold: float = 0.3,
                              snapshot_dir: str = None) -> dict:
        """Retrain from current labels, record to tracker, optionally save snapshot pkl."""
        import pickle, os, time

        prev_cv = self.tracker.records[-1].oof_f1 if self.tracker.records else None

        model = self.train_model()
        probas = self.get_full_probas(model)
        n_below = int(np.sum(np.abs(probas - 0.5) * 2 < confidence_threshold))
        labeled_mask, X_labeled, y_labeled, _ = self.get_labels_and_features()
        record = self.tracker.record(model, X_labeled, y_labeled, n_below)

        snapshot_path = None
        if snapshot_dir:
            os.makedirs(snapshot_dir, exist_ok=True)
            ts = time.strftime('%Y%m%d_%H%M%S')
            fname = f"al_iter{record.iteration}_{ts}.pkl"
            snapshot_path = os.path.join(snapshot_dir, fname)
            atomic_pickle_save({
                'clf_model': model,
                'iteration': record.iteration,
                'train_f1': record.train_f1,
                'cv_f1': record.oof_f1,
                'n_labeled_total': record.n_labeled_total,
            }, snapshot_path)

        delta_cv = None
        if prev_cv is not None and record.oof_f1 is not None:
            delta_cv = record.oof_f1 - prev_cv

        return {
            'model': model, 'probas': probas, 'record': record,
            'snapshot_path': snapshot_path, 'delta_cv': delta_cv,
        }

    def run_one_iteration(self, n_bouts: int, confidence_threshold: float,
                          min_bout_frames: int = 5, context_frames: int = 30,
                          max_bout_frames=None,          # None = adaptive
                          class_balanced: bool = True, diversity_radius: int = 0,
                          progress_callback=None, model=None) -> dict:
        if model is None:
            model = self.train_model()
        probas = self.get_full_probas(model)
        max_bout = max_bout_frames if max_bout_frames else self._adaptive_max_bout_frames()
        _candidates = self._engine.find_uncertain_bouts(
            probas, self._labels, n_bouts * 3, confidence_threshold,
            min_bout_frames, context_frames, max_bout_frames=max_bout,
            class_balanced=class_balanced, diversity_radius=diversity_radius)
        _unseen = [b for b in _candidates if (b.start_frame, b.end_frame) not in self._seen_bouts]
        _seen_c = [b for b in _candidates if (b.start_frame, b.end_frame) in self._seen_bouts]
        bouts = (_unseen + _seen_c)[:n_bouts]
        for b in bouts:
            self._seen_bouts.add((b.start_frame, b.end_frame))
        for b in bouts:
            b.session_idx = 0
            b.video_path = self.video_path
        n_eligible = int(np.sum(np.abs(probas - 0.5) * 2 < confidence_threshold))
        return {
            'model': model,
            'probas': probas,
            'bouts': bouts,
            'n_eligible': n_eligible,
        }

    def apply_labels(self, new_labels: dict, confidence_threshold: float,
                     propagate: bool = False,
                     probas: np.ndarray = None, record_curve: bool = True) -> dict:
        n_propagated = 0

        for key, label in new_labels.items():
            if isinstance(key, tuple):
                # Bout-keyed: expand to per-frame
                # Support both 2-tuple (start, end) and 3-tuple (session_idx, start, end)
                if len(key) == 3:
                    _, start, end = key
                else:
                    start, end = key
                for f in range(start, min(end + 1, len(self._labels))):
                    self._labels[f] = label
            else:
                # Frame-keyed (legacy)
                self._labels[key] = label
                if propagate:
                    propagated = self._propagator.propagate(
                        key, label, self._features, self._labels
                    )
                    for pf, pl in propagated.items():
                        self._labels[pf] = pl
                    n_propagated += len(propagated)

        # Save CSV
        self.save_labels_csv()

        # Cheap counts always; the heavy train + all-frame predict + CV that builds
        # a curve point is skipped when record_curve=False (the GUI retrains and
        # records right after labeling, so doing it here is redundant work + a
        # duplicate curve point — and the cause of the pre-prompt CPU spike).
        labeled_mask, X_labeled, y_labeled, _ = self.get_labels_and_features()
        n_labeled_total = int(labeled_mask.sum())
        n_positive = int((y_labeled == 1).sum())

        record = None
        if record_curve:
            try:
                model = self.train_model()
                n_below = int(np.sum(
                    np.abs(self._engine.score_all_frames(model, self._features) - 0.5) * 2
                    < confidence_threshold
                ))
                record = self.tracker.record(model, X_labeled, y_labeled, n_below)
            except Exception:
                record = None

        self._iteration += 1

        return {
            'propagated_count': n_propagated,
            'n_labeled_total': n_labeled_total,
            'n_positive': n_positive,
            'iteration_record': record,
        }

    def count_eligible(self, probas: np.ndarray, threshold: float,
                       min_bout_frames: int = 5) -> tuple:
        """Returns (n_frames, n_bouts, stats) where n_frames = unlabeled & uncertain."""
        # Extend probas with 0.5 (max uncertainty) for frames beyond features cache
        n_lab = len(self._labels)
        if n_lab > len(probas):
            ext_probas = np.concatenate([probas, np.full(n_lab - len(probas), 0.5)])
        else:
            ext_probas = probas[:n_lab]
        uncertain = np.abs(ext_probas - 0.5) * 2 < threshold
        unlabeled = self._labels < 0
        n_frames = int(np.sum(uncertain & unlabeled))
        stats = {}
        bouts = self._engine.find_uncertain_bouts(
            probas, self._labels, n_bouts=9999,
            confidence_threshold=threshold, min_bout_frames=min_bout_frames,
            _stats=stats)
        return n_frames, len(bouts), stats

    def count_positive(self) -> int:
        return int((self._labels == 1).sum())

    def is_converged(self, probas: np.ndarray, confidence_threshold: float) -> bool:
        return np.sum(np.abs(probas - 0.5) * 2 < confidence_threshold) == 0

    def save_labels_csv(self):
        """Read existing CSV, update first column with self._labels, grow if needed.

        For a from-scratch pool session (no CSV yet), create one named by the
        behavior with one row per labeled frame.
        """
        try:
            if not (self.labels_csv and os.path.isfile(self.labels_csv)):
                # Brand-new label file: write all frames up to the last labeled one.
                out_all = self._labels.astype(float)
                out_all[out_all == -1] = np.nan
                labeled = np.where(~np.isnan(out_all))[0]
                if len(labeled) == 0:
                    return  # nothing labeled yet — nothing to persist
                n_keep = int(labeled.max()) + 1
                df = pd.DataFrame({self.behavior_name: out_all[:n_keep]})
                if self.labels_csv:
                    os.makedirs(os.path.dirname(self.labels_csv) or '.', exist_ok=True)
                    atomic_dataframe_to_csv(df, self.labels_csv, index=False)
                return
            df = pd.read_csv(self.labels_csv)
            # Write into the TARGET behavior column, preserving every other behavior
            # column. If the CSV doesn't have this behavior yet, add it (NaN elsewhere)
            # rather than clobbering df.columns[0].
            col = self.behavior_name or df.columns[0]
            if col not in df.columns:
                df[col] = np.nan
            n_df  = len(df)
            n_lbl = len(self._labels)

            # Build output array: -1 → NaN
            out = self._labels.astype(float)
            out[out == -1] = np.nan

            if n_lbl > n_df:
                # Check whether any frames beyond the original CSV have been labeled
                extra = out[n_df:]
                last_new = -1
                for i in range(len(extra) - 1, -1, -1):
                    if not np.isnan(extra[i]):
                        last_new = i
                        break
                if last_new >= 0:
                    # Extend df with NaN rows up to and including the furthest new label
                    new_rows = pd.DataFrame({col: extra[:last_new + 1]})
                    df = pd.concat([df, new_rows], ignore_index=True)

            # Update all rows within current df
            n = min(len(df), n_lbl)
            df.loc[:n - 1, col] = out[:n]
            atomic_dataframe_to_csv(df, self.labels_csv, index=False)
        except Exception:
            traceback.print_exc()


# ============================================================================
# MultiSessionAL
# ============================================================================

class MultiSessionAL:
    def __init__(self, sessions: list, min_frame_spacing: int = 5,
                 behavior_name: str = None):
        """
        sessions: list of {'labels_csv': str, 'video_path': str, 'features_cache': str}
        Sessions whose 'labels_csv' does not exist are treated as fully-unlabeled
        pool members (all -1) — requires a warm-start classifier to score.
        """
        import pickle
        self._subs = []
        self.behavior_name = behavior_name
        # Sessions whose CSV lacks the target behavior column (treated as unlabeled).
        self._missing_behavior = []
        for i, s in enumerate(sessions):
            feats = _robust_unpickle(s['features_cache'])
            feat_cols = list(feats.columns) if isinstance(feats, pd.DataFrame) else None
            if isinstance(feats, pd.DataFrame):
                feats = feats.values
            feats = feats.astype(np.float32)
            _lcsv = s.get('labels_csv')
            if _lcsv and os.path.isfile(_lcsv):
                df = pd.read_csv(_lcsv)
                # Read the TARGET behavior column, not df.columns[0] — multi-behavior
                # CSVs (BORIS exports) would otherwise feed the wrong behavior.
                col = _select_behavior_column(df, behavior_name)
                if self.behavior_name is None:
                    self.behavior_name = behavior_name or col or 'behavior'
                if col is None:
                    # CSV exists but has no column for this behavior → unlabeled here.
                    labels = np.full(len(feats), -1, dtype=int)
                    self._missing_behavior.append(
                        os.path.splitext(os.path.basename(_lcsv))[0])
                else:
                    raw = df[col].values
                    labels = np.where(np.isnan(raw.astype(float)), -1, raw.astype(int))
            else:
                # Fully-unlabeled pool member.
                if self.behavior_name is None:
                    self.behavior_name = behavior_name or 'behavior'
                labels = np.full(len(feats), -1, dtype=int)
            n_feat = len(feats)
            n_csv = len(labels)
            _truncation_warning = None
            if n_feat > n_csv:
                # Features cover more frames than CSV — extend labels with -1 (unlabeled)
                labels = np.concatenate([labels, np.full(n_feat - n_csv, -1, dtype=int)])
            elif n_feat < n_csv:
                truncated_tail = labels[n_feat:]
                n_truncated_labeled = int(np.sum(truncated_tail >= 0))
                labels = labels[:n_feat]
                _truncation_warning = (n_csv - n_feat, n_truncated_labeled)
            self._subs.append({
                'labels': labels, 'features': feats, 'feature_cols': feat_cols,
                'video_path': s['video_path'], 'labels_csv': s['labels_csv'],
                # DLC pose path per sub — egocentric augmentation re-reads it.
                'dlc_path': s.get('dlc_path', ''),
                # behavior name to write into a from-scratch label CSV's header.
                'behavior_name': self.behavior_name,
                '_truncation_warning': _truncation_warning,
            })
        self._engine = UncertaintyEngineV2(min_frame_spacing)
        self._propagator = LabelPropagator()
        self.tracker = LearningCurveTracker()
        self._iteration = 0
        self._last_model = None
        self._seen_bouts: set = set()   # (session_idx, start_frame, end_frame) of shown bouts

    def train_model(self):
        """Pool labeled frames from all sessions and train XGBoost."""
        from xgboost import XGBClassifier
        dfs, ys = [], []
        _trim = getattr(self, '_trim_to_positive', True)
        _dropped = 0
        for sub in self._subs:
            lab = _effective_train_labels(sub['labels'], self)
            mask = lab >= 0
            # Per-session trim-to-positive-span parity with the Train tab.
            if _trim:
                keep = _trim_to_positive_span(lab)
                _dropped += int(np.sum(mask & ~keep))
                mask = mask & keep
            if not mask.any():
                continue
            X_sub = sub['features'][mask]
            if sub.get('feature_cols'):
                dfs.append(pd.DataFrame(X_sub, columns=sub['feature_cols']))
            else:
                dfs.append(pd.DataFrame(X_sub))
            ys.append(lab[mask])
        self._trim_dropped = _dropped
        if not dfs:
            raise ValueError("No labeled frames across any session.")
        X = pd.concat(dfs, ignore_index=True).fillna(0.0)
        y = np.concatenate(ys)
        if (y == 1).sum() == 0:
            raise ValueError("No positive-labeled frames.")
        model = XGBClassifier(n_estimators=100, max_depth=6,
                              learning_rate=0.1, random_state=42, verbosity=0)
        model.fit(X, y)
        return model

    def run_one_iteration(self, n_bouts, confidence_threshold,
                          min_bout_frames=5, context_frames=30,
                          max_bout_frames=None,          # None = adaptive (300)
                          class_balanced: bool = True, diversity_radius: int = 0,
                          progress_callback=None, model=None) -> dict:
        if model is None:
            model = self.train_model()
        self._last_model = model
        all_bouts = []
        n_eligible = 0
        max_bout = max_bout_frames if max_bout_frames else 300
        for i, sub in enumerate(self._subs):
            probas = model.predict_proba(
                _align_features(model, sub['features'], sub.get('feature_cols')))[:, 1]
            n_eligible += int(np.sum(np.abs(probas - 0.5) * 2 < confidence_threshold))
            bouts = self._engine.find_uncertain_bouts(
                probas, sub['labels'], n_bouts=n_bouts * 3,
                confidence_threshold=confidence_threshold,
                min_bout_frames=min_bout_frames, context_frames=context_frames,
                max_bout_frames=max_bout,
                class_balanced=class_balanced, diversity_radius=diversity_radius)
            for b in bouts:
                b.session_idx = i
                b.video_path = sub['video_path']
            all_bouts.extend(bouts)
        # Global selection via the headless engine: uncertainty x feature-space
        # diversity (k-means++) x class-balance quota x temporal min-gap, pooled
        # across ALL videos ("select globally"). Falls back to the legacy
        # per-session round-robin interleave on any error.
        _selected = None
        try:
            _selected = _engine_select_bouts(
                all_bouts, [sub['features'] for sub in self._subs], n_bouts,
                min_gap=self._engine.min_frame_spacing, seed=self._iteration)
        except Exception:
            _selected = None
        if not _selected:
            from collections import deque as _deque
            _by_sess = {}
            for _b in all_bouts:
                _by_sess.setdefault(_b.session_idx, []).append(_b)
            _queues = [_deque(_bouts) for _bouts in _by_sess.values()]
            _selected = []
            while len(_selected) < n_bouts and _queues:
                _next_queues = []
                for _q in _queues:
                    if _q:
                        _selected.append(_q.popleft())
                        if len(_selected) >= n_bouts:
                            break
                    if _q:
                        _next_queues.append(_q)
                _queues = _next_queues
        # Seen-bout deduplication: prefer bouts not shown in prior iterations
        _unseen = [b for b in _selected
                   if (b.session_idx, b.start_frame, b.end_frame) not in self._seen_bouts]
        _seen_c = [b for b in _selected
                   if (b.session_idx, b.start_frame, b.end_frame) in self._seen_bouts]
        _selected = (_unseen + _seen_c)[:n_bouts]
        for b in _selected:
            self._seen_bouts.add((b.session_idx, b.start_frame, b.end_frame))
        all_probas = np.concatenate([
            model.predict_proba(
                _align_features(model, sub['features'], sub.get('feature_cols')))[:, 1]
            for sub in self._subs])
        return {'model': model, 'probas': all_probas,
                'bouts': _selected, 'n_eligible': n_eligible}

    def apply_labels(self, new_labels: dict, confidence_threshold: float,
                     propagate: bool = False,
                     probas: np.ndarray = None, record_curve: bool = True) -> dict:
        """new_labels keys: (session_idx, start, end) tuples.
        probas, if given, is the concatenated array across all sub-sessions in order.
        record_curve=False skips the heavy train+predict+CV (the GUI retrains after).
        """
        # Write the user's label LITERALLY across the marked span (YES→1, NO→0),
        # matching the single-session path and the Train tab. Previously a YES was
        # refined against the model (`1 if proba>0.5 else 0`), which silently turned a
        # YES marked on an uncertain bout (proba≤0.5 — exactly what AL surfaces) into a
        # negative. The user's explicit label must win. `probas` is unused now but kept
        # in the signature for call-site compatibility. Unmarked frames stay -1.
        for key, label in new_labels.items():
            if len(key) == 3:
                sess_idx, start, end = key
            else:
                sess_idx, (start, end) = 0, key
            sub = self._subs[sess_idx]
            for f in range(start, min(end + 1, len(sub['labels']))):
                sub['labels'][f] = label

        for sub in self._subs:
            self._save_labels_csv(sub)

        Xs, ys = [], []
        for sub in self._subs:
            mask = sub['labels'] >= 0
            if mask.any():
                Xs.append(sub['features'][mask])
                ys.append(sub['labels'][mask])
        n_labeled = sum(len(y) for y in ys)
        n_positive = sum((y == 1).sum() for y in ys)

        record = None
        if record_curve:
            try:
                model = self.train_model()
                X = np.concatenate(Xs)
                y_all = np.concatenate(ys)
                # Compute actual n_below from fresh predictions
                all_p = np.concatenate([
                    model.predict_proba(sub['features'])[:, 1] for sub in self._subs])
                n_below = int(np.sum(np.abs(all_p - 0.5) * 2 < confidence_threshold))
                record = self.tracker.record(model, X, y_all, n_below)
            except Exception:
                import traceback
                traceback.print_exc()

        self._iteration += 1
        return {'propagated_count': 0, 'n_labeled_total': n_labeled,
                'n_positive': int(n_positive), 'iteration_record': record}

    def is_converged(self, probas, confidence_threshold) -> bool:
        return np.sum(np.abs(probas - 0.5) * 2 < confidence_threshold) == 0

    def count_eligible(self, probas, threshold, min_bout_frames=5) -> tuple:
        n_frames = 0
        n_bouts = 0
        pos = 0
        for sub in self._subs:
            n = len(sub['features'])
            sub_probas = probas[pos:pos + n]
            pos += n
            uncertain = np.abs(sub_probas - 0.5) * 2 < threshold
            unlabeled = sub['labels'] < 0
            n_frames += int(np.sum(uncertain & unlabeled))
            bouts = self._engine.find_uncertain_bouts(
                sub_probas, sub['labels'], n_bouts=9999,
                confidence_threshold=threshold, min_bout_frames=min_bout_frames)
            n_bouts += len(bouts)
        return n_frames, n_bouts, {}

    def count_positive(self) -> int:
        return sum(int((s['labels'] == 1).sum()) for s in self._subs)

    def retrain_and_snapshot(self, confidence_threshold: float = 0.3,
                              snapshot_dir: str = None) -> dict:
        import pickle, os, time

        prev_cv = self.tracker.records[-1].oof_f1 if self.tracker.records else None

        model = self.train_model()
        all_p = np.concatenate([
            model.predict_proba(sub['features'])[:, 1] for sub in self._subs])
        n_below = int(np.sum(np.abs(all_p - 0.5) * 2 < confidence_threshold))

        Xs, ys = [], []
        for sub in self._subs:
            mask = sub['labels'] >= 0
            if mask.any():
                Xs.append(sub['features'][mask])
                ys.append(sub['labels'][mask])
        X = np.concatenate(Xs)
        y_all = np.concatenate(ys)
        record = self.tracker.record(model, X, y_all, n_below)

        snapshot_path = None
        if snapshot_dir:
            os.makedirs(snapshot_dir, exist_ok=True)
            ts = time.strftime('%Y%m%d_%H%M%S')
            fname = f"al_iter{record.iteration}_{ts}.pkl"
            snapshot_path = os.path.join(snapshot_dir, fname)
            atomic_pickle_save({
                'clf_model': model,
                'iteration': record.iteration,
                'train_f1': record.train_f1,
                'cv_f1': record.oof_f1,
                'n_labeled_total': record.n_labeled_total,
            }, snapshot_path)

        delta_cv = None
        if prev_cv is not None and record.oof_f1 is not None:
            delta_cv = record.oof_f1 - prev_cv

        return {
            'model': model, 'probas': all_p, 'record': record,
            'snapshot_path': snapshot_path, 'delta_cv': delta_cv,
        }

    @staticmethod
    def _save_labels_csv(sub: dict):
        try:
            lcsv = sub.get('labels_csv')
            if not (lcsv and os.path.isfile(lcsv)):
                # From-scratch pool member: create a CSV named by the behavior,
                # rows up to the furthest labeled frame.
                if not lcsv:
                    return
                out_all = sub['labels'].astype(float)
                out_all[out_all == -1] = np.nan
                labeled = np.where(~np.isnan(out_all))[0]
                if len(labeled) == 0:
                    return
                n_keep = int(labeled.max()) + 1
                col = sub.get('behavior_name') or 'behavior'
                df = pd.DataFrame({col: out_all[:n_keep]})
                os.makedirs(os.path.dirname(lcsv) or '.', exist_ok=True)
                atomic_dataframe_to_csv(df, lcsv, index=False)
                return
            df = pd.read_csv(lcsv)
            # Update the TARGET behavior column only — preserve every other behavior
            # column; add the column if this CSV doesn't have it yet.
            col = sub.get('behavior_name') or df.columns[0]
            if col not in df.columns:
                df[col] = np.nan
            n = min(len(df), len(sub['labels']))
            df.loc[:n-1, col] = sub['labels'][:n].astype(float)
            df.loc[df[col] == -1, col] = np.nan
            atomic_dataframe_to_csv(df, lcsv, index=False)
        except Exception:
            import traceback
            traceback.print_exc()


# ============================================================================
# run_directed_discovery
# ============================================================================

def run_directed_discovery(project_folder: str, labels_csv: str, features_cache: str,
                            behavior_name: str, run_name: str = 'al_discovery',
                            min_cluster_size: int = 50) -> Optional[str]:
    """
    Run UMAP + HDBSCAN on positive-labeled frames to find sub-behaviors.
    Saves per-cluster CSVs to <project>/unsupervised/<run_name>/.
    Returns output path or None on failure.
    Does NOT import from unsupervised_tab.py.
    """
    if not UMAP_HDBSCAN_AVAILABLE:
        print("run_directed_discovery: umap-learn/hdbscan not installed. Skipping.")
        return None
    try:
        # Load labels — prefer the requested behavior's column over df.columns[0]
        # (label CSVs may hold many behaviors).
        df_labels = pd.read_csv(labels_csv)
        bname = (behavior_name if (behavior_name and behavior_name in df_labels.columns)
                 else df_labels.columns[0])
        raw = df_labels[bname].values
        labels_arr = np.where(np.isnan(raw.astype(float)), -1, raw.astype(int))

        # Load features (joblib+LZ4 for canonical 8aed1c22 caches, else pickle)
        feats = _robust_unpickle(features_cache)
        if isinstance(feats, pd.DataFrame):
            feats = feats.values
        feats = feats.astype(np.float32)

        min_len = min(len(labels_arr), len(feats))
        labels_arr = labels_arr[:min_len]
        feats = feats[:min_len]

        pos_mask = labels_arr == 1
        pos_indices = np.where(pos_mask)[0]
        if len(pos_indices) < 50:
            print(f"run_directed_discovery: only {len(pos_indices)} positive frames, need >=50.")
            return None

        X = feats[pos_indices]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.0, n_components=2, random_state=42)
        embedding = reducer.fit_transform(X_scaled)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        cluster_labels = clusterer.fit_predict(embedding)

        # Save outputs
        out_dir = os.path.join(project_folder, 'unsupervised', run_name)
        os.makedirs(out_dir, exist_ok=True)

        unique_clusters = sorted(set(cluster_labels))
        for c in unique_clusters:
            if c == -1:
                cluster_name = 'noise'
            else:
                cluster_name = f'cluster_{c:02d}'

            mask = cluster_labels == c
            frame_indices = pos_indices[mask]

            # Build output DataFrame — one binary column per cluster
            total_frames = len(labels_arr)
            col = np.zeros(total_frames, dtype=int)
            col[frame_indices] = 1

            out_df = pd.DataFrame({cluster_name: col})
            out_path = os.path.join(out_dir, f'{cluster_name}.csv')
            out_df.to_csv(out_path, index=False)

        # Save summary
        summary = {
            'behavior': bname,
            'run_name': run_name,
            'n_positive_frames': int(len(pos_indices)),
            'n_clusters': len([c for c in unique_clusters if c != -1]),
            'n_noise': int(np.sum(cluster_labels == -1)),
            'timestamp': datetime.now().isoformat(),
        }
        with open(os.path.join(out_dir, 'discovery_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"run_directed_discovery: saved {len(unique_clusters)} cluster CSVs to {out_dir}")
        return out_dir
    except Exception as e:
        traceback.print_exc()
        return None


# ============================================================================
# LabelingInterface (migrated verbatim from active_learning.py)
# ============================================================================

class LabelingInterface:
    """
    GUI for labeling suggested frames.
    """

    def __init__(self,
                 video_path: str,
                 suggested_frames: np.ndarray,
                 confidences: np.ndarray,
                 behavior_name: str):
        """
        Args:
            video_path: Path to video file
            suggested_frames: Frame indices to label
            confidences: Model confidence for each frame
            behavior_name: Name of behavior being labeled
        """
        self.video_path = video_path
        self.suggested_frames = suggested_frames
        self.confidences = confidences
        self.behavior_name = behavior_name

        # Results
        self.labels = {}  # frame_idx -> 0 or 1
        self.current_idx = 0

        # Video
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # UI
        self.window = None
        self.video_label = None
        self.context_start = -60  # Show 1 sec before
        self.context_end = 60     # Show 1 sec after

    def run(self) -> dict:
        """
        Show labeling interface and return labeled frames.

        Returns:
            Dictionary mapping frame_idx -> label (0 or 1)
        """
        self.create_ui()
        self.show_current_frame()

        # If using Toplevel, use wait_window instead of mainloop
        if isinstance(self.window, tk.Toplevel):
            self.window.wait_window()
        else:
            self.window.mainloop()

        self.cap.release()
        return self.labels

    def create_ui(self):
        """Create the labeling interface window"""
        # Use Toplevel instead of Tk() since we're being called from PixelPaws
        # which already has a Tk() root
        import tkinter as tk
        from tkinter import ttk

        # Try to get existing root, otherwise create new one
        try:
            root = tk._default_root
            if root:
                self.window = tk.Toplevel(root)
            else:
                self.window = tk.Tk()
        except Exception:
            self.window = tk.Tk()

        self.window.title(f"Active Learning - {self.behavior_name}")
        _sw = self.window.winfo_screenwidth()
        _sh = self.window.winfo_screenheight()
        _w = int(_sw * 0.75)
        _h = int(_sh * 0.75)
        self.window.geometry(f"{_w}x{_h}+{(_sw-_w)//2}+{(_sh-_h)//2}")
        self.window.resizable(True, True)

        # Title
        title_frame = ttk.Frame(self.window)
        title_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(
            title_frame,
            text=f"🧠 Active Learning - {self.behavior_name}",
            font=(FONT_FAMILY, 16, "bold")
        ).pack()

        # Progress
        progress_frame = ttk.Frame(self.window)
        progress_frame.pack(fill=tk.X, padx=10, pady=5)

        self.progress_label = ttk.Label(
            progress_frame,
            text=f"Frame 1 of {len(self.suggested_frames)}",
            font=(FONT_FAMILY, 12)
        )
        self.progress_label.pack()

        self.progress_bar = ttk.Progressbar(
            progress_frame,
            length=800,
            mode='determinate',
            maximum=len(self.suggested_frames)
        )
        self.progress_bar.pack(pady=5)

        # Video display
        video_frame = ttk.Frame(self.window)
        video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.video_label = tk.Label(video_frame, bg='black')
        self.video_label.pack()

        # Frame info
        info_frame = ttk.Frame(self.window)
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        self.info_label = ttk.Label(
            info_frame,
            text="",
            font=(FONT_FAMILY, 10)
        )
        self.info_label.pack()

        # Question
        question_frame = ttk.Frame(self.window)
        question_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(
            question_frame,
            text=f"Is this {self.behavior_name} behavior?",
            font=(FONT_FAMILY, 14, "bold")
        ).pack()

        # Buttons
        button_frame = ttk.Frame(self.window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        _bs = {'bootstyle': 'success'} if _TTKBOOTSTRAP else {}
        btn_yes = ttk.Button(
            button_frame, text="✓ YES (Y)", width=15,
            command=lambda: self.label_frame(1), **_bs)
        btn_yes.pack(side=tk.LEFT, padx=5, expand=True)

        _bd = {'bootstyle': 'danger'} if _TTKBOOTSTRAP else {}
        btn_no = ttk.Button(
            button_frame, text="✗ NO (N)", width=15,
            command=lambda: self.label_frame(0), **_bd)
        btn_no.pack(side=tk.LEFT, padx=5, expand=True)

        _bsec = {'bootstyle': 'secondary'} if _TTKBOOTSTRAP else {}
        btn_skip = ttk.Button(
            button_frame, text="? SKIP (S)", width=15,
            command=self.skip_frame, **_bsec)
        btn_skip.pack(side=tk.LEFT, padx=5, expand=True)

        # Context playback button
        context_frame = ttk.Frame(self.window)
        context_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(
            context_frame,
            text="▶ Play Context (±2 sec)",
            command=self.play_context
        ).pack()

        # Keyboard shortcuts
        self.window.bind('y', lambda e: self.label_frame(1))
        self.window.bind('Y', lambda e: self.label_frame(1))
        self.window.bind('n', lambda e: self.label_frame(0))
        self.window.bind('N', lambda e: self.label_frame(0))
        self.window.bind('s', lambda e: self.skip_frame())
        self.window.bind('S', lambda e: self.skip_frame())
        self.window.bind('<space>', lambda e: self.play_context())

        # Shortcuts label
        ttk.Label(
            self.window,
            text="Shortcuts: Y=Yes | N=No | S=Skip | Space=Play Context",
            font=(FONT_FAMILY, 9),
            foreground="gray"
        ).pack(pady=5)

    def show_current_frame(self):
        """Display the current frame to label"""
        if self.current_idx >= len(self.suggested_frames):
            self.finish_labeling()
            return

        frame_idx = self.suggested_frames[self.current_idx]
        confidence = self.confidences[self.current_idx]

        # Update progress
        self.progress_label.config(
            text=f"Frame {self.current_idx + 1} of {len(self.suggested_frames)}"
        )
        self.progress_bar['value'] = self.current_idx

        # Update info
        timestamp = frame_idx / self.fps
        self.info_label.config(
            text=f"Frame: {frame_idx} / {self.total_frames}  |  "
                 f"Time: {timestamp:.2f}s  |  "
                 f"Confidence: {confidence:.1%} (Uncertain!)"
        )

        # Load and display frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if ret:
            # Resize for display
            height, width = frame.shape[:2]
            max_width = 800
            if width > max_width:
                scale = max_width / width
                new_width = max_width
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to PhotoImage
            from PIL import Image, ImageTk
            img = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image=img)

            # CRITICAL: Keep reference to prevent garbage collection
            self.current_photo = photo
            self.video_label.config(image=photo)

    def label_frame(self, label: int):
        """Label current frame and move to next"""
        frame_idx = self.suggested_frames[self.current_idx]

        # Store label in dict (no bounds check needed - dict can store any frame index)
        self.labels[frame_idx] = label

        print(f"  Frame {frame_idx}: {'YES' if label == 1 else 'NO'}")

        self.current_idx += 1

        # Check if we're done
        if self.current_idx >= len(self.suggested_frames):
            self.close_interface()
        else:
            self.show_current_frame()

    def skip_frame(self):
        """Skip current frame without labeling"""
        frame_idx = self.suggested_frames[self.current_idx]
        print(f"  Frame {frame_idx}: SKIPPED")

        self.current_idx += 1

        # Check if we're done
        if self.current_idx >= len(self.suggested_frames):
            self.close_interface()
        else:
            self.show_current_frame()

    def play_context(self):
        """Play video context around current frame"""
        frame_idx = self.suggested_frames[self.current_idx]

        # Calculate context window
        start_frame = max(0, frame_idx + self.context_start)
        end_frame = min(self.total_frames, frame_idx + self.context_end)

        # Create playback window
        play_window = tk.Toplevel(self.window)
        play_window.title("Context Playback")
        _sw = play_window.winfo_screenwidth()
        _sh = play_window.winfo_screenheight()
        _w = int(_sw * 0.65)
        _h = int(_sh * 0.65)
        play_window.geometry(f"{_w}x{_h}+{(_sw-_w)//2}+{(_sh-_h)//2}")
        play_window.resizable(True, True)

        play_label = tk.Label(play_window, bg='black')
        play_label.pack(fill=tk.BOTH, expand=True)

        # Play frames
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for i in range(start_frame, end_frame):
            ret, frame = self.cap.read()
            if not ret:
                break

            # Highlight target frame
            if i == frame_idx:
                cv2.rectangle(frame, (10, 10),
                              (frame.shape[1]-10, frame.shape[0]-10),
                              (0, 255, 0), 5)
                cv2.putText(frame, "TARGET FRAME", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Resize and display
            height, width = frame.shape[:2]
            max_width = 750
            if width > max_width:
                scale = max_width / width
                frame = cv2.resize(frame, (max_width, int(height * scale)))

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            from PIL import Image, ImageTk
            img = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image=img)

            play_label.config(image=photo)
            play_label.photo = photo  # Keep reference
            play_window.update()

            # Delay to match FPS
            time.sleep(1.0 / self.fps)

            if not play_window.winfo_exists():
                break

        if play_window.winfo_exists():
            play_window.destroy()

    def finish_labeling(self):
        """Show completion message and close"""
        n_labeled = len(self.labels)
        n_total = len(self.suggested_frames)

        messagebox.showinfo(
            "Active Learning Complete!",
            f"Labeled {n_labeled} out of {n_total} suggested frames.\n\n"
            f"Labels will be saved to your per-frame CSV file.\n"
            f"The model will now be retrained with these new labels."
        )

        self.window.destroy()

    def close_interface(self):
        """Alias for finish_labeling"""
        self.finish_labeling()


# ============================================================================
# BoutLabelingInterface — bout-level labeling (A-SOiD style)
# ============================================================================

class BoutLabelingInterface:
    """
    GUI for bout-level labeling.  Shows a looping video clip for each uncertain
    bout; user clicks YES / NO / SKIP for the entire clip.

    Returns {(start_frame, end_frame): 0_or_1} — consumed by ALSessionV2.apply_labels().
    """

    MAX_CLIP_FRAMES = 600   # cap to avoid memory issues on long clips

    def __init__(self, video_path: str, bouts: List[BoutCandidate],
                 probas: np.ndarray, behavior_name: str, fps: float,
                 log_cb=None):
        self.video_path = video_path
        self.bouts = bouts
        self.probas = probas
        self.behavior_name = behavior_name
        # Optional callback (msg:str)->None to mirror per-label confirmations into the
        # main AL log. Wrapped so a failing/None callback never breaks labeling.
        self._log_cb = log_cb
        self._n_labeled = 0   # running count of bouts/spans labeled this session (for progress)

        self.cap = cv2.VideoCapture(video_path)
        self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _cap_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = max(fps if fps > 0 else _cap_fps, 1.0)

        self._result_labels: Dict[Tuple[int, int], int] = {}
        self._current_idx = 0
        self._loop_frames: list = []   # list of (PhotoImage, x_off, y_off)
        self._loop_pos = 0
        self._paused = False
        self._after_id = None
        self._window_open = False

        # Tk widgets (set in _build_window)
        self._window = None
        self._video_canvas = None
        self._progress_label = None
        self._info_var = None
        self._trace_ax = None
        self._trace_canvas_wgt = None
        self._cursor_line = None
        self._goto_entry = None                # "Go to frame" input (set in _build_window)
        # Multi-span selection within ONE bout: a list of committed (start,end) spans
        # plus a single PENDING open mark. Mark In opens a span (sets _mark_in); Mark Out
        # closes it (appends to _mark_spans, clears _mark_in). Marking In→Out repeatedly —
        # without pressing Y between — accumulates multiple spans, all labeled at once on Y.
        # Empty _mark_spans + no pending mark = label the whole bout.
        self._mark_in: Optional[int] = None    # pending OPEN mark (absolute frame), or None
        self._mark_spans: list = []            # committed [(start,end), ...] absolute frames
        self._sel_var = None                   # tk.StringVar created in _build_window
        # Navigation state: remember each bout's spans so returning to it (Prev) restores
        # the selection; track which result keys each bout created so a re-label on revisit
        # REPLACES the old ones (fixing a mis-label) instead of duplicating.
        self._saved_marks: Dict[int, Tuple[Optional[int], list]] = {}
        self._bout_keys: Dict[int, list] = {}
        self._bout_fresh = True                # first label after (re)entering a bout wipes its old labels
        # Optional DLC keypoint overlay (resolved from each bout's video).
        self._show_dlc_var = None              # tk.BooleanVar created in _build_window
        self._pose_cache: Dict[str, dict] = {} # video_path -> {bodypart: (x, y, likelihood)}

    # ------------------------------------------------------------------
    def run(self) -> dict:
        if not self.bouts:
            return {}
        self._build_window()
        if self._window_open:
            self._load_bout(0)
            self._window.wait_window()
        self.cap.release()
        return self._result_labels

    # ------------------------------------------------------------------
    def _log(self, msg: str):
        """Mirror a confirmation line into the main AL log (if a callback was given)."""
        cb = self._log_cb
        if cb is None:
            return
        try:
            cb(msg)
        except Exception:
            pass

    def _update_progress(self):
        """Refresh the top-right progress readout: current bout, total, and running
        count of labels committed this session."""
        if self._progress_label is None:
            return
        idx = self._current_idx
        self._progress_label.config(
            text=f"Bout {idx + 1} / {len(self.bouts)}   ·   {self._n_labeled} labeled")

    # ------------------------------------------------------------------
    def _build_window(self):
        try:
            root = tk._default_root
            self._window = tk.Toplevel(root)
        except Exception:
            self._window = tk.Tk()

        self._window_open = True
        self._window.title(f"Active Learning — {self.behavior_name}")
        _sw = self._window.winfo_screenwidth()
        _sh = self._window.winfo_screenheight()
        _w = int(_sw * 0.90)
        _h = int(_sh * 0.90)
        self._window.geometry(f"{_w}x{_h}+{(_sw-_w)//2}+{(_sh-_h)//2}")
        self._window.resizable(True, True)
        self._window.protocol("WM_DELETE_WINDOW", self._on_close)

        # Header
        hdr = ttk.Frame(self._window)
        hdr.pack(fill='x', padx=10, pady=(8, 2))
        self._header_label = ttk.Label(hdr, text=f"Active Learning — {self.behavior_name}",
                                       font=(FONT_FAMILY, 13, 'bold'))
        self._header_label.pack(side='left')
        self._progress_label = ttk.Label(hdr, text="", font=(FONT_FAMILY, 12, 'bold'))
        self._progress_label.pack(side='right')

        ttk.Separator(self._window, orient='horizontal').pack(fill='x', padx=8, pady=2)

        # Video canvas — fills available width, fixed height for pre-rendered frames
        self._video_canvas = tk.Canvas(self._window, bg='black', width=900, height=520)
        self._video_canvas.pack(padx=10, pady=4)

        # Info bar
        self._info_var = tk.StringVar(value="")
        ttk.Label(self._window, textvariable=self._info_var,
                  font=(FONT_FAMILY, 9), foreground='navy').pack()

        ttk.Separator(self._window, orient='horizontal').pack(fill='x', padx=8, pady=2)

        # Probability trace
        if MATPLOTLIB_AVAILABLE:
            try:
                _fig, self._trace_ax = plt.subplots(figsize=(6.4, 0.9), dpi=90, constrained_layout=True)
                self._trace_canvas_wgt = FigureCanvasTkAgg(_fig, master=self._window)
                self._trace_canvas_wgt.get_tk_widget().pack(fill='x', padx=10, pady=2)
                _bind_tight_layout_on_resize(self._trace_canvas_wgt, _fig)
                self._trace_fig = _fig
                # Click the probability trace to seek the video to that frame.
                self._trace_canvas_wgt.mpl_connect('button_press_event', self._on_trace_click)
            except Exception:
                self._trace_ax = None
                self._trace_canvas_wgt = None

        ttk.Separator(self._window, orient='horizontal').pack(fill='x', padx=8, pady=2)

        # Label buttons
        btn_frame = ttk.Frame(self._window)
        btn_frame.pack(pady=4)
        _bs = {'bootstyle': 'success'} if _TTKBOOTSTRAP else {}
        ttk.Button(btn_frame, text="✓ YES (Y)", width=12,
                  command=lambda: self._label_bout(1), **_bs).pack(side='left', padx=8)
        _bd = {'bootstyle': 'danger'} if _TTKBOOTSTRAP else {}
        ttk.Button(btn_frame, text="✗ NO (N)", width=12,
                  command=lambda: self._label_bout(0), **_bd).pack(side='left', padx=8)
        _bsec = {'bootstyle': 'secondary'} if _TTKBOOTSTRAP else {}
        ttk.Button(btn_frame, text="◀ Prev bout [P]", width=15,
                  command=self._previous_bout, **_bsec).pack(side='left', padx=8)
        ttk.Button(btn_frame, text="? SKIP / Next [S/↵]", width=18,
                  command=self._skip_bout, **_bsec).pack(side='left', padx=8)

        # Step controls
        step_frame = ttk.Frame(self._window)
        step_frame.pack(pady=2)
        ttk.Button(step_frame, text="◀ Step (←)", width=12,
                   command=lambda: self._step_frame(-1)).pack(side='left', padx=6)
        ttk.Button(step_frame, text="Step (→) ▶", width=12,
                   command=lambda: self._step_frame(1)).pack(side='left', padx=6)
        # Jump to an exact (absolute) frame number within the current clip.
        ttk.Label(step_frame, text="Go to frame [G]:").pack(side='left', padx=(18, 3))
        self._goto_entry = ttk.Entry(step_frame, width=8)
        self._goto_entry.pack(side='left')
        self._goto_entry.bind('<Return>', lambda e: (self._goto_frame(), 'break')[1])
        ttk.Button(step_frame, text="Go",
                   command=self._goto_frame).pack(side='left', padx=6)

        # Mark In / Mark Out row
        mark_frame = ttk.Frame(self._window)
        mark_frame.pack(pady=2)
        ttk.Button(mark_frame, text="Mark In [I]", width=12,
                   command=self._set_mark_in).pack(side='left', padx=6)
        ttk.Button(mark_frame, text="Mark Out [O]", width=12,
                   command=self._set_mark_out).pack(side='left', padx=6)
        ttk.Button(mark_frame, text="Clear [C]", width=10,
                   command=self._clear_marks).pack(side='left', padx=6)

        # Overlay toggles
        tog_frame = ttk.Frame(self._window)
        tog_frame.pack(pady=2)
        self._show_dlc_var = tk.BooleanVar(value=True)   # DLC overlay ON by default
        ttk.Checkbutton(tog_frame, text="Show DLC points [D]",
                        variable=self._show_dlc_var,
                        command=self._toggle_dlc).pack(side='left', padx=6)

        # Selection status label
        self._sel_var = tk.StringVar(value="Selection: full bout")
        ttk.Label(self._window, textvariable=self._sel_var,
                  font=(FONT_FAMILY, 9), foreground='darkorange').pack(pady=1)

        ttk.Label(self._window,
                  text="Shortcuts: Enter=save marked span as YES & next  Y=Yes (stay)  "
                       "N=No (record negative)  S=Skip (record nothing, just advance)  "
                       "P=Prev bout  Space=Pause  ←/→=Step  G=Go to frame  "
                       "I=Mark In  O=Mark Out (again = extend span)  U/Bksp=Undo last mark  C=Clear all",
                  font=(FONT_FAMILY, 9), foreground='gray').pack(pady=2)

        # Wrap window-level shortcuts so they DON'T fire while the "Go to frame" box has
        # focus (otherwise typing/Enter in the entry would also trigger label/step keys).
        def g(fn):
            def _h(e):
                try:
                    if isinstance(self._window.focus_get(), (tk.Entry, ttk.Entry)):
                        return
                except Exception:
                    pass
                return fn()
            return _h

        self._window.bind('y', g(lambda: self._label_bout(1)))
        self._window.bind('Y', g(lambda: self._label_bout(1)))
        self._window.bind('n', g(lambda: self._label_bout(0)))
        self._window.bind('N', g(lambda: self._label_bout(0)))
        self._window.bind('s', g(lambda: self._skip_bout()))
        self._window.bind('S', g(lambda: self._skip_bout()))
        self._window.bind('<space>', g(lambda: self._toggle_pause()))
        self._window.bind('<Left>',  g(lambda: self._step_frame(-1)))
        self._window.bind('<Right>', g(lambda: self._step_frame(1)))
        self._window.bind('i', g(lambda: self._set_mark_in()))
        self._window.bind('I', g(lambda: self._set_mark_in()))
        self._window.bind('o', g(lambda: self._set_mark_out()))
        self._window.bind('O', g(lambda: self._set_mark_out()))
        self._window.bind('c', g(lambda: self._clear_marks()))
        self._window.bind('C', g(lambda: self._clear_marks()))
        self._window.bind('u', g(lambda: self._undo_last_mark()))
        self._window.bind('U', g(lambda: self._undo_last_mark()))
        self._window.bind('<BackSpace>', g(lambda: self._undo_last_mark()))
        self._window.bind('p', g(lambda: self._previous_bout()))
        self._window.bind('P', g(lambda: self._previous_bout()))
        self._window.bind('<Prior>', g(lambda: self._previous_bout()))  # PageUp
        self._window.bind('<Return>', g(lambda: self._on_enter()))
        self._window.bind('d', g(lambda: self._toggle_dlc(flip=True)))
        self._window.bind('D', g(lambda: self._toggle_dlc(flip=True)))
        self._window.bind('g', lambda e: self._focus_goto())
        self._window.bind('G', lambda e: self._focus_goto())

    # ------------------------------------------------------------------
    def _load_bout(self, idx: int):
        if idx >= len(self.bouts):
            self._finish()
            return

        # Remember the marks of the bout we're leaving so returning (Prev) restores them.
        if 0 <= self._current_idx < len(self.bouts):
            self._saved_marks[self._current_idx] = (self._mark_in, list(self._mark_spans))

        self._current_idx = idx
        # First label after (re)entering a bout wipes its prior recorded labels (redo).
        self._bout_fresh = True
        # Restore any previously-set marks for this bout (selection persistence).
        _saved = self._saved_marks.get(idx, (None, []))
        self._mark_in, self._mark_spans = _saved[0], list(_saved[1])
        bout = self.bouts[idx]

        # Reopen cap if this bout's video differs from current
        if bout.video_path and bout.video_path != getattr(self, '_current_video_path', None):
            self.cap.release()
            self.cap = cv2.VideoCapture(bout.video_path)
            _fps = self.cap.get(cv2.CAP_PROP_FPS)
            if _fps > 0:
                self.fps = _fps
            self._total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._current_video_path = bout.video_path

        self._update_progress()
        if bout.video_path:
            vname = os.path.basename(bout.video_path)
            self._header_label.config(
                text=f"Active Learning — {self.behavior_name}  [{vname}]")
        dur_sec = bout.duration_frames / self.fps
        _vname = os.path.basename(bout.video_path) if bout.video_path else ""
        _vpart = f"{_vname}  |  " if _vname else ""
        self._info_base = (_vpart +
                           f"Frames {bout.start_frame}–{bout.end_frame} | "
                           f"{dur_sec:.1f} sec | Mean P(1)={bout.mean_proba:.2f}")
        self._info_var.set(self._info_base + " | \u25B6 LOOPING")

        # Clamp clip to MAX_CLIP_FRAMES
        clip_start = bout.clip_start
        clip_end = min(bout.clip_end, clip_start + self.MAX_CLIP_FRAMES - 1)
        if bout.clip_end > clip_end:
            self._info_base += (f"  | TRIMMED (showing {clip_end - clip_start + 1}"
                                f" of {bout.clip_end - bout.clip_start + 1} frames)")

        # Read frame dimensions first
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
        ret, first_frame = self.cap.read()
        if not ret:
            self._advance_bout()
            return

        display_w, display_h = 900, 520
        h, w = first_frame.shape[:2]
        scale = min(display_w / w, display_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        x_off = (display_w - new_w) // 2
        y_off = (display_h - new_h) // 2

        # Pre-read all clip frames
        from PIL import Image, ImageTk
        clip_frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
        for fi in range(clip_start, clip_end + 1):
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_resized = cv2.resize(frame, (new_w, new_h))
            # Optional DLC keypoint overlay (drawn in resized coords).
            if self._show_dlc_var is not None and self._show_dlc_var.get():
                self._draw_pose(frame_resized, bout.video_path or self.video_path,
                                fi, scale)
            # Green border for core uncertain region
            if bout.start_frame <= fi <= bout.end_frame:
                cv2.rectangle(frame_resized, (2, 2), (new_w - 3, new_h - 3), (0, 200, 0), 3)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            clip_frames.append((photo, x_off, y_off))

        if not clip_frames:
            self._advance_bout()
            return

        self._loop_frames = clip_frames
        self._loop_pos = 0
        self._paused = False

        # Draw probability trace + restore the selection label for this bout's marks.
        self._update_mark_display(bout)

        # Cancel any pending loop callback, then start fresh
        if self._after_id is not None:
            try:
                self._window.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

        self._schedule_next_frame()

    def _toggle_dlc(self, flip=False):
        """Toggle the DLC overlay and re-render the current bout's clip."""
        if self._show_dlc_var is None:
            return
        if flip:
            self._show_dlc_var.set(not self._show_dlc_var.get())
        # Re-render the current bout so pre-rendered frames pick up the change.
        if self.bouts and 0 <= self._current_idx < len(self.bouts):
            self._load_bout(self._current_idx)

    def _pose_for_video(self, video_path):
        """Lazily load DLC keypoints for *video_path* → {bodypart: (x, y, likelihood)}.
        Resolves the .h5 next to the video (prefers a filtered one). Cached per video."""
        if not video_path:
            return {}
        if video_path in self._pose_cache:
            return self._pose_cache[video_path]
        bp_xy = {}
        try:
            import glob as _glob
            d = os.path.dirname(video_path)
            base = os.path.splitext(os.path.basename(video_path))[0]
            h5s = sorted(_glob.glob(os.path.join(d, f"{base}*.h5")))
            filt = [h for h in h5s if 'filtered' in os.path.basename(h).lower()]
            h5s = filt or h5s
            if h5s:
                _dlc = pd.read_hdf(h5s[0])
                try:
                    _dlc.columns = pd.MultiIndex.from_tuples(
                        [(c[1], c[2]) for c in _dlc.columns])
                except Exception:
                    pass
                for bp in _dlc.columns.get_level_values(0).unique():
                    bp_xy[bp] = (_dlc[bp]['x'].values.astype(float),
                                 _dlc[bp]['y'].values.astype(float),
                                 _dlc[bp]['likelihood'].values.astype(float))
        except Exception:
            bp_xy = {}
        self._pose_cache[video_path] = bp_xy
        return bp_xy

    _POSE_PALETTE = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
                     (255, 0, 255), (255, 255, 0), (255, 128, 0), (128, 0, 255),
                     (0, 128, 255), (128, 255, 0)]

    def _draw_pose(self, frame_bgr, video_path, frame_idx, scale, thr=0.5):
        """Draw DLC keypoints onto a resized BGR frame (coords scaled by *scale*)."""
        bp_xy = self._pose_for_video(video_path)
        if not bp_xy:
            return
        for i, (bp, (xs, ys, ls)) in enumerate(bp_xy.items()):
            if frame_idx >= len(xs) or ls[frame_idx] < thr:
                continue
            x = int(xs[frame_idx] * scale)
            y = int(ys[frame_idx] * scale)
            cv2.circle(frame_bgr, (x, y), 3,
                       self._POSE_PALETTE[i % len(self._POSE_PALETTE)], -1)

    def _schedule_next_frame(self):
        delay_ms = max(1, int(1000.0 / self.fps))
        self._after_id = self._window.after(delay_ms, self._next_frame)

    def _next_frame(self):
        if not self._window_open or not self._loop_frames:
            return
        if self._paused:
            self._after_id = self._window.after(50, self._next_frame)
            return

        photo, x_off, y_off = self._loop_frames[self._loop_pos]
        self._video_canvas.delete('all')
        self._video_canvas.create_image(x_off, y_off, anchor='nw', image=photo)
        self._video_canvas.image = photo  # keep reference

        # Update probability trace cursor
        if (self._trace_ax is not None and self._cursor_line is not None
                and self._trace_canvas_wgt is not None):
            bout = self.bouts[self._current_idx]
            abs_frame = bout.clip_start + self._loop_pos
            self._cursor_line.set_xdata([abs_frame, abs_frame])
            self._trace_canvas_wgt.draw_idle()

        self._loop_pos = (self._loop_pos + 1) % len(self._loop_frames)
        self._schedule_next_frame()

    def _draw_trace(self, bout: BoutCandidate):
        if self._trace_ax is None or self._trace_canvas_wgt is None:
            return
        try:
            self._trace_ax.clear()
            clip_probas = self.probas[bout.clip_start:bout.clip_end + 1]
            x = np.arange(bout.clip_start, bout.clip_start + len(clip_probas))
            self._trace_ax.plot(x, clip_probas, color='steelblue', linewidth=1.0)
            self._trace_ax.axhspan(0.35, 0.65, color='orange', alpha=0.2)
            self._trace_ax.axvspan(bout.start_frame, bout.end_frame,
                                   color='green', alpha=0.15)
            # User-selected sub-ranges: shade every committed span, plus a dotted line
            # at the pending (open) Mark In if one is awaiting its Mark Out.
            for (sel_s, sel_e) in self._mark_spans:
                self._trace_ax.axvspan(sel_s, sel_e, color='gold', alpha=0.40, zorder=3)
            if self._mark_in is not None:
                self._trace_ax.axvline(self._mark_in, color='goldenrod',
                                       linestyle=':', linewidth=1.4, zorder=4)
            self._trace_ax.axhline(0.5, color='black', linestyle='--', linewidth=0.8)
            self._trace_ax.set_ylim(0, 1)
            self._trace_ax.set_ylabel("P(1)", fontsize=7)
            self._trace_ax.tick_params(labelsize=6)
            self._cursor_line = self._trace_ax.axvline(bout.clip_start,
                                                        color='red', linewidth=1.0)
            self._trace_canvas_wgt.draw()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _toggle_pause(self):
        self._paused = not self._paused
        if self._paused:
            bout = self.bouts[self._current_idx]
            abs_frame = bout.clip_start + self._loop_pos
            self._info_var.set(self._info_base + f" | \u23F8 Frame {abs_frame}")
        else:
            self._info_var.set(self._info_base + " | \u25B6 LOOPING")

    def _render_loop_pos(self):
        """Show the frame at the current self._loop_pos, move the trace cursor, and
        update the info line. Assumes playback is paused (caller's responsibility)."""
        if not self._window_open or not self._loop_frames:
            return
        photo, x_off, y_off = self._loop_frames[self._loop_pos]
        self._video_canvas.delete('all')
        self._video_canvas.create_image(x_off, y_off, anchor='nw', image=photo)
        self._video_canvas.image = photo
        bout = self.bouts[self._current_idx]
        abs_frame = bout.clip_start + self._loop_pos
        if (self._trace_ax is not None and self._cursor_line is not None
                and self._trace_canvas_wgt is not None):
            self._cursor_line.set_xdata([abs_frame, abs_frame])
            self._trace_canvas_wgt.draw_idle()
        self._info_var.set(self._info_base + f" | \u23F8 Frame {abs_frame}")

    def _step_frame(self, delta: int):
        """Step one frame forward (delta=1) or backward (delta=-1). Auto-pauses."""
        if not self._window_open or not self._loop_frames:
            return
        self._paused = True
        self._loop_pos = max(0, min(len(self._loop_frames) - 1, self._loop_pos + delta))
        self._render_loop_pos()

    def _on_trace_click(self, event):
        """Click anywhere on the probability trace to seek the looping clip to that
        frame. The trace x-axis is in absolute frame indices, so the clicked
        position maps to loop_pos = round(xdata) - bout.clip_start. Auto-pauses."""
        if not self._window_open or not self._loop_frames:
            return
        if event.inaxes is not self._trace_ax or event.xdata is None:
            return
        if not (0 <= self._current_idx < len(self.bouts)):
            return
        bout = self.bouts[self._current_idx]
        target = int(round(event.xdata)) - bout.clip_start
        self._paused = True
        self._loop_pos = max(0, min(len(self._loop_frames) - 1, target))
        self._render_loop_pos()

    def _focus_goto(self):
        """Put the cursor in the Go-to-frame box (pauses playback so the jump is visible)."""
        if not self._window_open or getattr(self, '_goto_entry', None) is None:
            return
        self._paused = True
        try:
            self._goto_entry.focus_set()
            self._goto_entry.select_range(0, 'end')
        except Exception:
            pass

    def _goto_frame(self):
        """Seek the clip to an absolute frame number typed in the Go-to-frame box. The
        target is clamped to the current bout's clip range and playback is paused."""
        if not self._window_open or not self._loop_frames:
            return
        if not (0 <= self._current_idx < len(self.bouts)):
            return
        raw = ''
        try:
            raw = self._goto_entry.get().strip()
        except Exception:
            pass
        if not raw:
            return
        try:
            target_abs = int(round(float(raw)))
        except ValueError:
            if self._sel_var:
                self._sel_var.set(f"Go to frame: '{raw}' is not a number")
            return
        bout = self.bouts[self._current_idx]
        clip_lo = bout.clip_start
        clip_hi = bout.clip_start + len(self._loop_frames) - 1
        clamped = max(clip_lo, min(clip_hi, target_abs))
        self._paused = True
        self._loop_pos = clamped - bout.clip_start
        self._render_loop_pos()
        try:
            self._window.focus_set()   # release the entry so shortcuts work again
        except Exception:
            pass
        if clamped != target_abs and self._sel_var:
            self._sel_var.set(f"Go to frame: {target_abs} outside clip "
                              f"[{clip_lo}–{clip_hi}] → jumped to {clamped}")

    def _set_mark_in(self):
        if not self._window_open or not self._loop_frames:
            return
        bout = self.bouts[self._current_idx]
        # Open a new span at the current frame. If a previous Mark In is still open
        # (no Mark Out yet), just move it — you can't have two open marks at once.
        self._mark_in = bout.clip_start + self._loop_pos
        self._update_mark_display(bout)

    def _set_mark_out(self):
        if not self._window_open or not self._loop_frames:
            return
        bout = self.bouts[self._current_idx]
        cur = bout.clip_start + self._loop_pos
        if self._mark_in is None:
            # No open span. If a span is already committed, pressing Out again EXTENDS the
            # most recent span to the current frame (so you can grow/shrink it without
            # re-marking In). With nothing marked at all, prompt for Mark In first.
            if self._mark_spans:
                last_s, last_e = self._mark_spans[-1]
                # Anchor on whichever end of the last span is farther from the cursor so
                # the span always spans from that anchor to the current frame.
                anchor = last_s if abs(cur - last_s) >= abs(cur - last_e) else last_e
                s, e = sorted((anchor, cur))
                self._mark_spans[-1] = (s, e)
                self._update_mark_display(bout)
            elif self._sel_var:
                self._sel_var.set("Selection: press Mark In [i] first")
            return
        s, e = sorted((self._mark_in, cur))
        self._mark_spans.append((s, e))   # commit the span
        self._mark_in = None              # close it; ready for the next Mark In
        self._update_mark_display(bout)

    def _undo_last_mark(self):
        """Clear just the most recent mark: cancel a pending open Mark In, or if none is
        open, remove the last committed span (so you can re-mark its Out) — unlike
        Clear [C] which wipes every span on the bout."""
        if not self._window_open:
            return
        if self._mark_in is not None:
            self._mark_in = None
        elif self._mark_spans:
            self._mark_spans.pop()
        if self._current_idx < len(self.bouts):
            self._update_mark_display(self.bouts[self._current_idx])

    def _clear_marks(self):
        self._mark_in = None
        self._mark_spans = []
        if self._current_idx < len(self.bouts):
            self._update_mark_display(self.bouts[self._current_idx])

    def _update_mark_display(self, bout: BoutCandidate):
        if not self._mark_spans and self._mark_in is None:
            if self._sel_var:
                self._sel_var.set("Selection: full bout")
        else:
            parts = [f"{s}–{e}" for (s, e) in self._mark_spans]
            txt = (f"Selection: {len(self._mark_spans)} span(s): " + ", ".join(parts)) \
                if self._mark_spans else "Selection:"
            if self._mark_in is not None:
                txt += f"  | …marking from {self._mark_in} (press Mark Out)"
            if self._sel_var:
                self._sel_var.set(txt)
        self._draw_trace(bout)

    def _label_bout(self, label: int):
        if not self._window_open or self._current_idx >= len(self.bouts):
            return
        bout = self.bouts[self._current_idx]
        # Auto-close any dangling open Mark In at the current frame so a forgotten
        # Mark Out still produces a usable span instead of being silently dropped.
        if self._mark_in is not None and self._loop_frames:
            cur = bout.clip_start + self._loop_pos
            s, e = sorted((self._mark_in, cur))
            self._mark_spans.append((s, e))
            self._mark_in = None
        spans = list(self._mark_spans)
        # First label after (re)entering this bout wipes its prior recorded labels, so a
        # revisit-to-fix REPLACES the old call(s) instead of leaving stale spans behind.
        if self._bout_fresh:
            for _k in self._bout_keys.get(self._current_idx, []):
                self._result_labels.pop(_k, None)
            self._bout_keys[self._current_idx] = []
            self._bout_fresh = False
        _tag = "YES (positive)" if label == 1 else "NO (negative)"
        _vname = os.path.basename(bout.video_path) if bout.video_path else ""
        _vpart = f"{_vname} " if _vname else ""
        if not spans:
            # Full-bout label — record the whole core bout, then advance immediately.
            key = (bout.session_idx, bout.start_frame, bout.end_frame)
            self._result_labels[key] = label
            self._bout_keys.setdefault(self._current_idx, []).append(key)
            self._n_labeled += 1
            _nf = bout.end_frame - bout.start_frame + 1
            self._log(f"Bout {self._current_idx + 1}/{len(self.bouts)}: marked {_tag} — "
                      f"{_vpart}frames {bout.start_frame}–{bout.end_frame} ({_nf} frames).")
            self._update_progress()
            self._advance_bout()
        else:
            # One label per marked span; stay on this bout so more spans can be added.
            for (s, e) in spans:
                key = (bout.session_idx, s, e)
                self._result_labels[key] = label
                self._bout_keys.setdefault(self._current_idx, []).append(key)
                self._n_labeled += 1
                self._log(f"Bout {self._current_idx + 1}/{len(self.bouts)}: marked {_tag} — "
                          f"{_vpart}span {s}–{e} ({e - s + 1} frames).")
            self._update_progress()
            self._clear_marks()

    def _skip_bout(self):
        if self._window_open:
            self._advance_bout()

    def _on_enter(self):
        """Enter: if any sub-segments are marked, save them all as YES and advance to the
        next bout; otherwise just advance. (Use Y to mark YES and STAY for more segments.)"""
        if not self._window_open:
            return
        if self._mark_spans or self._mark_in is not None:
            self._label_bout(1)    # record all marked spans as YES (also clears marks)
            self._advance_bout()   # …then move to the next bout
        else:
            self._skip_bout()

    def _advance_bout(self):
        next_idx = self._current_idx + 1
        if next_idx >= len(self.bouts):
            self._finish()
        else:
            self._load_bout(next_idx)

    def _previous_bout(self):
        """Go back to the previous bout to fix a mis-label. `_load_bout` saves the
        current bout's marks and restores the previous one's selection."""
        if not self._window_open:
            return
        if self._current_idx > 0:
            self._load_bout(self._current_idx - 1)

    def _finish(self):
        if self._after_id is not None:
            try:
                self._window.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        if self._window_open:
            self._window_open = False
            try:
                self._window.destroy()
            except Exception:
                pass

    def _on_close(self):
        self._finish()


class ALRetrainWindow:
    """Popup shown while AL retrains — header status + scrolling log + live learning
    curve. Mirrors the Train tab's TrainingVisualizationWindow, AL-shaped."""

    def __init__(self, parent_root, behavior='behavior', on_next=None):
        self._root = parent_root
        self._closed = False
        self._on_next = on_next   # callback to launch the next labeling iteration
        self.win = tk.Toplevel(parent_root)
        self.win.title(f"Active Learning — Retraining ({behavior})")
        try:
            sw, sh = self.win.winfo_screenwidth(), self.win.winfo_screenheight()
            w, h = int(sw * 0.50), int(sh * 0.62)
            self.win.geometry(f"{w}x{h}+{(sw - w)//2}+{(sh - h)//2}")
        except Exception:
            pass
        self.win.transient(parent_root)
        self._header = ttk.Label(self.win, text="⏳ Retraining will now commence…",
                                 font=(FONT_FAMILY, 13, 'bold'))
        self._header.pack(pady=(10, 2))
        # Metric headline (honest F1 / AUPRC / smoothing winner), set on completion.
        self._metric_var = tk.StringVar(value="")
        self._metric_lbl = ttk.Label(self.win, textvariable=self._metric_var,
                                     font=(FONT_FAMILY, 11, 'bold'), foreground='#1a7f37')
        self._metric_lbl.pack(pady=(0, 2))
        # Hover legend explaining each headline endpoint (HONEST F1 / bout-F1 / AUPRC / …).
        self._metric_help = ttk.Label(self.win, text="ⓘ what these metrics mean",
                                      font=(FONT_FAMILY, 8), foreground='#6b7280', cursor='hand2')
        self._metric_help.pack(pady=(0, 4))
        ToolTip(self._metric_help, METRICS_HELP)
        # Next-iteration button — enabled when the retrain finishes, so the user can
        # start the next labeling round straight from this window.
        self._next_btn = None
        if on_next is not None:
            self._next_btn = ttk.Button(self.win, text="Next Iteration → (label more)",
                                        command=self._on_next_click, state='disabled')
            self._next_btn.pack(pady=(0, 4))
        # Walking 🐾 loading indicator (replaces the marquee bar): paws march left→right
        # while indeterminate; fill proportionally during the sweep (set_progress).
        self._PAW_MAX = 8
        self._paw_n = 0
        self._paw_running = False
        try:
            _bg = self.win.cget('bg')
        except Exception:
            _bg = None
        self._paw_canvas = tk.Canvas(self.win, height=40, width=440,
                                     highlightthickness=0, **({'bg': _bg} if _bg else {}))
        self._paw_canvas.pack(pady=(0, 4))
        if MATPLOTLIB_AVAILABLE:
            # Two panels: per-fold P/R/F1 bars (left) + learning curve (right).
            self._fig, (self._ax_folds, self._ax) = plt.subplots(
                1, 2, figsize=(9, 3), dpi=90, constrained_layout=True,
                gridspec_kw={'width_ratios': [1.0, 1.3]})
            self._ax_folds.set_title("Per-fold metrics", fontsize=9)
            self._ax_folds.text(0.5, 0.5, "(awaiting CV…)", ha='center', va='center',
                                fontsize=8, color='gray', transform=self._ax_folds.transAxes)
            self._ax_folds.set_xticks([]); self._ax_folds.set_yticks([])
            self._canvas = FigureCanvasTkAgg(self._fig, master=self.win)
            self._canvas.get_tk_widget().pack(fill='both', expand=True, padx=8, pady=4)
        else:
            self._fig = self._ax = self._ax_folds = self._canvas = None
        self._log = scrolledtext.ScrolledText(self.win, height=9, wrap='word',
                                              font=('Consolas', 9))
        self._log.pack(fill='both', expand=True, padx=8, pady=(4, 8))
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)
        # Surface above the main window so the user sees retraining is underway.
        try:
            self.win.lift()
            self.win.attributes('-topmost', True)
            self.win.after(400, lambda: self.win.attributes('-topmost', False))
        except Exception:
            pass

    def _on_close(self):
        self._closed = True
        try:
            self.win.destroy()
        except Exception:
            pass

    def _on_next_click(self):
        """Launch the next labeling iteration (callback into the AL panel)."""
        if not self._on_next:
            return
        try:
            self._next_btn.configure(state='disabled')
        except Exception:
            pass
        try:
            self._on_next()
        except Exception:
            pass

    def log(self, msg):
        if self._closed:
            return
        try:
            self._log.insert('end', msg + "\n")
            self._log.see('end')
        except Exception:
            pass

    def _draw_paws(self, n, done=False):
        """Draw n 🐾 rotated 90° (angle=270 → toes point right) marching left→right."""
        if self._closed:
            return
        try:
            self._paw_canvas.delete('all')
            for i in range(n):
                x = 16 + i * 46
                y = 20 + (4 if i % 2 else -4)   # alternate up/down → walking steps
                self._paw_canvas.create_text(x, y, text="🐾", angle=270,
                                             font=(FONT_FAMILY, 15))
            if done:
                self._paw_canvas.create_text(16 + n * 46 + 6, 20, text="✓",
                                             font=(FONT_FAMILY, 14, 'bold'), fill='#2ca02c')
        except Exception:
            self._paw_running = False

    def _animate_paws(self):
        if self._closed or not self._paw_running:
            return
        self._paw_n = (self._paw_n % self._PAW_MAX) + 1
        self._draw_paws(self._paw_n)
        try:
            self.win.after(320, self._animate_paws)
        except Exception:
            self._paw_running = False

    def start_busy(self):
        if self._closed:
            return
        self._paw_running = True
        self._animate_paws()

    def stop_busy(self):
        if self._closed:
            return
        self._paw_running = False
        self._draw_paws(self._PAW_MAX, done=True)

    def set_progress(self, done, total):
        """Determinate progress (e.g. sweep thresholds done/total) → proportional paw-fill."""
        if self._closed:
            return
        self._paw_running = False   # stop the walk; show progress as a paw fill
        frac = float(done) / max(float(total), 1.0)
        n = max(1, min(self._PAW_MAX, int(round(frac * self._PAW_MAX))))
        self._draw_paws(n)

    def set_stage(self, msg):
        if self._closed:
            return
        try:
            self._header.config(text=f"⏳ {msg}")
        except Exception:
            pass
        self.start_busy()   # resume walking paws for this stage

    def set_metrics(self, text):
        """Show the headline metrics (honest F1 / AUPRC / smoothing winner)."""
        if self._closed:
            return
        try:
            self._metric_var.set(text)
        except Exception:
            pass

    def set_done(self, ok=True):
        if self._closed:
            return
        self.stop_busy()
        try:
            self._header.config(text="✓ Retraining complete — review the curve below"
                                if ok else "Retraining ended")
        except Exception:
            pass
        if ok and self._next_btn is not None:
            try:
                self._next_btn.configure(state='normal')
            except Exception:
                pass

    def reset_for_new_run(self):
        """Prepare a REUSED popup for a fresh retrain iteration: clear the prior
        headline metrics, re-disable the Next button, resume the busy animation and
        bring the window forward — so 'Next Iteration → retrain' updates this window
        in place instead of opening a new one."""
        if self._closed:
            return
        try:
            self._metric_var.set("")
        except Exception:
            pass
        try:
            self._header.config(text="⏳ Retraining will now commence…")
        except Exception:
            pass
        if self._next_btn is not None:
            try:
                self._next_btn.configure(state='disabled')
            except Exception:
                pass
        try:
            self.win.lift()
            self.win.attributes('-topmost', True)
            self.win.after(400, lambda: self.win.attributes('-topmost', False))
        except Exception:
            pass
        self.start_busy()

    def draw_curve(self, draw_fn):
        """draw_fn(ax) renders the learning curve into our embedded axes."""
        if self._closed or self._ax is None:
            return
        try:
            self._ax.clear()
            draw_fn(self._ax)
            self._canvas.draw()
        except Exception:
            pass

    def draw_folds(self, fold_detail, cv_mode='session', at_operating_point=False):
        """Grouped bar chart of per-fold F1/Precision/Recall; x-tick labels name the
        held-out session(s) so the train/test split is visible at a glance.
        at_operating_point=True → these are the nested-LOFO swept metrics (matches the
        honest headline); False → the raw @0.5 preview shown live during CV."""
        if self._closed or getattr(self, '_ax_folds', None) is None:
            return
        try:
            import numpy as _np
            ax = self._ax_folds
            ax.clear()
            n = len(fold_detail)
            if n == 0:
                ax.text(0.5, 0.5, "(no folds)", ha='center', va='center',
                        fontsize=8, color='gray', transform=ax.transAxes)
                self._canvas.draw(); return
            x = _np.arange(n); w = 0.26
            f1 = [d['f1'] for d in fold_detail]
            pr = [d['precision'] for d in fold_detail]
            rc = [d['recall'] for d in fold_detail]
            ax.bar(x - w, f1, w, label='F1', color='#2c7fb8')
            ax.bar(x,     pr, w, label='Precision', color='#7fcdbb')
            ax.bar(x + w, rc, w, label='Recall', color='#fdae61')
            labs = []
            for i, d in enumerate(fold_detail):
                h = d.get('held') or f"fold {i+1}"
                if len(h) > 14:
                    h = h[:6] + "…" + h[-6:]
                labs.append(h)
            ax.set_xticks(x)
            ax.set_xticklabels(labs, rotation=30, ha='right', fontsize=7)
            ax.set_ylim(0, 1.0)
            ax.set_ylabel('score', fontsize=8)
            _base = "Per-fold @ operating point" if at_operating_point else "Per-fold @0.5 (raw)"
            _ttl = _base + (f"  (held-out {cv_mode})" if cv_mode else "")
            ax.set_title(_ttl, fontsize=9)
            ax.legend(fontsize=7, loc='lower right', ncol=3, framealpha=0.6)
            ax.tick_params(axis='y', labelsize=7)
            ax.grid(axis='y', alpha=0.3)
            self._canvas.draw()
        except Exception:
            pass


# ===========================================================================
# ActiveLearningTabV2 — main tab class (moved from PixelPaws_GUI.py)
# ===========================================================================

class ActiveLearningTabV2(ttk.Frame):
    """
    Active Learning v2 tab.
    Layout: horizontal PanedWindow — left=controls, right=plot+log.
    """
    def __init__(self, parent, parent_app):
        super().__init__(parent)
        self.app = parent_app
        self.pack(fill='both', expand=True)

        # State
        self._session = None
        self._last_probas = None
        self._last_model = None
        self._last_frames = None
        self._n_labeled_at_load = 0
        self._sessions_list = []
        self._clf_options = {}
        self._base_clf_f1 = None   # CV F1 of pre-loaded classifier (baseline for plot)

        # SharedVars
        self._threshold_var = tk.DoubleVar(value=0.30)
        self._n_suggestions_var = tk.IntVar(value=10)
        self._n_suggestions_var.trace_add('write', lambda *a: self._update_bout_estimate())
        self._min_spacing_var = tk.IntVar(value=5)
        self._context_frames_var = tk.IntVar(value=10)
        self._max_bout_var = tk.IntVar(value=0)   # 0 = adaptive
        # Auto-populated bout-length indicators (set in _build_left); guard so the
        # auto-set doesn't clear its own indicator via the manual-edit trace.
        self._min_bout_ind = None
        self._max_bout_ind = None
        self._bout_auto_guard = False
        self._min_spacing_var.trace_add('write', lambda *a: self._on_bout_var_edit('min'))
        self._max_bout_var.trace_add('write', lambda *a: self._on_bout_var_edit('max'))
        self._eligible_count_var = tk.StringVar(value="— not scored yet —")
        self._bout_aware_cv_var = tk.BooleanVar(value=True)
        self._propagate_var = tk.BooleanVar(value=False)
        self._class_balanced_var = tk.BooleanVar(value=True)
        self._diversity_radius_var = tk.IntVar(value=0)
        self._auto_iter_var = tk.IntVar(value=3)
        self._auto_remaining = 0
        self._stop_auto_var = tk.BooleanVar(value=False)
        self._include_unlabeled_var = tk.BooleanVar(value=True)   # include label-less pool sessions by default
        self._n_unlabeled_var = tk.IntVar(value=0)               # cap unlabeled pool videos (0=all)
        self._clf_bout_seed = {}                                  # bout params seeded from warm-start clf
        self._retrain_window = None                               # ALRetrainWindow popup during retrain
        self._last_score_stats = None                             # cached eligibility stats for live bout-count estimate
        self._trim_to_positive_var = tk.BooleanVar(value=False)   # OFF for AL: every 0 is a deliberate reviewed negative; only ON to clean imported BORIS trailing-zeros
        self._n_folds_var = tk.IntVar(value=5)                    # Train-tab parity: CV fold count (mirrors train_n_folds)
        self._test_session_names = set()                         # sessions held out as a fixed test set (never queried)
        self._auto_holdout_var = tk.BooleanVar(value=False)      # auto-reserve a fraction of sessions as test
        self._auto_holdout_pct_var = tk.IntVar(value=20)         # fraction (%) to auto-reserve
        self._btn_next_iter = None  # reference set in _build_left

        self._build_ui()

        # React to project changes
        self.app.current_project_folder.trace_add('write', lambda *_: self._on_project_changed())

    def _build_ui(self):
        # Header
        hdr = ttk.Frame(self)
        hdr.pack(fill='x', padx=10, pady=(8, 2))
        ttk.Label(hdr, text="🧠 Active Learning v2",
                  font=(FONT_FAMILY, 14, 'bold')).pack(side='left')

        paned = ttk.PanedWindow(self, orient='horizontal')
        paned.pack(fill='both', expand=True, padx=6, pady=4)

        left = ttk.Frame(paned, width=300)
        right = ttk.Frame(paned, width=500)
        paned.add(left, weight=1)
        paned.add(right, weight=2)

        self._build_left(left)
        self._build_right(right)

    def _build_left(self, parent):
        # Sessions
        sf = ttk.LabelFrame(parent, text="Sessions", padding=5)
        sf.pack(fill='both', expand=True, padx=4, pady=4)

        btn_row = ttk.Frame(sf)
        btn_row.pack(fill='x', pady=(0, 4))
        ttk.Button(btn_row, text="🔄 Scan", width=8,
                   command=self._scan_sessions).pack(side='left', padx=(0, 4))
        _btn_all = ttk.Button(btn_row, text="✓ All", width=6,
                              command=self._select_all_sessions)
        _btn_all.pack(side='left', padx=(0, 4))
        ToolTip(_btn_all, "Select every scanned session.")
        _btn_unl = ttk.Button(btn_row, text="✓ Unlbl", width=7,
                              command=self._select_unlabeled_sessions)
        _btn_unl.pack(side='left', padx=(0, 4))
        ToolTip(_btn_unl, "Select only the feature-ready UNLABELED videos. Score with a warm-start "
                          "classifier to query bouts from just the new videos (the classifier carries "
                          "the labeled-data knowledge). Tip: bout queries always come from -1 "
                          "(unlabeled) frames, so fully-labeled videos are never re-queried anyway.")
        _btn_test = ttk.Button(btn_row, text="⊘ Test set", width=10,
                               command=self._toggle_test_sessions)
        _btn_test.pack(side='left', padx=(0, 4))
        ToolTip(_btn_test, "Mark/unmark the selected session(s) as a fixed held-out TEST set. "
                           "Test sessions are never queried or trained on; each iteration the "
                           "model is evaluated on their full ground truth to give an unbiased F1 "
                           "(the orange line on the learning curve), directly comparable to a "
                           "Train-tab F1. Needs labeled sessions.")

        _cb_unl = ttk.Checkbutton(btn_row, text="Incl. unlabeled",
                                  variable=self._include_unlabeled_var,
                                  command=self._scan_sessions)
        _cb_unl.pack(side='left')
        ToolTip(_cb_unl, "Also list sessions with NO label CSV yet (unlabeled pool). These need a "
                         "warm-start classifier to score; a fresh CSV is written on first save.")
        _spn_unl = ttk.Spinbox(btn_row, from_=0, to=999, width=4,
                               textvariable=self._n_unlabeled_var,
                               command=self._scan_sessions)
        _spn_unl.pack(side='left', padx=(4, 1))
        ttk.Label(btn_row, text="use (0=all)").pack(side='left')
        ToolTip(_spn_unl, "Cap how many UNLABELED pool videos enter active learning (0 = all). "
                          "Work a manageable batch at a time — fewer videos to score and label. "
                          "The N are picked deterministically + spread across the cohort; all "
                          "labeled sessions are always kept.")

        # Save / load session row
        sess_row = ttk.Frame(sf)
        sess_row.pack(fill='x', pady=(2, 0))
        _btn_save = ttk.Button(sess_row, text="💾 Save session", command=self._save_al_session)
        _btn_save.pack(side='left', padx=(0, 4))
        _btn_load = ttk.Button(sess_row, text="📂 Load session", command=self._load_al_session)
        _btn_load.pack(side='left')
        ToolTip(_btn_save, "Save this AL session — the exact selected videos, the held-out test set, "
                           "the warm-start classifier and CV params — so you can reload the SAME "
                           "train/test split later for reproducible/comparable F1. (Also auto-saved "
                           "each time you Score.) The learning curve persists alongside it.")
        ToolTip(_btn_load, "Reload a saved AL session: restores the exact video selection, test set, "
                           "classifier and params, and redraws the saved learning curve — so you "
                           "continue where you left off on the same data.")

        # Auto hold-out row
        ho_row = ttk.Frame(sf)
        ho_row.pack(fill='x', pady=(2, 0))
        _cb_ho = ttk.Checkbutton(ho_row, text="Auto hold-out", variable=self._auto_holdout_var)
        _cb_ho.pack(side='left')
        ttk.Spinbox(ho_row, from_=5, to=50, textvariable=self._auto_holdout_pct_var,
                    width=4).pack(side='left', padx=(4, 1))
        ttk.Label(ho_row, text="% as test").pack(side='left')
        ToolTip(_cb_ho, "When ON (and no session is manually marked as Test), Score automatically "
                        "reserves this fraction of the selected LABELED sessions as a fixed held-out "
                        "test set — excluded from querying/training, plotted as the orange 'Held-out F1' "
                        "line. Deterministic (same sessions each run). Off by default so the default "
                        "behaviour trains on all sessions + session-level CV, like the Train tab.")

        lb_frame = ttk.Frame(sf)
        lb_frame.pack(fill='both', expand=True)
        # exportselection=False: keep the video selection when another widget (the
        # classifier combobox, etc.) claims the X selection — otherwise picking a
        # classifier would silently clear the selected videos.
        self._session_lb = tk.Listbox(lb_frame, selectmode='extended', height=5,
                                      exportselection=False)
        lb_sb = ttk.Scrollbar(lb_frame, command=self._session_lb.yview)
        self._session_lb.configure(yscrollcommand=lb_sb.set)
        self._session_lb.pack(side='left', fill='both', expand=True)
        lb_sb.pack(side='right', fill='y')

        # Classifier
        cf = ttk.LabelFrame(parent, text="Classifier (for scoring)", padding=5)
        cf.pack(fill='x', padx=4, pady=4)
        clf_row = ttk.Frame(cf)
        clf_row.pack(fill='x')
        self._clf_combo = ttk.Combobox(clf_row, state='readonly', width=28)
        self._clf_combo.pack(side='left', padx=(0, 4))
        self._clf_combo.bind('<<ComboboxSelected>>', lambda e: self._on_clf_selected())
        ttk.Button(clf_row, text="↺", width=3,
                   command=self._refresh_classifiers).pack(side='left')
        ttk.Button(clf_row, text="📁", width=3,
                   command=self._browse_classifier).pack(side='left', padx=(2, 0))
        # Hide the per-run pruned/ and all_features/ variant pkls by default — only the
        # primary classifier per run is shown unless this is ticked (declutters the list).
        self._clf_show_variants_var = tk.BooleanVar(value=False)
        _cb_var = ttk.Checkbutton(clf_row, text="variants",
                                  variable=self._clf_show_variants_var,
                                  command=self._refresh_classifiers)
        _cb_var.pack(side='left', padx=(6, 0))
        ToolTip(_cb_var, "Also list each run's pruned / all-features variant classifiers. "
                         "Off by default — only the primary classifier per training run is shown.")
        ToolTip(self._clf_combo,
                "Optional: select a pre-trained classifier (.pkl) to score frames. "
                "Labels read 'behavior · F1 · date'; the best-F1 one is auto-selected. "
                "If left blank, the tab trains a fresh model from your current labels.")

        # Parameters
        pf = ttk.LabelFrame(parent, text="Parameters", padding=5)
        pf.pack(fill='x', padx=4, pady=4)

        def _row(parent, label, var, from_, to, width=6, tooltip=None):
            r = ttk.Frame(parent)
            r.pack(fill='x', pady=1)
            lbl = ttk.Label(r, text=label, width=22)
            lbl.pack(side='left')
            spx = ttk.Spinbox(r, from_=from_, to=to, textvariable=var, width=width)
            spx.pack(side='left')
            if tooltip:
                ToolTip(lbl, tooltip)
                ToolTip(spx, tooltip)
            return r

        _row(pf, "Bouts / iteration:", self._n_suggestions_var, 1, 200,
             tooltip="Number of uncertain video clips to present per labeling round. Lower = shorter sessions; higher = more frames labeled per click.")
        _minrow = _row(pf, "Min bout frames:", self._min_spacing_var, 1, 100,
             tooltip="Minimum number of consecutive uncertain frames required to form a bout. Shorter runs are ignored.")
        self._min_bout_ind = ttk.Label(_minrow, text="", font=(FONT_FAMILY, 8),
                                       foreground='#1a7f37')
        self._min_bout_ind.pack(side='left', padx=(6, 0))
        _row(pf, "Context frames:", self._context_frames_var, 0, 300,
             tooltip="Extra frames shown before and after the uncertain region so you can see the behavior in context. Does not affect which frames get labeled.")
        _maxrow = _row(pf, "Max bout frames (0=auto):", self._max_bout_var, 0, 2000,
             tooltip="Cap the maximum clip length. 0 = auto (uses 90th-percentile of positive bout lengths). Increase if bouts are being cut short; decrease to avoid very long clips.")
        self._max_bout_ind = ttk.Label(_maxrow, text="", font=(FONT_FAMILY, 8),
                                       foreground='#1a7f37')
        self._max_bout_ind.pack(side='left', padx=(6, 0))
        _row(pf, "CV folds:", self._n_folds_var, 2, 10,
             tooltip="Cross-validation fold count for the held-out F1, mirroring the Train tab (default 5). Uses session-level folds when there are at least this many labeled sessions, else falls back to bout-level grouping.")

        _cb_ba = ttk.Checkbutton(pf, text="Leave-animals-out CV", variable=self._bout_aware_cv_var)
        _cb_ba.pack(anchor='w', pady=(4, 0))
        ToolTip(_cb_ba, "Use bout-grouped cross-validation (GroupKFold) instead of frame-level "
                        "StratifiedKFold. Recommended — prevents data leakage across bouts. "
                        "Auto-falls back to frame-level when too few bout groups exist.")

        _cb_cb = ttk.Checkbutton(pf, text="Class-balanced queries",
                                  variable=self._class_balanced_var)
        _cb_cb.pack(anchor='w', pady=(2, 0))
        ToolTip(_cb_cb, "Alternate positive-predicted and negative-predicted bouts in each "
                        "query round, so labels stay balanced even when one class dominates.")

        _cb_trim = ttk.Checkbutton(pf, text="Trim train set to positive span",
                                   variable=self._trim_to_positive_var)
        _cb_trim.pack(anchor='w', pady=(2, 0))
        ToolTip(_cb_trim, "OFF by default for active learning — every 0 you label is a deliberate "
                          "reviewed negative (incl. far-from-positive bouts AL surfaces), so all "
                          "labels are kept. Turn ON only to clean an IMPORTED BORIS/Targets CSV whose "
                          "frames past the last annotation are unreviewed trailing zeros: it then "
                          "drops frames before the first / after the last positive from TRAINING "
                          "(scoring still uses the whole video). The Log reports how many it drops.")

        # Label-bout-length filter — shares the SAME vars as the Train tab, so a value
        # set in either tab applies in both (drops too-short/long labeled bouts).
        try:
            _lbmin = self.app.train_min_label_bout
            _lbmax = self.app.train_max_label_bout
            _lbrow = ttk.Frame(pf)
            _lbrow.pack(anchor='w', pady=(2, 0))
            ttk.Label(_lbrow, text="Exclude labeled bouts  <").pack(side='left')
            ttk.Spinbox(_lbrow, from_=0, to=100000, width=5,
                        textvariable=_lbmin).pack(side='left', padx=2)
            ttk.Label(_lbrow, text="fr  or  >").pack(side='left')
            ttk.Spinbox(_lbrow, from_=0, to=100000, width=5,
                        textvariable=_lbmax).pack(side='left', padx=2)
            ttk.Label(_lbrow, text="fr").pack(side='left')
            ToolTip(_lbrow, "Drop accidentally short / implausibly long labeled positive bouts "
                            "from TRAINING (set to -1, in-memory only — CSV untouched). 0 = off. "
                            "Shared with the Train Classifier tab.")
        except Exception:
            pass

        # CV eligibility (shared vars) — sparse sessions train-only, not held out for eval.
        try:
            _cverow = ttk.Frame(pf)
            _cverow.pack(anchor='w', pady=(2, 0))
            ttk.Label(_cverow, text="CV eligibility:").pack(side='left')
            ttk.Combobox(_cverow, textvariable=self.app.train_cv_eligibility_mode,
                         values=['auto', 'manual', 'off'], state='readonly',
                         width=8).pack(side='left', padx=3)
            ttk.Label(_cverow, text="manual ≥").pack(side='left')
            ttk.Spinbox(_cverow, from_=0, to=100000, width=4,
                        textvariable=self.app.train_min_cv_pos_bouts).pack(side='left', padx=2)
            ttk.Label(_cverow, text="bouts").pack(side='left')
            ToolTip(_cverow, "Sparse sessions (too few positive events) are used for TRAINING but "
                             "not held out for evaluation — stabilizes rare-behavior F1 without "
                             "wasting labels. Leave 'auto'. Shared with the Train Classifier tab.")
        except Exception:
            pass

        _div_row = ttk.Frame(pf)
        _div_row.pack(fill='x', pady=(2, 0))
        _div_lbl = ttk.Label(_div_row, text="Diversity radius (0=off):", width=22)
        _div_lbl.pack(side='left')
        _div_spx = ttk.Spinbox(_div_row, from_=0, to=500,
                                textvariable=self._diversity_radius_var, width=6)
        _div_spx.pack(side='left')
        ToolTip(_div_lbl, "Minimum frame gap between bouts in a query. "
                          "0 = disabled. Increase (e.g. 150) to spread queries across the video.")
        ToolTip(_div_spx, "Minimum frame gap between bouts in a query. "
                          "0 = disabled. Increase (e.g. 150) to spread queries across the video.")

        _cb_prop = ttk.Checkbutton(pf, text="Label propagation (cosine sim)",
                                    variable=self._propagate_var)
        _cb_prop.pack(anchor='w', pady=(2, 0))
        ToolTip(_cb_prop, "After labeling a frame, auto-label nearby frames with cosine "
                          "similarity ≥ 0.92 in feature space. Speeds up labeling "
                          "when behavior is temporally clustered.")

        _btn_auto = ttk.Button(pf, text="🔍 Auto-detect from labels",
                               command=self._auto_detect_bout_lengths)
        _btn_auto.pack(fill='x', pady=(4, 0))
        ToolTip(_btn_auto, "Scan positive labels in the selected session(s) to detect actual bout lengths "
                           "and set Min/Max bout frames automatically.")

        # Threshold
        tf = ttk.LabelFrame(parent, text="Uncertainty Threshold", padding=5)
        tf.pack(fill='x', padx=4, pady=4)
        thresh_row = ttk.Frame(tf)
        thresh_row.pack(fill='x')
        _thresh_lbl = ttk.Label(thresh_row, text="Threshold:")
        _thresh_lbl.pack(side='left')
        _thresh_scale = ttk.Scale(thresh_row, from_=0.05, to=1.0, variable=self._threshold_var,
                                  orient='horizontal', length=150,
                                  command=lambda _: self._update_eligible_count())
        _thresh_scale.pack(side='left', padx=4)
        ToolTip(_thresh_lbl, "Frames whose model confidence is within this distance of P=0.5 are considered uncertain and eligible for labeling.")
        ToolTip(_thresh_scale, "Frames whose model confidence is within this distance of P=0.5 are considered uncertain and eligible for labeling.")
        ttk.Label(thresh_row, textvariable=tk.StringVar()).pack(side='left')  # placeholder
        ttk.Label(tf, textvariable=self._eligible_count_var,
                  font=(FONT_FAMILY, 9), foreground='navy').pack(anchor='w', pady=2)

        # Buttons
        btn_f = ttk.LabelFrame(parent, text="Actions", padding=5)
        btn_f.pack(fill='x', padx=4, pady=4)
        _btn_score = ttk.Button(btn_f, text="1. Score + Histogram",
                                command=self._score_and_histogram)
        _btn_score.pack(fill='x', pady=2)
        ToolTip(_btn_score, "Train a model on current labels, score every frame, and open the confidence distribution chart.")
        _btn_label = ttk.Button(btn_f, text="2. Start Labeling",
                                command=self._start_labeling)
        _btn_label.pack(fill='x', pady=2)
        ToolTip(_btn_label, "Find the most uncertain video clips and open the bout-labeling interface.")
        _btn_retrain = ttk.Button(btn_f, text="Retrain & Save Snapshot",
                                   command=self._retrain_and_compare)
        _btn_retrain.pack(fill='x', pady=2)
        ToolTip(_btn_retrain, "Retrain on all current labels, save a snapshot pkl to classifiers/, and update the learning curve.")
        self._btn_next_iter = ttk.Button(btn_f, text="Next Iteration →",
                                         command=self._start_labeling,
                                         state='disabled')
        self._btn_next_iter.pack(fill='x', pady=2)
        ToolTip(self._btn_next_iter, "Score frames with the latest model and open another "
                                     "labeling round. Enabled after first scoring or retrain.")

        auto_row = ttk.Frame(btn_f)
        auto_row.pack(fill='x', pady=(4, 2))
        ttk.Button(auto_row, text="🔁 Auto-iterate",
                   command=self._auto_iterate).pack(side='left', fill='x', expand=True)
        ttk.Spinbox(auto_row, from_=1, to=20, textvariable=self._auto_iter_var,
                    width=4).pack(side='left', padx=(4, 0))
        ttk.Label(auto_row, text="iters").pack(side='left', padx=(2, 0))
        ttk.Button(btn_f, text="⏹ Stop Auto",
                   command=lambda: self._stop_auto_var.set(True)).pack(fill='x', pady=(0, 2))

        _btn_disc = ttk.Button(btn_f, text="3. Run Discovery",
                               command=self._run_discovery)
        _btn_disc.pack(fill='x', pady=2)
        ToolTip(_btn_disc, "Use UMAP + HDBSCAN to find sub-behaviors within your positive-labeled frames.")

        _btn_clear = ttk.Button(btn_f, text="🗑 Clear AL progress",
                                command=self._clear_al_progress)
        _btn_clear.pack(fill='x', pady=(8, 2))
        ToolTip(_btn_clear, "Restart fresh: reverts AL-added labels to the pre-AL backup, deletes the "
                            "learning curve and the saved PixelPaws_<behavior>_AL.pkl. Irreversible — "
                            "you'll be warned first. The curve then re-anchors on the honest baseline.")

    def _build_right(self, parent):
        # Learning curve plot
        plot_lf = ttk.LabelFrame(parent, text="Learning Curve", padding=4)
        plot_lf.pack(fill='x', padx=4, pady=4)

        if MATPLOTLIB_AVAILABLE:
            self._lc_fig, self._lc_ax = plt.subplots(figsize=(5, 2.5), dpi=90,
                                                      constrained_layout=True)
            self._lc_canvas = FigureCanvasTkAgg(self._lc_fig, master=plot_lf)
            self._lc_canvas.get_tk_widget().pack(fill='both', expand=True)
            _bind_tight_layout_on_resize(self._lc_canvas, self._lc_fig)
            self._draw_empty_curve()
        else:
            ttk.Label(plot_lf, text="(install matplotlib to see learning curve)").pack()

        # Log
        log_lf = ttk.LabelFrame(parent, text="Log", padding=4)
        log_lf.pack(fill='both', expand=True, padx=4, pady=4)
        from tkinter import scrolledtext
        self._log = scrolledtext.ScrolledText(log_lf, height=14, wrap='word',
                                              font=('Consolas', 9))
        self._log.pack(fill='both', expand=True)

    # ------------------------------------------------------------------
    # Project / session / classifier helpers
    # ------------------------------------------------------------------

    def _on_project_changed(self):
        self._scan_sessions()
        self._refresh_classifiers()

    def _scan_sessions(self):
        folder = self.app.current_project_folder.get()
        if not folder or not os.path.isdir(folder):
            return
        if not _FIND_SESSIONS_AVAILABLE:
            self._log_msg("Session discovery unavailable (evaluation_tab not found)")
            return
        try:
            # recursive=True descends into nested sub-folders (e.g. per-cohort
            # subject folders). "Incl. unlabeled" also lists label-less but
            # feature-ready sessions (the AL pool).
            include_unlabeled = self._include_unlabeled_var.get()
            sessions = find_session_triplets(folder, prefer_filtered=True,
                                             require_labels=not include_unlabeled,
                                             recursive=True)
            if include_unlabeled:
                sessions = [s for s in sessions if self._get_features_cache(s)]
            self._sessions_list = sessions
            n_missing_cache = self._refresh_session_listbox()
            n_unl = sum(1 for s in sessions if not self._session_has_labels(s))
            _smsg = f"Scanned: {len(sessions)} session(s) found"
            if include_unlabeled:
                _smsg += f" ({n_unl} unlabeled, feature-ready)"
                _cap = int(self._n_unlabeled_var.get())
                if _cap > 0 and n_unl > _cap:
                    _smsg += f"  [will use {_cap} unlabeled at Score]"
            self._log_msg(_smsg + ".")
            if n_missing_cache > 0:
                self._log_msg(f"\u26a0 {n_missing_cache} session(s) missing feature cache — extract features first (Train tab).")
            # Sessions (re)loaded → if a classifier is selected, auto-populate bout lengths.
            self._auto_bouts_from_labels()
        except Exception as e:
            self._log_msg(f"Scan error: {e}")

    @staticmethod
    def _session_has_labels(s):
        tgt = s.get('target_path') or s.get('labels')
        return bool(tgt and os.path.isfile(tgt))

    def _select_all_sessions(self):
        if self._session_lb.size() > 0:
            self._session_lb.selection_set(0, 'end')
            self._log_msg(f"Selected all {self._session_lb.size()} session(s).")

    def _select_unlabeled_sessions(self):
        """Select only the feature-ready UNLABELED videos (the AL query targets)."""
        self._session_lb.selection_clear(0, 'end')
        n = 0
        for i, s in enumerate(self._sessions_list):
            if (i < self._session_lb.size()
                    and not self._session_has_labels(s)
                    and self._get_features_cache(s)):
                self._session_lb.selection_set(i)
                n += 1
        if n:
            self._log_msg(f"Selected {n} unlabeled (feature-ready) video(s) — Score with a "
                          f"warm-start classifier to query bouts from just these.")
        else:
            self._log_msg("No unlabeled feature-ready videos listed — tick 'Incl. unlabeled' and Scan.")

    def _update_bout_estimate(self, *args):
        """Recompute the '+frames / %' estimate for the current 'Bouts / iteration'
        value from cached score stats — lets the user dial the bout count up/down
        AFTER scoring and immediately see how the labeled set would grow, no re-score."""
        st = getattr(self, '_last_score_stats', None)
        if not st:
            return
        try:
            n_suggest = int(self._n_suggestions_var.get())
        except Exception:
            return
        n_bouts = st['n_bouts']
        n_label = min(n_suggest, max(n_bouts, 1))          # can't label more than exist
        pct_bouts = 100.0 * n_label / max(n_bouts, 1)       # share of the uncertain backlog
        est_add = int(st['avg_len'] * n_label)              # approx frames added
        est = (f"  ·  label {n_label} of {n_bouts} uncertain bouts ({pct_bouts:.0f}%) "
               f"≈ +{est_add:,} frames")
        self._eligible_count_var.set(
            f"{st['n_pos']:,} pos labeled | {st['n_eligible']:,} uncertain in {n_bouts} bouts{est}")

    def _behavior_name(self):
        b = getattr(self._session, 'behavior_name', None) if self._session else None
        if not b:
            b = (self._load_selected_classifier_data() or {}).get('Behavior_type')
        return b or 'behavior'

    def _al_session_path(self, behavior=None):
        folder = self.app.current_project_folder.get() or '.'
        b = behavior or self._behavior_name()
        return os.path.join(folder, 'features', f'{b}_al_session.json')

    def _save_al_session(self, silent=False):
        """Persist the exact AL session (selected videos + test set + classifier +
        params) so the SAME train/test split can be reloaded for comparable F1."""
        import json
        folder = self.app.current_project_folder.get()
        if not folder:
            if not silent:
                messagebox.showwarning("No project", "Open a project first.")
            return
        behavior = self._behavior_name()
        selected = self._get_selected_sessions() or list(self._sessions_list)
        data = {
            'behavior':          behavior,
            'selected':          [s['session_name'] for s in selected],
            'test':              sorted(self._test_session_names),
            'classifier':        self._clf_combo.get(),
            # Persist the resolved path too — the display label format can change
            # (F1/date/variant), so reload matches by path first, label second.
            'classifier_path':   (self._clf_options.get(self._clf_combo.get())
                                  if getattr(self, '_clf_options', None) else None),
            'include_unlabeled': bool(self._include_unlabeled_var.get()),
            'n_unlabeled':       int(self._n_unlabeled_var.get()),
            'n_folds':           int(self._n_folds_var.get()),
            'trim':              bool(self._trim_to_positive_var.get()),
            'auto_holdout':      bool(self._auto_holdout_var.get()),
            'auto_holdout_pct':  int(self._auto_holdout_pct_var.get()),
            'min_spacing':       int(self._min_spacing_var.get()),
            # Bout sampling parameters
            'n_suggestions':     int(self._n_suggestions_var.get()),
            'max_bout':          int(self._max_bout_var.get()),
            'context_frames':    int(self._context_frames_var.get()),
            'threshold':         float(self._threshold_var.get()),
        }
        # Last scoring's bout statistics (count + avg length etc.) so the saved session
        # records how many uncertain bouts were found and their typical length.
        _ss = getattr(self, '_last_score_stats', None)
        if _ss:
            data['score_stats'] = {
                'n_bouts':    int(_ss.get('n_bouts', 0)),
                'avg_len':    round(float(_ss.get('avg_len', 0.0)), 1),
                'n_eligible': int(_ss.get('n_eligible', 0)),
                'n_pos':      int(_ss.get('n_pos', 0)),
                'cur_lab':    int(_ss.get('cur_lab', 0)),
            }
        path = self._al_session_path(behavior)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            if not silent:
                _bs = data.get('score_stats')
                _btxt = (f", {_bs['n_bouts']} uncertain bouts (avg {_bs['avg_len']:.0f} frames)"
                         if _bs else "")
                self._log_msg(f"Saved AL session → {os.path.basename(path)} "
                              f"({len(data['selected'])} videos, {len(data['test'])} test, "
                              f"{data['n_suggestions']} bouts/iter{_btxt}).")
        except Exception as e:
            if not silent:
                self._log_msg(f"⚠ could not save session: {e}")

    def _load_al_session(self):
        """Reload a saved AL session: restore the exact selection + test set + clf +
        params, and redraw the persisted learning curve."""
        import json
        from tkinter import filedialog
        folder = self.app.current_project_folder.get() or '.'
        default = self._al_session_path()
        path = default if os.path.isfile(default) else filedialog.askopenfilename(
            title="Load AL session", initialdir=os.path.join(folder, 'features'),
            filetypes=[("AL session", "*_al_session.json"), ("All files", "*.*")])
        if not path or not os.path.isfile(path):
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            self._log_msg(f"⚠ could not load session: {e}")
            return
        # restore params + test set BEFORE scanning (scan tags test sessions)
        self._include_unlabeled_var.set(bool(data.get('include_unlabeled', False)))
        self._n_unlabeled_var.set(int(data.get('n_unlabeled', 0)))
        self._n_folds_var.set(int(data.get('n_folds', 5)))
        self._trim_to_positive_var.set(bool(data.get('trim', False)))
        self._auto_holdout_var.set(bool(data.get('auto_holdout', False)))
        self._auto_holdout_pct_var.set(int(data.get('auto_holdout_pct', 20)))
        # NOTE: bout params (min_spacing/n_suggestions/max_bout/context/threshold) are
        # restored AFTER the scan/clf/auto-detect below, so the saved values win over
        # `_auto_bouts_from_labels` (which would otherwise re-derive them from labels).
        self._test_session_names = set(data.get('test', []))
        self._scan_sessions()
        # restore classifier — prefer matching by saved path (label format can change),
        # fall back to the saved display label for back-compat with older sessions.
        clf = data.get('classifier')
        clf_path = data.get('classifier_path')
        sel_label = None
        if clf_path and getattr(self, '_clf_options', None):
            sel_label = next((lbl for lbl, p in self._clf_options.items()
                              if p == clf_path), None)
        if sel_label is None and clf and clf in list(self._clf_combo['values'] or []):
            sel_label = clf
        if sel_label is not None:
            self._clf_combo.set(sel_label)
            self._on_clf_selected()
        # re-select the saved videos
        want = set(data.get('selected', []))
        self._session_lb.selection_clear(0, 'end')
        n_sel = 0
        for i, s in enumerate(self._sessions_list):
            if s['session_name'] in want and i < self._session_lb.size():
                self._session_lb.selection_set(i)
                n_sel += 1
        # redraw the persisted curve so progress comes back
        self._show_saved_curve(data.get('behavior', 'behavior'))
        # selection + clf restored → auto-populate bout lengths from the labels…
        self._auto_bouts_from_labels()
        # …then RESTORE the saved bout params on top (saved session wins; setting these
        # clears the green "auto" indicators since these are the user's saved values).
        try:
            self._min_spacing_var.set(int(data.get('min_spacing', 5)))
            if 'n_suggestions' in data:
                self._n_suggestions_var.set(int(data['n_suggestions']))
            if 'max_bout' in data:
                self._max_bout_var.set(int(data['max_bout']))
            if 'context_frames' in data:
                self._context_frames_var.set(int(data['context_frames']))
            if 'threshold' in data:
                self._threshold_var.set(float(data['threshold']))
        except Exception:
            pass
        # restore cached bout stats + reflect them in the eligibility display
        _ss0 = data.get('score_stats')
        if _ss0:
            self._last_score_stats = dict(_ss0)
            try:
                self._update_bout_estimate()
            except Exception:
                pass
        _ss = data.get('score_stats')
        _btxt = (f", last score: {_ss['n_bouts']} uncertain bouts (avg {_ss['avg_len']:.0f} frames)"
                 if _ss else "")
        self._log_msg(f"Loaded AL session: {n_sel}/{len(want)} videos re-selected, "
                      f"{len(self._test_session_names)} test, {data.get('n_suggestions', '?')} "
                      f"bouts/iter, clf={clf or 'none'}{_btxt}. Re-Score to continue.")

    def _show_saved_curve(self, behavior):
        """Load + draw the persisted multi-session learning curve without a live session."""
        folder = self.app.current_project_folder.get() or ''
        path = os.path.join(folder, 'features', f'{behavior}_multi_al_curve.json')
        if not (MATPLOTLIB_AVAILABLE and os.path.isfile(path)):
            return
        try:
            tr = LearningCurveTracker()
            tr.load(path)
            if not tr.records:
                return
            self._lc_ax.clear()
            self._draw_curve_into(self._lc_ax, tracker=tr)
            self._lc_canvas.draw()
        except Exception:
            pass

    def _refresh_session_listbox(self):
        """Render self._sessions_list with cache / unlabeled / test tags.
        Returns the count of sessions missing a feature cache."""
        prev = set(self._session_lb.curselection())
        self._session_lb.delete(0, 'end')
        n_missing = 0
        for idx, s in enumerate(self._sessions_list):
            name = s['session_name']
            cache = self._get_features_cache(s)
            label, color = name, None
            if not cache:
                label, color = f"{name}  [no features]", 'red'
                n_missing += 1
            elif name in self._test_session_names:
                label, color = f"{name}  [TEST]", '#d2691e'
            elif not self._session_has_labels(s):
                label, color = f"{name}  [unlabeled]", '#b8860b'
            self._session_lb.insert('end', label)
            if color:
                self._session_lb.itemconfig(idx, foreground=color)
        for i in prev:
            if i < self._session_lb.size():
                self._session_lb.selection_set(i)
        if self._sessions_list and not self._session_lb.curselection():
            self._session_lb.selection_set(0)
        return n_missing

    def _toggle_test_sessions(self):
        """Mark/unmark the selected session(s) as the fixed held-out test set."""
        sel = self._get_selected_sessions()
        if not sel:
            messagebox.showinfo("No selection",
                                "Select session(s) in the list, then click 'Test set'.")
            return
        for s in sel:
            nm = s['session_name']
            if nm in self._test_session_names:
                self._test_session_names.discard(nm)
            elif self._session_has_labels(s):
                self._test_session_names.add(nm)
            else:
                self._log_msg(f"'{nm}' has no labels — cannot be a held-out test set.")
        self._refresh_session_listbox()
        n = len(self._test_session_names)
        self._log_msg(f"Held-out test set: {n} session(s)." if n
                      else "Held-out test set cleared.")

    def _refresh_classifiers(self):
        """Populate the classifier dropdown with decluttered, human-readable labels.

        The Train tab saves each run into its own subfolder (classifiers/
        PixelPaws_<behavior>_<ts>/...pkl, plus nested pruned/ and all_features/
        variants). Listing every pkl with its raw path got convoluted, so we:
          • hide the pruned/ + all_features/ variants unless 'variants' is ticked,
          • label entries 'behavior · F1 0.67 · 06-01 · AL' (one read per pkl),
          • sort grouped by behavior, newest-first.
        ``_clf_options`` stays a {display-label → full path} map, so the loaders
        keyed on it are unaffected.
        """
        import re
        show_variants = bool(getattr(self, '_clf_show_variants_var', None)
                             and self._clf_show_variants_var.get())
        folder = self.app.current_project_folder.get()
        clf_dir = os.path.join(folder, 'classifiers')
        self._clf_options = {}
        entries = []   # (behavior, ts_sortkey, f1_or_-1, display_label, path)
        if os.path.isdir(clf_dir):
            for root, _dirs, files in os.walk(clf_dir):
                for f in files:
                    if not f.endswith('.pkl') or f.endswith('_train_set.pkl'):
                        continue   # _train_set = training-data dump, not a classifier
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, clf_dir)
                    parts = rel.split(os.sep)
                    # Variant = lives in a pruned/ or all_features/ subfolder of its run.
                    variant = next((p for p in parts[:-1]
                                    if p in ('pruned', 'all_features')), None)
                    if variant and not show_variants:
                        continue
                    # Read metadata once (also doubles as the best-F1 scan).
                    try:
                        data = _robust_unpickle(full)
                    except Exception:
                        data = None
                    if not isinstance(data, dict) or 'clf_model' not in data:
                        continue
                    beh = (data.get('behavior')
                           or (re.sub(r'^PixelPaws_|_AL$|\.pkl$', '',
                                      f.replace('.pkl', '')) or 'classifier'))
                    f1 = data.get('honest_cv_f1')
                    if f1 is None:
                        f1 = data.get('oof_best_f1')
                    if f1 is None:
                        f1 = data.get('mean_cv_f1')
                    f1 = float(f1) if f1 is not None else None
                    # Date: prefer the YYYYMMDD_HHMMSS embedded in the run-folder name;
                    # fall back to the file's mtime.
                    m = re.search(r'(\d{8})_(\d{6})', rel)
                    if m:
                        try:
                            dt = datetime.strptime(m.group(1) + m.group(2), '%Y%m%d%H%M%S')
                        except Exception:
                            dt = datetime.fromtimestamp(os.path.getmtime(full))
                    else:
                        dt = datetime.fromtimestamp(os.path.getmtime(full))
                    is_al = f.endswith('_AL.pkl')
                    label = (f"{beh} · F1 {f1:.2f}" if f1 is not None else f"{beh} · F1 n/a") \
                        + f" · {dt.strftime('%m-%d')}"
                    if is_al:
                        label += " · AL"
                    if variant:
                        label += f" · {variant}"
                    entries.append((str(beh), dt, (f1 if f1 is not None else -1.0),
                                    label, full))
        # Group by behavior (ascending), newest-first within each behavior.
        entries.sort(key=lambda e: (e[0].lower(), -e[1].timestamp()))
        for beh, dt, f1, label, path in entries:
            uniq = label
            n = 2
            while uniq in self._clf_options:   # disambiguate identical labels
                uniq = f"{label} ({n})"
                n += 1
            self._clf_options[uniq] = path
        self._clf_combo['values'] = list(self._clf_options.keys())
        if not self._clf_options:
            self._clf_combo.set('')
            return
        # Auto-select the highest-F1 entry (we already parsed F1 above).
        best = max(entries, key=lambda e: e[2]) if entries else None
        best_label = next((lbl for lbl, p in self._clf_options.items()
                           if best and p == best[4]), None)
        if best_label:
            self._clf_combo.set(best_label)
        else:
            self._clf_combo.current(0)

    def _browse_classifier(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select Classifier (.pkl)",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
        if not path:
            return
        name = os.path.basename(path)
        self._clf_options[name] = path
        self._clf_combo['values'] = list(self._clf_options.keys())
        self._clf_combo.set(name)

    def _get_selected_session(self):
        sel = self._session_lb.curselection()
        if not sel:
            return None
        idx = sel[0]
        if idx < len(self._sessions_list):
            return self._sessions_list[idx]
        return None

    def _get_selected_sessions(self) -> list:
        indices = self._session_lb.curselection()
        return [self._sessions_list[i] for i in indices if i < len(self._sessions_list)]

    def _get_features_cache(self, session):
        """Find the feature-cache .pkl for a session.

        Searches the project-level ``features/`` dir plus dirs co-located with
        the video, so sessions discovered in nested sub-folders (recursive
        scan) resolve their caches too. The lookup stem comes from the video
        filename — a nested session_name may carry a ``subfolder/`` prefix that
        is NOT part of the cache filename.

        When several ``*_features_<hash>.pkl`` variants exist, the canonical
        :data:`PREFERRED_FEATURE_HASH` schema is returned outright so every
        session loads the same column set; otherwise the first variant found is
        used (previous behaviour).
        """
        import glob as _glob
        folder = self.app.current_project_folder.get()
        video_path = session.get('video_path', '') or session.get('video', '')
        video_dir = os.path.dirname(video_path)
        # Bare file stem (strip any "subfolder/" prefix that recursive scan
        # bakes into session_name; cache files are named by the video stem).
        if video_path:
            base = os.path.splitext(os.path.basename(video_path))[0]
        else:
            base = os.path.basename(session.get('session_name', ''))

        search_dirs = [
            os.path.join(folder, 'features'),
            video_dir,
            os.path.join(video_dir, 'features'),
            os.path.join(video_dir, 'PredictionCache'),
            os.path.join(video_dir, 'FeatureCache'),
        ]
        fallback = None
        for d in search_dirs:
            if not d or not os.path.isdir(d):
                continue
            canonical = os.path.join(d, f"{base}_features_{PREFERRED_FEATURE_HASH}.pkl")
            if os.path.isfile(canonical):
                return canonical
            if fallback is None:
                matches = sorted(_glob.glob(os.path.join(d, f"{base}_features*.pkl")))
                if matches:
                    fallback = matches[0]
        return fallback

    def _load_selected_classifier(self):
        """Return the selected pre-trained classifier, or None to train from labels."""
        name = self._clf_combo.get()
        if name and name in self._clf_options:
            try:
                data = _robust_unpickle(self._clf_options[name])
                # PixelPaws .pkl files are dicts with 'clf_model' key
                if isinstance(data, dict):
                    return data['clf_model']
                return data
            except Exception as e:
                self._log_msg(f"Warning: could not load classifier '{name}': {e}")
        return None

    def _load_selected_classifier_data(self):
        """Return the full pkl dict for the selected classifier, or None."""
        name = self._clf_combo.get()
        if name and name in self._clf_options:
            try:
                data = _robust_unpickle(self._clf_options[name])
                if isinstance(data, dict) and 'clf_model' in data:
                    return data
            except Exception:
                pass
        return None

    def _on_clf_selected(self):
        """When a warm-start classifier is chosen: auto-populate the bout param
        from its stored values and anchor the curve baseline on its HONEST F1."""
        data = self._load_selected_classifier_data()
        if not data:
            return
        try:
            if data.get('min_bout'):
                self._min_spacing_var.set(int(data['min_bout']))
        except Exception:
            pass
        self._clf_bout_seed = {
            'min_bout':       int(data.get('min_bout', 1) or 1),
            'min_after_bout': int(data.get('min_after_bout', 0) or 0),
            'max_gap':        int(data.get('max_gap', 0) or 0),
        }
        hb = data.get('honest_cv_f1')
        if hb is None:
            hb = data.get('mean_cv_f1')
        self._base_clf_f1 = float(hb) if hb is not None else None
        _bf = f"{self._base_clf_f1:.3f}" if self._base_clf_f1 is not None else "n/a"
        self._log_msg(f"Loaded '{self._clf_combo.get()}': bout seed "
                      f"min_bout={self._clf_bout_seed['min_bout']} "
                      f"min_after={self._clf_bout_seed['min_after_bout']} "
                      f"max_gap={self._clf_bout_seed['max_gap']}  |  honest baseline F1={_bf}")
        try:
            self._refresh_plot()
        except Exception:
            pass
        # Classifier picked → if labeled sessions are loaded, auto-populate bout
        # lengths from the actual labels (overrides the clf's stored min_bout seed).
        self._auto_bouts_from_labels()

    def _clear_al_progress(self):
        """Restart fresh: revert AL-added labels to their pre-AL backup, delete the
        learning curve(s) and the saved AL classifier for this behavior. Warns first."""
        import glob as _glob, shutil as _shutil
        selected = self._get_selected_sessions() or list(self._sessions_list)
        folder = self.app.current_project_folder.get() or ''

        behavior = getattr(self._session, 'behavior_name', None) if self._session else None
        if not behavior:
            behavior = (self._load_selected_classifier_data() or {}).get('Behavior_type')
        behavior = behavior or 'behavior'

        label_csvs = [s.get('labels_path') or s.get('target_path') for s in selected]
        label_csvs = [lc for lc in label_csvs if lc]

        curve_paths = set()
        if folder:
            curve_paths.add(os.path.join(folder, 'features', f'{behavior}_multi_al_curve.json'))
        for lc in label_csvs:
            curve_paths.add(self._curve_path(lc))
        if self._session is not None:
            try:
                curve_paths.add(self._get_curve_path())
            except Exception:
                pass
        al_pkl = os.path.join(folder, 'classifiers', f'PixelPaws_{behavior}_AL.pkl') if folder else None

        revertable = []
        for lc in label_csvs:
            bdir = os.path.join(os.path.dirname(lc), 'label_backups')
            stem = os.path.splitext(os.path.basename(lc))[0]
            cands = sorted(_glob.glob(os.path.join(bdir, f'{stem}_preAL_*.csv')))
            if cands:
                revertable.append((lc, cands[0]))   # earliest pre-AL snapshot

        n_curve = sum(1 for p in curve_paths if p and os.path.isfile(p))
        n_pkl = 1 if (al_pkl and os.path.isfile(al_pkl)) else 0
        msg = (f"Clear ALL active-learning progress for '{behavior}'?\n\n"
               f"This will IRREVERSIBLY:\n"
               f"  • revert {len(revertable)} label CSV(s) to their pre-AL backup "
               f"(discarding labels added via AL)\n"
               f"  • delete {n_curve} learning-curve file(s)\n"
               f"  • delete {n_pkl} saved AL classifier(s) (PixelPaws_{behavior}_AL.pkl)\n\n"
               f"Labels with no pre-AL backup are left untouched. Continue?")
        if not messagebox.askyesno("Clear AL progress", msg, icon='warning'):
            return

        for lc, bak in revertable:
            try:
                _shutil.copy2(bak, lc)
                self._log_msg(f"Reverted {os.path.basename(lc)} ← {os.path.basename(bak)}")
            except Exception as e:
                self._log_msg(f"⚠ could not revert {os.path.basename(lc)}: {e}")
        if not revertable:
            self._log_msg("No pre-AL backups found — labels left as-is.")
        for p in curve_paths:
            try:
                if p and os.path.isfile(p):
                    os.remove(p)
                    self._log_msg(f"Deleted curve {os.path.basename(p)}")
            except Exception as e:
                self._log_msg(f"⚠ could not delete {os.path.basename(p)}: {e}")
        if self._session is not None and hasattr(self._session, 'tracker'):
            try:
                self._session.tracker.records.clear()
            except Exception:
                pass
        if n_pkl:
            try:
                os.remove(al_pkl)
                self._log_msg(f"Deleted {os.path.basename(al_pkl)}")
            except Exception as e:
                self._log_msg(f"⚠ could not delete AL pkl: {e}")

        # In-memory reset + re-anchor baseline on the honest number
        self._session = None
        self._last_probas = None
        self._last_model = None
        self._held_out_sessions = []
        self._refresh_classifiers()
        self._on_clf_selected()   # sets _base_clf_f1 from honest_cv_f1 of selected clf
        try:
            self._draw_empty_curve()
        except Exception:
            pass
        self._log_msg("AL progress cleared — fresh start. Re-Score to begin (honest baseline anchored).")

    # ------------------------------------------------------------------
    # Scoring + histogram
    # ------------------------------------------------------------------

    def _evaluate_held_out(self, model, best_params):
        """Unbiased F1 on the fixed held-out test set (sessions marked 'test',
        never queried). Predicts each test session's FULL video with the same
        threshold + bout post-processing as best_params, then scores against
        ground truth. Returns pooled F1, or None when no test set / on failure.
        """
        test_sessions = getattr(self, '_held_out_sessions', None)
        if not test_sessions:
            return None
        try:
            from prediction_pipeline import augment_features_post_cache as _aug
            from evaluation_tab import _apply_bout_filtering
            from sklearn.metrics import f1_score as _f1s
            clf_data = self._load_selected_classifier_data() or {}
            thr = float(best_params.get('thresh', 0.5))
            mb = int(best_params.get('min_bout', 1))
            mab = int(best_params.get('min_after_bout', 1))
            mg = int(best_params.get('max_gap', 0))
            model_cols = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else None
            y_all, p_all = [], []
            for s in test_sessions:
                fc = self._get_features_cache(s)
                lcsv = s.get('target_path') or s.get('labels')
                if not fc or not (lcsv and os.path.isfile(lcsv)):
                    continue
                feats = _robust_unpickle(fc)
                Xdf = feats if isinstance(feats, pd.DataFrame) else pd.DataFrame(feats)
                dlc = s.get('pose_path') or s.get('dlc') or ''
                try:
                    Xdf = _aug(Xdf, clf_data, model, dlc, log_fn=None)
                except Exception:
                    pass
                if model_cols is not None:
                    Xdf = Xdf.reindex(columns=model_cols, fill_value=0.0)
                proba = model.predict_proba(Xdf)[:, 1]
                gt = pd.read_csv(lcsv).iloc[:, 0].to_numpy().astype(float)
                n = min(len(proba), len(gt))
                proba, gt = proba[:n], gt[:n]
                pred = (proba >= thr).astype(int)
                try:
                    pred = np.asarray(_apply_bout_filtering(pred, mb, mab, mg)).astype(int)
                except Exception:
                    pass
                labeled = ~np.isnan(gt)            # score only reviewed frames
                if labeled.any():
                    y_all.append(gt[labeled].astype(int))
                    p_all.append(pred[labeled])
            if not y_all:
                return None
            return float(_f1s(np.concatenate(y_all), np.concatenate(p_all), zero_division=0))
        except Exception as e:
            self.app.root.after(0, lambda e=e: self._log_msg(f"Held-out eval failed: {e}"))
            return None

    def _augment_session_features(self, sess, model, clf_data, log):
        """Augment a session's cached BASE features up to a warm-start model's
        feature schema, in place.

        The 8aed1c22 cache is the 635-col base set; classifiers trained with the
        GUI's richer pipeline also need egocentric / contact / lag / multiscale
        columns. ``augment_features_post_cache`` derives those (egocentric ones
        re-read the DLC .h5), then we reindex to the model's exact column order
        so ``_align_features`` becomes a no-op at scoring time. Handles single
        sessions (``_features``) and multi-session pools (``_subs``) alike; a
        no-op when the model needs nothing beyond the base cache.
        """
        from prediction_pipeline import augment_features_post_cache as _aug
        # Align to what the model actually predicts on.
        if hasattr(model, 'feature_names_in_'):
            target_cols = list(model.feature_names_in_)
        elif clf_data and clf_data.get('selected_feature_cols'):
            target_cols = list(clf_data['selected_feature_cols'])
        else:
            return  # nothing to align to — leave features as-is

        def _aug_one(feats, cols, dlc_path, tag=""):
            if not cols:
                return feats, cols  # no column names -> can't augment safely
            df = pd.DataFrame(feats, columns=cols)
            try:
                df = _aug(df, clf_data or {}, model, dlc_path or '', log_fn=log)
            except Exception as e:
                log(f"⚠ augmentation failed{tag} ({type(e).__name__}: {e}); using base features.")
            missing = [c for c in target_cols if c not in df.columns]
            if missing:
                log(f"⚠ {len(missing)} model feature(s){tag} still absent after augmentation "
                    f"(e.g. {missing[:4]}) — 0-filled; check DLC path / feature config.")
            df = df.reindex(columns=target_cols, fill_value=0.0)
            return df.values.astype(np.float32), list(target_cols)

        if hasattr(sess, '_subs'):
            _reused = 0
            for i, sub in enumerate(sess._subs):
                # Memoize: a sub already augmented to THIS exact model schema needs no
                # recompute — every AL stage (initial Score, Next-Iteration re-score, …)
                # would otherwise re-derive the identical egocentric/contact/lag/brightness/
                # normalized columns per session. The warm-start clf and the retrained AL
                # model share one schema within a session, so this skips the bulk of it.
                if list(sub.get('feature_cols') or []) == target_cols:
                    _reused += 1
                    continue
                _tag = f" [{os.path.basename(sub.get('video_path', '')) or i}]"
                sub['features'], sub['feature_cols'] = _aug_one(
                    sub['features'], sub.get('feature_cols'), sub.get('dlc_path'), _tag)
            if _reused:
                log(f"  {_reused}/{len(sess._subs)} session(s) already augmented to the "
                    f"model schema — reused (no recompute).")
        elif hasattr(sess, '_features'):
            if list(getattr(sess, '_feature_cols', None) or []) == target_cols:
                return   # already augmented to this schema — reuse
            sess._features, sess._feature_cols = _aug_one(
                sess._features, getattr(sess, '_feature_cols', None),
                getattr(sess, '_dlc_path', ''))

    def _score_and_histogram(self):
        selected = self._get_selected_sessions()
        if not selected:
            messagebox.showwarning("No session", "Please select a session first.")
            return

        # Auto-save this session (selection + test + params) so it can be reloaded
        # for a reproducible train/test split.
        self._save_al_session(silent=True)

        # Held-out test sessions are never queried/trained — pull them out of the
        # pool and remember them for the per-iteration unbiased F1 checkpoint.
        self._held_out_sessions = [s for s in self._sessions_list
                                   if s['session_name'] in self._test_session_names]
        selected = [s for s in selected
                    if s['session_name'] not in self._test_session_names]

        # Cap the UNLABELED pool to N (the "use N" spinbox): keep all labeled, but
        # only N unlabeled (deterministic) even if more were selected.
        _cap = int(self._n_unlabeled_var.get())
        if _cap > 0:
            _lab = [s for s in selected if self._session_has_labels(s)]
            _unl = [s for s in selected if not self._session_has_labels(s)]
            if len(_unl) > _cap:
                _names = sorted(s['session_name'] for s in _unl)
                _keep = set(_names[i] for i in
                            np.random.RandomState(42).permutation(len(_names))[:_cap])
                _unl = [s for s in _unl if s['session_name'] in _keep]
                selected = [s for s in selected
                            if self._session_has_labels(s) or s['session_name'] in _keep]
                self._log_msg(f"Unlabeled pool capped to {_cap} of {len(_names)} selected.")

        # Auto hold-out: if enabled and the user hasn't marked a test set, reserve
        # a deterministic fraction of the LABELED selected sessions as the test set.
        if self._auto_holdout_var.get() and not self._held_out_sessions:
            labeled_pool = [s for s in selected if self._session_has_labels(s)]
            if len(labeled_pool) >= 2:
                pct = max(5, min(50, int(self._auto_holdout_pct_var.get())))
                k = max(1, int(round(len(labeled_pool) * pct / 100.0)))
                k = min(k, len(labeled_pool) - 1)   # always leave ≥1 for training
                names = sorted(s['session_name'] for s in labeled_pool)
                pick = np.random.RandomState(42).permutation(len(names))[:k]
                hold = {names[i] for i in pick}
                self._held_out_sessions = [s for s in labeled_pool
                                           if s['session_name'] in hold]
                selected = [s for s in selected if s['session_name'] not in hold]
                self._log_msg(f"Auto hold-out: reserved {k}/{len(labeled_pool)} "
                              f"labeled session(s) as test — {sorted(hold)}")
            else:
                self._log_msg("Auto hold-out skipped — need ≥2 labeled sessions.")

        if not selected:
            messagebox.showwarning(
                "No training session",
                "All selected sessions are marked as the held-out TEST set.\n"
                "Select at least one non-test session to actively learn on.")
            return

        # Validate all selected sessions. A session with no label CSV is allowed
        # ONLY when a warm-start classifier is selected (0 labels can't train).
        _clf_selected = self._load_selected_classifier() is not None
        _any_unlabeled = False
        for s in selected:
            lcsv = s.get('labels_path') or s.get('target_path')
            fc = self._get_features_cache(s)
            if not (lcsv and os.path.isfile(lcsv)):
                _any_unlabeled = True
            if not fc or not os.path.isfile(fc):
                messagebox.showerror("Missing file",
                                     f"Features cache not found for session '{s['session_name']}'.\n"
                                     "Please run feature extraction first (Train tab).")
                return
        if _any_unlabeled and not _clf_selected:
            messagebox.showerror("Classifier required",
                "One or more selected sessions have no label CSV (unlabeled pool).\n"
                "Select a warm-start classifier in 'Classifier (for scoring)' — "
                "0 labels cannot train a model from scratch.")
            return

        self._log_msg("Initializing session and scoring frames...")
        _selected_pool = list(selected)   # test-set already excluded

        def _run():
            try:
                selected_snap = _selected_pool
                import shutil as _shutil, datetime as _dt
                _ts = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
                _preAL_backups = []
                for _s in selected_snap:
                    _lcsv = _s.get('labels_path') or _s.get('target_path')
                    if _lcsv and os.path.isfile(_lcsv):
                        _bdir = os.path.join(os.path.dirname(_lcsv), 'label_backups')
                        os.makedirs(_bdir, exist_ok=True)
                        _stem = os.path.splitext(os.path.basename(_lcsv))[0]
                        _dst = os.path.join(_bdir, f'{_stem}_preAL_{_ts}.csv')
                        _shutil.copy2(_lcsv, _dst)
                        _preAL_backups.append(_dst)
                if _preAL_backups:
                    _bmsgs = [os.path.basename(p) for p in _preAL_backups]
                    self.app.root.after(0, lambda _bmsgs=_bmsgs: self._log_msg(
                        "Label backup(s): " + ", ".join(_bmsgs)))
                # Behavior name + target label-CSV path for from-scratch (unlabeled)
                # sessions: derived from the warm-start classifier, written to the
                # canonical behavior_labels/ dir so a later require_labels scan finds it.
                _clf_dat = self._load_selected_classifier_data()
                _behavior = ((_clf_dat or {}).get('Behavior_type')
                             or (_clf_dat or {}).get('behavior_name') or 'behavior')

                def _resolve_labels_csv(s):
                    tgt = s.get('labels_path') or s.get('target_path')
                    if tgt and os.path.isfile(tgt):
                        return tgt
                    vp = s.get('video_path') or s.get('video') or ''
                    base = os.path.splitext(os.path.basename(vp))[0]
                    proj = s.get('project_dir') or self.app.current_project_folder.get()
                    return os.path.join(proj, 'behavior_labels', f'{base}_labels.csv')

                if len(selected_snap) == 1:
                    s = selected_snap[0]
                    labels_csv = _resolve_labels_csv(s)
                    video_path = s.get('video_path', '')
                    features_cache = self._get_features_cache(s)
                    sess = ALSessionV2(
                        labels_csv=labels_csv,
                        video_path=video_path,
                        features_cache=features_cache,
                        min_frame_spacing=self._min_spacing_var.get(),
                        dlc_path=s.get('pose_path') or s.get('dlc') or '',
                        behavior_name=_behavior,
                    )
                else:
                    from active_learning_v2 import MultiSessionAL
                    sess = MultiSessionAL([{
                        'labels_csv': _resolve_labels_csv(s),
                        'video_path': s.get('video_path', ''),
                        'features_cache': self._get_features_cache(s),
                        'dlc_path': s.get('pose_path') or s.get('dlc') or '',
                    } for s in selected_snap],
                        min_frame_spacing=self._min_spacing_var.get(),
                        behavior_name=_behavior)
                    labels_csv = _resolve_labels_csv(selected_snap[0])
                # Train-tab parity: trim training set to the positive span (scoring
                # still uses the full video). Applied inside train_model.
                sess._trim_to_positive = bool(self._trim_to_positive_var.get())
                # Shared label-bout-length filter (same vars as the Train tab) — drop
                # too-short/long labeled bouts from AL training/eval too.
                try:
                    sess._label_bout_min = int(self.app.train_min_label_bout.get())
                    sess._label_bout_max = int(self.app.train_max_label_bout.get())
                except Exception:
                    sess._label_bout_min = sess._label_bout_max = 0
                self._session = sess

                # Note sessions whose label CSV has no column for this behavior —
                # they're treated as unlabeled (not as negatives), mirroring the Train
                # tab's "skipped (column not in labels)". Explains a smaller/​correct
                # labeled-frame count vs reading the wrong (first) column.
                _missing = list(getattr(sess, '_missing_behavior', None)
                                or ([] if not getattr(sess, '_behavior_missing', False)
                                    else [os.path.splitext(os.path.basename(labels_csv))[0]]))
                if _missing:
                    _bn = getattr(sess, 'behavior_name', _behavior)
                    _names = ", ".join(_missing[:8]) + (" …" if len(_missing) > 8 else "")
                    self.app.root.after(0, lambda _bn=_bn, _n=len(_missing), _names=_names:
                        self._log_msg(f"{_n} session(s) have no '{_bn}' column — "
                                      f"treated as unlabeled: {_names}"))

                # Check for feature-label truncation warnings
                if hasattr(sess, '_truncation_warning'):
                    warn = getattr(sess, '_truncation_warning', None)
                    if warn and warn[1] > 0:
                        _tw = warn
                        self.app.root.after(0, lambda _tw=_tw: messagebox.showwarning(
                            "Label Truncation",
                            f"Feature cache is shorter than labels CSV by {_tw[0]} rows.\n"
                            f"{_tw[1]} labeled frames beyond the feature range will be ignored.\n\n"
                            "Re-extract features to include all frames."))
                elif hasattr(sess, '_subs'):
                    for _sub in sess._subs:
                        _tw = _sub.get('_truncation_warning')
                        if _tw and _tw[1] > 0:
                            _sname = os.path.basename(_sub.get('video_path', ''))
                            self.app.root.after(0, lambda _tw=_tw, _sn=_sname: messagebox.showwarning(
                                "Label Truncation",
                                f"Session '{_sn}': feature cache is shorter than labels CSV by {_tw[0]} rows.\n"
                                f"{_tw[1]} labeled frames beyond the feature range will be ignored.\n\n"
                                "Re-extract features to include all frames."))

                # Capture how many frames were already labeled at load time
                # so the budget only counts newly annotated frames this session
                if hasattr(sess, '_labels'):
                    self._n_labeled_at_load = int(np.sum(sess._labels >= 0))
                elif hasattr(sess, '_subs'):
                    self._n_labeled_at_load = sum(int(np.sum(sub['labels'] >= 0))
                                                   for sub in sess._subs)
                else:
                    self._n_labeled_at_load = 0

                # Diagnostic: show label breakdown after loading
                if hasattr(sess, '_labels'):
                    import numpy as _np2
                    _lbl = sess._labels
                    _n_pos = int(_np2.sum(_lbl == 1))
                    _n_neg = int(_np2.sum(_lbl == 0))
                    _n_unl = int(_np2.sum(_lbl < 0))
                    _n_feat = len(sess._features)
                    _msg = (f"Labels loaded: {_n_pos} positive, {_n_neg} negative, "
                            f"{_n_unl} unlabeled (total {len(_lbl)})")
                    if _n_feat < len(_lbl):
                        _msg += (f" — features cover only {_n_feat} frames; "
                                 f"{len(_lbl) - _n_feat} unlabeled tail frames "
                                 f"scored at max uncertainty")
                    self.app.root.after(0, lambda: self._log_msg(_msg))

                # Load curve from previous session if exists
                curve_path = self._get_curve_path()
                if os.path.isfile(curve_path):
                    try:
                        sess.tracker.load(curve_path)
                        self.app.root.after(0, lambda: self._log_msg(
                            f"Loaded {len(sess.tracker.records)} previous iteration(s) from curve."))
                        self.app.root.after(0, self._refresh_plot)
                    except Exception:
                        pass

                clf_override = self._load_selected_classifier()
                if clf_override is not None:
                    model = clf_override
                    self.app.root.after(0, lambda: self._log_msg(
                        f"Using pre-trained classifier: {self._clf_combo.get()}"))
                    clf_data_full = self._load_selected_classifier_data()
                    base_f1 = None
                    if clf_data_full:
                        base_f1 = clf_data_full.get('honest_cv_f1')   # prefer honest
                        if base_f1 is None:
                            base_f1 = clf_data_full.get('mean_cv_f1')
                    self.app.root.after(0, lambda v=base_f1: setattr(self, '_base_clf_f1', v))
                    # Augment cached BASE features up to the warm-start model's
                    # schema (egocentric / contact / lag / multiscale), per-session
                    # using each session's DLC pose file. Handles BOTH single- and
                    # multi-session; no-op when the model needs nothing extra.
                    def _log_sync(m):
                        self.app.root.after(0, lambda m=m: self._log_msg(m))
                    self._augment_session_features(sess, model, clf_data_full, _log_sync)
                    # Make the learning curve start from the warm-start classifier's
                    # HONEST F1 (and purge any stale leaky iteration-0 point) so it no
                    # longer "drops" from an inflated value. Runs on the Tk thread AFTER
                    # the _base_clf_f1 setattr queued above.
                    self.app.root.after(0, self._seed_baseline_curve)
                else:
                    model = sess.train_model()
                    self._base_clf_f1 = None
                    _dropped = int(getattr(sess, '_trim_dropped', 0) or 0)
                    if _dropped > 0:
                        self.app.root.after(0, lambda d=_dropped: self._log_msg(
                            f"Trim-to-positive-span dropped {d:,} labeled frame(s) outside the "
                            f"positive span from the training set (scoring still uses the full video)."))
                if hasattr(sess, 'get_full_probas'):
                    probas = sess.get_full_probas(model)
                else:
                    # MultiSessionAL: concatenate probas from all sub-sessions
                    import numpy as _np
                    from active_learning_v2 import _align_features as _al_align
                    probas = _np.concatenate([
                        model.predict_proba(
                            _al_align(model, sub['features'], sub.get('feature_cols')))[:, 1]
                        for sub in sess._subs])
                self._last_probas = probas
                self._last_model = model
                self.app.root.after(0, lambda: self._btn_next_iter.configure(state='normal'))

                threshold = self._threshold_var.get()
                n_frames, n_bouts_eligible, bout_stats = sess.count_eligible(
                    probas, threshold, self._min_spacing_var.get())
                n_eligible = n_frames
                n_pos = sess.count_positive()
                if hasattr(sess, '_labels'):
                    _cur_lab = int((sess._labels >= 0).sum())
                else:
                    _cur_lab = sum(int((s['labels'] >= 0).sum()) for s in sess._subs)
                # Cache eligibility stats so the "Bouts / iteration" spinbox can
                # recompute the +frames/% estimate live without re-scoring.
                self._last_score_stats = {
                    'n_pos': int(n_pos), 'n_eligible': int(n_eligible),
                    'n_bouts': int(n_bouts_eligible),
                    'avg_len': float(n_frames / max(n_bouts_eligible, 1)),
                    'cur_lab': int(_cur_lab),
                }
                self.app.root.after(0, self._update_bout_estimate)
                self.app.root.after(0, lambda: self._log_msg(
                    f"Scored {len(probas):,} frames. "
                    f"{n_pos:,} pos labeled. "
                    f"{n_eligible:,} eligible unlabeled frames in {n_bouts_eligible} bouts "
                    f"at threshold={threshold:.2f}"))
                # --- Convergence hint (A-SOID-style) ---
                _n_sugg = self._n_suggestions_var.get()
                if 0 < n_bouts_eligible < _n_sugg:
                    self.app.root.after(0, lambda nb=n_bouts_eligible, ns=_n_sugg: self._log_msg(
                        f"⚑ Only {nb} uncertain bout(s) remain (< {ns} requested) — "
                        f"model may be converging. Consider stopping or lowering the "
                        f"confidence threshold to find more candidates."))
                elif n_eligible > 0 and (n_eligible / max(len(probas), 1)) < 0.02:
                    self.app.root.after(0, lambda ne=n_eligible, nt=len(probas): self._log_msg(
                        f"⚑ Only {ne:,} / {nt:,} frames ({ne/nt*100:.1f}%) remain uncertain — "
                        f"model is converging."))
                if n_bouts_eligible == 0:
                    _n_runs = bout_stats.get('n_runs', 0)
                    _n_short = bout_stats.get('n_too_short', 0)
                    _min_bf = self._min_spacing_var.get()
                    if _n_runs == 0:
                        self.app.root.after(0, lambda: self._log_msg(
                            "  → No unlabeled frame runs found. Session may be fully labeled."))
                    else:
                        self.app.root.after(0, lambda _n_runs=_n_runs, _n_short=_n_short, _min_bf=_min_bf: self._log_msg(
                            f"  → {_n_runs} unlabeled run(s) found but all filtered: "
                            f"{_n_short} too short (<{_min_bf} frames). "
                            f"Try lowering 'Min bout frames'."))
                self.app.root.after(0, self._show_histogram)
            except Exception as e:
                _e = e
                self.app.root.after(0, lambda _e=_e: messagebox.showerror("Error", str(_e)))
                self.app.root.after(0, lambda _e=_e: self._log_msg(f"Error: {_e}"))

        threading.Thread(target=_run, daemon=True).start()

    def _show_histogram(self):
        if self._last_probas is None:
            return
        ConfidenceHistogramDialog(
            parent_root=self.app.root,
            probas=self._last_probas,
            threshold_var=self._threshold_var,
            on_proceed=None,
            on_cancel=None,
        )

    # ------------------------------------------------------------------
    # Labeling
    # ------------------------------------------------------------------

    def _start_labeling(self):
        if self._session is None or self._last_probas is None:
            messagebox.showwarning("Not scored", "Run 'Score + Histogram' first.")
            return

        selected = self._get_selected_sessions()
        if not selected:
            return

        # For single session validate video path upfront
        if len(selected) == 1:
            video_path = selected[0].get('video_path', '')
            if not os.path.isfile(video_path):
                messagebox.showerror("Missing file", f"Video not found:\n{video_path}")
                return
        else:
            video_path = selected[0].get('video_path', '')

        self._log_msg(f"▶ Launching labeling — scoring {len(selected)} session(s) with the "
                      f"latest model and finding uncertain bouts to label…")

        threshold = self._threshold_var.get()
        n_bouts = self._n_suggestions_var.get()
        min_bout_frames = self._min_spacing_var.get()
        context_frames = self._context_frames_var.get()
        class_balanced = self._class_balanced_var.get()
        diversity_radius = self._diversity_radius_var.get()

        def _run():
            try:
                clf_override = self._load_selected_classifier()
                # Re-augment the session features up to the warm-start classifier's
                # schema BEFORE scoring with it — exactly as the initial score-and-load
                # path does (~4286). A retrain (or session reload) can reset the subs
                # back to the base/AL schema, dropping the clf's egocentric / contact /
                # multiscale columns; without this, _align_features sees too many missing
                # features and refuses to predict (the "Next Iteration" crash). Idempotent:
                # a no-op when the features already match the model's schema.
                if clf_override is not None:
                    clf_data_full = self._load_selected_classifier_data()
                    def _log_sync(m):
                        self.app.root.after(0, lambda m=m: self._log_msg(m))
                    self._augment_session_features(
                        self._session, clf_override, clf_data_full, _log_sync)
                result = self._session.run_one_iteration(
                    n_bouts=n_bouts,
                    confidence_threshold=threshold,
                    min_bout_frames=min_bout_frames,
                    context_frames=context_frames,
                    max_bout_frames=self._max_bout_var.get() or None,  # None → adaptive
                    class_balanced=class_balanced,
                    diversity_radius=diversity_radius,
                    model=clf_override,   # None = retrain from labels
                )
                bouts = result['bouts']
                self._last_probas = result['probas']
                self.app.root.after(0, lambda: self._btn_next_iter.configure(state='normal'))

                if len(bouts) == 0:
                    self.app.root.after(0, lambda: messagebox.showinfo(
                        "Converged", "No uncertain bouts remain. Model has converged!"))
                    return

                self.app.root.after(0, lambda: self._run_bout_labeling_ui(
                    bouts, result['probas'], video_path))
            except Exception as e:
                import traceback
                traceback.print_exc()
                _e = e
                self.app.root.after(0, lambda _e=_e: messagebox.showerror("Error", str(_e)))

        threading.Thread(target=_run, daemon=True).start()

    def _retrain_and_compare(self, _auto_continue=False):
        if self._session is None:
            messagebox.showwarning("No session", "Run '1. Score + Histogram' first.")
            return

        # Pre-flight: need positive + negative labels
        if hasattr(self._session, '_labels'):
            lbl = self._session._labels
            n_pos = int((lbl == 1).sum()); n_neg = int((lbl == 0).sum())
        else:
            n_pos = sum(int((s['labels'] == 1).sum()) for s in self._session._subs)
            n_neg = sum(int((s['labels'] == 0).sum()) for s in self._session._subs)
        if n_pos == 0:
            messagebox.showwarning("No positive labels",
                                   "Label at least one YES bout before retraining.")
            return
        if n_neg == 0:
            messagebox.showwarning("No negative labels",
                                   "Label at least one NO bout before retraining.")
            return

        # Read UI vars on main thread
        threshold_var_val = self._threshold_var.get()
        pf = self.app.current_project_folder.get()
        snap_dir = os.path.join(pf, 'classifiers') if pf else None
        if not snap_dir:
            self._log_msg("⚠ No project folder — classifier will not be saved.")
        behavior_name = getattr(self._session, 'behavior_name', 'behavior')
        base_clf_data = self._load_selected_classifier_data()  # full dict or None
        _al_min_bout = int(base_clf_data.get('min_bout', 1)) if base_clf_data else 1

        # Open / reuse the retrain progress popup (mirrors the Train tab). A subsequent
        # 'Next Iteration → retrain' reuses the SAME window (reset to busy) rather than
        # stacking a new one; only opens a fresh window if none exists or it was closed.
        if getattr(self, '_retrain_window', None) is None or getattr(self._retrain_window, '_closed', True):
            try:
                self._retrain_window = ALRetrainWindow(self.app.root, behavior_name,
                                                       on_next=self._start_labeling)
                self._retrain_window.start_busy()
            except Exception:
                self._retrain_window = None
        else:
            self._retrain_window.reset_for_new_run()

        self._log_msg("Retraining (full pipeline)…")

        def _stage(msg):
            self.app.root.after(0, lambda m=msg: self._log_msg(m))
            rw = getattr(self, '_retrain_window', None)
            if rw is not None:
                self.app.root.after(0, lambda m=msg: rw.set_stage(m))

        def _run():
            try:
                from xgboost import XGBClassifier
                from sklearn.model_selection import StratifiedKFold
                from sklearn.metrics import f1_score as _f1

                # ── Backup label CSVs before this retrain iteration ──────────────
                import shutil as _shutil, datetime as _dt
                _ts = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
                _selected_snap = self._get_selected_sessions() if hasattr(self, '_get_selected_sessions') else []
                _backup_paths = []
                for _s in _selected_snap:
                    _lcsv = _s.get('labels_path') or _s.get('target_path')
                    if _lcsv and os.path.isfile(_lcsv):
                        _bdir = os.path.join(os.path.dirname(_lcsv), 'label_backups')
                        os.makedirs(_bdir, exist_ok=True)
                        _stem = os.path.splitext(os.path.basename(_lcsv))[0]
                        _bdst = os.path.join(_bdir, f'{_stem}_backup_{_ts}.csv')
                        _shutil.copy2(_lcsv, _bdst)
                        _backup_paths.append(_bdst)

                # --- Gather labeled data (trim-to-positive-span parity, training only) ---
                _trim_on = bool(getattr(self._session, '_trim_to_positive', True))
                if hasattr(self._session, '_labels'):
                    _lab = _effective_train_labels(self._session._labels, self._session)
                    mask = _lab >= 0
                    if _trim_on:
                        mask = mask & _trim_to_positive_span(_lab)
                    X = self._session._features[mask]
                    y = _lab[mask]
                    _lab_indices = np.where(mask)[0]
                    _sess_ids = None
                else:
                    import pandas as _pd_fin
                    _sub_dfs_lbl = []
                    ys = []
                    _lab_idx_parts = []
                    _sess_id_parts = []
                    _offset = 0
                    for _si, sub in enumerate(self._session._subs):
                        _lab = _effective_train_labels(sub['labels'], self._session)
                        m = _lab >= 0
                        if _trim_on:
                            m = m & _trim_to_positive_span(_lab)
                        if m.any():
                            _sub_dfs_lbl.append(_pd_fin.DataFrame(
                                sub['features'][m],
                                columns=sub.get('feature_cols') or range(sub['features'].shape[1])))
                            ys.append(_lab[m])
                            # index/session bookkeeping must align with the (trimmed)
                            # labeled rows that go into y — use the SAME mask m.
                            _idx = np.where(m)[0] + _offset
                            _lab_idx_parts.append(_idx)
                            _sess_id_parts.append(np.full(len(_idx), _si, dtype=int))
                        _offset += len(sub['labels'])
                    X_df = _pd_fin.concat(_sub_dfs_lbl, ignore_index=True).fillna(0.0)
                    feature_cols = list(X_df.columns)
                    X = X_df.values.astype(np.float32)
                    y = np.concatenate(ys)
                    _lab_indices = np.concatenate(_lab_idx_parts) if _lab_idx_parts else np.array([], dtype=int)
                    _sess_ids = np.concatenate(_sess_id_parts) if _sess_id_parts else np.array([], dtype=int)

                n_labeled = len(y)
                if not hasattr(self._session, '_labels'):
                    pass  # feature_cols already set above
                else:
                    feature_cols = getattr(self._session, '_feature_cols', None)

                # --- Class imbalance weight (mirrors full training pipeline) ---
                n_pos = int((y == 1).sum())
                n_neg = int((y == 0).sum())
                spw = float(n_neg / n_pos) if n_pos > 0 else 1.0

                # --- CV-OOF: session-level → bout → frame (Train-tab parity) ---
                # Session-level grouping when ≥ n_folds labeled sessions exist
                # (matches the Train tab); else bout-level; else frame-level.
                from active_learning_v2 import (_cv_oof, _make_bout_groups,
                                                _honest_pipeline_oof_f1)
                _n_folds = int(self._n_folds_var.get())
                _bgroups = _make_bout_groups(_lab_indices, _al_min_bout, session_ids=_sess_ids)

                # Map session-group id (sub index) → short session name, built BEFORE CV so
                # the live per-fold callback can name each fold's held-out animals.
                _grp_names = {}
                try:
                    for _gi, _sub in enumerate(getattr(self._session, '_subs', []) or []):
                        _nm = os.path.splitext(os.path.basename(_sub.get('video_path', '') or ''))[0]
                        _grp_names[_gi] = _nm or f"grp{_gi}"
                except Exception:
                    pass

                def _held_str(detail, mode):
                    """Readable held-out-sessions label for a fold (test set)."""
                    tg = detail.get('test_groups') or []
                    if mode == 'session' and tg:
                        names = [_grp_names.get(g, f"grp{g}") for g in tg]
                        if len(names) > 4:
                            return f"{len(names)} sessions ({', '.join(names[:3])}, …)"
                        return ", ".join(names)
                    if tg:
                        return f"{len(tg)} {mode}(s)"
                    return mode

                # Live per-fold streaming: as each CV fold finishes, log its P/R/F1 + the
                # held-out sessions AND grow the bar chart (replaces the "(awaiting CV…)"
                # placeholder fold-by-fold instead of all-at-once after the last fold).
                _live_fold_det = []
                def _on_fold_done(done, total, detail, mode):
                    held = _held_str(detail, mode)
                    _train_note = ""
                    if mode == 'session':
                        n_train = max(len(_grp_names) - len(detail.get('test_groups') or []), 0)
                        _train_note = f" | train: {n_train} others"
                    _live_fold_det.append(dict(detail, held=held))
                    def _ui():
                        self._log_msg(
                            f"  Fold {done}/{total} [test: {held}{_train_note}]  "
                            f"F1={detail['f1']:.3f}  P={detail['precision']:.3f}  "
                            f"R={detail['recall']:.3f} @0.5  (n={detail['n_test']:,}, pos={detail['n_pos']:,})")
                        rw2 = getattr(self, '_retrain_window', None)
                        if rw2 is not None and not getattr(rw2, '_closed', True):
                            rw2.draw_folds(list(_live_fold_det), mode)
                    self.app.root.after(0, _ui)

                # CV eligibility: sparse sessions train-only (shared Train-tab settings).
                _al_elig = None
                try:
                    _mode_e = self.app.train_cv_eligibility_mode.get()
                    if _mode_e != 'off' and _sess_ids is not None:
                        _floor = max(int(self.app.train_min_label_bout.get() or 0), 3)
                        _cnts = {int(si): session_positive_counts(
                                    y[np.asarray(_sess_ids) == si], min_bout_len=_floor)
                                 for si in np.unique(_sess_ids)}
                        _e, _t, _i = select_cv_eligible(
                            _cnts, mode=_mode_e,
                            min_frames=int(self.app.train_min_cv_pos_frames.get() or 0),
                            min_bouts=int(self.app.train_min_cv_pos_bouts.get() or 0))
                        if _t:
                            _al_elig = _e
                            self._log_msg(f"  CV eligibility ({_mode_e}): {len(_e)} eligible, "
                                          f"{len(_t)} sparse session(s) train-only (not evaluated)")
                except Exception:
                    _al_elig = None

                _stage(f"Cross-validating ({_n_folds} folds × {n_labeled:,} labeled frames) — "
                       f"all CPU cores, may take a few minutes…")
                oof_proba, fold_f1s, _cv_mode, _fold_of, _fold_det = _cv_oof(
                    X, y, n_folds=_n_folds,
                    session_ids=_sess_ids,
                    bout_groups=_bgroups,
                    bout_aware=self._bout_aware_cv_var.get(),
                    n_estimators=200,
                    progress_cb=_on_fold_done,
                    eligible_sessions=_al_elig)

                mean_cv_f1 = float(np.mean(fold_f1s))   # per-fold @0.5 (conservative floor)
                std_cv_f1  = float(np.std(fold_f1s))

                # stash for the retrain window's final per-fold bar redraw (_post_log)
                self._last_fold_detail = [dict(_d, held=_held_str(_d, _cv_mode)) for _d in _fold_det]
                self._last_cv_mode = _cv_mode

                # --- Honest, leak-free F1 of the FULL deployed pipeline (nested LOFO) ---
                # For each fold, threshold + bout params are chosen on the OTHER folds and
                # applied (per-session) to the held-out fold — no params-on-test leakage,
                # and (unlike the old threshold-only honest F1) this INCLUDES the bout
                # post-processing the deployed model actually uses. ~5 parallel sweeps.
                _stage("Honest per-fold operating-point selection (nested LOFO)…")
                def _honest_prog(done, total):
                    rw3 = getattr(self, '_retrain_window', None)
                    if rw3 is not None:
                        self.app.root.after(0, lambda d=done, t=total: rw3.set_progress(d, t))
                    self.app.root.after(0, lambda d=done, t=total: self._log_msg(
                        f"  nested fold {d}/{t}: operating point selected on other folds"))
                honest_f1, honest_per_fold, honest_mean, honest_std, honest_bout = \
                    _honest_pipeline_oof_f1(oof_proba, y, _fold_of, session_ids=_sess_ids,
                                            progress_cb=_honest_prog)

                # Threshold-free Average Precision (AUPRC, A-SOiD-style MAP) on the OOF.
                _oof_ap = None
                try:
                    from sklearn.metrics import average_precision_score as _aps
                    if 0 < int(np.sum(y)) < len(y):
                        _oof_ap = float(_aps(y, oof_proba))
                except Exception:
                    _oof_ap = None

                # Smoothing-method comparison (DIAGNOSTIC): HMM/Viterbi honest F1 vs the
                # deployed morphological honest F1. Reported so we can see whether HMM
                # would help; deployment stays morphological for now.
                _stage("Comparing smoothing (HMM/Viterbi vs morphological)…")
                _hmm_f1 = _honest_hmm_oof_f1(oof_proba, y, _fold_of, session_ids=_sess_ids)

                # Per-fold bars now reflect each fold's SWEPT operating point (matches the
                # honest headline), replacing the live @0.5 preview. honest_per_fold is in
                # np.unique(fold_of) order == _last_fold_detail order, so merge by position
                # (keeps each fold's held-out session label).
                try:
                    _lfd = getattr(self, '_last_fold_detail', None) or []
                    for i, sp in enumerate(honest_per_fold):
                        if i < len(_lfd):
                            _lfd[i]['f1'] = sp['f1']
                            _lfd[i]['precision'] = sp['precision']
                            _lfd[i]['recall'] = sp['recall']
                    self._last_fold_detail = _lfd
                    rw_f = getattr(self, '_retrain_window', None)
                    if rw_f is not None and not getattr(rw_f, '_closed', True):
                        self.app.root.after(0, lambda fd=[dict(d) for d in _lfd], m=_cv_mode:
                                            rw_f.draw_folds(fd, m, at_operating_point=True))
                except Exception:
                    pass

                # --- OOF parameter sweep (SHIPPED operating point, on ALL data) ---
                # Chooses the classifier's deployed best_thresh/bout. Session-aware
                # (bouts filtered within each session, no cross-boundary bridging). Its
                # F1 is OPTIMISTIC (params tuned on the same OOF they're scored on) — the
                # honest headline above is the nested-LOFO number; this is cross-ref only.
                _stage("Tuning threshold + bout params (8,288 combos, multi-core)…")
                # Parallel replacement for the Train tab's serial _sweep_postprocessing —
                # fans the 37 thresholds across all cores and uses an inline numpy F1, so
                # this stage pegs the CPU instead of idling on one core.
                def _sweep_prog(done, total):
                    rw2 = getattr(self, '_retrain_window', None)
                    if rw2 is not None:
                        self.app.root.after(0, lambda d=done, t=total: rw2.set_progress(d, t))
                try:
                    best_params = _sweep_postprocessing_fast(
                        oof_proba, y, progress_cb=_sweep_prog, session_ids=_sess_ids)
                except Exception as _swe:
                    self.app.root.after(0, lambda e=_swe: self._log_msg(
                        f"  (parallel sweep failed: {e}; using serial Train-tab sweep)"))
                    if hasattr(self.app, '_sweep_postprocessing'):
                        best_params = self.app._sweep_postprocessing(oof_proba, y)
                    else:
                        best_params = _fallback_sweep(oof_proba, y)

                # --- Final model on all labeled data ---
                # Trains on EVERY labeled frame → this is the classifier that gets saved
                # and used for scoring/prediction. The per-fold CV models above were
                # throwaway (only to estimate the honest F1); this one ships.
                _stage("Training final (deployed) model on all labeled data…")
                self.app.root.after(0, lambda n=n_labeled: self._log_msg(
                    f"  Training final model on all {n:,} labeled frames → this is the "
                    f"classifier saved & used for scoring (the per-fold CV models above "
                    f"were only to estimate the honest F1)."))
                final_clf = XGBClassifier(n_estimators=300, max_depth=6,
                                           learning_rate=0.1, scale_pos_weight=spw,
                                           random_state=42, verbosity=0)
                if feature_cols:
                    import pandas as _pd_fin
                    final_clf.fit(_pd_fin.DataFrame(X, columns=feature_cols), y)
                else:
                    final_clf.fit(X, y)
                from active_learning_v2 import _align_features as _al_align
                if hasattr(self._session, '_features'):
                    probas_all = final_clf.predict_proba(
                        _al_align(final_clf, self._session._features,
                                  getattr(self._session, '_feature_cols', None)))[:, 1]
                else:
                    import pandas as _pd_fin
                    _all_dfs = []
                    for _s in self._session._subs:
                        _all_dfs.append(_pd_fin.DataFrame(
                            _s['features'],
                            columns=_s.get('feature_cols') or range(_s['features'].shape[1])))
                    _X_all = _pd_fin.concat(_all_dfs, ignore_index=True).fillna(0.0)
                    if hasattr(final_clf, 'feature_names_in_'):
                        _X_all = _X_all.reindex(columns=final_clf.feature_names_in_, fill_value=0.0)
                    probas_all = final_clf.predict_proba(_X_all)[:, 1]

                # --- Build full classifier_data ---
                def _get(key, default=None):
                    return base_clf_data.get(key, default) if base_clf_data else default

                clf_data = {
                    'clf_model':             final_clf,
                    'Behavior_type':         behavior_name,
                    'selected_feature_cols': feature_cols,
                    'best_thresh':           best_params['thresh'],
                    'min_bout':              best_params['min_bout'],
                    'min_after_bout':        best_params['min_after_bout'],
                    'max_gap':               best_params['max_gap'],
                    'ui_min_bout':           best_params['min_bout'],
                    'ui_min_after_bout':     _get('ui_min_after_bout', 1),
                    'ui_max_gap':            best_params['max_gap'],
                    'bp_include_list':       _get('bp_include_list'),
                    'bp_pixbrt_list':        _get('bp_pixbrt_list', []),
                    'square_size':           _get('square_size', [40]),
                    'pix_threshold':         _get('pix_threshold', 0.3),
                    'include_optical_flow':  _get('include_optical_flow', True),
                    'bp_optflow_list':       _get('bp_optflow_list', []),
                    # Provenance
                    'training_source':       'active_learning',
                    'n_labeled_total':       n_labeled,
                    'n_positive':            int((y == 1).sum()),
                    'cv_f1_scores':          fold_f1s,
                    'mean_cv_f1':            mean_cv_f1,
                    'std_cv_f1':             std_cv_f1,
                    'oof_best_f1':           best_params['f1'],   # OPTIMISTIC (threshold tuned on same OOF)
                    # Honest, leak-free headline (leave-one-fold-out threshold):
                    'honest_cv_f1':          honest_f1,
                    'honest_cv_mean':        honest_mean,
                    'honest_cv_std':         honest_std,
                    'honest_cv_per_fold':    [d['f1'] for d in honest_per_fold],
                    'honest_bout_f1':        (honest_bout['f1'] if honest_bout else None),
                    'honest_bout_tol':       (honest_bout['tol'] if honest_bout else None),
                    'oof_ap':                _oof_ap,   # threshold-free AUPRC
                    'honest_hmm_f1':         _hmm_f1,   # HMM/Viterbi comparison (diagnostic)
                    'smoothing_method':      'morphological',
                }

                # --- Save ---
                saved_path = None
                if snap_dir:
                    os.makedirs(snap_dir, exist_ok=True)
                    fname = f"PixelPaws_{behavior_name}_AL.pkl"
                    saved_path = os.path.join(snap_dir, fname)
                    atomic_pickle_save(clf_data, saved_path)

                # --- Held-out test set checkpoint (unbiased, on reserved full videos) ---
                if getattr(self, '_held_out_sessions', None):
                    _stage("Evaluating held-out test set…")
                _held_out_f1 = self._evaluate_held_out(final_clf, best_params)

                # --- Learning curve record (for plot) ---
                # Plot the HONEST leave-one-fold-out F1 as the headline (leak-free);
                # the optimistic oof_best_f1 stays in clf_data for Train-tab cross-ref.
                n_below = int(np.sum(np.abs(probas_all - 0.5) * 2 < threshold_var_val))
                record = self._session.tracker.record(
                    final_clf, X, y, n_below,
                    labels_array=getattr(self._session, '_labels', None),
                    min_bout=_al_min_bout,
                    session_ids=_sess_ids, n_folds=int(self._n_folds_var.get()),
                    oof_f1=honest_f1, held_out_f1=_held_out_f1,
                    cv_mode=(_cv_mode or 'session'))
                self._session.tracker.save(self._get_curve_path())
                self._last_probas = probas_all
                self._last_model  = final_clf

                # --- Log ---
                def _post_log():
                    self._log_msg("=" * 52)
                    self._log_msg(f"  RETRAIN COMPLETE — iteration {self._session._iteration}")
                    self._log_msg("=" * 52)
                    self._log_msg(f"  Labeled frames : {n_labeled}  "
                                  f"(+{int((y==1).sum())}  /  -{int((y==0).sum())})")
                    self._log_msg(f"  Class balance  : 1:{spw:.1f}  (neg/pos weight)")
                    self._log_msg(f"  CV mode        : {_cv_mode}")
                    self._log_msg(f"  HONEST CV F1   : {honest_f1:.3f}  "
                                  f"(per-fold {honest_mean:.3f} ± {honest_std:.3f}) "
                                  f"← leak-free headline (nested LOFO: threshold+bouts "
                                  f"picked per-fold, session-aware)")
                    # Change in honest F1 vs the previous curve point (the seeded
                    # warm-start baseline on the first retrain, else the last retrain).
                    try:
                        _recs = [r for r in self._session.tracker.records
                                 if r.oof_f1 is not None]
                        if len(_recs) >= 2:
                            _prev, _cur = _recs[-2].oof_f1, _recs[-1].oof_f1
                            _dlt = _cur - _prev
                            self._log_msg(
                                f"  ΔF1 vs previous: {_dlt:+.3f}  "
                                f"({_prev:.3f} → {_cur:.3f}, iter {_recs[-2].iteration}→{_recs[-1].iteration})")
                        else:
                            self._log_msg("  ΔF1 vs previous: — (first honest point)")
                    except Exception:
                        pass
                    self._log_msg(f"  CV F1 @ 0.5    : {mean_cv_f1:.3f} ± {std_cv_f1:.3f}  "
                                  f"[{', '.join(f'{v:.3f}' for v in fold_f1s)}]  (no-postproc floor)")
                    self._log_msg(f"  OOF F1 (tuned) : {best_params['f1']:.3f}  "
                                  f"(optimistic — threshold+bouts tuned on the same OOF)")
                    if _oof_ap is not None:
                        self._log_msg(f"  OOF Avg Prec   : {_oof_ap:.3f}  "
                                      f"(AUPRC — threshold-free)")
                    if _hmm_f1 is not None:
                        _win = "HMM" if _hmm_f1 > honest_f1 + 0.005 else "morphological"
                        self._log_msg(f"  Smoothing      : morphological {honest_f1:.3f} vs "
                                      f"HMM/Viterbi {_hmm_f1:.3f}  → {_win} wins "
                                      f"(deployed: morphological)")
                    else:
                        self._log_msg("  Smoothing      : HMM comparison skipped "
                                      "(label set too large)")
                    if _held_out_f1 is not None:
                        self._log_msg(f"  Held-out F1    : {_held_out_f1:.3f}  "
                                      f"(unbiased, reserved test sessions)")
                    self._log_msg(f"  ship thresh    : {best_params['thresh']:.2f}  "
                                  f"min_bout={best_params['min_bout']}  "
                                  f"max_gap={best_params['max_gap']}")
                    if saved_path:
                        self._log_msg(f"  Saved → {os.path.basename(saved_path)}")
                    else:
                        self._log_msg("  ⚠ No project folder — classifier not saved.")
                    if _backup_paths:
                        self._log_msg("  Label backups → " +
                                      ", ".join(os.path.basename(p) for p in _backup_paths))
                    if self._btn_next_iter:
                        self._btn_next_iter.configure(state='normal')
                    # Render the curve into the retrain popup; mark done unless
                    # an auto-iterate round will continue into the same window.
                    rw = getattr(self, '_retrain_window', None)
                    if rw is not None and not getattr(rw, '_closed', True):
                        rw.draw_curve(self._draw_curve_into)
                        _fd = getattr(self, '_last_fold_detail', None)
                        if _fd:
                            rw.draw_folds(_fd, getattr(self, '_last_cv_mode', 'session'),
                                          at_operating_point=True)
                        # Headline metrics: honest frame F1 (+gain vs baseline AND vs last
                        # iteration) / bout F1 / AUPRC / smoothing. The current iteration's
                        # record was appended above (tracker.record with oof_f1=honest_f1),
                        # so records[-2].oof_f1 is the previous iteration's honest F1.
                        _mtxt = f"HONEST F1 {honest_f1:.3f}"
                        _deltas = []
                        _base = getattr(self, '_base_clf_f1', None)
                        if _base is not None:
                            _deltas.append(f"{honest_f1 - _base:+.3f} vs baseline {_base:.3f}")
                        _recs = getattr(getattr(self._session, 'tracker', None), 'records', []) \
                            if self._session else []
                        if len(_recs) >= 2 and _recs[-2].oof_f1 is not None:
                            _deltas.append(f"{honest_f1 - _recs[-2].oof_f1:+.3f} vs last iter")
                        if _deltas:
                            _mtxt += "  (Δ " + "; ".join(_deltas) + ")"
                        if honest_bout is not None:
                            _mtxt += f"   |   bout-F1 {honest_bout['f1']:.3f} (±{honest_bout['tol']}fr)"
                        if _oof_ap is not None:
                            _mtxt += f"   |   AUPRC {_oof_ap:.3f}"
                        if _hmm_f1 is not None:
                            _sw = "HMM" if _hmm_f1 > honest_f1 + 0.005 else "morphological"
                            _mtxt += f"   |   smoothing: {_sw}"
                        rw.set_metrics(_mtxt)
                        if not _auto_continue:
                            rw.set_done(True)
                            # Keep the reference so the next 'Next Iteration → retrain'
                            # reuses THIS window (reset_for_new_run) instead of opening a
                            # new one. It's only re-created if the user closes it (_closed).

                self.app.root.after(0, _post_log)
                self.app.root.after(0, self._refresh_plot)
                self.app.root.after(0, self._refresh_classifiers)
                if _auto_continue:
                    self.app.root.after(0, self._do_auto_next)

            except Exception as e:
                import traceback; traceback.print_exc()
                err = str(e)
                self.app.root.after(0, lambda: self._log_msg(f"✗ Retrain failed: {err}"))
                self.app.root.after(0, lambda: messagebox.showerror("Retrain error", err))

        threading.Thread(target=_run, daemon=True).start()

    def _auto_iterate(self):
        if self._session is None:
            messagebox.showwarning("No session", "Load a session first (Score + Histogram).")
            return
        self._auto_remaining = self._auto_iter_var.get()
        self._stop_auto_var.set(False)
        self._log_msg(f"Auto-iterate: {self._auto_remaining} iteration(s) queued.")
        self._start_labeling()

    def _do_auto_next(self):
        """Called after each auto-mode retrain; chains next iteration if remaining."""
        if self._auto_remaining <= 0 or self._stop_auto_var.get():
            self._log_msg("Auto-iterate: done.")
            return
        if (self._last_probas is not None and self._session is not None and
                self._session.is_converged(self._last_probas, self._threshold_var.get())):
            self._log_msg("Auto-iterate: model converged — stopping early.")
            return
        self._log_msg(f"Auto-iterate: {self._auto_remaining} iteration(s) remaining → next round...")
        self._start_labeling()

    def _run_bout_labeling_ui(self, bouts, probas, video_path):
        """Launch BoutLabelingInterface on the main thread, then apply labels."""
        bname = self._session.behavior_name if self._session else "behavior"

        # Read fps from video
        _cap = cv2.VideoCapture(video_path)
        fps = _cap.get(cv2.CAP_PROP_FPS) or 60.0
        _cap.release()

        interface = BoutLabelingInterface(
            video_path=video_path,
            bouts=bouts,
            probas=probas,
            behavior_name=bname,
            fps=fps,
            log_cb=self._log_msg,
        )
        new_labels = interface.run()  # {(start, end): 0 or 1}

        if not new_labels:
            self._log_msg("No bouts labeled.")
            return

        try:
            stats = self._session.apply_labels(
                new_labels=new_labels,
                confidence_threshold=self._threshold_var.get(),
                propagate=self._propagate_var.get(),
                probas=self._last_probas,
                record_curve=False,   # the retrain below does the train+CV+curve point
            )
            # Persist labels/curve (no new record was added here)
            self._session.tracker.save(self._get_curve_path())

            n_bouts_labeled = len(new_labels)
            _added = sum(int(k[-1]) - int(k[-2]) + 1
                         for k in new_labels if isinstance(k, tuple) and len(k) >= 2)
            _total = int(stats.get('n_labeled_total', 0))
            _pct = 100.0 * _added / max(_total - _added, 1)
            msg = (f"Labeled {n_bouts_labeled} bout(s) → +{_added:,} frames "
                   f"(+{_pct:.2f}% of labeled data). Total labeled: {_total:,}.")
            self._log_msg(msg)
            self._refresh_plot()
            self._check_convergence(stats)
            # Auto-retrain: rebuild classifier if both label classes present
            if hasattr(self._session, '_subs'):
                _n_pos = sum(int((s['labels'] == 1).sum()) for s in self._session._subs)
                _n_neg = sum(int((s['labels'] == 0).sum()) for s in self._session._subs)
            else:
                _n_pos = int((self._session._labels == 1).sum())
                _n_neg = int((self._session._labels == 0).sum())
            if _n_pos > 0 and _n_neg > 0:
                if self._auto_remaining > 0:
                    # Auto mode: skip dialog, retrain and chain
                    self._auto_remaining -= 1
                    self._log_msg("Auto mode: retraining...")
                    self._retrain_and_compare(_auto_continue=True)
                else:
                    _slow = ("\n\n⚠ Retraining trains on ALL {0:,} labeled frames and will "
                             "use every CPU core — this may take a few minutes on large "
                             "label sets (sparse AL on new videos is much faster)."
                             ).format(_n_pos + _n_neg) if (_n_pos + _n_neg) > 100_000 else ""
                    do_retrain = messagebox.askyesno(
                        "Retraining will now commence",
                        f"Labeling complete — {len(new_labels)} bout(s) → +{_added:,} frames "
                        f"(+{_pct:.1f}% labeled data; {_n_pos:,} pos / {_n_neg:,} neg total).\n\n"
                        "Retraining the classifier will now commence — a progress window "
                        "with the live learning curve will open." + _slow + "\n\n"
                        "Proceed now?  (Choose 'No' to review labels first and retrain "
                        "manually later.)")
                    if do_retrain:
                        self._log_msg("Retraining classifier...")
                        self._retrain_and_compare()
                    else:
                        self._log_msg("Retrain deferred — click 'Retrain & Save Snapshot' when ready.")
            else:
                self._log_msg("Auto-retrain skipped — need at least one YES and one NO bout.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error applying labels", str(e))

    def _on_bout_var_edit(self, which):
        """Clear the 'auto' indicator when the user manually edits the value."""
        if self._bout_auto_guard:
            return
        ind = self._min_bout_ind if which == 'min' else self._max_bout_ind
        if ind is not None:
            try:
                ind.config(text="")
            except Exception:
                pass

    def _auto_bouts_from_labels(self):
        """Auto-populate Min/Max bout frames from the labels once a classifier and
        labeled session(s) are loaded. Silent (no popups); no-op if nothing labeled."""
        try:
            if not self._clf_combo.get():
                return
            has_labeled = self._session is not None or any(
                self._session_has_labels(s) for s in self._get_selected_sessions())
            if has_labeled:
                self._auto_detect_bout_lengths(silent=True)
        except Exception:
            pass

    def _auto_detect_bout_lengths(self, silent=False):
        """Scan label data to set Min/Max bout frames from actual positive bouts.
        silent=True suppresses the popups (used for auto-population on load)."""
        import numpy as _np

        # --- Gather label sources: (labels_array, identifier, video_path_or_None) ---
        sources = []
        if self._session is not None:
            # Session already loaded (post-scoring): read in-memory arrays
            if hasattr(self._session, '_subs'):
                for sub in self._session._subs:
                    sources.append((sub['labels'],
                                    os.path.basename(sub['video_path']),
                                    sub['video_path']))
            else:
                sources.append((self._session._labels, "(loaded session)", None))
        else:
            # Pre-scoring fallback: read label CSVs directly from selected sessions
            selected = self._get_selected_sessions()
            if not selected:
                if not silent:
                    messagebox.showwarning("No session selected",
                        "Select a session in the list first.")
                return
            # Target behavior (from the warm-start classifier) so we read the RIGHT
            # column, not df.columns[0] (which is 'rearing' in multi-behavior CSVs).
            _beh = ((self._load_selected_classifier_data() or {}).get('Behavior_type')
                    or (self._load_selected_classifier_data() or {}).get('behavior_name'))
            for s in selected:
                lcsv = s.get('labels') or s.get('target_path')
                if not lcsv or not os.path.isfile(lcsv):
                    self._log_msg(f"  ⚠ Labels CSV not found: {lcsv}")
                    continue
                try:
                    import pandas as _pd
                    df = _pd.read_csv(lcsv)
                    col = _select_behavior_column(df, _beh)
                    if col is None:
                        self._log_msg(f"  ⚠ '{_beh}' not in {os.path.basename(lcsv)} — skipped")
                        continue
                    raw = df[col].values
                    labels = _np.where(_np.isnan(raw.astype(float)), -1, raw.astype(int))
                    vpath = s.get('video') or s.get('video_path')
                    sources.append((labels, s['session_name'], vpath))
                except Exception as e:
                    self._log_msg(f"  ⚠ Could not read {lcsv}: {e}")

        if not sources:
            if not silent:
                messagebox.showinfo("No labels", "No readable label files found.")
            return

        # --- FPS helper ---
        def _fps_for(vpath):
            if not vpath:
                return None
            try:
                import cv2 as _cv2
                cap = _cv2.VideoCapture(vpath)
                fps = cap.get(_cv2.CAP_PROP_FPS)
                cap.release()
                return float(fps) if fps and fps > 0 else None
            except Exception:
                return None

        def _fmt_bout(length, start, ident, vpath):
            fps = _fps_for(vpath)
            loc = f"frame {start}"
            if fps:
                loc += f" / {start / fps:.1f} s"
            return f"{length} frames, starts {loc} — \"{ident}\""

        # --- Find positive bout records: (length, start_frame, identifier, video_path) ---
        bout_records = []
        for labels, ident, vpath in sources:
            pos = (labels == 1).astype(int)
            if pos.sum() == 0:
                continue
            padded = _np.concatenate([[0], pos, [0]])
            starts = _np.where(_np.diff(padded) == 1)[0]
            ends   = _np.where(_np.diff(padded) == -1)[0]
            for s, e in zip(starts, ends):
                bout_records.append((int(e - s), int(s), ident, vpath))

        if not bout_records:
            if not silent:
                messagebox.showinfo("No labels",
                    "No positive-labeled frames found in the selected session(s).")
            return

        arr = _np.array([r[0] for r in bout_records])
        min_len   = int(arr.min())
        pct90_len = int(_np.percentile(arr, 90))
        max_len   = int(arr.max())
        median    = int(_np.median(arr))

        min_rec = next(r for r in bout_records if r[0] == min_len)
        max_rec = next(r for r in bout_records if r[0] == max_len)
        pct90_rec = min(bout_records, key=lambda r: abs(r[0] - pct90_len))  # source nearest the set value

        def _src(rec):
            _l, _start, _ident, _vp = rec
            _short = _ident.split('/')[-1]
            _short = ('…' + _short[-15:]) if len(_short) > 16 else _short
            _fps = _fps_for(_vp)
            _t = f" {_start/_fps:.0f}s" if _fps else ""
            return f"{_short} f{_start}{_t}"

        self._bout_auto_guard = True
        self._min_spacing_var.set(max(1, min_len))
        self._max_bout_var.set(pct90_len)
        # Indicators next to the boxes: mark auto-populated + show the source bout.
        if getattr(self, '_min_bout_ind', None) is not None:
            self._min_bout_ind.config(text=f"⟵ auto: {_src(min_rec)}")
        if getattr(self, '_max_bout_ind', None) is not None:
            self._max_bout_ind.config(text=f"⟵ auto(90th-pct): {_src(pct90_rec)}")
        self._bout_auto_guard = False

        self._log_msg(
            f"Auto-detected {len(arr)} positive bouts — "
            f"min={min_len}  median={median}  90th-pct={pct90_len}  max={max_len} frames\n"
            f"  Min bout ← {_fmt_bout(*min_rec)}\n"
            f"  Max bout ← 90th-pct, nearest: {_fmt_bout(*pct90_rec)}\n"
            f"  → Min bout frames = {max(1, min_len)}, Max bout frames = {pct90_len}"
        )

    def _check_convergence(self, stats):
        """Check auto-convergence and plateau after labeling."""
        if self._last_probas is not None:
            if self._session.is_converged(self._last_probas, self._threshold_var.get()):
                if messagebox.askyesno("Converged",
                        "No uncertain frames remain.\n\nSave final classifier now?"):
                    self._retrain_and_compare()
                return

        records = self._session.tracker.records
        if len(records) >= 3:
            last_3_cv = [r.oof_f1 for r in records[-3:] if r.oof_f1 is not None]
            if len(last_3_cv) == 3 and (max(last_3_cv) - min(last_3_cv)) < 0.01:
                if messagebox.askyesno("Plateau Detected",
                        f"OOF F1 stable at ~{last_3_cv[-1]:.3f} for 3 iterations.\n\n"
                        "Save final classifier and stop?"):
                    self._retrain_and_compare()
                    self._log_msg("Convergence — final classifier saved.")

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _run_discovery(self):
        if self._session is None:
            messagebox.showwarning("No session", "Score a session first.")
            return

        project_folder = self.app.current_project_folder.get()
        if hasattr(self._session, 'labels_csv'):
            labels_csv = self._session.labels_csv
            features_cache = self._session.features_cache
            behavior_name = self._session.behavior_name
        else:
            # MultiSessionAL — use first sub-session
            first = self._session._subs[0]
            labels_csv = first['labels_csv']
            features_cache = first['features_cache']
            behavior_name = self._session.behavior_name

        self._log_msg("Starting directed discovery (UMAP + HDBSCAN on positive frames)...")

        def _run():
            out = run_directed_discovery(
                project_folder=project_folder,
                labels_csv=labels_csv,
                features_cache=features_cache,
                behavior_name=behavior_name,
                run_name='al_discovery',
            )
            if out:
                self.app.root.after(0, lambda: self._log_msg(f"Discovery complete: {out}"))
                self.app.root.after(0, lambda: messagebox.showinfo(
                    "Discovery complete",
                    f"Sub-behavior clusters saved to:\n{out}\n\n"
                    "Open the Discover tab to visualize clusters."))
            else:
                self.app.root.after(0, lambda: self._log_msg(
                    "Discovery failed or insufficient positive frames (need >=50). "
                    "Ensure umap-learn and hdbscan are installed."))

        threading.Thread(target=_run, daemon=True).start()

    # ------------------------------------------------------------------
    # Threshold + eligible count
    # ------------------------------------------------------------------

    def _update_eligible_count(self):
        if self._last_probas is None or self._session is None:
            return
        t = self._threshold_var.get()
        n_frames, n_bouts, _ = self._session.count_eligible(
            self._last_probas, t, self._min_spacing_var.get())
        sess = self._session
        if hasattr(sess, '_labels'):
            cur_lab = int((sess._labels >= 0).sum())
        else:
            cur_lab = sum(int((s['labels'] >= 0).sum()) for s in sess._subs)
        # Refresh cached stats (threshold changes the eligible bouts) then redraw
        # the estimate so both the threshold slider and the bout-count spinbox stay live.
        self._last_score_stats = {
            'n_pos': int(sess.count_positive()), 'n_eligible': int(n_frames),
            'n_bouts': int(n_bouts), 'avg_len': float(n_frames / max(n_bouts, 1)),
            'cur_lab': int(cur_lab),
        }
        self._update_bout_estimate()

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _draw_empty_curve(self):
        ax = self._lc_ax
        ax.clear()
        ax.text(0.5, 0.62, "No iterations yet\nRun an iteration to build the curve",
                transform=ax.transAxes, ha='center', va='center',
                fontsize=9, color='#888888', style='italic')
        # Anchor on the honest baseline (starting point) when a warm-start clf is loaded.
        if self._base_clf_f1 is not None:
            ax.axhline(self._base_clf_f1, color='#e6a817', linestyle=':', linewidth=1.4)
            ax.set_ylim(0, 1)
            ax.text(0.5, 0.30, f"Honest baseline F1 = {self._base_clf_f1:.3f}",
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=8, color='#b8860b')
            ax.set_xticks([])
        else:
            ax.set_xticks([]); ax.set_yticks([])
        self._lc_canvas.draw()

    def _seed_baseline_curve(self):
        """Start the learning curve from the warm-start classifier's HONEST F1 instead
        of an inflated/leaky value. Purges stale leaky points (cv_mode not honest AND
        oof_f1 implausibly above the baseline) and, if no honest record exists yet, seeds
        an iteration-0 'baseline' point at _base_clf_f1 so the curve reads honest-vs-honest
        rather than dropping from a frame-level-inflated start."""
        sess = getattr(self, '_session', None)
        base = getattr(self, '_base_clf_f1', None)
        if sess is None or base is None:
            return
        try:
            tr = sess.tracker
        except Exception:
            return
        HONEST = ('session', 'baseline')
        # Keep honest history + anything at/under the baseline; drop only legacy/leaky
        # points that sit implausibly above the honest baseline (the frame-level 0.945).
        kept = [r for r in tr.records
                if getattr(r, 'cv_mode', None) in HONEST
                or r.oof_f1 is None
                or r.oof_f1 <= base + 0.08]
        n_removed = len(tr.records) - len(kept)
        tr.records = kept
        has_honest = any(getattr(r, 'cv_mode', None) in HONEST for r in tr.records)
        seeded = False
        if not has_honest:
            try:
                if hasattr(sess, '_labels'):
                    n_lab = int(np.sum(sess._labels >= 0))
                    n_pos = int(np.sum(sess._labels == 1))
                else:
                    n_lab = sum(int(np.sum(s['labels'] >= 0)) for s in sess._subs)
                    n_pos = sum(int(np.sum(s['labels'] == 1)) for s in sess._subs)
            except Exception:
                n_lab, n_pos = 0, 0
            base_rec = ALIterationRecord(
                iteration=0, n_labeled_total=n_lab, n_positive=n_pos,
                train_f1=float(base), oof_f1=float(base), n_below_threshold=0,
                timestamp=datetime.now().isoformat(),
                oof_precision=None, oof_recall=None, held_out_f1=None,
                cv_mode='baseline')
            tr.records.insert(0, base_rec)
            seeded = True
        if seeded or n_removed:
            for i, r in enumerate(tr.records):
                r.iteration = i
            try:
                tr.save(self._get_curve_path())
            except Exception:
                pass
            if n_removed:
                self._log_msg(f"Curve: dropped {n_removed} stale leaky point(s); "
                              f"baseline = warm-start honest F1 {base:.3f}.")
            elif seeded:
                self._log_msg(f"Curve baseline seeded at warm-start honest F1 {base:.3f}.")
            self._refresh_plot()

    def _refresh_plot(self):
        if not MATPLOTLIB_AVAILABLE or self._session is None:
            return
        tracker = self._session.tracker
        if not tracker.records:
            self._draw_empty_curve()
            return

        import numpy as _np
        df = tracker.to_dataframe()
        ax = self._lc_ax
        ax.clear()

        # --- Style ---
        ax.grid(True, alpha=0.3, zorder=0)

        # --- X axis = iteration number (avoids the giant frame-count range) ---
        iters = df['iteration'].values.astype(int)
        train_f1 = df['train_f1'].values

        cv_rows = df.dropna(subset=['oof_f1'])
        cv_iters = cv_rows['iteration'].values.astype(int)
        cv_f1 = cv_rows['oof_f1'].values

        # NOTE: the Train-F1 line + train-vs-CV overfit shading/badge are intentionally
        # NOT plotted — train F1 is near-1.0 and misleads users. The honest OOF/CV curve
        # (and the honest baseline) are what should be read.

        # CV F1 line
        if len(cv_rows) > 0:
            ax.plot(cv_iters, cv_f1, color='#2ca02c', linewidth=1.8,
                    marker='s', markersize=5, zorder=4, label='OOF F1 (CV)')

        # Held-out test-set F1 — unbiased, directly comparable to a Train-tab F1
        if 'held_out_f1' in df.columns:
            _ho_rows = df.dropna(subset=['held_out_f1'])
            if not _ho_rows.empty:
                ax.plot(_ho_rows['iteration'].values.astype(int),
                        _ho_rows['held_out_f1'].values, color='#d2691e',
                        linewidth=2.2, marker='D', markersize=5, zorder=6,
                        label='Held-out F1')

        # Precision & Recall lines
        if 'oof_precision' in df.columns and 'oof_recall' in df.columns:
            _pr_rows = df.dropna(subset=['oof_precision', 'oof_recall'])
            if not _pr_rows.empty:
                _pr_iters = _pr_rows['iteration'].values.astype(int)
                ax.plot(_pr_iters, _pr_rows['oof_precision'].values, 's--',
                        color='steelblue', linewidth=1.2, alpha=0.7,
                        markersize=4, zorder=3, label='OOF Precision')
                ax.plot(_pr_iters, _pr_rows['oof_recall'].values, '^--',
                        color='darkorange', linewidth=1.2, alpha=0.7,
                        markersize=4, zorder=3, label='OOF Recall')

        # Per-point n_labeled annotation (small, above each marker) — anchored to the
        # OOF/CV point (train F1 is no longer plotted).
        for _, row in df.iterrows():
            n = int(row['n_labeled_total'])
            label_str = f"{n//1000}k" if n >= 1000 else str(n)
            _yv = row['oof_f1'] if not _np.isnan(row.get('oof_f1', _np.nan)) else row['train_f1']
            ax.annotate(label_str,
                        (int(row['iteration']), _yv),
                        textcoords="offset points", xytext=(0, 6),
                        fontsize=6, color='#555555', ha='center', zorder=5)

        # Baseline classifier reference line
        if self._base_clf_f1 is not None:
            ax.axhline(self._base_clf_f1, color='#e6a817', linestyle=':',
                       linewidth=1.4, zorder=2,
                       label=f'Honest baseline={self._base_clf_f1:.3f}')

        # --- Axes ---
        ax.set_xlim(iters.min() - 0.5, iters.max() + 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Iteration", fontsize=8)
        ax.set_ylabel("F1", fontsize=8)
        ax.set_title("Learning Curve", fontsize=9, pad=4)
        # Integer x-ticks only
        ax.set_xticks(iters)
        ax.legend(fontsize=7, framealpha=0.7, loc='lower right')
        self._lc_canvas.draw()

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def _log_msg(self, msg: str):
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        full = f"[{ts}] {msg}\n"
        try:
            self._log.insert('end', full)
            self._log.see('end')
        except Exception:
            pass
        # Mirror into the retrain popup when one is open.
        rw = getattr(self, '_retrain_window', None)
        if rw is not None and not getattr(rw, '_closed', True):
            rw.log(f"[{ts}] {msg}")

    def _draw_curve_into(self, ax, tracker=None):
        """Compact learning-curve render (train / honest CV / held-out / baseline)
        used by the retrain popup and the saved-session reload. The right panel
        keeps its richer _refresh_plot."""
        if tracker is None:
            tracker = getattr(self._session, 'tracker', None) if self._session else None
        if tracker is None:
            return
        df = tracker.to_dataframe()
        if df.empty:
            return
        it = df['iteration'].values.astype(int)
        # Train F1 intentionally not plotted (near-1.0, misleading); show honest CV only.
        cv = df.dropna(subset=['oof_f1'])
        if not cv.empty:
            ax.plot(cv['iteration'].values.astype(int), cv['oof_f1'].values, '-s',
                    color='#2ca02c', ms=4, label='Honest CV F1')
        if 'held_out_f1' in df.columns:
            ho = df.dropna(subset=['held_out_f1'])
            if not ho.empty:
                ax.plot(ho['iteration'].values.astype(int), ho['held_out_f1'].values, '-D',
                        color='#d2691e', ms=4, label='Held-out F1')
        if self._base_clf_f1 is not None:
            ax.axhline(self._base_clf_f1, color='#e6a817', ls=':', lw=1.2,
                       label=f'Baseline {self._base_clf_f1:.2f}')
        ax.set_xlabel('Iteration'); ax.set_ylabel('F1'); ax.set_ylim(0, 1)
        ax.grid(alpha=0.3); ax.legend(fontsize=7, loc='lower right')
        ax.set_title('Active Learning — Learning Curve', fontsize=9)

    @staticmethod
    def _curve_path(labels_csv: str) -> str:
        base = os.path.splitext(labels_csv)[0]
        return base + '_al_curve.json'

    def _get_curve_path(self):
        """Return curve JSON path for current session (single or multi)."""
        if hasattr(self._session, 'labels_csv'):
            return self._curve_path(self._session.labels_csv)
        # Multi-session: derive from project folder + behavior name
        folder = self.app.current_project_folder.get()
        bname = getattr(self._session, 'behavior_name', 'behavior')
        return os.path.join(folder, 'features', f'{bname}_multi_al_curve.json')


# ============================================================================
# Main Application Entry Point
# ============================================================================
