"""Cross-validation helpers for classifier training.

What the Train tab needs after the out-of-fold probabilities are in: the
post-processing sweep (threshold, minimum bout, gap), the honest out-of-fold
F1, which sessions can take part in cross-validation, and label clean-up by
bout length.
"""
import numpy as np


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
    any CSV.
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


# Sweep grid — MUST mirror PixelPaws_GUI._sweep_postprocessing exactly so the parallel
# path produces bit-identical (threshold, min_bout, min_after_bout, max_gap, f1).
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
