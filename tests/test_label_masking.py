"""Regression tests for unlabeled-frame (-1 / NaN) handling, BORIS pairing and the
training early-stopping split (bug sweep 2026-09-18)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import boris_import as bi
from evaluation_tab import (EvaluationTab, count_bouts, _mask_unlabeled,
                            _zero_unlabeled, _bout_filter_by_session,
                            _slices_from_lengths, _pool_bout_counts)


# ── evaluation_tab: unlabeled frames never reach a metric ────────────────────

def test_mask_unlabeled_drops_minus_one_and_nan():
    y_true = np.array([0, 1, -1, np.nan, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    yt, yp = _mask_unlabeled(y_true, y_pred)
    assert yt.tolist() == [0, 1, 1, 0]
    assert yp.tolist() == [0, 1, 0, 0]


def test_zero_unlabeled_keeps_timeline():
    yt, yp = _zero_unlabeled(np.array([1, -1, 1, np.nan]), np.array([1, 1, 1, 1]))
    assert yt.tolist() == [1, 0, 1, 0]
    assert yp.tolist() == [1, 0, 1, 0]


def test_count_bouts_minus_one_ends_a_bout():
    assert count_bouts(np.array([1, 1, -1, 1]), 60.0)['n_bouts'] == 2


def test_grid_search_accepts_unlabeled_frames():
    rng = np.random.default_rng(0)
    y = (rng.random(400) > 0.7).astype(float)
    proba = np.clip(y * 0.6 + rng.random(400) * 0.4, 0, 1)
    y[::7] = -1      # previously: f1_score raised "multiclass"
    y[::11] = np.nan
    best = EvaluationTab._grid_search_params(proba, y)
    assert 0.0 <= best['f1'] <= 1.0


def test_bout_filter_by_session_does_not_bridge_boundaries():
    y = np.array([1, 1, 0, 0, 0, 1, 1])
    pooled = _bout_filter_by_session(y, None, 1, 0, 5)
    split = _bout_filter_by_session(y, _slices_from_lengths([4, 3]), 1, 0, 5)
    assert pooled.tolist() == [1, 1, 1, 1, 1, 1, 1]
    assert split.tolist() == [1, 1, 0, 0, 0, 1, 1]


def test_pool_bout_counts_sums_sessions():
    a = EvaluationTab._compute_bout_metrics(np.array([1, 1, 0]), np.array([1, 0, 0]))
    b = EvaluationTab._compute_bout_metrics(np.array([0, 1, 1]), np.array([0, 0, 0]))
    pooled = _pool_bout_counts([a, b])
    assert pooled['n_true_bouts'] == 2 and pooled['n_pred_bouts'] == 1
    assert pooled['bout_precision'] == 1.0 and pooled['bout_recall'] == 0.5


# ── boris_import: POINT events, per-subject pairing, OR-ed overlaps ──────────

def _ev(frame, subject, code, fps=10.0):
    return [frame / fps, subject, code, '', '', frame]


def test_point_events_mark_single_frames():
    events = [_ev(10, 'A', 'flinch'), _ev(20, 'A', 'flinch')]
    df, _ = bi.build_label_frame(events, ['flinch'], 30, 10.0,
                                 behavior_types={'flinch': 'Point event'})
    col = df['flinch'].to_numpy()
    assert col[10] == 1 and col[20] == 1
    assert (col[11:20] == 0).all()        # not a bout to the next point
    assert (col[21:] == -1).all()         # unobserved tail survives


def test_state_bouts_pair_per_subject_and_or_overlaps():
    events = [_ev(10, 'A', 'lick'), _ev(20, 'B', 'lick'),
              _ev(30, 'A', 'lick'), _ev(40, 'B', 'lick')]
    df, _ = bi.build_label_frame(events, ['lick'], 50, 10.0,
                                 behavior_types={'lick': 'State event'})
    col = df['lick'].to_numpy()
    assert (col[10:40] == 1).all()        # overlapping bouts no longer cancel
    assert (col[:10] == 0).all()
    assert (col[40:] == -1).all()


def test_state_bouts_counts_unclosed_per_subject():
    events = [_ev(5, 'A', 'lick'), _ev(8, 'A', 'lick'), _ev(9, 'B', 'lick')]
    bouts, n_open = bi._state_bouts(events, 'lick', 10.0)
    assert bouts == [(5, 8)] and n_open == 1


# ── PixelPaws_GUI: early stopping never uses the held-out fold ───────────────

@pytest.fixture(scope='module')
def gui_cls():
    import PixelPaws_GUI
    return PixelPaws_GUI.PixelPawsGUI


def test_early_stop_split_holds_out_training_sessions(gui_cls):
    sids = np.repeat([0, 1, 2], 10)
    y = np.tile([0, 1], 15)
    train_idx = np.where(sids != 2)[0]     # session 2 is the scored fold
    fit_idx, es_idx, _ = gui_cls._early_stop_split(train_idx, sids, y)
    assert len(es_idx) and set(sids[es_idx]).isdisjoint(set(sids[fit_idx]))
    assert 2 not in set(sids[es_idx]) | set(sids[fit_idx])


def test_early_stop_split_single_session_uses_tail(gui_cls):
    y = np.array([0, 1] * 10)
    fit_idx, es_idx, note = gui_cls._early_stop_split(np.arange(20), np.zeros(20, int), y)
    assert es_idx.tolist() == [16, 17, 18, 19] and 'tail' in note
    _, es_none, _ = gui_cls._early_stop_split(np.arange(20), np.zeros(20, int),
                                              np.zeros(20, int))
    assert len(es_none) == 0              # no positives -> early stopping off


def test_looks_like_boris_export(gui_cls):
    assert gui_cls._looks_like_boris_export(
        ['Observation id', 'Subject', 'Behavior', 'Behavior type', 'Time'])
    assert not gui_cls._looks_like_boris_export(['Frame', 'Left_licking', 'Grooming'])
