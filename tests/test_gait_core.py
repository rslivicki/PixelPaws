# -*- coding: utf-8 -*-
"""Fast unit tests for gait_core's pure helpers on synthetic arrays.

Run:  PYTHONIOENCODING=utf-8 python -m pytest tests/test_gait_core.py -v
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import gait_core
from gait_core import (
    debounce, gait_bouts, compute_speed_contact, compute_selection_masks,
    regularity_index, print_position, resample_contour, normalize_contour,
    shape_metrics, shape_metrics_batch, rebin_timecourse, compute_all_metrics,
    recompute_with_contact, GaitContext,
)

FPS = 100.0  # 1 frame = 10 ms; keeps ms→frame arithmetic obvious


# ─────────────────────────────────────────────────────────────────────────────
# debounce
# ─────────────────────────────────────────────────────────────────────────────

def test_debounce_removes_short_runs():
    mask = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1], dtype=bool)  # one 1-frame gap
    out = debounce(mask, min_frames=2)
    assert out.all()                       # gap flipped to surrounding stance


def test_debounce_keeps_long_runs():
    mask = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=bool)
    out = debounce(mask, min_frames=2)
    assert (out == mask).all()


def test_debounce_does_not_mutate_input():
    mask = np.array([1, 0, 1, 1, 1], dtype=bool)
    orig = mask.copy()
    debounce(mask, min_frames=3)
    assert (mask == orig).all()


# ─────────────────────────────────────────────────────────────────────────────
# gait_bouts
# ─────────────────────────────────────────────────────────────────────────────

def test_gait_bouts_basic():
    # stance 10 frames, swing 5, stance 10, swing 5, stance 10
    mask = np.r_[np.ones(10), np.zeros(5), np.ones(10),
                 np.zeros(5), np.ones(10)].astype(bool)
    stance, swing, onsets = gait_bouts(mask, FPS)
    assert onsets == [0, 15, 30]
    assert stance == pytest.approx([0.1, 0.1, 0.1])
    assert swing == pytest.approx([0.05, 0.05])


def test_gait_bouts_all_stance_and_all_swing():
    assert gait_bouts(np.ones(50, dtype=bool), FPS) == ([50 / FPS], [], [0])
    assert gait_bouts(np.zeros(50, dtype=bool), FPS) == ([], [50 / FPS], [])
    assert gait_bouts(np.array([], dtype=bool), FPS) == ([], [], [])


def test_gait_bouts_min_stride_filter():
    # 3-frame stance (30 ms) then 20-frame stance; 50 ms floor drops the first
    mask = np.r_[np.zeros(5), np.ones(3), np.zeros(5), np.ones(20)].astype(bool)
    stance, swing, onsets = gait_bouts(mask, FPS, min_stride_ms=50)
    assert onsets == [13]
    assert stance == pytest.approx([0.2])
    # without the filter both survive
    stance2, _, onsets2 = gait_bouts(mask, FPS, min_stride_ms=0)
    assert onsets2 == [5, 13]


# ─────────────────────────────────────────────────────────────────────────────
# compute_speed_contact
# ─────────────────────────────────────────────────────────────────────────────

def test_speed_contact_stationary_vs_moving():
    n = 200
    x = np.r_[np.zeros(100), np.arange(100) * 5.0]  # still, then 500 px/s
    y = np.zeros(n)
    stance = compute_speed_contact(x, y, FPS, threshold=100.0,
                                   median_ms=0, min_bout_ms=0)
    assert stance[:100].all()
    assert not stance[105:].any()          # (allow filter edge at transition)


def test_speed_contact_short_input():
    assert compute_speed_contact([5.0], [5.0], FPS).tolist() == [True]
    assert compute_speed_contact([], [], FPS).tolist() == []


def test_speed_contact_auto_threshold_is_20th_percentile():
    rng = np.random.RandomState(0)
    x = np.cumsum(rng.rand(500))
    y = np.cumsum(rng.rand(500))
    out = compute_speed_contact(x, y, FPS, threshold='auto',
                                median_ms=0, min_bout_ms=0)
    # ~20% of frames should be below the 20th percentile speed
    assert 0.10 < out.mean() < 0.30


# ─────────────────────────────────────────────────────────────────────────────
# compute_selection_masks (lick + 4-paw gating — the denominators)
# ─────────────────────────────────────────────────────────────────────────────

def _hind_only_masks(n):
    return {'HL': pd.Series(np.ones(n, dtype=bool)),
            'HR': pd.Series(np.ones(n, dtype=bool))}


def test_selection_masks_lick_exclusion():
    n = 100
    lick = np.zeros(n, dtype=bool)
    lick[:40] = True
    analyzed, base, four = compute_selection_masks(
        _hind_only_masks(n), lick, gate_4paw=False, n=n)
    assert base.sum() == 60 and analyzed.sum() == 60
    assert four is None                      # <4 paws present


def test_selection_masks_4paw_gate():
    n = 100
    masks = {r: pd.Series(np.ones(n, dtype=bool))
             for r in ('HL', 'HR', 'FL', 'FR')}
    masks['FR'].iloc[50:] = False            # FR lifts for the second half
    lick = np.zeros(n, dtype=bool)
    lick[:10] = True
    analyzed, base, four = compute_selection_masks(masks, lick, True, n)
    assert four is not None and four.sum() == 50
    assert base.sum() == 90
    assert analyzed.sum() == 40              # frames 10..49
    # gate off: analyzed == base even with 4 paws mapped
    analyzed2, base2, four2 = compute_selection_masks(masks, lick, False, n)
    assert analyzed2.sum() == 90 and four2.sum() == 50


def test_selection_masks_none_lick():
    n = 20
    analyzed, base, four = compute_selection_masks(_hind_only_masks(n),
                                                   None, False, n)
    assert base.all() and analyzed.all()


# ─────────────────────────────────────────────────────────────────────────────
# regularity_index / print_position edge cases
# ─────────────────────────────────────────────────────────────────────────────

def _four_paw_cycle(n_cycles=6, stance=8, gap=2):
    """Perfect walk: HL, HR, FL, FR strike in sequence each cycle."""
    period = 4 * (stance + gap)
    n = n_cycles * period
    masks = {}
    for k, role in enumerate(('HL', 'HR', 'FL', 'FR')):
        m = np.zeros(n, dtype=bool)
        for c in range(n_cycles):
            s = c * period + k * (stance + gap)
            m[s:s + stance] = True
        masks[role] = pd.Series(m)
    return masks, n


def test_regularity_index_perfect_sequence():
    masks, n = _four_paw_cycle()
    ri = regularity_index(masks, FPS, None, None, None, 0)
    assert ri == 100.0


def test_regularity_index_insufficient_data():
    masks = {r: pd.Series(np.zeros(10, dtype=bool))
             for r in ('HL', 'HR', 'FL', 'FR')}
    assert regularity_index(masks, FPS, None, None, None, 0) is None
    # missing a paw
    masks3 = {r: pd.Series(np.ones(10, dtype=bool)) for r in ('HL', 'HR', 'FL')}
    assert regularity_index(masks3, FPS, None, None, None, 0) is None


def test_print_position_requires_two_fore_onsets():
    masks, n = _four_paw_cycle(n_cycles=1)
    paw_xy = {r: (np.zeros(n), np.zeros(n)) for r in masks}
    assert np.isnan(print_position(masks, paw_xy, 'HL', 'FL', FPS,
                                   None, None, None, 0))


def test_print_position_distance():
    masks, n = _four_paw_cycle(n_cycles=4)
    # FL fixed at x=3,y=4 → distance from HL (0,0) is 5
    paw_xy = {r: (np.zeros(n), np.zeros(n)) for r in masks}
    paw_xy['FL'] = (np.full(n, 3.0), np.full(n, 4.0))
    pp = print_position(masks, paw_xy, 'HL', 'FL', FPS, None, None, None, 0)
    assert pp == pytest.approx(5.0)


# ─────────────────────────────────────────────────────────────────────────────
# contour helpers
# ─────────────────────────────────────────────────────────────────────────────

SQUARE = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)


def test_resample_contour():
    out = resample_contour(SQUARE, 64)
    assert out.shape == (64, 2)
    # perimeter approximately preserved (resampling cuts the corners slightly)
    d = np.diff(np.vstack([out, out[:1]]), axis=0)
    assert np.sqrt((d ** 2).sum(1)).sum() == pytest.approx(40.0, rel=5e-2)
    assert resample_contour(SQUARE[:2]) is None


def test_normalize_contour():
    out = normalize_contour(SQUARE, area=100.0)
    assert np.allclose(out.mean(axis=0), 0.0)
    assert out.max() == pytest.approx(0.5)   # 10 px / sqrt(100)
    assert normalize_contour(SQUARE, 0.0) is None
    assert normalize_contour(None, 10.0) is None


def test_shape_metrics_square():
    pts = resample_contour(SQUARE, 64)
    ar, circ = shape_metrics(pts)
    assert ar == pytest.approx(1.0)
    assert circ == pytest.approx(np.pi / 4, rel=5e-2)  # square: 4πA/P² = π/4


def test_shape_metrics_batch_matches_scalar():
    rect = np.array([[0, 0], [20, 0], [20, 5], [0, 5]], dtype=float)
    shapes = [resample_contour(SQUARE, 64), resample_contour(rect, 64)]
    stacked = np.stack(shapes)
    ar_b, circ_b = shape_metrics_batch(stacked)
    for i, pts in enumerate(shapes):
        ar_s, circ_s = shape_metrics(pts)
        assert ar_b[i] == pytest.approx(ar_s)
        assert circ_b[i] == pytest.approx(circ_s)


# ─────────────────────────────────────────────────────────────────────────────
# rebin_timecourse
# ─────────────────────────────────────────────────────────────────────────────

def test_rebin_timecourse():
    xs = [0, 1, 2, 3, 4, 5]
    means = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    errs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    nx, nm, ne = rebin_timecourse(xs, means, errs, 2)
    assert nx == [0, 2, 4]
    assert nm == pytest.approx([1.5, 3.5, 5.5])
    assert ne == pytest.approx([0.15, 0.35, 0.55])


def test_rebin_timecourse_passthrough():
    xs, ms, es = [0, 1], [1.0, 2.0], [0.1, 0.2]
    assert rebin_timecourse(xs, ms, es, 0) == (xs, ms, es)
    assert rebin_timecourse([], [], [], 5) == ([], [], [])


# ─────────────────────────────────────────────────────────────────────────────
# compute_all_metrics + recompute_with_contact on synthetic data
# (denominator behaviour: lick exclusion must reach BOTH paths)
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_data(n=1000, fps=FPS):
    hl = np.zeros(n, dtype=bool); hl[:600] = True        # 60% raw contact
    hr = np.zeros(n, dtype=bool); hr[:300] = True        # 30% raw contact
    heights = pd.DataFrame({
        'hlpaw_Height': np.where(hl, 5.0, 50.0),
        'hrpaw_Height': np.where(hr, 5.0, 50.0),
    })
    lick = np.zeros(n, dtype=bool); lick[500:] = True    # last half licking
    return {
        'session_name': 'synthetic',
        'height_df': heights,
        'bp_xcord': None, 'bp_ycord': None, 'bp_prob': None,
        'fps': fps, '_used_fallback_fps': True, 'n_frames': n,
        'active_paws': {'HL': 'hlpaw', 'HR': 'hrpaw'},
        'contact_masks': {'HL': pd.Series(hl), 'HR': pd.Series(hr)},
        'paw_xy': {},
        'brightness_series': {}, 'paw_contour_data': {},
        'confidence_mask': None, 'loco_mask': None, 'body_speed': None,
        'frame_displacements': None,
        'lick_mask': lick,
        'params': {'contact_threshold': 15, 'height_window': 500,
                   'bin_seconds': 5, 'bin_unit': 'seconds',
                   'fallback_fps': fps, 'use_brightness': False,
                   'brt_weight': 0.0, 'contact_method': 'height',
                   'speed_threshold': 'auto', 'median_filter_ms': 0,
                   'min_bout_ms': 0, 'min_stance_ms': 0,
                   'exclude_licking': True, 'gate_4paw': False},
        '_mm_per_px': None,
    }


def test_compute_all_metrics_lick_denominator():
    data = _synthetic_data()
    params = data['params']
    sel = compute_selection_masks(data['contact_masks'], data['lick_mask'],
                                  False, data['n_frames'])
    out = compute_all_metrics(data, sel, params)
    s = out['summary']
    assert s['n_frames'] == 1000
    assert s['n_analyzed'] == 500                       # licking excluded
    # HL contact within analyzed window: frames 0..499 all True → 100%
    assert s['contact_pct_HL'] == 100.0
    assert s['n_contact_HL'] == 500
    # HR: 300 of the 500 analyzed frames → 60%
    assert s['contact_pct_HR'] == 60.0
    assert s['WBI_hind'] == pytest.approx(62.5)         # 100/(100+60)*100
    # bins: 1000 frames / (5 s * 100 fps) = 2 bins
    assert len(out['bins']) == 2
    assert out['bins'][0]['bin_index'] == 0
    assert out['bins'][0]['bin_start_s'] == 0.0
    assert out['bins'][1]['n_analyzed'] == 0            # all licking


def test_compute_all_metrics_frame_slice_only():
    data = _synthetic_data()
    sel = compute_selection_masks(data['contact_masks'], data['lick_mask'],
                                  False, data['n_frames'])
    out = compute_all_metrics(data, sel, data['params'],
                              frame_slice=slice(0, 500))
    assert out['bins'] == []
    assert out['summary']['n_frames'] == 500
    assert out['summary']['n_analyzed'] == 500


def test_recompute_with_contact_keeps_lick_exclusion():
    """The old _recompute_contact DROPPED the licking/4-paw masks (drifted
    duplicate). The unified path must keep them."""
    data = _synthetic_data()
    ctx = GaitContext(project_folder='')
    out = recompute_with_contact(
        data, {'contact_method': 'height', 'contact_threshold': 15,
               'speed_threshold': 'auto', 'median_filter_ms': 0,
               'min_bout_ms': 0, 'brt_weight': 0.0}, ctx)
    s = out['summary']
    assert s['n_analyzed'] == 500                       # lick mask still applied
    assert s['contact_pct_HL'] == 100.0
    assert s['contact_pct_HR'] == 60.0
    # masks were rebuilt from height_df and stored back
    assert data['contact_masks']['HL'].sum() == 600
    assert data['params']['contact_threshold'] == 15


def test_recompute_with_contact_new_threshold_changes_masks():
    data = _synthetic_data()
    ctx = GaitContext(project_folder='')
    out = recompute_with_contact(
        data, {'contact_method': 'height', 'contact_threshold': 100,
               'speed_threshold': 'auto', 'median_filter_ms': 0,
               'min_bout_ms': 0, 'brt_weight': 0.0}, ctx)
    s = out['summary']
    # threshold 100 px → every frame is "contact" for both paws
    assert s['contact_pct_HL'] == 100.0 and s['contact_pct_HR'] == 100.0
    assert data['contact_masks']['HR'].sum() == 1000


def test_recompute_speed_threshold_string_parsing():
    data = _synthetic_data()
    ctx = GaitContext(project_folder='')
    recompute_with_contact(
        data, {'contact_method': 'height', 'speed_threshold': '42.5'}, ctx)
    assert data['params']['speed_threshold'] == 42.5
    # verbatim old-quirk: only exact lowercase 'auto' is normalized; other
    # spellings of auto pass through unchanged (harmless for height method)
    recompute_with_contact(
        data, {'contact_method': 'height', 'speed_threshold': 'AUTO'}, ctx)
    assert data['params']['speed_threshold'] == 'AUTO'
    recompute_with_contact(
        data, {'contact_method': 'height', 'speed_threshold': 'garbage'}, ctx)
    assert data['params']['speed_threshold'] == 'auto'


# ─────────────────────────────────────────────────────────────────────────────
# misc small API
# ─────────────────────────────────────────────────────────────────────────────

def test_extract_behavior_name():
    f = gait_core.extract_behavior_name
    assert f('mouse1_vehPixelPaws_Left_licking_predictions.csv') == 'Left_licking'
    assert f('Left_licking_predictions.csv') == 'Left_licking'
    assert f('whatever.csv') == 'whatever'


def test_resolve_subject_ladder():
    key = pd.DataFrame({'Subject': ['mouse1', '4321'],
                        'Treatment': ['Veh', 'Drug']})
    assert gait_core.resolve_subject('mouse1_veh', key) == 'mouse1'
    assert gait_core.resolve_subject('pre_4321_post', key) == '4321'
    # prefix strip (no key hit)
    assert gait_core.resolve_subject('exp_ABC_x', None,
                                     strip_prefix='exp_') == 'ABC'
    # 4-digit token heuristic
    assert gait_core.resolve_subject('cage_9876_day1', None) == '9876'


def test_get_treatment():
    key = pd.DataFrame({'Subject': ['m1'], 'Treatment': ['Veh']})
    assert gait_core.get_treatment('m1', key) == 'Veh'
    assert gait_core.get_treatment('m2', key) == ''
    assert gait_core.get_treatment('m1', None) == ''


def test_gait_context_defaults():
    ctx = GaitContext(project_folder='X')
    assert ctx.pawlike_thresholds == {'solidity': 1.00, 'aspect_ratio': 1.6,
                                      'circularity': 0.10}
    # instances must not share the dict
    ctx.pawlike_thresholds['solidity'] = 0.5
    assert GaitContext().pawlike_thresholds['solidity'] == 1.00
