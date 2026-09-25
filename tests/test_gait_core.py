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
# gait_cycles (validity-aware bouts + per-stride pairing)
# ─────────────────────────────────────────────────────────────────────────────

def test_gait_cycles_matches_gait_bouts_without_gate():
    rng = np.random.RandomState(3)
    for trial in range(40):
        mask = rng.rand(300) < rng.uniform(0.2, 0.8)
        mask = debounce(mask, 2)
        for ms in (0, 50):
            st, sw, on = gait_bouts(mask, FPS, min_stride_ms=ms)
            cyc = gait_core.gait_cycles(mask, FPS, min_stride_ms=ms)
            assert cyc['stance_durs'] == st
            assert cyc['swing_durs'] == sw
            assert cyc['stance_onsets'] == on
            # every stance but the last is followed by its swing
            assert len(cyc['strides']) == len(sw)
    assert gait_core.gait_cycles(np.ones(50, bool), FPS)['stance_durs'] == [0.5]
    assert gait_core.gait_cycles(np.zeros(50, bool), FPS)['swing_durs'] == [0.5]


def test_gait_cycles_invalid_frames_never_create_bouts():
    # stances 20-140 (120 f) and 200-260 (60 f); 20 invalid frames inside
    # the first stance.  mask & valid used to give stances [50, 50, 60] and
    # a fake 20-frame swing.
    m = np.zeros(300, bool); m[20:140] = True; m[200:260] = True
    valid = np.ones(300, bool); valid[70:90] = False
    cyc = gait_core.gait_cycles(m, FPS, valid=valid)
    assert cyc['stance_durs'] == pytest.approx([0.60])     # 120-f stance dropped
    assert cyc['stance_onsets'] == [200]
    assert cyc['swing_durs'] == pytest.approx([0.60])      # 140-200, all valid
    assert cyc['strides'] == []                            # no valid stance+swing pair
    # an invalid swing is dropped, and so is the stride it would close
    valid2 = np.ones(300, bool); valid2[150:160] = False
    cyc2 = gait_core.gait_cycles(m, FPS, valid=valid2)
    assert cyc2['stance_durs'] == pytest.approx([1.2, 0.6])
    assert cyc2['swing_durs'] == [] and cyc2['strides'] == []


def _stride_mask():
    # stances (start, len): swings between them are 10, 12, 15, 14 frames
    bouts = [(10, 20), (40, 18), (70, 25), (110, 6), (130, 30)]
    m = np.zeros(170, bool)
    for s, n in bouts:
        m[s:s + n] = True
    return m


def _gait_data(mask, loco=None, fps=FPS):
    n = len(mask)
    return {
        'session_name': 'syn', 'fps': fps, '_used_fallback_fps': True,
        'n_frames': n, 'height_df': pd.DataFrame({'hlpaw_Height': np.where(mask, 5.0, 50.0)}),
        'bp_xcord': None, 'bp_ycord': None, 'bp_prob': None,
        'active_paws': {'HL': 'hlpaw'}, 'contact_masks': {'HL': pd.Series(mask)},
        'paw_xy': {}, 'brightness_series': {}, 'paw_contour_data': {},
        'confidence_mask': None, 'loco_mask': loco, 'body_speed': None,
        'frame_displacements': None, 'lick_mask': np.zeros(n, dtype=bool),
        'params': {'contact_threshold': 15, 'height_window': 500,
                   'bin_seconds': 3600, 'bin_unit': 'seconds',
                   'fallback_fps': fps, 'use_brightness': False,
                   'brt_weight': 0.0, 'contact_method': 'height',
                   'speed_threshold': 'auto', 'median_filter_ms': 0,
                   'min_bout_ms': 0, 'min_stance_ms': 0,
                   'exclude_licking': False, 'gate_4paw': False,
                   'compute_gait': True},     # gait metrics are opt-in
        '_mm_per_px': None,
    }


def _summary(data):
    sel = compute_selection_masks(data['contact_masks'], data['lick_mask'],
                                  False, data['n_frames'])
    return compute_all_metrics(data, sel, data['params'])['summary']


def test_stride_duty_cadence_are_per_stride():
    # strides = stance_i + following swing_i = 30, 30, 40, 20 frames
    # -> mean stride 0.30 s, duty mean(20/30, 18/30, 25/40, 6/20) = 54.79 %,
    #    cadence 60 / 0.30 = 200 /min.  mean(stance)+mean(swing) gave
    #    0.3255 s / 60.8 % / 184.3 /min.
    s = _summary(_gait_data(_stride_mask()))
    assert s['stride_dur_HL'] == pytest.approx(0.30)
    assert s['duty_cycle_HL'] == pytest.approx(
        round(np.mean([20 / 30, 18 / 30, 25 / 40, 6 / 20]) * 100, 2))
    assert s['cadence_HL'] == pytest.approx(200.0)
    assert s['stance_dur_HL'] == pytest.approx(round(np.mean([20, 18, 25, 6, 30]) / FPS, 4))
    assert s['swing_dur_HL'] == pytest.approx(round(np.mean([10, 12, 15, 14]) / FPS, 4))
    assert s['n_strides_HL'] == 5


def test_loco_gate_inside_stance_does_not_split_it():
    m = np.zeros(300, bool); m[20:140] = True; m[200:260] = True
    loco = np.ones(300, bool); loco[70:90] = False
    s = _summary(_gait_data(m, loco=loco))
    # all-frames block is ungated: 2 stances, 1 swing, 1 stride
    assert s['n_strides_HL'] == 2
    assert s['stride_dur_HL'] == pytest.approx(1.8)
    # loco block: the stance overlapping non-locomotion frames is dropped,
    # the 60-frame swing stays, no fake 20-frame swing, no complete stride
    assert s['loco_n_strides_HL'] == 1
    assert s['loco_stance_dur_HL'] == pytest.approx(0.6)
    assert s['loco_swing_dur_HL'] == pytest.approx(0.6)
    assert np.isnan(s['loco_stride_dur_HL'])
    assert np.isnan(s['loco_duty_cycle_HL']) and np.isnan(s['loco_cadence_HL'])


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
# compute_selection_masks (lick + 4-paw gating - the denominators)
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
    params = {**data['params'], 'compute_gait': True}
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
               'min_bout_ms': 0, 'brt_weight': 0.0,
               'compute_gait': True}, ctx)
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
               'min_bout_ms': 0, 'brt_weight': 0.0,
               'compute_gait': True}, ctx)
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

def test_contour_gate_reports_hind_indices_as_nan():
    """Under the contour gate both hind paws share one mask, so WBI/SI/SBI
    hind would always read 50/0/0; the engine reports them as NaN instead."""
    data = _synthetic_data()
    params = {**data['params'], 'contact_method': 'contour_area',
              'compute_gait': True}
    sel = compute_selection_masks(data['contact_masks'], data['lick_mask'],
                                  False, data['n_frames'])
    out = compute_all_metrics(data, sel, params)
    for row in [out['summary']] + out['bins']:
        for k in ('WBI_hind', 'SI_hind', 'SBI_hind'):
            assert np.isnan(row[k]), (k, row[k])
    # everything else is untouched
    assert out['summary']['contact_pct_HL'] == 100.0
    assert out['summary']['WBI_HL'] == pytest.approx(62.5)


def test_full_stance_contour_metrics_populate_every_bin():
    """Regression: stance_mask_all is built from the already bin-sliced masks
    and was sliced a second time with the absolute bin slice, leaving every
    bin after the first NaN for the *_stance_* contour metrics."""
    data = _synthetic_data()
    n = data['n_frames']
    ones = pd.Series(np.ones(n, dtype=bool))
    data['contact_masks'] = {'HL': ones, 'HR': ones.copy()}
    data['lick_mask'] = np.zeros(n, dtype=bool)
    areas = np.arange(1, n + 1, dtype=float)
    for role in ('HL', 'HR'):
        data['paw_contour_data'][role] = {
            'areas': areas.copy(), 'spreads': areas.copy(),
            'intensities': areas.copy(), 'widths': areas.copy(),
            'solidities': np.full(n, 0.8), 'aspect_ratios': np.full(n, 1.2),
            'circularities': np.full(n, 0.5),
        }
    sel = compute_selection_masks(data['contact_masks'], data['lick_mask'],
                                  False, n)
    out = compute_all_metrics(data, sel, data['params'])
    assert len(out['bins']) == 2
    b0, b1 = out['bins']
    assert b0['paw_area_stance_HL'] == pytest.approx(areas[:500].mean(), abs=0.01)
    assert b1['paw_area_stance_HL'] == pytest.approx(areas[500:].mean(), abs=0.01)
    for k in ('contact_intensity_stance_HR', 'paw_area_ratio_stance_hind',
              'contact_intensity_ratio_stance_hind'):
        assert np.isfinite(b1[k]), k


# ─────────────────────────────────────────────────────────────────────────────
# compute_gait: default = paw-contour + ROI-brightness columns only
# ─────────────────────────────────────────────────────────────────────────────

ROLES_ALL = ('HL', 'HR', 'FL', 'FR')


def _four_paw_contour_data():
    """Four mapped paws under the contour gate, with brightness, contour
    arrays, paw positions and the locomotion arrays a gait run would use."""
    masks, n = _four_paw_cycle(n_cycles=12)
    rng = np.random.RandomState(5)
    areas = rng.uniform(1000, 5500, n)
    band = pd.Series((areas > 1500) & (areas <= 5000))
    contact = {'HL': band, 'HR': band.copy(),
               'FL': masks['FL'], 'FR': masks['FR']}
    pcd = {r: {'areas': areas.copy(), 'spreads': areas / 50,
               'intensities': rng.uniform(40, 90, n),
               'widths': areas / 80, 'solidities': np.full(n, 0.8),
               'aspect_ratios': np.full(n, 1.3),
               'circularities': np.full(n, 0.5)} for r in ('HL', 'HR')}
    speed = rng.uniform(0, 60, n)
    lick = np.zeros(n, dtype=bool)
    lick[:40] = True               # part of the first 1-s bin
    t = np.arange(n, dtype=float)
    return {
        'session_name': 'four', 'fps': FPS, '_used_fallback_fps': False,
        'n_frames': n, 'height_df': pd.DataFrame(index=range(n)),
        'bp_xcord': None, 'bp_ycord': None, 'bp_prob': None,
        'active_paws': {r: r.lower() for r in ROLES_ALL},
        'contact_masks': contact,
        'paw_xy': {r: (t + k, np.full(n, 10.0 * k))
                   for k, r in enumerate(ROLES_ALL)},
        'brightness_series': {r: pd.Series(rng.uniform(50, 100, n))
                              for r in ROLES_ALL},
        'paw_contour_data': pcd,
        'confidence_mask': None,
        'loco_mask': speed > 20, 'body_speed': speed,
        'frame_displacements': speed / FPS,
        'lick_mask': lick,
        'params': {'bin_seconds': 1, 'bin_unit': 'seconds',
                   'contact_method': 'contour_area', 'min_stance_ms': 0,
                   'exclude_licking': True, 'gate_4paw': False},
        '_mm_per_px': None,
    }


def _run(data, **extra):
    params = {**data['params'], **extra}
    sel = compute_selection_masks(data['contact_masks'], data['lick_mask'],
                                  False, data['n_frames'])
    return compute_all_metrics(data, sel, params)


def test_default_run_emits_no_gait_or_movement_columns():
    out = _run(_four_paw_contour_data())
    assert out['bins'], 'expected time bins'
    for row in [out['summary']] + out['bins']:
        gait = [k for k in row if gait_core.is_gait_column(k)]
        assert not gait, gait
        for k in ('contour_pass_pct', 'brightness_HL', 'brightness_HR',
                  'brightness_ratio_HL_HR', 'paw_area_HL',
                  'contact_intensity_HR', 'paw_area_ratio_hind',
                  'contact_intensity_ratio_hind', 'pawlike_area_ratio_hind',
                  'paw_area_stance_HL', 'n_analyzed', 'analyzed_pct'):
            assert k in row, k
    # the same flag is honoured when it is passed explicitly
    explicit = _run(_four_paw_contour_data(), compute_gait=False)['summary']
    assert set(explicit) == set(out['summary'])


def test_compute_gait_true_still_emits_the_gait_columns():
    data = _four_paw_contour_data()
    default = _run(data)['summary']
    s = _run(data, compute_gait=True)['summary']
    for k in ('contact_pct_HL', 'contact_pct_HR', 'n_contact_HL',
              'contact_pct_FL', 'WBI_fore', 'SI_fore', 'SBI_fore',
              'WBI_HL', 'hind_fore_ratio', 'quad_stance_pct',
              'stance_dur_HL', 'swing_dur_HL', 'stride_dur_HL',
              'duty_cycle_HL', 'cadence_HL', 'stride_len_HL',
              'swing_speed_HL', 'stride_cv_HL', 'n_strides_HL',
              'step_len_hind', 'step_width_hind', 'stance_SI_hind',
              'stride_len_SI_hind', 'phase_HL_HR', 'phase_diagonal',
              'print_position_L', 'support_4paw_pct',
              'total_distance', 'loco_total_distance', 'time_moving_s',
              'time_moving_pct', 'body_speed_mean', 'body_speed_loco',
              'loco_stance_dur_HL', 'loco_step_len_hind'):
        assert k in s, k
        assert gait_core.is_gait_column(k), k
    # hind WBI/SI/SBI are reported (as NaN under the contour gate)
    assert np.isnan(s['WBI_hind']) and np.isnan(s['SBI_hind'])
    # the contour / brightness values do not depend on the flag
    for k, v in default.items():
        assert k in s, k
        assert (v == s[k]) or (np.isnan(v) and np.isnan(s[k])), k
    # contour_pass_pct is the old contact_pct_HL (== HR) under the gate
    assert s['contour_pass_pct'] == s['contact_pct_HL'] == s['contact_pct_HR']


def test_contour_pass_pct_counts_band_frames_over_analyzed():
    data = _four_paw_contour_data()
    s = _run(data)['summary']
    band = data['contact_masks']['HL'].values
    analyzed = ~data['lick_mask']
    want = round(float((band & analyzed).sum()) / analyzed.sum() * 100, 2)
    assert s['contour_pass_pct'] == pytest.approx(want)
    assert s['n_analyzed'] == int(analyzed.sum())
    # no contour gate -> no pass %
    h = _run(data, contact_method='height')['summary']
    assert 'contour_pass_pct' not in h


def test_gait_column_classification():
    kept = ['session', 'subject', 'treatment', 'n_frames', 'n_analyzed',
            'analyzed_pct', 'fps', 'fallback_fps_used', 'contour_pass_pct',
            'brightness_HL', 'brightness_ratio_HL_HR', 'paw_area_HL',
            'paw_area_stance_HR', 'contact_intensity_HL',
            'contact_intensity_stance_HR', 'pawlike_intensity_HL',
            'paw_solidity_HL', 'paw_area_ratio_hind',
            'contact_intensity_ratio_stance_hind',
            'pawlike_intensity_ratio_hind', 'bin_index', 'bin_start_s']
    assert not [c for c in kept if gait_core.is_gait_column(c)]
    df = pd.DataFrame(columns=kept + ['contact_pct_HL', 'WBI_hind',
                                      'loco_cadence_HR', 'regularity_index'])
    assert list(gait_core.drop_gait_columns(df).columns) == kept
    assert gait_core.drop_gait_columns(None) is None


def test_load_lick_mask_requires_stem_boundary(tmp_path):
    """..._S1 must not borrow ..._S10's predictions (stem-prefix glob)."""
    beh_dir = tmp_path / 'results' / 'licking'
    beh_dir.mkdir(parents=True)
    ones = pd.DataFrame({'licking': [1] * 10})
    ones.to_csv(beh_dir / 'X_S10_classifier_licking_predictions.csv',
                index=False)
    out = gait_core.load_lick_mask(str(tmp_path), 'X_S1', 'licking', 0.5, 10)
    assert not out.any()
    ones.to_csv(beh_dir / 'X_S1_classifier_licking_predictions.csv',
                index=False)
    out = gait_core.load_lick_mask(str(tmp_path), 'X_S1', 'licking', 0.5, 10)
    assert out.all()
    # the bare '{stem}_predictions.csv' form is accepted too
    ones.to_csv(beh_dir / 'X_S2_predictions.csv', index=False)
    out = gait_core.load_lick_mask(str(tmp_path), 'X_S2', 'licking', 0.5, 10)
    assert out.all()


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


def test_analysis_dir_new_vs_existing_project(tmp_path):
    """New projects get paw_contour/; a project that already has
    gait_limb_analysis/ keeps it (nothing is renamed)."""
    new_proj = tmp_path / "new"; new_proj.mkdir()
    assert gait_core.analysis_dir(str(new_proj)) == str(new_proj / "paw_contour")
    old_proj = tmp_path / "old"; (old_proj / "gait_limb_analysis").mkdir(parents=True)
    assert gait_core.analysis_dir(str(old_proj)) == str(old_proj / "gait_limb_analysis")
    both = tmp_path / "both"
    (both / "gait_limb_analysis").mkdir(parents=True); (both / "paw_contour").mkdir()
    assert gait_core.analysis_dir(str(both)) == str(both / "paw_contour")


def _run_gated(data, gate_4paw):
    params = {**data['params'], 'gate_4paw': gate_4paw}
    sel = compute_selection_masks(data['contact_masks'], data['lick_mask'],
                                  gate_4paw, data['n_frames'])
    return compute_all_metrics(data, sel, params), params


def test_contour_pass_pct_with_4paw_gate_is_not_stuck_at_100():
    """The 4-paw mask is built from the same band masks, so over analyzed
    frames the pass % always read 100 with the gate on. It is now the share
    of licking-excluded frames the contour metrics come from."""
    data = _four_paw_contour_data()
    out, _p = _run_gated(data, True)
    s = out['summary']
    band = data['contact_masks']['HL'].values
    four = (band & data['contact_masks']['FL'].values
            & data['contact_masks']['FR'].values)
    base = ~data['lick_mask']
    want = round(float((band & four & base).sum()) / base.sum() * 100, 2)
    assert s['contour_pass_pct'] == pytest.approx(want)
    assert s['contour_pass_pct'] < 100
    # gate off: unchanged (band frames over licking-excluded frames)
    off = _run_gated(data, False)[0]['summary']
    assert off['contour_pass_pct'] == pytest.approx(
        round(float((band & base).sum()) / base.sum() * 100, 2))


@pytest.mark.parametrize("gate_4paw", [False, True])
def test_filter_preview_apply_reproduces_the_run(gate_4paw):
    """Filter Preview's Apply with unchanged limits must give back the run's
    Filtered (pawlike_*) numbers. It used to average over the contour gate
    alone, putting licking frames back in (example mouse1_veh: Filtered
    area ratio 0.94 -> 1.29 on Apply)."""
    import types
    import gait_views
    data = _four_paw_contour_data()
    out, params = _run_gated(data, gate_4paw)
    data['params'] = params
    summ = pd.DataFrame([out['summary']]).assign(session='four')
    bins = pd.DataFrame(out['bins']).assign(session='four')
    cols = [c for c in summ.columns if c.startswith('pawlike_')]
    bcols = [c for c in bins.columns if c.startswith('pawlike_')]
    assert cols and bcols

    def apply(inter):
        host = types.SimpleNamespace(
            summary_df=summ.copy(), bins_df=bins.copy(),
            intermediates={'four': inter},
            pawlike_thresholds=dict(gait_core._default_pawlike()))
        gait_views.recompute_pawlike_metrics(host)
        return host

    host = apply(data)
    pd.testing.assert_frame_equal(host.summary_df[cols].astype(float),
                                  summ[cols].astype(float))
    pd.testing.assert_frame_equal(host.bins_df[bcols].astype(float),
                                  bins[bcols].astype(float))
    # the licking frames matter here, so the check above is not vacuous
    no_lick = apply({**data, 'lick_mask': np.zeros(data['n_frames'], bool)})
    assert not np.allclose(no_lick.summary_df['pawlike_area_HL'].astype(float),
                           summ['pawlike_area_HL'].astype(float))


def test_shape_ratios_circularity_and_solidity():
    """The two shape ratios the manuscript reports beside intensity and area:
    mean HL / mean HR over the gated frames, in the summary and in every bin,
    for all three contour variants."""
    data = _synthetic_data()
    n = data['n_frames']
    ones = pd.Series(np.ones(n, dtype=bool))
    data['contact_masks'] = {'HL': ones, 'HR': ones.copy()}
    data['lick_mask'] = np.zeros(n, dtype=bool)
    areas = np.full(n, 2000.0)
    shape = {'HL': (0.60, 0.90), 'HR': (0.50, 0.75)}      # (circularity, solidity)
    for role, (circ, sol) in shape.items():
        data['paw_contour_data'][role] = {
            'areas': areas.copy(), 'spreads': areas / 50, 'intensities': areas / 20,
            'widths': areas / 80, 'solidities': np.full(n, sol),
            'aspect_ratios': np.full(n, 1.2), 'circularities': np.full(n, circ),
        }
    sel = compute_selection_masks(data['contact_masks'], data['lick_mask'], False, n)
    out = compute_all_metrics(data, sel, data['params'])
    assert out['bins'], 'expected time bins'
    for row in [out['summary']] + out['bins']:
        for stem in ('paw_{}_ratio_hind', 'paw_{}_ratio_stance_hind', 'pawlike_{}_ratio_hind'):
            assert row[stem.format('circularity')] == pytest.approx(0.60 / 0.50, abs=1e-3), stem
            assert row[stem.format('solidity')] == pytest.approx(0.90 / 0.75, abs=1e-3), stem
        assert not [k for k in row if 'ratio' in k and gait_core.is_gait_column(k)]

    # no HR contour -> no shape ratio, and nothing raises
    del data['paw_contour_data']['HR']
    lone = compute_all_metrics(data, sel, data['params'])['summary']
    assert 'paw_circularity_ratio_hind' not in lone
    assert 'paw_solidity_ratio_hind' not in lone
