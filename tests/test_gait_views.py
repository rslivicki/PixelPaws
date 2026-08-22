# -*- coding: utf-8 -*-
"""Headless tests for gait_views.py (run with PYTHONIOENCODING=utf-8
python -m pytest tests/test_gait_views.py). Pattern per test_contract_shim:
Agg backend + withdrawn Tk root."""
import os
import sys

sys.path.insert(0, r"E:/Code/PixelPaws")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import tkinter as tk
from tkinter import ttk

import gait_views as gv

RNG = np.random.default_rng(7)
ROLES = ('HL', 'HR', 'FL', 'FR')


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data
# ─────────────────────────────────────────────────────────────────────────────

SUMMARY_COLS = (
    ['WBI_hind', 'SI_hind', 'SBI_hind', 'WBI_fore', 'SI_fore', 'SBI_fore',
     'brightness_ratio_HL_HR',
     'stance_SI_hind', 'stride_len_SI_hind',
     'step_len_hind', 'step_width_hind', 'step_len_fore', 'step_width_fore',
     'phase_HL_HR', 'phase_diagonal', 'phase_FL_FR', 'phase_HL_FL',
     'phase_HR_FR', 'phase_HL_FR',
     'total_distance', 'loco_total_distance', 'time_moving_s',
     'time_moving_pct', 'body_speed_mean', 'body_speed_loco',
     'regularity_index', 'print_position_L', 'print_position_R',
     'support_0paw_pct', 'support_1paw_pct', 'support_2paw_pct',
     'support_3paw_pct', 'support_4paw_pct',
     'paw_area_ratio_hind', 'contact_intensity_ratio_hind',
     'pawlike_area_ratio_hind', 'pawlike_intensity_ratio_hind']
    + [f'{m}_{r}' for r in ROLES for m in
       ('contact_pct', 'stance_dur', 'swing_dur', 'duty_cycle',
        'swing_speed', 'stride_len', 'stride_cv',
        'paw_area', 'paw_spread', 'contact_intensity', 'paw_width',
        'paw_solidity', 'paw_aspect_ratio', 'paw_circularity',
        'pawlike_area', 'pawlike_spread', 'pawlike_intensity',
        'pawlike_width', 'pawlike_solidity', 'pawlike_aspect_ratio',
        'pawlike_circularity')]
)

BIN_COLS = (
    ['WBI_hind', 'SI_hind', 'SBI_hind', 'WBI_fore', 'SI_fore', 'SBI_fore',
     'hind_fore_ratio',
     'brightness_HL', 'brightness_HR', 'brightness_FL', 'brightness_FR',
     'brightness_ratio_HL_HR',
     'stance_SI_hind', 'stride_len_SI_hind',
     'total_distance', 'loco_total_distance', 'time_moving_s',
     'time_moving_pct', 'body_speed_mean', 'body_speed_loco',
     'paw_area_ratio_hind', 'contact_intensity_ratio_hind',
     'pawlike_area_ratio_hind', 'pawlike_intensity_ratio_hind']
    + [f'{m}_{r}' for r in ROLES for m in
       ('contact_pct', 'stance_dur', 'duty_cycle', 'cadence',
        'swing_speed', 'stride_len',
        'paw_area', 'pawlike_area')]
)


def _mk_summary():
    n = 8
    rows = {
        'session': [f'S{i+1}' for i in range(n)],
        'subject': [f'subj{i+1}' for i in range(n)],
        'treatment': ['Vehicle'] * 4 + ['Drug'] * 4,
    }
    for c in SUMMARY_COLS:
        base = 1.0 if 'ratio' in c else 50.0
        rows[c] = np.round(base + RNG.normal(0, base * 0.1, n), 4)
    return pd.DataFrame(rows)


def _mk_bins(summary):
    recs = []
    for _, r in summary.iterrows():
        for b in range(3):
            rec = {'session': r['session'], 'subject': r['subject'],
                   'treatment': r['treatment'],
                   'bin_start_s': b * 300.0, 'bin_end_s': (b + 1) * 300.0}
            for c in BIN_COLS:
                base = 1.0 if 'ratio' in c else 50.0
                rec[c] = round(base + RNG.normal(0, base * 0.1), 4)
            recs.append(rec)
    return pd.DataFrame(recs)


def _mk_shapes(k=24):
    """k noisy unit circles resampled to (64, 2)."""
    shapes = []
    theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    for _ in range(k):
        rad = 1.0 + RNG.normal(0, 0.05, 64)
        shapes.append(np.column_stack([rad * np.cos(theta),
                                       rad * np.sin(theta)]))
    return shapes


def _mk_intermediates(sessions, with_frame_arrays=False):
    inter = {}
    n_frames = 900
    for s in sessions:
        pcd = {}
        for role in ('HL', 'HR'):
            d = {
                'contour_shapes': _mk_shapes(),
                'contour_solidities': list(RNG.uniform(0.6, 0.95, 24)),
            }
            if with_frame_arrays:
                d.update({
                    'areas': RNG.uniform(0, 3000, n_frames),
                    'spreads': RNG.uniform(10, 80, n_frames),
                    'intensities': RNG.uniform(0, 255, n_frames),
                    'widths': RNG.uniform(5, 40, n_frames),
                    'solidities': RNG.uniform(0.5, 1.0, n_frames),
                    'aspect_ratios': RNG.uniform(1.0, 3.0, n_frames),
                    'circularities': RNG.uniform(0.05, 0.9, n_frames),
                })
            pcd[role] = d
        inter[s] = {
            'paw_contour_data': pcd,
            'fps': 1,
            'n_frames': n_frames,
            'contact_masks': {r: RNG.random(n_frames) > 0.4
                              for r in ('HL', 'HR')},
        }
    return inter


def _mk_cfg():
    return {
        'colors': {'Vehicle': '#888888', 'Drug': '#cc3311'},
        'order': ['Vehicle', 'Drug'],
        'time_window': None,
        'error_type': 'SEM',
        'error_display': 'circles_caps',
        'rebin_minutes': 0,
        'opacities': {}, 'marker_sizes': {}, 'marker_shapes': {},
        'marker_fills': {}, 'marker_edge_colors': {},
        'line_widths': {}, 'line_styles': {},
        'marker_size': 5, 'marker_shape': 'o',
        'show_individual': False,
        'show_stats': True,
        'sig_style': 'asterisk',
        'graph_sets': {},
        'stats_paradigm': 'parametric',
        'stats_alpha': 0.05,
        'stats_test': 'auto',
        'timecourse_posthoc': False,
    }


def _mk_host(injured='HL', intermediates=None, with_frame_arrays=False):
    summary = _mk_summary()
    bins_df = _mk_bins(summary)
    if intermediates is None:
        intermediates = _mk_intermediates(summary['session'][:2],
                                          with_frame_arrays=with_frame_arrays)
    return gv.ViewHost(
        summary_df=summary,
        bins_df=bins_df,
        intermediates=intermediates,
        cfg=_mk_cfg(),
        injured_paw=injured,
        enable_stats=lambda: True,
        log=lambda m: None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def root():
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip('no display / Tk unavailable')
    r.withdraw()
    yield r
    r.destroy()


@pytest.fixture(scope='module')
def host():
    return _mk_host()


@pytest.fixture(scope='module')
def registry(host):
    return gv.build_registry(host)


def _widgets(w):
    yield w
    for c in w.winfo_children():
        yield from _widgets(c)


def _has_figure(frame):
    return any(getattr(w, 'figure', None) is not None for w in _widgets(frame))


def _has_label(frame):
    return any(isinstance(w, (ttk.Label, tk.Label)) for w in _widgets(frame))


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_registry_categories(registry):
    cats = list(registry.keys())
    for expected in ['Paw Contact', 'Limb Use — Hind', 'Limb Use — Fore',
                     'Contact % / Brightness', 'Gait — Timing',
                     'Gait — Spatial', 'Gait — Symmetry', 'Movement',
                     'Coordination', 'Paw Contour — Hind Left',
                     'Paw Contour — Ratios',
                     'Paw Contour — Filter Preview',
                     'Paw Contour — Filtered — Hind Left',
                     'Paw Contour — Filtered — Ratios',
                     'Statistics']:
        assert expected in cats, f'missing category {expected}: {cats}'
    # Full Stance stays hidden unless graph_sets enables it (old default)
    assert not any(c.startswith('Paw Contour — Full Stance') for c in cats)
    # entry contract
    for cat, entries in registry.items():
        assert entries, f'empty category {cat}'
        for e in entries:
            assert e['display_name']
            assert 'description' in e
            assert callable(e['build'])
            assert isinstance(e['has_stats'], bool)


def test_every_entry_builds(root, host, registry):
    failures = []
    frame = ttk.Frame(root)
    for cat, entries in registry.items():
        for e in entries:
            try:
                gv.render_entry(host, frame, e)
                root.update_idletasks()
                ok = _has_figure(frame) or _has_label(frame)
                if not ok:
                    failures.append((cat, e['display_name'], 'no fig/label'))
            except Exception as exc:  # noqa: BLE001
                failures.append((cat, e['display_name'], repr(exc)))
    # final cleanup
    gv.render_entry(host, frame, {'build': lambda f: None})
    assert not failures, failures


def test_graceful_degradation_without_intermediates(root):
    h = _mk_host(intermediates={})
    reg = gv.build_registry(h)
    shape_entry = next(e for e in reg['Paw Contour — Hind Left']
                       if e['display_name'] == 'Shape')
    frame = ttk.Frame(root)
    gv.render_entry(h, frame, shape_entry)
    assert _has_label(frame) and not _has_figure(frame)
    # filter preview also degrades
    fp_entry = reg['Paw Contour — Filter Preview'][0]
    gv.render_entry(h, frame, fp_entry)
    assert _has_label(frame) and not _has_figure(frame)


def test_stats_mode(root, host, registry):
    # a data-backed entry gets a real table
    e = next(x for x in registry['Limb Use — Hind']
             if x['display_name'] == 'WBI Hind')
    assert e['has_stats']
    frame = ttk.Frame(root)
    gv.render_entry(host, frame, e, stats_mode=True)
    txts = [w for w in _widgets(frame) if isinstance(w, tk.Text)]
    assert txts
    content = txts[0].get('1.0', 'end')
    assert 'WBI Hind' in content
    assert 'Vehicle' in content and 'Drug' in content
    # custom entry shows the informative fallback in stats mode
    ce = next(x for x in registry['Contact % / Brightness']
              if x['display_name'] == 'Contact % (All Paws)')
    assert not ce['has_stats']
    gv.render_entry(host, frame, ce, stats_mode=True)
    txts = [w for w in _widgets(frame) if isinstance(w, tk.Text)]
    assert 'statistics unavailable here' in txts[0].get('1.0', 'end')


def test_injured_flip_hr_inverts_display_without_mutation():
    h = _mk_host(injured='HR')
    before = h.summary_df.copy(deep=True)
    reg = gv.build_registry(h)
    e = next(x for x in reg['Paw Contact']
             if x['display_name'] == 'Paw Area Ratio')
    assert e['flip'] and 'HR/HL' in e['y_label']
    disp, col = gv._entry_data(h, e)
    orig = h.summary_df['paw_area_ratio_hind']
    np.testing.assert_allclose(disp[col].values, 1.0 / orig.values)
    # host frames untouched: values identical, no leaked helper columns
    pd.testing.assert_frame_equal(h.summary_df, before)
    assert not any(c.endswith('_injflip') for c in h.summary_df.columns)
    assert not any(c.endswith('_injflip') for c in h.bins_df.columns)
    # HL host displays the stored value unchanged
    h2 = _mk_host(injured='HL')
    reg2 = gv.build_registry(h2)
    e2 = next(x for x in reg2['Paw Contact']
              if x['display_name'] == 'Paw Area Ratio')
    disp2, col2 = gv._entry_data(h2, e2)
    assert disp2 is h2.summary_df
    assert 'HL/HR' in e2['y_label']


def test_export_entry_csv(tmp_path, host, registry):
    bar = next(x for x in registry['Limb Use — Hind']
               if x['display_name'] == 'WBI Hind')
    p1 = tmp_path / 'bar.csv'
    assert gv.export_entry_csv(host, bar, str(p1)) == str(p1)
    out = pd.read_csv(p1)
    assert list(out.columns) == ['treatment', 'WBI_hind']
    assert len(out) == 8
    tc = next(x for x in registry['Limb Use — Hind']
              if x['display_name'] == 'WBI Hind — TC')
    p2 = tmp_path / 'tc.csv'
    assert gv.export_entry_csv(host, tc, str(p2)) == str(p2)
    out2 = pd.read_csv(p2)
    assert 'bin_start_min' in out2.columns and 'bin_start_s' in out2.columns
    # custom entries have no tabular data behind them
    ce = next(x for x in registry['Coordination']
              if x['display_name'] == 'Support Patterns')
    assert gv.export_entry_csv(host, ce, str(tmp_path / 'x.csv')) is None
    # flipped export carries displayed values under the original column name
    h = _mk_host(injured='HR')
    reg = gv.build_registry(h)
    fe = next(x for x in reg['Paw Contact']
              if x['display_name'] == 'Paw Area Ratio')
    p3 = tmp_path / 'flip.csv'
    gv.export_entry_csv(h, fe, str(p3))
    out3 = pd.read_csv(p3)
    assert 'paw_area_ratio_hind' in out3.columns
    assert not any(c.endswith('_injflip') for c in out3.columns)


def test_wb_statistics(tmp_path, host):
    stats_data = {'summary_df': host.summary_df, 'bins_df': host.bins_df,
                  'treatments': ['Vehicle', 'Drug'],
                  'metrics': ['WBI_hind', 'SI_hind']}
    frame_df = gv.wb_statistics_frame(host, stats_data)
    assert (frame_df['Section'] == 'Summary — WBI_hind').any()
    assert (frame_df['Section'].str.startswith('Test — ')).any()
    p = tmp_path / 'stats.csv'
    assert gv.export_wb_statistics(host, stats_data, path=str(p)) == str(p)
    assert p.exists()


def test_recompute_pawlike_metrics_updates_host():
    h = _mk_host(with_frame_arrays=True)
    s0 = h.summary_df['session'].iloc[0]
    h.summary_df.loc[h.summary_df['session'] == s0, 'pawlike_area_HL'] = np.nan
    gv.recompute_pawlike_metrics(h)
    val = h.summary_df.loc[h.summary_df['session'] == s0,
                           'pawlike_area_HL'].iloc[0]
    assert np.isfinite(val)
    ratio = h.summary_df.loc[h.summary_df['session'] == s0,
                             'pawlike_area_ratio_hind'].iloc[0]
    aHL = h.summary_df.loc[h.summary_df['session'] == s0, 'pawlike_area_HL'].iloc[0]
    aHR = h.summary_df.loc[h.summary_df['session'] == s0, 'pawlike_area_HR'].iloc[0]
    assert ratio == pytest.approx(round(aHL / aHR, 4))
    # bins updated too
    b0 = h.bins_df[h.bins_df['session'] == s0].iloc[0]
    assert np.isfinite(b0['pawlike_area_HL'])


def test_pawlike_change_hook_via_apply(root):
    """The Apply flow: thresholds stored on host + hook fired."""
    h = _mk_host(with_frame_arrays=True)
    fired = []
    h.on_pawlike_change = lambda: fired.append(True)
    reg = gv.build_registry(h)
    fp_entry = reg['Paw Contour — Filter Preview'][0]
    frame = ttk.Frame(root)
    gv.render_entry(h, frame, fp_entry)
    root.update_idletasks()
    # find the Apply button and invoke it
    btns = [w for w in _widgets(frame)
            if isinstance(w, ttk.Button) and w.cget('text') == 'Apply']
    assert btns, 'Apply button not found in filter preview'
    btns[0].invoke()
    assert fired, 'on_pawlike_change did not fire'


def test_render_entry_closes_previous_figures(root, host, registry):
    frame = ttk.Frame(root)
    e = next(x for x in registry['Limb Use — Hind']
             if x['display_name'] == 'WBI Hind')
    gv.render_entry(host, frame, e)
    n_open = len(plt.get_fignums())
    for _ in range(3):
        gv.render_entry(host, frame, e)
    assert len(plt.get_fignums()) <= n_open, 'figures leaked across renders'
