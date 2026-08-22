# -*- coding: utf-8 -*-
"""Golden equality: gait_core vs the old GaitLimbTab on E:/PixelPaws_practice.

The golden CSVs were captured from the OLD tab by
scripts/capture_gait_golden.py (manuscript preset, 30-s bins).  These tests
run the NEW headless gait_core on the same sessions/params and assert frame
equality - plus, in the slow test, cache re-extraction equality.

Run:  PYTHONIOENCODING=utf-8 python -m pytest tests/test_gait_golden.py -v
Skip the slow re-extraction test with:  -m "not slow"
"""
import os
import sys
import json
import glob
import shutil

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

PROJ = r"E:/PixelPaws_practice"
GOLDEN_DIR = os.path.join(REPO, "tests", "golden")
CFG_PATH = os.path.join(GOLDEN_DIR, "gait_capture_config.json")

pytestmark = pytest.mark.skipif(
    not (os.path.isdir(PROJ) and os.path.isfile(CFG_PATH)),
    reason="practice project or golden capture not present")

import gait_core
from gait_core import GaitContext


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_cfg():
    with open(CFG_PATH) as f:
        return json.load(f)


def _sessions_in_capture_order(cfg):
    sessions = gait_core.find_session_triplets(PROJ, require_labels=False)
    by_name = {s['session_name']: s for s in sessions}
    missing = [n for n in cfg['sessions'] if n not in by_name]
    assert not missing, f"sessions from capture no longer found: {missing}"
    return [by_name[n] for n in cfg['sessions']]


def _ctx(cfg):
    return GaitContext(project_folder=PROJ,
                       pawlike_thresholds=dict(cfg['pawlike_thresholds']))


def _roundtrip(df, tmp_path, name):
    """Normalize float formatting the same way the golden CSVs were written."""
    p = os.path.join(str(tmp_path), name)
    df.to_csv(p, index=False)
    return pd.read_csv(p)


def assert_frames_equal(new_df, golden_df, sort_keys, label):
    assert set(new_df.columns) == set(golden_df.columns), (
        f"{label}: column sets differ; "
        f"only-new={sorted(set(new_df.columns) - set(golden_df.columns))}, "
        f"only-golden={sorted(set(golden_df.columns) - set(new_df.columns))}")
    new_df = (new_df[golden_df.columns.tolist()]
              .sort_values(sort_keys).reset_index(drop=True))
    golden_df = golden_df.sort_values(sort_keys).reset_index(drop=True)
    assert len(new_df) == len(golden_df), (
        f"{label}: row counts differ ({len(new_df)} vs {len(golden_df)})")
    for col in golden_df.columns:
        g, n = golden_df[col], new_df[col]
        if pd.api.types.is_numeric_dtype(g) and pd.api.types.is_numeric_dtype(n):
            ok = np.allclose(n.astype(float).values, g.astype(float).values,
                             rtol=1e-9, atol=0, equal_nan=True)
        else:
            ok = (g.fillna('').astype(str) == n.fillna('').astype(str)).all()
        assert ok, (f"{label}: column '{col}' differs\n"
                    f"golden: {g.tolist()}\nnew:    {n.tolist()}")


def _run_core(cfg, sessions=None, cancel=None):
    ctx = _ctx(cfg)
    key_df = gait_core.load_key_file(cfg['key_file'])
    return gait_core.run_sessions(
        sessions if sessions is not None else _sessions_in_capture_order(cfg),
        cfg['paw_map'], cfg['params'], key_df, ctx)


# ─────────────────────────────────────────────────────────────────────────────
# Fast: warm-cache equality on all 6 sessions
# ─────────────────────────────────────────────────────────────────────────────

def test_golden_equality_warm_caches(tmp_path):
    cfg = _load_cfg()
    summary_rows, bin_rows, not_done = _run_core(cfg)
    assert not_done == cfg.get('not_done', []), not_done
    assert len(summary_rows) == len(cfg['sessions'])

    new_sum = _roundtrip(pd.DataFrame(summary_rows), tmp_path, "s.csv")
    new_bin = _roundtrip(pd.DataFrame(bin_rows), tmp_path, "b.csv")
    gold_sum = pd.read_csv(os.path.join(GOLDEN_DIR, "gait_summary.csv"))
    gold_bin = pd.read_csv(os.path.join(GOLDEN_DIR, "gait_bins.csv"))

    assert_frames_equal(new_sum, gold_sum, ['session'], "summary")
    assert_frames_equal(new_bin, gold_bin, ['session', 'bin_index'], "bins")


def test_recompute_with_contact_matches_analyze(tmp_path):
    """The Adjust-Contact path (rebuild masks → same compute_all_metrics) must
    reproduce the analyze_session results when given the same contact params."""
    cfg = _load_cfg()
    sess = _sessions_in_capture_order(cfg)[0]
    ctx = _ctx(cfg)
    result = gait_core.analyze_session(sess, cfg['paw_map'], cfg['params'], ctx)
    assert result is not None
    data = result['intermediates']

    redo = gait_core.recompute_with_contact(
        data,
        {'contact_method': cfg['params']['contact_method'],
         'contact_threshold': cfg['params']['contact_threshold'],
         'speed_threshold': cfg['params']['speed_threshold'],
         'median_filter_ms': cfg['params']['median_filter_ms'],
         'min_bout_ms': cfg['params']['min_bout_ms'],
         'brt_weight': cfg['params']['brt_weight']},
        ctx)

    a = _roundtrip(pd.DataFrame([result['summary']]), tmp_path, "a.csv")
    b = _roundtrip(pd.DataFrame([redo['summary']]), tmp_path, "b.csv")
    a['session'] = b['session'] = sess['session_name']
    assert_frames_equal(b, a, ['session'], "recompute summary")

    a_bins = pd.DataFrame(result['bins'])
    b_bins = pd.DataFrame(redo['bins'])
    a_bins['session'] = b_bins['session'] = sess['session_name']
    assert_frames_equal(_roundtrip(b_bins, tmp_path, "bb.csv"),
                        _roundtrip(a_bins, tmp_path, "ab.csv"),
                        ['session', 'bin_index'], "recompute bins")


def test_analyze_session_intermediates_contract():
    """The graph window / Adjust-Contact consumers rely on these keys."""
    cfg = _load_cfg()
    sess = _sessions_in_capture_order(cfg)[0]
    result = gait_core.analyze_session(sess, cfg['paw_map'], cfg['params'],
                                       _ctx(cfg))
    assert result is not None
    inter = result['intermediates']
    for key in ('height_df', 'bp_xcord', 'bp_ycord', 'bp_prob', 'fps',
                '_used_fallback_fps', 'n_frames', 'active_paws',
                'contact_masks', 'paw_xy', 'brightness_series',
                'paw_contour_data', 'confidence_mask', 'loco_mask',
                'body_speed', 'frame_displacements', 'lick_mask', 'params'):
        assert key in inter, key
    # contour graph tabs need shapes + extraction-time solidity lists
    for role, d in inter['paw_contour_data'].items():
        assert 'areas' in d and 'contour_shapes' in d, role


# ─────────────────────────────────────────────────────────────────────────────
# Slow: cache re-extraction equality (one session; decodes the whole video)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_extraction_regenerates_identical_caches(tmp_path):
    cfg = _load_cfg()
    sess = _sessions_in_capture_order(cfg)[0]
    name = sess['session_name']
    cache_dir = os.path.join(PROJ, 'gait_limb_analysis')

    cache_files = sorted(
        glob.glob(os.path.join(cache_dir, f'{name}_brt_*')) +
        glob.glob(os.path.join(cache_dir, f'{name}_contour_*')))
    cache_files = [p for p in cache_files if not p.endswith('.bak')]
    assert cache_files, f"no warm caches for {name} - run the capture first"

    backups = {}
    try:
        for p in cache_files:
            bak = p + '.bak'
            shutil.move(p, bak)
            backups[p] = bak

        # Re-extract from video via the new core
        result = gait_core.analyze_session(sess, cfg['paw_map'],
                                           cfg['params'], _ctx(cfg))
        assert result is not None

        # 1) regenerated caches exist under the SAME names (same md5 keys)
        for p in backups:
            assert os.path.isfile(p), f"cache not regenerated: {p}"

        # 2) regenerated cache contents match the old tab's caches
        for p, bak in backups.items():
            if p.endswith('.csv'):
                old_df, new_df = pd.read_csv(bak), pd.read_csv(p)
                assert list(old_df.columns) == list(new_df.columns), p
                for col in old_df.columns:
                    assert np.allclose(new_df[col].values.astype(float),
                                       old_df[col].values.astype(float),
                                       rtol=1e-9, atol=0, equal_nan=True), \
                        (p, col)
            elif p.endswith('.json'):
                with open(bak) as f1, open(p) as f2:
                    assert json.load(f1) == json.load(f2), p
            elif p.endswith('.npz'):
                # context managers: np.load keeps the handle open, which
                # blocks the file swap in the finally block on Windows
                with np.load(bak) as old_z, np.load(p) as new_z:
                    assert sorted(old_z.files) == sorted(new_z.files), p
                    for k in old_z.files:
                        assert np.allclose(new_z[k], old_z[k],
                                           rtol=1e-9, atol=0), (p, k)

        # 3) freshly-extracted metrics still match golden for this session
        summary_rows, bin_rows, not_done = _run_core(cfg, sessions=[sess])
        assert not not_done
        gold_sum = pd.read_csv(os.path.join(GOLDEN_DIR, "gait_summary.csv"))
        gold_bin = pd.read_csv(os.path.join(GOLDEN_DIR, "gait_bins.csv"))
        assert_frames_equal(
            _roundtrip(pd.DataFrame(summary_rows), tmp_path, "s.csv"),
            gold_sum[gold_sum['session'] == name].reset_index(drop=True),
            ['session'], "re-extracted summary")
        assert_frames_equal(
            _roundtrip(pd.DataFrame(bin_rows), tmp_path, "b.csv"),
            gold_bin[gold_bin['session'] == name].reset_index(drop=True),
            ['session', 'bin_index'], "re-extracted bins")
    finally:
        # Restore the original (old-tab-written) caches byte-identically.
        for p, bak in backups.items():
            if os.path.isfile(bak):
                if os.path.isfile(p):
                    os.remove(p)
                shutil.move(bak, p)
