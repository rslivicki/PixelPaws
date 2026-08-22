# -*- coding: utf-8 -*-
"""
Integration test for the rebuilt Gait & Limb tab (gait_tab_v2).

Drives the real widget headlessly against E:/PixelPaws_practice: scan →
readiness (video_path bug regression) → analysis via gait_core → golden
equality → registry render (gait_views) → Adjust-Contact recompute
reproducibility → session-bundle roundtrip.

Skips cleanly when the practice project or goldens are absent.
"""

import os
import sys
import threading

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

PRACTICE = r"E:/PixelPaws_practice"
GOLDEN = os.path.join(REPO, "tests", "golden", "gait_summary.csv")

pytestmark = pytest.mark.skipif(
    not (os.path.isdir(PRACTICE) and os.path.isfile(GOLDEN)),
    reason="practice project or gait goldens not available")


class _FakeThread:
    """Captures a Thread target so the test can run it synchronously."""
    captured = {}

    def __init__(self, target=None, args=(), daemon=None):
        _FakeThread.captured["target"] = target
        _FakeThread.captured["args"] = args

    def start(self):
        pass

    def is_alive(self):
        return False


@pytest.fixture(scope="module")
def tab():
    import matplotlib
    matplotlib.use("Agg")
    import tkinter as tk
    from tkinter import ttk
    from gait_tab_v2 import GaitLimbTabV2

    root = tk.Tk()
    root.withdraw()

    class StubApp:
        notebook = None
        key_file_data = {}

    StubApp.root = root
    StubApp.current_project_folder = tk.StringVar(value=PRACTICE)

    t = GaitLimbTabV2(ttk.Notebook(root), StubApp())
    t._test_root = root
    yield t
    # Teardown hygiene so later Tk-based test modules start clean.
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass
    try:
        root.destroy()
    except Exception:
        pass
    tk._default_root = None
    # ttkbootstrap keeps a Style singleton bound to the first root; clear it
    # so later Tk-based test modules can build themed widgets.
    try:
        from ttkbootstrap.style import Style as _BsStyle
        from ttkbootstrap.publisher import Publisher as _BsPub
        _BsStyle.instance = None
        _BsPub.clear_subscribers()
    except Exception:
        pass


def _run_sync(tab, launcher):
    """Run a thread-spawning tab method with the thread made synchronous."""
    _FakeThread.captured.clear()
    orig = threading.Thread
    threading.Thread = _FakeThread
    try:
        launcher()
    finally:
        threading.Thread = orig
    assert "target" in _FakeThread.captured, "no worker launched"
    _FakeThread.captured["target"](*_FakeThread.captured.get("args", ()))
    tab._test_root.update()


def test_scan_readiness_and_golden_run(tab):
    root = tab._test_root
    assert tab._preset_var.get() == "Paw contact (manuscript gate)"
    assert tab._contact_method_var.get() == "contour_area"

    tab.on_project_changed()
    assert len(tab._sessions) == 6
    assert tab._key_df is not None and len(tab._key_df) == 6
    assert tab._exclude_lick_var.get() and tab._lick_behavior_var.get()

    tab._select_all()
    ready, issues, notes = tab._check_readiness()
    assert ready, issues
    # regression: the old readiness check read s['video_path'] (never set)
    # and warned that every session lacked video whenever brightness was on
    assert not any("lack video" in x for x in notes), notes

    # match the golden capture's binning (5-min preset bins yield 0 bins
    # on the 2-minute practice clips)
    tab._bin_seconds_var.set(30)
    tab._bin_unit_var.set("seconds")
    _run_sync(tab, tab._start_analysis)

    assert tab._summary_df is not None and len(tab._summary_df) == 6
    assert len(tab._bins_df) == 24
    assert len(tab._session_intermediates) == 6
    assert len(tab._res_tree.get_children()) == 6

    golden = pd.read_csv(GOLDEN).sort_values("session").reset_index(drop=True)
    new = tab._summary_df.sort_values("session").reset_index(drop=True)
    missing = set(golden.columns) - set(new.columns)
    assert not missing, missing
    for c in golden.columns:
        a, b = golden[c], new[c]
        if a.dtype.kind in "fc":
            assert np.allclose(a, b.astype(float), rtol=1e-9,
                               equal_nan=True), c
        else:
            assert (a.fillna("") == b.fillna("")).all(), c


def test_registry_renders(tab):
    root = tab._test_root
    cats = list(tab._registry.keys())
    assert cats, "registry not populated by _on_analysis_complete"
    assert sum(len(v) for v in tab._registry.values()) > 50
    for cat in ("Paw Contact", "Limb Use — Hind", "Contact % / Brightness",
                "Statistics"):
        if cat not in tab._registry:
            continue
        tab._cat_var.set(cat)
        tab._on_category_changed()
        root.update()
        assert tab._graph_container.winfo_children(), cat

    # Σ stats flip on a stats-capable entry
    tab._cat_var.set("Limb Use — Hind")
    tab._on_category_changed()
    entry = tab._current_entry()
    if entry and entry.get("has_stats"):
        tab._stats_mode_var.set(True)
        tab._render_current()
        root.update()
        tab._stats_mode_var.set(False)
        tab._render_current()


def test_recompute_same_params_reproduces(tab):
    before = (tab._summary_df.sort_values("session")
              .reset_index(drop=True)["contact_pct_HL"].copy())
    _run_sync(tab, lambda: tab._recompute_contact({
        "contact_method": "contour_area",
        "contact_threshold": tab._contact_thresh_var.get(),
        "speed_threshold": "auto",
        "median_filter_ms": tab._median_filter_var.get(),
        "min_bout_ms": tab._min_bout_var.get(),
        "brt_weight": tab._brt_weight_var.get(),
    }))
    log = tab._log_text.get("1.0", "end")
    assert "Recompute done: 6" in log, log[-500:]
    after = (tab._summary_df.sort_values("session")
             .reset_index(drop=True)["contact_pct_HL"])
    assert np.allclose(before.values, after.values)


def test_bundle_roundtrip(tab):
    tab._save_session_file("named", "pytest-roundtrip")
    tab._summary_df = None
    tab._bins_df = None
    tab._refresh_saved_sessions()
    vals = list(tab._saved_combo["values"])
    idx = [i for i, v in enumerate(vals) if "pytest-roundtrip" in v]
    assert idx, vals
    tab._saved_combo.current(idx[0])
    tab._load_selected_session()
    assert tab._summary_df is not None and len(tab._summary_df) == 6
    # cleanup the named bundle
    tab._refresh_saved_sessions()
    for label, path in tab._saved_items:
        if "pytest-roundtrip" in label:
            os.remove(path)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-p", "no:dash"]))
