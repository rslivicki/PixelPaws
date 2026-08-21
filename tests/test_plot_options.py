# -*- coding: utf-8 -*-
"""Headless tests for the shared plot-options facility (plot_style) and its
Locomotion pilot wiring. Run directly: python tests/test_plot_options.py"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib.figure import Figure

import plot_style as ps


def test_roundtrip():
    d = tempfile.mkdtemp()
    cfg = os.path.join(d, "PixelPaws_project.json")
    json.dump({"group_colors": {"A": "#ff0000"}, "mystery": 1}, open(cfg, "w"))
    ps.save_options(d, {"error_type": "SD", "line_styles": {"A": "dashed"},
                        "grid": True, "not_a_key": 5})
    o = ps.get_options(d)
    assert o["error_type"] == "SD" and o["grid"] is True
    assert o["line_styles"] == {"A": "dashed"}
    assert o["error_display"] == "band"          # untouched default
    data = json.load(open(cfg))
    assert data["group_colors"] == {"A": "#ff0000"}
    assert data["mystery"] == 1
    assert "not_a_key" not in data["plot_options"]
    ps.reset_options(d)
    assert ps.get_options(d) == ps.DEFAULT_OPTIONS
    # corrupt file -> defaults, no raise
    open(cfg, "w").write("{not json")
    assert ps.get_options(d) == ps.DEFAULT_OPTIONS
    print("roundtrip OK")


def test_calc_error():
    v = [1, 2, 3, 4]
    assert abs(ps.calc_error(v, "SD") - 1.29099) < 1e-4
    assert abs(ps.calc_error(v, "SEM") - 0.64550) < 1e-4
    assert abs(ps.calc_error(v, "95CI") - 2.05426) < 1e-4
    assert ps.calc_error([5.0], "SEM") == 0.0
    assert abs(ps.calc_error([1, np.nan, 3], "SD")
               - np.std([1, 3], ddof=1)) < 1e-9
    assert ps.error_label("SD").endswith("SD")
    print("calc_error OK")


def test_draw_series():
    x = np.arange(5)
    rows = np.array([[1, 2, 3, 2, 1], [2, 3, 4, 3, 2], [1.5, 2, 3, 2, 1.5]])
    for disp in ("band", "caps", "none"):
        for spag in (False, True):
            for mark in (False, True):
                fig = Figure()
                ax = fig.add_subplot(111)
                o = dict(ps.DEFAULT_OPTIONS, error_display=disp,
                         show_individual=spag, show_markers=mark,
                         line_styles={"G": "dashed"})
                ps.draw_series(ax, x, rows, "#123456", o, label="G", group="G")
                nlines = len(ax.get_lines())
                expect = 1 + (3 if spag else 0)
                if disp == "caps":
                    assert len(ax.containers) or ax.collections or nlines >= expect
                assert nlines >= expect, (disp, spag, mark, nlines)
                mean_line = ax.get_lines()[3 if spag else 0]
                assert mean_line.get_linestyle() == "--"
                if disp == "band":
                    assert len(ax.collections) >= 1
    print("draw_series OK")


def test_frame_options():
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [1, 2])
    ax.set_xlabel("x")
    o = dict(ps.DEFAULT_OPTIONS, grid=True, font_size=12, y_from_zero=True)
    ps.apply_frame_options(ax, o)
    assert ax.get_ylim()[0] == 0
    assert ax.xaxis.label.get_size() == 12
    assert any(gl.get_visible() for gl in ax.get_xgridlines())
    print("frame_options OK")


def test_dialog_and_pilot():
    import tkinter as tk
    from tkinter import ttk
    root = tk.Tk()
    root.withdraw()
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "videos"), exist_ok=True)
    json.dump({}, open(os.path.join(d, "PixelPaws_project.json"), "w"))
    fired = []
    ps.open_options_dialog(root, d, ["A", "B"], on_apply=lambda: fired.append(1))
    wins = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
    assert wins, "dialog not created"
    win = wins[-1]
    win._pp_apply()
    assert fired == [1]
    assert "plot_options" in json.load(
        open(os.path.join(d, "PixelPaws_project.json")))
    # empty-groups construction must not refuse
    ps.open_options_dialog(root, d, [], on_apply=None)
    wins2 = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)
             and w.winfo_exists()]
    assert wins2, "dialog refused empty groups"
    for w in wins2:
        w.destroy()

    # pilot wiring: locomotion constructs with the gear button; no gait import
    from locomotion_tab import LocomotionTab

    class StubApp:
        notebook = None
        transitions_tab = None
        key_file_path = None
    StubApp.root = root
    StubApp.current_project_folder = tk.StringVar(value=r"E:/PixelPaws_practice")
    lt = LocomotionTab(ttk.Notebook(root), StubApp())

    def walk(w):
        yield w
        for c in w.winfo_children():
            yield from walk(c)
    gears = [w for w in walk(lt) if isinstance(w, ttk.Button)
             and "\u2699" in str(w.cget("text"))]
    assert len(gears) == 1, "style (gear) button missing"
    assert "gait_limb_tab" not in sys.modules
    # draw with non-default options against the practice project
    if os.path.isdir(r"E:/PixelPaws_practice/videos"):
        lt.scan_sessions()
        ps.save_options(r"E:/PixelPaws_practice",
                        dict(ps.DEFAULT_OPTIONS, error_type="SD",
                             error_display="caps", show_individual=True,
                             grid=True, line_styles={"Drug": "dashed"}))
        lt._view.set("cumulative")
        lt.refresh()
        ax = lt._fig.axes[0]
        assert len(ax.get_lines()) >= 2 + 6, "spaghetti lines missing"
        styles = {l.get_linestyle() for l in ax.get_lines()}
        assert "--" in styles, styles
        assert any(gl.get_visible() for gl in ax.get_xgridlines())
        ps.reset_options(r"E:/PixelPaws_practice")
    root.destroy()
    print("dialog + pilot OK")


if __name__ == "__main__":
    test_roundtrip()
    test_calc_error()
    test_draw_series()
    test_frame_options()
    test_dialog_and_pilot()
    print("ALL PLOT OPTION TESTS PASSED")
