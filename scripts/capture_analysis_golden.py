"""
capture_analysis_golden.py — Golden capture for the Single-Classifier rebuild
=============================================================================

Runs the OLD analysis_tab.AnalysisTab headless (withdrawn tk.Tk + stub
main_gui) on the practice project E:/PixelPaws_practice and writes golden
outputs that tests/test_golden_practice.py later compares against the new
pure-compute analysis_core.run_analysis.

Outputs (tests/golden/):
  practice_results_default.csv              Default preset, analyze_mode='both'
  practice_results_formalin.csv             Formalin (2-phase) preset (windows
                                            0-10 / 10-60 min — do NOT fit the
                                            2-min practice videos, so no phase
                                            rows; kept as the spec'd capture)
  practice_results_formalin_shortphase.csv  Formalin preset with phase windows
                                            shrunk to 0-1 / 1-2 min so phase
                                            rows actually exist and the
                                            phase_rows port gets golden
                                            coverage (documented deviation:
                                            additive third capture)
  practice_perframe_default.npz             perframe_data of the Default run,
                                            keys "subject|treatment|behavior"
  capture_config.json                       exact AnalysisConfig equivalents
                                            for each run + input paths

Stubs used to run the tab headless (each documented):
  * StubApp — minimal main_gui: root, current_project_folder (StringVar),
    key_file_data ({}), _ba_scan_sessions (no-op; only the key-file *generate*
    flow calls it, which we never trigger).
  * tab._dash_show_graphs / tab._render_results_view — replaced with no-op
    lambdas on the INSTANCE (run_analysis calls them after computing
    results_df; they build Toplevel graph windows / ttk stats tables, which
    are pure display and irrelevant to the golden data).

Run with:  PYTHONIOENCODING=utf-8 python scripts/capture_analysis_golden.py
"""

import matplotlib
matplotlib.use("Agg")  # must precede anything that pulls pyplot

import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

PRACTICE = r"E:/PixelPaws_practice"
GOLDEN_DIR = os.path.join(REPO, "tests", "golden")


def main():
    if not os.path.isdir(PRACTICE):
        print(f"SKIP: practice project not found at {PRACTICE}")
        sys.exit(1)

    os.makedirs(GOLDEN_DIR, exist_ok=True)

    import tkinter as tk
    from tkinter import ttk  # noqa: F401  (tab builds ttk widgets)

    root = tk.Tk()
    root.withdraw()

    class StubApp:
        """Minimal main_gui stand-in (pattern from tests/test_plot_options.py)."""
        key_file_data = {}
        _ba_scan_sessions = staticmethod(lambda *a, **k: None)  # generate-flow only
    StubApp.root = root
    StubApp.current_project_folder = tk.StringVar(value=PRACTICE)

    from analysis_tab import AnalysisTab
    tab = AnalysisTab(root, StubApp())

    # Display-only calls made at the end of run_analysis; no-op them so the
    # capture never builds graph Toplevels / stats tables (see module docstring).
    tab._dash_show_graphs = lambda *a, **k: None
    tab._render_results_view = lambda *a, **k: None

    # Construction already ran scan_project_folder via _sync_project_folder_from_gui;
    # run it explicitly per spec (idempotent).
    tab.scan_project_folder(PRACTICE)

    # Key file: auto-loaded when the scan finds exactly one candidate; fall back
    # to the canonical discovery helper otherwise.
    from project_config import find_key_files
    key_candidates = find_key_files(PRACTICE)
    assert key_candidates, "no key file found in practice project"
    key_path = key_candidates[0]
    if tab.key_df is None:
        tab.load_key_file(key_path)

    assert tab.key_df is not None, "key_df not loaded"
    assert tab.prediction_files, "no prediction files found"
    print(f"key file: {key_path}  ({len(tab.key_df)} subjects)")
    print(f"prediction files: {len(tab.prediction_files)}  "
          f"behaviors: {sorted(tab.available_behaviors.keys())}")

    # analyze_mode 'both' so the golden also covers the combined path.
    tab.analyze_mode.set('both')

    cfg_record = {
        "project_folder": PRACTICE,
        "pred_folder": tab.pred_folder_var.get(),
        "key_file": key_path,
        "runs": {},
    }

    def _run_and_save(tag, csv_name, save_perframe=False,
                      allow_phase_nan_auc=False):
        tab.run_analysis(silent=True)
        assert tab.results_df is not None and len(tab.results_df) > 0, \
            f"{tag}: run_analysis produced no results"
        df = tab.results_df

        # Document that AUC is a literal duplicate of Total_Time_s (the new
        # core drops it).  Phase rows never carried AUC (bin_idx=None) so they
        # are excluded where present.
        if allow_phase_nan_auc:
            chk = df[df['AUC'].notna()]
            nan_rows = df[df['AUC'].isna()]
            assert (nan_rows['Bin_Index'] < 0).all(), \
                f"{tag}: NaN AUC outside phase rows"
        else:
            chk = df
        assert chk['AUC'].notna().all(), f"{tag}: unexpected NaN AUC"
        assert (chk['AUC'] == chk['Total_Time_s']).all(), \
            f"{tag}: AUC != Total_Time_s"
        print(f"{tag}: AUC == Total_Time_s confirmed on {len(chk)} rows")

        out = os.path.join(GOLDEN_DIR, csv_name)
        df.to_csv(out, index=False)
        print(f"{tag}: wrote {out}  ({len(df)} rows, {len(df.columns)} cols)")

        if save_perframe:
            npz_path = os.path.join(GOLDEN_DIR, "practice_perframe_default.npz")
            arrays = {f"{s}|{t}|{b}": arr
                      for (s, t, b), arr in tab.perframe_data.items()}
            np.savez(npz_path, **arrays)
            print(f"{tag}: wrote {npz_path}  ({len(arrays)} arrays)")

        # Record the exact config equivalent for the replay test.
        phases = []
        if tab.enable_phase_analysis.get():
            phases = [
                {"name": "Acute", "start_min": tab.acute_start_var.get(),
                 "end_min": tab.acute_end_var.get(), "bin_index": -1},
                {"name": "Phase_II", "start_min": tab.phase2_start_var.get(),
                 "end_min": tab.phase2_end_var.get(), "bin_index": -2},
            ]
        bin_size = tab.bin_size_var.get()
        if tab.bin_unit_var.get() == 'seconds':
            bin_size = bin_size / 60.0
        cfg_record["runs"][tag] = {
            "csv": csv_name,
            "bin_size_min": bin_size,
            "whole_session": bool(tab.no_time_bins_var.get()),
            "fps": tab.fps_var.get(),
            "analyze_mode": tab.analyze_mode.get(),
            "phases": phases,
            "filename_prefix": tab.filename_prefix_var.get(),
        }

    # ── (a) Default preset ────────────────────────────────────────────────
    tab.analysis_preset_var.set('Default')
    tab._apply_analysis_preset()
    _run_and_save("default", "practice_results_default.csv", save_perframe=True)

    # ── (b) Formalin (2-phase) preset, as it ships (0-10 / 10-60 min) ────
    tab.analysis_preset_var.set('Formalin (2-phase)')
    tab._apply_analysis_preset()
    _run_and_save("formalin", "practice_results_formalin.csv",
                  allow_phase_nan_auc=True)

    # ── (c) Formalin preset with phase windows that FIT the 2-min practice
    #        videos (0-1 / 1-2 min), so phase rows exist in a golden ────────
    tab.acute_start_var.set(0)
    tab.acute_end_var.set(1)
    tab.phase2_start_var.set(1)
    tab.phase2_end_var.set(2)
    _run_and_save("formalin_shortphase",
                  "practice_results_formalin_shortphase.csv",
                  allow_phase_nan_auc=True)
    _sp = pd.read_csv(os.path.join(GOLDEN_DIR,
                                   "practice_results_formalin_shortphase.csv"))
    n_phase = int((_sp['Bin_Index'] < 0).sum())
    assert n_phase > 0, "shortphase capture produced no phase rows"
    print(f"formalin_shortphase: {n_phase} phase rows captured")

    with open(os.path.join(GOLDEN_DIR, "capture_config.json"), "w",
              encoding="utf-8") as f:
        json.dump(cfg_record, f, indent=2)
    print(f"wrote {os.path.join(GOLDEN_DIR, 'capture_config.json')}")

    root.destroy()
    print("CAPTURE OK")


if __name__ == "__main__":
    main()
