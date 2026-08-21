# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, r"E:/Code/PixelPaws")
import matplotlib; matplotlib.use("Agg")
import tkinter as tk
from tkinter import ttk
import numpy as np

PROJ = r"E:/PixelPaws_practice"
if not os.path.isdir(PROJ):
    print("practice project not present - skipping contract test")
    sys.exit(0)
root = tk.Tk(); root.withdraw()
import plot_style as ps
ps.reset_options(PROJ)
from single_classifier_tab import SingleClassifierTab

class StubApp:
    notebook = None
    key_file_data = {}
StubApp.root = root
StubApp.current_project_folder = tk.StringVar(value=PROJ)

tab = SingleClassifierTab(ttk.Notebook(root), StubApp())
# contract members
for m in ("analysis_project_var", "key_df", "results_df",
          "scan_project_folder", "_dash_subjects_overview", "run_analysis"):
    assert hasattr(tab, m), m
print("contract members present")

tab.scan_project_folder(PROJ)
assert tab.key_df is not None, "key not auto-loaded"
assert len(tab._behaviors) == 8, tab._behaviors.keys()
print("scan:", tab._scan_status.get())
print("subjects:", tab._subjects_status.get())
assert "6 subject(s)" in tab._subjects_status.get()

# synchronous silent run (batch handoff path)
tab.run_analysis(silent=True)
assert tab.results_df is not None and len(tab.results_df) > 0
print("silent run:", len(tab.results_df), "rows;",
      sorted(tab.results_df['Behavior'].unique())[:3], "...")
assert "AUC" not in tab.results_df.columns
assert tab.perframe_data, "perframe missing"

# all views render
import tempfile
OUT = tempfile.mkdtemp()
rendered = []
for key, label in tab._VIEW_LABELS:
    if key == "phase":
        continue          # no phase rows in Default preset
    tab._view.set(key)
    tab.refresh()
    axes = tab._fig.axes
    assert axes, key
    txts = [t.get_text() for a in axes for t in a.texts]
    assert not any("Could not draw" in t for t in txts), (key, txts)
    rendered.append(key)
print("views rendered:", rendered)
tab._view.set("timecourse"); tab.refresh()
tab._fig.savefig(os.path.join(OUT, "sct_timecourse.png"), dpi=110)
tab._view.set("total_time"); tab.refresh()
tab._fig.savefig(os.path.join(OUT, "sct_total_time.png"), dpi=110)
tab._view.set("heatmap_time"); tab.refresh()
tab._fig.savefig(os.path.join(OUT, "sct_heatmap.png"), dpi=110)

# stats flip
tab.enable_stats_var.set(True)
tab._view.set("timecourse")
tab._stats_mode.set(True)
tab._apply_stats_mode()
txt = tab._stats_widget.get("1.0", "end")
assert "Omnibus" in txt and ("Welch" in txt or "Mann" in txt or "test" in txt)
assert "Vehicle" in txt and "Drug" in txt
print("stats text OK:", len(txt), "chars")
tab._stats_mode.set(False); tab._apply_stats_mode()

# stats markers render on graph
tab.refresh()
print("stats-enabled graph OK")

# phase run (short windows to fit 2-min clips)
tab._preset_var.set("Formalin (2-phase)")
tab._apply_preset()
tab.acute_start_var.set(0); tab.acute_end_var.set(1)
tab.phase2_start_var.set(1); tab.phase2_end_var.set(2)
tab.run_analysis(silent=True)
assert (tab.results_df["Bin_Index"] < 0).any(), "no phase rows"
tab._refresh_view_choices()
labels = list(tab._view_cb.cget("values"))
assert "Phase Analysis" in labels
tab._view.set("phase"); tab.refresh()
txts = [t.get_text() for a in tab._fig.axes for t in a.texts]
assert not any("Could not draw" in t for t in txts), txts
tab._fig.savefig(os.path.join(OUT, "sct_phase.png"), dpi=110)
print("phase view OK")

# axis override
tab._view.set("timecourse")
tab._axis_overrides["timecourse"] = (0.0, 100.0)
tab.refresh()
assert tab._fig.axes[0].get_ylim() == (0.0, 100.0)
print("axis override OK")

# sidecar export
import tempfile, json
tmp = os.path.join(tempfile.mkdtemp(), "out.csv")
tab.results_df.to_csv(tmp, index=False)
tab._write_sidecar(tmp)
meta = json.load(open(tmp + ".meta.json"))
assert meta["rows"] == len(tab.results_df) and meta["config"]["fps"] == 60.0
print("sidecar OK:", sorted(meta.keys()))

root.destroy()
print("SCT INTEGRATION PASSED")
