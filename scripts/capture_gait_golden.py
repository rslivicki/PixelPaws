# -*- coding: utf-8 -*-
"""Capture the gait golden dataset from the OLD GaitLimbTab, headlessly.

Runs the legacy tab's `_analyze_session` on every session of the practice
project (E:/PixelPaws_practice) with the manuscript-preset params — the same
params dict `_start_analysis` would build — and saves:

    tests/golden/gait_summary.csv         summary rows (base + summary metrics)
    tests/golden/gait_bins.csv            bin rows (base + per-bin metrics)
    tests/golden/gait_capture_config.json params / paw_map / sessions used

First run may extract brightness+contour from video (minutes/session); the
caches then live in the practice project's gait_limb_analysis/ and later
runs (and the golden equality test) reuse them.

Usage:  PYTHONIOENCODING=utf-8 python scripts/capture_gait_golden.py
"""
import sys, os, json, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import matplotlib
matplotlib.use("Agg")
import tkinter as tk
from tkinter import ttk
import pandas as pd

PROJ = r"E:/PixelPaws_practice"
GOLDEN_DIR = os.path.join(REPO, "tests", "golden")

if not os.path.isdir(PROJ):
    print("practice project not present - cannot capture golden")
    sys.exit(1)

root = tk.Tk()
root.withdraw()


class StubApp:
    notebook = None
    key_file_data = {}


StubApp.root = root
StubApp.current_project_folder = tk.StringVar(value=PROJ)

from gait_limb_tab import GaitLimbTab

tab = GaitLimbTab(ttk.Notebook(root), StubApp())

# Scan sessions + key files exactly as the GUI would on project change.
tab._scan_sessions()
assert tab._sessions, "no sessions found in practice project"
print(f"{len(tab._sessions)} sessions found:",
      [s['session_name'] for s in tab._sessions])

# Ensure the key file is the practice key (auto-load may or may not have hit
# depending on how many Subject/Treatment CSVs the walk finds).
key_path = os.path.join(PROJ, "practice_key.csv")
tab._load_key_file(key_path)
assert tab._key_df is not None, "key file failed to load"

# Apply the manuscript preset (the default profile) deterministically.
tab._preset_var.set('Paw contact (manuscript gate)')
tab._apply_gait_preset()
assert tab._exclude_lick_var.get(), (
    "licking exclusion did not auto-enable - are behavior predictions "
    "missing from the practice project's results/ folder?")
print("lick behavior:", tab._lick_behavior_var.get())

# The preset's 5-min bins yield ZERO bins on the 2-minute practice clips, so
# the golden would never exercise the per-bin path. Override to 30-s bins for
# the capture (everything else stays manuscript-preset). Recorded in config.
tab._bin_seconds_var.set(30)
tab._bin_unit_var.set('seconds')

# ── params: the exact dict _start_analysis builds ──────────────────────────
speed_thresh_raw = tab._speed_thresh_var.get().strip()
speed_thresh = 'auto'
if speed_thresh_raw.lower() != 'auto':
    try:
        speed_thresh = float(speed_thresh_raw)
    except ValueError:
        speed_thresh = 'auto'

params = {
    'contact_threshold': tab._contact_thresh_var.get(),
    'height_window':     tab._height_window_var.get(),
    'bin_seconds':       tab._bin_seconds_var.get(),
    'bin_unit':          tab._bin_unit_var.get(),
    'fallback_fps':      float(tab._fallback_fps_var.get()),
    'use_brightness':    tab._use_brightness_var.get(),
    'brt_threshold':     tab._brt_thresh_var.get(),
    'brt_weight':        tab._brt_weight_var.get(),
    'roi_sizes':         {role: tab._roi_size_vars[role].get()
                          for role in tab.ROLES},
    'crop_offset_x':      tab._crop_x_var.get(),
    'crop_offset_y':      tab._crop_y_var.get(),
    'extraction_stride':  tab._extraction_stride_var.get(),
    'contact_method':     tab._contact_method_var.get(),
    'speed_threshold':    speed_thresh,
    'median_filter_ms':   tab._median_filter_var.get(),
    'min_bout_ms':        tab._min_bout_var.get(),
    'min_stance_ms':      tab._min_stance_ms_var.get(),
    'use_likelihood':     tab._use_likelihood_var.get(),
    'likelihood_threshold': tab._likelihood_thresh_var.get(),
    'loco_filter':        tab._loco_filter_var.get(),
    'loco_threshold':     tab._loco_thresh_var.get(),
    'paw_contour':        tab._paw_contour_var.get(),
    'contour_roi_sizes':  {role: tab._contour_roi_size_vars[role].get()
                           for role in tab.ROLES},
    'contour_forelimbs':  tab._contour_forelimbs_var.get(),
    'contour_area_threshold': tab._contour_area_thresh_var.get(),
    'contour_area_max': tab._contour_area_max_var.get(),
    'exclude_licking':    bool(tab._exclude_lick_var.get()),
    'lick_behavior':      tab._lick_behavior_var.get(),
    'lick_threshold':     float(tab._lick_thresh_var.get()),
    'gate_4paw':          bool(tab._gate_4paw_var.get()),
}

paw_map = {role: tab._role_vars[role].get().strip() for role in tab.ROLES}
print("paw_map:", paw_map)
print("params:", json.dumps(params, indent=1, default=str))

# ── run: replicate _analysis_thread's loop (no thread, no widgets) ─────────
summary_rows = []
bin_rows = []
not_done = []
t0 = time.time()
for sess in tab._sessions:
    name = sess['session_name']
    t1 = time.time()
    print(f"Processing: {name} ...", flush=True)
    result = tab._analyze_session(sess, paw_map, params)
    if result:
        subj = tab._resolve_subject(name)
        treatment = tab._get_treatment(subj)
        base = dict(session=name, subject=subj, treatment=treatment)
        srow = {**base, **result['summary']}
        summary_rows.append(srow)
        for brow in result['bins']:
            bin_rows.append({**base, **brow})
        print(f"  done in {time.time()-t1:.1f}s "
              f"({len(result['bins'])} bins)")
    else:
        not_done.append(name)
        print(f"  SKIPPED (no result)")

print(f"total: {time.time()-t0:.1f}s; "
      f"{len(summary_rows)} summaries, {len(bin_rows)} bin rows, "
      f"not_done={not_done}")

os.makedirs(GOLDEN_DIR, exist_ok=True)
sum_df = pd.DataFrame(summary_rows)
bin_df = pd.DataFrame(bin_rows)
sum_df.to_csv(os.path.join(GOLDEN_DIR, "gait_summary.csv"), index=False)
bin_df.to_csv(os.path.join(GOLDEN_DIR, "gait_bins.csv"), index=False)
with open(os.path.join(GOLDEN_DIR, "gait_capture_config.json"), "w") as f:
    json.dump({
        'project': PROJ,
        'preset': 'Paw contact (manuscript gate)',
        'note': 'bin_seconds/bin_unit overridden to 30 s for the 2-min '
                'practice clips (preset 5-min bins would produce 0 bins)',
        'key_file': key_path,
        'params': params,
        'paw_map': paw_map,
        'sessions': [s['session_name'] for s in tab._sessions],
        'not_done': not_done,
        'pawlike_thresholds': tab._pawlike_thresholds,
    }, f, indent=2, default=str)

print("golden written:",
      sum_df.shape, "summary;", bin_df.shape, "bins")
root.destroy()
print("CAPTURE OK")
