"""
Body Contact Analysis Tab  (body_contact_tab.py)
==================================================
Companion to ``gait_limb_tab.py``. Where the Gait & Limb tab tracks
per-paw contact / stride / cadence, this tab tracks the same kind of
contact + brightness metrics for the **midline keypoints** that the
gait tab deliberately ignores: ``centroid`` (a usable belly proxy)
and ``tailbase`` (a usable perineum / anus proxy).

Both keypoints are produced by the existing 9-keypoint DLC model so
no DLC retraining is needed. The brightness side requires that the
project's ``bp_pixbrt_list`` includes ``centroid`` and ``tailbase`` —
add them via the project setup wizard before running this tab.

Metrics per session, per keypoint (raw keypoint names throughout):
  contact_pct_<bp>            — % frames classified as in contact
  n_bouts_<bp>                — number of contact bouts after debounce
  mean_bout_dur_<bp>          — mean contact-bout duration (seconds)
  mean_brightness_<bp>        — mean Pix_<bp> over all frames
  mean_brightness_contact_<bp>— mean Pix_<bp> during contact frames

Detection method (radio): ``brightness`` (default), ``height``,
``combined`` (AND).

Caveats (also surfaced in the tab):
- Centroid and tailbase are virtual midpoints, not literal
  floor-contact points. Height < threshold conflates "pressing into
  floor" with "lying flat / sleeping"; brightness is the cleaner
  pressure signal.
- The 15 px paw-contact threshold is too tight for these midline
  points. Defaults here are 40 px, exposed for tuning.
"""

from __future__ import annotations

import os
import re
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime

import numpy as np
import pandas as pd
from ui_utils import FONT_FAMILY

try:
    from evaluation_tab import find_session_triplets
except ImportError as _fst_err:
    # If evaluation_tab fails to import, the Body Contact tab gets an
    # empty session list. Pre-2026-05-01 this happened silently; now we
    # log it once at module load so the failure is at least traceable.
    print(f'[body_contact_tab] WARNING: could not import '
          f'find_session_triplets from evaluation_tab: {_fst_err}. '
          f'The Body Contact tab will show an empty session list.')
    def find_session_triplets(folder, **kw):
        return []

try:
    from feature_cache import FeatureCacheManager
    _FEATURE_CACHE_AVAILABLE = True
except ImportError:
    FeatureCacheManager = None
    _FEATURE_CACHE_AVAILABLE = False

try:
    from PixelPaws_GUI import extract_subject_id_from_filename as _extract_sid
except ImportError:
    _extract_sid = None

try:
    from io_utils import get_git_sha
except ImportError:
    def get_git_sha():
        return 'unknown'

try:
    import cv2
    _CV2_OK = True
except ImportError:
    cv2 = None
    _CV2_OK = False

try:
    from pose_features import PoseFeatureExtractor
    _POSE_OK = True
except ImportError:
    PoseFeatureExtractor = None
    _POSE_OK = False


# Midline keypoints we score. Order matters for column layout.
_TARGETS = ('centroid', 'tailbase')

# Detection-method options surfaced in the radio.
_METHODS = ('brightness', 'height', 'combined')

# Brightness signal options. Each picks a different scaling of the
# per-frame ROI brightness. Defaults: ΔBrt for analysis (animal-invariant)
# but all four are computed and visible in the preview readout so the
# user can build intuition on which signal best separates contact.
#
#   raw         Pix_<bp>                                      — animal-dependent, lighting-dependent
#   dbrt        Pix_baseline_sub_<bp>  (raw − session_median) — animal-invariant via per-session normalization
#   z           (raw − session_mean) / session_std            — same idea as ΔBrt but in standard-deviation units
#   frac_bright count(ROI_pixels > floor) / count(ROI_pixels) — closest to a physical unit; needs an absolute floor
_SIGNALS = ('raw', 'dbrt', 'z', 'frac_bright')

# Pretty labels for the radio + threshold spinbox.
_SIGNAL_LABELS = {
    'raw':         'raw  (Pix_<bp>)',
    'dbrt':        'ΔBrt (Pix_baseline_sub_<bp>)',
    'z':           'z-score',
    'frac_bright': 'frac_bright (preview only)',
}


class BodyContactTab(ttk.Frame):
    """Body-contact analysis (centroid + tailbase) — sibling of GaitLimbTab."""

    def __init__(self, parent, main_gui):
        super().__init__(parent)
        self.app = main_gui
        self.pack(fill='both', expand=True)

        self._sessions: list = []
        self._key_df: pd.DataFrame | None = None
        self._summary_df: pd.DataFrame | None = None
        self._bouts_df: pd.DataFrame | None = None
        self._session_bouts: dict[str, dict] = {}  # session -> {bp: [bout, ...]}
        self._fit_thread: threading.Thread | None = None
        self._cancel_flag = threading.Event()
        self._session_features: dict[str, pd.DataFrame] = {}

        self._method_var = tk.StringVar(value='brightness')
        # Signal selector for brightness-mode thresholding. See
        # _SIGNALS for the full list. ΔBrt is the default because it's
        # animal-invariant (session-median normalized) without
        # needing per-rig calibration.
        self._signal_var = tk.StringVar(value='dbrt')
        self._frac_bright_floor_var = tk.IntVar(value=180)
        self._height_thresh_vars = {bp: tk.IntVar(value=40) for bp in _TARGETS}
        # Per-signal brightness threshold dicts. The brightness-mode
        # contact decision compares the *selected* signal against
        # *its* threshold, so we keep one dict per signal rather than
        # forcing the user to retune when switching signals.
        self._brightness_thresh_vars = {bp: tk.DoubleVar(value=5.0) for bp in _TARGETS}
        self._raw_thresh_vars = {bp: tk.DoubleVar(value=80.0) for bp in _TARGETS}
        self._z_thresh_vars   = {bp: tk.DoubleVar(value=2.0)  for bp in _TARGETS}
        self._fb_thresh_vars  = {bp: tk.DoubleVar(value=0.30) for bp in _TARGETS}
        # ROI half-size per keypoint — mirrors gait tab's per-paw
        # ROI half-size spinboxes; 30 px is a sensible midpoint
        # default for centroid / tailbase (paws use 25 in the gait tab).
        self._roi_half_vars = {bp: tk.IntVar(value=30) for bp in _TARGETS}

        # Smoothing controls (step 2). Likelihood gating drops frames
        # where DLC confidence < threshold; the gap is linearly
        # interpolated up to `like_gap_limit` frames. Rolling-median
        # on the brightness signal kills single-frame flicker spikes
        # before the threshold is applied. min_bout_ms (already below)
        # is the last-line debounce.
        self._use_like_gate_var      = tk.BooleanVar(value=True)
        self._like_thresh_var        = tk.DoubleVar(value=0.6)
        self._like_gap_limit_var     = tk.IntVar(value=10)   # frames
        self._use_signal_smooth_var  = tk.BooleanVar(value=True)
        self._signal_smooth_window_var = tk.IntVar(value=5)  # frames

        self._min_bout_ms_var = tk.IntVar(value=100)
        self._fallback_fps_var = tk.DoubleVar(value=60.0)
        self._key_file_var = tk.StringVar()
        self._prefix_var = tk.StringVar()

        # State the borrowed gait graph helpers expect to find on self.
        # _open_graphs binds the gait helpers as MethodType on this
        # instance the first time it runs; these vars feed them.
        self._enable_stats_var = tk.BooleanVar(value=True)
        self._stats_paradigm_var = tk.StringVar(value='auto')
        self._last_graph_cfg = None
        # Plain attribute; gait's _add_stat_annotation reads this lazily.
        self._sig_style = 'asterisk'
        self._gait_graph_helpers_bound = False

        self._build_ui()

    # ═══════════════════════════════════════════════════════════════════════
    # UI
    # ═══════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        hdr = ttk.Frame(self)
        hdr.pack(fill='x', padx=10, pady=(8, 2))
        ttk.Label(hdr, text="Body Contact Analysis  (centroid + tailbase)",
                  font=(FONT_FAMILY, 13, 'bold')).pack(side='left')
        ttk.Label(
            hdr,
            text="  midline keypoints — brightness is the cleaner pressure signal",
            foreground='grey').pack(side='left', padx=(6, 0))

        paned = ttk.PanedWindow(self, orient='horizontal')
        paned.pack(fill='both', expand=True, padx=6, pady=4)

        left  = ttk.Frame(paned, width=240)
        mid   = ttk.Frame(paned, width=300)
        right = ttk.Frame(paned, width=420)
        paned.add(left,  weight=1)
        paned.add(mid,   weight=1)
        paned.add(right, weight=2)

        self._build_sessions_panel(left)
        self._build_settings_panel(mid)
        self._build_results_panel(right)

    # ── Left: sessions list ────────────────────────────────────────────────

    def _build_sessions_panel(self, parent):
        lf = ttk.LabelFrame(parent, text="Sessions", padding=5)
        lf.pack(fill='both', expand=True, padx=4, pady=4)

        btn_row = ttk.Frame(lf)
        btn_row.pack(fill='x', pady=(0, 4))
        ttk.Button(btn_row, text="Scan",
                   command=self._scan_sessions).pack(side='left', padx=2)
        ttk.Button(btn_row, text="All",
                   command=lambda: self._sess_tree.selection_set(
                       self._sess_tree.get_children())).pack(side='left', padx=2)
        ttk.Button(btn_row, text="Clear",
                   command=lambda: self._sess_tree.selection_remove(
                       self._sess_tree.get_children())).pack(side='left', padx=2)

        cols = ('name', 'subject', 'cache')
        self._sess_tree = ttk.Treeview(lf, columns=cols, show='headings',
                                        selectmode='extended', height=20)
        self._sess_tree.heading('name', text='Session')
        self._sess_tree.heading('subject', text='Subject')
        self._sess_tree.heading('cache', text='Cache?')
        self._sess_tree.column('name', width=140, stretch=True)
        self._sess_tree.column('subject', width=60, stretch=False)
        self._sess_tree.column('cache', width=50, stretch=False)
        sb = ttk.Scrollbar(lf, orient='vertical', command=self._sess_tree.yview)
        self._sess_tree.config(yscrollcommand=sb.set)
        self._sess_tree.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')

        self._sess_lbl = ttk.Label(parent, text='No sessions scanned yet',
                                    foreground='grey', wraplength=220)
        self._sess_lbl.pack(anchor='w', padx=6, pady=(2, 0))

    # ── Middle: settings ───────────────────────────────────────────────────

    def _build_settings_panel(self, parent):
        run_frame = ttk.Frame(parent)
        run_frame.pack(fill='x', padx=4, pady=(4, 4))
        self._run_btn = ttk.Button(run_frame, text="▶ Run Analysis",
                                    command=self._start_analysis)
        self._run_btn.pack(side='left', padx=2)
        self._cancel_btn = ttk.Button(run_frame, text="■ Cancel",
                                       command=self._cancel_analysis,
                                       state='disabled')
        self._cancel_btn.pack(side='left', padx=2)
        self._progress = ttk.Progressbar(run_frame, mode='determinate',
                                          length=120)
        self._progress.pack(side='left', padx=4)

        # Sub-progress label — names the current step within each
        # session. Mirrors the gait tab's `_sub_progress_label` pattern.
        # Empty string when idle.
        self._sub_progress_var = tk.StringVar(value='')
        ttk.Label(parent, textvariable=self._sub_progress_var,
                   foreground='#444', font=('TkDefaultFont', 8),
                   wraplength=290, justify='left').pack(
            anchor='w', padx=8, pady=(0, 4))

        # Brightness-extraction status banner — shown when the project
        # has bp_pixbrt_list including centroid/tailbase but the cached
        # features for the current sessions don't have those columns
        # (i.e., re-extraction is required for brightness-mode to work
        # in this tab). One-click button launches watch_dlc_extract
        # --continuous on the project folder. Hidden by default;
        # _refresh_extraction_banner() decides when to show it.
        self._banner_frame = ttk.Frame(parent)
        # not packed yet — _refresh_extraction_banner controls visibility
        self._banner_var = tk.StringVar(value='')
        ttk.Label(self._banner_frame, textvariable=self._banner_var,
                   foreground='#993300', font=('TkDefaultFont', 9, 'bold'),
                   wraplength=290, justify='left').pack(anchor='w', padx=4,
                                                          pady=(2, 2))
        self._banner_btn = ttk.Button(
            self._banner_frame, text="Launch feature extraction watcher",
            command=self._launch_watcher)
        self._banner_btn.pack(anchor='w', padx=4, pady=(0, 4))

        # Key file picker
        kf_lf = ttk.LabelFrame(parent, text="Key File (optional)", padding=5)
        kf_lf.pack(fill='x', padx=4, pady=(0, 4))
        # Stable anchor for the extraction banner: the banner gets
        # packed with before=self._kf_lf when needed, so it always
        # appears between the run row and the key-file frame.
        self._kf_lf = kf_lf
        kf_row = ttk.Frame(kf_lf)
        kf_row.pack(fill='x', pady=2)
        ttk.Label(kf_row, text="File:", width=6).pack(side='left')
        ttk.Entry(kf_row, textvariable=self._key_file_var,
                  width=22).pack(side='left', padx=3)
        ttk.Button(kf_row, text="Browse",
                   command=self._browse_key_file).pack(side='left')

        pfx_row = ttk.Frame(kf_lf)
        pfx_row.pack(fill='x', pady=2)
        ttk.Label(pfx_row, text="Prefix:", width=6).pack(side='left')
        ttk.Entry(pfx_row, textvariable=self._prefix_var,
                  width=22).pack(side='left', padx=3)

        # Detection method
        method_lf = ttk.LabelFrame(parent, text="Contact Detection", padding=5)
        method_lf.pack(fill='x', padx=4, pady=(0, 4))
        for i, m in enumerate(_METHODS):
            ttk.Radiobutton(method_lf, text=m, variable=self._method_var,
                            value=m).grid(row=0, column=i, sticky='w', padx=4)
        ttk.Label(method_lf,
                  text=("brightness: <signal> > thresh (pressure proxy)\n"
                        "height: <bp>_Height < thresh (geometric proxy)\n"
                        "combined: both must agree (AND)"),
                  foreground='grey', font=('TkDefaultFont', 8),
                  justify='left').grid(row=1, column=0, columnspan=3,
                                        sticky='w', pady=(2, 0))

        # Brightness signal selector — controls which derived signal
        # the brightness-mode threshold compares against. All four are
        # visible in the preview readout regardless. raw / dbrt / z
        # work today; frac_bright is preview-only in v1 because it
        # requires per-frame ROI pixel counts that aren't in the
        # cached features (see plan step 1.5 for analysis-time
        # frac_bright support).
        sig_lf = ttk.LabelFrame(parent, text="Brightness signal", padding=5)
        sig_lf.pack(fill='x', padx=4, pady=(0, 4))
        for i, s in enumerate(_SIGNALS):
            ttk.Radiobutton(sig_lf, text=_SIGNAL_LABELS[s],
                             variable=self._signal_var,
                             value=s).grid(row=i, column=0, sticky='w',
                                            padx=2, pady=1)
        # frac_bright floor (rig-level calibration, default 180/255)
        fb_row = ttk.Frame(sig_lf)
        fb_row.grid(row=len(_SIGNALS), column=0, sticky='w', pady=(4, 0))
        ttk.Label(fb_row, text="frac_bright floor (0-255):",
                  font=('TkDefaultFont', 8)).pack(side='left')
        ttk.Spinbox(fb_row, from_=0, to=255, increment=5, width=5,
                    textvariable=self._frac_bright_floor_var).pack(
            side='left', padx=4)

        # Per-keypoint thresholds — one column per signal so the user
        # can keep separate thresholds for each, then switch signals
        # via the radio above without retuning every time.
        thr_lf = ttk.LabelFrame(parent, text="Thresholds", padding=5)
        thr_lf.pack(fill='x', padx=4, pady=(0, 4))
        # Header row
        for ci, lbl in enumerate(['', 'Height', 'raw', 'ΔBrt', 'z', 'fb'],
                                  start=0):
            ttk.Label(thr_lf, text=lbl,
                      font=('TkDefaultFont', 8, 'bold')).grid(
                row=0, column=ci, padx=2, pady=(0, 2))
        for ri, bp in enumerate(_TARGETS, start=1):
            ttk.Label(thr_lf, text=bp).grid(row=ri, column=0, sticky='w',
                                              pady=1)
            for ci, vars_dict in enumerate(
                    [self._height_thresh_vars,
                     self._raw_thresh_vars,
                     self._brightness_thresh_vars,
                     self._z_thresh_vars,
                     self._fb_thresh_vars], start=1):
                ttk.Spinbox(thr_lf, from_=0, to=300, increment=0.5,
                             width=5,
                             textvariable=vars_dict[bp]).grid(
                    row=ri, column=ci, padx=2, pady=1)
        ttk.Label(thr_lf,
                  text=("paws default 15 px; midline ~40 px. fb = frac_bright."),
                  foreground='grey', font=('TkDefaultFont', 8)).grid(
            row=len(_TARGETS) + 1, column=0, columnspan=6, sticky='w',
            pady=(4, 0))

        # ROI half-size (per keypoint) — used by the live preview to
        # draw the brightness sample square. Larger than the paw default
        # because centroid / tailbase sit higher above the floor and a
        # narrow ROI misses the surrounding pressed-flesh halo.
        roi_lf = ttk.LabelFrame(parent, text="Preview ROI half-size (px)",
                                  padding=5)
        roi_lf.pack(fill='x', padx=4, pady=(0, 4))
        for i, bp in enumerate(_TARGETS):
            ttk.Label(roi_lf, text=bp).grid(row=i, column=0, sticky='w',
                                              pady=2)
            ttk.Spinbox(roi_lf, from_=5, to=200, increment=5, width=6,
                        textvariable=self._roi_half_vars[bp]).grid(
                row=i, column=1, padx=4, pady=2)

        # Smoothing — DLC dropout gating + rolling-median on the
        # brightness signal. Defaults are tuned for 60 fps:
        # 0.6 likelihood threshold (matches gait tab), 10-frame
        # interpolation gap limit (~167 ms), 5-frame rolling-median
        # window (~83 ms) which covers single-frame flicker without
        # smearing real bouts.
        sm_lf = ttk.LabelFrame(parent, text="Smoothing (DLC dropouts)",
                                padding=5)
        sm_lf.pack(fill='x', padx=4, pady=(0, 4))
        ttk.Checkbutton(sm_lf,
                        text="Likelihood gate (drop low-confidence frames)",
                        variable=self._use_like_gate_var).pack(anchor='w')
        gate_row = ttk.Frame(sm_lf)
        gate_row.pack(fill='x', pady=(2, 0))
        ttk.Label(gate_row, text="prob threshold:",
                  font=('TkDefaultFont', 8)).pack(side='left', padx=(16, 2))
        ttk.Spinbox(gate_row, from_=0.0, to=1.0, increment=0.05, width=5,
                    textvariable=self._like_thresh_var).pack(side='left')
        ttk.Label(gate_row, text="  gap limit (fr):",
                  font=('TkDefaultFont', 8)).pack(side='left', padx=(8, 2))
        ttk.Spinbox(gate_row, from_=0, to=200, increment=1, width=5,
                    textvariable=self._like_gap_limit_var).pack(side='left')
        ttk.Checkbutton(sm_lf,
                        text="Signal rolling-median",
                        variable=self._use_signal_smooth_var).pack(
            anchor='w', pady=(4, 0))
        smooth_row = ttk.Frame(sm_lf)
        smooth_row.pack(fill='x', pady=(2, 0))
        ttk.Label(smooth_row, text="window (frames):",
                  font=('TkDefaultFont', 8)).pack(side='left', padx=(16, 2))
        ttk.Spinbox(smooth_row, from_=1, to=51, increment=2, width=5,
                    textvariable=self._signal_smooth_window_var).pack(
            side='left')

        # Bout filtering + fallback fps
        misc_lf = ttk.LabelFrame(parent, text="Misc", padding=5)
        misc_lf.pack(fill='x', padx=4, pady=(0, 4))
        mb_row = ttk.Frame(misc_lf)
        mb_row.pack(fill='x')
        ttk.Label(mb_row, text="Min bout (ms):", width=14).pack(side='left')
        ttk.Spinbox(mb_row, from_=0, to=2000, increment=20, width=6,
                    textvariable=self._min_bout_ms_var).pack(side='left')
        fps_row = ttk.Frame(misc_lf)
        fps_row.pack(fill='x', pady=(2, 0))
        ttk.Label(fps_row, text="Fallback FPS:", width=14).pack(side='left')
        ttk.Spinbox(fps_row, from_=15, to=240, increment=15, width=6,
                    textvariable=self._fallback_fps_var).pack(side='left')

        # Live preview button — opens a frame-scrubbing window with
        # centroid + tailbase ROIs and live contact decisions, mirroring
        # the gait tab's "Preview brightness…" button on its Parameters
        # tab (gait_limb_tab.py:431).
        preview_lf = ttk.LabelFrame(parent, text="Live Preview", padding=5)
        preview_lf.pack(fill='x', padx=4, pady=(0, 4))
        ttk.Button(preview_lf, text="Preview brightness / contact…",
                   command=self._open_preview).pack(fill='x', pady=2)
        ttk.Label(preview_lf,
                  text=("Pick a session in the list, then click to scrub\n"
                        "frames with the centroid + tailbase ROIs drawn\n"
                        "live. Filled ROI = currently in contact under the\n"
                        "selected method."),
                  foreground='grey', font=('TkDefaultFont', 8),
                  justify='left').pack(anchor='w', pady=(2, 0))

        # Caveat banner
        caveat = ttk.Label(
            parent,
            text=("Centroid + tailbase are MIDLINE points, not literal\n"
                  "floor-contact landmarks. Height alone can't distinguish\n"
                  "'lying flat' from 'pressing belly down' — brightness is\n"
                  "the cleaner pressure proxy."),
            foreground='#993300', font=('TkDefaultFont', 8),
            justify='left', wraplength=290)
        caveat.pack(fill='x', padx=4, pady=(2, 0))

    # ── Right: results ────────────────────────────────────────────────────

    def _build_results_panel(self, parent):
        # ── Action buttons ─────────────────────────────────────────
        btn_row = ttk.Frame(parent)
        btn_row.pack(fill='x', padx=4, pady=(4, 2))
        self._export_btn = ttk.Button(btn_row, text="Export Summary CSV",
                                       command=self._export_summary,
                                       state='disabled')
        self._export_btn.pack(side='left', padx=2)
        self._export_bouts_btn = ttk.Button(btn_row, text="Export Bouts CSV",
                                             command=self._export_bouts,
                                             state='disabled')
        self._export_bouts_btn.pack(side='left', padx=2)
        self._graphs_btn = ttk.Button(btn_row, text="Graphs",
                                       command=self._open_graphs,
                                       state='disabled')
        self._graphs_btn.pack(side='left', padx=2)
        self._plot_btn = ttk.Button(btn_row, text="Preview Session",
                                     command=self._plot_session,
                                     state='disabled')
        self._plot_btn.pack(side='left', padx=2)

        # ── Summary panel: mean ± SEM by treatment (matches gait tab) ──
        self._summary_frame = ttk.LabelFrame(parent, text="Summary",
                                              padding=5)
        self._summary_frame.pack(fill='x', padx=4, pady=(0, 4))
        ttk.Label(self._summary_frame, text="No results yet",
                  foreground='grey',
                  font=('TkDefaultFont', 9)).pack(anchor='w')

        # ── Trimmed results treeview ───────────────────────────────
        # Drop fps/n_frames (uninteresting per row), drop n_bouts +
        # whole-session brightness (kept in the CSV export, not the
        # table). Replace the brt / brt_contact pair with a single
        # `dbrt` = brt_contact - brt — the pressure-driven brightness
        # shift that's the actual point of the metric.
        res_lf = ttk.LabelFrame(parent, text="Results", padding=3)
        res_lf.pack(fill='both', expand=True, padx=4, pady=(0, 2))

        self._RES_COLS = (
            'session', 'subject', 'treatment',
            'cpct_centroid', 'mbout_centroid', 'dbrt_centroid',
            'cpct_tailbase', 'mbout_tailbase', 'dbrt_tailbase',
        )
        hdrs = [
            ('session',         'Session',         180),
            ('subject',         'Subj',             60),
            ('treatment',       'Trt',              70),
            ('cpct_centroid',   '% Centroid',       70),
            ('mbout_centroid',  'MBd Cent (s)',     80),
            ('dbrt_centroid',   'ΔBrt Cent',        70),
            ('cpct_tailbase',   '% Tailbase',       70),
            ('mbout_tailbase',  'MBd Tail (s)',     80),
            ('dbrt_tailbase',   'ΔBrt Tail',        70),
        ]
        tree_frame = ttk.Frame(res_lf)
        tree_frame.pack(fill='both', expand=True)
        self._res_tree = ttk.Treeview(tree_frame, columns=self._RES_COLS,
                                       show='headings', height=14,
                                       selectmode='browse')
        for col, hdr, w in hdrs:
            self._res_tree.heading(col, text=hdr)
            self._res_tree.column(col, width=w,
                                   stretch=(col == 'session'))
        sb = ttk.Scrollbar(tree_frame, orient='vertical',
                            command=self._res_tree.yview)
        self._res_tree.config(yscrollcommand=sb.set)
        self._res_tree.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')

        legend = ttk.Label(
            parent,
            text=("%  = contact_pct over the session  |  "
                  "MBd = mean bout duration (s)\n"
                  "ΔBrt = brightness during contact – overall brightness "
                  "(pressure-driven shift)\n"
                  "Whole-session brightness, n_bouts, fps, frame counts "
                  "are kept in the CSV export."),
            foreground='grey', font=('TkDefaultFont', 8),
            justify='left')
        legend.pack(anchor='w', padx=6, pady=(2, 0))

        # ── Log ────────────────────────────────────────────────────
        log_lf = ttk.LabelFrame(parent, text="Log", padding=3)
        log_lf.pack(fill='x', padx=4, pady=(2, 4))
        self._log_text = tk.Text(log_lf, height=4, wrap='word',
                                  state='disabled', font=('Consolas', 8))
        log_sb = ttk.Scrollbar(log_lf, orient='vertical',
                                command=self._log_text.yview)
        self._log_text.config(yscrollcommand=log_sb.set)
        self._log_text.pack(side='left', fill='both', expand=True)
        log_sb.pack(side='right', fill='y')

    def _log_ui(self, msg: str):
        try:
            self._log_text.config(state='normal')
            self._log_text.insert('end', msg + '\n')
            self._log_text.see('end')
            self._log_text.config(state='disabled')
        except Exception:
            pass

    def _update_sub_progress(self, msg: str):
        """Thread-safe sub-progress label update.

        Called from the analysis thread to surface what step is
        running for the current session — `Loading cache for X…`,
        `Computing centroid signal…`, etc.
        """
        try:
            self.app.root.after(0, self._sub_progress_var.set, msg)
        except Exception:
            try:
                self._sub_progress_var.set(msg)
            except Exception:
                pass

    def _refresh_extraction_banner(self):
        """Show/hide the missing-brightness banner based on the project's
        bp_pixbrt_list and the cached feature columns.

        Banner appears when one of:
          (a) centroid or tailbase is *not* in the project's
              `bp_pixbrt_list` — only height mode will work.
          (b) keypoint *is* in `bp_pixbrt_list` but the cached pkl
              for the first scanned session is missing `Pix_<bp>`.
              Caches predate the current config; re-extraction needed.
        """
        try:
            self._banner_frame.pack_forget()
        except Exception:
            pass

        msg = None
        try:
            cfg_path = os.path.join(self._project_folder(),
                                     'PixelPaws_project.json')
            if not os.path.isfile(cfg_path):
                return
            import json
            with open(cfg_path, encoding='utf-8') as f:
                cfg = json.load(f)
            wanted  = set(cfg.get('bp_pixbrt_list', []) or [])
            tracked = set(_TARGETS)
            not_in_list = sorted(tracked - wanted)
            if not_in_list:
                msg = (f"⚠ {' / '.join(not_in_list)} is not in this "
                       "project's `bp_pixbrt_list`. Only height-mode "
                       "metrics will work for those keypoints. Open "
                       "the project setup wizard to add them, then "
                       "re-extract features.")
            elif self._sessions and _FEATURE_CACHE_AVAILABLE:
                cache_root = os.path.join(self._project_folder(),
                                            'features')
                sample = None
                for sess in self._sessions:
                    video_dir = os.path.dirname(
                        sess.get('video') or sess.get('dlc') or '')
                    cp = FeatureCacheManager.find_any_cache(
                        sess['session_name'], cache_root, video_dir,
                        project_root=self._project_folder())
                    if cp is not None:
                        sample = cp
                        break
                if sample is not None:
                    try:
                        cached = pd.read_pickle(sample)
                        missing = [bp for bp in (wanted & tracked)
                                    if f'Pix_{bp}' not in cached.columns]
                    except Exception:
                        missing = []
                    if missing:
                        msg = (f"⚠ Brightness for "
                               f"{' / '.join(missing)} not yet "
                               "extracted on this project. Cached "
                               "pkls predate the current "
                               "`bp_pixbrt_list` — re-run feature "
                               "extraction to populate the "
                               "Pix_<bp> columns.")
        except Exception as e:
            self._log_ui(f"banner refresh error: {e}")
            return

        if msg:
            self._banner_var.set(msg)
            try:
                self._banner_frame.pack(fill='x', padx=4, pady=(0, 4),
                                          before=self._kf_lf)
            except Exception:
                self._banner_frame.pack(fill='x', padx=4, pady=(0, 4))

    def _launch_watcher(self):
        """Launch watch_dlc_extract.py --continuous on the current
        project in a detached subprocess. Lets the user kick off
        feature extraction without leaving the tab.
        """
        import subprocess, sys
        proj = self._project_folder()
        if not proj or not os.path.isdir(proj):
            messagebox.showwarning(
                "No project", "Open a project first.", parent=self)
            return
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'watch_dlc_extract.py')
        if not os.path.isfile(script):
            messagebox.showerror(
                "Watcher missing",
                f"Could not find watch_dlc_extract.py at:\n{script}",
                parent=self)
            return
        try:
            env = os.environ.copy()
            env.setdefault('PYTHONIOENCODING', 'utf-8')
            subprocess.Popen(
                [sys.executable, '-u', script, proj,
                 '--poll', '60', '--continuous'],
                env=env,
                creationflags=(subprocess.CREATE_NEW_CONSOLE
                                 if hasattr(subprocess, 'CREATE_NEW_CONSOLE')
                                 else 0))
            self._log_ui(
                f"Launched watcher (continuous mode) on {proj}.")
            messagebox.showinfo(
                "Watcher launched",
                "watch_dlc_extract.py --continuous is running in a "
                "new console window. It will auto-extract features as "
                "DLC outputs land. Closing this GUI doesn't stop it.",
                parent=self)
        except Exception as e:
            messagebox.showerror("Watcher launch failed", str(e),
                                  parent=self)

    def _refresh_summary_panel(self):
        """Render mean ± SEM by treatment for the key body-contact metrics.

        Mirrors `GaitLimbTab._refresh_summary_panel`'s grid layout but
        with the centroid + tailbase metrics; no symmetry / WBI / SI.
        """
        for w in self._summary_frame.winfo_children():
            w.destroy()
        if self._summary_df is None or self._summary_df.empty:
            ttk.Label(self._summary_frame, text="No results yet",
                      foreground='grey',
                      font=('TkDefaultFont', 9)).pack(anchor='w')
            return

        df = self._summary_df
        metrics = [
            ('% Centroid',  'cpct_centroid',     '% frames in contact'),
            ('% Tailbase',  'cpct_tailbase',     '% frames in contact'),
            ('MBd Cent',    'mbout_centroid',    'mean bout dur (s)'),
            ('MBd Tail',    'mbout_tailbase',    'mean bout dur (s)'),
            ('ΔBrt Cent',   'dbrt_centroid',     'brt_contact - brt'),
            ('ΔBrt Tail',   'dbrt_tailbase',     'brt_contact - brt'),
        ]

        has_treatment = ('treatment' in df.columns
                          and df['treatment'].notna().any()
                          and df['treatment'].astype(str).ne('').any())
        if has_treatment:
            groups = [(t, df[df['treatment'] == t])
                      for t in df['treatment'].dropna().unique()
                      if str(t).strip()]
        else:
            groups = [('All', df)]

        # Header row
        ttk.Label(self._summary_frame, text="Metric",
                  font=('TkDefaultFont', 8, 'bold')).grid(
            row=0, column=0, sticky='w', padx=(0, 8))
        for ci, (gname, gdf) in enumerate(groups):
            ttk.Label(self._summary_frame,
                      text=f"{gname} (n={len(gdf)})",
                      font=('TkDefaultFont', 8, 'bold')).grid(
                row=0, column=ci + 1, sticky='e', padx=(4, 8))
        ttk.Separator(self._summary_frame, orient='horizontal').grid(
            row=1, column=0, columnspan=len(groups) + 1,
            sticky='ew', pady=2)

        for ri, (label, col, _ref) in enumerate(metrics, start=2):
            if col not in df.columns:
                continue
            ttk.Label(self._summary_frame, text=label,
                      font=('TkDefaultFont', 8)).grid(
                row=ri, column=0, sticky='w', padx=(0, 8))
            for ci, (_, gdf) in enumerate(groups):
                vals = pd.to_numeric(gdf[col], errors='coerce').dropna()
                if len(vals) == 0:
                    txt = '--'
                elif len(vals) == 1:
                    txt = f'{vals.iloc[0]:.2f}'
                else:
                    m = vals.mean()
                    sem = vals.std(ddof=1) / np.sqrt(len(vals))
                    txt = f'{m:.2f} ± {sem:.2f}'
                ttk.Label(self._summary_frame, text=txt,
                          font=('TkDefaultFont', 8)).grid(
                    row=ri, column=ci + 1, sticky='e', padx=(4, 8))

    # ═══════════════════════════════════════════════════════════════════════
    # Sessions / key file
    # ═══════════════════════════════════════════════════════════════════════

    def _project_folder(self) -> str:
        try:
            return self.app.current_project_folder.get()
        except Exception:
            return ''

    def _scan_sessions(self):
        folder = self._project_folder()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning(
                "No project",
                "Open a project first (the project folder feeds session "
                "discovery and the features/ cache lookup).", parent=self)
            return
        try:
            self._sessions = find_session_triplets(
                folder, require_labels=False)
        except Exception as e:
            self._sessions = []
            messagebox.showerror("Scan error", str(e), parent=self)
            return

        for item in self._sess_tree.get_children():
            self._sess_tree.delete(item)

        cache_root = os.path.join(folder, 'features')
        for sess in self._sessions:
            name = sess['session_name']
            subj = self._resolve_subject(name)
            cache_hit = self._has_any_cache(name, cache_root, folder, sess)
            self._sess_tree.insert('', 'end',
                                    values=(name, subj,
                                             '✓' if cache_hit else '✗'))

        n = len(self._sessions)
        n_cache = sum(1 for s in self._sessions
                      if self._has_any_cache(s['session_name'],
                                              cache_root, folder, s))
        self._sess_lbl.config(
            text=f'{n} session{"s" if n != 1 else ""}; '
                 f'{n_cache} have cached features')

        # Step 5: refresh the missing-brightness banner now that we
        # know what sessions are available + can sample one cache.
        self._refresh_extraction_banner()

    @staticmethod
    def _has_any_cache(session_name: str, cache_root: str,
                        project_folder: str, sess: dict) -> bool:
        if not _FEATURE_CACHE_AVAILABLE:
            return False
        video_dir = os.path.dirname(sess.get('video') or sess.get('dlc') or '')
        return FeatureCacheManager.find_any_cache(
            session_name, cache_root, video_dir,
            project_root=project_folder) is not None

    def _browse_key_file(self):
        path = filedialog.askopenfilename(
            title="Select key file",
            filetypes=[('Key files', '*.csv *.xlsx'),
                       ('All files', '*.*')], parent=self)
        if not path:
            return
        try:
            df = (pd.read_excel(path) if path.lower().endswith('.xlsx')
                  else pd.read_csv(path))
            if 'Subject' not in df.columns or 'Treatment' not in df.columns:
                raise ValueError("Key file must have 'Subject' and "
                                 "'Treatment' columns.")
            df['Subject'] = df['Subject'].astype(str)
            self._key_df = df
            self._key_file_var.set(path)
        except Exception as e:
            messagebox.showerror("Key file error", str(e), parent=self)

    def _resolve_subject(self, session_name: str) -> str:
        """Same 4-strategy fallback as GaitLimbTab."""
        stem = session_name
        if self._key_df is not None:
            tokens = stem.split('_')
            for subj in self._key_df['Subject']:
                if str(subj) in tokens:
                    return str(subj)
            for subj in self._key_df['Subject']:
                if f'_{subj}_' in f'_{stem}_':
                    return str(subj)
        pfx = self._prefix_var.get().strip()
        if pfx and stem.startswith(pfx):
            remainder = stem[len(pfx):]
            token = remainder.split('_')[0] if remainder else ''
            if token:
                return token
        if _extract_sid is not None:
            sid = _extract_sid(session_name)
            if sid:
                return str(sid)
        for token in stem.split('_'):
            if re.match(r'^\d{4}$', token):
                return token
        return stem

    def _get_treatment(self, subject: str) -> str:
        if self._key_df is None:
            return ''
        row = self._key_df[self._key_df['Subject'] == str(subject)]
        return str(row.iloc[0]['Treatment']) if not row.empty else ''

    # ═══════════════════════════════════════════════════════════════════════
    # Analysis
    # ═══════════════════════════════════════════════════════════════════════

    def _start_analysis(self):
        if self._fit_thread and self._fit_thread.is_alive():
            messagebox.showwarning("Busy", "Analysis is already running.",
                                    parent=self)
            return
        items = self._sess_tree.selection()
        if not items:
            messagebox.showwarning("No sessions",
                                    "Select at least one session.",
                                    parent=self)
            return
        selected = {self._sess_tree.item(i, 'values')[0] for i in items}
        sessions = [s for s in self._sessions
                    if s['session_name'] in selected]

        # Per-signal threshold dicts; the brightness-mode contact
        # decision picks the dict matching `signal`.
        signal_thresholds = {
            'raw':         {bp: float(self._raw_thresh_vars[bp].get())
                             for bp in _TARGETS},
            'dbrt':        {bp: float(self._brightness_thresh_vars[bp].get())
                             for bp in _TARGETS},
            'z':           {bp: float(self._z_thresh_vars[bp].get())
                             for bp in _TARGETS},
            'frac_bright': {bp: float(self._fb_thresh_vars[bp].get())
                             for bp in _TARGETS},
        }
        params = {
            'method':              self._method_var.get(),
            'signal':              self._signal_var.get(),
            'height_thresholds':   {bp: float(self._height_thresh_vars[bp].get())
                                     for bp in _TARGETS},
            'signal_thresholds':   signal_thresholds,
            'frac_bright_floor':   int(self._frac_bright_floor_var.get()),
            'min_bout_ms':         int(self._min_bout_ms_var.get()),
            'fallback_fps':        float(self._fallback_fps_var.get()),
            # Step-2 smoothing knobs
            'use_like_gate':       bool(self._use_like_gate_var.get()),
            'like_thresh':         float(self._like_thresh_var.get()),
            'like_gap_limit':      int(self._like_gap_limit_var.get()),
            'use_signal_smooth':   bool(self._use_signal_smooth_var.get()),
            'signal_smooth_window': int(self._signal_smooth_window_var.get()),
        }

        # Reset
        for item in self._res_tree.get_children():
            self._res_tree.delete(item)
        self._summary_df = None
        self._session_features = {}
        self._cancel_flag.clear()
        self._run_btn.config(state='disabled')
        self._cancel_btn.config(state='normal')
        self._export_btn.config(state='disabled')
        self._plot_btn.config(state='disabled')
        self._progress.config(maximum=max(len(sessions), 1), value=0)

        self._fit_thread = threading.Thread(
            target=self._analysis_thread,
            args=(sessions, params),
            daemon=True)
        self._fit_thread.start()

    def _cancel_analysis(self):
        self._cancel_flag.set()

    def _analysis_thread(self, sessions, params):
        rows = []
        bout_rows = []
        session_bouts = {}
        project_folder = self._project_folder()
        cache_root = os.path.join(project_folder, 'features')
        for sess in sessions:
            if self._cancel_flag.is_set():
                break
            name = sess['session_name']
            self._update_sub_progress(
                f"Analyzing {name}…")
            try:
                metrics, X, bouts = self._analyze_session(
                    sess, params, cache_root, project_folder)
                if metrics is not None:
                    subj = self._resolve_subject(name)
                    trt = self._get_treatment(subj)
                    rows.append({'session': name, 'subject': subj,
                                  'treatment': trt, **metrics})
                    if X is not None:
                        # Keep only the columns used for plotting; the
                        # full X can be 100s of MB.
                        keep_cols = [c for c in X.columns if (
                            c.startswith('Pix_') or c.endswith('_Height')
                            or c.endswith('_ContactState'))]
                        self._session_features[name] = X[keep_cols].copy()
                    if bouts:
                        session_bouts[name] = bouts
                        for bp, bout_list in bouts.items():
                            for b in bout_list:
                                bout_rows.append({
                                    'session':   name,
                                    'subject':   subj,
                                    'treatment': trt,
                                    'keypoint':  bp,
                                    **b,
                                })
            except Exception as e:
                rows.append({'session': name, 'subject': '',
                              'treatment': '', 'error': str(e)})
            try:
                self.app.root.after(0, lambda: self._progress.step(1))
            except tk.TclError:
                pass

        self._session_bouts = session_bouts
        try:
            self.app.root.after(0, self._on_analysis_complete,
                                  rows, bout_rows)
        except tk.TclError:
            pass

    def _analyze_session(self, sess: dict, params: dict,
                          cache_root: str, project_folder: str):
        """Load cached features and compute body-contact metrics for one session.

        Returns (metrics_dict, X_features_df) or (None, None) on failure.
        """
        if not _FEATURE_CACHE_AVAILABLE:
            raise RuntimeError("FeatureCacheManager not available")

        name = sess['session_name']
        video_dir = os.path.dirname(sess.get('video') or sess.get('dlc') or '')
        cache_path = FeatureCacheManager.find_any_cache(
            name, cache_root, video_dir, project_root=project_folder)
        if cache_path is None:
            raise FileNotFoundError(
                f"No cached feature pkl for '{name}'. Re-run feature "
                f"extraction with centroid + tailbase in bp_pixbrt_list.")

        cache_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
        self._update_sub_progress(
            f"Loading cache for {name} ({cache_size_mb:.0f} MB)…")
        X = pd.read_pickle(cache_path)
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"{cache_path}: expected DataFrame, got "
                              f"{type(X).__name__}")

        # FPS — try video metadata, fall back to user setting
        fps = self._lookup_fps(sess, params['fallback_fps'])

        method = params['method']
        signal = params.get('signal', 'dbrt')
        height_thr = params['height_thresholds']
        sig_thr_table = params.get(
            'signal_thresholds',
            {signal: params.get('brt_thresholds', {})})
        sig_thr = sig_thr_table.get(signal, {})
        min_bout_frames = int(round(params['min_bout_ms'] / 1000.0 * fps))

        # Step-2 smoothing knobs. Likelihood gating reads per-keypoint
        # `<bp>_prob` from the DLC h5 (since cached features don't keep
        # the raw probabilities). Loaded once per session, even if the
        # gate is off — cheap and avoids the load if the pkl is huge.
        use_gate         = bool(params.get('use_like_gate', False))
        like_thresh      = float(params.get('like_thresh', 0.6))
        like_gap_limit   = int(params.get('like_gap_limit', 10))
        use_smooth       = bool(params.get('use_signal_smooth', False))
        smooth_window    = int(params.get('signal_smooth_window', 1))
        if not use_smooth:
            smooth_window = 1

        prob_lookup = {}
        if use_gate and _POSE_OK and sess.get('dlc') and \
                os.path.isfile(sess['dlc']):
            try:
                _ext = PoseFeatureExtractor(list(_TARGETS))
                _dlc = _ext.load_dlc_data(sess['dlc'])
                _, _, _bp_prob = _ext.get_bodypart_coords(_dlc)
                for bp in _TARGETS:
                    pcol = next((c for c in _bp_prob.columns
                                  if bp.lower() in c.lower()), None)
                    if pcol is not None:
                        prob_lookup[bp] = _bp_prob[pcol].values
            except Exception:
                # Gate without prob_array still works as a no-op
                # (the `prob_array is None` path in _smooth_signal_series).
                prob_lookup = {}

        metrics = {'fps': round(fps, 2), 'n_frames': len(X)}
        bouts_per_keypoint = {bp: [] for bp in _TARGETS}
        for bp in _TARGETS:
            mask = self._compute_contact_mask(
                X, bp, method,
                height_thr[bp],
                sig_thr.get(bp, 0.0),
                signal=signal,
                prob_array=prob_lookup.get(bp) if use_gate else None,
                like_thresh=like_thresh,
                like_gap_limit=like_gap_limit,
                smooth_window=smooth_window)
            if mask is None:
                # Required column missing — emit NaNs but keep going so
                # the user can see which keypoint is missing.
                for k in (f'cpct_{bp}', f'nbouts_{bp}', f'mbout_{bp}',
                           f'brt_{bp}', f'brt_contact_{bp}',
                           f'dbrt_{bp}'):
                    metrics[k] = float('nan')
                continue
            mask = self._debounce(mask, min_bout_frames)
            n_bouts, mean_bout_dur = self._bout_stats(mask, fps)

            # Per-bout list — uses the same smoothed signal that drove
            # the mask, plus the raw <bp>_Height column, so the stats
            # in the bouts CSV match what the threshold actually fired
            # on. Empty list when the brightness signal isn't available
            # (e.g. height-only mode without cached features).
            sig_for_bouts = self._smooth_signal_series(
                self._build_brightness_series(X, bp, signal),
                prob_array=prob_lookup.get(bp) if use_gate else None,
                like_thresh=like_thresh,
                like_gap_limit=like_gap_limit,
                smooth_window=smooth_window)
            sig_arr = (sig_for_bouts.values
                        if sig_for_bouts is not None else None)
            h_arr = (X[f'{bp}_Height'].values
                      if f'{bp}_Height' in X.columns else None)
            bouts_per_keypoint[bp] = self._extract_bouts(
                mask, fps,
                signal_array=sig_arr, height_array=h_arr)

            pix_col = f'Pix_{bp}'
            pix = X[pix_col].astype(float) if pix_col in X.columns else None

            mean_brt = (float(pix.mean())
                         if pix is not None else float('nan'))
            mean_brt_contact = (float(pix[mask].mean())
                                 if pix is not None and mask.any()
                                 else float('nan'))
            # ΔBrt = pressure-driven shift = brightness during contact
            # minus whole-session brightness. Positive = pressed harder
            # than baseline. NaN if either side is missing.
            if (mean_brt == mean_brt and mean_brt_contact == mean_brt_contact):
                dbrt = mean_brt_contact - mean_brt
            else:
                dbrt = float('nan')

            metrics[f'cpct_{bp}']    = round(float(mask.mean()) * 100, 2)
            metrics[f'nbouts_{bp}']  = int(n_bouts)
            metrics[f'mbout_{bp}']   = round(float(mean_bout_dur), 3)
            metrics[f'brt_{bp}']     = (round(mean_brt, 2)
                                         if mean_brt == mean_brt
                                         else float('nan'))
            metrics[f'brt_contact_{bp}'] = (
                round(mean_brt_contact, 2)
                if mean_brt_contact == mean_brt_contact else float('nan'))
            metrics[f'dbrt_{bp}'] = (round(dbrt, 2)
                                      if dbrt == dbrt else float('nan'))

        return metrics, X, bouts_per_keypoint

    @staticmethod
    def _build_brightness_series(X: pd.DataFrame, bp: str,
                                   signal: str) -> pd.Series:
        """Return the per-frame brightness signal as a Series, or None
        when the required cache column is missing.

        Computed BEFORE smoothing — the caller is responsible for
        applying likelihood-gating + rolling-median if requested.
        """
        if signal == 'raw':
            col = f'Pix_{bp}'
            if col in X.columns:
                return X[col].astype(float).copy()
        elif signal == 'dbrt':
            col = f'Pix_baseline_sub_{bp}'
            if col in X.columns:
                return X[col].astype(float).copy()
        elif signal == 'z':
            col = f'Pix_{bp}'
            if col in X.columns:
                v = X[col].astype(float)
                std = float(v.std())
                if std > 1e-12:
                    return ((v - v.mean()) / std).copy()
        elif signal == 'frac_bright':
            # Preview-only — needs per-frame ROI pixel counts.
            return None
        return None

    @staticmethod
    def _smooth_signal_series(series: pd.Series,
                                prob_array=None,
                                like_thresh: float = 0.6,
                                like_gap_limit: int = 10,
                                smooth_window: int = 1) -> pd.Series:
        """Apply likelihood gating + rolling-median to a signal Series.

        Pipeline (each layer is conditional on its kwarg):
        1. NaN-out frames where ``prob_array < like_thresh`` (when
           ``prob_array`` is provided).
        2. Linear-interpolate the resulting NaN gaps up to
           ``like_gap_limit`` frames in length.
        3. Rolling-median centered on each frame, window
           ``smooth_window`` (set 1 to disable). Centered windows
           don't shift bout edges.

        Returns the smoothed Series. Original is not modified.
        """
        if series is None:
            return None
        s = series.copy()
        if prob_array is not None:
            n = min(len(s), len(prob_array))
            low = pd.Series(prob_array[:n] < like_thresh,
                              index=s.index[:n])
            s.loc[s.index[:n]] = s.loc[s.index[:n]].where(~low.values)
            if like_gap_limit and like_gap_limit > 0:
                s = s.interpolate(method='linear',
                                    limit=int(like_gap_limit),
                                    limit_direction='both')
        if smooth_window and smooth_window > 1:
            s = s.rolling(int(smooth_window),
                            center=True, min_periods=1).median()
        return s

    @staticmethod
    def _compute_contact_mask(X: pd.DataFrame, bp: str, method: str,
                                height_thr: float, sig_thr: float,
                                signal: str = 'dbrt',
                                prob_array=None,
                                like_thresh: float = 0.6,
                                like_gap_limit: int = 10,
                                smooth_window: int = 1):
        """Return a boolean numpy array of the contact mask, or None when
        required columns are missing for the chosen method.

        Smoothing knobs (step 2): if ``prob_array`` is provided, low-
        confidence frames get NaN'd and short gaps interpolated.
        ``smooth_window`` applies a centered rolling-median to the
        brightness signal before thresholding (set 1 to disable).
        """
        height_col = f'{bp}_Height'
        h_mask = (X[height_col].values < height_thr
                   if height_col in X.columns else None)

        b_mask = None
        sig_series = BodyContactTab._build_brightness_series(X, bp, signal)
        if sig_series is not None:
            sig_series = BodyContactTab._smooth_signal_series(
                sig_series,
                prob_array=prob_array,
                like_thresh=like_thresh,
                like_gap_limit=like_gap_limit,
                smooth_window=smooth_window)
            # `>` against NaN returns False, so gated/uninterpolated
            # frames are correctly classified as "not in contact"
            # rather than triggering a positive on stale data.
            b_mask = (sig_series.values > sig_thr)

        if method == 'height':
            return h_mask
        if method == 'brightness':
            return b_mask
        if method == 'combined':
            if h_mask is None or b_mask is None:
                return None
            return h_mask & b_mask
        return None

    @staticmethod
    def _debounce(mask: np.ndarray, min_bout_frames: int) -> np.ndarray:
        """Drop bouts shorter than `min_bout_frames`. Operates on the
        True runs only; gaps shorter than min_bout_frames are NOT
        bridged (matches gait tab semantics for stance bouts).
        """
        if min_bout_frames <= 1 or mask.size == 0:
            return mask.astype(bool)
        m = mask.astype(bool).copy()
        # Find runs of True
        diff = np.diff(np.concatenate(([0], m.astype(int), [0])))
        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            if (e - s) < min_bout_frames:
                m[s:e] = False
        return m

    @staticmethod
    def _bout_stats(mask: np.ndarray, fps: float) -> tuple:
        """(n_bouts, mean_bout_dur_seconds) from a boolean contact mask."""
        if mask.size == 0 or not mask.any():
            return 0, 0.0
        diff = np.diff(np.concatenate(([0], mask.astype(int), [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        durs = (ends - starts) / max(fps, 1.0)
        return len(durs), float(durs.mean()) if len(durs) else 0.0

    @staticmethod
    def _extract_bouts(mask: np.ndarray, fps: float,
                         signal_array=None,
                         height_array=None) -> list:
        """Return per-bout dicts from a boolean contact mask.

        Each dict: bout_idx (1-based), start_frame, end_frame (exclusive),
        start_s, end_s, duration_s, peak_signal, mean_signal, mean_height.

        ``signal_array`` should be the post-smoothing brightness signal
        used for thresholding; ``height_array`` is the per-frame
        ``<bp>_Height`` for the geometric proxy. Either may be None if
        unavailable; the corresponding stat is then NaN.
        """
        out = []
        if mask is None or mask.size == 0 or not mask.any():
            return out
        diff = np.diff(np.concatenate(([0], mask.astype(int), [0])))
        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0]
        for i, (s, e) in enumerate(zip(starts, ends), start=1):
            sig_slice = (signal_array[s:e]
                          if signal_array is not None else None)
            h_slice = (height_array[s:e]
                        if height_array is not None else None)
            peak = (float(np.nanmax(sig_slice))
                     if sig_slice is not None and len(sig_slice)
                     else float('nan'))
            mean_sig = (float(np.nanmean(sig_slice))
                         if sig_slice is not None and len(sig_slice)
                         else float('nan'))
            mean_h = (float(np.nanmean(h_slice))
                       if h_slice is not None and len(h_slice)
                       else float('nan'))
            out.append({
                'bout_idx':    i,
                'start_frame': int(s),
                'end_frame':   int(e),
                'start_s':     round(float(s) / max(fps, 1.0), 3),
                'end_s':       round(float(e) / max(fps, 1.0), 3),
                'duration_s':  round(float(e - s) / max(fps, 1.0), 3),
                'peak_signal': (round(peak, 3)
                                  if peak == peak else float('nan')),
                'mean_signal': (round(mean_sig, 3)
                                  if mean_sig == mean_sig else float('nan')),
                'mean_height': (round(mean_h, 3)
                                  if mean_h == mean_h else float('nan')),
            })
        return out

    def _lookup_fps(self, sess: dict, fallback: float) -> float:
        """Read FPS from the video file via OpenCV; fall back if unavailable."""
        video = sess.get('video') or sess.get('video_path')
        if video and os.path.isfile(video):
            try:
                import cv2
                cap = cv2.VideoCapture(video)
                f = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if f and f > 0:
                    return float(f)
            except Exception:
                pass
        return float(fallback)

    def _on_analysis_complete(self, rows: list, bout_rows: list = None):
        self._cancel_btn.config(state='disabled')
        self._run_btn.config(state='normal')
        self._update_sub_progress("")
        if not rows:
            messagebox.showinfo("No results",
                                 "No sessions produced metrics.",
                                 parent=self)
            self._log_ui("Analysis produced no rows.")
            return
        df = pd.DataFrame(rows)
        # CSV export keeps every column (whole-session brightness,
        # n_bouts, fps, frame counts) — only the treeview is trimmed.
        cols = ['session', 'subject', 'treatment', 'fps', 'n_frames']
        for bp in _TARGETS:
            cols += [f'cpct_{bp}', f'nbouts_{bp}', f'mbout_{bp}',
                     f'brt_{bp}', f'brt_contact_{bp}', f'dbrt_{bp}']
        cols += [c for c in df.columns if c not in cols]
        df = df.reindex(columns=cols)
        self._summary_df = df

        bout_rows = bout_rows or []
        if bout_rows:
            self._bouts_df = pd.DataFrame(bout_rows)
            self._log_ui(
                f"Analysis complete: {len(df)} session(s), "
                f"{len(self._bouts_df)} bout(s) detected.")
        else:
            self._bouts_df = None
            self._log_ui(
                f"Analysis complete: {len(df)} session(s) "
                "(no bouts detected).")

        for item in self._res_tree.get_children():
            self._res_tree.delete(item)
        for _, r in df.iterrows():
            vals = tuple(self._fmt(r.get(c)) for c in self._RES_COLS)
            self._res_tree.insert('', 'end', values=vals)

        self._refresh_summary_panel()
        self._export_btn.config(state='normal')
        self._plot_btn.config(state='normal')
        self._graphs_btn.config(state='normal')
        if hasattr(self, '_export_bouts_btn'):
            self._export_bouts_btn.config(
                state='normal' if self._bouts_df is not None else 'disabled')

    @staticmethod
    def _fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return '—'
        return v if isinstance(v, str) else str(v)

    # ═══════════════════════════════════════════════════════════════════════
    # Export
    # ═══════════════════════════════════════════════════════════════════════

    def _export_summary(self):
        if self._summary_df is None or self._summary_df.empty:
            return
        path = filedialog.asksaveasfilename(
            title="Save Summary CSV",
            defaultextension='.csv',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
            parent=self)
        if not path:
            return
        try:
            self._summary_df.to_csv(path, index=False)
            self._write_meta_sidecar(path)
            messagebox.showinfo(
                "Exported",
                f"Wrote {os.path.basename(path)}\n"
                f"+ reproducibility sidecar "
                f"{os.path.basename(path)}.meta.json",
                parent=self)
        except Exception as e:
            messagebox.showerror("Export error", str(e), parent=self)

    def _export_bouts(self):
        """Save the per-bout DataFrame plus a meta-json sidecar.

        Each row is one detected contact bout for one keypoint in one
        session, with start/end frames, duration, peak/mean signal,
        and mean height. Sort the resulting CSV by `peak_signal` to
        eyeball the strongest belly-press events first.
        """
        if self._bouts_df is None or self._bouts_df.empty:
            messagebox.showinfo(
                "No bouts",
                "No bouts to export — either no contact was detected "
                "or the analysis hasn't run on a brightness mode that "
                "produces a signal trace.", parent=self)
            return
        path = filedialog.asksaveasfilename(
            title="Save Bouts CSV",
            defaultextension='.csv',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
            parent=self)
        if not path:
            return
        try:
            self._bouts_df.to_csv(path, index=False)
            self._write_meta_sidecar(path, kind='bouts')
            messagebox.showinfo(
                "Exported",
                f"Wrote {os.path.basename(path)}\n"
                f"({len(self._bouts_df)} bouts across "
                f"{self._bouts_df['session'].nunique()} sessions)\n"
                f"+ reproducibility sidecar "
                f"{os.path.basename(path)}.meta.json",
                parent=self)
        except Exception as e:
            messagebox.showerror("Export error", str(e), parent=self)

    def _write_meta_sidecar(self, csv_path: str, kind: str = 'summary'):
        import json
        if kind == 'bouts':
            row_src = self._bouts_df
        else:
            row_src = self._summary_df
        meta = {
            'csv_path':            os.path.abspath(csv_path),
            'csv_kind':            kind,
            'csv_row_count':       (len(row_src) if row_src is not None
                                     else 0),
            'git_sha':             get_git_sha(),
            'export_timestamp_utc':
                datetime.utcnow().isoformat(timespec='seconds') + 'Z',
            'export_timestamp_local':
                datetime.now().isoformat(timespec='seconds'),
            'pixelpaws_module':    'body_contact_tab.py',
            'tracked_keypoints':   list(_TARGETS),
            'method':              self._method_var.get(),
            'signal':              self._signal_var.get(),
            'frac_bright_floor':   int(self._frac_bright_floor_var.get()),
            'height_thresholds':   {bp: int(self._height_thresh_vars[bp].get())
                                     for bp in _TARGETS},
            'signal_thresholds': {
                'raw':   {bp: float(self._raw_thresh_vars[bp].get())
                           for bp in _TARGETS},
                'dbrt':  {bp: float(self._brightness_thresh_vars[bp].get())
                           for bp in _TARGETS},
                'z':     {bp: float(self._z_thresh_vars[bp].get())
                           for bp in _TARGETS},
                'frac_bright': {bp: float(self._fb_thresh_vars[bp].get())
                                  for bp in _TARGETS},
            },
            'min_bout_ms':         int(self._min_bout_ms_var.get()),
            'fallback_fps':        float(self._fallback_fps_var.get()),
            'key_file':            self._key_file_var.get() or None,
        }
        try:
            with open(csv_path + '.meta.json', 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2, default=str)
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════════════════
    # Live preview (mirrors gait_limb_tab._open_brightness_preview pattern)
    # ═══════════════════════════════════════════════════════════════════════

    def _open_preview(self):
        """Open a frame-scrubbing preview window with centroid + tailbase
        ROIs drawn live on the video, contact state decided per the
        currently selected method/thresholds.

        Mirrors `gait_limb_tab._open_brightness_preview` (line 1038)
        but tracks the body-contact target list instead of paws and
        excludes the brightness-weighted contact blend (we use the
        cleaner method radio: height / brightness / combined).
        """
        if not _CV2_OK:
            messagebox.showerror(
                "Missing dependency",
                "OpenCV (cv2) is required for the live preview.",
                parent=self)
            return
        if not _POSE_OK:
            messagebox.showerror(
                "Missing dependency",
                "pose_features.PoseFeatureExtractor not available.",
                parent=self)
            return

        # Pick the first selected session that has both a video and a
        # DLC h5; fall back to "any" session if none is selected.
        selected = [self._sess_tree.item(i, 'values')[0]
                    for i in self._sess_tree.selection()]
        candidates = [s for s in self._sessions
                      if (not selected or s['session_name'] in selected)
                      and (s.get('video') and os.path.isfile(s['video']))
                      and (s.get('dlc')   and os.path.isfile(s['dlc']))]
        if not candidates:
            messagebox.showwarning(
                "No session",
                "Select a session in the list that has both a video "
                "and a DLC file.", parent=self)
            return

        sess = candidates[0]
        video_path = sess['video']
        dlc_path   = sess['dlc']

        # Load DLC + Height frame so the preview can show the per-frame
        # height alongside the brightness readout.
        try:
            ext = PoseFeatureExtractor(list(_TARGETS))
            dlc_df = ext.load_dlc_data(dlc_path)
            bp_xcord, bp_ycord, bp_prob = ext.get_bodypart_coords(dlc_df)
        except Exception as e:
            messagebox.showerror("DLC error", str(e), parent=self)
            return
        try:
            height_df = ext.calculate_paw_height(
                bp_xcord, bp_ycord, window=500)
        except Exception:
            height_df = None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Video error",
                                  f"Cannot open:\n{video_path}",
                                  parent=self)
            return
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # No local var copies — preview reads/writes the main panel
        # vars directly so changing a threshold in the preview takes
        # effect for analysis runs too. Simpler state, less to keep
        # in sync. Per-signal threshold dicts are mapped here for
        # convenience in the render loop below.
        signal_thresh_vars = {
            'raw':         self._raw_thresh_vars,
            'dbrt':        self._brightness_thresh_vars,
            'z':           self._z_thresh_vars,
            'frac_bright': self._fb_thresh_vars,
        }

        win = tk.Toplevel(self)
        win.title(f"Body-Contact Preview — {sess['session_name']}")
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        w, h = int(sw * 0.65), int(sh * 0.70)
        win.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")
        win.protocol('WM_DELETE_WINDOW',
                     lambda: (cap.release(), win.destroy()))

        # ── Layout ─────────────────────────────────────────────────
        # Left  — video canvas (fills remaining space)
        # Right — compact controls panel, narrower than v1.
        ctrl = ttk.Frame(win, padding=6)
        ctrl.pack(side='right', fill='y')

        video_canvas = tk.Canvas(win, bg='black', width=520, height=420)
        video_canvas.pack(side='left', fill='both', expand=True,
                           padx=4, pady=4)

        # ── Frame slider (tk.Scale w/ resolution=1 → no float drift) ──
        f_lf = ttk.LabelFrame(ctrl, text="Frame", padding=4)
        f_lf.pack(fill='x', pady=(0, 4))
        frame_var = tk.IntVar(value=0)
        f_top = ttk.Frame(f_lf)
        f_top.pack(fill='x')
        ttk.Spinbox(f_top, from_=0, to=max(n_frames - 1, 0),
                     textvariable=frame_var, width=8).pack(side='left')
        ttk.Label(f_top, text=f" / {n_frames - 1}",
                   font=('TkDefaultFont', 8)).pack(side='left', padx=(2, 0))
        # tk.Scale (not ttk.Scale) so resolution=1 snaps to integers.
        # ttk.Scale has no resolution kwarg and writes floats into
        # IntVar via traces, producing the 5976.617647-style display.
        tk.Scale(f_lf, from_=0, to=max(n_frames - 1, 0),
                  orient='horizontal', resolution=1,
                  variable=frame_var, showvalue=False,
                  highlightthickness=0).pack(fill='x', pady=(2, 0))

        # ── Bout navigation (active only after analysis) ──────────
        # Pulls the bout list for THIS session that the analysis run
        # produced. If you change thresholds in the panel, run the
        # analysis again to refresh the bouts before navigating.
        bout_nav_row = ttk.Frame(f_lf)
        bout_nav_row.pack(fill='x', pady=(2, 0))
        sess_bout_dict = self._session_bouts.get(sess['session_name'], {})
        all_bouts_sorted = []
        for bp in _TARGETS:
            for b in sess_bout_dict.get(bp, []):
                all_bouts_sorted.append((bp, b))
        all_bouts_sorted.sort(key=lambda kb: kb[1]['start_frame'])

        def _jump_to_bout(direction: int):
            if not all_bouts_sorted:
                return
            cur = frame_var.get()
            ordered_starts = [b['start_frame']
                                for _, b in all_bouts_sorted]
            if direction > 0:
                next_starts = [s for s in ordered_starts if s > cur]
                target = (next_starts[0] if next_starts
                            else ordered_starts[-1])
            else:
                prev_starts = [s for s in ordered_starts if s < cur]
                target = (prev_starts[-1] if prev_starts
                            else ordered_starts[0])
            frame_var.set(int(target))

        ttk.Button(bout_nav_row, text="◀ Prev bout", width=10,
                    command=lambda: _jump_to_bout(-1)).pack(side='left',
                                                              padx=2)
        ttk.Button(bout_nav_row, text="Next bout ▶", width=10,
                    command=lambda: _jump_to_bout(+1)).pack(side='left',
                                                              padx=2)
        bout_count_lbl = ttk.Label(
            bout_nav_row,
            text=(f"  {len(all_bouts_sorted)} bouts"
                  if all_bouts_sorted
                  else "  (run analysis to populate)"),
            font=('TkDefaultFont', 8), foreground='grey')
        bout_count_lbl.pack(side='left', padx=(4, 0))

        # ── Method + signal radios ─────────────────────────────────
        ms_lf = ttk.LabelFrame(ctrl, text="Detection", padding=4)
        ms_lf.pack(fill='x', pady=(0, 4))
        ttk.Label(ms_lf, text="Method:",
                  font=('TkDefaultFont', 8, 'bold')).pack(anchor='w')
        m_row = ttk.Frame(ms_lf)
        m_row.pack(fill='x')
        for m in _METHODS:
            ttk.Radiobutton(m_row, text=m, variable=self._method_var,
                             value=m).pack(side='left', padx=4)
        ttk.Label(ms_lf, text="Signal:",
                  font=('TkDefaultFont', 8, 'bold')).pack(anchor='w',
                                                            pady=(4, 0))
        for s in _SIGNALS:
            ttk.Radiobutton(ms_lf, text=_SIGNAL_LABELS[s],
                             variable=self._signal_var,
                             value=s).pack(anchor='w', padx=4)

        # ── Per-keypoint thresholds — a row per (signal × Height/ROI) ──
        # Active signal's brightness threshold is highlighted bold; the
        # others are grey so the user knows which one is in effect.
        thr_lf = ttk.LabelFrame(ctrl, text="Thresholds", padding=4)
        thr_lf.pack(fill='x', pady=(0, 4))
        # Header row
        for ci, bp in enumerate(_TARGETS, start=1):
            ttk.Label(thr_lf, text=bp,
                      font=('TkDefaultFont', 8, 'bold')).grid(
                row=0, column=ci, padx=4, sticky='w')
        # Rows: Height, raw, ΔBrt, z, fb, ROI
        rows = [
            ('Height (px)',  self._height_thresh_vars,        5,  200, 5),
            ('raw thr',      self._raw_thresh_vars,           0,  255, 1.0),
            ('ΔBrt thr',     self._brightness_thresh_vars,    0,  200, 0.5),
            ('z thr',        self._z_thresh_vars,            -5,    8, 0.1),
            ('fb thr',       self._fb_thresh_vars,            0,    1, 0.05),
            ('ROI half (px)', self._roi_half_vars,            5,  200, 5),
        ]
        for ri, (label, vars_dict, lo, hi, inc) in enumerate(rows, start=1):
            ttk.Label(thr_lf, text=label,
                      font=('TkDefaultFont', 8)).grid(
                row=ri, column=0, sticky='w', padx=(0, 4), pady=1)
            for ci, bp in enumerate(_TARGETS, start=1):
                ttk.Spinbox(thr_lf, from_=lo, to=hi, increment=inc,
                             width=6,
                             textvariable=vars_dict[bp]).grid(
                    row=ri, column=ci, padx=4, pady=1)

        # ── Live readouts (one row per keypoint, monospace) ───────
        # Shows ALL FOUR brightness signals so the user can compare
        # raw / ΔBrt / z / frac_bright at a glance and pick the right
        # signal/threshold combination for their rig.
        readout_lf = ttk.LabelFrame(ctrl, text="Per-frame", padding=4)
        readout_lf.pack(fill='x', pady=(0, 4))
        readout_lbls = {}
        for bp in _TARGETS:
            readout_lbls[bp] = ttk.Label(
                readout_lf, text=f"{bp}: —", anchor='w',
                font=('Consolas', 8), justify='left')
            readout_lbls[bp].pack(fill='x', pady=1)
        cache_hint_lbl = ttk.Label(
            readout_lf, text='', foreground='grey',
            font=('TkDefaultFont', 8, 'italic'))
        cache_hint_lbl.pack(anchor='w', pady=(2, 0))

        # ── Render: ROI overlays + contact decisions ───────────────
        # All four signals are computed per-frame and shown in the
        # readout. The contact decision uses the user-selected signal
        # (self._signal_var) against its dedicated threshold dict.
        _palette = {'centroid': (0, 200, 255),   # cyan-ish
                     'tailbase': (255, 100, 200)}  # magenta-ish

        cached_X = self._session_features.get(sess['session_name'])

        # Pre-compute per-keypoint session stats for z-score; cheap.
        session_stats = {bp: {'mean': None, 'std': None} for bp in _TARGETS}
        if cached_X is not None:
            for bp in _TARGETS:
                col = f'Pix_{bp}'
                if col in cached_X.columns:
                    v = cached_X[col].astype(float).values
                    session_stats[bp]['mean'] = float(v.mean())
                    sigma = float(v.std())
                    session_stats[bp]['std'] = sigma if sigma > 1e-12 else None

        if cached_X is not None:
            cache_hint_lbl.config(
                text=("✓ cached features loaded — ΔBrt, z use cache; "
                      "raw, frac_bright recomputed live"))
        else:
            cache_hint_lbl.config(
                text=("⚠ no cached features for this session — "
                      "ΔBrt and z will read '—'; raw and frac_bright "
                      "still work"))

        _after_id = [None]

        def _render(*_):
            if _after_id[0]:
                win.after_cancel(_after_id[0])
            _after_id[0] = win.after(50, _do_render)

        def _safe_int(var, default=0):
            try:
                return int(var.get())
            except Exception:
                return default

        def _safe_float(var, default=0.0):
            try:
                return float(var.get())
            except Exception:
                return default

        def _fmt_signed(v, w=6, prec=2):
            return f'{v:+{w}.{prec}f}' if (v is not None and v == v) \
                                          else f'{"—":>{w}}'

        def _fmt_pos(v, w=6, prec=2):
            return f'{v:{w}.{prec}f}' if (v is not None and v == v) \
                                          else f'{"—":>{w}}'

        def _do_render():
            fi = max(0, min(_safe_int(frame_var, 0), n_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                return
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            vis = frame.copy()
            method = self._method_var.get()
            signal = self._signal_var.get()
            fb_floor = _safe_int(self._frac_bright_floor_var, 180)

            for bp in _TARGETS:
                col_x = next((c for c in bp_xcord.columns
                               if bp.lower() in c.lower()), None)
                col_y = next((c for c in bp_ycord.columns
                               if bp.lower() in c.lower()), None)
                if (col_x is None or col_y is None
                        or fi >= len(bp_xcord)):
                    readout_lbls[bp].config(text=f"{bp:<8s}: (not in DLC)")
                    continue

                bx = int(bp_xcord[col_x].iloc[fi])
                by = int(bp_ycord[col_y].iloc[fi])
                rh = _safe_int(self._roi_half_vars[bp], 30)
                fh, fw = frame.shape[:2]
                x1 = max(0, bx - rh); x2 = min(fw, bx + rh)
                y1 = max(0, by - rh); y2 = min(fh, by + rh)

                # === Compute all four signals per-frame ===
                # raw: ROI mean from current video frame (no cache needed)
                roi = gray[y1:y2, x1:x2]
                if roi.size == 0:
                    raw_val = float('nan')
                    fb_val  = float('nan')
                else:
                    raw_val = float(roi.mean())
                    fb_val  = float(np.mean(roi > fb_floor))

                # dbrt: from cached Pix_baseline_sub_<bp>
                dbrt_val = float('nan')
                if cached_X is not None:
                    bs_col = f'Pix_baseline_sub_{bp}'
                    if bs_col in cached_X.columns and fi < len(cached_X):
                        dbrt_val = float(cached_X[bs_col].iloc[fi])

                # z: (raw - session_mean) / session_std (computed from
                # cached Pix_<bp>; falls back to NaN if no cache).
                z_val = float('nan')
                stats = session_stats[bp]
                if stats['mean'] is not None and stats['std'] is not None:
                    # Use cached Pix_<bp> at this frame, not the live
                    # ROI value, so z matches what the analysis path
                    # produces.
                    if cached_X is not None:
                        pcol = f'Pix_{bp}'
                        if pcol in cached_X.columns and fi < len(cached_X):
                            cached_raw = float(cached_X[pcol].iloc[fi])
                            z_val = (cached_raw - stats['mean']) / stats['std']

                # === Per-frame Height (geometric proxy) ===
                h_val = None
                if height_df is not None:
                    hcol = next((c for c in height_df.columns
                                  if bp.lower() in c.lower()), None)
                    if hcol and fi < len(height_df):
                        h_val = float(height_df[hcol].iloc[fi])

                # === Contact decision ===
                h_thr = _safe_int(self._height_thresh_vars[bp], 40)
                sig_value = {'raw': raw_val, 'dbrt': dbrt_val,
                             'z': z_val, 'frac_bright': fb_val}[signal]
                sig_thr = _safe_float(
                    signal_thresh_vars[signal][bp], 0.0)
                h_hit = (h_val is not None and h_val < h_thr)
                b_hit = (sig_value == sig_value
                          and sig_value > sig_thr)

                if method == 'height':
                    in_contact = h_hit
                elif method == 'brightness':
                    in_contact = b_hit
                else:  # combined
                    in_contact = h_hit and b_hit

                # === Draw ROI box ===
                color = _palette.get(bp, (255, 255, 255))
                if in_contact:
                    overlay = vis.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2),
                                   color, -1)
                    cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color,
                               3 if in_contact else 1)
                cv2.circle(vis, (bx, by), 4, color, -1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th_), _ = cv2.getTextSize(bp, font, 0.55, 1)
                cv2.rectangle(vis, (x1, max(0, y1 - th_ - 4)),
                                (x1 + tw + 4, y1), (0, 0, 0), -1)
                cv2.putText(vis, bp, (x1 + 2, y1 - 2), font, 0.55,
                              color, 1)

                # === Readout: all four signals stacked ===
                h_str = (f'h={h_val:5.1f}' if h_val is not None
                          else f'h={"?":>5}')
                contact_str = '  ✓' if in_contact else '   '
                readout_lbls[bp].config(
                    text=(f"{bp:<8s} {h_str}  "
                          f"raw={_fmt_pos(raw_val)}  "
                          f"ΔBrt={_fmt_signed(dbrt_val)}  "
                          f"z={_fmt_signed(z_val)}  "
                          f"fb={_fmt_pos(fb_val, 5, 2)}"
                          f"{contact_str}"))

            # HUD — frame / method / signal + per-keypoint bout context
            # when the current frame is inside a detected bout.
            hud_lines = [f"frame {fi}/{n_frames - 1}  "
                          f"method={method}  signal={signal}"]
            for bp in _TARGETS:
                bouts_bp = sess_bout_dict.get(bp, [])
                if not bouts_bp:
                    continue
                hit = next((b for b in bouts_bp
                             if b['start_frame'] <= fi < b['end_frame']),
                            None)
                if hit is not None:
                    bidx = hit['bout_idx']
                    btot = len(bouts_bp)
                    hud_lines.append(
                        f"{bp}: bout {bidx}/{btot}  "
                        f"dur={hit['duration_s']:.2f}s  "
                        f"peak={hit['peak_signal']:+.2f}")
            for li, line in enumerate(hud_lines):
                cv2.putText(vis, line, (6, 22 + li * 18),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                              (200, 220, 0), 1)

            cw = video_canvas.winfo_width() or 700
            ch = video_canvas.winfo_height() or 540
            vh, vw = vis.shape[:2]
            scale = min(cw / vw, ch / vh) if vw and vh else 1.0
            nw, nh = max(1, int(vw * scale)), max(1, int(vh * scale))
            vis_small = cv2.resize(vis, (nw, nh))
            rgb = cv2.cvtColor(vis_small, cv2.COLOR_BGR2RGB)
            try:
                from PIL import Image, ImageTk
            except ImportError:
                messagebox.showerror(
                    "Missing dependency",
                    "Pillow (PIL) is required for the live preview.",
                    parent=self)
                return
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            video_canvas.delete('all')
            video_canvas.create_image(cw // 2, ch // 2, image=photo,
                                        anchor='center')
            video_canvas.image = photo

        # Wire all live controls to the debounced render. Reads main
        # panel vars directly (no preview-local copies).
        all_vars = [frame_var, self._method_var, self._signal_var,
                     self._frac_bright_floor_var]
        for d in (self._height_thresh_vars,
                   self._raw_thresh_vars,
                   self._brightness_thresh_vars,
                   self._z_thresh_vars,
                   self._fb_thresh_vars,
                   self._roi_half_vars):
            all_vars.extend(d.values())
        for v in all_vars:
            try:
                v.trace_add('write', _render)
            except tk.TclError:
                pass
        win.bind('<Configure>', _render)
        win.after(120, _do_render)

    # ═══════════════════════════════════════════════════════════════════════
    # Graphs (per-metric notebook with stats / export, gait helpers reused)
    # ═══════════════════════════════════════════════════════════════════════

    def _bind_gait_graph_helpers(self):
        """Attach GaitLimbTab graph helpers to this instance.

        Why: the gait tab carries ~600 LOC of polished plotting infra
        (per-metric notebook, treatment grouping, stats annotations,
        navigation toolbar, axis editor, Export Graph / Export Data).
        Rather than duplicate or refactor that, we bind the relevant
        helpers as instance methods on this BodyContactTab so any
        ``self.X`` access inside them resolves against our own state.
        """
        if self._gait_graph_helpers_bound:
            return True
        try:
            from gait_limb_tab import GaitLimbTab as _G
        except Exception:
            return False
        import types as _types

        # Bind plain instance methods.
        for _name in ('_treatment_groups', '_add_stat_annotation',
                       '_embed_figure', '_style_ax',
                       '_make_metric_selector', '_register_metric',
                       '_build_bar_graph', '_build_box_graph',
                       '_build_violin_graph', '_add_export_buttons'):
            _fn = _G.__dict__.get(_name)
            if _fn is None:
                continue
            setattr(self, _name, _types.MethodType(_fn, self))

        # Static methods are callable as plain functions.
        for _name in ('_calc_error', '_error_label'):
            _sm = _G.__dict__.get(_name)
            if isinstance(_sm, staticmethod):
                setattr(self, _name, _sm.__func__)

        self._gait_graph_helpers_bound = True
        return True

    def _open_graphs(self):
        """Per-metric graph window for body-contact summary.

        One outer notebook with three categories (Contact %, Bout
        duration, Brightness Δ); each category exposes a metric
        selector with centroid / tailbase variants. Charts, stats
        overlays, axis editor, and CSV / figure export buttons are
        provided by the borrowed gait-tab helpers.
        """
        if self._summary_df is None or self._summary_df.empty:
            messagebox.showinfo("No data", "Run analysis first.",
                                 parent=self)
            return
        try:
            import matplotlib
            matplotlib.use('TkAgg', force=False)
            import matplotlib.pyplot as plt
        except Exception as e:
            messagebox.showerror("matplotlib missing", str(e),
                                  parent=self)
            return
        if not self._bind_gait_graph_helpers():
            messagebox.showerror(
                "Gait tab unavailable",
                "Could not import GaitLimbTab graph helpers; the body\n"
                "contact graphs depend on them.", parent=self)
            return

        df = self._summary_df

        # Build the treatment ordering / cfg the gait helpers expect.
        if ('treatment' in df.columns
                and df['treatment'].astype(str).str.strip().ne('').any()):
            order = (df['treatment'].astype(str).str.strip()
                     .loc[lambda s: s.ne('')].drop_duplicates().tolist())
        else:
            order = []
        graph_cfg = {
            'order':        order,
            'colors':       {},
            'error_type':   'SEM',
            'show_stats':   self._enable_stats_var.get(),
            'sig_style':    self._sig_style,
        }
        self._last_graph_cfg = graph_cfg

        win = tk.Toplevel(self)
        win.title("Body Contact — Graphs")
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        gw = int(sw * 0.72)
        gh = int(sh * 0.80)
        win.geometry(f"{gw}x{gh}+{(sw - gw) // 2}+{(sh - gh) // 2}")

        # Description label updates as the user picks metrics.
        desc_lbl = ttk.Label(win, text='', wraplength=int(gw * 0.95),
                              foreground='#444',
                              font=(FONT_FAMILY, 9, 'italic'),
                              padding=(6, 2))
        desc_lbl.pack(fill='x', padx=6)

        nb = ttk.Notebook(win)
        nb.pack(fill='both', expand=True, padx=6, pady=6)

        tab_descs = {}

        # Per-category metric registration.
        sel_pct  = self._make_metric_selector(nb, "Contact %",
                                                desc_lbl, tab_descs)
        sel_bout = self._make_metric_selector(nb, "Bout Duration",
                                                desc_lbl, tab_descs)
        sel_dbrt = self._make_metric_selector(nb, "Brightness Δ",
                                                desc_lbl, tab_descs)

        # ── Contact % (cpct_centroid, cpct_tailbase) ────────────────────
        for col, label, bp in [
            ('cpct_centroid', 'Contact % — Centroid', 'centroid'),
            ('cpct_tailbase', 'Contact % — Tailbase', 'tailbase'),
        ]:
            if col not in df.columns or not df[col].notna().any():
                continue
            self._register_metric(
                sel_pct, f'{label} (Box)', 'box', df, col,
                reference=None, y_label='% frames in contact',
                graph_cfg=graph_cfg,
                description=f'Percent of frames classified as in contact '
                            f'for {bp}. Boxplot, treatment-grouped, with '
                            f'jittered points and significance overlay.')
            self._register_metric(
                sel_pct, f'{label} (Violin)', 'violin', df, col,
                reference=None, y_label='% frames in contact',
                graph_cfg=graph_cfg,
                description=f'Same data as the {bp} contact-percent box '
                            f'plot, drawn as a violin to show the full '
                            f'distribution shape.')

        # ── Mean bout duration (s) ──────────────────────────────────────
        for col, label, bp in [
            ('mbout_centroid', 'Mean Bout Duration — Centroid',
             'centroid'),
            ('mbout_tailbase', 'Mean Bout Duration — Tailbase',
             'tailbase'),
        ]:
            if col not in df.columns or not df[col].notna().any():
                continue
            self._register_metric(
                sel_bout, f'{label} (Box)', 'box', df, col,
                reference=None, y_label='mean bout duration (s)',
                graph_cfg=graph_cfg,
                description=f'Average duration (seconds) of one '
                            f'continuous {bp} contact bout, after the '
                            f'min-bout debounce.')
            self._register_metric(
                sel_bout, f'{label} (Bar)', 'bar', df, col,
                reference=None, y_label='mean bout duration (s)',
                graph_cfg=graph_cfg,
                description=f'Bar with mean ± SEM of the {bp} mean '
                            f'bout duration, per treatment.')

        # ── ΔBrt = brightness during contact – overall brightness ───────
        for col, label, bp in [
            ('dbrt_centroid', 'ΔBrt — Centroid', 'centroid'),
            ('dbrt_tailbase', 'ΔBrt — Tailbase', 'tailbase'),
        ]:
            if col not in df.columns or not df[col].notna().any():
                continue
            # Reference 0 = no brightness change between contact and
            # background. Useful zero line for signed metric.
            self._register_metric(
                sel_dbrt, f'{label} (Box)', 'box', df, col,
                reference=0.0, y_label='ΔBrt (contact − overall)',
                graph_cfg=graph_cfg,
                description=f'Mean ROI brightness during contact frames '
                            f'minus mean brightness over all frames, for '
                            f'{bp}. Reference 0 = no change.')
            self._register_metric(
                sel_dbrt, f'{label} (Violin)', 'violin', df, col,
                reference=0.0, y_label='ΔBrt (contact − overall)',
                graph_cfg=graph_cfg,
                description=f'Same {bp} ΔBrt data as a violin plot.')

        # Show first metric in each populated category, prefer Contact %.
        for sel in (sel_pct, sel_bout, sel_dbrt):
            if sel['registry']:
                win.after_idle(lambda _s=sel: _s['show'](0))
                break

        # If nothing got registered, surface that to the user.
        if not (sel_pct['registry'] or sel_bout['registry']
                or sel_dbrt['registry']):
            messagebox.showinfo(
                "No metrics", "No plottable metric columns found in the\n"
                              "current summary.", parent=win)
            win.destroy()
            return

        # ── Cleanup: close every embedded matplotlib figure on exit ────
        def _close_figures_recursive(widget):
            fig_obj = getattr(widget, 'figure', None)
            if fig_obj is not None:
                try:
                    plt.close(fig_obj)
                except Exception:
                    pass
            for child in getattr(widget, 'winfo_children',
                                   lambda: [])():
                _close_figures_recursive(child)

        def _on_close():
            for w in win.winfo_children():
                _close_figures_recursive(w)
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", _on_close)

    # ═══════════════════════════════════════════════════════════════════════
    # Per-session preview plot
    # ═══════════════════════════════════════════════════════════════════════

    def _plot_session(self):
        """Per-session time-course preview.

        Shows a row per signal (raw, ΔBrt, z) and a column per
        bodypart (centroid, tailbase). Each panel: the signal trace,
        the user's threshold for *that* signal as a dashed line, and
        gold shading where the active method+signal classifies the
        frame as in contact. The contact shading uses the *current*
        method/signal selection so the user can see whether the
        chosen signal's threshold actually bounds the right regions.
        """
        sel = self._res_tree.selection()
        if not sel:
            messagebox.showwarning(
                "No selection",
                "Select a row in Results to plot the time-course.",
                parent=self)
            return
        name = self._res_tree.item(sel[0], 'values')[0]
        X = self._session_features.get(name)
        if X is None:
            messagebox.showwarning(
                "Not loaded",
                f"Features for '{name}' weren't kept in memory; re-run "
                f"the analysis on this session.", parent=self)
            return

        method = self._method_var.get()
        active_signal = self._signal_var.get()
        height_thr = {bp: float(self._height_thresh_vars[bp].get())
                      for bp in _TARGETS}
        # Per-signal threshold dicts so each panel's dashed line
        # matches what *that* signal would use to call contact.
        signal_thresh = {
            'raw':         self._raw_thresh_vars,
            'dbrt':        self._brightness_thresh_vars,
            'z':           self._z_thresh_vars,
            'frac_bright': self._fb_thresh_vars,
        }
        active_sig_thr = {
            bp: float(signal_thresh[active_signal][bp].get())
            for bp in _TARGETS}

        try:
            import matplotlib
            matplotlib.use('TkAgg', force=False)
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import (
                FigureCanvasTkAgg, NavigationToolbar2Tk)
        except Exception as e:
            messagebox.showerror("matplotlib missing", str(e), parent=self)
            return

        # Convert frame index to seconds when we have an fps for this
        # session in the summary; fall back to the spinbox default.
        fps = float(self._fallback_fps_var.get() or 60.0)
        try:
            sdf = self._summary_df
            if sdf is not None and 'session' in sdf.columns:
                row = sdf.loc[sdf['session'] == name]
                if not row.empty and 'fps' in row.columns:
                    rfps = float(row['fps'].iloc[0])
                    if rfps > 0:
                        fps = rfps
        except Exception:
            pass

        win = tk.Toplevel(self)
        win.title(f"Body contact — {name}")
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        gw = int(sw * 0.70)
        gh = int(sh * 0.78)
        win.geometry(f"{gw}x{gh}+{(sw - gw) // 2}+{(sh - gh) // 2}")

        # Header: which method / signal is driving the contact shading.
        hdr_txt = (f"Session: {name}    method={method}    "
                   f"contact-shading signal={active_signal}    "
                   f"fps={fps:.1f}")
        ttk.Label(win, text=hdr_txt, foreground='#444',
                   font=(FONT_FAMILY, 9, 'italic'),
                   padding=(8, 4)).pack(fill='x')

        # Signal × bodypart grid. Rows = signals, cols = bodyparts.
        # frac_bright is preview-only (no cached column), so we
        # auto-skip rows whose builder returns None for every bp.
        sig_specs = [
            ('raw',  'Raw  Pix_{bp}',                  'steelblue'),
            ('dbrt', 'ΔBrt  Pix_baseline_sub_{bp}',     'crimson'),
            ('z',    'Z  (raw − μ)/σ',                  'darkgreen'),
        ]
        # Pre-compute series so empty rows can be dropped.
        series_grid = {}  # (signal, bp) -> Series or None
        present_signals = []
        for sig, _, _ in sig_specs:
            row_has_data = False
            for bp in _TARGETS:
                s = self._build_brightness_series(X, bp, sig)
                series_grid[(sig, bp)] = s
                if s is not None and s.notna().any():
                    row_has_data = True
            if row_has_data:
                present_signals.append(sig)
        if not present_signals:
            messagebox.showinfo(
                "No signals",
                "None of raw / ΔBrt / Z had usable data for this "
                "session. Re-run extraction with brightness "
                "features enabled.", parent=win)
            win.destroy()
            return

        # Build the figure. sharex=True per column so panning one
        # bodypart's column scrolls all 3 signals together.
        ncols = len(_TARGETS)
        nrows = len(present_signals)
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(10, 2.2 * nrows + 0.6),
                                  sharex='col',
                                  constrained_layout=True)
        # Normalise to 2-D array for indexing.
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = np.asarray(axes).reshape(1, -1)
        elif ncols == 1:
            axes = np.asarray(axes).reshape(-1, 1)

        # Per-bodypart contact masks — one per column.
        bp_masks = {}
        for bp in _TARGETS:
            try:
                bp_masks[bp] = self._compute_contact_mask(
                    X, bp, method, height_thr[bp],
                    active_sig_thr[bp], signal=active_signal)
            except Exception:
                bp_masks[bp] = None

        n_frames = len(X)
        t = (np.arange(n_frames) / fps if fps > 0
             else np.arange(n_frames))
        x_label = 'Time (s)' if fps > 0 else 'Frame'

        for ri, sig in enumerate(present_signals):
            label_tmpl, color = next(
                (lab, col) for s, lab, col in sig_specs if s == sig)
            for ci, bp in enumerate(_TARGETS):
                ax = axes[ri, ci]
                s = series_grid.get((sig, bp))
                if s is None or not s.notna().any():
                    ax.text(0.5, 0.5,
                            f"{sig}: not available\n(needs cached "
                            f"Pix_{bp} or Pix_baseline_sub_{bp})",
                            transform=ax.transAxes,
                            ha='center', va='center', fontsize=8,
                            color='#888')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    continue
                # Trace
                ax.plot(t[:len(s)], s.values[:len(t)], color=color,
                         linewidth=0.8)
                # Threshold line for this signal
                thr = float(signal_thresh[sig][bp].get())
                ax.axhline(thr, color=color, linestyle='--',
                            linewidth=0.8, alpha=0.6,
                            label=f'thr={thr:g}')
                ax.legend(loc='upper right', fontsize=7,
                           frameon=False)
                # Contact mask shading from active method+signal
                m = bp_masks.get(bp)
                if m is not None and len(m) > 0:
                    yl = ax.get_ylim()
                    ax.fill_between(t[:len(m)], yl[0], yl[1],
                                     where=m[:len(t)],
                                     color='gold', alpha=0.18,
                                     step='mid')
                    ax.set_ylim(yl)
                # Cosmetics
                ax.tick_params(axis='both', labelsize=7)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                if ci == 0:
                    ax.set_ylabel(label_tmpl.format(bp=bp),
                                   fontsize=8, color=color)
                if ri == 0:
                    ax.set_title(bp, fontsize=10, loc='left',
                                  fontweight='bold')
                if ri == nrows - 1:
                    ax.set_xlabel(x_label, fontsize=9)

        fig.suptitle(
            f"{name}  —  per-signal time-course "
            f"(gold shading = contact under method={method}, "
            f"signal={active_signal})",
            fontsize=10)

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        # Cleanup on close so per-session previews don't leak figures.
        def _on_close():
            try:
                plt.close(fig)
            except Exception:
                pass
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", _on_close)
