"""
Paw Contour Analysis Tab  (gait_tab_v2.py)
==========================================
Ground-up Phase-B rebuild of the Gait & Limb tab on the app's shared
architecture, reduced (2026-09-18) to paw-contour and ROI-brightness
measures: the gait and movement metrics are no longer computed
(``gait_core`` runs with ``compute_gait`` off) and their graphs, settings
and table columns are gone.

  • Compute lives in headless ``gait_core`` (golden-tested; ONE metrics
    implementation shared by the analyze and Adjust-Contact paths, so the
    old recompute drift - missing licking/4-paw masks - is dead by
    construction).
  • Graphs live in ``gait_views`` (the paw-contour and brightness part of
    the old graph window as a registry) and render INTO the right pane
    behind two dropdowns: Category → Graph, with the shared 🎨⚙ style
    dialog and a Σ Stats flip.
  • The left rail reads: Data (key file, then the Sessions picker) →
    Quick Setup (preset with Run/Cancel beside it and the readiness strip -
    the single source of truth; the old 'video_path'/'video' mismatch is
    fixed) → Setup → Detection → ▸ Advanced (collapsed) → Results & Export.

The old ``gait_limb_tab.py`` remains importable as the fallback and as a
"(legacy)" tab under INCLUDE_DEV_TABS. On-disk formats (caches, sidecars,
session bundles schema v1; caches and bundles in gc.analysis_dir():
paw_contour/, or gait_limb_analysis/ where a project already has it).
"""

import os
import re
import json
import pickle
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from datetime import datetime

import numpy as np
import pandas as pd

try:
    from io_utils import atomic_pickle_save
except Exception:  # pragma: no cover - fallback if io_utils unavailable
    atomic_pickle_save = None

try:
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False

try:
    import matplotlib
    import matplotlib.pyplot as plt
    _PLOT_OK = True
except ImportError:
    _PLOT_OK = False

from pose_features import PoseFeatureExtractor

from ui_utils import (ToolTip as _ToolTip,
                      bind_mousewheel, FONT_FAMILY)
from ui_tooltip import Tip, collapsible
from ui_session_filter import SessionFilter

import gait_core as gc
from gait_core import find_session_triplets

try:
    import gait_views
except Exception as _gv_err:  # pragma: no cover
    gait_views = None
    print(f'[gait_tab_v2] WARNING: gait_views unavailable ({_gv_err}); '
          f'graphs disabled.')

try:
    import plot_style as ps
except Exception:  # pragma: no cover
    ps = None

_robust_unpickle = gc.robust_unpickle

try:
    from PixelPaws_GUI import extract_subject_id_from_filename as _extract_sid
except Exception:  # pragma: no cover - GUI module unavailable headless
    _extract_sid = None


class GaitLimbTabV2(ttk.Frame):
    """Paw Contour analysis tab (paw-contour shape + ROI brightness)."""

    ROLES = ('HL', 'HR', 'FL', 'FR')
    ROLE_LABELS = {
        'HL': 'Hind-Left paw:',
        'HR': 'Hind-Right paw:',
        'FL': 'Fore-Left paw:',
        'FR': 'Fore-Right paw:',
    }
    ROLE_DEFAULTS = {'HL': 'hlpaw', 'HR': 'hrpaw', 'FL': 'flpaw', 'FR': 'frpaw'}

    # Quick-setup presets - each maps to the interdependent toggle vars so the
    # user never has to know that contour needs brightness needs video. Applied
    # only when the user picks one; the on-load defaults (set in __init__) are
    # left untouched. Keys are shown in the "Quick setup" combobox in order.
    # Single source: gait_core.GAIT_PRESETS (the manuscript gate first, then
    # contour-area presets; the gait presets were dropped 2026-09-18 and an
    # old bundle's preset name maps through gait_core.LEGACY_PRESET_NAMES).
    GAIT_PRESETS = gc.GAIT_PRESETS

    CONTOUR_PASS_TIP = (
        "Contour gate pass %\n"
        "Share of the analyzed frames in which the hind-paw contours fall\n"
        "inside the area band. The contour metrics come from these frames.\n"
        "A quality-control number, not a behavior.")

    # Session table (Results tree): (tree column, header, width,
    # summary column, value format, header tooltip).  Ratios are the stored
    # HL/HR values; the graphs show them as injured/contralateral.
    _RESULT_COLUMNS = (
        ('session',   'Session',             130, 'session',   None, None),
        ('subject',   'Subject',              70, 'subject',   None, None),
        ('treatment', 'Treatment',            80, 'treatment', None, None),
        ('pass_pct',  'Contour gate pass %', 118, 'contour_pass_pct', '{:.1f}',
         CONTOUR_PASS_TIP),
        ('area_r',    'Area HL/HR',           72, 'paw_area_ratio_hind', '{:.3f}',
         'Paw area ratio\nMean contour area HL ÷ HR over the gated frames.'),
        ('int_r',     'Int HL/HR',            66, 'contact_intensity_ratio_hind',
         '{:.3f}',
         'Intensity ratio\nMean brightness inside the contour, HL ÷ HR.'),
        ('brt_r',     'Brt HL/HR',            66, 'brightness_ratio_HL_HR', '{:.3f}',
         'Brightness ratio\nMean ROI brightness HL ÷ HR over the gated frames.'),
        ('area_hl',   'Area HL',              60, 'paw_area_HL', '{:.0f}',
         'Mean paw contour area (px²) - hind left.'),
        ('area_hr',   'Area HR',              60, 'paw_area_HR', '{:.0f}',
         'Mean paw contour area (px²) - hind right.'),
        ('int_hl',    'Int HL',               56, 'contact_intensity_HL', '{:.1f}',
         'Mean brightness inside the contour - hind left.'),
        ('int_hr',    'Int HR',               56, 'contact_intensity_HR', '{:.1f}',
         'Mean brightness inside the contour - hind right.'),
        ('brt_hl',    'Brt HL',               56, 'brightness_HL', '{:.1f}',
         'Mean ROI brightness - hind left.'),
        ('brt_hr',    'Brt HR',               56, 'brightness_HR', '{:.1f}',
         'Mean ROI brightness - hind right.'),
    )


    def __init__(self, parent, main_gui):
        super().__init__(parent)
        self.app = main_gui
        self.pack(fill='both', expand=True)

        self._sessions: list = []
        self._key_df: pd.DataFrame = None
        self._key_scan_paths: list = []
        self._summary_df: pd.DataFrame = None
        self._bins_df: pd.DataFrame = None
        self._enable_stats_var       = tk.BooleanVar(value=True)
        self._stats_test_var         = tk.StringVar(value='auto')
        self._stats_alpha_var        = tk.DoubleVar(value=0.05)
        self._timecourse_posthoc_var = tk.BooleanVar(value=False)
        self._stats_paradigm_var    = tk.StringVar(value='parametric')
        # Contour area is the only contact method this tab produces; the var
        # is read-only state for readiness notes and is not persisted.
        # (Median filter / min bout / min stance, the DLC-likelihood filter and
        # the locomotion filter were dropped 2026-09-18: they only acted on the
        # speed contact method or the gait metrics.)
        self._contact_method_var  = tk.StringVar(value='contour_area')
        self._paw_contour_var    = tk.BooleanVar(value=True)
        self._contour_forelimbs_var = tk.BooleanVar(value=False)
        # px^2 band for contour-area contact. Defaults are the manuscript's deployed gate
        # (the oxycodone-dose analysis): a real paw runs ~2,800 px^2 in the 100x100 ROI,
        # so 1,500-5,000 excludes spill-over contours; the old floor of 20 never excluded
        # a frame.
        self._contour_area_thresh_var = tk.IntVar(value=1500)
        self._contour_area_max_var = tk.IntVar(value=5000)
        # Which hind paw carries the injury/injection. Stored ratio columns stay HL/HR;
        # the results window relabels (and inverts when HR) so every ratio graph reads
        # injured/contralateral regardless of side.
        self._injured_paw_var = tk.StringVar(value='HL')
        # Exclude-licking (formalin): drop frames predicted as a licking behavior
        # from the contour gate pass % and the ROI brightness.
        self._exclude_lick_var   = tk.BooleanVar(value=False)
        self._lick_behavior_var  = tk.StringVar(value='')
        self._lick_thresh_var    = tk.DoubleVar(value=0.5)
        self._lick_behaviors     = []   # behaviors found on disk (results/)
        # Hard gate: restrict the gate pass % and ROI brightness to frames where
        # all four paws are in contact (fore paws by height).
        self._gate_4paw_var      = tk.BooleanVar(value=False)
        self._fit_thread: threading.Thread = None
        self._cancel_flag = threading.Event()
        self._bodyparts: list = []
        self._session_intermediates = {}
        self._loading_session = False      # True while restoring (suppresses auto-save)
        self._saved_items = []             # [(label, path)] for the saved-session combo
        self._pawlike_thresholds = {'solidity': 1.00, 'aspect_ratio': 1.6, 'circularity': 0.10}


        # v2 results-pane state
        self._registry = {}          # {category: [entry, ...]} from gait_views
        self._host = None            # gait_views.ViewHost for current results
        self._stats_mode_var = tk.BooleanVar(value=False)
        self._cat_var = tk.StringVar(value='')
        self._graph_var = tk.StringVar(value='')

        self._build_ui()
        # Discover licking behaviors on disk before applying the default preset,
        # so the Formalin profile can auto-enable licking exclusion when available.
        try:
            self._refresh_lick_behaviors()
        except Exception:
            pass
        # Open in the default analysis profile (formalin contour/brightness).
        try:
            self._apply_gait_preset()
        except Exception:
            pass
        try:
            self._refresh_saved_sessions()
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════════════════
    # UI construction
    # ═══════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        hdr = ttk.Frame(self)
        hdr.pack(fill='x', padx=12, pady=(10, 2))
        ttk.Label(hdr, text="🐾  Paw Contour Analysis",
                  font=(FONT_FAMILY, 14, 'bold')).pack(side='left')
        ttk.Label(hdr,
                  text="   Check Data → pick a preset → run.",
                  foreground='grey', font=(FONT_FAMILY, 9)).pack(side='left')

        paned = ttk.PanedWindow(self, orient='horizontal')
        paned.pack(fill='both', expand=True)

        # ── LEFT: scrollable control rail ──
        left_host = ttk.Frame(paned)
        paned.add(left_host, weight=0)
        lcanvas = tk.Canvas(left_host, width=430, highlightthickness=0)
        lscroll = ttk.Scrollbar(left_host, orient='vertical', command=lcanvas.yview)
        rail = ttk.Frame(lcanvas)
        rail.bind('<Configure>',
                  lambda e: lcanvas.configure(scrollregion=lcanvas.bbox('all')))
        _rail_id = lcanvas.create_window((0, 0), window=rail, anchor='nw')
        lcanvas.bind('<Configure>',
                     lambda e: lcanvas.itemconfig(_rail_id, width=e.width))
        lcanvas.configure(yscrollcommand=lscroll.set)
        lcanvas.pack(side='left', fill='both', expand=True)
        lscroll.pack(side='right', fill='y')
        try:
            bind_mousewheel(lcanvas)
        except Exception:
            pass

        self._build_data_panel(rail)
        self._build_quicksetup(rail)     # preset + Run/Cancel + readiness
        self._build_settings_panel(rail)
        self._build_export_panel(rail)

        # ── RIGHT: results pane (graphs + tables + log) ──
        right = ttk.Frame(paned)
        paned.add(right, weight=1)
        self._build_results_pane(right)

        # Re-render the registry when display/stats knobs change, keep the
        # readiness strip and the picker's Subject column live.
        for _v in (self._injured_paw_var, self._enable_stats_var,
                   self._stats_paradigm_var, self._stats_alpha_var):
            _v.trace_add('write',
                         lambda *_: self._populate_registry(keep=True))
        self._contact_method_var.trace_add('write', self._update_readiness)
        self._prefix_var.trace_add('write', self._on_prefix_changed)

        # Initial readiness state (all widgets now exist)
        self._update_readiness()

        # Offer to reload the last saved results when this tab becomes active.
        try:
            self.app.notebook.bind('<<NotebookTabChanged>>',
                                   self._on_tab_shown, add='+')
        except Exception:
            pass

    def _build_quicksetup(self, parent):
        """Preset chooser with Run/Cancel beside it, readiness underneath."""
        lf = ttk.LabelFrame(parent, text="Quick Setup", padding=6)
        lf.pack(fill='x', padx=4, pady=(6, 2))
        row = ttk.Frame(lf)
        row.pack(fill='x')
        ttk.Label(row, text="Preset:").pack(side='left')
        self._preset_var = tk.StringVar(value='Paw contour (manuscript gate)')
        self._preset_combo = ttk.Combobox(
            row, textvariable=self._preset_var, state='readonly',
            width=26, values=list(self.GAIT_PRESETS.keys()))
        self._preset_combo.pack(side='left', padx=4)
        self._preset_combo.bind('<<ComboboxSelected>>', self._apply_gait_preset)
        self._tip(self._preset_combo,
                  "One-click setup for common runs. Sets the brightness / contour /\n"
                  "forepaw options for you so you don't have to know their\n"
                  "dependencies. Fine-tune anything afterward in the sections below.")
        self._run_btn = ttk.Button(row, text="▶  Run Analysis",
                                   command=self._start_analysis,
                                   state='disabled')
        self._run_btn.pack(side='left', padx=(8, 2))
        self._cancel_btn = ttk.Button(row, text="■", width=3,
                                      command=self._cancel_analysis,
                                      state='disabled')
        self._cancel_btn.pack(side='left', padx=2)
        self._tip(self._cancel_btn, "Cancel the running analysis.")

        # Readiness strip - always shows the next actionable step.
        self._readiness_lbl = ttk.Label(
            lf, text='', foreground='grey', wraplength=380,
            justify='left', font=(FONT_FAMILY, 8))
        self._readiness_lbl.pack(fill='x', pady=(3, 0))

        # Progress bar (hidden until analysis runs; shown/hidden by
        # _start_analysis / _on_analysis_complete).
        self._progress_frame = ttk.Frame(lf)
        self._progress = ttk.Progressbar(self._progress_frame,
                                         mode='determinate')
        self._progress.pack(fill='x', padx=4, pady=(0, 1))
        self._sub_progress_label = ttk.Label(
            self._progress_frame, text="", font=('TkDefaultFont', 8))
        self._sub_progress_label.pack(fill='x', padx=4, pady=(0, 2))

    # ── Left: Data (key file first, then the session picker) ───────────────

    def _build_data_panel(self, parent):
        self._override_folder_var = tk.StringVar(value='')

        lf = ttk.LabelFrame(parent, text="Data", padding=6)
        lf.pack(fill='x', padx=4, pady=4)

        # Key file first - same order as the other analysis tabs.
        kf_row = ttk.Frame(lf)
        kf_row.pack(fill='x', pady=2)
        ttk.Label(kf_row, text="Key file:", width=8).pack(side='left')
        self._key_file_var = tk.StringVar()
        self._key_combo = ttk.Combobox(kf_row, textvariable=self._key_file_var,
                                       state='normal', width=16)
        self._key_combo.pack(side='left', padx=3, fill='x', expand=True)
        self._key_combo.bind('<<ComboboxSelected>>', self._on_key_combo_selected)
        ttk.Button(kf_row, text="Browse", width=7,
                   command=self._browse_key_file).pack(side='left')
        ttk.Button(kf_row, text="Generate…", width=9,
                   command=self._generate_key_file).pack(side='left', padx=(4, 0))

        pfx_row = ttk.Frame(lf)
        pfx_row.pack(fill='x', pady=2)
        ttk.Label(pfx_row, text="Prefix:", width=8).pack(side='left')
        self._prefix_var = tk.StringVar()
        pfx_ent = ttk.Entry(pfx_row, textvariable=self._prefix_var, width=16)
        pfx_ent.pack(side='left', padx=3)
        self._tip(pfx_ent,
                  "Filename prefix to strip before extracting subject ID.\n"
                  "e.g. '260129_Formalin_' → next underscore-token = subject.")

        self._key_status_lbl = ttk.Label(lf, text='No key file loaded',
                                         foreground='grey', wraplength=380,
                                         justify='left')
        self._key_status_lbl.pack(anchor='w', pady=(0, 4))

        # Sessions - same dropdown-table picker as the other tabs
        # (opens the session table with Subject / Video / Cache columns).
        sess_row = ttk.Frame(lf)
        sess_row.pack(fill='x', pady=2)
        self._session_filter = SessionFilter(sess_row,
                                             on_change=self._update_readiness)
        self._session_filter.pack(side='left', fill='x', expand=True)
        ttk.Button(sess_row, text="Rescan", width=7,
                   command=self._scan_sessions).pack(side='left', padx=(4, 0))
        _br = ttk.Button(sess_row, text="Browse…", width=8,
                         command=self._browse_sessions_folder)
        _br.pack(side='left', padx=(4, 0))
        self._tip(_br, "Analyze sessions from a different folder than the\n"
                       "current project.")
        self._sess_lbl = ttk.Label(lf, text='', foreground='grey')
        self._sess_lbl.pack(anchor='w', pady=(2, 0))
        self._folder_lbl = ttk.Label(lf, text='', foreground='grey',
                                     wraplength=380, font=(FONT_FAMILY, 8))
        self._folder_lbl.pack(anchor='w')

    # ── Session scanning (feeds the picker) ────────────────────────────────

    def _scan_sessions(self):
        folder = (self._override_folder_var.get()
                  or self.app.current_project_folder.get())
        if not folder or not os.path.isdir(folder):
            return

        try:
            self._sessions = find_session_triplets(folder, require_labels=False)
        except Exception as e:
            self._log_ui(f"Session scan error: {e}")
            self._sessions = []

        # Auto-detect body parts from first available DLC h5
        for sess in self._sessions:
            if sess.get('dlc') and os.path.isfile(sess['dlc']):
                self._auto_populate_bodyparts(sess['dlc'])
                break

        self._publish_sessions(folder)

        n = len(self._sessions)
        self._sess_lbl.config(text=f'{n} session{"s" if n != 1 else ""} found')
        self._folder_lbl.config(
            text=os.path.basename(folder) if folder else '')

        self._scan_key_files(folder)
        self._refresh_lick_behaviors()
        self._update_readiness()

    def _publish_sessions(self, folder=None):
        """Feed the session picker: name + Subject / Video / Cache columns."""
        folder = folder or (self._override_folder_var.get()
                            or self.app.current_project_folder.get())
        extra = {}
        for sess in self._sessions:
            name = sess['session_name']
            has_vid = '✓' if (sess.get('video')
                              and os.path.isfile(sess['video'])) else '✗'
            extra[name] = {'Subject': self._resolve_subject(name),
                           'Video': has_vid,
                           'Cache': self._session_cache_status(name, folder)}
        self._session_filter.set_sessions(
            [s['session_name'] for s in self._sessions],
            key_df=self._key_df, extra=extra)

    def _scan_key_files(self, folder: str):
        """Find key-file candidates (Subject+Treatment; results exports
        excluded) and auto-pick when the choice is unambiguous."""
        candidates = []
        try:
            from project_config import find_key_files, _KEY_RESULTS_COLS
            candidates = list(find_key_files(folder))  # ranked, csv only
        except Exception:
            _KEY_RESULTS_COLS = set()
        # xlsx keys (find_key_files is csv-only)
        _SKIP = {'__pycache__', '.git', 'node_modules', '.idea'}
        for root, dirs, files in os.walk(folder):
            dirs[:] = [d for d in sorted(dirs)
                       if d not in _SKIP and not d.startswith('.')]
            for fname in files:
                if not fname.lower().endswith('.xlsx'):
                    continue
                full = os.path.join(root, fname)
                try:
                    cols = set(pd.read_excel(full, nrows=0).columns)
                except Exception:
                    continue
                if ('Subject' in cols and 'Treatment' in cols
                        and not (_KEY_RESULTS_COLS & cols)):
                    candidates.append(full)

        self._key_scan_paths = candidates
        labels = [os.path.relpath(p, folder).replace(os.sep, '/')
                  for p in candidates]
        self._key_combo.config(values=labels)
        # Auto-pick the top-ranked candidate when the choice is unambiguous:
        # it is the only candidate, or it is clearly named as a key.
        if candidates and not self._key_file_var.get():
            top = candidates[0]
            if (len(candidates) == 1
                    or 'key' in os.path.basename(top).lower()):
                self._key_combo.current(0)
                self._on_key_combo_selected()

    def _on_prefix_changed(self, *_):
        if getattr(self, '_sessions', None):
            self._publish_sessions()

    def _select_all(self):
        self._session_filter.select_all(True)

    def _rail_section(self, parent, title):
        """A bold section header + content frame stacked in the rail (replaces the
        old settings sub-tabs). Returns the content frame to build widgets into."""
        ttk.Label(parent, text=title,
                  font=(FONT_FAMILY, 10, 'bold')).pack(anchor='w', padx=6, pady=(10, 0))
        frame = ttk.Frame(parent)
        frame.pack(fill='x', padx=2)
        return frame

    def _build_settings_panel(self, parent):
        # ── Parameter vars (declared up front so any tab can build them) ──
        self._contact_thresh_var = tk.IntVar(value=15)
        self._height_window_var  = tk.IntVar(value=500)
        self._bin_seconds_var    = tk.IntVar(value=1)
        self._bin_unit_var       = tk.StringVar(value='minutes')
        self._fallback_fps_var   = tk.DoubleVar(value=60.0)
        self._use_brightness_var = tk.BooleanVar(value=True)
        self._brt_thresh_var     = tk.IntVar(value=0)
        # No widget since 2026-09-18: the brightness weight never reaches the
        # contour gate. It stays persisted because it is part of the
        # brightness / contour extraction cache keys (a new value would only
        # force a re-extraction).
        self._brt_weight_var        = tk.DoubleVar(value=1.0)
        self._extraction_stride_var = tk.IntVar(value=1)
        self._roi_size_vars      = {
            'HL': tk.IntVar(value=20),
            'HR': tk.IntVar(value=20),
            'FL': tk.IntVar(value=15),
            'FR': tk.IntVar(value=15),
        }
        self._contour_roi_size_vars = {
            'HL': tk.IntVar(value=60),
            'HR': tk.IntVar(value=60),
            'FL': tk.IntVar(value=40),
            'FR': tk.IntVar(value=40),
        }
        self._crop_x_var     = tk.IntVar(value=0)
        self._crop_y_var     = tk.IntVar(value=0)

        # Settings flattened into stacked sections in the scrollable rail (the old
        # Setup/Detection/Advanced sub-tabs). The Quick-setup preset and the
        # Run/Cancel strip now lives in _build_quicksetup.
        setup_inner  = self._rail_section(parent, "Setup")
        detect_inner = self._rail_section(parent, "Detection")
        adv_inner    = collapsible(parent, "Advanced", collapsed=True,
                                   fill='x', padx=2, pady=(10, 0))


        pm_lf = ttk.LabelFrame(setup_inner, text="Paw Mapping", padding=5)
        pm_lf.pack(fill='x', pady=(0, 6), padx=2)

        self._role_vars   = {}
        self._role_combos = {}
        for i, role in enumerate(self.ROLES):
            ttk.Label(pm_lf, text=self.ROLE_LABELS[role]).grid(
                row=i, column=0, sticky='w', pady=2)
            var = tk.StringVar(value=self.ROLE_DEFAULTS[role])
            self._role_vars[role] = var
            cb = ttk.Combobox(pm_lf, textvariable=var, width=14, state='normal')
            cb.grid(row=i, column=1, sticky='w', padx=4, pady=2)
            self._role_combos[role] = cb
            if role in ('FL', 'FR'):
                ttk.Label(pm_lf, text='(optional)',
                          foreground='grey').grid(row=i, column=2, sticky='w')

        _inj_lbl = ttk.Label(pm_lf, text='Injured / injected paw:')
        _inj_lbl.grid(row=len(self.ROLE_LABELS), column=0, sticky='w', pady=(6, 2))
        _inj_cb = ttk.Combobox(pm_lf, textvariable=self._injured_paw_var, width=14,
                               state='readonly', values=['HL', 'HR'])
        _inj_cb.grid(row=len(self.ROLE_LABELS), column=1, sticky='w', padx=4, pady=(6, 2))
        _inj_tip = ('Which hind paw received the injury or injection. Ratio graphs are\n'
                    'shown as injured/contralateral: with HL this is HL/HR as stored;\n'
                    'with HR the displayed ratio is inverted so values below 1.0 always\n'
                    'mean the injured paw is the lower one (a smaller or less intense print).')
        self._tip(_inj_lbl, _inj_tip)
        self._tip(_inj_cb, _inj_tip)

        self._use_fore_var = tk.BooleanVar(value=False)
        self._use_fore_chk = ttk.Checkbutton(
            pm_lf, text='Include fore paws',
            variable=self._use_fore_var,
            command=self._on_use_fore_changed)
        self._use_fore_chk.grid(
            row=len(self.ROLES), column=0, columnspan=3, sticky='w', pady=(4, 0))

        ttk.Button(pm_lf, text="Auto-detect from DLC",
                   command=self._autodetect_bodyparts).grid(
            row=len(self.ROLES) + 1, column=0, columnspan=3, sticky='ew', pady=(6, 2))

        # === Detection: contour gate + fore-paw contact + time bins ===
        cd_lf = ttk.LabelFrame(detect_inner, text="Contour detection", padding=5)
        cd_lf.pack(fill='x', pady=(0, 6), padx=2)

        # Contour area is the only detection method (Height / Speed / Combined
        # were dropped 2026-09-17); _contact_method_var stays "contour_area".
        # The band is the only gate setting: the median filter / min bout
        # (speed method only) and min stance (gait metrics only) went
        # 2026-09-18.
        _cat_lbl = ttk.Label(cd_lf, text="Contour-area band (px²):")
        _cat_lbl.grid(row=0, column=0, sticky='w', pady=2)
        _cat_row = ttk.Frame(cd_lf)
        _cat_row.grid(row=0, column=1, sticky='w', padx=4, pady=2)
        _cat_sb = ttk.Spinbox(_cat_row, from_=0, to=100000,
                              textvariable=self._contour_area_thresh_var, width=7)
        _cat_sb.pack(side='left')
        ttk.Label(_cat_row, text="-").pack(side='left', padx=2)
        _cat_max_sb = ttk.Spinbox(_cat_row, from_=0, to=100000,
                                  textvariable=self._contour_area_max_var, width=7)
        _cat_max_sb.pack(side='left')
        _cat_tip = ("A hind paw counts as on the floor when its contour area falls\n"
                    "inside this band (both hind paws gated together). The contour\n"
                    "metrics come from the frames that pass; the Contour gate pass %\n"
                    "column reports that share. Default 1,500-5,000 px^2, the deployed\n"
                    "manuscript gate: a real paw runs ~2,800 px^2 in the 100x100 ROI, and\n"
                    "the band excludes contours that spilled onto the body or background.")
        self._tip(_cat_lbl, _cat_tip)
        self._tip(_cat_sb, _cat_tip)
        self._tip(_cat_max_sb, _cat_tip)

        _prev_btn = ttk.Button(cd_lf, text="Preview contour…",
                               command=self._open_contour_preview)
        _prev_btn.grid(row=1, column=0, columnspan=2, sticky='w', pady=(2, 0))
        self._tip(_prev_btn,
                  "Visual check of the Otsu contour on sample frames; per-paw "
                  "ROI half-sizes are editable inside the preview.")

        # Fore-paw contact (height) + frame rate. Hind-paw contact is the
        # contour band; the height threshold / window only decide when a FORE
        # paw is down, which feeds the 4-paw gate and the fore-paw ROI
        # brightness, so both are disabled unless fore paws are included.
        kin_lf = ttk.LabelFrame(detect_inner, text="Fore paws and frame rate",
                                padding=5)
        kin_lf.pack(fill='x', pady=(0, 6), padx=2)
        kin_rows = [
            ("Fore-paw down height (px):", self._contact_thresh_var, 0, 500,
             "Fore paws only (hind-paw contact comes from the contour band).\n"
             "A fore paw counts as down when its height is below this.\n\n"
             "Height = rolling_max(paw_y, window=height_window) − current_paw_y,\n"
             "how far the paw has risen above the estimated floor level.\n\n"
             "Used by the 4-paw gate and for fore-paw ROI brightness.\n"
             "Ignored unless 'Include fore paws' is on. Typical: 5-30 px.", True),
            ("Height window (frames):", self._height_window_var, 1, 10000,
             "Rolling-max window for the fore-paw floor estimate.\n"
             "Larger = more stable. Ignored unless 'Include fore paws' is on.",
             True),
            ("Fallback fps:", self._fallback_fps_var, 1, 500,
             "Used when the video cannot be opened to read its actual fps.",
             False),
        ]
        self._fore_contact_widgets = []
        for r, (lbl_text, var, from_, to, tip, fore_only) in enumerate(kin_rows):
            lbl = ttk.Label(kin_lf, text=lbl_text)
            lbl.grid(row=r, column=0, sticky='w', pady=2)
            sb = ttk.Spinbox(kin_lf, from_=from_, to=to, textvariable=var, width=8)
            sb.grid(row=r, column=1, sticky='w', padx=4, pady=2)
            self._tip(lbl, tip)
            self._tip(sb, tip)
            if fore_only:
                self._fore_contact_widgets += [lbl, sb]

        # Time Bins
        tb_lf = ttk.LabelFrame(detect_inner, text="Time Bins", padding=5)
        tb_lf.pack(fill='x', pady=(0, 6), padx=2)

        tb_row = ttk.Frame(tb_lf)
        tb_row.pack(fill='x', pady=2)
        ttk.Label(tb_row, text="Bin size:", width=10).pack(side='left')
        _bin_spx = ttk.Spinbox(tb_row, from_=0, to=3600,
                                textvariable=self._bin_seconds_var, width=7)
        _bin_spx.pack(side='left', padx=4)

        unit_frame = ttk.Frame(tb_row)
        unit_frame.pack(side='left', padx=4)
        ttk.Radiobutton(unit_frame, text="minutes", variable=self._bin_unit_var,
                        value='minutes').pack(side='left')
        ttk.Radiobutton(unit_frame, text="seconds", variable=self._bin_unit_var,
                        value='seconds').pack(side='left', padx=(6, 0))

        ttk.Label(tb_lf,
                  text="Video divided into equal bins. 0 = full session only (no bins).",
                  font=(FONT_FAMILY, 8), foreground='gray').pack(anchor='w', pady=(2, 0))

        # Frame selection: exclude licking frames + restrict to 4-paw stance.
        fs_lf = ttk.LabelFrame(detect_inner, text="Frame Selection", padding=5)
        fs_lf.pack(fill='x', pady=(0, 6), padx=2)

        lick_row = ttk.Frame(fs_lf)
        lick_row.pack(fill='x', pady=2)
        self._exclude_lick_chk = ttk.Checkbutton(
            lick_row, text="Exclude licking frames",
            variable=self._exclude_lick_var, command=self._on_lick_toggle)
        self._exclude_lick_chk.pack(side='left')
        self._lick_combo = ttk.Combobox(
            lick_row, textvariable=self._lick_behavior_var, state='disabled',
            width=16, values=[])
        self._lick_combo.pack(side='left', padx=(6, 2))
        ttk.Label(lick_row, text="thr").pack(side='left')
        self._lick_thr_spin = ttk.Spinbox(
            lick_row, from_=0.0, to=1.0, increment=0.05,
            textvariable=self._lick_thresh_var, width=5, state='disabled')
        self._lick_thr_spin.pack(side='left', padx=(2, 0))
        self._lick_hint_lbl = ttk.Label(
            fs_lf, text="", font=(FONT_FAMILY, 8), foreground='gray',
            wraplength=520, justify='left')
        self._lick_hint_lbl.pack(anchor='w', pady=(0, 2))
        self._tip(self._exclude_lick_chk,
                  "Drop frames the classifier predicts as licking from every\n"
                  "Paw Contour metric: the contour measures, the ROI brightness\n"
                  "and the gate pass %. The area band alone does not remove\n"
                  "licking (a paw held to the mouth is still paw-sized).\n"
                  "Sourced from the project's behavior\n"
                  "predictions (results/…/*_predictions.csv). The default\n"
                  "paw-contact preset enables this automatically.")

        self._gate_4paw_chk = ttk.Checkbutton(
            fs_lf, text="Only analyze frames with all four paws in contact",
            variable=self._gate_4paw_var, command=self._update_readiness)
        self._gate_4paw_chk.pack(anchor='w', pady=(2, 0))
        self._tip(self._gate_4paw_chk,
                  "Four-paw gate: restrict every Paw Contour\n"
                  "metric to frames where all four paws are down (fore paws\n"
                  "by the fore-paw height threshold). The gate pass % then\n"
                  "counts only those frames.\n"
                  "Requires fore paws enabled.")
        self._tip(_bin_spx,
                  "Number of minutes (or seconds) per time bin.\n"
                  "0 = output the full session as a single row only.")

        # === Advanced: brightness / contour / crop / filters (all optional) ===
        brt_lf = ttk.LabelFrame(adv_inner, text="Brightness (needs video)", padding=5)
        brt_lf.pack(fill='x', pady=(0, 6), padx=2)

        brt_rows = [
            ("Brightness threshold:", self._brt_thresh_var, 0, 255,
             "Pixel intensity cutoff for brightness extraction.\n"
             "0 = auto-detect (≈ mean frame brightness × 0.5).\n"
             "Use the Preview button to set this visually."),
            ("Extraction stride:", self._extraction_stride_var, 1, 16,
             "Read every Nth video frame during brightness extraction.\n"
             "1 = every frame (full accuracy).\n"
             "2 = every 2nd frame (~2× faster, minimal accuracy loss for 30-s bins).\n"
             "4 = every 4th frame (~4× faster, suitable for quick exploration runs)."),
        ]
        for r, (lbl_text, var, from_, to, tip) in enumerate(brt_rows):
            lbl = ttk.Label(brt_lf, text=lbl_text)
            lbl.grid(row=r, column=0, sticky='w', pady=2)
            sb = ttk.Spinbox(brt_lf, from_=from_, to=to, textvariable=var, width=8)
            sb.grid(row=r, column=1, sticky='w', padx=4, pady=2)
            self._tip(lbl, tip)
            self._tip(sb, tip)

        brt_cb = ttk.Checkbutton(brt_lf, text="ROI brightness (needs video)",
                                  variable=self._use_brightness_var)
        brt_cb.grid(row=len(brt_rows), column=0, columnspan=2, sticky='w', pady=(4, 0))
        self._tip(brt_cb,
                  "Extract mean pixel brightness in each paw ROI during the\n"
                  "frames that pass the contour gate. Requires a video file for\n"
                  "each session. Skipped if video is missing.")

        ttk.Button(brt_lf, text="Preview brightness…",
                   command=self._open_brightness_preview).grid(
            row=len(brt_rows) + 1, column=0, columnspan=2, sticky='w', pady=(4, 0))

        # Paw Contour + DLC Crop Offset panels removed 2026-08-28: the
        # Contact Detection section above governs the contour gate (preset,
        # method, area band), per-paw ROI is editable inside the contour
        # preview, and crop offsets are editable in the brightness preview.
        # All backing vars remain live and persisted.
        # The brightness weight, DLC Confidence Filter and Locomotion Filter
        # panels (and the locomotion preview) were removed 2026-09-18: none
        # of them reaches the contour or brightness results.

        # Statistics (consumed by the Statistics graphs and the Σ flip)
        st_lf = ttk.LabelFrame(adv_inner, text="Statistics", padding=5)
        st_lf.pack(fill='x', pady=(0, 6), padx=2)
        _st_chk = ttk.Checkbutton(st_lf, text="Show significance on graphs",
                                  variable=self._enable_stats_var)
        _st_chk.pack(anchor='w')
        self._tip(_st_chk,
                  "Annotate group graphs with significance markers and\n"
                  "enable the Statistics category / Σ views.")
        _st_row = ttk.Frame(st_lf)
        _st_row.pack(fill='x', pady=2)
        ttk.Label(_st_row, text="Paradigm:").pack(side='left')
        _st_par = ttk.Combobox(_st_row, textvariable=self._stats_paradigm_var,
                               state='readonly', width=13,
                               values=['parametric', 'nonparametric'])
        _st_par.pack(side='left', padx=4)
        ttk.Label(_st_row, text="α:").pack(side='left', padx=(8, 0))
        ttk.Spinbox(_st_row, from_=0.001, to=0.2, increment=0.005,
                    textvariable=self._stats_alpha_var, width=6,
                    format='%.3f').pack(side='left', padx=4)
        _st_ph = ttk.Checkbutton(
            st_lf, text="Per-bin post-hoc in timecourse statistics",
            variable=self._timecourse_posthoc_var)
        _st_ph.pack(anchor='w', pady=(2, 0))
        self._tip(_st_ph,
                  "Timecourse statistics additionally test each time bin\n"
                  "(Bonferroni-corrected across bins).")

        # ── Readiness wiring: re-evaluate the Run gate when inputs change ──
        for _rv in ('HL', 'HR'):
            self._role_vars[_rv].trace_add('write', self._update_readiness)
        self._use_brightness_var.trace_add('write', self._update_readiness)
        self._key_file_var.trace_add('write', self._update_readiness)

    # ── Right: results pane ────────────────────────────────────────────────

    def _build_results_pane(self, parent):
        # Topbar: Category → Graph selection + styling + stats flip.
        topbar = ttk.Frame(parent)
        topbar.pack(fill='x', padx=8, pady=(8, 2))

        ttk.Label(topbar, text="Category:").pack(side='left')
        self._cat_cb = ttk.Combobox(topbar, textvariable=self._cat_var,
                                    state='readonly', width=20, values=[])
        self._cat_cb.pack(side='left', padx=(3, 8))
        self._cat_cb.bind('<<ComboboxSelected>>', self._on_category_changed)
        Tip(self._cat_cb, "Metric family. Pick a category, then a graph.")

        ttk.Label(topbar, text="Graph:").pack(side='left')
        self._graph_cb = ttk.Combobox(topbar, textvariable=self._graph_var,
                                      state='readonly', width=34, values=[])
        self._graph_cb.pack(side='left', padx=(3, 8))
        self._graph_cb.bind('<<ComboboxSelected>>',
                            lambda e: self._render_current())
        Tip(self._graph_cb, "Graphs in this category. Built on demand.")

        style_btn = ttk.Button(topbar, text="🎨⚙", width=4,
                               command=self._open_style_dialog)
        style_btn.pack(side='left', padx=(0, 2))
        Tip(style_btn, "Group colors and plot options (error bars, lines,\n"
                       "markers, significance style…). Shared across tabs.")

        disp_btn = ttk.Button(topbar, text="Display…", width=9,
                              command=self._open_display_dialog)
        disp_btn.pack(side='left', padx=(0, 2))
        Tip(disp_btn, "Display options: treatment order, per-treatment\n"
                      "markers, timecourse window and re-bin.")

        self._stats_btn = ttk.Checkbutton(
            topbar, text="Σ Stats", style='Toolbutton',
            variable=self._stats_mode_var, command=self._render_current)
        self._stats_btn.pack(side='left', padx=(4, 0))
        Tip(self._stats_btn, "Flip between the graph and its statistics\n"
                             "(descriptives + tests) for the current metric.")

        self._desc_lbl = ttk.Label(parent, text='', foreground='#444',
                                   font=(FONT_FAMILY, 9, 'italic'),
                                   wraplength=900, justify='left')
        self._desc_lbl.pack(fill='x', padx=10, pady=(0, 2))
        # Wrap to the pane's real width: a fixed 900 px ran past the edge of a narrower
        # window and the end of the first line was cut off.

        def _fit_desc(e, lbl=self._desc_lbl):
            w = max(200, e.width - 4)
            if int(str(lbl.cget('wraplength'))) != w:
                lbl.config(wraplength=w)
        self._desc_lbl.bind('<Configure>', _fit_desc)

        # Run outcome (N analyzed / M skipped) - set by _on_analysis_complete
        self._outcome_lbl = ttk.Label(parent, text='', font=(FONT_FAMILY, 9))
        self._outcome_lbl.pack(fill='x', padx=10, pady=(0, 2))

        # Bottom: per-session table + run summary (collapsed by default -
        # packed before the graph container so it keeps the bottom slot).
        bottom = collapsible(parent, "Session table",
                             collapsed=True, side='bottom', fill='x',
                             padx=6, pady=(0, 6))
        self._build_session_table(bottom)

        # Graph container - gait_views renders the selected entry in here.
        self._graph_container = ttk.Frame(parent)
        self._graph_container.pack(fill='both', expand=True, padx=6, pady=2)
        self._placeholder_lbl = ttk.Label(
            self._graph_container,
            text="Run an analysis (or load a saved session) to see graphs here.",
            foreground='grey')
        self._placeholder_lbl.pack(expand=True)

    def _build_export_panel(self, parent):
        """Rail section: exports, Adjust Contact, saved sessions, log."""
        lf = ttk.LabelFrame(parent, text="Results & Export", padding=6)
        lf.pack(fill='x', padx=4, pady=(0, 8))

        btn_row = ttk.Frame(lf)
        btn_row.pack(fill='x', pady=(0, 2))
        self._export_sum_btn = ttk.Button(btn_row, text="Export Summary",
                                          command=self._export_summary,
                                          state='disabled')
        self._export_sum_btn.pack(side='left', padx=2)
        self._export_bin_btn = ttk.Button(btn_row, text="Export Bins",
                                          command=self._export_bins,
                                          state='disabled')
        self._export_bin_btn.pack(side='left', padx=2)
        self._adjust_contact_btn = ttk.Button(
            btn_row, text="Adjust Contact",
            command=self._open_contact_adjustment, state='disabled')
        self._adjust_contact_btn.pack(side='left', padx=2)
        Tip(self._adjust_contact_btn,
            "Re-compute the contour gate (area band) from the stored run -\n"
            "no video re-extraction. Uses the same metrics code as the\n"
            "analysis itself.")

        # Saved sessions - pick a previous run and reload it (no re-analysis).
        saved_row = ttk.Frame(lf)
        saved_row.pack(fill='x', pady=(0, 2))
        ttk.Label(saved_row, text="Saved:").pack(side='left')
        self._saved_combo = ttk.Combobox(saved_row, state='readonly', width=26,
                                         values=[])
        self._saved_combo.pack(side='left', padx=(4, 2))
        self._saved_load_btn = ttk.Button(saved_row, text="Load", width=5,
                                          command=self._load_selected_session,
                                          state='disabled')
        self._saved_load_btn.pack(side='left', padx=1)
        ttk.Button(saved_row, text="Save…", width=6,
                   command=self._save_current_session_named).pack(side='left',
                                                                  padx=1)
        self._saved_del_btn = ttk.Button(saved_row, text="Del", width=4,
                                         command=self._delete_selected_session,
                                         state='disabled')
        self._saved_del_btn.pack(side='left', padx=1)
        Tip(self._saved_combo,
            "Previous runs auto-save here; pick one and Load to restore\n"
            "results without re-analyzing.")

        # Log
        log_lf = ttk.LabelFrame(lf, text="Log", padding=3)
        log_lf.pack(fill='x', pady=(2, 0))
        self._log_text = tk.Text(log_lf, height=4, wrap='word',
                                 state='disabled', font=('Consolas', 8))
        log_sb = ttk.Scrollbar(log_lf, orient='vertical',
                               command=self._log_text.yview)
        self._log_text.config(yscrollcommand=log_sb.set)
        self._log_text.pack(side='left', fill='both', expand=True)
        log_sb.pack(side='right', fill='y')

    def _build_session_table(self, parent):
        # ── Key Metrics Summary ───────────────────────────────────
        self._summary_frame = ttk.LabelFrame(parent, text="Summary", padding=5)
        self._summary_frame.pack(fill='x', padx=4, pady=(0, 4))
        self._summary_placeholder = ttk.Label(
            self._summary_frame, text="No results yet",
            foreground='grey', font=('TkDefaultFont', 9))
        self._summary_placeholder.pack(anchor='w')

        # Results treeview
        res_lf = ttk.LabelFrame(parent, text="Results", padding=3)
        res_lf.pack(fill='both', expand=True, padx=4, pady=(0, 2))

        tree_frame = ttk.Frame(res_lf)
        tree_frame.pack(fill='both', expand=True)

        _RES_COLS = tuple(c[0] for c in self._RESULT_COLUMNS)
        self._res_tree = ttk.Treeview(tree_frame, columns=_RES_COLS,
                                      show='headings', height=8)
        for col, hdr, w, _src, _fmt, _tip in self._RESULT_COLUMNS:
            self._res_tree.heading(col, text=hdr)
            self._res_tree.column(col, width=w, stretch=(col == 'session'))

        # Column-header mouseover tooltips
        _COL_TIPS = {c[0]: c[5] for c in self._RESULT_COLUMNS}
        _tree_tip = [None]

        def _on_tree_motion(event):
            region = self._res_tree.identify_region(event.x, event.y)
            if region != 'heading':
                if _tree_tip[0]:
                    _tree_tip[0].destroy()
                    _tree_tip[0] = None
                return
            col_idx = int(self._res_tree.identify_column(event.x).lstrip('#')) - 1
            try:
                col_name = _RES_COLS[col_idx]
            except IndexError:
                return
            tip_text = _COL_TIPS.get(col_name)
            if not tip_text:
                if _tree_tip[0]:
                    _tree_tip[0].destroy()
                    _tree_tip[0] = None
                return
            if _tree_tip[0]:
                _tree_tip[0].destroy()
            tip = tk.Toplevel(self._res_tree)
            tip.wm_overrideredirect(True)
            tip.wm_geometry(
                f'+{self._res_tree.winfo_rootx() + event.x + 12}'
                f'+{self._res_tree.winfo_rooty() + event.y + 16}')
            tk.Label(tip, text=tip_text, background='#ffffcc', relief='solid',
                     borderwidth=1, font=(FONT_FAMILY, 9), wraplength=280,
                     justify='left').pack(ipadx=4, ipady=2)
            _tree_tip[0] = tip

        def _on_tree_leave(event):
            if _tree_tip[0]:
                _tree_tip[0].destroy()
                _tree_tip[0] = None

        self._res_tree.bind('<Motion>', _on_tree_motion)
        self._res_tree.bind('<Leave>',  _on_tree_leave)

        res_vsb = ttk.Scrollbar(tree_frame, orient='vertical',
                                command=self._res_tree.yview)
        res_hsb = ttk.Scrollbar(tree_frame, orient='horizontal',
                                command=self._res_tree.xview)
        self._res_tree.config(yscrollcommand=res_vsb.set,
                              xscrollcommand=res_hsb.set)
        self._res_tree.grid(row=0, column=0, sticky='nsew')
        res_vsb.grid(row=0, column=1, sticky='ns')
        res_hsb.grid(row=1, column=0, sticky='ew')
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)


    # ═══════════════════════════════════════════════════════════════════════
    # Graph registry: Category → Graph dropdowns driven by gait_views
    # ═══════════════════════════════════════════════════════════════════════

    def _project_folder(self):
        try:
            return (self._override_folder_var.get()
                    or self.app.current_project_folder.get())
        except Exception:
            return ''

    def _core_ctx(self, project_folder=None):
        """Build the gait_core context. Worker threads must pass a
        ``project_folder`` snapshotted on the main thread (Tk vars are not
        thread-safe); main-thread callers may leave it None."""
        if project_folder is None:
            project_folder = self._project_folder()
        return gc.GaitContext(project_folder=project_folder,
                              pawlike_thresholds=self._pawlike_thresholds)

    def _gait_display(self):
        """Gait-specific display prefs stored in PixelPaws_project.json."""
        proj = self._project_folder()
        try:
            with open(os.path.join(proj, 'PixelPaws_project.json'),
                      encoding='utf-8') as f:
                return dict(json.load(f).get('gait_display', {}))
        except Exception:
            return {}

    def _save_gait_display(self, gd):
        proj = self._project_folder()
        path = os.path.join(proj, 'PixelPaws_project.json')
        try:
            data = {}
            if os.path.isfile(path):
                with open(path, encoding='utf-8') as f:
                    data = json.load(f)
            data['gait_display'] = gd
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self._log_ui(f"(could not save display options: {e})")

    def _graph_cfg(self):
        """Synthesize the builders' graph_cfg from the shared plot style +
        gait display extras (replaces the old 800-line settings dialog)."""
        df = self._summary_df
        treatments = []
        if df is not None and 'treatment' in df.columns:
            treatments = [str(t) for t in df['treatment'].dropna().unique()
                          if str(t).strip()]
        if not treatments:
            treatments = ['All sessions']
        gd = self._gait_display()
        order = [t for t in gd.get('order', []) if t in treatments]
        order += [t for t in treatments if t not in order]

        proj = self._project_folder()
        if ps is not None:
            opts = ps.get_options(proj)
            colors = ps.get_colors(proj, order)
        else:  # pragma: no cover
            opts = {}
            colors = {}
        try:
            _alpha = float(self._stats_alpha_var.get())
        except Exception:
            _alpha = 0.05
        lw = float(opts.get('line_width', 1.8))
        raw_ls = opts.get('line_styles', {}) or {}
        ms = float(opts.get('marker_size', 5.0))
        cfg = {
            'colors': colors,
            'order': order,
            'time_window': gd.get('time_window'),
            'rebin_minutes': gd.get('rebin_minutes', 0),
            'error_type': opts.get('error_type', 'SEM'),
            'error_display': opts.get('error_display', 'caps'),
            'show_individual': bool(opts.get('show_individual', False)),
            'sig_style': opts.get('sig_style', 'asterisk'),
            'show_stats': bool(self._enable_stats_var.get()),
            'marker_size': ms,
            'marker_shape': gd.get('marker_shape', 'o'),
            'line_widths': {g: lw for g in order},
            'line_styles': {g: raw_ls.get(g, '-') for g in order},
            'marker_sizes': {g: gd.get('marker_sizes', {}).get(g, ms)
                             for g in order},
            'marker_shapes': {g: gd.get('marker_shapes', {}).get(g, 'o')
                              for g in order},
            # builders expect matplotlib fillstyle strings, not booleans
            'marker_fills': {g: ('full' if gd.get('marker_fills', {})
                                 .get(g, True) in (True, 'full', 1)
                                 else 'none') for g in order},
            'marker_edge_colors': {g: gd.get('marker_edge_colors', {}).get(g)
                                   for g in order},
            'opacities': {g: gd.get('opacities', {}).get(g, 1.0)
                          for g in order},
            'graph_sets': {**{k: True for k in ('brightness', 'paw_contour',
                                                'statistics')},
                           'full_stance': bool(gd.get('full_stance',
                                                      False))},
            # Stats knobs consumed by the Statistics category / Σ views.
            'stats_alpha': _alpha,
            'stats_paradigm': self._stats_paradigm_var.get(),
            'stats_test': self._stats_test_var.get(),
            'timecourse_posthoc': bool(self._timecourse_posthoc_var.get()),
            'analysis_dir': gc.analysis_dir(proj or ''),
        }
        return cfg

    def _make_host(self):
        return gait_views.ViewHost(
            summary_df=self._summary_df,
            bins_df=(self._bins_df
                     if self._bins_df is not None and not self._bins_df.empty
                     else None),
            intermediates=self._session_intermediates,
            cfg=self._graph_cfg(),
            injured_paw=self._injured_paw_var.get(),
            enable_stats=lambda: bool(self._enable_stats_var.get()),
            log=self._log_ui,
            tip=self._tip,
            pawlike_thresholds=self._pawlike_thresholds,
            on_pawlike_change=lambda: self._populate_registry(keep=True),
        )

    def _populate_registry(self, keep=True):
        """(Re)build the Category/Graph registry from the current results."""
        if (gait_views is None or not _PLOT_OK
                or self._summary_df is None or self._summary_df.empty):
            if (self._summary_df is not None and not self._summary_df.empty
                    and (gait_views is None or not _PLOT_OK)):
                self._log_ui("Graphs unavailable: gait_views/matplotlib "
                             "failed to load.")
            return
        try:
            self._host = self._make_host()
            self._registry = gait_views.build_registry(self._host)
        except Exception as e:
            self._log_ui(f"Graph registry error: {e}")
            return
        cats = list(self._registry.keys())
        cur = self._cat_var.get()
        self._cat_cb['values'] = cats
        if not keep or cur not in cats:
            self._cat_var.set(cats[0] if cats else '')
        self._on_category_changed(keep=keep)

    def _on_category_changed(self, event=None, keep=False):
        entries = self._registry.get(self._cat_var.get(), [])
        names = [e['display_name'] for e in entries]
        self._graph_cb['values'] = names
        if not keep or self._graph_var.get() not in names:
            self._graph_var.set(names[0] if names else '')
        self._render_current()

    def _current_entry(self):
        for e in self._registry.get(self._cat_var.get(), []):
            if e['display_name'] == self._graph_var.get():
                return e
        return None

    def _render_current(self):
        entry = self._current_entry()
        if entry is None or self._host is None:
            return
        self._desc_lbl.config(text=entry.get('description', ''))
        stats = bool(self._stats_mode_var.get() and entry.get('has_stats'))
        try:
            gait_views.render_entry(self._host, self._graph_container,
                                    entry, stats_mode=stats)
        except Exception as e:
            for w in self._graph_container.winfo_children():
                w.destroy()
            ttk.Label(self._graph_container,
                      text=f"Could not draw this graph:\n{e}",
                      foreground='#b00020').pack(expand=True)
            self._log_ui(f"Graph error ({entry.get('display_name')}): {e}")

    def _open_style_dialog(self):
        if ps is None:
            return
        df = self._summary_df
        groups = []
        if df is not None and 'treatment' in df.columns:
            groups = [str(t) for t in df['treatment'].dropna().unique()
                      if str(t).strip()]
        ps.open_options_dialog(self, self._project_folder(), groups,
                               on_apply=lambda: self._populate_registry(True))

    def _open_display_dialog(self):
        """Display extras: treatment order, per-treatment markers,
        timecourse window / re-bin, Full-Stance contour categories."""
        gd = self._gait_display()
        cfg = self._graph_cfg()
        order = list(cfg['order'])

        win = tk.Toplevel(self)
        win.title("Paw Contour display")
        win.resizable(False, False)
        win.grab_set()
        frm = ttk.Frame(win, padding=10)
        frm.pack(fill='both', expand=True)

        # Treatment display order
        ord_lf = ttk.LabelFrame(frm, text="Treatment order", padding=6)
        ord_lf.grid(row=0, column=0, sticky='nsew', padx=(0, 8))
        lb = tk.Listbox(ord_lf, height=max(3, len(order)), exportselection=False,
                        width=22)
        for t in order:
            lb.insert('end', t)
        lb.pack(side='left')
        bcol = ttk.Frame(ord_lf)
        bcol.pack(side='left', padx=4)

        def _move(delta):
            sel = lb.curselection()
            if not sel:
                return
            i = sel[0]
            j = i + delta
            if not (0 <= j < lb.size()):
                return
            txt = lb.get(i)
            lb.delete(i)
            lb.insert(j, txt)
            lb.selection_set(j)
        ttk.Button(bcol, text="↑", width=3,
                   command=lambda: _move(-1)).pack(pady=2)
        ttk.Button(bcol, text="↓", width=3,
                   command=lambda: _move(+1)).pack(pady=2)

        # Per-treatment markers
        mk_lf = ttk.LabelFrame(frm, text="Markers", padding=6)
        mk_lf.grid(row=0, column=1, sticky='nsew')
        shapes = ['o', 's', '^', 'v', 'D', 'P', 'X', '*']
        mk_vars = {}
        for r, t in enumerate(order):
            ttk.Label(mk_lf, text=t).grid(row=r, column=0, sticky='w', pady=1)
            sv = tk.StringVar(value=cfg['marker_shapes'].get(t, 'o'))
            ttk.Combobox(mk_lf, textvariable=sv, values=shapes, width=3,
                         state='readonly').grid(row=r, column=1, padx=3)
            fv = tk.BooleanVar(
                value=gd.get('marker_fills', {}).get(t, True)
                in (True, 'full', 1))
            ttk.Checkbutton(mk_lf, text="filled", variable=fv).grid(
                row=r, column=2, padx=3)
            ov = tk.DoubleVar(value=float(cfg['opacities'].get(t, 1.0)))
            ttk.Spinbox(mk_lf, from_=0.1, to=1.0, increment=0.1, width=4,
                        textvariable=ov, format='%.1f').grid(row=r, column=3)
            mk_vars[t] = (sv, fv, ov)

        # Timecourse window / re-bin
        tc_lf = ttk.LabelFrame(frm, text="Timecourse", padding=6)
        tc_lf.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(8, 0))
        ttk.Label(tc_lf, text="Show first (min, blank = all):").grid(
            row=0, column=0, sticky='w')
        tw_var = tk.StringVar(
            value='' if gd.get('time_window') in (None, '')
            else str(gd.get('time_window')))
        ttk.Entry(tc_lf, textvariable=tw_var, width=6).grid(row=0, column=1,
                                                            padx=4)
        ttk.Label(tc_lf, text="Re-bin to (min, 0 = native):").grid(
            row=1, column=0, sticky='w')
        rb_var = tk.DoubleVar(value=float(gd.get('rebin_minutes', 0) or 0))
        ttk.Spinbox(tc_lf, from_=0, to=60, increment=0.5, width=6,
                    textvariable=rb_var).grid(row=1, column=1, padx=4)
        fs_var = tk.BooleanVar(value=bool(gd.get('full_stance', False)))
        _fs_chk = ttk.Checkbutton(
            tc_lf, text="Show Full-Stance contour categories",
            variable=fs_var)
        _fs_chk.grid(row=2, column=0, columnspan=2, sticky='w', pady=(4, 0))
        Tip(_fs_chk, "Adds the Paw Contour - Full Stance categories\n"
                     "(contour metrics restricted to frames where every\n"
                     "contour paw passes the gate; under the both-hind\n"
                     "contour gate these match the main categories).")

        def _apply():
            new_gd = dict(gd)
            new_gd['order'] = list(lb.get(0, 'end'))
            new_gd['marker_shapes'] = {t: v[0].get()
                                       for t, v in mk_vars.items()}
            new_gd['marker_fills'] = {t: bool(v[1].get())
                                      for t, v in mk_vars.items()}
            new_gd['opacities'] = {t: float(v[2].get())
                                   for t, v in mk_vars.items()}
            tw = tw_var.get().strip()
            try:
                new_gd['time_window'] = float(tw) if tw else None
            except ValueError:
                new_gd['time_window'] = None
            new_gd['rebin_minutes'] = float(rb_var.get())
            new_gd['full_stance'] = bool(fs_var.get())
            self._save_gait_display(new_gd)
            win.destroy()
            self._populate_registry(keep=True)

        btns = ttk.Frame(frm)
        btns.grid(row=2, column=0, columnspan=2, pady=(10, 0))
        ttk.Button(btns, text="Apply", command=_apply).pack(side='left', padx=6)
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side='left',
                                                                  padx=6)

    def _tip(self, widget, text):
        _ToolTip(widget, text)

    # ═══════════════════════════════════════════════════════════════════════
    # Fore-paw toggle
    # ═══════════════════════════════════════════════════════════════════════

    def _on_use_fore_changed(self):
        enabled = self._use_fore_var.get()
        state = 'normal' if enabled else 'disabled'
        for role in ('FL', 'FR'):
            self._role_combos[role].configure(state=state)
        # The height threshold / window only decide fore-paw contact.
        for w in getattr(self, '_fore_contact_widgets', []):
            try:
                w.configure(state=state)
            except tk.TclError:
                pass
        if not enabled:
            for role in ('FL', 'FR'):
                self._role_vars[role].set('')

    # ═══════════════════════════════════════════════════════════════════════
    # Thread-safe logging
    # ═══════════════════════════════════════════════════════════════════════

    def _log(self, msg: str):
        self.app.root.after(0, self._log_ui, msg)

    def _log_ui(self, msg: str):
        self._log_text.config(state='normal')
        self._log_text.insert('end', msg + '\n')
        self._log_text.see('end')
        self._log_text.config(state='disabled')

    def on_project_changed(self):
        """Called by PixelPawsGUI._on_project_folder_changed."""
        self._scan_sessions()
        self._refresh_lick_behaviors()
        self._refresh_saved_sessions()

    # ═══════════════════════════════════════════════════════════════════════
    # Session scanning
    # ═══════════════════════════════════════════════════════════════════════

    def _browse_sessions_folder(self):
        folder = filedialog.askdirectory(title="Select sessions folder")
        if folder:
            self._override_folder_var.set(folder)
            self._scan_sessions()

    def _session_cache_status(self, session_name, folder=None):
        """Report which extraction caches exist for a session: 'brt+contour',
        'brt', 'contour', or '-'. Checks both the current and legacy cache dirs."""
        folder = folder or self._override_folder_var.get() or \
            self.app.current_project_folder.get()
        if not folder:
            return '-'
        import glob as _glob
        has_brt = has_ctr = False
        for _d in (gc.ANALYSIS_DIR,) + gc.LEGACY_ANALYSIS_DIRS:
            base = os.path.join(folder, _d)
            if not os.path.isdir(base):
                continue
            if _glob.glob(os.path.join(base, f'{session_name}_brt_*.csv')):
                has_brt = True
            if _glob.glob(os.path.join(base, f'{session_name}_contour_*.csv')):
                has_ctr = True
        if has_brt and has_ctr:
            return 'brt+contour'
        if has_ctr:
            return 'contour'
        if has_brt:
            return 'brt'
        return '-'

    def _auto_populate_bodyparts(self, h5_path: str):
        """Detect body parts from a DLC h5 file and fill paw comboboxes."""
        try:
            df = pd.read_hdf(h5_path)
            if isinstance(df.columns, pd.MultiIndex):
                bps = list(df.columns.get_level_values(1).unique())
            else:
                bps = list({c.rsplit('_', 1)[0] for c in df.columns
                            if c.endswith(('_x', '_y'))})
            self._bodyparts = sorted(bps)
        except Exception:
            self._bodyparts = []

        for cb in self._role_combos.values():
            cb.config(values=self._bodyparts)

    def _autodetect_bodyparts(self):
        """Button: re-scan first DLC h5 and fill combos."""
        for sess in self._sessions:
            if sess.get('dlc') and os.path.isfile(sess['dlc']):
                self._auto_populate_bodyparts(sess['dlc'])
                self._log_ui(f"Detected {len(self._bodyparts)} body parts")
                return
        messagebox.showinfo("No sessions", "No sessions found - press Rescan in the Data section.", parent=self)

    # ═══════════════════════════════════════════════════════════════════════
    # Licking-frame exclusion (behavior predictions on disk)
    # ═══════════════════════════════════════════════════════════════════════

    def _scan_lick_behaviors(self, folder=None):
        """Behavior names available as predictions in the project's results/ folder.
        Extracted from prediction filenames (robust to per-session subfolders),
        with a consolidated per_frame header fallback."""
        folder = folder or self._project_folder()
        names = set()
        if not folder:
            return []
        results = os.path.join(folder, 'results')
        if not os.path.isdir(results):
            return []
        import glob as _glob
        sessions = {s['session_name'] for s in getattr(self, '_sessions', [])}

        def _is_session_named(name):
            return any(name == sn or name.startswith(sn + '_') or sn in name
                       for sn in sessions)

        # Preferred: immediate behavior subfolders (results/{behavior}/…) that
        # actually hold prediction files - this is the canonical batch layout.
        for entry in os.listdir(results):
            p = os.path.join(results, entry)
            if not os.path.isdir(p) or entry.lower() == 'per_frame':
                continue
            if _is_session_named(entry):
                continue
            if _glob.glob(os.path.join(p, '*_predictions.csv')) or \
               _glob.glob(os.path.join(p, '**', '*_predictions.csv'), recursive=True):
                names.add(entry)
        # Fallback: parse behavior from filenames, dropping session-named noise.
        if not names:
            for path in _glob.glob(os.path.join(results, '**', '*_predictions.csv'),
                                   recursive=True):
                beh = gc.extract_behavior_name(path)
                if beh and not _is_session_named(beh):
                    names.add(beh)
        # Last resort: consolidated per_frame headers.
        if not names:
            pf = os.path.join(results, 'per_frame')
            if os.path.isdir(pf):
                for fn in os.listdir(pf):
                    if fn.endswith('_frames.csv'):
                        try:
                            cols = pd.read_csv(os.path.join(pf, fn), nrows=0).columns
                            for c in cols:
                                if c.endswith('_pred'):
                                    names.add(c[:-5])
                        except Exception:
                            pass
                        break
        return sorted(names)

    def _guess_lick_behavior(self):
        """Best-guess licking behavior from the available list."""
        for b in getattr(self, '_lick_behaviors', []):
            if 'lick' in b.lower():
                return b
        return self._lick_behaviors[0] if getattr(self, '_lick_behaviors', None) else ''

    def _refresh_lick_behaviors(self):
        """Populate the licking behavior combo from disk; enable/disable the toggle."""
        if not hasattr(self, '_lick_combo'):
            return
        self._lick_behaviors = self._scan_lick_behaviors()
        self._lick_combo.config(values=self._lick_behaviors)
        if self._lick_behaviors and not self._lick_behavior_var.get():
            self._lick_behavior_var.set(self._guess_lick_behavior())
        if not self._lick_behaviors:
            self._exclude_lick_var.set(False)
            self._exclude_lick_chk.config(state='disabled')
        else:
            self._exclude_lick_chk.config(state='normal')
            # Auto-enable when the active preset requests it and predictions exist.
            preset = self.GAIT_PRESETS.get(
                self._preset_var.get() if hasattr(self, '_preset_var') else '', {})
            if preset.get('exclude_licking') and not self._exclude_lick_var.get():
                self._exclude_lick_var.set(True)
                if not self._lick_behavior_var.get():
                    self._lick_behavior_var.set(self._guess_lick_behavior())
        self._on_lick_toggle()

    def _on_lick_toggle(self):
        """Enable/disable the behavior picker + threshold and update the hint."""
        if not hasattr(self, '_lick_combo'):
            return
        on = bool(self._exclude_lick_var.get()) and bool(self._lick_behaviors)
        self._lick_combo.config(state='readonly' if on else 'disabled')
        self._lick_thr_spin.config(state='normal' if on else 'disabled')
        if not self._lick_behaviors:
            self._lick_hint_lbl.config(
                text="No behavior predictions found - run classifiers first to "
                     "enable licking exclusion.")
        elif on:
            self._lick_hint_lbl.config(
                text=f"Excluding frames predicted as '{self._lick_behavior_var.get()}' "
                     f"(≥ threshold) from every Paw Contour metric.")
        else:
            self._lick_hint_lbl.config(text="")
        try:
            self._update_readiness()
        except Exception:
            pass

    def _on_key_combo_selected(self, event=None):
        idx = self._key_combo.current()
        if 0 <= idx < len(self._key_scan_paths):
            self._load_key_file(self._key_scan_paths[idx])

    def _browse_key_file(self):
        path = filedialog.askopenfilename(
            title="Select Key File",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv"),
                       ("All files", "*.*")],
            parent=self)
        if path:
            self._key_file_var.set(path)
            self._load_key_file(path)

    def _generate_key_file(self):
        """Open KeyFileGeneratorDialog to create a key_file.csv for this project."""
        try:
            from project_setup import KeyFileGeneratorDialog
        except ImportError:
            messagebox.showerror("Unavailable",
                                 "project_setup.py not found - cannot open key file generator.",
                                 parent=self)
            return

        folder = getattr(self.app, 'current_project_folder', None)
        folder = folder.get() if folder else ''
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("No Project",
                                   "Please open a project first so PixelPaws knows "
                                   "where to find your videos and save the key file.",
                                   parent=self)
            return

        import glob as _g
        videos_dir = os.path.join(folder, 'videos')
        _seen = {}
        for ext in ('.mp4', '.avi', '.mov', '.wmv', '.MP4', '.AVI', '.MOV', '.WMV'):
            for vf in _g.glob(os.path.join(videos_dir, f'*{ext}')):
                _seen[os.path.normcase(vf)] = vf
        basenames = [os.path.splitext(os.path.basename(v))[0]
                     for v in sorted(_seen.values())]
        if not basenames:
            messagebox.showinfo("No Videos",
                                "No video files found in videos/.\n"
                                "Add your videos first, then generate the key file.",
                                parent=self)
            return

        existing = {}
        key_path = os.path.join(folder, 'key_file.csv')
        if os.path.isfile(key_path):
            try:
                import csv
                with open(key_path, newline='') as f:
                    for row in csv.DictReader(f):
                        s = row.get('Subject', '').strip()
                        t = row.get('Treatment', '').strip()
                        if s:
                            existing[s] = t
            except Exception:
                pass

        def _on_save(data):
            if hasattr(self.app, 'key_file_data'):
                self.app.key_file_data = data
            saved_path = os.path.join(folder, 'key_file.csv')
            if os.path.isfile(saved_path):
                self._key_file_var.set(saved_path)
                self._load_key_file(saved_path)

        KeyFileGeneratorDialog(
            self.winfo_toplevel(), folder, basenames,
            existing_groups=existing, on_save=_on_save)

    def _load_key_file(self, path: str):
        try:
            df = pd.read_excel(path) if path.endswith('.xlsx') else pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("Key file error", str(e), parent=self)
            return
        missing = [c for c in ('Subject', 'Treatment') if c not in df.columns]
        if missing:
            messagebox.showerror("Invalid key file",
                                 f"Missing columns: {', '.join(missing)}", parent=self)
            return
        df['Subject'] = df['Subject'].astype(str)
        self._key_df = df
        n_subj = len(df)
        treatments = df['Treatment'].unique()
        self._key_status_lbl.config(
            text=(f"✓ {n_subj} subjects, {len(treatments)} treatment(s): "
                  f"{', '.join(map(str, treatments))}"),
            foreground='green')
        self._log_ui(f"Key file loaded: {os.path.basename(path)}")

        # Refresh the picker's Subject column now that the key is known.
        if getattr(self, '_sessions', None):
            self._publish_sessions()

    # ═══════════════════════════════════════════════════════════════════════
    # Subject / treatment resolution
    # ═══════════════════════════════════════════════════════════════════════

    def _resolve_subject(self, session_name: str, prefix=None) -> str:
        """Extract subject ID via 4-strategy fallback (mirrors analysis_tab).

        ``prefix`` is the strip-prefix; worker threads pass a value snapshotted
        on the main thread (Tk vars are not thread-safe), main-thread callers
        may leave it None to read the entry."""
        stem = session_name

        # 1. Key-file token match
        if self._key_df is not None:
            tokens = stem.split('_')
            for subj in self._key_df['Subject']:
                if str(subj) in tokens:
                    return str(subj)
            for subj in self._key_df['Subject']:
                if f'_{subj}_' in f'_{stem}_':
                    return str(subj)

        # 2. Prefix strip
        pfx = (self._prefix_var.get() if prefix is None else prefix).strip()
        if pfx and stem.startswith(pfx):
            remainder = stem[len(pfx):]
            token = remainder.split('_')[0] if remainder else ''
            if token:
                return token

        # 3. PixelPaws_GUI legacy helper
        if _extract_sid is not None:
            sid = _extract_sid(session_name)
            if sid:
                return str(sid)

        # 4. First 4-digit token heuristic
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
    # Analysis: launch / cancel
    # ═══════════════════════════════════════════════════════════════════════

    def _apply_gait_preset(self, event=None):
        """Set the interdependent toggle vars (and, for richer presets, the ROI /
        bin / contour-band knobs) from the chosen Quick-setup preset."""
        preset = self.GAIT_PRESETS.get(self._preset_var.get())
        if not preset:
            return
        self._use_brightness_var.set(bool(preset['brightness']))
        self._paw_contour_var.set(bool(preset['contour']))
        self._use_fore_var.set(bool(preset['fore']))
        # Optional richer-preset knobs - applied only when the preset carries them,
        # so the simple 3-toggle presets keep their existing behaviour.
        if 'contour_roi' in preset:
            for _r in ('HL', 'HR'):
                if _r in self._contour_roi_size_vars:
                    self._contour_roi_size_vars[_r].set(int(preset['contour_roi']))
        if 'bin_seconds' in preset:
            self._bin_seconds_var.set(int(preset['bin_seconds']))
        if 'bin_unit' in preset:
            self._bin_unit_var.set(str(preset['bin_unit']))
        if 'contact_method' in preset:
            self._contact_method_var.set(str(preset['contact_method']))
        if 'contour_area_threshold' in preset:
            self._contour_area_thresh_var.set(int(preset['contour_area_threshold']))
        if 'contour_area_max' in preset:
            self._contour_area_max_var.set(int(preset['contour_area_max']))
        if 'exclude_licking' in preset:
            # Only enable if licking predictions are actually available on disk;
            # otherwise leave the toggle off (the control is disabled with a hint).
            want = bool(preset['exclude_licking'])
            if want and getattr(self, '_lick_behaviors', None):
                self._exclude_lick_var.set(True)
                if not self._lick_behavior_var.get():
                    self._lick_behavior_var.set(self._guess_lick_behavior())
            elif not want:
                self._exclude_lick_var.set(False)
        try:
            self._on_lick_toggle()
        except Exception:
            pass
        # keep the fore-paw mapping widgets in sync with the checkbox
        try:
            self._on_use_fore_changed()
        except Exception:
            pass
        self._update_readiness()

    def _check_readiness(self):
        """Return (ready, issues, notes).

        issues = hard blockers that must be fixed before a run can start.
        notes  = soft warnings that do not block (surfaced but non-fatal).
        This is the single source of truth for both the readiness strip and
        the guard at the top of ``_start_analysis``.
        """
        issues, notes = [], []

        sel = (self._session_filter.selected()
               if hasattr(self, '_session_filter') else [])
        if not sel:
            issues.append("Select at least one session (Data → Sessions)")

        hl = self._role_vars['HL'].get().strip() if 'HL' in self._role_vars else ''
        hr = self._role_vars['HR'].get().strip() if 'HR' in self._role_vars else ''
        if not hl or not hr:
            issues.append("Map HL and HR paws (Setup → Paw Mapping)")

        if (self._gate_4paw_var.get()
                and not (self._use_fore_var.get()
                         and self._role_vars['FL'].get().strip()
                         and self._role_vars['FR'].get().strip())):
            issues.append("4-paw gate needs fore paws mapped (Setup)")

        if self._key_df is None:
            notes.append("no key file - treatment will be blank")
        # Licking exclusion is part of the deployed contact gate: a licking paw is at the
        # mouth, not on the floor, and licking rates differ between groups, so including
        # those frames biases the intensity ratio. If predictions are absent the toggle
        # is silently off - say so, and say how to fix it.
        if (getattr(self, '_contact_method_var', None) is not None
                and self._contact_method_var.get() == 'contour_area'
                and not self._exclude_lick_var.get()):
            notes.append("licking frames will NOT be excluded - score licking first "
                         "(Run Classifiers), then Rescan; the manuscript gate drops them")

        if self._use_brightness_var.get() and sel:
            sel_names = set(sel)
            missing = [s['session_name'] for s in self._sessions
                       if s['session_name'] in sel_names and not s.get('video')]
            if missing:
                notes.append(f"{len(missing)} selected session(s) lack video - "
                             "they will be skipped")

        return (not issues), issues, notes

    def _update_readiness(self, *args):
        """Refresh the readiness strip and gate the Run button. Trace/UI callback."""
        if not hasattr(self, '_readiness_lbl'):
            return
        try:
            ready, issues, notes = self._check_readiness()
        except Exception:
            return

        running = bool(self._fit_thread and self._fit_thread.is_alive())
        if hasattr(self, '_run_btn'):
            self._run_btn.config(
                state=('disabled' if (running or not ready) else 'normal'))

        if running:
            self._readiness_lbl.config(text="Running…", foreground='grey')
            return
        if issues:
            self._readiness_lbl.config(text="⚠ " + "  •  ".join(issues),
                                       foreground='#b00020')
            return
        n_sel = len(self._session_filter.selected())
        bits = [f"✓ {n_sel} session(s)", "✓ paws HL,HR"]
        bits += ["⚠ " + n for n in notes]
        self._readiness_lbl.config(
            text="  ·  ".join(bits) + "  -  Ready to run",
            foreground=('#7a6a00' if notes else '#127a12'))

    # ── Brightness preview ───────────────────────────────────────────────

    def _open_brightness_preview(self):
        if not _CV2_OK:
            messagebox.showerror("Missing dependency",
                                 "OpenCV (cv2) is required for the brightness preview.",
                                 parent=self)
            return

        # --- find a usable session (needs video + DLC) ---
        selected = self._session_filter.selected()
        candidates = [s for s in self._sessions
                      if (not selected or s['session_name'] in selected)
                      and s.get('video') and os.path.isfile(s['video'])
                      and s.get('dlc')   and os.path.isfile(s['dlc'])]
        if not candidates:
            messagebox.showwarning(
                "No session",
                "Select a session that has both a video and DLC file.",
                parent=self)
            return

        sess = candidates[0]
        video_path = sess['video']
        dlc_path   = sess['dlc']

        # --- read DLC coordinates + paw heights ---
        active_paws = {r: self._role_vars[r].get().strip()
                       for r in self.ROLES if self._role_vars[r].get().strip()}
        active_bps  = list(set(active_paws.values()))
        try:
            ext      = PoseFeatureExtractor(active_bps)
            dlc_df   = ext.load_dlc_data(dlc_path)
            bp_xcord, bp_ycord, bp_prob = ext.get_bodypart_coords(dlc_df)
        except Exception as e:
            messagebox.showerror("DLC error", str(e), parent=self)
            return

        try:
            height_df = ext.calculate_paw_height(
                bp_xcord, bp_ycord, window=self._height_window_var.get())
        except Exception:
            height_df = None

        # --- open video ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Video error",
                                 f"Cannot open:\n{video_path}", parent=self)
            return
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # --- local preview vars ---
        contact_var    = tk.IntVar(value=self._contact_thresh_var.get())
        roi_vars       = {role: tk.IntVar(value=self._roi_size_vars[role].get())
                          for role in active_paws}
        crop_x_var     = tk.IntVar(value=self._crop_x_var.get())
        crop_y_var     = tk.IntVar(value=self._crop_y_var.get())

        # --- build window ---
        win = tk.Toplevel(self)
        win.title(f"Brightness Preview - {sess['session_name']}")
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        w, h = int(sw * 0.55), int(sh * 0.60)
        win.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        win.protocol('WM_DELETE_WINDOW', lambda: (cap.release(), win.destroy()))

        # Scrollable ctrl panel - pack FIRST so it gets priority for space
        ctrl_outer = ttk.Frame(win)
        ctrl_outer.pack(side='right', fill='both', padx=6, pady=6)
        ctrl_canvas = tk.Canvas(ctrl_outer, borderwidth=0, highlightthickness=0, width=300)
        ctrl_sb_y = ttk.Scrollbar(ctrl_outer, orient='vertical', command=ctrl_canvas.yview)
        ctrl_sb_x = ttk.Scrollbar(ctrl_outer, orient='horizontal', command=ctrl_canvas.xview)
        ctrl_canvas.configure(yscrollcommand=ctrl_sb_y.set, xscrollcommand=ctrl_sb_x.set)
        ctrl_sb_x.pack(side='bottom', fill='x')
        ctrl_sb_y.pack(side='right', fill='y')
        ctrl_canvas.pack(side='left', fill='both', expand=True)
        ctrl = ttk.Frame(ctrl_canvas)
        ctrl_win_id = ctrl_canvas.create_window((0, 0), window=ctrl, anchor='nw')
        ctrl.bind('<Configure>',
                  lambda e: ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox('all')))
        ctrl_canvas.bind('<Configure>',
                         lambda e: ctrl_canvas.itemconfig(ctrl_win_id, width=e.width))

        # Video canvas - packed after ctrl so it fills remaining space
        canvas = tk.Canvas(win, bg='black', width=480, height=380)
        canvas.pack(side='left', fill='both', expand=True, padx=4, pady=4)

        # Frame slider
        ttk.Label(ctrl, text="Frame:").pack(anchor='w')
        frame_var = tk.IntVar(value=0)
        frame_sb = ttk.Spinbox(ctrl, from_=0, to=max(n_frames - 1, 0),
                                textvariable=frame_var, width=8)
        frame_sb.pack(anchor='w', pady=(0, 2))
        frame_slider = ttk.Scale(ctrl, from_=0, to=max(n_frames - 1, 0),
                                  variable=frame_var, orient='vertical', length=180)
        frame_slider.pack(anchor='w', pady=(0, 8))

        # Brightness threshold slider
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=4)
        ttk.Label(ctrl, text="Brt threshold (0=auto):").pack(anchor='w')
        thresh_var = tk.IntVar(value=self._brt_thresh_var.get())
        thresh_sb = ttk.Spinbox(ctrl, from_=0, to=255, textvariable=thresh_var, width=8)
        thresh_sb.pack(anchor='w', pady=(0, 2))
        thresh_slider = ttk.Scale(ctrl, from_=0, to=255,
                                   variable=thresh_var, orient='vertical', length=160)
        thresh_slider.pack(anchor='w', pady=(0, 4))

        # Fore-paw contact threshold spinbox (hind contact is the contour band)
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=4)
        ttk.Label(ctrl, text="Fore-paw down height (px):").pack(anchor='w')
        contact_sb = ttk.Spinbox(ctrl, from_=0, to=500, textvariable=contact_var, width=8)
        contact_sb.pack(anchor='w', pady=(0, 6))

        # Crop offset spinboxes
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=4)
        ttk.Label(ctrl, text="Crop offset (px):").pack(anchor='w')
        crop_grid = ttk.Frame(ctrl)
        crop_grid.pack(anchor='w', pady=(0, 6))
        ttk.Label(crop_grid, text="X:").grid(row=0, column=0, sticky='w', padx=(0, 2))
        ttk.Spinbox(crop_grid, from_=0, to=4000, textvariable=crop_x_var,
                    width=6).grid(row=0, column=1, sticky='w')
        ttk.Label(crop_grid, text="Y:").grid(row=1, column=0, sticky='w', padx=(0, 2))
        ttk.Spinbox(crop_grid, from_=0, to=4000, textvariable=crop_y_var,
                    width=6).grid(row=1, column=1, sticky='w')

        # Per-paw ROI half-sizes
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=4)
        ttk.Label(ctrl, text="ROI half-size (px):").pack(anchor='w')
        roi_grid = ttk.Frame(ctrl)
        roi_grid.pack(anchor='w', pady=(0, 6))
        for i, (role, var) in enumerate(roi_vars.items()):
            ttk.Label(roi_grid, text=f"{role}:").grid(
                row=i // 2, column=(i % 2) * 2, sticky='w', padx=(0, 2))
            ttk.Spinbox(roi_grid, from_=5, to=200, textvariable=var, width=5).grid(
                row=i // 2, column=(i % 2) * 2 + 1, sticky='w', padx=(0, 6))

        # Per-paw readout labels
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=4)
        ttk.Label(ctrl, text="Paw readouts:").pack(anchor='w')
        paw_labels = {}
        for role, bp in active_paws.items():
            lbl = ttk.Label(ctrl, text=f"{role} ({bp}): -", wraplength=270)
            lbl.pack(anchor='w')
            paw_labels[role] = lbl

        # Auto-detect brightness label
        auto_lbl = ttk.Label(ctrl, text='', foreground='grey', wraplength=270)
        auto_lbl.pack(anchor='w', pady=(4, 0))

        # "Apply all to main settings" button
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=6)

        def _apply_all():
            self._brt_thresh_var.set(thresh_var.get())
            self._contact_thresh_var.set(contact_var.get())
            self._crop_x_var.set(crop_x_var.get())
            self._crop_y_var.set(crop_y_var.get())
            for role in active_paws:
                self._roi_size_vars[role].set(roi_vars[role].get())
            roi_str = ', '.join(f'{r}={roi_vars[r].get()}' for r in active_paws)
            self._log_ui(
                f"Applied: brt_thresh={thresh_var.get() or 'auto'}, "
                f"fore_contact_thresh={contact_var.get()}, "
                f"crop=({crop_x_var.get()},{crop_y_var.get()}), "
                f"ROI={{{roi_str}}}"
            )

        ttk.Button(ctrl, text="Apply all to main settings",
                   command=_apply_all).pack(fill='x', pady=(0, 2))

        # --- render function (debounced 50 ms) ---
        _after_id = [None]

        def _render(*_):
            if _after_id[0]:
                win.after_cancel(_after_id[0])
            _after_id[0] = win.after(50, _do_render)

        def _safe_int(var, default=0):
            try:
                return int(var.get())
            except (tk.TclError, ValueError):
                return default

        def _do_render():
            fi   = _safe_int(frame_var, 0)
            thr  = _safe_int(thresh_var, 0)
            cthr = _safe_int(contact_var, self._contact_thresh_var.get())
            cx   = _safe_int(crop_x_var, 0)
            cy   = _safe_int(crop_y_var, 0)

            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                return

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

            # Effective brightness threshold
            if thr == 0:
                effective = float(gray.mean() * 0.5)
                auto_lbl.config(text=f'Auto → {effective:.1f}')
            else:
                effective = float(thr)
                auto_lbl.config(text='')

            # Pixels below brightness threshold → dimmed blue
            vis = frame.copy()
            below_mask = gray < effective
            vis[below_mask] = (vis[below_mask] * 0.25).astype(np.uint8)
            vis[below_mask, 0] = np.clip(
                vis[below_mask, 0].astype(int) + 60, 0, 255).astype(np.uint8)

            _paw_colors = {'HL': (0, 255, 0), 'HR': (0, 0, 255),
                           'FL': (255, 255, 0), 'FR': (0, 165, 255)}

            # Draw ROI boxes per paw
            for role, bp in active_paws.items():
                x_col = next((c for c in bp_xcord.columns if bp.lower() in c.lower()), None)
                y_col = next((c for c in bp_ycord.columns if bp.lower() in c.lower()), None)
                if x_col is None or y_col is None:
                    continue
                if fi >= len(bp_xcord):
                    continue

                bx = int(bp_xcord[x_col].iloc[fi]) + cx
                by = int(bp_ycord[y_col].iloc[fi]) + cy

                rh = _safe_int(roi_vars[role], 25)   # half-size in px
                fh, fw = frame.shape[:2]
                x1 = max(0, bx - rh);  x2 = min(fw, bx + rh)
                y1 = max(0, by - rh);  y2 = min(fh, by + rh)

                # Brightness in ROI
                roi_gray = gray[y1:y2, x1:x2].copy()
                roi_gray[roi_gray < effective] = 1.0
                brt = float(roi_gray.mean()) if roi_gray.size > 0 else 0.0

                # Paw height + contact state. Only the fore paws use the
                # height threshold; hind-paw contact is the contour band
                # (Preview contour…), so no height call is drawn for them.
                paw_h = None
                if height_df is not None:
                    h_col = next((c for c in height_df.columns
                                  if bp.lower() in c.lower()), None)
                    if h_col and fi < len(height_df):
                        paw_h = float(height_df[h_col].iloc[fi])
                in_contact = (role in ('FL', 'FR')
                              and paw_h is not None and paw_h < cthr)

                # Update paw label
                h_str = f'h={paw_h:.1f}' if paw_h is not None else 'h=?'
                contact_str = ' ✓' if in_contact else ''
                paw_labels[role].config(
                    text=f"{role} ({bp}): brt={brt:.1f} {h_str}{contact_str}")

                color = _paw_colors.get(role, (255, 255, 255))

                # Contact fill overlay (semi-transparent)
                if in_contact:
                    overlay = vis.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)

                lw = 3 if in_contact else 1
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, lw)
                cv2.circle(vis, (bx, by), 4, color, -1)

                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(role, font, 0.55, 1)
                cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw + 4, y1), (0, 0, 0), -1)
                cv2.putText(vis, role, (x1 + 2, y1 - 2), font, 0.55, color, 1)

            # HUD text
            hud = f'brt_thr={effective:.1f}  fore_contact_thr={cthr}'
            if cx or cy:
                hud += f'  crop=({cx},{cy})'
            cv2.putText(vis, hud, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)

            # Display on canvas
            cw = canvas.winfo_width()  or 680
            ch = canvas.winfo_height() or 520
            vh, vw = vis.shape[:2]
            scale = min(cw / vw, ch / vh)
            nw, nh = int(vw * scale), int(vh * scale)
            vis_small = cv2.resize(vis, (nw, nh))
            rgb = cv2.cvtColor(vis_small, cv2.COLOR_BGR2RGB)
            from PIL import Image, ImageTk
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            canvas.delete('all')
            canvas.create_image(cw // 2, ch // 2, image=photo, anchor='center')
            canvas.image = photo   # keep reference

        # Wire all traces → debounced render
        frame_var.trace_add('write', _render)
        thresh_var.trace_add('write', _render)
        contact_var.trace_add('write', _render)
        crop_x_var.trace_add('write', _render)
        crop_y_var.trace_add('write', _render)
        for var in roi_vars.values():
            var.trace_add('write', _render)
        win.bind('<Configure>', _render)
        win.after(100, _do_render)   # initial render after window maps

    # ── Contour preview ─────────────────────────────────────────────────────

    def _open_contour_preview(self):
        if not _CV2_OK:
            messagebox.showerror("Missing dependency",
                                 "OpenCV (cv2) is required for the contour preview.",
                                 parent=self)
            return

        # --- find a usable session (needs video + DLC) ---
        selected = self._session_filter.selected()
        candidates = [s for s in self._sessions
                      if (not selected or s['session_name'] in selected)
                      and s.get('video') and os.path.isfile(s['video'])
                      and s.get('dlc')   and os.path.isfile(s['dlc'])]
        if not candidates:
            messagebox.showwarning(
                "No session",
                "Select a session that has both a video and DLC file.",
                parent=self)
            return

        sess = candidates[0]
        video_path = sess['video']
        dlc_path   = sess['dlc']

        # --- read DLC coordinates ---
        active_paws = {r: self._role_vars[r].get().strip()
                       for r in self.ROLES if self._role_vars[r].get().strip()}
        active_bps  = list(set(active_paws.values()))
        try:
            ext      = PoseFeatureExtractor(active_bps)
            dlc_df   = ext.load_dlc_data(dlc_path)
            bp_xcord, bp_ycord, bp_prob = ext.get_bodypart_coords(dlc_df)
        except Exception as e:
            messagebox.showerror("DLC error", str(e), parent=self)
            return

        # --- open video ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Video error",
                                 f"Cannot open:\n{video_path}", parent=self)
            return
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # --- local preview vars ---
        roi_vars       = {role: tk.IntVar(value=self._contour_roi_size_vars[role].get())
                          for role in active_paws}
        crop_x_var     = tk.IntVar(value=self._crop_x_var.get())
        crop_y_var     = tk.IntVar(value=self._crop_y_var.get())
        thresh_mode_var = tk.StringVar(value='otsu')
        manual_thresh_var = tk.IntVar(value=128)
        blur_kernel_var = tk.IntVar(value=3)
        min_area_var    = tk.IntVar(value=10)

        # --- build window ---
        win = tk.Toplevel(self)
        win.title(f"Contour Preview \u2014 {sess['session_name']}")
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        w, h = int(sw * 0.55), int(sh * 0.60)
        win.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        win.protocol('WM_DELETE_WINDOW', lambda: (cap.release(), win.destroy()))

        # Scrollable ctrl panel - pack FIRST so it gets priority for space
        ctrl_outer = ttk.Frame(win)
        ctrl_outer.pack(side='right', fill='both', padx=6, pady=6)
        ctrl_canvas = tk.Canvas(ctrl_outer, borderwidth=0, highlightthickness=0, width=300)
        ctrl_sb_y = ttk.Scrollbar(ctrl_outer, orient='vertical', command=ctrl_canvas.yview)
        ctrl_sb_x = ttk.Scrollbar(ctrl_outer, orient='horizontal', command=ctrl_canvas.xview)
        ctrl_canvas.configure(yscrollcommand=ctrl_sb_y.set, xscrollcommand=ctrl_sb_x.set)
        ctrl_sb_x.pack(side='bottom', fill='x')
        ctrl_sb_y.pack(side='right', fill='y')
        ctrl_canvas.pack(side='left', fill='both', expand=True)
        ctrl = ttk.Frame(ctrl_canvas)
        ctrl_win_id = ctrl_canvas.create_window((0, 0), window=ctrl, anchor='nw')
        ctrl.bind('<Configure>',
                  lambda e: ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox('all')))
        ctrl_canvas.bind('<Configure>',
                         lambda e: ctrl_canvas.itemconfig(ctrl_win_id, width=e.width))

        # Video canvas - packed after ctrl so it fills remaining space
        canvas = tk.Canvas(win, bg='black', width=480, height=380)
        canvas.pack(side='left', fill='both', expand=True, padx=4, pady=4)

        # Frame slider
        ttk.Label(ctrl, text="Frame:").pack(anchor='w')
        frame_var = tk.IntVar(value=0)
        frame_sb = ttk.Spinbox(ctrl, from_=0, to=max(n_frames - 1, 0),
                                textvariable=frame_var, width=8)
        frame_sb.pack(anchor='w', pady=(0, 2))
        frame_slider = ttk.Scale(ctrl, from_=0, to=max(n_frames - 1, 0),
                                  variable=frame_var, orient='vertical', length=180)
        frame_slider.pack(anchor='w', pady=(0, 8))

        # Contour threshold mode
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=4)
        ttk.Label(ctrl, text="Threshold mode:").pack(anchor='w')
        ttk.Radiobutton(ctrl, text="Otsu (auto)", variable=thresh_mode_var,
                        value='otsu').pack(anchor='w')
        ttk.Radiobutton(ctrl, text="Manual", variable=thresh_mode_var,
                        value='manual').pack(anchor='w')

        # Manual threshold
        ttk.Label(ctrl, text="Manual threshold:").pack(anchor='w', pady=(4, 0))
        manual_sb = ttk.Spinbox(ctrl, from_=0, to=255, textvariable=manual_thresh_var,
                                width=8, state='disabled')
        manual_sb.pack(anchor='w', pady=(0, 4))

        def _on_mode_change(*_):
            manual_sb.config(state='normal' if thresh_mode_var.get() == 'manual' else 'disabled')
        thresh_mode_var.trace_add('write', _on_mode_change)

        # Gaussian blur kernel
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=4)
        ttk.Label(ctrl, text="Gaussian blur kernel:").pack(anchor='w')
        ttk.Spinbox(ctrl, values=(1, 3, 5, 7), textvariable=blur_kernel_var,
                    width=8).pack(anchor='w', pady=(0, 4))

        # Min contour area
        ttk.Label(ctrl, text="Min contour area (px\u00b2):").pack(anchor='w')
        ttk.Spinbox(ctrl, from_=0, to=5000, textvariable=min_area_var,
                    width=8).pack(anchor='w', pady=(0, 4))

        # Per-paw ROI half-sizes
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=4)
        ttk.Label(ctrl, text="Contour ROI half-size (px):").pack(anchor='w')
        roi_grid = ttk.Frame(ctrl)
        roi_grid.pack(anchor='w', pady=(0, 6))
        for i, (role, var) in enumerate(roi_vars.items()):
            ttk.Label(roi_grid, text=f"{role}:").grid(
                row=i // 2, column=(i % 2) * 2, sticky='w', padx=(0, 2))
            ttk.Spinbox(roi_grid, from_=5, to=200, textvariable=var, width=5).grid(
                row=i // 2, column=(i % 2) * 2 + 1, sticky='w', padx=(0, 6))

        # Crop offset spinboxes
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=4)
        ttk.Label(ctrl, text="Crop offset (px):").pack(anchor='w')
        crop_grid = ttk.Frame(ctrl)
        crop_grid.pack(anchor='w', pady=(0, 6))
        ttk.Label(crop_grid, text="X:").grid(row=0, column=0, sticky='w', padx=(0, 2))
        ttk.Spinbox(crop_grid, from_=0, to=4000, textvariable=crop_x_var,
                    width=6).grid(row=0, column=1, sticky='w')
        ttk.Label(crop_grid, text="Y:").grid(row=1, column=0, sticky='w', padx=(0, 2))
        ttk.Spinbox(crop_grid, from_=0, to=4000, textvariable=crop_y_var,
                    width=6).grid(row=1, column=1, sticky='w')

        # Per-paw readout labels
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=4)
        ttk.Label(ctrl, text="Paw readouts:").pack(anchor='w')
        paw_labels = {}
        for role, bp in active_paws.items():
            lbl = ttk.Label(ctrl, text=f"{role} ({bp}): \u2014", wraplength=270)
            lbl.pack(anchor='w')
            paw_labels[role] = lbl

        # "Apply contour ROI sizes to main settings" button
        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=6)

        def _apply_roi():
            self._crop_x_var.set(crop_x_var.get())
            self._crop_y_var.set(crop_y_var.get())
            for role in active_paws:
                self._contour_roi_size_vars[role].set(roi_vars[role].get())
            roi_str = ', '.join(f'{r}={roi_vars[r].get()}' for r in active_paws)
            self._log_ui(
                f"Applied: crop=({crop_x_var.get()},{crop_y_var.get()}), "
                f"contour ROI={{{roi_str}}}"
            )

        ttk.Button(ctrl, text="Apply contour ROI to main settings",
                   command=_apply_roi).pack(fill='x', pady=(0, 2))

        # --- render function (debounced 50 ms) ---
        _after_id = [None]

        def _render(*_):
            if _after_id[0]:
                win.after_cancel(_after_id[0])
            _after_id[0] = win.after(50, _do_render)

        def _safe_int(var, default=0):
            try:
                return int(var.get())
            except (tk.TclError, ValueError):
                return default

        def _do_render():
            fi  = _safe_int(frame_var, 0)
            cx  = _safe_int(crop_x_var, 0)
            cy  = _safe_int(crop_y_var, 0)
            k   = _safe_int(blur_kernel_var, 3)
            min_ca = _safe_int(min_area_var, 10)
            m_thr  = _safe_int(manual_thresh_var, 128)

            # Ensure kernel is odd and >= 1
            if k < 1:
                k = 1
            if k % 2 == 0:
                k += 1

            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                return

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

            _paw_colors = {'HL': (0, 255, 0), 'HR': (0, 0, 255),
                           'FL': (255, 255, 0), 'FR': (0, 165, 255)}

            for role, bp in active_paws.items():
                x_col = next((c for c in bp_xcord.columns if bp.lower() in c.lower()), None)
                y_col = next((c for c in bp_ycord.columns if bp.lower() in c.lower()), None)
                if x_col is None or y_col is None:
                    continue
                if fi >= len(bp_xcord):
                    continue

                bx = int(bp_xcord[x_col].iloc[fi]) + cx
                by = int(bp_ycord[y_col].iloc[fi]) + cy

                rh = _safe_int(roi_vars[role], 25)
                fh, fw = frame.shape[:2]
                x1 = max(0, bx - rh);  x2 = min(fw, bx + rh)
                y1 = max(0, by - rh);  y2 = min(fh, by + rh)

                color = _paw_colors.get(role, (255, 255, 255))

                if x2 <= x1 or y2 <= y1:
                    paw_labels[role].config(
                        text=f"{role} ({bp}): ROI out of bounds")
                    continue

                roi = gray[y1:y2, x1:x2]
                if roi.size == 0:
                    paw_labels[role].config(
                        text=f"{role} ({bp}): empty ROI")
                    continue

                # Gaussian blur + threshold
                blurred = cv2.GaussianBlur(roi, (k, k), 0)
                if thresh_mode_var.get() == 'otsu':
                    _, thresh_img = cv2.threshold(
                        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                else:
                    _, thresh_img = cv2.threshold(
                        blurred, m_thr, 255, cv2.THRESH_BINARY)

                # Find contours
                contours, _ = cv2.findContours(
                    thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Filter by min area
                contours = [c for c in contours if cv2.contourArea(c) >= min_ca]

                area = 0.0
                spread = 0.0
                mean_int = 0.0

                if contours:
                    best = max(contours, key=lambda c: cv2.contourArea(c))
                    area = cv2.contourArea(best)

                    xb, yb, wb, hb = cv2.boundingRect(best)
                    spread = float(max(wb, hb))

                    # Mean intensity within contour
                    mask_c = np.zeros(roi.shape, dtype=np.uint8)
                    cv2.drawContours(mask_c, [best], -1, 255, -1)
                    mean_int = cv2.mean(roi, mask=mask_c)[0]

                    # Draw contour on visualization frame, offset to frame coords
                    contour_offset = best.copy()
                    contour_offset[:, :, 0] += x1
                    contour_offset[:, :, 1] += y1
                    cv2.drawContours(vis, [contour_offset], -1, (0, 255, 0), 2)

                # Draw ROI rectangle
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
                cv2.circle(vis, (bx, by), 4, color, -1)

                # Label
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th_t), _ = cv2.getTextSize(role, font, 0.55, 1)
                cv2.rectangle(vis, (x1, y1 - th_t - 4), (x1 + tw + 4, y1), (0, 0, 0), -1)
                cv2.putText(vis, role, (x1 + 2, y1 - 2), font, 0.55, color, 1)

                # Area / spread / intensity text overlay
                info_txt = f"A={area:.0f} S={spread:.0f} I={mean_int:.1f}"
                cv2.putText(vis, info_txt, (x1 + 2, y2 + 14),
                            font, 0.4, color, 1)

                # Update readout label
                paw_labels[role].config(
                    text=f"{role} ({bp}): area={area:.0f}  spread={spread:.0f}  int={mean_int:.1f}")

            # HUD text
            mode_str = thresh_mode_var.get()
            if mode_str == 'manual':
                mode_str += f'={m_thr}'
            hud = f'thresh={mode_str}  blur={k}  min_area={min_ca}'
            if cx or cy:
                hud += f'  crop=({cx},{cy})'
            cv2.putText(vis, hud, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)

            # Display on canvas
            cw = canvas.winfo_width()  or 680
            ch = canvas.winfo_height() or 520
            vh, vw = vis.shape[:2]
            scale = min(cw / vw, ch / vh)
            nw, nh = int(vw * scale), int(vh * scale)
            vis_small = cv2.resize(vis, (nw, nh))
            rgb = cv2.cvtColor(vis_small, cv2.COLOR_BGR2RGB)
            from PIL import Image, ImageTk
            photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            canvas.delete('all')
            canvas.create_image(cw // 2, ch // 2, image=photo, anchor='center')
            canvas.image = photo   # keep reference

        # Wire all traces -> debounced render
        frame_var.trace_add('write', _render)
        thresh_mode_var.trace_add('write', _render)
        manual_thresh_var.trace_add('write', _render)
        blur_kernel_var.trace_add('write', _render)
        min_area_var.trace_add('write', _render)
        crop_x_var.trace_add('write', _render)
        crop_y_var.trace_add('write', _render)
        for var in roi_vars.values():
            var.trace_add('write', _render)
        win.bind('<Configure>', _render)
        win.after(100, _do_render)   # initial render after window maps

    # ── Detect cached brightness ─────────────────────────────────────────────


    # ═══════════════════════════════════════════════════════════════════════
    # Analysis: run / cancel / thread (compute delegated to gait_core)
    # ═══════════════════════════════════════════════════════════════════════

    def auto_run(self):
        """One-click chain entry (pose flow / batch handoff): rescan, include
        every session, re-apply the preset (licking predictions may have just
        arrived), and run with the current settings."""
        self._scan_sessions()
        self._select_all()
        try:
            self._refresh_lick_behaviors()
            self._apply_gait_preset()
        except Exception:
            pass
        ready, issues, _notes = self._check_readiness()
        if not ready:
            self._log_ui("Automatic paw contour run skipped: " + "; ".join(issues))
            return
        self._log_ui("Starting paw contour analysis (one-click chain)...")
        self._start_analysis(confirm_ungrouped=False)

    def _start_analysis(self, confirm_ungrouped=True):
        if self._fit_thread and self._fit_thread.is_alive():
            messagebox.showwarning("Busy", "Analysis is already running.",
                                   parent=self)
            return

        ready, issues, _notes = self._check_readiness()
        if not ready:
            messagebox.showwarning(
                "Not ready to run",
                "Please resolve:\n\n• " + "\n• ".join(issues), parent=self)
            return

        selected_names = set(self._session_filter.selected())
        if not selected_names:
            messagebox.showwarning("No sessions",
                                   "Select at least one session.", parent=self)
            return

        # A missing key is only a soft readiness note, but the run is
        # expensive - confirm before producing an ungrouped cohort.
        if self._key_df is None and confirm_ungrouped:
            if not messagebox.askyesno(
                    "No key file",
                    "No key file is loaded, so every session will be "
                    "analyzed as one ungrouped cohort (blank Treatment).\n\n"
                    "Load a key in Data → Key file to get group "
                    "comparisons.\n\nRun ungrouped anyway?",
                    parent=self):
                return

        sessions = [s for s in self._sessions
                    if s['session_name'] in selected_names]

        paw_map = {role: self._role_vars[role].get().strip()
                   for role in self.ROLES}
        if not paw_map.get('HL') or not paw_map.get('HR'):
            messagebox.showerror("Paw mapping",
                                 "Configure at least HL and HR body parts.",
                                 parent=self)
            return

        # Validate brightness requires video
        if self._use_brightness_var.get():
            no_video = [s['session_name'] for s in sessions
                        if not s.get('video')]
            if no_video:
                msg = ("Brightness extraction is enabled but these sessions "
                       "have no video:\n\n"
                       + "\n".join(no_video[:10])
                       + ("\n..." if len(no_video) > 10 else "")
                       + "\n\nBrightness will be skipped for those sessions.")
                messagebox.showwarning("Missing videos", msg, parent=self)

        params = {
            'contact_threshold': self._contact_thresh_var.get(),
            'height_window':     self._height_window_var.get(),
            'bin_seconds':       self._bin_seconds_var.get(),
            'bin_unit':          self._bin_unit_var.get(),
            'fallback_fps':      float(self._fallback_fps_var.get()),
            'use_brightness':    self._use_brightness_var.get(),
            'brt_threshold':     self._brt_thresh_var.get(),
            'brt_weight':        self._brt_weight_var.get(),
            'roi_sizes':         {role: self._roi_size_vars[role].get()
                                  for role in self.ROLES},
            'crop_offset_x':      self._crop_x_var.get(),
            'crop_offset_y':      self._crop_y_var.get(),
            'extraction_stride':  self._extraction_stride_var.get(),
            # Contour area is the only method the tab can produce (the
            # engine still runs height/speed/combined for scripts/tests).
            'contact_method':     'contour_area',
            # Paw-contour + ROI-brightness measures only: no gait / movement
            # metrics (their speed / stance / likelihood / locomotion
            # settings are engine defaults the gait blocks alone read).
            'compute_gait':       False,
            # The contour gate needs contour data, so extraction is always on
            # (an old bundle may have persisted the toggle off).
            'paw_contour':        True,
            'contour_roi_sizes':  {role: self._contour_roi_size_vars[role].get()
                                   for role in self.ROLES},
            'contour_forelimbs':  self._contour_forelimbs_var.get(),
            'contour_area_threshold': self._contour_area_thresh_var.get(),
            'contour_area_max': self._contour_area_max_var.get(),
            'exclude_licking':    bool(self._exclude_lick_var.get()),
            'lick_behavior':      self._lick_behavior_var.get(),
            'lick_threshold':     float(self._lick_thresh_var.get()),
            'gate_4paw':          bool(self._gate_4paw_var.get()),
        }

        self._cancel_flag.clear()
        self._session_intermediates = {}
        self._run_btn.config(state='disabled')
        self._cancel_btn.config(state='normal')
        self._progress.config(maximum=max(len(sessions), 1), value=0)
        self._sub_progress_label.config(text='')
        self._progress_frame.pack(fill='x', padx=4, pady=(0, 4))
        self._export_sum_btn.config(state='disabled')
        self._export_bin_btn.config(state='disabled')

        # Snapshot Tk-backed values on the main thread; the worker must not
        # touch Tk variables.
        ctx = self._core_ctx(self._project_folder())
        prefix = self._prefix_var.get().strip()

        self._fit_thread = threading.Thread(
            target=self._analysis_thread,
            args=(sessions, paw_map, params, ctx, prefix),
            daemon=True)
        self._fit_thread.start()

    def _cancel_analysis(self):
        self._cancel_flag.set()
        self._log("Cancelling…")

    def _sub_progress(self, done, total, text):
        """gait_core extraction progress → sub-progress label (thread-safe)."""
        try:
            self.app.root.after(
                0, lambda: self._sub_progress_label.config(text=text or ''))
        except tk.TclError:
            pass

    def _analysis_thread(self, sessions, paw_map, params, ctx, prefix=''):
        summary_rows = []
        bin_rows = []
        not_done = []   # sessions that raised or produced no result
        skip_reasons = {}   # session_name -> why (from gait_core)
        for sess in sessions:
            if self._cancel_flag.is_set():
                self._log("Cancelled.")
                break
            name = sess['session_name']
            self._log(f"Processing: {name}")
            try:
                result = gc.analyze_session(
                    sess, paw_map, params, ctx,
                    log=self._log, progress_cb=self._sub_progress,
                    cancel=self._cancel_flag, skip_reasons=skip_reasons)
                if result:
                    subj      = self._resolve_subject(name, prefix)
                    treatment = self._get_treatment(subj)
                    base = dict(session=name, subject=subj,
                                treatment=treatment)
                    srow = {**base, **result['summary']}
                    summary_rows.append(srow)
                    for brow in result['bins']:
                        bin_rows.append({**base, **brow})
                    self._session_intermediates[name] = \
                        result['intermediates']
                else:
                    not_done.append(name)
                    self._log(f"  Skipped: {name} "
                              f"({skip_reasons.get(name, 'no result')})")
            except Exception as e:
                self._log(f"  ERROR: {e}")
                skip_reasons[name] = f"error: {e}"
                not_done.append(name)
            try:
                self.app.root.after(0, self._progress.step, 1)
            except tk.TclError:
                pass

        self._log(f"Analysis loop finished: {len(summary_rows)}/{len(sessions)} "
                  f"sessions produced results.")

        try:
            self.app.root.after(0, self._on_analysis_complete,
                                summary_rows, bin_rows, not_done, skip_reasons)
        except tk.TclError:
            pass

    def _on_analysis_complete(self, summary_rows, bin_rows, not_done=None,
                              skip_reasons=None):
        self._cancel_btn.config(state='disabled')
        self._progress_frame.pack_forget()
        self._sub_progress_label.config(text='')

        not_done = not_done or []
        skip_reasons = skip_reasons or {}
        n_ok = len(summary_rows)
        n_bad = len(not_done)
        skipped_lines = [f"{n} - {skip_reasons.get(n, 'no result')}"
                         for n in not_done]
        if skipped_lines:
            self._log_ui(f"Skipped {n_bad} session(s):")
            for line in skipped_lines:
                self._log_ui(f"  {line}")

        # Visible run outcome
        if hasattr(self, '_outcome_lbl'):
            if n_bad:
                self._outcome_lbl.config(
                    text=f"{n_ok} analyzed · {n_bad} skipped/failed",
                    foreground='#b00020')
            else:
                self._outcome_lbl.config(
                    text=f"{n_ok} analyzed · 0 skipped/failed",
                    foreground='#127a12')

        if not summary_rows:
            self._log_ui("No results - all sessions were skipped or failed. "
                         "Check log for ERROR messages.")
            names = "\n".join(skipped_lines[:8]) + ("\n…" if n_bad > 8 else "")
            messagebox.showwarning(
                "No results",
                "All selected sessions were skipped or failed"
                + (f":\n\n{names}" if names else ".")
                + "\n\nSee the Log panel for the specific errors.",
                parent=self)
            self._update_readiness()
            return

        self._summary_df = pd.DataFrame(summary_rows)
        self._bins_df    = pd.DataFrame(bin_rows) if bin_rows else pd.DataFrame()

        self._refresh_results_table()
        self._refresh_summary_panel()
        self._export_sum_btn.config(state='normal')
        if not self._bins_df.empty:
            self._export_bin_btn.config(state='normal')
        if _PLOT_OK:
            self._populate_registry()
        if self._session_intermediates:
            self._adjust_contact_btn.config(state='normal')

        # Persist results so the tab can auto-reload them next time (Transitions-style).
        self._save_last_session()

        # Warn about fallback FPS usage
        fallback_sessions = [r.get('session', '?') for r in summary_rows
                             if r.get('fallback_fps_used', False)]
        if fallback_sessions:
            self._log_ui(f"  ⚠ {len(fallback_sessions)} session(s) used fallback FPS "
                         f"(video FPS could not be detected).")

        self._log_ui(f"Done. {len(summary_rows)} session(s) processed"
                     + (f", {n_bad} skipped (see above)." if n_bad else "."))
        self._update_readiness()

    # ═══════════════════════════════════════════════════════════════════════
    # Results persistence + auto-reload (Transitions-style)
    # ═══════════════════════════════════════════════════════════════════════

    SESSION_SCHEMA_VERSION = 1

    # Scalar analysis vars persisted with a session (name → attribute).
    _PERSIST_SCALAR_VARS = (
        '_contact_thresh_var', '_height_window_var', '_bin_seconds_var',
        '_bin_unit_var', '_fallback_fps_var', '_use_brightness_var',
        '_brt_thresh_var', '_brt_weight_var', '_extraction_stride_var',
        # _contact_method_var / _speed_thresh_var were dropped 2026-09-17
        # (contour area is the only method), and the median-filter, min-bout,
        # min-stance, likelihood and locomotion vars 2026-09-18 (gait only);
        # their keys in old bundles are ignored by _apply_settings.
        '_crop_x_var', '_crop_y_var',
        '_paw_contour_var', '_contour_forelimbs_var',
        '_contour_area_thresh_var', '_contour_area_max_var', '_injured_paw_var',
        '_exclude_lick_var', '_lick_behavior_var',
        '_enable_stats_var', '_stats_paradigm_var', '_stats_alpha_var',
        '_timecourse_posthoc_var',
        '_lick_thresh_var', '_gate_4paw_var', '_use_fore_var', '_preset_var',
        '_key_file_var', '_prefix_var',
    )
    _PERSIST_DICT_VARS = ('_roi_size_vars', '_contour_roi_size_vars', '_role_vars')

    def _collect_settings(self):
        s = {}
        for name in self._PERSIST_SCALAR_VARS:
            v = getattr(self, name, None)
            if v is not None:
                try:
                    s[name] = v.get()
                except Exception:
                    pass
        for name in self._PERSIST_DICT_VARS:
            d = getattr(self, name, None)
            if isinstance(d, dict):
                s[name] = {k: (var.get() if hasattr(var, 'get') else var)
                           for k, var in d.items()}
        return s

    def _apply_settings(self, s):
        for name in self._PERSIST_SCALAR_VARS:
            if name in s and getattr(self, name, None) is not None:
                val = s[name]
                if name == '_preset_var':
                    # an old bundle may name one of the dropped gait presets
                    val = gc.LEGACY_PRESET_NAMES.get(val, val)
                try:
                    getattr(self, name).set(val)
                except Exception:
                    pass
        for name in self._PERSIST_DICT_VARS:
            d = getattr(self, name, None)
            if isinstance(d, dict) and isinstance(s.get(name), dict):
                for k, val in s[name].items():
                    if k in d and hasattr(d[k], 'set'):
                        try:
                            d[k].set(val)
                        except Exception:
                            pass

    def _session_bundle(self):
        return {
            'schema_version': self.SESSION_SCHEMA_VERSION,
            'saved_at': datetime.now().isoformat(timespec='seconds'),
            'settings': self._collect_settings(),
            'summary_records': (self._summary_df.to_dict('records')
                                if self._summary_df is not None else []),
            'bins_records': (self._bins_df.to_dict('records')
                             if self._bins_df is not None
                             and not self._bins_df.empty else []),
        }

    _SESSION_AUTO_KEEP = 15   # prune older auto_* saves beyond this (named_* kept)

    def _last_session_path(self, proj=None):
        """Legacy single-file path (still listed in the dropdown if present)."""
        proj = proj or self._project_folder()
        if not proj:
            return None
        return os.path.join(gc.analysis_dir(proj), '_last_session.pkl')

    def _sessions_dir(self, proj=None):
        proj = proj or self._project_folder()
        if not proj:
            return None
        d = os.path.join(gc.analysis_dir(proj), 'sessions')
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            return None
        return d

    def _save_last_session(self):
        """Auto-save after a run - unless we're mid-load (loads never spawn one)."""
        if getattr(self, '_loading_session', False):
            return
        self._save_session_file('auto')

    def _save_session_file(self, kind, name=None):
        if atomic_pickle_save is None:
            return
        d = self._sessions_dir()
        if not d:
            return
        n = 0 if self._summary_df is None else len(self._summary_df)
        ts = datetime.now().strftime('%Y%m%d-%H%M%S-%f')   # μs keeps names unique
        if kind == 'named' and name:
            slug = re.sub(r'[^A-Za-z0-9._-]+', '-', name).strip('-') or 'session'
            fname = f'named_{slug}_{ts}_{n}sess.pkl'
        else:
            fname = f'auto_{ts}_{n}sess.pkl'
        try:
            atomic_pickle_save(self._session_bundle(), os.path.join(d, fname))
        except Exception as e:
            self._log_ui(f"(could not save session: {e})")
            return
        self._prune_auto_sessions(d)
        self._refresh_saved_sessions(select_path=os.path.join(d, fname))

    def _prune_auto_sessions(self, d):
        try:
            autos = sorted(
                [f for f in os.listdir(d) if f.startswith('auto_') and f.endswith('.pkl')])
        except Exception:
            return
        for f in autos[:-self._SESSION_AUTO_KEEP]:
            try:
                os.remove(os.path.join(d, f))
            except OSError:
                pass

    @staticmethod
    def _parse_session_fname(fn):
        """(sort_key, label) for a saved-session filename, or None if unrecognized."""
        _TS = r'(\d{8}-\d{6}(?:-\d{1,6})?)'
        m = re.match(r'auto_' + _TS + r'_(\d+)sess\.pkl$', fn)
        if m:
            ts, n = m.group(1), int(m.group(2))
            return ts, f"{GaitLimbTabV2._fmt_ts(ts)} · {n} session{'s' if n != 1 else ''}"
        m = re.match(r'named_(.+)_' + _TS + r'_(\d+)sess\.pkl$', fn)
        if m:
            slug, ts, n = m.group(1), m.group(2), int(m.group(3))
            return ts, f"★ {slug} · {n} session{'s' if n != 1 else ''}"
        return None

    @staticmethod
    def _fmt_ts(ts):
        try:
            return datetime.strptime(ts[:15], '%Y%m%d-%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            return ts

    def _list_saved_sessions(self):
        """[(label, path)] newest-first for the current project (+ legacy file)."""
        items = []
        d = self._sessions_dir()
        if d and os.path.isdir(d):
            for fn in os.listdir(d):
                parsed = self._parse_session_fname(fn)
                if parsed:
                    items.append((parsed[0], parsed[1], os.path.join(d, fn)))
        legacy = self._last_session_path()
        if legacy and os.path.isfile(legacy):
            items.append(('00000000-000000', '(previous)', legacy))
        items.sort(key=lambda t: t[0], reverse=True)
        return [(label, path) for _sk, label, path in items]

    def _load_last_session(self, path=None):
        if path is None:
            path = self._last_session_path()
        if not path or not os.path.isfile(path):
            return False
        self._loading_session = True   # loads never create a new saved entry
        try:
            return self._load_bundle_file(path)
        finally:
            self._loading_session = False

    @staticmethod
    def _contour_only(df, settings=None):
        """A bundle's summary / bins frame without the gait and movement
        columns that runs before 2026-09-18 saved. A contour-gated old run
        carries its gate pass % as contact_pct_HL (== contact_pct_HR), so
        that becomes contour_pass_pct; a pre-2026-09-17 height-gated run
        (its settings name another contact method) gets no pass %."""
        if df is None or df.empty:
            return df
        method = (settings or {}).get('_contact_method_var', 'contour_area')
        if ('contour_pass_pct' not in df.columns
                and 'contact_pct_HL' in df.columns
                and method == 'contour_area'):
            df = df.copy()
            df.insert(df.columns.get_loc('contact_pct_HL'),
                      'contour_pass_pct', df['contact_pct_HL'])
        return gc.drop_gait_columns(df)

    def _load_bundle_file(self, path):
        try:
            bundle = _robust_unpickle(path)
        except Exception as e:
            self._log_ui(f"(could not load session cache: {e})")
            return False
        if bundle.get('schema_version') != self.SESSION_SCHEMA_VERSION:
            self._log_ui("(saved session cache is a different version; ignoring)")
            return False
        try:
            self._apply_settings(bundle.get('settings', {}))
            self._on_lick_toggle()
            self._on_use_fore_changed()
        except Exception:
            pass
        # Re-resolve the restored key path so the status/Subject column and
        # readiness reflect the loaded bundle (not just the combo text).
        try:
            kf = self._key_file_var.get().strip()
            if kf:
                kp = kf if os.path.isabs(kf) else os.path.join(
                    self._project_folder(), kf)
                if os.path.isfile(kp):
                    self._load_key_file(kp)
        except Exception:
            pass
        summ = bundle.get('summary_records') or []
        bins = bundle.get('bins_records') or []
        settings = bundle.get('settings') or {}
        self._summary_df = (self._contour_only(pd.DataFrame(summ), settings)
                            if summ else None)
        self._bins_df = (self._contour_only(pd.DataFrame(bins), settings)
                         if bins else pd.DataFrame())
        self._refresh_results_table()
        try:
            self._refresh_summary_panel()
        except Exception:
            pass
        # Enable the result actions that only need the DataFrames.
        if self._summary_df is not None:
            self._export_sum_btn.config(state='normal')
            if self._bins_df is not None and not self._bins_df.empty:
                self._export_bin_btn.config(state='normal')
            if _PLOT_OK:
                self._populate_registry()
        if hasattr(self, '_outcome_lbl'):
            n = 0 if self._summary_df is None else len(self._summary_df)
            self._outcome_lbl.config(
                text=f"Reloaded {n} session(s) from cache "
                     f"(saved {bundle.get('saved_at', '?')}). Re-run for "
                     f"contour-shape graphs / Adjust Contact.",
                foreground='#127a12')
        self._log_ui(f"Reloaded {0 if self._summary_df is None else len(self._summary_df)} "
                     f"session(s) from saved cache.")
        return True

    def _on_tab_shown(self, event=None):
        """Fires on any main-tab change; refresh the saved-session list when this
        tab becomes active (no more reload prompt)."""
        try:
            nb = self.app.notebook
            label = nb.tab(nb.select(), 'text')   # select() is a widget path
        except Exception:
            return
        if isinstance(label, str) and ('Paw Contour' in label or 'Gait & Limb' in label
                                       or label.endswith('Limb Use')):
            self._refresh_saved_sessions()

    def _refresh_saved_sessions(self, select_path=None):
        """Repopulate the Saved-sessions combobox from disk (index-keyed, so
        duplicate labels never collide)."""
        combo = getattr(self, '_saved_combo', None)
        if combo is None:
            return
        self._saved_items = self._list_saved_sessions()   # [(label, path)]
        labels = [label for label, _ in self._saved_items]
        combo.config(values=labels)
        idx = 0
        if select_path is not None:
            idx = next((i for i, (_l, p) in enumerate(self._saved_items)
                        if os.path.abspath(p) == os.path.abspath(select_path)), 0)
        if labels:
            combo.current(idx)
        else:
            combo.set('')
        state = 'normal' if labels else 'disabled'
        for b in ('_saved_load_btn', '_saved_del_btn'):
            w = getattr(self, b, None)
            if w is not None:
                w.config(state=state)

    def _selected_session_path(self):
        combo = getattr(self, '_saved_combo', None)
        items = getattr(self, '_saved_items', [])
        if combo is None or not items:
            return None
        i = combo.current()
        return items[i][1] if 0 <= i < len(items) else None

    def _load_selected_session(self):
        path = self._selected_session_path()
        if not path:
            messagebox.showinfo("Load session", "No saved session selected.", parent=self)
            return
        self._load_last_session(path)

    def _save_current_session_named(self):
        if self._summary_df is None:
            messagebox.showinfo("Save session", "Run an analysis first.", parent=self)
            return
        name = simpledialog.askstring(
            "Save session", "Name for this saved session:", parent=self)
        if name:
            self._save_session_file('named', name)

    def _delete_selected_session(self):
        path = self._selected_session_path()
        if not path:
            return
        label = self._saved_combo.get()
        if messagebox.askyesno("Delete session",
                               f"Delete this saved session?\n\n{label}", parent=self):
            try:
                os.remove(path)
            except OSError as e:
                self._log_ui(f"(could not delete: {e})")
            self._refresh_saved_sessions()

    # ═══════════════════════════════════════════════════════════════════════
    # Post-analysis contact re-adjustment
    # ═══════════════════════════════════════════════════════════════════════


    # ═══════════════════════════════════════════════════════════════════════
    # Adjust Contact - recompute via the SAME metrics code as the analysis
    # ═══════════════════════════════════════════════════════════════════════

    def _open_contact_adjustment(self):
        """Open dialog to adjust contact detection params and recompute metrics."""
        if not self._session_intermediates:
            messagebox.showinfo("No data", "Run analysis first.", parent=self)
            return

        win = tk.Toplevel(self)
        win.title("Adjust contour detection")
        win.resizable(False, False)
        win.grab_set()

        ttk.Label(win, text="Re-compute the contour gate with new parameters",
                  font=(FONT_FAMILY, 10, 'bold')).pack(padx=12, pady=(10, 6))

        frm = ttk.Frame(win, padding=10)
        frm.pack(fill='x')

        # Contour-area band (the hind-paw contour gate)
        ttk.Label(frm, text="Contour-area band (px²):").grid(
            row=0, column=0, sticky='w', pady=3)
        _ca_row = ttk.Frame(frm)
        _ca_row.grid(row=0, column=1, sticky='w', padx=4)
        ca_min_var = tk.IntVar(value=self._contour_area_thresh_var.get())
        ttk.Spinbox(_ca_row, from_=0, to=100000, textvariable=ca_min_var,
                    width=7).pack(side='left')
        ttk.Label(_ca_row, text="-").pack(side='left', padx=2)
        ca_max_var = tk.IntVar(value=self._contour_area_max_var.get())
        ttk.Spinbox(_ca_row, from_=0, to=100000, textvariable=ca_max_var,
                    width=7).pack(side='left')

        # Fore-paw height threshold - only when the stored run mapped fore
        # paws (it feeds the 4-paw gate and the fore-paw ROI brightness).
        ct_var = tk.IntVar(value=self._contact_thresh_var.get())
        has_fore = any({'FL', 'FR'} & set(inter.get('active_paws', {}))
                       for inter in self._session_intermediates.values())
        if has_fore:
            ttk.Label(frm, text="Fore-paw down height (px):").grid(
                row=1, column=0, sticky='w', pady=3)
            ttk.Spinbox(frm, from_=0, to=500, textvariable=ct_var,
                        width=8).grid(row=1, column=1, sticky='w', padx=4)

        # Buttons
        dlg_btn_row = ttk.Frame(win)
        dlg_btn_row.pack(pady=(6, 12))

        def _apply():
            win.destroy()
            self._recompute_contact({
                'contact_method': 'contour_area',   # the only method the tab runs
                'contact_threshold': ct_var.get(),
                'contour_area_threshold': ca_min_var.get(),
                'contour_area_max': ca_max_var.get(),
            })

        ttk.Button(dlg_btn_row, text="Apply", command=_apply).pack(
            side='left', padx=6)
        ttk.Button(dlg_btn_row, text="Cancel", command=win.destroy).pack(
            side='left', padx=6)


    def _recompute_contact(self, new_params):
        """Recompute contact masks and metrics using stored intermediates.
        Delegates to gait_core.recompute_with_contact - the same
        compute_all_metrics as the analysis itself (licking exclusion and the
        4-paw gate stay in force, unlike the old drifted copy)."""
        if not self._session_intermediates:
            return
        self._log_ui("Re-computing contact with adjusted parameters...")
        self._run_btn.config(state='disabled')
        # Snapshot Tk-backed values here (main thread); the worker must not
        # touch Tk variables.
        ctx = self._core_ctx(self._project_folder())
        prefix = self._prefix_var.get().strip()

        def _worker():
            rows, brows, errs = [], [], []
            for name, inter in self._session_intermediates.items():
                try:
                    res = gc.recompute_with_contact(inter, new_params, ctx,
                                                    log=self._log)
                    subj = self._resolve_subject(name, prefix)
                    treatment = self._get_treatment(subj)
                    base = dict(session=name, subject=subj,
                                treatment=treatment)
                    rows.append({**base, **res['summary']})
                    for b in res['bins']:
                        brows.append({**base, **b})
                except Exception as e:
                    errs.append(f"{name}: {e}")
            try:
                self.app.root.after(0, self._recompute_done, rows, brows, errs)
            except tk.TclError:
                pass

        threading.Thread(target=_worker, daemon=True).start()

    def _recompute_done(self, rows, brows, errs):
        for e in errs:
            self._log_ui(f"  ERROR: {e}")
        self._update_readiness()
        if not rows:
            self._log_ui("Recompute produced no results.")
            return
        self._summary_df = pd.DataFrame(rows)
        self._bins_df = pd.DataFrame(brows) if brows else pd.DataFrame()
        self._refresh_results_table()
        self._refresh_summary_panel()
        self._save_last_session()
        self._populate_registry(keep=True)
        self._log_ui(f"Recompute done: {len(rows)} session(s).")

    def _refresh_summary_panel(self):
        """Populate the summary frame with key metrics (mean +/- SEM)."""
        for w in self._summary_frame.winfo_children():
            w.destroy()

        if self._summary_df is None or self._summary_df.empty:
            ttk.Label(self._summary_frame, text="No results yet",
                      foreground='grey', font=('TkDefaultFont', 9)).pack(anchor='w')
            return

        df = self._summary_df
        # (label, column, value format, tooltip). Ratios are the stored HL/HR.
        metrics = [
            ('Contour gate pass %', 'contour_pass_pct',            '{:.1f}',
             self.CONTOUR_PASS_TIP),
            ('Area ratio HL/HR',    'paw_area_ratio_hind',          '{:.3f}',
             'Paw area ratio, HL ÷ HR (1.0 = equal).'),
            ('Intensity ratio HL/HR', 'contact_intensity_ratio_hind', '{:.3f}',
             'Contact intensity ratio, HL ÷ HR (1.0 = equal).'),
            ('Circularity ratio HL/HR', 'paw_circularity_ratio_hind', '{:.3f}',
             'Contour circularity ratio, HL ÷ HR (1.0 = equal).'),
            ('Solidity ratio HL/HR', 'paw_solidity_ratio_hind', '{:.3f}',
             'Contour solidity ratio, HL ÷ HR (1.0 = equal).'),
            ('Brightness ratio HL/HR', 'brightness_ratio_HL_HR',   '{:.3f}',
             'ROI brightness ratio, HL ÷ HR (1.0 = equal).'),
        ]

        has_treatment = ('treatment' in df.columns
                         and df['treatment'].notna().any()
                         and df['treatment'].ne('').any())
        if has_treatment:
            groups = [(t, df[df['treatment'] == t])
                      for t in df['treatment'].dropna().unique()
                      if str(t).strip()]
        else:
            groups = [('All', df)]

        row_idx = 0
        # Header row
        ttk.Label(self._summary_frame, text="Metric",
                  font=('TkDefaultFont', 8, 'bold')).grid(
            row=row_idx, column=0, sticky='w', padx=(0, 8))
        for ci, (gname, _) in enumerate(groups):
            ttk.Label(self._summary_frame, text=str(gname),
                      font=('TkDefaultFont', 8, 'bold')).grid(
                row=row_idx, column=ci + 1, sticky='e', padx=(4, 8))
        row_idx += 1

        ttk.Separator(self._summary_frame, orient='horizontal').grid(
            row=row_idx, column=0, columnspan=len(groups) + 1,
            sticky='ew', pady=2)
        row_idx += 1

        for label, col, fmt, tip in metrics:
            # Skip metrics with no values at all (e.g. no brightness run).
            if col not in df.columns or df[col].dropna().empty:
                continue
            _lbl = ttk.Label(self._summary_frame, text=label,
                             font=('TkDefaultFont', 8))
            _lbl.grid(row=row_idx, column=0, sticky='w', padx=(0, 8))
            if tip:
                Tip(_lbl, tip)
            for ci, (_, gdf) in enumerate(groups):
                vals = gdf[col].dropna()
                if len(vals) == 0:
                    txt = '--'
                elif len(vals) == 1:
                    txt = fmt.format(vals.iloc[0])
                else:
                    m = vals.mean()
                    sem = (vals.std(ddof=1) / np.sqrt(len(vals)))
                    txt = f'{fmt.format(m)} \u00b1 {fmt.format(sem)}'
                ttk.Label(self._summary_frame, text=txt,
                          font=('TkDefaultFont', 8)).grid(
                    row=row_idx, column=ci + 1, sticky='e', padx=(4, 8))
            row_idx += 1

    # ═══════════════════════════════════════════════════════════════════════
    # Tooltip helper
    # ═══════════════════════════════════════════════════════════════════════

    def _refresh_results_table(self):
        for item in self._res_tree.get_children():
            self._res_tree.delete(item)
        if self._summary_df is None:
            return

        def _fmt(v, fmt):
            if fmt is None:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return ''
                return str(v)
            try:
                v = float(v)
            except (TypeError, ValueError):
                return ''
            return '' if np.isnan(v) else fmt.format(v)

        for _, row in self._summary_df.iterrows():
            self._res_tree.insert('', 'end', values=tuple(
                _fmt(row.get(src, float('nan') if fmt else ''), fmt)
                for _c, _h, _w, src, fmt, _t in self._RESULT_COLUMNS))

    # ═══════════════════════════════════════════════════════════════════════
    # Export
    # ═══════════════════════════════════════════════════════════════════════

    def _export_summary(self):
        self._export_df(self._summary_df, 'wb_summary')

    def _export_bins(self):
        self._export_df(self._bins_df, 'wb_bins')

    def _export_df(self, df: pd.DataFrame, prefix: str):
        if df is None or df.empty:
            messagebox.showinfo("Nothing to export", "Run analysis first.", parent=self)
            return
        # Paw-contour + ROI-brightness columns only (the tab never computes
        # the gait metrics; this also covers frames restored another way).
        df = gc.drop_gait_columns(df)
        folder = self.app.current_project_folder.get()
        analysis_dir = os.path.join(folder, 'analysis') if folder else ''
        if analysis_dir:
            os.makedirs(analysis_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = filedialog.asksaveasfilename(
            title="Save CSV",
            initialdir=analysis_dir or None,
            initialfile=f'{prefix}_{ts}.csv',
            defaultextension='.csv',
            filetypes=[('CSV files', '*.csv')],
            parent=self)
        if path:
            df.to_csv(path, index=False)
            self._log_ui(f"Saved: {os.path.basename(path)}")

    # ═══════════════════════════════════════════════════════════════════════
    # Graphs
    # ═══════════════════════════════════════════════════════════════════════

