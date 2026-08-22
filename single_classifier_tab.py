"""Single-Classifier Analysis — rebuilt tab (streamlined, shared-style).

Ground-up rebuild of the original analysis_tab on the app's standard
architecture: left rail (Data / Options / Statistics / collapsed Advanced /
Run / Export), right pane with a Graph dropdown topbar, the shared 🎨⚙ style
dialog, and a Σ Stats flip. All compute lives in ``analysis_core`` (pure,
threaded-safe); this module is presentation only.

External contract (PixelPaws_GUI relies on exactly this surface):
    SingleClassifierTab(parent, main_gui)   # a ttk.Frame
    .analysis_project_var                   # tk.StringVar
    .key_df / .results_df
    .scan_project_folder(folder=None)
    ._dash_subjects_overview()
    .run_analysis(silent=False)             # silent=True runs SYNCHRONOUSLY
                                            # (the batch handoff reads
                                            # results_df immediately after)
"""
from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)

from ui_tooltip import Tip, collapsible
from ui_session_filter import SessionFilter

# Metric dropdown: (column, aggregation across bins, label)
_METRICS = [
    ("Total_Time_s", "sum", "Time in behavior (s)"),
    ("N_Bouts", "sum", "Number of bouts"),
    ("Mean_Bout_Duration_s", "mean", "Mean bout duration (s)"),
    ("Bout_Frequency_per_min", "mean", "Bout frequency (/min)"),
    ("Percent_Time", "mean", "% time in behavior"),
]
_METRIC_LABELS = [m[2] for m in _METRICS]

_PRESETS = {
    "Default": dict(bin_size=5.0, bin_unit="minutes", whole=False,
                    phases=False, stats=False),
    "Formalin (2-phase)": dict(bin_size=5.0, bin_unit="minutes", whole=False,
                               phases=True, stats=True,
                               acute=(0, 10), phase2=(10, 60)),
}


class SingleClassifierTab(ttk.Frame):

    def __init__(self, parent, main_gui):
        super().__init__(parent)
        self.app = main_gui
        self.analysis_project_var = tk.StringVar(value="")
        self.key_df = None
        self.key_file_path = None
        self.results_df = None
        self.perframe_data = {}
        self._files = []                 # list[analysis_core.FileInfo]
        self._behaviors = {}             # {behavior: [FileInfo]}
        self._subjects_rows = []         # [(subject, group)]
        self._axis_overrides = {}        # {view_key: (ymin, ymax)}
        self._run_queue = queue.Queue()
        self._cancel = threading.Event()
        self._running = False
        self._build_ui()

    # ------------------------------------------------------------------ UI

    def _project(self):
        import plot_style
        p = self.analysis_project_var.get()
        return p or plot_style.project_of(self.app)

    def _build_ui(self):
        def _T(w, tip):
            Tip(w, tip)
            return w

        left_outer = ttk.Frame(self, width=300)
        left_outer.pack(side="left", fill="y", padx=(8, 4), pady=8)
        left_outer.pack_propagate(False)
        canvas = tk.Canvas(left_outer, highlightthickness=0, width=290)
        vsb = ttk.Scrollbar(left_outer, orient="vertical",
                            command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        left = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=left, anchor="nw")
        left.bind("<Configure>",
                  lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True, padx=(4, 8), pady=8)

        # ---------------- Data ----------------
        df_ = ttk.LabelFrame(left, text="Data")
        df_.pack(fill="x", pady=(0, 6))

        # Key file first, then Sessions — same order on every analysis tab.
        krow = ttk.Frame(df_)
        krow.pack(fill="x", padx=6, pady=(6, 1))
        ttk.Label(krow, text="Key file:").pack(side="left")
        self.key_file_var = tk.StringVar(value="")
        self._key_combo = ttk.Combobox(krow, textvariable=self.key_file_var,
                                       state="readonly", width=14)
        self._key_combo.pack(side="left", fill="x", expand=True, padx=(4, 4))
        self._key_combo.bind("<<ComboboxSelected>>", self._on_key_pick)
        Tip(self._key_combo, "Key files discovered in the project; groups "
                             "come from the Treatment column.")
        _T(ttk.Button(krow, text="Browse", width=7,
                      command=self._browse_key),
           "Pick a key file (CSV/XLSX with Subject and Treatment columns)."
           ).pack(side="left")
        _T(ttk.Button(krow, text="Generate…", width=9,
                      command=self._generate_key),
           "Create a key file: type a Treatment for each video subject; "
           "saved as key_file.csv in the project.").pack(side="left",
                                                         padx=(4, 0))
        self._key_status = tk.StringVar(value="no key loaded")
        ttk.Label(df_, textvariable=self._key_status, foreground="#666666",
                  wraplength=270, justify="left").pack(anchor="w", padx=6)

        # Sessions picker + Rescan on one row.
        sess_row = ttk.Frame(df_)
        sess_row.pack(fill="x", padx=6, pady=(4, 1))
        self._session_filter = SessionFilter(sess_row)
        self._session_filter.pack(side="left", fill="x", expand=True)
        _T(ttk.Button(sess_row, text="Rescan", width=7,
                      command=lambda: self.scan_project_folder()),
           "Re-scan the project for prediction folders and key files."
           ).pack(side="left", padx=(4, 0))
        self._scan_status = tk.StringVar(value="No project scanned.")
        ttk.Label(df_, textvariable=self._scan_status, foreground="#666666",
                  wraplength=270, justify="left").pack(anchor="w", padx=6)

        srow = ttk.Frame(df_)
        srow.pack(fill="x", padx=6, pady=(2, 1))
        self._subjects_status = tk.StringVar(value="")
        ttk.Label(srow, textvariable=self._subjects_status,
                  foreground="#666666", wraplength=200,
                  justify="left").pack(side="left")
        _T(ttk.Button(srow, text="table", width=6,
                      command=self._show_subjects_table),
           "Show the full subject → group assignment table."
           ).pack(side="right")

        ttk.Label(df_, text="Behaviors:").pack(anchor="w", padx=6,
                                               pady=(4, 0))
        self._beh_list = tk.Listbox(df_, selectmode="extended", height=6,
                                    exportselection=False)
        self._beh_list.pack(fill="x", padx=6, pady=(0, 2))
        Tip(self._beh_list,
            "Behaviors found in the predictions folder. Highlighted = "
            "analyzed (Ctrl/Shift-click); all selected by default.")

        mrow = ttk.Frame(df_)
        mrow.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Label(mrow, text="Analyze:").pack(side="left")
        self.analyze_mode = tk.StringVar(value="separate")
        for val, lab in (("separate", "separate"), ("combined", "combined"),
                         ("both", "both")):
            _T(ttk.Radiobutton(mrow, text=lab, value=val,
                               variable=self.analyze_mode),
               "separate: each behavior on its own. combined: one merged "
               "any-behavior signal. both: run both ways."
               ).pack(side="left", padx=(6, 0))

        # ---------------- Options ----------------
        opt = ttk.LabelFrame(left, text="Options")
        opt.pack(fill="x", pady=6)
        prow = ttk.Frame(opt)
        prow.pack(fill="x", padx=6, pady=(6, 1))
        ttk.Label(prow, text="Preset:").pack(side="left")
        self._preset_var = tk.StringVar(value="Default")
        pcb = ttk.Combobox(prow, textvariable=self._preset_var,
                           state="readonly", width=18,
                           values=list(_PRESETS))
        pcb.pack(side="right")
        pcb.bind("<<ComboboxSelected>>", self._apply_preset)
        Tip(pcb, "Formalin (2-phase) enables acute/Phase II windows and "
                 "statistics; windows are editable under Advanced.")

        brow = ttk.Frame(opt)
        brow.pack(fill="x", padx=6, pady=1)
        ttk.Label(brow, text="Bin size:").pack(side="left")
        self.bin_unit_var = tk.StringVar(value="minutes")
        ucb = ttk.Combobox(brow, textvariable=self.bin_unit_var,
                           state="readonly", width=8,
                           values=["minutes", "seconds"])
        ucb.pack(side="right")
        self.bin_size_var = tk.DoubleVar(value=5.0)
        bsp = ttk.Spinbox(brow, textvariable=self.bin_size_var, from_=1,
                          to=3600, width=6)
        bsp.pack(side="right", padx=(0, 4))
        Tip(bsp, "Width of the analysis time bins.")
        self.no_time_bins_var = tk.BooleanVar(value=False)
        _T(ttk.Checkbutton(opt, text="Entire session (no bins)",
                           variable=self.no_time_bins_var),
           "One value per session instead of a timecourse."
           ).pack(anchor="w", padx=6, pady=1)

        frow = ttk.Frame(opt)
        frow.pack(fill="x", padx=6, pady=(1, 6))
        ttk.Label(frow, text="FPS:").pack(side="left")
        _T(ttk.Button(frow, text="Auto-detect", width=10,
                      command=self._auto_fps),
           "Read the frame rate from a project video."
           ).pack(side="right")
        self.fps_var = tk.IntVar(value=60)
        ttk.Spinbox(frow, textvariable=self.fps_var, from_=1, to=240,
                    width=6).pack(side="right", padx=(0, 4))

        # ---------------- Statistics ----------------
        st = ttk.LabelFrame(left, text="Statistics")
        st.pack(fill="x", pady=6)
        self.enable_stats_var = tk.BooleanVar(value=False)
        _T(ttk.Checkbutton(st, text="Enable statistics",
                           variable=self.enable_stats_var,
                           command=lambda: self.refresh()),
           "Group tests + significance markers on the graphs and the "
           "Σ Stats tables.").pack(anchor="w", padx=6, pady=(4, 1))
        trow = ttk.Frame(st)
        trow.pack(fill="x", padx=6, pady=1)
        ttk.Label(trow, text="Test:").pack(side="left")
        self.stats_paradigm_var = tk.StringVar(value="auto")
        tcb = ttk.Combobox(trow, textvariable=self.stats_paradigm_var,
                           state="readonly", width=14,
                           values=["auto", "parametric", "non-parametric"])
        tcb.pack(side="right")
        tcb.bind("<<ComboboxSelected>>", lambda e: self.refresh())
        Tip(tcb, "auto = Shapiro-Wilk decides; 2 groups use Welch t / "
                 "Mann-Whitney, 3+ use ANOVA / Kruskal-Wallis. Pairwise "
                 "p-values are Bonferroni-corrected.")
        arow = ttk.Frame(st)
        arow.pack(fill="x", padx=6, pady=(1, 6))
        ttk.Label(arow, text="Alpha:").pack(side="left")
        self.stats_alpha_var = tk.DoubleVar(value=0.05)
        ttk.Spinbox(arow, textvariable=self.stats_alpha_var, from_=0.001,
                    to=0.2, increment=0.005, width=6,
                    command=self.refresh).pack(side="right")

        # ---------------- Advanced (collapsed) ----------------
        adv = collapsible(left, "Advanced", collapsed=True, fill="x", pady=6)
        ttk.Label(adv, text="Predictions folder:").pack(anchor="w")
        self.pred_folder_var = tk.StringVar(value="")
        pfrow = ttk.Frame(adv)
        pfrow.pack(fill="x", pady=1)
        self._pred_combo = ttk.Combobox(pfrow,
                                        textvariable=self.pred_folder_var,
                                        state="readonly", width=24)
        self._pred_combo.pack(side="left", fill="x", expand=True)
        self._pred_combo.bind("<<ComboboxSelected>>", self._on_pred_pick)
        _T(ttk.Button(pfrow, text="…", width=3,
                      command=self._browse_predictions),
           "Point at a different folder of *_predictions.csv."
           ).pack(side="left", padx=(4, 0))
        prow2 = ttk.Frame(adv)
        prow2.pack(fill="x", pady=1)
        ttk.Label(prow2, text="Filename prefix:").pack(side="left")
        self.filename_prefix_var = tk.StringVar(value="")
        pe = ttk.Entry(prow2, textvariable=self.filename_prefix_var, width=12)
        pe.pack(side="right")
        Tip(pe, "Optional prefix stripped from filenames before matching "
                "subjects against the key file.")
        self._phase_frame = ttk.Frame(adv)
        ttk.Label(self._phase_frame, text="Phase windows (min):").pack(
            anchor="w")
        self.acute_start_var = tk.DoubleVar(value=0)
        self.acute_end_var = tk.DoubleVar(value=10)
        self.phase2_start_var = tk.DoubleVar(value=10)
        self.phase2_end_var = tk.DoubleVar(value=60)
        for lab, v1, v2 in (("Acute", self.acute_start_var,
                             self.acute_end_var),
                            ("Phase II", self.phase2_start_var,
                             self.phase2_end_var)):
            r = ttk.Frame(self._phase_frame)
            r.pack(fill="x", pady=1)
            ttk.Label(r, text=lab, width=8).pack(side="left")
            ttk.Spinbox(r, textvariable=v1, from_=0, to=240,
                        width=5).pack(side="left")
            ttk.Label(r, text="–").pack(side="left")
            ttk.Spinbox(r, textvariable=v2, from_=0, to=240,
                        width=5).pack(side="left")
        self.enable_phase_analysis = tk.BooleanVar(value=False)

        # ---------------- Run + export ----------------
        rrow = ttk.Frame(left)
        rrow.pack(fill="x", pady=(8, 0))
        self._run_btn = ttk.Button(rrow, text="▶ Run analysis",
                                   command=self.run_analysis)
        self._run_btn.pack(side="left", fill="x", expand=True)
        Tip(self._run_btn, "Analyze the selected behaviors on a worker "
                           "thread; Stop cancels between files.")
        self._stop_btn = ttk.Button(rrow, text="■ Stop", width=7,
                                    state="disabled", command=self._stop)
        self._stop_btn.pack(side="left", padx=(6, 0))
        self._progress = ttk.Progressbar(left, mode="determinate")
        self._progress.pack(fill="x", pady=(4, 0))
        self._status = tk.StringVar(value="")
        ttk.Label(left, textvariable=self._status, foreground="#444444",
                  wraplength=270, justify="left").pack(anchor="w",
                                                       pady=(2, 0))
        erow = ttk.Frame(left)
        erow.pack(fill="x", pady=(6, 0))
        _T(ttk.Button(erow, text="Export CSV…", command=self._export_master),
           "Full results table + a .meta.json sidecar recording the exact "
           "inputs and settings (incl. git version)."
           ).pack(side="left", fill="x", expand=True)
        _T(ttk.Button(erow, text="Export figure…", command=self._export_fig),
           "Save the current graph as PNG, SVG, or PDF."
           ).pack(side="left", fill="x", expand=True, padx=(6, 0))

        # ---------------- right pane: topbar + canvas ----------------
        self._VIEW_LABELS = [
            ("timecourse", "Time Course"),
            ("traces", "Individual Traces"),
            ("total_time", "Total Time"),
            ("bouts", "Bout Analysis"),
            ("phase", "Phase Analysis"),
            ("heatmap_time", "Heatmap (Time)"),
            ("heatmap_bouts", "Heatmap (Bouts)"),
            ("cumulative", "Cumulative Time"),
            ("cumulative_bouts", "Cumulative Bouts"),
            ("mean_timecourse", "Mean Timecourse (1 Hz)"),
            ("latency", "Latency"),
        ]
        top1 = ttk.Frame(right)
        top1.pack(fill="x", pady=(0, 2))
        ttk.Label(top1, text="Graph:",
                  font=("Segoe UI", 9, "bold")).pack(side="left")
        self._view = tk.StringVar(value="timecourse")
        self._view_label_var = tk.StringVar(value="Time Course")
        self._view_cb = ttk.Combobox(top1, textvariable=self._view_label_var,
                                     state="readonly", width=22)
        self._view_cb.pack(side="left", padx=(6, 0))
        self._view_cb.bind("<<ComboboxSelected>>", self._on_view_pick)
        Tip(self._view_cb, "Which graph to show. Phase and Mean-Timecourse "
                           "views appear only when their data exists.")
        ttk.Label(top1, text="Behavior:").pack(side="left", padx=(10, 2))
        self._behavior_var = tk.StringVar(value="")
        self._behavior_cb = ttk.Combobox(top1,
                                         textvariable=self._behavior_var,
                                         state="readonly", width=18)
        self._behavior_cb.pack(side="left")
        self._behavior_cb.bind("<<ComboboxSelected>>",
                               lambda e: self.refresh())
        _T(ttk.Button(top1, text="🎨 ⚙", command=self._edit_style),
           "Plot style: group colors, error bars, markers, significance "
           "markers, fonts, heatmap colormap. Saved with the project and "
           "shared by every analysis tab.").pack(side="left", padx=(8, 0))
        self._stats_mode = tk.BooleanVar(value=False)
        _T(ttk.Checkbutton(top1, text="Σ Stats", variable=self._stats_mode,
                           style="Toolbutton",
                           command=self._apply_stats_mode),
           "Flip between the graph and its statistics (descriptives, group "
           "tests, timecourse ANOVA + per-bin post-hoc)."
           ).pack(side="left", padx=(6, 0))

        top2 = ttk.Frame(right)
        top2.pack(fill="x", pady=(0, 4))
        self._metric_row = ttk.Frame(top2)
        ttk.Label(self._metric_row, text="Metric:").pack(side="left")
        self._metric_var = tk.StringVar(value=_METRIC_LABELS[0])
        mcb = ttk.Combobox(self._metric_row, textvariable=self._metric_var,
                           state="readonly", width=22,
                           values=_METRIC_LABELS)
        mcb.pack(side="left", padx=(4, 0))
        mcb.bind("<<ComboboxSelected>>", lambda e: self.refresh())
        wrow = ttk.Frame(top2)
        wrow.pack(side="right")
        _T(ttk.Button(wrow, text="Axis…", width=6,
                      command=self._axis_dialog),
           "Manual y-axis limits for this view (blank = auto)."
           ).pack(side="right")
        ttk.Label(wrow, text="Window (min):").pack(side="left")
        self.time_window_var = tk.DoubleVar(value=0)
        wsp = ttk.Spinbox(wrow, textvariable=self.time_window_var, from_=0,
                          to=600, width=6, command=self.refresh)
        wsp.pack(side="left", padx=(2, 8))
        wsp.bind("<Return>", lambda e: self.refresh())
        Tip(wsp, "Show data up to this many minutes (0 = everything).")

        self._fig = Figure(figsize=(9, 5.5), constrained_layout=True)
        self._canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        self._nav = NavigationToolbar2Tk(self._canvas, right)
        from tkinter import scrolledtext as _stext
        self._stats_widget = _stext.ScrolledText(right, wrap="none",
                                                 font=("Consolas", 9))
        self._refresh_view_choices()
        self.refresh()

    # ------------------------------------------------------------------ data

    def scan_project_folder(self, folder=None):
        import analysis_core as core
        folder = folder or self._project()
        if not folder or not os.path.isdir(folder):
            self._scan_status.set("No project scanned.")
            return
        self.analysis_project_var.set(folder)
        scan = core.scan_project(folder)
        key_files = scan["key_files"]
        pred_folders = scan["pred_folders"]
        self._key_combo.configure(values=[os.path.relpath(k, folder)
                                          for k in key_files])
        self._key_paths = list(key_files)
        if key_files and self.key_df is None:
            try:
                self._load_key(key_files[0])
            except Exception:
                pass
        # scan_project returns (folder, n_files) tuples for pred folders
        pred_paths = [pf[0] if isinstance(pf, (tuple, list)) else pf
                      for pf in pred_folders]
        self._pred_combo.configure(values=[os.path.relpath(p, folder)
                                           for p in pred_paths])
        self._pred_paths = pred_paths
        pred = self.pred_folder_var.get()
        target = None
        if pred and os.path.isdir(os.path.join(folder, pred)):
            target = os.path.join(folder, pred)
        elif scan["auto_pred_folder"]:
            target = scan["auto_pred_folder"]
            self.pred_folder_var.set(os.path.relpath(target, folder))
        if target:
            self._scan_pred_folder(target)
        n = len(self._files)
        self._scan_status.set(
            f"{os.path.basename(folder)}: {len(pred_folders)} "
            f"prediction folder(s), {n} file(s), "
            f"{len(self._behaviors)} behavior(s)."
            if n else f"{os.path.basename(folder)}: no prediction files — "
                      "run classifiers first.")
        self._dash_subjects_overview()
        # Predictions + key already on disk -> analyze automatically so the
        # tab arrives with graphs. Deferred so a following synchronous run
        # (the batch handoff) wins; skipped once results exist.
        if self._files and self.key_df is not None:
            self.after(400, self._auto_run_if_idle)

    def _auto_run_if_idle(self):
        if (self.results_df is None and not self._running
                and self._files and self.key_df is not None):
            self.run_analysis(silent=False)

    def _scan_pred_folder(self, folder):
        import analysis_core as core
        self._files, self._behaviors = core.scan_predictions(folder)
        self._beh_list.delete(0, "end")
        for b in sorted(self._behaviors):
            self._beh_list.insert("end", b)
        self._beh_list.select_set(0, "end")
        self._session_filter.set_sessions(
            sorted({self._session_of(f) for f in self._files}),
            key_df=self.key_df)

    @staticmethod
    def _session_of(f):
        """Session name of a prediction FileInfo: stem minus the
        _predictions suffix and the classifier/behavior token (same
        derivation the Sequencing tab uses)."""
        stem = os.path.splitext(f.filename)[0]
        for suf in ("_predictions", "_prediction"):
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                break
        for tok in ("_classifier", "_PixelPaws"):
            if tok in stem:
                stem = stem.split(tok)[0]
                break
        else:
            beh = getattr(f, "behavior", "") or ""
            if beh and stem.lower().endswith("_" + beh.lower()):
                stem = stem[:-(len(beh) + 1)]
        return stem

    def _selected_behaviors(self):
        sel = [self._beh_list.get(i) for i in self._beh_list.curselection()]
        return sel or sorted(self._behaviors)

    def _load_key(self, path):
        import analysis_core as core
        self.key_df = core.load_key_file(path)
        self.key_file_path = path
        self.key_file_var.set(os.path.relpath(path, self._project())
                              if self._project() else path)
        groups = list(self.key_df["Treatment"].unique())
        self._key_status.set(f"{len(self.key_df)} subjects · "
                             f"{len(groups)} group(s): "
                             + ", ".join(map(str, groups))[:80])
        if self._files:
            self._session_filter.set_sessions(
                sorted({self._session_of(f) for f in self._files}),
                key_df=self.key_df)

    def _on_key_pick(self, _evt=None):
        idx = self._key_combo.current()
        if 0 <= idx < len(getattr(self, "_key_paths", [])):
            try:
                self._load_key(self._key_paths[idx])
                self._dash_subjects_overview()
            except Exception as e:
                messagebox.showerror("Key file", str(e), parent=self)

    def _browse_key(self):
        p = filedialog.askopenfilename(
            title="Key file", filetypes=[("Tables", "*.csv *.xlsx"),
                                         ("All files", "*.*")])
        if p:
            try:
                self._load_key(p)
                self._dash_subjects_overview()
            except Exception as e:
                messagebox.showerror("Key file", str(e), parent=self)

    def _generate_key(self):
        try:
            self.app._open_key_file_dialog()
        except Exception as e:
            messagebox.showerror("Key file", str(e), parent=self)

    def _on_pred_pick(self, _evt=None):
        idx = self._pred_combo.current()
        if 0 <= idx < len(getattr(self, "_pred_paths", [])):
            self._scan_pred_folder(self._pred_paths[idx])

    def _browse_predictions(self):
        d = filedialog.askdirectory(title="Predictions folder")
        if d:
            self.pred_folder_var.set(d)
            self._scan_pred_folder(d)

    def _dash_subjects_overview(self):
        import analysis_core as core
        try:
            self._subjects_rows = core.subjects_overview(self._project(),
                                                         self.key_df)
        except Exception:
            self._subjects_rows = []
        n = len(self._subjects_rows)
        groups = {g for _, g in self._subjects_rows if g}
        unmatched = sum(1 for _, g in self._subjects_rows if not g)
        txt = f"{n} subject(s) · {len(groups)} group(s)"
        if unmatched:
            txt += f" · {unmatched} unmatched"
        self._subjects_status.set(txt if n else "")

    def _show_subjects_table(self):
        win = tk.Toplevel(self)
        win.title("Subjects")
        win.transient(self.winfo_toplevel())
        tree = ttk.Treeview(win, columns=("subject", "group"),
                            show="headings", height=16)
        tree.heading("subject", text="Subject")
        tree.heading("group", text="Group")
        tree.column("subject", width=220)
        tree.column("group", width=140)
        for s, g in self._subjects_rows:
            tree.insert("", "end", values=(s, g or "— unmatched —"))
        tree.pack(fill="both", expand=True, padx=8, pady=8)

    def _apply_preset(self, _evt=None):
        p = _PRESETS.get(self._preset_var.get(), _PRESETS["Default"])
        self.bin_size_var.set(p["bin_size"])
        self.bin_unit_var.set(p["bin_unit"])
        self.no_time_bins_var.set(p["whole"])
        self.enable_stats_var.set(p["stats"])
        self.enable_phase_analysis.set(p["phases"])
        if p["phases"]:
            self.acute_start_var.set(p["acute"][0])
            self.acute_end_var.set(p["acute"][1])
            self.phase2_start_var.set(p["phase2"][0])
            self.phase2_end_var.set(p["phase2"][1])
            self._phase_frame.pack(fill="x", pady=(4, 0))
        else:
            self._phase_frame.pack_forget()

    def _auto_fps(self):
        p = filedialog.askopenfilename(
            title="Pick a video to read FPS from",
            initialdir=os.path.join(self._project(), "videos"),
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv")])
        if not p:
            return
        try:
            import cv2
            cap = cv2.VideoCapture(p)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps > 0:
                self.fps_var.set(int(round(fps)))
                self._status.set(f"FPS detected: {fps:g}")
        except Exception as e:
            messagebox.showerror("FPS", str(e), parent=self)

    # ------------------------------------------------------------------ run

    def _build_cfg(self):
        import analysis_core as core
        size = float(self.bin_size_var.get())
        bin_min = size / 60.0 if self.bin_unit_var.get() == "seconds" else size
        phases = ()
        if self.enable_phase_analysis.get():
            phases = (core.PhaseWindow("Acute",
                                       float(self.acute_start_var.get()),
                                       float(self.acute_end_var.get()), -1),
                      core.PhaseWindow("Phase_II",
                                       float(self.phase2_start_var.get()),
                                       float(self.phase2_end_var.get()), -2))
        return core.AnalysisConfig(
            bin_size_min=bin_min,
            whole_session=bool(self.no_time_bins_var.get()),
            fps=float(self.fps_var.get() or 60),
            analyze_mode=self.analyze_mode.get(),
            phases=phases,
            filename_prefix=self.filename_prefix_var.get().strip())

    def _files_for_run(self):
        beh = set(self._selected_behaviors())
        return [f for f in self._files if f.behavior in beh
                and self._session_filter.allows(self._session_of(f))]

    def run_analysis(self, silent=False):
        import analysis_core as core
        if self._running:
            if not silent:
                messagebox.showinfo("Running",
                                    "An analysis is already in progress.",
                                    parent=self)
            return
        if self.key_df is None:
            if not silent:
                messagebox.showwarning("No key file",
                                       "Load a key file first.", parent=self)
            return
        files = self._files_for_run()
        if not files:
            if not silent:
                messagebox.showwarning("No predictions",
                                       "Scan a predictions folder first.",
                                       parent=self)
            return
        cfg = self._build_cfg()
        if silent:
            # Batch handoff reads results_df immediately after this call, so
            # the silent path is synchronous (core is headless and safe).
            res = core.run_analysis(files, self.key_df, cfg)
            self._finish_run(res)
            return
        self._running = True
        self._cancel.clear()
        self._run_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._progress.configure(value=0, maximum=max(len(files), 1))
        self._status.set("Analyzing…")

        def _worker():
            try:
                res = core.run_analysis(
                    files, self.key_df, cfg,
                    progress_cb=lambda d, t, m: self._run_queue.put(
                        ("prog", d, t, m)),
                    cancel=self._cancel)
                self._run_queue.put(("done", res))
            except Exception as e:
                self._run_queue.put(("err", e))

        threading.Thread(target=_worker, daemon=True).start()
        self.after(100, self._poll_run)

    def _poll_run(self):
        try:
            while True:
                msg = self._run_queue.get_nowait()
                if msg[0] == "prog":
                    _, d, t, m = msg
                    self._progress.configure(value=d, maximum=max(t, 1))
                    self._status.set(m)
                elif msg[0] == "done":
                    self._finish_run(msg[1])
                    return
                elif msg[0] == "err":
                    self._running = False
                    self._run_btn.configure(state="normal")
                    self._stop_btn.configure(state="disabled")
                    self._status.set(f"Analysis failed: {msg[1]}")
                    return
        except queue.Empty:
            pass
        if self._running:
            self.after(100, self._poll_run)

    def _finish_run(self, res):
        self._running = False
        try:
            self._run_btn.configure(state="normal")
            self._stop_btn.configure(state="disabled")
            self._progress.configure(value=self._progress["maximum"])
        except Exception:
            pass
        self.results_df = res.results_df
        self.perframe_data = res.perframe_data
        n = 0 if self.results_df is None else len(self.results_df)
        skip = f"; {len(res.skipped)} file(s) skipped (no key match)" \
            if res.skipped else ""
        self._status.set(f"Done: {n} rows{skip}.")
        behs = (sorted(self.results_df["Behavior"].unique())
                if n else [])
        self._behavior_cb.configure(values=behs)
        if behs and self._behavior_var.get() not in behs:
            self._behavior_var.set(behs[0])
        self._refresh_view_choices()
        self.refresh()

    def _stop(self):
        self._cancel.set()
        self._status.set("Stopping…")

    # ------------------------------------------------------------------ views

    def _refresh_view_choices(self):
        have_phase = (self.results_df is not None
                      and "Bin_Index" in self.results_df.columns
                      and (self.results_df["Bin_Index"] < 0).any())
        have_pf = bool(self.perframe_data)
        labels = [l for k, l in self._VIEW_LABELS
                  if not (k == "phase" and not have_phase)
                  and not (k == "mean_timecourse" and not have_pf)]
        self._view_cb.configure(values=labels)
        cur = dict(self._VIEW_LABELS).get(self._view.get())
        if cur not in labels and labels:
            self._view_label_var.set(labels[0])
            self._view.set([k for k, l in self._VIEW_LABELS
                            if l == labels[0]][0])

    def _on_view_pick(self, _evt=None):
        lab = self._view_label_var.get()
        for k, l in self._VIEW_LABELS:
            if l == lab:
                self._view.set(k)
                break
        self.refresh()

    def _metric(self):
        lab = self._metric_var.get()
        for col, agg, l in _METRICS:
            if l == lab:
                return col, agg, l
        return _METRICS[0]

    def _groups(self, df=None):
        if self.key_df is None:
            return []
        order = [str(t) for t in self.key_df["Treatment"].unique()]
        if df is not None:
            present = set(df["Treatment"].astype(str))
            order = [g for g in order if g in present]
        return order

    def _opts(self):
        import plot_style
        return plot_style.get_options(self._project())

    def _colors(self, groups):
        import plot_style
        return plot_style.get_colors(self._project(), groups)

    def _df(self, include_phase=False):
        df = self.results_df
        if df is None or df.empty:
            return None
        b = self._behavior_var.get()
        if b:
            df = df[df["Behavior"] == b]
        if "Bin_Index" in df.columns and not include_phase:
            df = df[df["Bin_Index"] >= 0]
        w = float(self.time_window_var.get() or 0)
        if w > 0 and "Bin_Start_Min" in df.columns:
            df = df[df["Bin_Start_Min"] < w]
        return df.copy()

    def _binned_matrices(self, df, col):
        """(bin_starts, {group: matrix subjects×bins}) for series views."""
        piv = df.pivot_table(values=col, index="Subject",
                             columns="Bin_Start_Min", aggfunc="mean")
        piv = piv.sort_index(axis=1)
        subj_grp = (df[["Subject", "Treatment"]].drop_duplicates()
                    .set_index("Subject")["Treatment"].astype(str).to_dict())
        out = {}
        for g in self._groups(df):
            subs = [s for s in piv.index if subj_grp.get(s) == g]
            if subs:
                out[g] = piv.loc[subs].to_numpy(float)
        return piv.columns.to_numpy(float), out

    def _per_subject(self, df, col, agg):
        per = df.groupby(["Subject", "Treatment"])[col].agg(agg).reset_index()
        per["Treatment"] = per["Treatment"].astype(str)
        return {g: per.loc[per["Treatment"] == g, col].to_numpy(float)
                for g in self._groups(df)}

    def refresh(self):
        if self._stats_mode.get():
            self._fill_stats()
            return
        self._fig.clear()
        df = self._df()
        if df is None or df.empty:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, "Run an analysis to see graphs.",
                    ha="center", va="center")
            ax.set_axis_off()
            self._canvas.draw_idle()
            return
        view = self._view.get()
        # metric row only where it applies
        if view in ("timecourse", "traces", "total_time"):
            self._metric_row.pack(side="left")
        else:
            self._metric_row.pack_forget()
        try:
            getattr(self, f"_draw_{view}")(df)
            ymin_ymax = self._axis_overrides.get(view)
            if ymin_ymax and len(self._fig.axes) == 1:
                lo, hi = ymin_ymax
                ax = self._fig.axes[0]
                cur = ax.get_ylim()
                ax.set_ylim(lo if lo is not None else cur[0],
                            hi if hi is not None else cur[1])
        except Exception as e:
            self._fig.clear()
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Could not draw: {e}", ha="center",
                    va="center", wrap=True)
            ax.set_axis_off()
        self._canvas.draw_idle()

    # ---- series views -------------------------------------------------

    def _series_view(self, df, col, ylabel, title, cumulative=False,
                     force_traces=False):
        import plot_style
        opts = self._opts()
        if force_traces:
            opts = dict(opts, show_individual=True)
        groups = self._groups(df)
        cmap = self._colors(groups)
        ax = self._fig.add_subplot(111)
        if cumulative:
            df = df.sort_values("Bin_Start_Min").copy()
            df[col + "_cum"] = df.groupby("Subject")[col].cumsum()
            col = col + "_cum"
        x, mats = self._binned_matrices(df, col)
        if len(x) == 1:
            # a one-bin run draws invisible single-point lines otherwise
            opts = dict(opts, show_markers=True)
        for gi, g in enumerate(groups):
            if g not in mats:
                continue
            plot_style.draw_series(
                ax, x, mats[g], cmap[g], opts,
                label=plot_style.group_label(g, mats[g].shape[0], opts),
                group=g)
        ax.legend(frameon=False, fontsize=8)
        ax.set_xlabel("Time (minutes)", fontsize=9)
        elab = ("" if opts["error_display"] == "none"
                else " (" + plot_style.error_label(opts["error_type"]) + ")")
        ax.set_ylabel(ylabel + elab, fontsize=9)
        if opts.get("show_titles", True):
            b = self._behavior_var.get()
            ax.set_title(f"{b} — {title}" if b else title, fontsize=11)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        # per-bin significance markers
        if self.enable_stats_var.get() and len(groups) >= 2:
            self._annotate_bins(ax, x, mats, groups, opts)
        plot_style.apply_frame_options(ax, opts)

    def _annotate_bins(self, ax, x, mats, groups, opts):
        import analysis_core as core
        import plot_style
        alpha = float(self.stats_alpha_var.get())
        paradigm = {"auto": "auto", "parametric": "parametric",
                    "non-parametric": "nonparametric"}[
            self.stats_paradigm_var.get()]
        top = max((m.max() for m in mats.values() if m.size), default=0)
        for j in range(len(x)):
            samples = [mats[g][:, j] for g in groups if g in mats
                       and mats[g].shape[1] > j]
            samples = [s[np.isfinite(s)] for s in samples]
            samples = [s for s in samples if len(s) >= 2]
            if len(samples) < 2:
                continue
            try:
                res = core.perform_statistical_test(
                    dict(enumerate(samples)), alpha=alpha, paradigm=paradigm)
            except Exception:
                continue
            if not res:
                continue
            p = res.get("p_value")
            if p is None:
                continue
            mark = plot_style.sig_text(p, alpha,
                                       opts.get("sig_style", "asterisk"))
            if opts.get("sig_style") == "p-value":
                mark = ""
            if mark:
                ax.text(x[j], top * 1.04, mark, ha="center", fontsize=8)

    def _draw_timecourse(self, df):
        col, _agg, lab = self._metric()
        self._series_view(df, col, lab, f"{lab} over time")

    def _draw_traces(self, df):
        col, _agg, lab = self._metric()
        self._series_view(df, col, lab, f"{lab} over time (individual)",
                          force_traces=True)

    def _draw_cumulative(self, df):
        self._series_view(df, "Total_Time_s", "Cumulative time (s)",
                          "Cumulative behavior time", cumulative=True)

    def _draw_cumulative_bouts(self, df):
        self._series_view(df, "N_Bouts", "Cumulative bouts",
                          "Cumulative number of bouts", cumulative=True)

    # ---- bar / scatter views ------------------------------------------

    def _bar_points(self, ax, per_group, ylabel, title):
        import plot_style
        import analysis_core as core
        opts = self._opts()
        groups = [g for g in per_group if len(per_group[g])]
        cmap = self._colors(groups)
        x = np.arange(len(groups))
        rng = np.random.default_rng(42)
        means, errs = [], []
        for g in groups:
            v = per_group[g]
            means.append(float(np.mean(v)))
            errs.append(plot_style.calc_error(v, opts["error_type"]))
        yerr = errs if opts["error_display"] != "none" else None
        edge = (dict(edgecolor="black", linewidth=0.8)
                if opts["bar_edges"] else {})
        ax.bar(x, means, width=0.65, yerr=yerr, capsize=4,
               color=[cmap[g] for g in groups], alpha=0.75,
               error_kw=dict(lw=1.0), zorder=2, **edge)
        for i, g in enumerate(groups):
            v = per_group[g]
            jit = rng.normal(0, 0.05, len(v))
            ax.plot(np.full(len(v), x[i]) + jit, v, ls="none", marker="o",
                    ms=6, mfc="white", mec=cmap[g], mew=1.2, alpha=0.85,
                    zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([plot_style.group_label(g, len(per_group[g]),
                                                   opts) for g in groups],
                           fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        if opts.get("show_titles", True):
            ax.set_title(title, fontsize=11)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if self.enable_stats_var.get() and len(groups) >= 2:
            alpha = float(self.stats_alpha_var.get())
            paradigm = {"auto": "auto", "parametric": "parametric",
                        "non-parametric": "nonparametric"}[
                self.stats_paradigm_var.get()]
            try:
                res = core.perform_statistical_test(
                    {g: per_group[g] for g in groups}, alpha=alpha,
                    paradigm=paradigm)
            except Exception:
                res = None
            if res and res.get("p_value") is not None:
                p = res["p_value"]
                mark = plot_style.sig_text(p, alpha,
                                           opts.get("sig_style", "asterisk"))
                name = res.get("test_type", "test")
                txt = f"{name} p={p:.3g}" + (f" {mark}" if mark and
                                             opts.get("sig_style")
                                             != "p-value" else "")
                ax.text(0.98, 0.98, txt, transform=ax.transAxes, ha="right",
                        va="top", fontsize=8,
                        color="#b00020" if p < alpha else "#666666")
        plot_style.apply_frame_options(ax, opts)

    def _draw_total_time(self, df):
        col, agg, lab = self._metric()
        ax = self._fig.add_subplot(111)
        b = self._behavior_var.get()
        self._bar_points(ax, self._per_subject(df, col, agg), lab,
                         f"{b} — {lab}" if b else lab)

    def _draw_latency(self, df):
        ax = self._fig.add_subplot(111)
        active = df[df["Total_Time_s"] > 0]
        if "Latency_s" in active.columns and active["Latency_s"].notna().any():
            lat = (active.dropna(subset=["Latency_s"])
                   .groupby(["Subject", "Treatment"])["Latency_s"].min()
                   .reset_index())
            lat["val"] = lat["Latency_s"] / 60.0
        else:
            lat = (active.groupby(["Subject", "Treatment"])["Bin_Start_Min"]
                   .min().reset_index())
            lat["val"] = lat["Bin_Start_Min"]
        lat["Treatment"] = lat["Treatment"].astype(str)
        per = {g: lat.loc[lat["Treatment"] == g, "val"].to_numpy(float)
               for g in self._groups(df)}
        b = self._behavior_var.get()
        self._bar_points(ax, per, "Latency to first bout (min)",
                         f"{b} — Latency" if b else "Latency")

    def _draw_bouts(self, df):
        import plot_style
        opts = self._opts()
        ax1, ax2 = self._fig.subplots(1, 2)
        for ax, col, agg, lab in ((ax1, "N_Bouts", "sum", "Total bouts"),
                                  (ax2, "Mean_Bout_Duration_s", "mean",
                                   "Mean bout duration (s)")):
            per = self._per_subject(df, col, agg)
            groups = [g for g in per if len(per[g])]
            cmap = self._colors(groups)
            data = [per[g] for g in groups]
            bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                            widths=0.6,
                            medianprops=dict(color="#222222"))
            for patch, g in zip(bp["boxes"], groups):
                patch.set_facecolor(cmap[g])
                patch.set_alpha(0.5)
            rng = np.random.default_rng(42)
            for i, g in enumerate(groups):
                v = per[g]
                ax.plot(rng.normal(i + 1, 0.05, len(v)), v, ls="none",
                        marker="o", ms=6, mfc="white", mec=cmap[g],
                        mew=1.2, alpha=0.85, zorder=3)
            ax.set_xticks(np.arange(1, len(groups) + 1))
            ax.set_xticklabels(
                [plot_style.group_label(g, len(per[g]), opts)
                 for g in groups], fontsize=8)
            ax.set_ylabel(lab, fontsize=9)
            if opts.get("show_titles", True):
                ax.set_title(lab, fontsize=10)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            plot_style.apply_frame_options(ax, opts)

    def _draw_phase(self, _df):
        import plot_style
        df = self._df(include_phase=True)
        pdf = df[df["Bin_Index"] < 0]
        if pdf.empty:
            raise ValueError("no phase rows — enable the Formalin preset "
                             "and re-run")
        ax1, ax2 = self._fig.subplots(1, 2)
        for ax, idx, lab in ((ax1, -1, "Acute phase"),
                             (ax2, -2, "Phase II")):
            sub = pdf[pdf["Bin_Index"] == idx]
            per = self._per_subject(sub, "Total_Time_s", "sum")
            self._bar_points(ax, per, "Time in behavior (s)", lab)

    def _draw_mean_timecourse(self, _df):
        import plot_style
        opts = self._opts()
        b = self._behavior_var.get()
        arrays = {}
        for (subj, trt, beh), arr in self.perframe_data.items():
            if beh != b:
                continue
            arrays.setdefault(str(trt), []).append(arr)
        if not arrays:
            raise ValueError("no per-frame data for this behavior")
        groups = [g for g in self._groups() if g in arrays]
        cmap = self._colors(groups)
        min_len = min(len(a) for arrs in arrays.values() for a in arrs)
        size = float(self.bin_size_var.get())
        bin_sec = max(int(size if self.bin_unit_var.get() == "seconds"
                          else size * 60), 1)
        n_bins = int(np.ceil(min_len / bin_sec))
        ax = self._fig.add_subplot(111)
        for g in groups:
            mat = np.array([a[:min_len] for a in arrays[g]], float)
            means, times = [], []
            for i in range(n_bins):
                s0, s1 = i * bin_sec, min((i + 1) * bin_sec, min_len)
                means.append(float(np.mean(np.sum(mat[:, s0:s1], axis=1))))
                times.append(s0)
            tx = (np.array(times, float) + bin_sec / 2)
            unit_min = bin_sec >= 60
            tx = tx / 60.0 if unit_min else tx
            ax.fill_between(tx, 0, means, alpha=0.3, color=cmap[g],
                            label=plot_style.group_label(g, mat.shape[0],
                                                         opts))
            ax.plot(tx, means, color=cmap[g],
                    lw=float(opts.get("line_width", 1.6)))
        unit = "min" if bin_sec >= 60 else "s"
        ax.set_xlabel(f"Time ({'minutes' if unit == 'min' else 'seconds'})",
                      fontsize=9)
        ax.set_ylabel(f"Mean time in behavior (s / "
                      f"{bin_sec // 60 if unit == 'min' else bin_sec}"
                      f"{unit} bin)", fontsize=9)
        ax.legend(frameon=False, fontsize=8)
        if opts.get("show_titles", True):
            ax.set_title(f"{b} — Mean timecourse (1 Hz per-frame)",
                         fontsize=11)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        plot_style.apply_frame_options(ax, opts)

    def _draw_heatmap(self, df, col, what):
        import plot_style
        opts = self._opts()
        groups = self._groups(df)
        subj_grp = (df[["Subject", "Treatment"]].drop_duplicates()
                    .set_index("Subject")["Treatment"].astype(str).to_dict())
        ordered = [s for g in groups
                   for s in sorted(x for x, gg in subj_grp.items()
                                   if gg == g)]
        piv = df.pivot_table(values=col, index="Subject",
                             columns="Bin_Start_Min", aggfunc="mean")
        piv = piv.reindex(ordered).sort_index(axis=1)
        ax = self._fig.add_subplot(111)
        im = ax.imshow(piv.to_numpy(float), aspect="auto",
                       interpolation="nearest",
                       cmap=opts.get("heatmap_cmap", "viridis"))
        ax.set_xticks(np.arange(len(piv.columns)))
        ax.set_xticklabels([f"{c:g}" for c in piv.columns], fontsize=7)
        ax.set_yticks(np.arange(len(piv.index)))
        ax.set_yticklabels(piv.index, fontsize=7)
        ax.set_xlabel("Time bin (minutes)", fontsize=9)
        # group separators + labels
        i0 = 0
        for g in groups:
            n = sum(1 for s in ordered if subj_grp.get(s) == g)
            if n == 0:
                continue
            if i0 > 0:
                ax.axhline(i0 - 0.5, color="white", lw=2)
            ax.text(1.01, i0 + n / 2 - 0.5,
                    plot_style.group_label(g, n, opts),
                    transform=ax.get_yaxis_transform(), ha="left",
                    va="center", fontsize=8, rotation=270)
            i0 += n
        cb = self._fig.colorbar(im, ax=ax, shrink=0.8, pad=0.10)
        cb.set_label(what, fontsize=8)
        cb.ax.tick_params(labelsize=7)
        if opts.get("show_titles", True):
            b = self._behavior_var.get()
            ax.set_title(f"{b} — {what} per subject", fontsize=11)

    def _draw_heatmap_time(self, df):
        self._draw_heatmap(df, "Total_Time_s", "Time in behavior (s)")

    def _draw_heatmap_bouts(self, df):
        self._draw_heatmap(df, "N_Bouts", "Number of bouts")

    # ------------------------------------------------------------------ stats

    def _apply_stats_mode(self):
        if self._stats_mode.get():
            self._canvas.get_tk_widget().pack_forget()
            self._nav.pack_forget()
            self._stats_widget.pack(fill="both", expand=True)
        else:
            self._stats_widget.pack_forget()
            self._canvas.get_tk_widget().pack(fill="both", expand=True)
            self._nav.pack(fill="x")
        self.refresh()

    def _fill_stats(self):
        self._stats_widget.configure(state="normal")
        self._stats_widget.delete("1.0", "end")
        try:
            txt = self._stats_text()
        except Exception as e:
            txt = f"Could not compute statistics: {e}"
        self._stats_widget.insert("1.0", txt)
        self._stats_widget.configure(state="disabled")

    def _stats_text(self):
        import analysis_core as core
        import plot_style
        df = self._df()
        if df is None or df.empty:
            return "Run an analysis first."
        opts = self._opts()
        alpha = float(self.stats_alpha_var.get())
        paradigm = {"auto": "auto", "parametric": "parametric",
                    "non-parametric": "nonparametric"}[
            self.stats_paradigm_var.get()]
        col, agg, lab = self._metric()
        b = self._behavior_var.get()
        per = self._per_subject(df, col, agg)
        groups = [g for g in per if len(per[g])]
        lines = [f"{b} — {lab}", "=" * 40, ""]
        rows_by_g = {g: per[g].reshape(-1, 1) for g in groups}
        lines.append(plot_style.stats_table(
            f"Per-subject {lab} ({agg} across bins)", [lab], rows_by_g,
            opts, test_fn=None, unit=lab))
        res = core.perform_statistical_test(
            {g: per[g] for g in groups}, alpha=alpha, paradigm=paradigm)
        lines.append("")
        if res:
            lines.append(f"Omnibus: {res.get('test_type', '?')}  "
                         f"p={res.get('p_value', float('nan')):.4g}")
            eff = res.get("effect_size")
            if eff is not None:
                lines.append(f"Effect size: "
                             f"{res.get('effect_size_type', '')} "
                             f"= {eff:.3g}")
            pw = res.get("pairwise") or {}
            if pw:
                lines.append("Pairwise (Bonferroni-corrected):")
                for pair, row in pw.items():
                    lines.append(
                        f"  {pair.replace('_vs_', ' vs ')}: "
                        f"p={row['p_corrected']:.4g}"
                        f"{'  *' if row['significant'] else ''}")
        else:
            lines.append("No omnibus test (need ≥2 groups with n ≥ 2).")
        # timecourse stats for series views
        if self._view.get() in ("timecourse", "traces",
                                "mean_timecourse") and len(groups) >= 2:
            lines.append("")
            try:
                an = core.timecourse_anova(df, col, alpha)
                if an:
                    lines.append(f"Two-way ANOVA ({lab}):")
                    for k, v in an.items():
                        lines.append(f"  {k}: {v}")
            except Exception as e:
                lines.append(f"Timecourse ANOVA unavailable: {e}")
            try:
                ph = core.timecourse_posthoc(df, groups, col, alpha,
                                             paradigm)
                if ph is not None and len(ph):
                    sig = ph[ph["significant"]]
                    lines.append("")
                    lines.append(f"Per-bin post-hoc: "
                                 f"{len(sig)}/{len(ph)} comparisons "
                                 f"significant (Bonferroni)")
                    for _, r in sig.iterrows():
                        lines.append(
                            f"  {r['Bin_Start_Min']:g} min  "
                            f"{r['group_a']} vs {r['group_b']}  "
                            f"p={r['p_corrected']:.4g}")
            except Exception as e:
                lines.append(f"Post-hoc unavailable: {e}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ misc

    def _edit_style(self):
        import plot_style
        plot_style.open_options_dialog(self, self._project(),
                                       self._groups(),
                                       on_apply=self.refresh)

    def _axis_dialog(self):
        view = self._view.get()
        cur = self._axis_overrides.get(view, (None, None))
        win = tk.Toplevel(self)
        win.title(f"Axis — {dict(self._VIEW_LABELS).get(view, view)}")
        win.transient(self.winfo_toplevel())
        frm = ttk.Frame(win, padding=10)
        frm.pack()
        ttk.Label(frm, text="Y min (blank = auto):").grid(row=0, column=0,
                                                          sticky="w")
        v1 = tk.StringVar(value="" if cur[0] is None else str(cur[0]))
        ttk.Entry(frm, textvariable=v1, width=10).grid(row=0, column=1)
        ttk.Label(frm, text="Y max (blank = auto):").grid(row=1, column=0,
                                                          sticky="w")
        v2 = tk.StringVar(value="" if cur[1] is None else str(cur[1]))
        ttk.Entry(frm, textvariable=v2, width=10).grid(row=1, column=1)

        def _apply():
            def _f(s):
                s = s.strip()
                return float(s) if s else None
            try:
                self._axis_overrides[view] = (_f(v1.get()), _f(v2.get()))
            except ValueError:
                pass
            win.destroy()
            self.refresh()

        ttk.Button(frm, text="Apply", command=_apply).grid(
            row=2, column=0, columnspan=2, pady=(8, 0), sticky="ew")
        win.grab_set()

    def _export_master(self):
        if self.results_df is None or self.results_df.empty:
            messagebox.showinfo("Nothing to export", "Run an analysis first.",
                                parent=self)
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".csv", initialfile="analysis_results.csv",
            filetypes=[("CSV", "*.csv")])
        if not p:
            return
        self.results_df.to_csv(p, index=False)
        try:
            self._write_sidecar(p)
        except Exception:
            pass
        messagebox.showinfo("Exported",
                            f"{len(self.results_df)} rows → {p}",
                            parent=self)

    def _write_sidecar(self, csv_path):
        import json
        from datetime import datetime, timezone
        try:
            from io_utils import get_git_sha
            sha = get_git_sha()
        except Exception:
            sha = None
        cfg = self._build_cfg()
        meta = {
            "pixelpaws_module": "single_classifier_tab.py",
            "git_sha": sha,
            "exported_utc": datetime.now(timezone.utc).isoformat(),
            "rows": int(len(self.results_df)),
            "behaviors": sorted(self.results_df["Behavior"].unique()
                                .tolist()),
            "key_file": self.key_file_path,
            "prediction_files": [f.path for f in self._files_for_run()],
            "config": {
                "bin_size_min": cfg.bin_size_min,
                "whole_session": cfg.whole_session,
                "fps": cfg.fps,
                "analyze_mode": cfg.analyze_mode,
                "phases": [(ph.name, ph.start_min, ph.end_min)
                           for ph in cfg.phases],
                "filename_prefix": cfg.filename_prefix,
            },
        }
        with open(csv_path + ".meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def _export_fig(self):
        p = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("SVG", "*.svg"), ("PDF", "*.pdf")])
        if p:
            self._fig.savefig(p, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Exported", p, parent=self)
