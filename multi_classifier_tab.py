"""Multi-Classifier Analysis tab — cross-classifier views of a scored cohort.

Consumes the consolidated per-frame sheets Run Classifiers writes to
``<results>/per_frame/<session>_frames.csv`` (one ``<behavior>_prob`` and one
``<behavior>_pred`` column per classifier). Three views:

  1. Probability traces — per session, one panel per behavior: the classifier's
     frame-by-frame probability with predicted bouts shaded (the supplement's
     probability-plot format).
  2. State occupancy — per-frame states resolved with a priority order
     (top wins a disputed frame; unscored = no classifier fired), expressed as
     % of session time per state, mean ± SEM across animals per group.
  3. Group timecourse — an N-panel grid, one panel per behavior: % time in
     behavior per time bin, mean ± SEM lines, one line per group.

All views scale to any number of groups (color cycle over unique Treatment
values — e.g. the manuscript's 5-group formalin design), and work ungrouped
(each session its own trace) when no key file is available.
"""
from __future__ import annotations

import os
import glob as _glob
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)

from ui_tooltip import Tip
from ui_session_filter import SessionFilter

# Okabe-Ito-adjacent cycle, matches the sequencing tab; cycles past 6 groups.
_GROUP_COLS = ["#8D99AE", "#CC79A7", "#3b528b", "#21918c", "#f59f00", "#2f9e44"]

# Behavior line colors for the all-behavior probability view -- the
# manuscript's "bright" palette (fig_ethogram_lines): maximally separated
# hues, warm readouts, grey background states. Fuzzy-matched by name;
# unmatched behaviors take the fallback cycle.
_BEHAVIOR_COLS = {
    "jump": "#14213D", "lick": "#D62828", "scratch": "#F3722C",
    "rear": "#7209B7", "body_groom": "#1D4E89", "bodygroom": "#1D4E89",
    "facial": "#48CAE4", "face": "#48CAE4", "walk": "#43AA8B",
    "moving": "#43AA8B", "still": "#8D99AE", "idle": "#8D99AE",
}
_BEHAVIOR_FALLBACK = ["#264653", "#E76F51", "#2A9D8F", "#E9C46A", "#6A4C93"]


def _behavior_color(name, i=0):
    n = name.lower()
    for tok, c in _BEHAVIOR_COLS.items():
        if tok in n:
            return c
    return _BEHAVIOR_FALLBACK[i % len(_BEHAVIOR_FALLBACK)]

# Manuscript priority order, most specific first (same template as Sequencing).
_PRIORITY_TEMPLATE = ["lick", "scratch", "jump", "rear", "body_groom", "bodygroom",
                      "back_groom", "belly_groom", "facial", "face",
                      "walk", "moving", "locomot", "still", "idle"]


def _template_rank(name):
    n = name.lower()
    for i, tok in enumerate(_PRIORITY_TEMPLATE):
        if tok in n:
            return i
    return len(_PRIORITY_TEMPLATE) - 2.5


def _group_color(i):
    return _GROUP_COLS[i % len(_GROUP_COLS)]


class MultiClassifierTab(ttk.Frame):

    def __init__(self, parent, main_gui):
        super().__init__(parent)
        self.app = main_gui
        self._frames = {}          # {session: csv_path}
        self._behaviors = []       # priority order, highest first
        self._key_df = None        # Subject / Treatment
        self._fps = 60.0
        self._cache = {}           # {session: DataFrame} lazy-loaded sheets
        self._build_ui()
        self._autofill_results()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        def _T(widget, tip):
            Tip(widget, tip)
            return widget

        left_outer = ttk.Frame(self, width=340)
        left_outer.pack(side="left", fill="y", padx=(8, 4), pady=8)
        left_outer.pack_propagate(False)
        canvas = tk.Canvas(left_outer, highlightthickness=0, width=330)
        vsb = ttk.Scrollbar(left_outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        left = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=left, anchor="nw")
        left.bind("<Configure>",
                  lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True, padx=(4, 8), pady=8)

        # -- data -------------------------------------------------------
        df_ = ttk.LabelFrame(left, text="Data")
        df_.pack(fill="x", pady=(0, 6))
        # Data folder → Key file → Sessions: same order on every analysis tab.
        row = ttk.Frame(df_)
        row.pack(fill="x", padx=6, pady=(6, 2))
        ttk.Label(row, text="Results folder:").pack(side="left")
        self._results_var = tk.StringVar(value="")
        _T(ttk.Entry(row, textvariable=self._results_var),
           "Results folder written by Run Classifiers. The consolidated "
           "per-frame sheets in its per_frame/ subfolder feed this tab.").pack(
            side="left", fill="x", expand=True, padx=(4, 4))
        _T(ttk.Button(row, text="Browse", command=self._browse_results,
                      width=7),
           "Pick a results folder and scan it.").pack(side="left")

        krow = ttk.Frame(df_)
        krow.pack(fill="x", padx=6, pady=(4, 2))
        ttk.Label(krow, text="Key file:").pack(side="left")
        self._key_status = tk.StringVar(value="none — sessions plotted ungrouped")
        ttk.Label(krow, textvariable=self._key_status, foreground="#666666",
                  justify="left", wraplength=170).pack(side="left", padx=(4, 4))
        _T(ttk.Button(krow, text="Browse", command=self._browse_key, width=7),
           "CSV/XLSX with Subject and Treatment columns. Subjects are matched "
           "as whole underscore-separated tokens of the session name, so "
           "'mouse1' matches 'mouse1_veh'. Any number of groups is "
           "supported.").pack(side="right")

        # Sessions picker + Rescan on one row.
        sess_row = ttk.Frame(df_)
        sess_row.pack(fill="x", padx=6, pady=(4, 2))
        self._session_filter = SessionFilter(sess_row, on_change=self.refresh)
        self._session_filter.pack(side="left", fill="x", expand=True)
        _T(ttk.Button(sess_row, text="Rescan", command=self.scan_results,
                      width=7),
           "Find the per-session sheets, list the behaviors, and auto-locate "
           "a key file.").pack(side="left", padx=(4, 0))
        self._scan_status = tk.StringVar(value="Not scanned.")
        ttk.Label(df_, textvariable=self._scan_status, foreground="#666666",
                  justify="left", wraplength=240).pack(anchor="w", padx=6,
                                                       pady=(0, 6))

        # -- options (view selection lives in the dropdown above the graph)
        vf = ttk.LabelFrame(left, text="Options")
        vf.pack(fill="x", pady=6)
        self._view = tk.StringVar(value="traces")

        ttk.Label(vf, text="Behaviors:").pack(anchor="w", padx=6,
                                              pady=(4, 0))
        self._beh_list = tk.Listbox(vf, selectmode="extended", height=5,
                                    exportselection=False)
        self._beh_list.pack(fill="x", padx=6, pady=(0, 2))
        Tip(self._beh_list,
            "Which classifiers appear in the behavior-summary graph and the "
            "timecourse panels. Highlighted = included (Ctrl/Shift-click); "
            "all are included by default.")
        self._beh_list.bind("<<ListboxSelect>>", lambda e: self.refresh())

        srow = ttk.Frame(vf)
        srow.pack(fill="x", padx=6, pady=(2, 2))
        ttk.Label(srow, text="Session:").pack(side="left")
        self._session_var = tk.StringVar(value="")
        self._session_combo = ttk.Combobox(srow, textvariable=self._session_var,
                                           width=18, state="readonly")
        self._session_combo.pack(side="left", padx=(4, 0))
        self._session_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh())
        Tip(self._session_combo, "Session shown in the probability-traces view.")

        wrow = ttk.Frame(vf)
        wrow.pack(fill="x", padx=6, pady=(2, 2))
        ttk.Label(wrow, text="Rolling window (s):").pack(side="left")
        self._win_var = tk.DoubleVar(value=30.0)
        wsp = ttk.Spinbox(wrow, textvariable=self._win_var, from_=1, to=300,
                          increment=5, width=6, command=self.refresh)
        wsp.pack(side="right")
        wsp.bind("<Return>", lambda e: self.refresh())
        Tip(wsp, "Width of the rolling window that smooths the all-behavior "
                 "probability lines (the manuscript supplement uses 30 s).")

        brow = ttk.Frame(vf)
        brow.pack(fill="x", padx=6, pady=(2, 2))
        ttk.Label(brow, text="Bin size (min):").pack(side="left")
        self._bin_var = tk.DoubleVar(value=5.0)
        sp = ttk.Spinbox(brow, textvariable=self._bin_var, from_=0.5, to=60,
                         increment=0.5, width=6, command=self.refresh)
        sp.pack(side="right")
        sp.bind("<Return>", lambda e: self.refresh())
        Tip(sp, "Width of the time bins in the group-timecourse view.")

        _T(ttk.Button(vf, text="\u21bb Update graphs", command=self.update_graphs),
           "Re-read the prediction sheets (picks up a re-run of the "
           "classifiers) and redraw the current view. Priority order and "
           "session selection are kept.").pack(fill="x", padx=6, pady=(4, 6))

        # -- statistics -------------------------------------------------
        sf = ttk.LabelFrame(left, text="Statistics")
        sf.pack(fill="x", pady=6)
        self._stats_on = tk.BooleanVar(value=True)
        _T(ttk.Checkbutton(sf, text="Annotate group comparisons",
                           variable=self._stats_on, command=self.refresh),
           "Occupancy: test each state across groups and star significant "
           "ones. Timecourse: test total % time per behavior across groups "
           "and print the p-value on each panel.").pack(anchor="w", padx=6,
                                                        pady=(4, 1))
        trow = ttk.Frame(sf)
        trow.pack(fill="x", padx=6, pady=1)
        ttk.Label(trow, text="Test:").pack(side="left")
        self._stats_test = tk.StringVar(value="auto")
        tcb = ttk.Combobox(trow, textvariable=self._stats_test, width=14,
                           state="readonly",
                           values=["auto", "parametric", "non-parametric"])
        tcb.pack(side="right")
        tcb.bind("<<ComboboxSelected>>", lambda e: self.refresh())
        Tip(tcb, "2 groups: Welch t-test / Mann-Whitney U. 3+ groups: one-way "
                 "ANOVA / Kruskal-Wallis. Auto picks non-parametric unless "
                 "every group has n \u2265 8 and passes Shapiro-Wilk.")
        arow = ttk.Frame(sf)
        arow.pack(fill="x", padx=6, pady=(1, 6))
        ttk.Label(arow, text="Alpha:").pack(side="left")
        self._stats_alpha = tk.DoubleVar(value=0.05)
        asp = ttk.Spinbox(arow, textvariable=self._stats_alpha, from_=0.001,
                          to=0.2, increment=0.005, width=6,
                          command=self.refresh)
        asp.pack(side="right")
        asp.bind("<Return>", lambda e: self.refresh())
        Tip(asp, "Significance threshold for the annotations.")

        # -- priority (collapsible; click the header to show/hide) ------
        from ui_tooltip import collapsible
        pf = collapsible(left, "Priority (state occupancy)",
                         collapsed=True, fill="x", pady=6)
        self._prio_list = tk.Listbox(pf, height=7, exportselection=False)
        self._prio_list.pack(fill="x", padx=6, pady=(6, 1))
        Tip(self._prio_list,
            "When two classifiers mark the same frame, the behavior higher in "
            "this list wins the frame in the occupancy view. Seeded with the "
            "manuscript priority ranking.")
        pb = ttk.Frame(pf)
        pb.pack(fill="x", padx=6, pady=(0, 6))
        _T(ttk.Button(pb, text="▲", width=3, command=lambda: self._move_prio(-1)),
           "Move the selected behavior up in priority.").pack(side="left")
        _T(ttk.Button(pb, text="▼", width=3, command=lambda: self._move_prio(+1)),
           "Move the selected behavior down in priority.").pack(
            side="left", padx=(4, 0))

        erow = ttk.Frame(left)
        erow.pack(fill="x", pady=(8, 0))
        _T(ttk.Button(erow, text="Export figure…", command=self._export),
           "Save the current figure as PNG, SVG, or PDF.").pack(
            side="left", fill="x", expand=True)
        _T(ttk.Button(erow, text="Export CSV…", command=self._export_csv),
           "Save the data behind the current view as a CSV: traces = the "
           "session's per-frame probabilities and calls; occupancy = % time "
           "per state per session; timecourse = % time per bin per session "
           "(long format).").pack(side="left", fill="x", expand=True,
                                  padx=(6, 0))
        self._status = tk.StringVar(value="")
        ttk.Label(left, textvariable=self._status, foreground="#444444",
                  justify="left", wraplength=240).pack(anchor="w", pady=(4, 0))

        # -- canvas (view dropdown on top) ------------------------------
        self._VIEW_LABELS = [
            ("traces", "Probability traces (per session)"),
            ("summary", "Probability lines (all behaviors, per group)"),
            ("occupancy", "State occupancy (per group)"),
            ("timecourse", "Group timecourse (per behavior)"),
        ]
        topbar = ttk.Frame(right)
        topbar.pack(fill="x", pady=(0, 4))
        ttk.Label(topbar, text="Graph:",
                  font=("Segoe UI", 9, "bold")).pack(side="left")
        self._view_label_var = tk.StringVar(value=self._VIEW_LABELS[0][1])
        _vcb = ttk.Combobox(topbar, textvariable=self._view_label_var,
                            state="readonly", width=30,
                            values=[l for _, l in self._VIEW_LABELS])
        _vcb.pack(side="left", padx=(6, 0))

        def _on_view_pick(_evt=None):
            lab = self._view_label_var.get()
            for k, l in self._VIEW_LABELS:
                if l == lab:
                    self._view.set(k)
                    break
            self.refresh()
        _vcb.bind("<<ComboboxSelected>>", _on_view_pick)
        Tip(_vcb, "Pick which graph to show. Traces need a session; occupancy "
                  "uses the priority order; timecourse uses the bin size.")
        _T(ttk.Button(topbar, text="🎨 ⚙",
                      command=self._edit_plot_options),
           "Plot style: group colors, error bars, line and marker styles, "
           "significance markers, bar/box plots, grid and fonts. Saved with "
           "the project and shared by the analysis tabs.").pack(
            side="left", padx=(6, 0))
        self._stats_mode = tk.BooleanVar(value=False)
        _T(ttk.Checkbutton(topbar, text="Σ Stats",
                           variable=self._stats_mode, style="Toolbutton",
                           command=self._apply_stats_mode),
           "Flip between the graph and a statistics table for the current "
           "view (group descriptives + tests).").pack(side="left",
                                                      padx=(6, 0))

        self._fig = Figure(figsize=(9, 5.5), constrained_layout=True)
        self._canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        self._nav = NavigationToolbar2Tk(self._canvas, right)
        from tkinter import scrolledtext as _stext
        self._stats_widget = _stext.ScrolledText(right, wrap="none",
                                                 font=("Consolas", 9))


    # ------------------------------------------------------------------ data

    def _autofill_results(self):
        proj = getattr(self.app, "current_project_folder", None)
        if hasattr(proj, "get"):
            proj = proj.get()
        if proj:
            for cand in ("results", "Results", "classifier_results"):
                d = os.path.join(proj, cand)
                if os.path.isdir(d):
                    self._results_var.set(d)
                    return

    def _browse_results(self):
        d = filedialog.askdirectory(title="Results folder (with per_frame/)")
        if d:
            self._results_var.set(d)
            self.scan_results()

    def scan_results(self):
        folder = self._results_var.get()
        self._frames = {}
        self._behaviors = []
        self._cache = {}
        pf_dir = os.path.join(folder, "per_frame")
        pats = ([os.path.join(pf_dir, "*_frames.csv")] if os.path.isdir(pf_dir)
                else [os.path.join(folder, "**", "*_frames.csv")])
        hits = []
        for pat in pats:
            hits += _glob.glob(pat, recursive=True)
        beh = set()
        for h in sorted(hits):
            sess = os.path.basename(h)[:-len("_frames.csv")]
            self._frames[sess] = h
            try:
                cols = pd.read_csv(h, nrows=0).columns
            except Exception:
                continue
            beh |= {c[:-len("_pred")] for c in cols if c.endswith("_pred")}
        self._behaviors = sorted(beh, key=_template_rank)
        self._fps = self._infer_fps(folder)
        self._prio_list.delete(0, "end")
        for b in self._behaviors:
            self._prio_list.insert("end", b)
        self._beh_list.delete(0, "end")
        for b in self._behaviors:
            self._beh_list.insert("end", b)
        self._beh_list.select_set(0, "end")
        self._session_combo.configure(values=sorted(self._frames))
        self._session_filter.set_sessions(sorted(self._frames),
                                          key_df=self._key_df)
        if self._frames and not self._session_var.get():
            self._session_var.set(sorted(self._frames)[0])
        self._scan_status.set(
            f"{len(self._frames)} session(s), {len(self._behaviors)} "
            f"behavior(s), {self._fps:g} fps." if self._frames else
            "No per-frame sheets found — run classifiers first (the batch "
            "writes results/per_frame/).")
        if self._key_df is None:
            self._autofind_key(folder)
        self.refresh()

    def _infer_fps(self, folder):
        """fps from any timebins CSV (start/end frames + seconds); fallback 60."""
        for tb in _glob.glob(os.path.join(folder, "**", "*_timebins.csv"),
                             recursive=True)[:3]:
            try:
                d = pd.read_csv(tb, nrows=2)
                fr = float(d["end_frame"].iloc[0] - d["start_frame"].iloc[0])
                sec = float(d["end_time_sec"].iloc[0] - d["start_time_sec"].iloc[0])
                if sec > 0 and fr > 0:
                    return round(fr / sec, 3)
            except Exception:
                continue
        return 60.0

    def _autofind_key(self, folder):
        for pat in ("*key*.csv", "*key*.xlsx", "*Key*.csv", "*Key*.xlsx"):
            for cand in (_glob.glob(os.path.join(folder, pat))
                         + _glob.glob(os.path.join(os.path.dirname(folder), pat))):
                try:
                    self._load_key(cand)
                    return
                except Exception:
                    continue
        kp = getattr(self.app, "key_file_path", None)
        if kp and os.path.isfile(kp):
            try:
                self._load_key(kp)
            except Exception:
                pass

    def _browse_key(self):
        p = filedialog.askopenfilename(
            title="Key file (Subject, Treatment)",
            filetypes=[("Tables", "*.csv *.xlsx"), ("All files", "*.*")])
        if p:
            try:
                self._load_key(p)
                self.refresh()
            except Exception as e:
                messagebox.showerror("Key file", str(e), parent=self)

    def _load_key(self, path):
        df = pd.read_excel(path) if path.lower().endswith(".xlsx") else pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        if "subject" not in cols or "treatment" not in cols:
            raise ValueError("Key file needs Subject and Treatment columns.")
        out = pd.DataFrame({
            "Subject": df[cols["subject"]].astype(str).str.strip(),
            "Treatment": df[cols["treatment"]].astype(str).str.strip(),
        })
        self._key_df = out[out["Subject"] != ""]
        self._key_status.set(f"{os.path.basename(path)} — "
                             f"{self._key_df['Treatment'].nunique()} group(s)")
        if self._frames:
            self._session_filter.set_sessions(sorted(self._frames),
                                              key_df=self._key_df)

    def _agg_sessions(self):
        """Sessions feeding group aggregations, after the optional filter."""
        return [s for s in sorted(self._frames)
                if self._session_filter.allows(s)]

    def _group_of(self, session):
        """Whole-token subject match (mouse1 matches mouse1_veh)."""
        if self._key_df is None:
            return None
        toks = set(session.split("_")) | {session}
        for _, r in self._key_df.iterrows():
            if r["Subject"] in toks:
                return r["Treatment"]
        return None

    def _project(self):
        import plot_style
        return plot_style.project_of(self.app)

    def _color_map(self, groups):
        import plot_style
        return plot_style.get_colors(self._project(), groups)

    def _edit_colors(self):
        self._edit_plot_options()

    def _edit_plot_options(self):
        import plot_style
        plot_style.open_options_dialog(self, self._project(),
                                       self._groups_present(),
                                       on_apply=self.refresh)

    def _groups_present(self):
        """Ordered unique groups over scanned sessions (key order first)."""
        if self._key_df is None:
            return []
        found = {self._group_of(s) for s in self._frames}
        return [g for g in self._key_df["Treatment"].unique() if g in found]

    def _sheet(self, session):
        if session not in self._cache:
            self._cache[session] = pd.read_csv(self._frames[session])
        return self._cache[session]

    def _priority(self):
        return list(self._prio_list.get(0, "end"))

    def _selected_behaviors(self):
        """Behaviors highlighted in the selector (all when nothing/empty)."""
        try:
            sel = [self._beh_list.get(i) for i in self._beh_list.curselection()]
        except Exception:
            sel = []
        return sel or list(self._behaviors)

    def _move_prio(self, delta):
        sel = self._prio_list.curselection()
        if not sel:
            return
        i = sel[0]
        j = i + delta
        if not (0 <= j < self._prio_list.size()):
            return
        txt = self._prio_list.get(i)
        self._prio_list.delete(i)
        self._prio_list.insert(j, txt)
        self._prio_list.select_set(j)
        self.refresh()

    def update_graphs(self):
        """Re-read the sheets (new/changed predictions) and redraw, keeping
        the user's priority order and session choice."""
        self._cache = {}
        folder = self._results_var.get()
        if folder and os.path.isdir(folder):
            keep_prio = self._priority()
            keep_sess = self._session_var.get()
            self.scan_results()
            # scan_results reseeds priority from the template; restore the
            # user's order for behaviors that still exist, new ones append.
            cur = self._priority()
            merged = ([b for b in keep_prio if b in cur]
                      + [b for b in cur if b not in keep_prio])
            self._prio_list.delete(0, "end")
            for b in merged:
                self._prio_list.insert("end", b)
            if keep_sess in self._frames:
                self._session_var.set(keep_sess)
        self.refresh()

    # ------------------------------------------------------------------ stats

    @staticmethod
    def _stars(pval, alpha):
        if pval < 1e-4:
            return "****"
        if pval < 1e-3:
            return "***"
        if pval < 1e-2:
            return "**"
        if pval < alpha:
            return "*"
        return ""

    def _group_test(self, rows_by_group):
        """Compare 1-D samples across N groups. Returns (p, label) or None.

        2 groups: Welch t-test / Mann-Whitney U. 3+: one-way ANOVA /
        Kruskal-Wallis. 'auto' uses non-parametric unless every group has
        n >= 8 and passes Shapiro-Wilk.
        """
        from scipy import stats as _st
        samples = [np.asarray(v, float) for v in rows_by_group if len(v) >= 2]
        if len(samples) < 2:
            return None
        mode = self._stats_test.get()
        if mode == "auto":
            parametric = all(len(v) >= 8 for v in samples)
            if parametric:
                try:
                    parametric = all(_st.shapiro(v)[1] > 0.05 for v in samples)
                except Exception:
                    parametric = False
        else:
            parametric = (mode == "parametric")
        try:
            if len(samples) == 2:
                if parametric:
                    p = _st.ttest_ind(*samples, equal_var=False)[1]
                    lab = "Welch t"
                else:
                    p = _st.mannwhitneyu(*samples,
                                         alternative="two-sided")[1]
                    lab = "MW U"
            else:
                if parametric:
                    p = _st.f_oneway(*samples)[1]
                    lab = "ANOVA"
                else:
                    p = _st.kruskal(*samples)[1]
                    lab = "KW"
        except Exception:
            return None
        if not np.isfinite(p):
            return None
        return float(p), lab

    # ------------------------------------------------------------------ views

    def refresh(self):
        if getattr(self, "_stats_mode", None) is not None \
                and self._stats_mode.get():
            self._stats_widget.configure(state="normal")
            self._stats_widget.delete("1.0", "end")
            try:
                txt = (self._stats_text() if self._frames
                       else "Scan a results folder first.")
            except Exception as e:
                txt = f"Could not compute statistics: {e}"
            self._stats_widget.insert("1.0", txt)
            self._stats_widget.configure(state="disabled")
            return
        self._fig.clear()
        if not self._frames:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, "Scan a results folder with per_frame/ sheets.",
                    ha="center", va="center")
            ax.set_axis_off()
            self._canvas.draw_idle()
            return
        view = self._view.get()
        try:
            if view == "traces":
                self._draw_traces()
            elif view == "summary":
                self._draw_prob_lines()
            elif view == "occupancy":
                self._draw_occupancy()
            else:
                self._draw_timecourse()
        except Exception as e:
            self._fig.clear()
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Could not draw: {e}", ha="center", va="center",
                    wrap=True)
            ax.set_axis_off()
        self._canvas.draw_idle()

    @staticmethod
    def _panel_grid(n):
        ncols = 1 if n <= 1 else (2 if n <= 4 else 3)
        nrows = int(np.ceil(n / ncols))
        return nrows, ncols

    def _draw_traces(self):
        sess = self._session_var.get()
        if sess not in self._frames:
            raise ValueError("pick a session")
        d = self._sheet(sess)
        beh = [b for b in self._priority() if f"{b}_prob" in d.columns
               and b in self._selected_behaviors()]
        n = len(beh)
        nrows, ncols = self._panel_grid(n)
        t = d["frame"].to_numpy() / self._fps / 60.0
        stride = max(1, len(d) // 6000)
        for i, b in enumerate(beh):
            ax = self._fig.add_subplot(nrows, ncols, i + 1)
            prob = d[f"{b}_prob"].to_numpy()
            pred = d[f"{b}_pred"].to_numpy().astype(bool)
            ax.plot(t[::stride], prob[::stride], lw=0.6, color="#3b528b")
            # shade predicted bouts
            if pred.any():
                edges = np.flatnonzero(np.diff(np.r_[0, pred.view(np.int8), 0]))
                for s0, s1 in zip(edges[::2], edges[1::2]):
                    ax.axvspan(t[s0], t[min(s1, len(t) - 1)], color="#CC79A7",
                               alpha=0.35, lw=0)
            ax.set_ylim(0, 1.02)
            ax.set_title(b.replace("_", " "), fontsize=9)
            ax.tick_params(labelsize=7)
            if i // ncols == nrows - 1:
                ax.set_xlabel("Time (min)", fontsize=8)
            if i % ncols == 0:
                ax.set_ylabel("Probability", fontsize=8)
            for sp_ in ("top", "right"):
                ax.spines[sp_].set_visible(False)
        import plot_style as _ps_tr
        if _ps_tr.get_options(self._project()).get("show_titles", True):
            self._fig.suptitle(f"{sess} — classifier probabilities "
                               f"(shaded = predicted bouts)", fontsize=10)
        self._status.set(f"{sess}: {n} behaviors, {len(d)} frames.")

    def _states_for(self, session):
        """Priority-resolved per-frame state. 0 = unscored; 1..K per priority."""
        d = self._sheet(session)
        prio = [b for b in self._priority() if f"{b}_pred" in d.columns]
        state = np.zeros(len(d), np.int16)
        for idx in range(len(prio) - 1, -1, -1):
            v = d[f"{prio[idx]}_pred"].to_numpy()
            state[v == 1] = idx + 1
        return state, prio

    def _draw_prob_lines(self):
        """All selected behaviors overlaid on one axes per group: the
        group-mean classifier probability smoothed with a rolling window
        (default 30 s), colored per behavior -- the manuscript supplement's
        line-ethogram format."""
        import plot_style
        opts = plot_style.get_options(self._project())
        win_s = max(float(self._win_var.get()), 1.0)
        win_f = max(int(round(win_s * self._fps)), 1)
        sessions = self._agg_sessions()
        beh = [b for b in self._priority() if b in self._selected_behaviors()]
        if not beh:
            raise ValueError("select at least one behavior")
        groups = self._groups_present() or [None]

        # rolling-mean probability per session/behavior (cached per window)
        def _smoothed(s_, b):
            key = ("roll", s_, b, win_f)
            if key not in self._cache:
                d = self._sheet(s_)
                if f"{b}_prob" not in d.columns:
                    self._cache[key] = None
                else:
                    self._cache[key] = (d[f"{b}_prob"]
                                        .rolling(win_f, center=True,
                                                 min_periods=1)
                                        .mean().to_numpy())
            return self._cache[key]

        n = len(groups)
        first_ax = None
        for gi, g in enumerate(groups):
            ax = self._fig.add_subplot(1, n, gi + 1, sharey=first_ax)
            if first_ax is None:
                first_ax = ax
            members = [s_ for s_ in sessions
                       if g is None or self._group_of(s_) == g]
            for bi, b in enumerate(beh):
                traces = [t for t in (_smoothed(s_, b) for s_ in members)
                          if t is not None]
                if not traces:
                    continue
                m = min(len(t) for t in traces)
                mean = np.mean([t[:m] for t in traces], axis=0)
                stride = max(1, m // 2500)
                tx = np.arange(m)[::stride] / self._fps / 60.0
                ax.plot(tx, mean[::stride], color=_behavior_color(b, bi),
                        lw=float(opts.get("line_width", 1.6)) * 0.8,
                        label=(b.replace("_", " ") if gi == 0 else None))
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("Time (min)", fontsize=8)
            if gi == 0:
                ax.set_ylabel(f"Probability ({win_s:g} s rolling)",
                              fontsize=8)
            if opts.get("show_titles", True):
                ax.set_title(plot_style.group_label(g, len(members), opts)
                             if g is not None else
                             plot_style.group_label("all sessions",
                                                    len(members), opts),
                             fontsize=9)
            for sp_ in ("top", "right"):
                ax.spines[sp_].set_visible(False)
            plot_style.apply_frame_options(
                ax, opts, base=max(opts["font_size"] - 1, 6))
        if first_ax is not None:
            plot_style.fig_legend(self._fig,
                                  *first_ax.get_legend_handles_labels(),
                                  opts=opts,
                                  fontsize=max(opts["font_size"] - 2, 6))
        if opts.get("show_titles", True):
            self._fig.suptitle(
            f"Classifier probabilities — group mean, "
            f"{win_s:g} s rolling window", fontsize=10)
        self._status.set(f"{len(sessions)} session(s); {len(beh)} "
                         f"behavior(s); {win_s:g} s rolling window; colors "
                         "per behavior.")

    def _draw_occupancy(self):
        sessions = self._agg_sessions()
        prio = None
        occ = {}                                    # {session: fractions}
        for s in sessions:
            st, prio = self._states_for(s)
            k = len(prio)
            counts = np.bincount(st, minlength=k + 1)
            occ[s] = counts / max(len(st), 1)
        labels = ["unscored"] + prio
        groups = self._groups_present()
        ax = self._fig.add_subplot(111)
        x = np.arange(len(labels))
        import plot_style
        opts = plot_style.get_options(self._project())
        _rng = np.random.default_rng(0)             # seeded: stable jitter
        if groups:
            cmap = self._color_map(groups)
            width = 0.8 / len(groups)
            for gi, g in enumerate(groups):
                rows = np.array([occ[s] for s in sessions
                                 if self._group_of(s) == g]) * 100
                if not len(rows):
                    continue
                mean = rows.mean(axis=0)
                err = np.array([plot_style.calc_error(rows[:, j],
                                                      opts["error_type"])
                                for j in range(rows.shape[1])])
                xo = x - 0.4 + width * (gi + 0.5)
                col = cmap.get(g, _group_color(gi))
                edge_kw = (dict(edgecolor="black", linewidth=0.8)
                           if opts["bar_edges"] else {})
                if opts["bar_kind"] == "box":
                    bp = ax.boxplot([rows[:, j] for j in range(rows.shape[1])],
                                    positions=xo, widths=width * 0.85,
                                    patch_artist=True, showfliers=False,
                                    medianprops=dict(color="#222222"))
                    for box in bp["boxes"]:
                        box.set_facecolor(col)
                        box.set_alpha(0.85)
                    ax.bar([xo[0]], [0], color=col,
                           label=plot_style.group_label(g, len(rows),
                                                        opts))
                else:
                    yerr = (err if opts["error_display"] != "none" else None)
                    ax.bar(xo, mean, width=width * 0.9, yerr=yerr, capsize=2,
                           color=col,
                           label=plot_style.group_label(g, len(rows), opts),
                           error_kw=dict(lw=0.8), **edge_kw)
                if opts["show_individual"]:
                    j = opts["point_jitter"] * width
                    for row in rows:
                        xj = xo + (_rng.uniform(-j, j, len(xo)) if j else 0)
                        ax.plot(xj, row, ls="none", marker="o", ms=2.5,
                                mfc="white", mec=col, mew=0.6, alpha=0.8)
            ax.legend(frameon=False, fontsize=8)
            if self._stats_on.get() and len(groups) >= 2:
                alpha = float(self._stats_alpha.get())
                rows_by_g = {g: np.array([occ[s_] for s_ in sessions
                                          if self._group_of(s_) == g]) * 100
                             for g in groups}
                test_lab = None
                for si in range(len(labels)):
                    res = self._group_test(
                        [r[:, si] for r in rows_by_g.values() if len(r)])
                    if res is None:
                        continue
                    pv, test_lab = res
                    star = plot_style.sig_text(pv, alpha,
                                               opts.get("sig_style",
                                                        "asterisk"))
                    if opts.get("sig_style") == "p-value" and pv >= alpha:
                        star = ""
                    if star:
                        top = max(r[:, si].max() for r in rows_by_g.values()
                                  if len(r))
                        ax.text(x[si], top + 1.5, star, ha="center",
                                fontsize=(7 if opts.get("sig_style")
                                          == "p-value" else 10))
                if test_lab and opts.get("show_titles", True):
                    ax.set_title(f"State occupancy (priority-resolved) "
                                 f"\u2014 {test_lab}, \u03b1={alpha:g}",
                                 fontsize=10)
        else:
            rows = np.array([occ[s] for s in sessions]) * 100
            mean = rows.mean(axis=0)
            err = np.array([plot_style.calc_error(rows[:, j],
                                                  opts["error_type"])
                            for j in range(rows.shape[1])])
            yerr = err if opts["error_display"] != "none" else None
            ax.bar(x, mean, width=0.7, yerr=yerr, capsize=2, color="#8D99AE",
                   error_kw=dict(lw=0.8),
                   **(dict(edgecolor="black", linewidth=0.8)
                      if opts["bar_edges"] else {}))
        ax.set_xticks(x)
        ax.set_xticklabels([l.replace("_", " ") for l in labels],
                           rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("% of session time")
        if not ax.get_title() and opts.get("show_titles", True):
            ax.set_title("State occupancy (priority-resolved)", fontsize=10)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        plot_style.apply_frame_options(ax, opts)
        self._status.set(f"{len(sessions)} session(s); "
                         f"{len(groups) or 'no'} group(s).")

    def _draw_timecourse(self, kind="pred"):
        """Panel grid, one behavior per panel, group mean over time bins.
        kind="pred": % time in behavior (the classifier's thresholded calls).
        kind="prob": mean classifier probability (the supplement's
        probability-over-time panels)."""
        bin_min = max(float(self._bin_var.get()), 0.1)
        sessions = self._agg_sessions()
        # Clamp the bin so short sessions still yield a timecourse (>= 4 bins
        # for the shortest session, floor 0.1 min).
        shortest = min(len(self._sheet(s)) for s in sessions)
        max_bin_min = shortest / self._fps / 60.0 / 4.0
        clamped = bin_min > max_bin_min > 0
        if clamped:
            bin_min = max(round(max_bin_min, 2), 0.1)
        bin_frames = max(int(round(bin_min * 60 * self._fps)), 1)
        groups = self._groups_present()
        cmap = self._color_map(groups)
        import plot_style as _ps_tc
        opts = _ps_tc.get_options(self._project())
        beh = [b for b in self._priority()
               if b in self._selected_behaviors()]
        n = len(beh)
        nrows, ncols = self._panel_grid(n)
        first_ax = None
        for i, b in enumerate(beh):
            ax = self._fig.add_subplot(nrows, ncols, i + 1, sharex=first_ax)
            if first_ax is None:
                first_ax = ax
            per_sess = {}
            for s in sessions:
                d = self._sheet(s)
                if f"{b}_{kind}" not in d.columns:
                    continue
                v = d[f"{b}_{kind}"].to_numpy()
                nb = len(v) // bin_frames
                if nb == 0:
                    continue
                scale = 100.0 if kind == "pred" else 1.0
                per_sess[s] = (v[:nb * bin_frames]
                               .reshape(nb, bin_frames).mean(axis=1) * scale)
            if not per_sess:
                ax.set_axis_off()
                continue
            nb_min = min(len(v) for v in per_sess.values())
            tx = (np.arange(nb_min) + 0.5) * bin_min
            if groups:
                for gi, g in enumerate(groups):
                    rows = np.array([v[:nb_min] for s, v in per_sess.items()
                                     if self._group_of(s) == g])
                    if not len(rows):
                        continue
                    _col = cmap.get(g, _group_color(gi))
                    _ps_tc.draw_series(
                        ax, tx, rows, _col, opts,
                        label=(_ps_tc.group_label(g, len(rows), opts)
                               if i == 0 else None),
                        group=g)
            else:
                for s, v in per_sess.items():
                    ax.plot(tx, v[:nb_min], lw=0.9, alpha=0.8,
                            label=s if i == 0 else None)
            ax.set_title(b.replace("_", " "), fontsize=9)
            if groups and self._stats_on.get() and len(groups) >= 2:
                totals = {g: [per_sess[s_].mean() for s_ in per_sess
                              if self._group_of(s_) == g] for g in groups}
                res = self._group_test(list(totals.values()))
                if res is not None:
                    pv, lab = res
                    alpha = float(self._stats_alpha.get())
                    star = _ps_tc.sig_text(pv, alpha,
                                           opts.get("sig_style",
                                                        "asterisk"))
                    if opts.get("sig_style") == "p-value":
                        txt = f"{lab} {star}"
                    else:
                        txt = f"{lab} p={pv:.3g}" + (f" {star}" if star
                                                     else "")
                    ax.text(0.98, 0.95, txt,
                            transform=ax.transAxes, ha="right", va="top",
                            fontsize=7,
                            color="#b00020" if pv < alpha else "#666666")
            if i // ncols == nrows - 1:
                ax.set_xlabel("Time (min)", fontsize=8)
            if i % ncols == 0:
                ax.set_ylabel("% time" if kind == "pred"
                              else "Mean probability", fontsize=8)
            for sp_ in ("top", "right"):
                ax.spines[sp_].set_visible(False)
            _ps_tc.apply_frame_options(
                ax, opts, base=max(opts["font_size"] - 2, 6))
        if first_ax is not None:
            _ps_tc.fig_legend(self._fig,
                              *first_ax.get_legend_handles_labels(),
                              opts=opts,
                              fontsize=max(opts["font_size"] - 1, 6))
        _elab = ("mean" if opts["error_display"] == "none"
                 else _ps_tc.error_label(opts["error_type"]))
        _what = ("Group timecourse" if kind == "pred"
                 else "Classifier probability over time")
        if opts.get("show_titles", True):
            self._fig.suptitle(
            f"{_what} — {bin_min:g}-min bins, {_elab}"
            + (" (bin clamped for short sessions)" if clamped else ""),
            fontsize=10)
        self._status.set(f"{len(sessions)} session(s); "
                         f"{len(groups) or 'no'} group(s); "
                         f"{bin_min:g}-min bins.")

    # ------------------------------------------------------------------ misc

    def _export(self):
        if not self._frames:
            messagebox.showinfo("Nothing to export", "Scan and draw first.",
                                parent=self)
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("SVG", "*.svg"), ("PDF", "*.pdf")])
        if p:
            self._fig.savefig(p, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Exported", p, parent=self)

    def _view_dataframe(self):
        """The data behind the current view, as a tidy DataFrame."""
        view = self._view.get()
        if view == "summary":
            # rolling-mean probabilities at 1 Hz, long format
            win_s = max(float(self._win_var.get()), 1.0)
            win_f = max(int(round(win_s * self._fps)), 1)
            sessions = self._agg_sessions()
            beh = [b for b in self._priority()
                   if b in self._selected_behaviors()]
            step = max(int(round(self._fps)), 1)
            rows = []
            for s_ in sessions:
                d = self._sheet(s_)
                grp = self._group_of(s_) or ""
                for b in beh:
                    if f"{b}_prob" not in d.columns:
                        continue
                    r = (d[f"{b}_prob"].rolling(win_f, center=True,
                                                min_periods=1)
                         .mean().to_numpy())
                    for fi in range(0, len(r), step):
                        rows.append({"session": s_, "group": grp,
                                     "behavior": b,
                                     "time_min": fi / self._fps / 60.0,
                                     "probability": r[fi]})
            return (pd.DataFrame(rows),
                    f"probability_lines_{win_s:g}s_rolling")
        if view == "traces":
            sess = self._session_var.get()
            if sess not in self._frames:
                raise ValueError("pick a session first")
            d = self._sheet(sess).copy()
            d.insert(1, "time_min", d["frame"] / self._fps / 60.0)
            d.insert(0, "session", sess)
            return d, f"{sess}_probability_traces"
        sessions = self._agg_sessions()
        if view == "occupancy":
            rows = []
            for s_ in sessions:
                st, prio = self._states_for(s_)
                counts = np.bincount(st, minlength=len(prio) + 1)
                frac = counts / max(len(st), 1) * 100
                row = {"session": s_, "group": self._group_of(s_) or "",
                       "unscored_pct": frac[0]}
                for i, b in enumerate(prio):
                    row[f"{b}_pct"] = frac[i + 1]
                rows.append(row)
            return pd.DataFrame(rows), "state_occupancy"
        # timecourse (long format)
        bin_min = max(float(self._bin_var.get()), 0.1)
        shortest = min(len(self._sheet(s_)) for s_ in sessions)
        max_bin_min = shortest / self._fps / 60.0 / 4.0
        if bin_min > max_bin_min > 0:
            bin_min = max(round(max_bin_min, 2), 0.1)
        bin_frames = max(int(round(bin_min * 60 * self._fps)), 1)
        rows = []
        for s_ in sessions:
            d = self._sheet(s_)
            grp = self._group_of(s_) or ""
            for b in self._priority():
                if f"{b}_pred" not in d.columns:
                    continue
                v = d[f"{b}_pred"].to_numpy()
                nb = len(v) // bin_frames
                for bi in range(nb):
                    seg = v[bi * bin_frames:(bi + 1) * bin_frames]
                    rows.append({"session": s_, "group": grp, "behavior": b,
                                 "bin_start_min": bi * bin_min,
                                 "bin_end_min": (bi + 1) * bin_min,
                                 "percent_time": seg.mean() * 100})
        return pd.DataFrame(rows), f"group_timecourse_{bin_min:g}min_bins"

    def _export_csv(self):
        if not self._frames:
            messagebox.showinfo("Nothing to export", "Scan and draw first.",
                                parent=self)
            return
        try:
            df, stem = self._view_dataframe()
        except Exception as e:
            messagebox.showerror("Export CSV", str(e), parent=self)
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".csv", initialfile=f"{stem}.csv",
            filetypes=[("CSV", "*.csv")])
        if p:
            df.to_csv(p, index=False)
            messagebox.showinfo("Exported", f"{len(df)} rows → {p}",
                                parent=self)

    def _apply_stats_mode(self):
        """Swap the canvas and the stats table; refresh fills the visible one."""
        if self._stats_mode.get():
            self._canvas.get_tk_widget().pack_forget()
            self._nav.pack_forget()
            self._stats_widget.pack(fill="both", expand=True)
        else:
            self._stats_widget.pack_forget()
            self._canvas.get_tk_widget().pack(fill="both", expand=True)
            self._nav.pack(fill="x")
        self.refresh()

    def _stats_text(self):
        import plot_style
        opts = plot_style.get_options(self._project())
        sessions = self._agg_sessions()
        groups = self._groups_present()
        view = self._view.get()
        if not groups:
            return ("No key file loaded - statistics need Treatment "
                    "groups.")
        if view == "occupancy":
            occ = {}
            prio = None
            for s_ in sessions:
                st, prio = self._states_for(s_)
                counts = np.bincount(st, minlength=len(prio) + 1)
                occ[s_] = counts / max(len(st), 1) * 100
            labels = ["unscored"] + prio
            rows_by_g = {g: np.array([occ[s_] for s_ in sessions
                                      if self._group_of(s_) == g])
                         for g in groups}
            return plot_style.stats_table(
                "State occupancy (% of session time, priority-resolved)",
                labels, rows_by_g, opts, test_fn=self._group_test,
                unit="% of session time")
        # summary + timecourse + traces: per-behavior session values
        beh = [b for b in self._priority()
               if b in self._selected_behaviors()]
        if view == "summary":
            def val(d, b):
                return (d[f"{b}_prob"].mean()
                        if f"{b}_prob" in d.columns else np.nan)
            title = "Mean classifier probability (whole session)"
            unit = "mean probability"
        else:
            def val(d, b):
                return (d[f"{b}_pred"].mean() * 100
                        if f"{b}_pred" in d.columns else np.nan)
            title = "Time in behavior (% of session, per-classifier calls)"
            unit = "% of session time"
        per = {s_: np.array([val(self._sheet(s_), b) for b in beh])
               for s_ in sessions}
        rows_by_g = {g: np.array([per[s_] for s_ in sessions
                                  if self._group_of(s_) == g])
                     for g in groups}
        return plot_style.stats_table(title, beh, rows_by_g, opts,
                                      test_fn=self._group_test, unit=unit)

    def on_project_changed(self):
        self._frames = {}
        self._behaviors = []
        self._key_df = None
        self._cache = {}
        self._session_var.set("")
        self._session_combo.configure(values=[])
        self._prio_list.delete(0, "end")
        self._beh_list.delete(0, "end")
        self._scan_status.set("Not scanned.")
        self._key_status.set("none — sessions plotted ungrouped")
        self._autofill_results()
        self.refresh()
        rd = self._results_var.get()
        if rd and os.path.isdir(rd):
            self.after(250, self._auto_scan_existing)

    def _auto_scan_existing(self):
        try:
            if not self._frames and os.path.isdir(self._results_var.get()):
                self.scan_results()
        except Exception:
            pass
