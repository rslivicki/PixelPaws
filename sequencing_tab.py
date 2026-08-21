"""Sequencing tab — behavioural syntax with its OWN pipeline, end to end.

This tab does not inherit the Transitions tab's state assembly, deliberately. That path
re-thresholds every classifier's probability at 0.5 and lets the highest probability win,
which discards the per-classifier operating points and bout post-processing the classifiers
were validated at. The correct inputs already exist: the Run Classifiers tab writes
*_predictions.csv with a binary column that was thresholded at each classifier's own
operating point and passed through its own bout filter. This tab consumes those calls
directly and resolves overlaps by a user-ordered priority list — the same construction as
the manuscript's battery figures — with no smoothing, no downsampling and no re-threshold.

Pipeline:  results folder -> scan *_predictions.csv -> per-frame priority resolution
           -> key file (Subject / Treatment / optional Animal for paired designs)
           -> bout-level syntax analysis (pipeline/syntax_analysis.py):
              quasi-independence residuals, rarefied pooled networks, exact-enumeration
              PERMANOVA (Behrens-Fisher F2), PCoA.

"Pull from Transitions" remains available as a secondary source for unsupervised /
already-computed sequences, but the native pipeline is the primary path.
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

from pipeline import syntax_analysis as SA

_GROUP_COLS = ["#8D99AE", "#CC79A7", "#3b528b", "#21918c", "#f59f00", "#2f9e44"]
_MARKS = ["o", "s", "^", "D", "v", "P"]

# The manuscript's priority order, most specific first. Discovered behaviours are slotted
# by fuzzy name match; anything unmatched appends after the specific behaviours, before
# the two background states.
_PRIORITY_TEMPLATE = ["lick", "scratch", "jump", "rear", "body_groom", "bodygroom",
                      "back_groom", "belly_groom", "facial", "face",
                      "walk", "moving", "locomot", "still", "idle"]


def _template_rank(name):
    n = name.lower()
    for i, tok in enumerate(_PRIORITY_TEMPLATE):
        if tok in n:
            return i
    return len(_PRIORITY_TEMPLATE) - 2.5          # unknown: above walk/still


class SequencingTab(ttk.Frame):

    def __init__(self, parent, main_gui):
        super().__init__(parent)
        self.app = main_gui
        self._cohort = None
        self._groups = []
        self._files = {}          # {(session, behavior): csv_path}
        self._behaviors = []      # priority order, highest first
        self._key_df = None
        self._build_ui()
        self._autofill_results()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        from ui_tooltip import Tip

        def _T(widget, tip):
            Tip(widget, tip)
            return widget

        left_outer = ttk.Frame(self, width=280)
        left_outer.pack(side="left", fill="y", padx=(8, 4), pady=8)
        left_outer.pack_propagate(False)
        canvas = tk.Canvas(left_outer, highlightthickness=0, width=270)
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

        # -- pipeline ---------------------------------------------------
        pf = ttk.LabelFrame(left, text="Pipeline")
        pf.pack(fill="x", pady=(0, 6))
        row = ttk.Frame(pf)
        row.pack(fill="x", padx=6, pady=(6, 2))
        ttk.Label(row, text="Results folder:").pack(anchor="w")
        self._results_var = tk.StringVar(value="")
        _T(ttk.Entry(row, textvariable=self._results_var),
           "Folder of *_predictions.csv written by Run Classifiers. "
           "Per-behavior subfolders are scanned recursively.").pack(fill="x", pady=1)
        btns = ttk.Frame(pf)
        btns.pack(fill="x", padx=6, pady=2)
        _T(ttk.Button(btns, text="Browse…", command=self._browse_results, width=9),
           "Pick a results folder and scan it.").pack(side="left")
        _T(ttk.Button(btns, text="Scan", command=self.scan_results, width=9),
           "Find sessions and behaviors in the results folder, order the "
           "priority list, and auto-locate a key file.").pack(side="left", padx=(6, 0))
        self._scan_status = tk.StringVar(value="Not scanned.")
        ttk.Label(pf, textvariable=self._scan_status, foreground="#666666",
                  justify="left").pack(anchor="w", padx=6)

        ttk.Label(pf, text="Sessions:").pack(anchor="w", padx=6, pady=(4, 0))
        self._sess_list = tk.Listbox(pf, selectmode="extended", height=6,
                                     exportselection=False)
        self._sess_list.pack(fill="x", padx=6, pady=1)
        Tip(self._sess_list,
            "Sessions included in the cohort. Highlighted rows are included; "
            "Ctrl/Shift-click to exclude sessions before Compute.")

        from ui_tooltip import collapsible
        prio_body = collapsible(pf, "Priority (top wins a frame)",
                                collapsed=True, fill="x", padx=6, pady=(4, 2))
        self._prio_list = tk.Listbox(prio_body, height=7, exportselection=False)
        self._prio_list.pack(fill="x", padx=2, pady=1)
        Tip(self._prio_list,
            "When two classifiers mark the same frame, the behavior higher in "
            "this list wins. Seeded with the manuscript priority ranking; "
            "select a row and use the arrows to reorder, then re-Compute.")
        pb = ttk.Frame(prio_body)
        pb.pack(fill="x", padx=2, pady=(0, 4))
        _T(ttk.Button(pb, text="▲", width=3, command=lambda: self._move_prio(-1)),
           "Move the selected behavior up in priority.").pack(side="left")
        _T(ttk.Button(pb, text="▼", width=3, command=lambda: self._move_prio(+1)),
           "Move the selected behavior down in priority.").pack(side="left", padx=(4, 0))

        row = ttk.Frame(pf)
        row.pack(fill="x", padx=6, pady=(2, 2))
        ttk.Label(row, text="Key file:").pack(side="left")
        _T(ttk.Button(row, text="Browse…", command=self._browse_key, width=9),
           "CSV/XLSX with Subject and Treatment columns. Optional Animal "
           "(or Pair/Block) column marks rows sharing a mouse and switches "
           "on within-animal permutation.").pack(side="right")
        self._key_status = tk.StringVar(value="none — required for groups")
        ttk.Label(pf, textvariable=self._key_status, foreground="#666666",
                  justify="left", wraplength=240).pack(anchor="w", padx=6)
        ttk.Label(pf, foreground="#888888", wraplength=240, justify="left",
                  text="Key columns: Subject, Treatment, and optionally Animal "
                       "(or Pair/Block) naming which rows share a mouse — that "
                       "switches on within-animal permutation.").pack(
            anchor="w", padx=6, pady=(2, 2))

        # (Compute + status live at the BOTTOM of the rail, below the view/
        # threshold options; the Transitions import button was removed --
        # the native prediction-CSV pipeline is the only source.)

        # -- view options (selection lives in the dropdown above the graph)
        vf = ttk.LabelFrame(left, text="View options")
        vf.pack(fill="x", pady=6)
        self._view = tk.StringVar(value="networks")
        rf = ttk.Frame(vf)
        rf.pack(fill="x", padx=6, pady=(2, 6))
        ttk.Label(rf, text="Reference:").pack(side="left")
        self._ref = tk.StringVar(value="")
        self._ref_combo = ttk.Combobox(rf, textvariable=self._ref, width=14,
                                       state="readonly")
        self._ref_combo.pack(side="left", padx=(4, 0))
        Tip(self._ref_combo, "Group the others are compared against in the "
                             "difference view.")
        self._ref_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh())

        # -- ordination options ----------------------------------------
        of = ttk.LabelFrame(left, text="Ordination options")
        of.pack(fill="x", pady=6)
        self._centroids = tk.BooleanVar(value=True)
        self._dodge = tk.BooleanVar(value=True)
        self._stats = tk.BooleanVar(value=True)
        _ord_tips = {
            "Group centroids (+)": "Mark each group's centroid with a +.",
            "Dodge coincident points": "Spread exactly-overlapping animals "
                                       "slightly so every point stays visible.",
            "PERMANOVA annotation": "Print the PERMANOVA F, p, and R² on "
                                    "the ordination panel.",
        }
        for var, lab in ((self._centroids, "Group centroids (+)"),
                         (self._dodge, "Dodge coincident points"),
                         (self._stats, "PERMANOVA annotation")):
            _T(ttk.Checkbutton(of, text=lab, variable=var,
                               command=self.refresh),
               _ord_tips[lab]).pack(anchor="w", padx=6, pady=1)
        bf = ttk.Frame(of)
        bf.pack(fill="x", padx=6, pady=(2, 6))
        ttk.Label(bf, text="Boundary:").pack(side="left")
        self._boundary = tk.StringVar(value="None")
        cb = ttk.Combobox(bf, textvariable=self._boundary, width=12, state="readonly",
                          values=["None", "95% ellipse", "Convex hull"])
        cb.pack(side="left", padx=(4, 0))
        cb.bind("<<ComboboxSelected>>", lambda e: self.refresh())
        Tip(cb, "Draw a 95% confidence ellipse or convex hull around each "
                "group in the ordination.")

        # -- thresholds -------------------------------------------------
        tf = ttk.LabelFrame(left, text="Display thresholds")
        tf.pack(fill="x", pady=6)

        def spin(parent, label, var, frm, to, inc, tip=""):
            row = ttk.Frame(parent)
            row.pack(fill="x", padx=6, pady=1)
            lbl = ttk.Label(row, text=label)
            lbl.pack(side="left")
            sp = ttk.Spinbox(row, textvariable=var, from_=frm, to=to, increment=inc,
                             width=6, command=self.refresh)
            sp.pack(side="right")
            sp.bind("<Return>", lambda e: self.refresh())
            if tip:
                Tip(lbl, tip)
                Tip(sp, tip)

        self._min_bouts = tk.IntVar(value=SA.MIN_BOUTS)
        self._min_p = tk.DoubleVar(value=SA.MIN_P)
        self._min_dz = tk.DoubleVar(value=SA.MIN_DZ)
        self._min_node = tk.IntVar(value=SA.MIN_NODE)
        spin(tf, "Animal floor (bouts)", self._min_bouts, 10, 1000, 10,
             tip="Animals with fewer total bouts than this are dropped from "
                 "the cohort (prevents unstable transition estimates). "
                 "Changing it needs a re-Compute.")
        spin(tf, "Edge share of exits", self._min_p, 0.0, 0.5, 0.01,
             tip="Hide edges carrying less than this share of the exits from "
                 "their source behavior.")
        spin(tf, "Change threshold (units)", self._min_dz, 0.5, 20.0, 0.5,
             tip="Difference view: only show routes whose residual changed by "
                 "at least this many units versus the reference group.")
        spin(tf, "Node floor (exits)", self._min_node, 1, 200, 1,
             tip="Hide behaviors with fewer total exits than this.")
        ttk.Label(tf, text="Animal-floor changes need a re-compute.",
                  foreground="#888888").pack(anchor="w", padx=6, pady=(0, 4))

        _seq_erow = ttk.Frame(left)
        _seq_erow.pack(fill="x", pady=(8, 0))
        _T(ttk.Button(_seq_erow, text="Export figure…", command=self._export),
           "Save the current figure as PNG, SVG, or PDF.").pack(
            side="left", fill="x", expand=True)

        _T(ttk.Button(left, text="Compute", command=self.compute),
           "Assemble per-frame states with the priority order, extract bout "
           "sequences per animal, and run the syntax analysis (transition "
           "networks + PERMANOVA).").pack(fill="x", pady=(8, 0))
        self._status = tk.StringVar(value="No cohort yet.")
        ttk.Label(left, textvariable=self._status, foreground="#444444",
                  justify="left", wraplength=240).pack(anchor="w",
                                                       pady=(4, 6))

        # -- canvas (view dropdown on top) ------------------------------
        self._VIEW_LABELS = [
            ("networks", "Group networks"),
            ("difference", "Difference vs reference"),
            ("ordination", "Ordination (PCoA)"),
        ]
        topbar = ttk.Frame(right)
        topbar.pack(fill="x", pady=(0, 4))
        ttk.Label(topbar, text="Graph:",
                  font=("Segoe UI", 9, "bold")).pack(side="left")
        self._view_label_var = tk.StringVar(value=self._VIEW_LABELS[0][1])
        _vcb = ttk.Combobox(topbar, textvariable=self._view_label_var,
                            state="readonly", width=26,
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
        Tip(_vcb, "Networks: per-group transition networks (edge color = "
                  "quasi-independence residual). Difference: routes changed "
                  "vs the reference group. Ordination: PCoA + PERMANOVA.")
        _T(ttk.Button(topbar, text="🎨 ⚙",
                      command=self._edit_plot_options),
           "Plot style: group colors, marker size and alpha, line width, "
           "grid and fonts. Saved with the project and shared by the "
           "analysis tabs.").pack(side="left", padx=(6, 0))
        self._stats_mode = tk.BooleanVar(value=False)
        _T(ttk.Checkbutton(topbar, text="Σ Stats",
                           variable=self._stats_mode, style="Toolbutton",
                           command=self._apply_stats_mode),
           "Flip between the graph and a statistics table for the current "
           "view (group descriptives + tests).").pack(side="left",
                                                      padx=(6, 0))


        self._fig = Figure(figsize=(9, 5), constrained_layout=True)
        self._canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        self._nav = NavigationToolbar2Tk(self._canvas, right)
        from tkinter import scrolledtext as _stext
        self._stats_widget = _stext.ScrolledText(right, wrap="none",
                                                 font=("Consolas", 9))


    # ------------------------------------------------------------------ pipeline

    def _autofill_results(self):
        proj = getattr(self.app, "current_project_folder", None)
        if hasattr(proj, "get"):        # the GUI stores this as a tk.StringVar
            proj = proj.get()
        if proj:
            for cand in ("results", "Results", "classifier_results"):
                d = os.path.join(proj, cand)
                if os.path.isdir(d):
                    self._results_var.set(d)
                    return

    def _browse_results(self):
        d = filedialog.askdirectory(title="Results folder with *_predictions.csv")
        if d:
            self._results_var.set(d)
            self.scan_results()

    def scan_results(self):
        """Find *_predictions.csv, group by (session, behavior) from the CSV headers.

        The behaviour name is read from the CSV header (the column that is not frame /
        probability), which is authoritative; filename parsing is only used to derive
        the session stem.
        """
        folder = self._results_var.get()
        if not folder or not os.path.isdir(folder):
            self._scan_status.set("Folder not found.")
            return
        self._files = {}
        behaviors = set()
        for path in _glob.glob(os.path.join(folder, "**", "*_predictions.csv"),
                               recursive=True):
            try:
                head = pd.read_csv(path, nrows=0)
            except Exception:
                continue
            beh = next((c for c in head.columns
                        if c.lower() not in ("frame", "probability", "prob")), None)
            if beh is None:
                continue
            stem = os.path.basename(path)[:-len("_predictions.csv")]
            # strip the classifier token to get the session stem
            for tok in ("_classifier", "_PixelPaws"):
                if tok in stem:
                    stem = stem.split(tok)[0]
                    break
            else:
                if stem.lower().endswith("_" + beh.lower()):
                    stem = stem[:-(len(beh) + 1)]
            self._files[(stem, beh)] = path
            behaviors.add(beh)
        sessions = sorted({s for s, _b in self._files})
        self._behaviors = sorted(behaviors, key=_template_rank)
        self._sess_list.delete(0, "end")
        for s in sessions:
            self._sess_list.insert("end", s)
        self._sess_list.select_set(0, "end")
        self._prio_list.delete(0, "end")
        for b in self._behaviors:
            self._prio_list.insert("end", b)
        self._scan_status.set(f"{len(sessions)} session(s), "
                              f"{len(behaviors)} behaviour(s).")
        if self._key_df is None:
            self._autofind_key(folder)

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
        self._behaviors = list(self._prio_list.get(0, "end"))

    def _browse_key(self):
        path = filedialog.askopenfilename(
            title="Key file (Subject, Treatment[, Animal])",
            filetypes=[("Key files", "*.csv *.xlsx"), ("All", "*.*")])
        if path:
            self._load_key(path)

    def _autofind_key(self, folder):
        for pat in ("*key*.csv", "*key*.xlsx", "*Key*.csv", "*Key*.xlsx"):
            for cand in (_glob.glob(os.path.join(folder, pat))
                         + _glob.glob(os.path.join(os.path.dirname(folder), pat))):
                try:
                    self._load_key(cand)
                    return
                except Exception:
                    continue
        # Fall back to the project-level key the main GUI resolved
        # (config entry / legacy key_file.csv / project-wide discovery).
        kp = getattr(self.app, "key_file_path", None)
        if kp and os.path.isfile(kp):
            try:
                self._load_key(kp)
            except Exception:
                pass

    def _load_key(self, path):
        df = pd.read_excel(path) if path.lower().endswith(".xlsx") else pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        if "subject" not in cols or "treatment" not in cols:
            raise ValueError("Key file needs Subject and Treatment columns.")
        df = df.rename(columns={cols["subject"]: "Subject",
                                cols["treatment"]: "Treatment"})
        for opt in ("animal", "pair", "block"):
            if opt in cols:
                df = df.rename(columns={cols[opt]: "Animal"})
                break
        self._key_df = df
        paired = "Animal" in df.columns
        self._key_status.set(os.path.basename(path)
                             + (" (paired: Animal column)" if paired else ""))

    def _subject_of(self, session):
        """Match a session stem to a key-file Subject: exact, then substring."""
        if self._key_df is None:
            return None
        subs = [str(s) for s in self._key_df["Subject"]]
        if session in subs:
            return session
        hits = [s for s in subs if s and s in session]
        return max(hits, key=len) if hits else None

    def compute(self):
        """Assemble per-frame states from the filtered binary calls and build the cohort.

        Frame resolution: state 0 (unscored) unless some selected classifier's filtered
        call is 1; overlaps go to the highest-priority behaviour. The binary columns were
        produced at each classifier's own operating point with its own bout filter, so no
        thresholding, smoothing or downsampling happens here.
        """
        if not self._files:
            messagebox.showinfo("No data", "Scan a results folder first.")
            return
        if self._key_df is None:
            messagebox.showinfo("No key file",
                                "Load a key file (Subject, Treatment[, Animal]) —\n"
                                "sequencing compares groups.")
            return
        self._behaviors = list(self._prio_list.get(0, "end"))
        sessions = [self._sess_list.get(i) for i in self._sess_list.curselection()] \
            or list(self._sess_list.get(0, "end"))
        prio = self._behaviors                     # index 0 = highest priority
        seqs, group_of, strata_of = {}, {}, {}
        skipped = []
        for sess in sessions:
            subj = self._subject_of(sess)
            row = (self._key_df[self._key_df["Subject"].astype(str) == str(subj)]
                   if subj else None)
            if row is None or row.empty:
                skipped.append(sess)
                continue
            cols = {}
            n = 0
            for beh in prio:
                path = self._files.get((sess, beh))
                if path is None:
                    continue
                df = pd.read_csv(path)
                bcol = next((c for c in df.columns
                             if c.lower() not in ("frame", "probability", "prob")), None)
                v = df[bcol].to_numpy()
                v = (v > 0.5).astype(np.int8)      # binary already; guard against floats
                cols[beh] = v
                n = max(n, len(v))
            if not cols:
                skipped.append(sess)
                continue
            state = np.zeros(n, np.int8)
            for pi in range(len(prio) - 1, -1, -1):    # low priority first, high overwrites
                beh = prio[pi]
                if beh in cols:
                    v = cols[beh]
                    state[:len(v)][v == 1] = pi + 1
            seqs[sess] = state
            group_of[sess] = str(row.iloc[0]["Treatment"])
            strata_of[sess] = (str(row.iloc[0]["Animal"])
                               if "Animal" in self._key_df.columns else sess)
        if not seqs:
            self._status.set("No session matched the key file.")
            return
        keep_idx = {i + 1: i for i in range(len(prio))}
        rep = len(set(strata_of.values())) < len(seqs)
        self._cohort = SA.SyntaxCohort(seqs, group_of, keep_idx, prio,
                                       min_bouts=int(self._min_bouts.get()),
                                       strata_of=strata_of if rep else None)
        if not self._cohort.sessions:
            self._cohort = None
            self._status.set(f"Every session fell under the "
                             f"{int(self._min_bouts.get())}-bout floor.")
            return
        self._groups = list(dict.fromkeys(
            [group_of[s] for s in self._cohort.sessions]))
        self._ref_combo["values"] = self._groups
        if self._ref.get() not in self._groups:
            self._ref.set(self._groups[0])
        c = self._cohort
        self._status.set(f"{len(c.sessions)} session(s), {len(self._groups)} group(s)\n"
                         f"subsampled to {c.n_sub} transitions"
                         + (f"\nexcluded: {len(c.dropped)} under bout floor"
                            if c.dropped else "")
                         + (f"\nno key match: {len(skipped)}" if skipped else "")
                         + ("\npaired (within-animal)" if c.strata is not None else ""))
        self.refresh()

    # -------------------------------------------------- secondary source

    def pull_from_transitions(self):
        tt = getattr(self.app, "transitions_tab", None)
        seqs = getattr(tt, "_state_seqs", None) if tt else None
        if not seqs:
            messagebox.showinfo("No data",
                                "The Transitions tab has no computed sequences.")
            return
        group_of = {n: tt._session_group(n) for n in seqs}
        if not any(group_of.values()):
            messagebox.showinfo("No groups",
                                "Load a key file on the Transitions tab first.")
            return
        named = [s for s in tt._states if s != 0]
        keep_idx = {sid: i for i, sid in enumerate(named)}
        labels = [tt._state_name(s) for s in named]
        subs = getattr(tt, "_session_subjects", None) or {}
        strata_of = {n: subs.get(n, n) for n in seqs}
        kdf = getattr(tt, "_key_df", None)
        if kdf is not None:
            pcol = next((c for c in ("Animal", "Pair", "Block") if c in kdf.columns), None)
            if pcol:
                m = {str(r["Subject"]): str(r[pcol]) for _, r in kdf.iterrows()}
                strata_of = {n: m.get(str(subs.get(n, n)), subs.get(n, n)) for n in seqs}
        rep = len(set(strata_of.values())) < len([n for n in seqs if group_of.get(n)])
        self._cohort = SA.SyntaxCohort(seqs, group_of, keep_idx, labels,
                                       min_bouts=int(self._min_bouts.get()),
                                       strata_of=strata_of if rep else None)
        if not self._cohort.sessions:
            self._cohort = None
            self._status.set("Every pulled session fell under the bout floor.")
            return
        order = getattr(tt, "_ordered_groups", None)
        self._groups = order(self._cohort.groups) if order else list(self._cohort.groups)
        self._ref_combo["values"] = self._groups
        if self._ref.get() not in self._groups:
            self._ref.set(self._groups[0])
        c = self._cohort
        self._status.set(f"[from Transitions] {len(c.sessions)} session(s), "
                         f"{len(self._groups)} group(s)"
                         + ("\npaired (within-animal)" if c.strata is not None else ""))
        self.refresh()

    # ------------------------------------------------------------------ drawing

    def _gcol(self, gi):
        try:
            import plot_style
            proj = getattr(self.app, "current_project_folder", None)
            if hasattr(proj, "get"):
                proj = proj.get()
            if gi < len(self._groups):
                g = self._groups[gi]
                # pass the FULL ordered list so positional defaults line up
                return plot_style.get_colors(proj or "", list(self._groups)).get(
                    g, _GROUP_COLS[gi % len(_GROUP_COLS)])
        except Exception:
            pass
        return _GROUP_COLS[gi % len(_GROUP_COLS)]

    def _edit_colors(self):
        self._edit_plot_options()

    def _edit_plot_options(self):
        import plot_style
        plot_style.open_options_dialog(
            self, plot_style.project_of(self.app), list(self._groups),
            on_apply=self.refresh)

    def refresh(self):
        if getattr(self, "_stats_mode", None) is not None \
                and self._stats_mode.get():
            self._stats_widget.configure(state="normal")
            self._stats_widget.delete("1.0", "end")
            try:
                txt = self._stats_text()
            except Exception as e:
                txt = f"Could not compute statistics: {e}"
            self._stats_widget.insert("1.0", txt)
            self._stats_widget.configure(state="disabled")
            return
        self._fig.clf()
        if self._cohort is None:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, "Scan a results folder, load a key file,\nthen Compute.",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            self._canvas.draw_idle()
            return
        {"networks": self._draw_networks,
         "difference": self._draw_difference,
         "ordination": self._draw_ordination}[self._view.get()]()
        self._canvas.draw_idle()

    def _draw_networks(self):
        c, groups = self._cohort, self._groups
        axes = self._fig.subplots(1, len(groups), squeeze=False)[0]
        for gi, g in enumerate(groups):
            col = self._gcol(gi)
            SA.draw_network(axes[gi], c.pooled[g], c.labels,
                            f"{g}\n(rarefied to {int(c.pooled[g].sum())} transitions)",
                            col, SA.mono_cmap(col),
                            min_p=float(self._min_p.get()),
                            min_node=int(self._min_node.get()))

    def _draw_difference(self):
        c, groups = self._cohort, self._groups
        ref = self._ref.get() or groups[0]
        others = [g for g in groups if g != ref]
        if not others:
            self._fig.add_subplot(111).text(0.5, 0.5, "Need two or more groups.",
                                            ha="center", va="center")
            return
        cmap = SA.div_cmap()
        axes = self._fig.subplots(1, len(others), squeeze=False)[0]
        for gi, g in enumerate(others):
            vmax, n_ch, n_el = SA.draw_diff(
                axes[gi], c.pooled[ref], c.pooled[g], c.labels, cmap,
                min_dz=float(self._min_dz.get()),
                min_p=float(self._min_p.get()), min_node=int(self._min_node.get()))
            axes[gi].set_title(
                f"{g} vs {ref}\n{n_ch} of {n_el} routes changed "
                f"> {float(self._min_dz.get()):g} residual units", fontsize=9, pad=4)

    def _draw_ordination(self):
        import plot_style
        _seq_opts = plot_style.get_options(plot_style.project_of(self.app))
        c, groups = self._cohort, self._groups
        xy, pcvar, _ = c.ordination()
        pts = SA.dodge(xy) if self._dodge.get() else xy
        ax = self._fig.add_subplot(111)
        for gi, g in enumerate(groups):
            m = c.groups_per_session == g
            col = self._gcol(gi)
            if self._boundary.get() == "95% ellipse":
                SA.ellipse(ax, xy[m], col)
            elif self._boundary.get() == "Convex hull":
                SA.hull(ax, xy[m], col)
            # marker_size is in pt; scatter s is pt^2 (4.0 -> ~46, the old
            # hardcoded look); line_width 1.6 -> 1.28 ~ the old 1.3
            _o = _seq_opts
            ax.scatter(pts[m, 0], pts[m, 1],
                       s=(float(_o["marker_size"]) * 1.7) ** 2,
                       marker=_MARKS[gi % len(_MARKS)],
                       facecolors="none", edgecolors=col,
                       linewidth=max(float(_o["line_width"]) * 0.8, 0.5),
                       alpha=float(_o["marker_alpha"]))
            if self._centroids.get() and m.sum() > 1:
                ax.scatter(*xy[m].mean(0), s=150, marker="+", color=col,
                           linewidth=1.8, zorder=4)
        if self._stats.get():
            try:
                F2, p, R2 = c.test()
                paired = c.strata is not None
                ax.set_title(f"p = {p:.4g}, R² = {R2:.2f}  (PERMANOVA"
                             + (", within-animal permutation)" if paired else ")"),
                             fontsize=10, pad=6)
            except Exception as ex:
                ax.set_title(f"PERMANOVA unavailable: {ex}", fontsize=9)
        ax.set_xlabel(f"PC1 ({pcvar[0]:.0f}%)")
        ax.set_ylabel(f"PC2 ({pcvar[1]:.0f}%)")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#DDDDDD")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(frameon=False, fontsize=8, loc="best")

    # ------------------------------------------------------------------ misc

    def _export(self):
        if self._cohort is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("SVG", "*.svg"), ("PDF", "*.pdf")])
        if path:
            self._fig.savefig(path, dpi=300, bbox_inches="tight")

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
        if self._cohort is None:
            return "Compute a cohort first."
        c = self._cohort
        lines = ["Sequencing cohort", "=================", ""]
        try:
            F2, p, R2 = c.test()
            lines.append(f"PERMANOVA (Behrens-Fisher F2): "
                         f"F2={F2:.3g}  p={p:.4g}  R\u00b2={R2:.3g}")
        except Exception as e:
            lines.append(f"PERMANOVA unavailable: {e}")
        lines.append(f"groups: {', '.join(map(str, self._groups))}")
        try:
            import numpy as _np
            for g in self._groups:
                subs = [s for s, gg in c.group_of.items() if gg == g]
                bouts = [len(c.seqs[s]) for s in subs if s in c.seqs]
                if bouts:
                    lines.append(f"  {g}: {len(subs)} animal(s), "
                                 f"median bouts {int(_np.median(bouts))} "
                                 f"(range {min(bouts)}-{max(bouts)})")
        except Exception:
            pass
        lines.append("")
        lines.append(f"status: {self._status.get()}")
        return "\n".join(lines)

    def on_project_changed(self):
        self._cohort = None
        self._groups = []
        self._files = {}
        self._key_df = None
        self._sess_list.delete(0, "end")
        self._prio_list.delete(0, "end")
        self._key_status.set("none — required for groups")
        self._scan_status.set("Not scanned.")
        self._status.set("No cohort yet.")
        self._autofill_results()
        self.refresh()
        # Auto-scan when the project already has results so the tab arrives
        # populated (sessions selected, priority ordered, key auto-found).
        # Deferred so project switching stays snappy.
        rd = self._results_var.get()
        if rd and os.path.isdir(rd):
            self.after(200, self._auto_scan_existing)

    def _auto_scan_existing(self):
        try:
            if not self._files and os.path.isdir(self._results_var.get()):
                self.scan_results()
        except Exception:
            pass
