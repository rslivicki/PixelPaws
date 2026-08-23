"""Locomotion tab - distance traveled and velocity from the pose skeleton.

Works directly from the DLC ``.h5`` files next to each project video, so it
needs no classifiers: pick a stable body point (centroid by default), gate it
by tracking likelihood, smooth it, and integrate frame-to-frame displacement
over time bins.

Not egocentric on purpose: egocentric normalization removes translation and
rotation - exactly the signal distance traveled is made of. Locomotion is an
allocentric quantity, so the tab tracks the animal's centroid through arena
coordinates and instead controls tracking jitter with a likelihood gate,
median smoothing, and a per-frame displacement dead-band.

Units come from the PawCapture spatial calibration embedded in each video
(``mm_per_pixel`` container tag): when every session is calibrated the tab
reports cm and cm/s; otherwise everything stays in pixels and the status line
names the uncalibrated sessions. All views scale to any number of groups.
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

_GROUP_COLS = ["#8D99AE", "#CC79A7", "#3b528b", "#21918c", "#f59f00", "#2f9e44"]
_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
_PREFERRED_BP = ("centroid", "tailbase", "neck", "snout")
_SKEL = "skeleton centroid"          # virtual: mean of core body points
# Trunk points only: snout head-sweeps are posture, not travel, and
# would inflate distance during licking/grooming.
_SKEL_PARTS = ("neck", "centroid", "tailbase")
_LIKELIHOOD_GATE = 0.6
_MAX_JUMP_PX = 100.0          # per-frame displacement above this = tracking teleport


def _group_color(i):
    return _GROUP_COLS[i % len(_GROUP_COLS)]


class LocomotionTab(ttk.Frame):

    def __init__(self, parent, main_gui):
        super().__init__(parent)
        self.app = main_gui
        self._sessions = {}        # {session: {'video':…, 'h5':…, 'fps':…, 'mm':…}}
        self._key_df = None
        self._traj_cache = {}      # {(session, bodypart, deadband): per-frame displacement}
        self._bodyparts = []
        self._build_ui()
        self._autofill()

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
        # Key file first, then Sessions - same order on every analysis tab.
        krow = ttk.Frame(df_)
        krow.pack(fill="x", padx=6, pady=(6, 2))
        ttk.Label(krow, text="Key file:").pack(side="left")
        self._key_status = tk.StringVar(value="none - sessions plotted ungrouped")
        ttk.Label(krow, textvariable=self._key_status, foreground="#666666",
                  justify="left", wraplength=170).pack(side="left", padx=(4, 4))
        _T(ttk.Button(krow, text="Browse", command=self._browse_key, width=7),
           "CSV/XLSX with Subject and Treatment columns; subjects match whole "
           "underscore tokens of the session name. Any number of groups is "
           "supported.").pack(side="right")

        # Sessions picker + Rescan on one row.
        sess_row = ttk.Frame(df_)
        sess_row.pack(fill="x", padx=6, pady=(4, 2))
        self._session_filter = SessionFilter(sess_row, on_change=self.refresh)
        self._session_filter.pack(side="left", fill="x", expand=True)
        _T(ttk.Button(sess_row, text="Rescan", command=self.scan_sessions,
                      width=7),
           "Find video + DLC .h5 pairs in the project's videos/ folder, read "
           "each video's fps and PawCapture spatial calibration, and list the "
           "tracked body parts.").pack(side="left", padx=(4, 0))
        self._scan_status = tk.StringVar(value="Not scanned.")
        ttk.Label(df_, textvariable=self._scan_status, foreground="#666666",
                  justify="left", wraplength=240).pack(anchor="w", padx=6,
                                                       pady=(0, 6))

        # -- settings ---------------------------------------------------
        st = ttk.LabelFrame(left, text="Tracking settings")
        st.pack(fill="x", pady=6)
        brow = ttk.Frame(st)
        brow.pack(fill="x", padx=6, pady=(6, 1))
        ttk.Label(brow, text="Body part:").pack(side="left")
        self._bp_var = tk.StringVar(value=_SKEL)
        self._bp_combo = ttk.Combobox(brow, textvariable=self._bp_var, width=12,
                                      state="readonly")
        self._bp_combo.pack(side="right")
        self._bp_combo.bind("<<ComboboxSelected>>", lambda e: self._recompute())
        Tip(self._bp_combo,
            "Body point whose trajectory is integrated. 'skeleton centroid' "
            "(default) averages the trunk points (neck, centroid, tailbase) "
            "per frame, which is more robust than any single tracked point. "
            "Snout is excluded on purpose: head sweeps during licking and "
            "grooming are posture, not travel.")
        drow = ttk.Frame(st)
        drow.pack(fill="x", padx=6, pady=1)
        ttk.Label(drow, text="Dead-band (px/frame):").pack(side="left")
        self._deadband_var = tk.DoubleVar(value=1.0)
        dsp = ttk.Spinbox(drow, textvariable=self._deadband_var, from_=0.0,
                          to=10.0, increment=0.25, width=6,
                          command=self._recompute)
        dsp.pack(side="right")
        dsp.bind("<Return>", lambda e: self._recompute())
        Tip(dsp, "Frame-to-frame displacements below this are counted as "
                 "tracking jitter and contribute zero distance. Raise it if a "
                 "still mouse accumulates distance; 0 disables the filter.")
        binrow = ttk.Frame(st)
        binrow.pack(fill="x", padx=6, pady=(1, 6))
        ttk.Label(binrow, text="Bin size (min):").pack(side="left")
        self._bin_var = tk.DoubleVar(value=5.0)
        bsp = ttk.Spinbox(binrow, textvariable=self._bin_var, from_=0.5, to=60,
                          increment=0.5, width=6, command=self.refresh)
        bsp.pack(side="right")
        bsp.bind("<Return>", lambda e: self.refresh())
        Tip(bsp, "Width of the time bins; clamps automatically so short "
                 "sessions still yield a timecourse.")
        _T(ttk.Button(st, text="Preview video…",
                      command=self._open_video_preview),
           "Play a session's video with the tracked trajectory overlaid -- "
           "sanity-check what the distance numbers are integrating.").pack(
            fill="x", padx=6, pady=(2, 6))

        # View selection lives in a dropdown ABOVE the graph (see the
        # right-side top bar below); only its backing var is created here.
        self._view = tk.StringVar(value="distance")

        # -- statistics -------------------------------------------------
        sf = ttk.LabelFrame(left, text="Statistics")
        sf.pack(fill="x", pady=6)
        self._stats_on = tk.BooleanVar(value=True)
        _T(ttk.Checkbutton(sf, text="Annotate group comparison",
                           variable=self._stats_on, command=self.refresh),
           "Test total distance (or mean velocity) across groups and print "
           "the p-value on the panel. 2 groups: Welch t / Mann-Whitney; 3+: "
           "ANOVA / Kruskal-Wallis.").pack(anchor="w", padx=6, pady=(4, 1))
        trow = ttk.Frame(sf)
        trow.pack(fill="x", padx=6, pady=(1, 6))
        ttk.Label(trow, text="Test:").pack(side="left")
        self._stats_test = tk.StringVar(value="auto")
        tcb = ttk.Combobox(trow, textvariable=self._stats_test, width=14,
                           state="readonly",
                           values=["auto", "parametric", "non-parametric"])
        tcb.pack(side="right")
        tcb.bind("<<ComboboxSelected>>", lambda e: self.refresh())
        Tip(tcb, "Auto picks non-parametric unless every group has n ≥ 8 and "
                 "passes Shapiro-Wilk.")

        erow = ttk.Frame(left)
        erow.pack(fill="x", pady=(8, 0))
        _T(ttk.Button(erow, text="Export figure…", command=self._export_fig),
           "Save the current figure as PNG, SVG, or PDF.").pack(
            side="left", fill="x", expand=True)
        _T(ttk.Button(erow, text="Export CSV…", command=self._export_csv),
           "Save the binned per-session distance/velocity table (long "
           "format).").pack(side="left", fill="x", expand=True, padx=(6, 0))
        self._status = tk.StringVar(value="")
        ttk.Label(left, textvariable=self._status, foreground="#444444",
                  justify="left", wraplength=240).pack(anchor="w", pady=(4, 0))

        self._VIEW_LABELS = [
            ("distance", "Distance per bin"),
            ("cumulative", "Cumulative distance"),
            ("velocity", "Velocity per bin"),
            ("trails", "Arena trails (per animal)"),
            ("grouptrails", "Arena trails (group overlay)"),
            ("representative", "Arena trails (representative)"),
            ("heatmap", "Arena occupancy heatmap"),
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
        Tip(_vcb, "Pick which graph to show. Timecourse views use the bin "
                  "size; arena views are drawn in normalized coordinates "
                  "(each video's frame mapped to the unit square).")
        _T(ttk.Button(topbar, text="🎨 ⚙",
                      command=self._edit_plot_options),
           "Plot style: group colors and colorways, error value (SEM/SD/"
           "95% CI) and display, line widths and per-group styles, markers, "
           "significance markers, individual-animal traces, grid, fonts, "
           "heatmap colormap. Saved with the project and shared by the "
           "analysis tabs.").pack(side="left", padx=(6, 0))
        # Heatmap colormap picker: appears only on the heatmap view.
        self._hm_bar = ttk.Frame(topbar)
        ttk.Label(self._hm_bar, text="Colormap:").pack(side="left",
                                                       padx=(10, 2))
        self._hm_cmap_var = tk.StringVar(value="viridis")
        _hmcb = ttk.Combobox(self._hm_bar, textvariable=self._hm_cmap_var,
                             state="readonly", width=9)
        _hmcb.pack(side="left")
        Tip(_hmcb, "Colormap for the occupancy heatmap; saved with the "
                   "project's plot style.")

        def _on_hm_pick(_evt=None):
            import plot_style
            o = plot_style.get_options(self._project())
            o["heatmap_cmap"] = self._hm_cmap_var.get()
            plot_style.save_options(self._project(), o)
            self.refresh()
        _hmcb.bind("<<ComboboxSelected>>", _on_hm_pick)
        self._hm_cmap_cb = _hmcb
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
        from ui_utils import neutralize_toolbar
        neutralize_toolbar(self._nav)
        from tkinter import scrolledtext as _stext
        self._stats_widget = _stext.ScrolledText(right, wrap="none",
                                                 font=("Consolas", 9))


    # ------------------------------------------------------------------ data

    def _project(self):
        import plot_style
        return plot_style.project_of(self.app)

    def _autofill(self):
        pass    # scanning is explicit or driven by on_project_changed

    def scan_sessions(self):
        self._sessions = {}
        self._traj_cache = {}
        vd = os.path.join(self._project(), "videos")
        if not os.path.isdir(vd):
            vd = self._project()
        if not os.path.isdir(vd):
            self._scan_status.set("Open a project first.")
            return
        try:
            from pawcapture_meta import read_calibration
        except Exception:
            read_calibration = None
        bps = set()
        for ext in _VIDEO_EXTS:
            for vp in sorted(_glob.glob(os.path.join(vd, f"*{ext}"))):
                stem = os.path.splitext(os.path.basename(vp))[0]
                if "DLC_" in stem or "_labeled" in stem:
                    continue
                h5s = _glob.glob(os.path.join(vd, f"{stem}DLC*.h5"))
                if not h5s:
                    continue
                fps, fw, fh = 60.0, None, None
                try:
                    import cv2
                    cap = cv2.VideoCapture(vp)
                    if cap.isOpened():
                        if cap.get(cv2.CAP_PROP_FPS) > 0:
                            fps = float(cap.get(cv2.CAP_PROP_FPS))
                        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
                        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
                    cap.release()
                except Exception:
                    pass
                mm = None
                if read_calibration is not None:
                    try:
                        c = read_calibration(vp)
                        if c and c.get("mm_per_pixel"):
                            mm = float(c["mm_per_pixel"])
                    except Exception:
                        pass
                self._sessions[stem] = {"video": vp, "h5": sorted(h5s)[0],
                                        "fps": fps, "mm": mm,
                                        "w": fw, "h": fh}
        # body parts from the first h5
        for info in self._sessions.values():
            try:
                cols = pd.read_hdf(info["h5"]).columns
                bps = {c[1] for c in cols}
            except Exception:
                bps = set()
            break
        self._bodyparts = ([_SKEL]
                           + [b for b in _PREFERRED_BP if b in bps]
                           + sorted(bps - set(_PREFERRED_BP)))
        if self._bodyparts:
            self._bp_combo.configure(values=self._bodyparts)
            if self._bp_var.get() not in self._bodyparts:
                self._bp_var.set(_SKEL)
        ncal = sum(1 for i in self._sessions.values() if i["mm"])
        n = len(self._sessions)
        if n:
            unit_note = ("(reporting cm)" if ncal == n else
                         "(reporting px - calibrate with PawCapture for "
                         "real units)")
            self._scan_status.set(f"{n} session(s); {ncal}/{n} calibrated "
                                  + unit_note)
        else:
            self._scan_status.set("No video + DLC .h5 pairs found - run pose "
                                  "tracking first.")
        self._session_filter.set_sessions(
            sorted(self._sessions), key_df=self._key_df,
            extra={s: {"Cal": ("✓" if i.get("mm") else "-")}
                   for s, i in self._sessions.items()})
        if self._key_df is None:
            self._autofind_key()
        self.refresh()

    def _unit(self):
        """('cm', factor_from_px) when every INCLUDED session is calibrated,
        else px (excluding an uncalibrated session restores cm)."""
        act = self._active_sessions() or sorted(self._sessions)
        infos = [self._sessions[k] for k in act]
        if infos and all(i["mm"] for i in infos):
            return "cm", None            # per-session factor applied individually
        return "px", None

    def _autofind_key(self):
        proj = self._project()
        kp = getattr(self.app, "key_file_path", None)
        if kp and os.path.isfile(kp):
            try:
                self._load_key(kp)
                return
            except Exception:
                pass
        for pat in ("*key*.csv", "*Key*.csv"):
            for cand in _glob.glob(os.path.join(proj, pat)):
                try:
                    self._load_key(cand)
                    return
                except Exception:
                    continue

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
        self._key_status.set(f"{os.path.basename(path)} - "
                             f"{self._key_df['Treatment'].nunique()} group(s)")
        if self._sessions:
            self._session_filter.set_sessions(
                sorted(self._sessions), key_df=self._key_df,
                extra={s: {"Cal": ("✓" if i.get("mm") else "-")}
                       for s, i in self._sessions.items()})

    def _group_of(self, session):
        if self._key_df is None:
            return None
        toks = set(session.split("_")) | {session}
        for _, r in self._key_df.iterrows():
            if r["Subject"] in toks:
                return r["Treatment"]
        # Multi-token subjects (e.g. "2605_FormOxy_S1"): match on
        # underscore boundaries so S1 can never claim S10's session.
        for _, r in self._key_df.iterrows():
            if f"_{r['Subject']}_" in f"_{session}_":
                return r["Treatment"]
        return None

    def _groups_present(self):
        if self._key_df is None:
            return []
        found = {self._group_of(s) for s in self._active_sessions()}
        return [g for g in self._key_df["Treatment"].unique() if g in found]

    # ------------------------------------------------------------------ compute

    def _trajectory(self, session):
        """Smoothed (x, y) of the chosen body part in px, likelihood-gated
        (p >= 0.6), gap-interpolated, 5-frame rolling-median filtered."""
        bp = self._bp_var.get()
        key = ("xy", session, bp)
        if key in self._traj_cache:
            return self._traj_cache[key]
        info = self._sessions[session]
        d = pd.read_hdf(info["h5"])
        d.columns = d.columns.droplevel(0)          # drop scorer level
        tracked = {c[0] for c in d.columns}
        if bp == _SKEL:
            # Virtual skeleton centroid: per-frame mean of the core trunk
            # points that clear the likelihood gate. Robust to any single
            # point dropping out; frames with no valid point interpolate.
            parts = [b for b in _SKEL_PARTS if b in tracked] or sorted(tracked)
            xs, ys = [], []
            for b in parts:
                bx = d[(b, "x")].astype(float)
                by = d[(b, "y")].astype(float)
                bp_ = d[(b, "likelihood")].astype(float)
                bx[bp_ < _LIKELIHOOD_GATE] = np.nan
                by[bp_ < _LIKELIHOOD_GATE] = np.nan
                # interpolate each point BEFORE averaging: a per-frame mean
                # over whichever points cleared the gate changes composition
                # when a point drops in or out, and those composition jumps
                # read as large fake displacements (verified 3-4x inflation).
                xs.append(bx.interpolate(limit_direction="both"))
                ys.append(by.interpolate(limit_direction="both"))
            x = pd.concat(xs, axis=1).mean(axis=1)
            y = pd.concat(ys, axis=1).mean(axis=1)
        else:
            if bp not in tracked:
                raise ValueError(f"{session}: body part '{bp}' not tracked")
            x = d[(bp, "x")].astype(float)
            y = d[(bp, "y")].astype(float)
            p = d[(bp, "likelihood")].astype(float)
            x[p < _LIKELIHOOD_GATE] = np.nan
            y[p < _LIKELIHOOD_GATE] = np.nan
        x = x.interpolate(limit_direction="both").rolling(5, center=True,
                                                          min_periods=1).median()
        y = y.interpolate(limit_direction="both").rolling(5, center=True,
                                                          min_periods=1).median()
        xy = (x.to_numpy(), y.to_numpy())
        self._traj_cache[key] = xy
        return xy

    def _displacement(self, session):
        """Per-frame displacement of the chosen body part, jitter-controlled:
        dead-band subtraction and a teleport cap on top of the smoothed
        trajectory. Units: px/frame."""
        dead = float(self._deadband_var.get())
        key = ("disp", session, self._bp_var.get(), dead)
        if key in self._traj_cache:
            return self._traj_cache[key]
        x, y = self._trajectory(session)
        disp = np.hypot(np.diff(x), np.diff(y))
        disp = np.nan_to_num(disp, nan=0.0)
        disp[disp < dead] = 0.0
        disp[disp > _MAX_JUMP_PX] = 0.0
        disp = np.r_[0.0, disp]
        self._traj_cache[key] = disp
        return disp

    def _binned(self):
        """{session: (bin_edges_min, per-bin distance in display units)},
        plus (bin_min, unit)."""
        unit, _ = self._unit()
        bin_min = max(float(self._bin_var.get()), 0.1)
        _act = self._active_sessions() or sorted(self._sessions)
        shortest = min(len(self._displacement(s)) / self._sessions[s]["fps"]
                       for s in _act) / 60.0
        if bin_min > shortest / 4 > 0:
            bin_min = max(round(shortest / 4, 2), 0.1)
        out = {}
        for s, info in self._sessions.items():
            if not self._session_filter.allows(s):
                continue
            disp = self._displacement(s)
            scale = (info["mm"] / 10.0) if (unit == "cm" and info["mm"]) else 1.0
            bin_frames = max(int(round(bin_min * 60 * info["fps"])), 1)
            nb = len(disp) // bin_frames
            if nb == 0:
                continue
            per_bin = (disp[:nb * bin_frames]
                       .reshape(nb, bin_frames).sum(axis=1) * scale)
            out[s] = per_bin
        return out, bin_min, unit

    # ------------------------------------------------------------------ stats

    def _group_test(self, samples):
        from scipy import stats as _st
        samples = [np.asarray(v, float) for v in samples if len(v) >= 2]
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
                    return float(_st.ttest_ind(*samples, equal_var=False)[1]), "Welch t"
                return float(_st.mannwhitneyu(
                    *samples, alternative="two-sided")[1]), "MW U"
            if parametric:
                return float(_st.f_oneway(*samples)[1]), "ANOVA"
            return float(_st.kruskal(*samples)[1]), "KW"
        except Exception:
            return None

    # ------------------------------------------------------------------ views

    def _active_sessions(self):
        """Session names after the optional per-session filter."""
        return [s for s in sorted(self._sessions)
                if self._session_filter.allows(s)]

    def refresh(self):
        if getattr(self, "_stats_mode", None) is not None \
                and self._stats_mode.get():
            self._stats_widget.configure(state="normal")
            self._stats_widget.delete("1.0", "end")
            try:
                txt = (self._stats_text() if self._sessions
                       else "Scan the project videos first.")
            except Exception as e:
                txt = f"Could not compute statistics: {e}"
            self._stats_widget.insert("1.0", txt)
            self._stats_widget.configure(state="disabled")
            return
        # colormap picker only makes sense on the heatmap view
        try:
            import plot_style
            if self._view.get() == "heatmap":
                cm = plot_style.get_options(self._project())["heatmap_cmap"]
                self._hm_cmap_cb.configure(values=plot_style.HEATMAP_CMAPS)
                if self._hm_cmap_var.get() != cm:
                    self._hm_cmap_var.set(cm)
                self._hm_bar.pack(side="left")
            else:
                self._hm_bar.pack_forget()
        except Exception:
            pass
        self._fig.clear()
        if not self._sessions:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, "Scan the project's videos first "
                              "(needs pose-tracked .h5 files).",
                    ha="center", va="center")
            ax.set_axis_off()
            self._canvas.draw_idle()
            return
        try:
            view = self._view.get()
            if view == "trails":
                self._draw_trails()
            elif view == "grouptrails":
                self._draw_grouptrails()
            elif view == "representative":
                self._draw_representative()
            elif view == "heatmap":
                self._draw_heatmap()
            else:
                ax = self._fig.add_subplot(111)
                self._draw(ax)
        except Exception as e:
            self._fig.clear()
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Could not draw: {e}", ha="center", va="center",
                    wrap=True)
            ax.set_axis_off()
        self._canvas.draw_idle()

    def _recompute(self):
        self._traj_cache = {}
        self.refresh()

    def _draw(self, ax):
        per_sess, bin_min, unit = self._binned()
        if not per_sess:
            raise ValueError("no session long enough for one bin")
        view = self._view.get()
        nb_min = min(len(v) for v in per_sess.values())
        tx = (np.arange(nb_min) + 0.5) * bin_min

        def series(v):
            v = v[:nb_min]
            if view == "cumulative":
                return np.cumsum(v)
            if view == "velocity":
                return v / (bin_min * 60.0)
            return v

        groups = self._groups_present()
        import plot_style
        cmap = plot_style.get_colors(self._project(), groups)
        opts = plot_style.get_options(self._project())
        if groups:
            for gi, g in enumerate(groups):
                col = cmap.get(g, _group_color(gi))
                rows = np.array([series(v) for s, v in per_sess.items()
                                 if self._group_of(s) == g])
                if not len(rows):
                    continue
                plot_style.draw_series(
                    ax, tx, rows, col, opts,
                    label=plot_style.group_label(g, len(rows), opts),
                    group=g)
            ax.legend(frameon=False, fontsize=8)
            if self._stats_on.get() and len(groups) >= 2:
                totals = [[per_sess[s][:nb_min].sum() for s in per_sess
                           if self._group_of(s) == g] for g in groups]
                res = self._group_test(totals)
                if res is not None:
                    pv, lab = res
                    mark = plot_style.sig_text(pv, 0.05,
                                               opts.get("sig_style",
                                                        "asterisk"))
                    if opts.get("sig_style") == "p-value" or not mark:
                        txt = f"total distance: {lab} p={pv:.3g}"
                    else:
                        txt = f"total distance: {lab} p={pv:.3g} {mark}"
                    ax.text(0.98, 0.02, txt,
                            transform=ax.transAxes, ha="right", va="bottom",
                            fontsize=8,
                            color="#b00020" if pv < 0.05 else "#666666")
        else:
            for s, v in per_sess.items():
                ax.plot(tx, series(v), lw=1.0, alpha=0.85, label=s)
            if len(per_sess) <= 10:
                ax.legend(frameon=False, fontsize=7)

        vel_unit = f"{unit}/s"
        titles = {"distance": (f"Distance traveled per {bin_min:g}-min bin",
                               f"Distance ({unit})"),
                  "cumulative": ("Cumulative distance traveled",
                                 f"Distance ({unit})"),
                  "velocity": (f"Mean velocity per {bin_min:g}-min bin",
                               f"Velocity ({vel_unit})")}
        ttl, ylab = titles[view]
        if opts.get("show_titles", True):
            ax.set_title(ttl + (f" - {self._bp_var.get()}"),
                         fontsize=10)
        ax.set_xlabel("Time (min)", fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        plot_style.apply_frame_options(ax, opts)
        _act = self._active_sessions()
        ncal = sum(1 for k in _act if self._sessions[k]["mm"])
        n = len(_act)
        note = "" if ncal == n else (
            f"  |  {n - ncal} uncalibrated session(s) - px units")
        _ex = len(self._sessions) - n
        self._status.set(f"{n} session(s)"
                         + (f" ({_ex} excluded)" if _ex else "")
                         + f"; {len(groups) or 'no'} group(s); "
                         f"{bin_min:g}-min bins; unit {unit}{note}")

    def _norm_xy(self, session):
        """Trajectory mapped into the unit arena: each video's frame becomes
        the [0,1] square (PawCapture crops to the enclosure, so frame bounds
        approximate arena bounds and per-subject crops become comparable).
        y is flipped so up on the plot = up in the video."""
        x, y = self._trajectory(session)
        info = self._sessions[session]
        w = info.get("w") or np.nanmax(x) or 1.0
        h = info.get("h") or np.nanmax(y) or 1.0
        return x / float(w), 1.0 - (y / float(h))

    def _draw_trails(self):
        sessions = self._active_sessions()
        groups = self._groups_present()
        import plot_style
        cmap = plot_style.get_colors(self._project(), groups)
        # order panels by group, then name
        if groups:
            order = [s_ for g in groups for s_ in sessions
                     if self._group_of(s_) == g]
            order += [s_ for s_ in sessions if s_ not in order]
        else:
            order = sessions
        n = len(order)
        ncols = 2 if n <= 4 else (3 if n <= 9 else 4)
        nrows = int(np.ceil(n / ncols))
        for i, s_ in enumerate(order):
            ax = self._fig.add_subplot(nrows, ncols, i + 1)
            nx, ny = self._norm_xy(s_)
            stride = max(1, len(nx) // 8000)
            g = self._group_of(s_)
            gi = groups.index(g) if g in groups else 0
            col = cmap.get(g, _group_color(gi)) if g else "#555555"
            ax.plot(nx[::stride], ny[::stride], lw=0.5, color=col, alpha=0.85)
            fin = np.isfinite(nx) & np.isfinite(ny)
            idx = np.flatnonzero(fin)
            if len(idx):
                ax.plot(nx[idx[0]], ny[idx[0]], marker="o", ms=4,
                        mfc="white", mec=col, mew=1.2)
                ax.plot(nx[idx[-1]], ny[idx[-1]], marker="s", ms=4,
                        color=col)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp_ in ax.spines.values():
                sp_.set_color("#bbbbbb")
            ttl = f"{s_}" + (f"  ({g})" if g else "")
            ax.set_title(ttl, fontsize=8, color=col if g else "#333333")
        self._fig.suptitle("Arena trails - normalized coordinates "
                           "(\u25cb start, \u25a0 end)", fontsize=10)
        self._status.set(f"{n} session(s); arena-normalized; "
                         f"{len(groups) or 'no'} group(s).")

    def _draw_grouptrails(self):
        """One panel per group with every member's trail overlaid."""
        sessions = self._active_sessions()
        groups = self._groups_present() or [None]
        import plot_style
        cmap = plot_style.get_colors(self._project(),
                                     [g for g in groups if g is not None])
        n = len(groups)
        _ncols = n if n <= 2 else (2 if n <= 4 else 3)
        _nrows = int(np.ceil(n / _ncols))
        for i, g in enumerate(groups):
            ax = self._fig.add_subplot(_nrows, _ncols, i + 1)
            members = [s_ for s_ in sessions
                       if g is None or self._group_of(s_) == g]
            col = cmap.get(g, "#555555") if g else "#555555"
            for s_ in members:
                nx, ny = self._norm_xy(s_)
                stride = max(1, len(nx) // 6000)
                ax.plot(nx[::stride], ny[::stride], lw=0.5, color=col,
                        alpha=0.45)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp_ in ax.spines.values():
                sp_.set_color("#bbbbbb")
            ax.set_title(f"{g} (n={len(members)})" if g
                         else f"all sessions (n={len(members)})",
                         fontsize=9, color=col)
        self._fig.suptitle("Arena trails - all animals overlaid per group, "
                           "normalized coordinates", fontsize=10)
        self._status.set(f"{len(sessions)} session(s); overlay per group.")

    def _draw_representative(self):
        """One panel per group: the animal whose total distance is closest
        to the group median (the standard 'representative example' pick)."""
        sessions = self._active_sessions()
        groups = self._groups_present() or [None]
        import plot_style
        cmap = plot_style.get_colors(self._project(),
                                     [g for g in groups if g is not None])
        n = len(groups)
        _ncols = n if n <= 2 else (2 if n <= 4 else 3)
        _nrows = int(np.ceil(n / _ncols))
        for i, g in enumerate(groups):
            ax = self._fig.add_subplot(_nrows, _ncols, i + 1)
            members = [s_ for s_ in sessions
                       if g is None or self._group_of(s_) == g]
            col = cmap.get(g, "#555555") if g else "#555555"
            if not members:
                ax.set_axis_off()
                continue
            totals = {s_: float(self._displacement(s_).sum())
                      for s_ in members}
            med = float(np.median(list(totals.values())))
            rep = min(totals, key=lambda s_: abs(totals[s_] - med))
            nx, ny = self._norm_xy(rep)
            stride = max(1, len(nx) // 8000)
            ax.plot(nx[::stride], ny[::stride], lw=0.7, color=col,
                    alpha=0.95)
            fin = np.isfinite(nx) & np.isfinite(ny)
            idx = np.flatnonzero(fin)
            if len(idx):
                ax.plot(nx[idx[0]], ny[idx[0]], marker="o", ms=4,
                        mfc="white", mec=col, mew=1.2)
                ax.plot(nx[idx[-1]], ny[idx[-1]], marker="s", ms=4,
                        color=col)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp_ in ax.spines.values():
                sp_.set_color("#bbbbbb")
            ttl = (f"{g}: {rep}" if g else rep)
            ax.set_title(ttl, fontsize=8, color=col)
        self._fig.suptitle("Representative trails - animal nearest the "
                           "group-median total distance", fontsize=10)
        self._status.set("Representative = closest to group-median "
                         "total distance.")

    def _draw_heatmap(self):
        sessions = self._active_sessions()
        groups = self._groups_present() or [None]
        BINS = 40
        maps = []
        for g in groups:
            members = [s_ for s_ in sessions
                       if g is None or self._group_of(s_) == g]
            hists = []
            for s_ in members:
                nx, ny = self._norm_xy(s_)
                fin = np.isfinite(nx) & np.isfinite(ny)
                if not fin.any():
                    continue
                hh, _, _ = np.histogram2d(nx[fin], ny[fin], bins=BINS,
                                          range=[[0, 1], [0, 1]])
                hists.append(hh / hh.sum() * 100)   # % time per bin
            maps.append((g, np.mean(hists, axis=0) if hists else None,
                         len(hists)))
        vmax = max((m.max() for _, m, _ in maps if m is not None),
                   default=1.0)
        _n = len(maps)
        ncols = _n if _n <= 2 else (2 if _n <= 4 else 3)
        _nrows = int(np.ceil(_n / ncols))
        for i, (g, m, nn) in enumerate(maps):
            ax = self._fig.add_subplot(_nrows, ncols, i + 1)
            if m is None:
                ax.set_axis_off()
                continue
            import plot_style as _ps
            _cm = _ps.get_options(self._project())["heatmap_cmap"]
            im = ax.imshow(m.T, origin="lower", extent=[0, 1, 0, 1],
                           cmap=_cm, vmin=0, vmax=vmax,
                           interpolation="bilinear")
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{g} (n={nn})" if g else f"all sessions (n={nn})",
                         fontsize=9)
        cbar = self._fig.colorbar(im, ax=self._fig.axes, shrink=0.75,
                                  pad=0.02)
        cbar.set_label("% time per bin", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        self._fig.suptitle("Arena occupancy - normalized position, "
                           "animal-weighted group mean", fontsize=10)
        self._status.set(f"{len(sessions)} session(s); {BINS}\u00d7{BINS} "
                         f"bins; shared color scale.")

    # ------------------------------------------------------------------ preview

    def _open_video_preview(self):
        """Video player with the smoothed trajectory overlaid: full path so
        far (faint), a recent trail (bright), and the current position."""
        if not self._sessions:
            messagebox.showinfo("Scan first", "Scan the project videos first.",
                                parent=self)
            return
        try:
            import cv2
            from PIL import Image, ImageTk
        except Exception as e:
            messagebox.showerror("Missing dependency",
                                 f"OpenCV + Pillow are required: {e}",
                                 parent=self)
            return

        win = tk.Toplevel(self)
        win.title("Locomotion preview")
        win.transient(self.winfo_toplevel())
        outer = ttk.Frame(win, padding=8)
        outer.pack(fill="both", expand=True)
        vid_lbl = ttk.Label(outer)
        vid_lbl.pack(side="left", padx=(0, 8))
        rail = ttk.Frame(outer, width=200)
        rail.pack(side="left", fill="y")

        ttk.Label(rail, text="Session:").pack(anchor="w")
        _prev_sessions = self._active_sessions() or sorted(self._sessions)
        sess_var = tk.StringVar(value=_prev_sessions[0])
        sess_cb = ttk.Combobox(rail, textvariable=sess_var, state="readonly",
                               values=_prev_sessions, width=20)
        sess_cb.pack(anchor="w", pady=(0, 6))

        ttk.Label(rail, text="Frame:").pack(anchor="w")
        frame_var = tk.IntVar(value=0)
        slider = ttk.Scale(rail, from_=0, to=100, orient="horizontal",
                           length=180)
        slider.pack(anchor="w")
        time_lbl = ttk.Label(rail, text="")
        time_lbl.pack(anchor="w", pady=(0, 6))

        ttk.Label(rail, text="Trail (s):").pack(anchor="w")
        trail_var = tk.DoubleVar(value=5.0)
        ttk.Spinbox(rail, textvariable=trail_var, from_=0.5, to=120,
                    increment=0.5, width=6).pack(anchor="w", pady=(0, 6))
        full_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(rail, text="Show full path so far",
                        variable=full_var).pack(anchor="w", pady=(0, 6))
        dist_lbl = ttk.Label(rail, text="", foreground="#3b528b",
                             font=("Segoe UI", 9, "bold"))
        dist_lbl.pack(anchor="w", pady=(0, 6))

        playing = [False]
        play_btn = ttk.Button(rail, text="\u25b6 Play")
        play_btn.pack(anchor="w", pady=(2, 0))
        ttk.Button(rail, text="Close", command=lambda: _close()).pack(
            anchor="w", pady=(10, 0))

        state = {"cap": None, "n": 0, "fps": 60.0, "mm": None,
                 "xy": None, "cum": None, "sess": None}

        def _load_session(*_a):
            sess = sess_var.get()
            if state["cap"] is not None:
                state["cap"].release()
            info = self._sessions[sess]
            cap = cv2.VideoCapture(info["video"])
            state.update(cap=cap,
                         n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                         fps=info["fps"] or 60.0, mm=info["mm"], sess=sess)
            x, y = self._trajectory(sess)
            state["xy"] = (x, y)
            state["cum"] = np.cumsum(self._displacement(sess))
            slider.configure(to=max(state["n"] - 1, 0))
            frame_var.set(0)
            _render(0)

        def _render(fi):
            cap = state["cap"]
            if cap is None:
                return
            fi = int(max(0, min(fi, state["n"] - 1)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            okf, frame = cap.read()
            if not okf:
                return
            x, y = state["xy"]
            m = min(len(x), state["n"])
            j = min(fi, m - 1)
            pts = np.stack([x[:j + 1], y[:j + 1]], axis=1)
            pts = pts[np.isfinite(pts).all(axis=1)].astype(np.int32)
            if full_var.get() and len(pts) > 1:
                cv2.polylines(frame, [pts], False, (180, 160, 120), 1,
                              cv2.LINE_AA)
            tr = int(float(trail_var.get()) * state["fps"])
            if len(pts) > 1 and tr > 1:
                cv2.polylines(frame, [pts[-tr:]], False, (167, 121, 204), 2,
                              cv2.LINE_AA)
            if np.isfinite(x[j]) and np.isfinite(y[j]):
                cv2.circle(frame, (int(x[j]), int(y[j])), 6, (139, 82, 59),
                           -1, cv2.LINE_AA)
                cv2.circle(frame, (int(x[j]), int(y[j])), 6, (255, 255, 255),
                           1, cv2.LINE_AA)
            h, w = frame.shape[:2]
            scale = min(640 / w, 640 / h, 1.0)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tkimg = ImageTk.PhotoImage(img)
            vid_lbl.configure(image=tkimg)
            vid_lbl.image = tkimg
            t = fi / state["fps"]
            time_lbl.configure(text=f"frame {fi}  ({int(t // 60)}:"
                                    f"{t % 60:04.1f})")
            cum = state["cum"]
            cj = cum[j] if j < len(cum) else cum[-1]
            if state["mm"]:
                dist_lbl.configure(
                    text=f"distance so far: {cj * state['mm'] / 10:.1f} cm")
            else:
                dist_lbl.configure(text=f"distance so far: {cj:.0f} px")

        def _on_slide(v):
            if not playing[0]:
                fi = int(float(v))
                frame_var.set(fi)
                _render(fi)

        def _tick():
            if not playing[0] or not win.winfo_exists():
                return
            fi = frame_var.get() + max(1, int(state["fps"] / 20))
            if fi >= state["n"]:
                _toggle_play()
                return
            frame_var.set(fi)
            slider.set(fi)
            _render(fi)
            win.after(50, _tick)

        def _toggle_play():
            playing[0] = not playing[0]
            play_btn.configure(text="\u23f8 Pause" if playing[0]
                               else "\u25b6 Play")
            if playing[0]:
                _tick()

        def _close():
            playing[0] = False
            if state["cap"] is not None:
                state["cap"].release()
            win.destroy()

        play_btn.configure(command=_toggle_play)
        slider.configure(command=_on_slide)
        sess_cb.bind("<<ComboboxSelected>>", _load_session)
        win.protocol("WM_DELETE_WINDOW", _close)
        trail_var.trace_add("write",
                            lambda *a: (not playing[0]
                                        and _render(frame_var.get())))
        full_var.trace_add("write",
                           lambda *a: (not playing[0]
                                       and _render(frame_var.get())))
        _load_session()

    # ------------------------------------------------------------------ export

    def _export_fig(self):
        if not self._sessions:
            messagebox.showinfo("Nothing to export", "Scan first.", parent=self)
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("SVG", "*.svg"), ("PDF", "*.pdf")])
        if p:
            self._fig.savefig(p, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Exported", p, parent=self)

    def _export_csv(self):
        if not self._sessions:
            messagebox.showinfo("Nothing to export", "Scan first.", parent=self)
            return
        if self._view.get() in ("trails", "grouptrails",
                                "representative", "heatmap"):
            # normalized trajectory, long format
            rows = []
            for s_ in self._active_sessions():
                nx, ny = self._norm_xy(s_)
                fps = self._sessions[s_]["fps"]
                grp = self._group_of(s_) or ""
                stride = max(1, int(round(fps / 10)))   # 10 Hz is plenty
                for fi in range(0, len(nx), stride):
                    rows.append({"session": s_, "group": grp,
                                 "time_s": fi / fps,
                                 "x_norm": nx[fi], "y_norm": ny[fi]})
            df = pd.DataFrame(rows)
            p = filedialog.asksaveasfilename(
                defaultextension=".csv",
                initialfile="arena_trajectories_norm.csv",
                filetypes=[("CSV", "*.csv")])
            if p:
                df.to_csv(p, index=False)
                messagebox.showinfo("Exported", f"{len(df)} rows (10 Hz) "
                                                f"\u2192 {p}", parent=self)
            return
        per_sess, bin_min, unit = self._binned()
        rows = []
        for s, v in per_sess.items():
            grp = self._group_of(s) or ""
            cum = 0.0
            for bi, dist in enumerate(v):
                cum += dist
                rows.append({"session": s, "group": grp,
                             "bin_start_min": bi * bin_min,
                             "bin_end_min": (bi + 1) * bin_min,
                             f"distance_{unit}": dist,
                             f"cumulative_{unit}": cum,
                             f"velocity_{unit}_per_s": dist / (bin_min * 60.0)})
        df = pd.DataFrame(rows)
        p = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=f"locomotion_{bin_min:g}min_bins.csv",
            filetypes=[("CSV", "*.csv")])
        if p:
            df.to_csv(p, index=False)
            messagebox.showinfo("Exported", f"{len(df)} rows → {p}",
                                parent=self)

    # ------------------------------------------------------------------ hooks

    def _edit_plot_options(self):
        import plot_style
        plot_style.open_options_dialog(self, self._project(),
                                       self._groups_present(),
                                       on_apply=self.refresh)

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
        sessions = self._active_sessions()
        groups = self._groups_present()
        per_sess, bin_min, unit = self._binned()
        tot = {s_: float(np.sum(v)) for s_, v in per_sess.items()}
        dur = {s_: len(self._displacement(s_)) / self._sessions[s_]["fps"]
               for s_ in per_sess}
        vel = {s_: tot[s_] / max(dur[s_], 1e-9) for s_ in per_sess}
        lines = []
        if groups:
            rows_by_g = {g: np.array(
                [[tot[s_], vel[s_]] for s_ in per_sess
                 if self._group_of(s_) == g]) for g in groups}
            lines.append(plot_style.stats_table(
                "Locomotion summary",
                [f"total distance ({unit})", f"mean velocity ({unit}/s)"],
                rows_by_g, opts, test_fn=self._group_test, unit=unit))
            lines.append("")
        lines.append("Per session")
        lines.append("-----------")
        for s_ in sorted(per_sess):
            g = self._group_of(s_) or ""
            lines.append(f"  {s_:<24} {g:<12} "
                         f"{tot[s_]:>9.1f} {unit}   "
                         f"{vel[s_]:>7.3f} {unit}/s")
        return "\n".join(lines)

    def on_project_changed(self):
        self._sessions = {}
        self._traj_cache = {}
        self._key_df = None
        self._session_filter.set_sessions([])
        self._session_filter._close_popup()
        self._scan_status.set("Not scanned.")
        self._key_status.set("none - sessions plotted ungrouped")
        self.refresh()
        vd = os.path.join(self._project(), "videos")
        if os.path.isdir(vd) and _glob.glob(os.path.join(vd, "*DLC*.h5")):
            self.after(300, self._auto_scan)

    def _auto_scan(self):
        try:
            if not self._sessions:
                self.scan_sessions()
        except Exception:
            pass
