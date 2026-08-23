# -*- coding: utf-8 -*-
"""Quick Start tab - the guided videos -> data wizard.

Walks a naive user from raw videos to populated analysis tabs in one pass:

    Transcode -> Pose tracking -> Feature extraction -> Classifiers (Core 8)
    -> Gait & contour

The wizard drives the existing machinery (DLCProgressDialog for
transcode+pose, the Run Classifiers batch, GaitLimbTabV2.auto_run) and
mirrors their progress into ONE bar whose color changes per stage
(OC*.Horizontal.TProgressbar styles defined in PixelPaws_GUI.apply_theme).
The user stays on this tab for the whole run; the usual end-of-batch popups
and tab jumps are suppressed while the pipeline is active.
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from ui_tooltip import Tip
from ui_utils import FONT_FAMILY

# stage key -> (row label, progressbar style prefix)
STAGES = [
    ("transcode",   "Transcode videos (intake H.265)",       "OCtranscode"),
    ("pose",        "Pose tracking (DLC)",                   "OCpose"),
    ("features",    "Feature extraction",                    "OCfeatures"),
    ("classifiers", "Classifiers (Core 8)",                  "OCclassifiers"),
    ("gait",        "Gait & contour analysis",               "OCgait"),
]
_DOTS = {"pending": ("○", "#999999"),   # open circle
         "running": ("●", None),        # filled, stage color
         "done":    ("✓", "#2f9e44"),
         "failed":  ("✗", "#c92a2a"),
         "skipped": ("↯", "#999999")}
_STAGE_COLORS = {"OCtranscode": "#f59f00", "OCpose": "#4582ec",
                 "OCfeatures": "#20c997", "OCclassifiers": "#6f42c1",
                 "OCgait": "#2f9e44"}


def _fmt_eta(sec):
    """Compact remaining-time text."""
    if sec is None or sec != sec or sec < 0:
        return ""
    sec = int(round(sec))
    if sec >= 3600:
        return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"
    if sec >= 60:
        return f"{sec // 60}:{sec % 60:02d}"
    return f"{sec}s"


def _fmt_clock(sec_from_now):
    import time as _t
    if sec_from_now is None:
        return ""
    lt = _t.localtime(_t.time() + sec_from_now)
    return f"{lt.tm_hour:02d}:{lt.tm_min:02d}"


class _Eta:
    """Rate-based ETA from monotonic progress fractions (EMA-smoothed)."""

    def __init__(self):
        import time as _t
        self.t0 = _t.time()
        self._ema = None

    def update(self, frac):
        import time as _t
        if frac is None or frac <= 0.02:
            return None
        frac = min(float(frac), 1.0)
        elapsed = _t.time() - self.t0
        if elapsed < 3:
            return None
        raw = elapsed * (1.0 - frac) / frac
        self._ema = raw if self._ema is None else             0.7 * self._ema + 0.3 * raw
        return self._ema


class OneClickTab(ttk.Frame):

    def __init__(self, parent, main_gui):
        super().__init__(parent)
        self.app = main_gui
        self._videos = []            # scan rows (dicts)
        self._running = False
        self._active_stage = None
        self._stage_state = {}       # key -> pending/running/done/failed/skipped
        self._pose_dlg = None
        self._pose_h5s = None
        self._pose_log_index = "1.0"
        self._cancel_requested = False
        self._build_ui()
        self.after(400, self.on_project_changed)

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        hdr = ttk.Frame(self)
        hdr.pack(fill="x", padx=14, pady=(12, 2))
        ttk.Label(hdr, text="⚡  Quick Start",
                  font=(FONT_FAMILY, 15, "bold")).pack(side="left")

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=14, pady=6)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        # ── left: session table ─────────────────────────────────────────
        sess = ttk.LabelFrame(body, text="1.  Sessions", padding=8)
        sess.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        # bottom rows first so the tree gets exactly the remaining space
        self._scan_lbl = ttk.Label(sess, text="", foreground="#666666")
        self._scan_lbl.pack(side="bottom", anchor="w", pady=(4, 0))
        btns = ttk.Frame(sess)
        btns.pack(side="bottom", fill="x", pady=(8, 0))
        ttk.Button(btns, text="Select all",
                   command=lambda: self._tree.selection_set(
                       self._tree.get_children())).pack(side="left")
        ttk.Button(btns, text="Unanalyzed only",
                   command=self._select_unanalyzed).pack(side="left",
                                                         padx=(6, 0))
        ttk.Button(btns, text="Clear",
                   command=lambda: self._tree.selection_remove(
                       self._tree.get_children())).pack(side="left",
                                                        padx=(6, 0))
        ttk.Button(btns, text="Rescan",
                   command=self.on_project_changed).pack(side="left",
                                                         padx=(6, 0))
        _add = ttk.Button(btns, text="➕ Add videos…",
                          command=self._add_videos)
        _add.pack(side="right")
        Tip(_add, "Copy new videos into the project's videos/ folder.")
        _pv = ttk.Button(btns, text="🎬 Check tracking…",
                         command=self._open_pose_preview)
        _pv.pack(side="right", padx=(0, 6))
        Tip(_pv, "Play a tracked video with the pose keypoints overlaid -\n"
                 "sanity-check tracking quality before analyzing.")

        tf = ttk.Frame(sess)
        tf.pack(fill="both", expand=True)
        cols = ("name", "duration", "cal", "status")
        self._tree = ttk.Treeview(tf, columns=cols, show="headings",
                                  height=14)
        for c, h, w in (("name", "Video", 240), ("duration", "Length", 60),
                        ("cal", "Calibration", 100), ("status", "Status", 120)):
            self._tree.heading(c, text=h)
            self._tree.column(c, width=w, anchor="w",
                              stretch=(c == "name"))
        vsb = ttk.Scrollbar(tf, orient="vertical",
                            command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # ── right: steps + run ──────────────────────────────────────────
        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")

        # Walking-paws activity indicator (same as the active-learning
        # retrain window): paws march while the pipeline is running.
        self._PAW_MAX = 8
        self._paw_n = 0
        self._paw_running = False
        self._paw_canvas = tk.Canvas(right, height=36, width=420,
                                     highlightthickness=0)
        self._paw_canvas.pack(fill="x", pady=(0, 2))

        steps = ttk.LabelFrame(right, text="2.  Steps", padding=8)
        steps.pack(fill="x")
        self._stage_vars = {}
        self._stage_dots = {}
        self._stage_lbls = {}
        self._stage_etas = {}
        for key, label, style in STAGES:
            row = ttk.Frame(steps)
            row.pack(fill="x", pady=1)
            dot = ttk.Label(row, text=_DOTS["pending"][0], width=2,
                            foreground=_DOTS["pending"][1],
                            font=(FONT_FAMILY, 11, "bold"))
            dot.pack(side="left")
            var = tk.BooleanVar(value=True)
            chk = ttk.Checkbutton(row, text=label, variable=var)
            chk.pack(side="left")
            eta = ttk.Label(row, text="", foreground="#888888",
                            font=(FONT_FAMILY, 8))
            eta.pack(side="right")
            self._stage_vars[key] = var
            self._stage_dots[key] = dot
            self._stage_lbls[key] = chk
            self._stage_etas[key] = eta
        Tip(self._stage_lbls["features"],
            "When Classifiers is also ticked, features are extracted inside "
            "the classifier run (one pass per settings group) - this row "
            "then tracks that phase.")
        Tip(self._stage_lbls["gait"],
            "Runs the Gait & Limb analysis with the manuscript paw-contact "
            "preset (contour + brightness + licking exclusion).")

        # Active pose model - resolved fresh at every run, so switching or
        # upgrading the network in Pose Estimation is picked up automatically.
        mrow = ttk.Frame(steps)
        mrow.pack(fill="x", pady=(8, 0))
        ttk.Label(mrow, text="Model:").pack(side="left")
        self._model_lbl = ttk.Label(mrow, text="detecting…",
                                    foreground="#666666")
        self._model_lbl.pack(side="left", padx=(4, 6))
        _ch = ttk.Button(mrow, text="Change…", width=9,
                         command=lambda: self._jump(
                             "🦴 Pose Estimation"))
        _ch.pack(side="right")
        Tip(_ch, "Pose models are managed on the Pose Estimation tab\n"
                 "(Installed pose models). Whichever model is active there\n"
                 "is the one this pipeline uses - switching or installing a\n"
                 "new network takes effect on the next run.")
        Tip(self._model_lbl,
            "The DLC network the Pose tracking step will use.")
        self.after(700, self._refresh_model)

        dev = ttk.Frame(steps)
        dev.pack(fill="x", pady=(8, 0))
        ttk.Label(dev, text="Device:").pack(side="left")
        self._providers = []
        self._device_var = tk.StringVar(value="")
        self._device_cb = ttk.Combobox(dev, textvariable=self._device_var,
                                       state="readonly", width=22, values=[])
        self._device_cb.pack(side="left", padx=(4, 10))
        ttk.Label(dev, text="Batch:").pack(side="left")
        self._batch_var = tk.IntVar(value=16)
        ttk.Spinbox(dev, from_=1, to=64, textvariable=self._batch_var,
                    width=5).pack(side="left", padx=(4, 0))
        self.after(600, self._probe_devices)

        run = ttk.LabelFrame(right, text="3.  Run", padding=8)
        run.pack(fill="both", expand=True, pady=(8, 0))
        rb = ttk.Frame(run)
        rb.pack(fill="x")
        self._run_btn = ttk.Button(rb, text="▶  Run pipeline",
                                   command=self.run_pipeline)
        self._run_btn.pack(side="left", fill="x", expand=True)
        self._cancel_btn = ttk.Button(rb, text="⏹ Stop", width=8,
                                      command=self._cancel,
                                      state="disabled")
        self._cancel_btn.pack(side="left", padx=(6, 0))

        self._bar = ttk.Progressbar(run, mode="determinate", maximum=100)
        self._bar.pack(fill="x", pady=(8, 2))
        self._status_lbl = ttk.Label(run, text="Ready.",
                                     foreground="#666666", wraplength=340,
                                     justify="left")
        self._status_lbl.pack(anchor="w")

        self._log = tk.Text(run, height=8, wrap="word", state="disabled",
                            font=("Consolas", 8))
        _lsb = ttk.Scrollbar(run, orient="vertical", command=self._log.yview)
        self._log.configure(yscrollcommand=_lsb.set)
        self._log.pack(side="left", fill="both", expand=True, pady=(6, 0))
        _lsb.pack(side="right", fill="y", pady=(6, 0))

        jump = ttk.Frame(right)
        jump.pack(fill="x", pady=(6, 0))
        self._jump_single = ttk.Button(
            jump, text="Open Single-Classifier",
            command=lambda: self._jump("\U0001f4c8 Single-Classifier Analysis"),
            state="disabled")
        self._jump_single.pack(side="left", fill="x", expand=True)
        self._jump_gait = ttk.Button(
            jump, text="Open Gait & Limb",
            command=lambda: self._jump("\U0001f43e Gait & Limb Use"),
            state="disabled")
        self._jump_gait.pack(side="left", fill="x", expand=True, padx=(6, 0))

    # ------------------------------------------------------------------ data

    def _project(self):
        try:
            return self.app.current_project_folder.get()
        except Exception:
            return ""

    def on_project_changed(self):
        from dlc_run_dialog import scan_project_videos
        proj = self._project()
        videos_dir = os.path.join(proj, "videos") if proj else ""
        if proj and not os.path.isdir(videos_dir):
            videos_dir = proj
        self._videos = scan_project_videos(videos_dir) if proj else []
        for i in self._tree.get_children():
            self._tree.delete(i)
        for v in self._videos:
            self._tree.insert("", "end", iid=str(v["index"]),
                              values=(v["name"], v["duration"], v["cal"],
                                      v["status"]))
        self._tree.selection_set([str(v["index"]) for v in self._videos
                                  if v["select_default"]] or
                                 [str(v["index"]) for v in self._videos])
        n = len(self._videos)
        todo = sum(1 for v in self._videos if v["select_default"])
        self._scan_lbl.config(
            text=f"{n} video(s); {todo} not yet pose-tracked"
            if n else "No videos found - use ➕ Add videos…")

    def _select_unanalyzed(self):
        self._tree.selection_set([str(v["index"]) for v in self._videos
                                  if v["select_default"]])

    def _open_pose_preview(self):
        from pose_preview import open_pose_preview
        sel = self._tree.selection()
        vp = None
        if sel:
            v = next((v for v in self._videos
                      if str(v["index"]) == sel[0]), None)
            if v:
                vp = v["path"]
        open_pose_preview(self.winfo_toplevel(), self._project(), vp)

    def _add_videos(self):
        fn = getattr(self.app, "add_videos_to_project", None)
        if fn:
            fn(on_done=self.on_project_changed)

    def _refresh_model(self):
        def _worker():
            txt = "no pose model installed"
            try:
                from dlc_run_dialog import load_active_bundle
                b = load_active_bundle() if load_active_bundle else None
                if b is not None:
                    txt = (f"{b.display_name}  "
                           f"(shuffle {b.dlc.shuffle}, {b.dlc.snapshot})")
            except Exception:
                txt = "model detection failed"
            try:
                self.after(0, lambda: self._model_lbl.config(text=txt))
            except Exception:
                pass
        threading.Thread(target=_worker, daemon=True).start()

    def _probe_devices(self):
        def _worker():
            try:
                from dlc_run_dialog import _probe_providers
                provs = _probe_providers()
            except Exception:
                provs = []
            def _apply():
                self._providers = provs
                self._device_cb.configure(
                    values=[p["display"] for p in provs])
                if provs:
                    self._device_var.set(provs[0]["display"])
                    self._batch_var.set(provs[0].get("suggested_batch", 16))
            try:
                self.after(0, _apply)
            except Exception:
                pass
        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------ log

    def _log_line(self, msg):
        self._log.configure(state="normal")
        self._log.insert("end", msg.rstrip("\n") + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    def _eta_reset(self, unit_key=None):
        """Fresh trackers for the active stage (batch + optional per-unit)."""
        self._eta_batch = _Eta()
        self._eta_unit = _Eta()
        self._eta_unit_key = unit_key

    def _eta_show(self, key, batch_frac):
        """Row ETA: '~left · ≈clock'. Returns remaining seconds or None."""
        b = self._eta_batch.update(batch_frac)
        lbl = self._stage_etas.get(key)
        if lbl is not None:
            lbl.config(text=(f"~{_fmt_eta(b)} left · ≈{_fmt_clock(b)}"
                             if b is not None else "estimating…"))
        return b

    def _eta_clear(self, key=None):
        for k, lbl in self._stage_etas.items():
            if key is None or k == key:
                lbl.config(text="")

    def _draw_paws(self, n, done=False):
        """n 🐾 rotated to point right, marching left to right."""
        try:
            self._paw_canvas.delete("all")
            for i in range(n):
                x = 16 + i * 46
                y = 18 + (4 if i % 2 else -4)   # alternate: walking steps
                self._paw_canvas.create_text(x, y, text="🐾",
                                             angle=270,
                                             font=(FONT_FAMILY, 14))
            if done:
                self._paw_canvas.create_text(
                    16 + n * 46 + 6, 18, text="✓",
                    font=(FONT_FAMILY, 13, "bold"), fill="#2ca02c")
        except Exception:
            self._paw_running = False

    def _animate_paws(self):
        if not self._paw_running:
            return
        self._paw_n = (self._paw_n % self._PAW_MAX) + 1
        self._draw_paws(self._paw_n)
        try:
            self.after(320, self._animate_paws)
        except Exception:
            self._paw_running = False

    def _paws_start(self):
        if not self._paw_running:
            self._paw_running = True
            self._paw_n = 0
            self._animate_paws()

    def _paws_stop(self, done=False):
        self._paw_running = False
        try:
            if done:
                self._draw_paws(self._PAW_MAX, done=True)
            else:
                self._paw_canvas.delete("all")
        except Exception:
            pass

    def _set_stage(self, key, state):
        self._stage_state[key] = state
        dot, color = _DOTS[state]
        sty = next(s for k, _l, s in STAGES if k == key)
        fg = _STAGE_COLORS[sty] if state == "running" else color
        self._stage_dots[key].config(text=dot, foreground=fg)
        if state == "running":
            self._active_stage = key
            self._eta_reset()
            self._bar.configure(style=f"{sty}.Horizontal.TProgressbar",
                                value=0)
            self._status_lbl.config(
                text=next(l for k, l, _s in STAGES if k == key) + "…")
        elif state in ("done", "skipped", "failed"):
            self._eta_clear(key)

    # ------------------------------------------------------------------ run

    def run_pipeline(self):
        if self._running:
            return
        proj = self._project()
        if not proj or not os.path.isdir(proj):
            messagebox.showwarning("No project",
                                   "Load a project folder first.", parent=self)
            return
        sel = self._tree.selection()
        if not sel:
            messagebox.showwarning("No videos",
                                   "Select at least one video.", parent=self)
            return
        vids = [v for v in self._videos if str(v["index"]) in set(sel)]

        self._plan = [k for k, _l, _s in STAGES if self._stage_vars[k].get()]
        if not self._plan:
            return
        # pose only needed for videos without pose files (unless transcoding
        # changes them first - transcode keeps names, h5s stay valid)
        need_pose = [v for v in vids if v["select_default"]]
        self._vids = vids
        self._need_pose = need_pose

        self._running = True
        self._cancel_requested = False
        self.app._oneclick_active = True
        self._run_btn.config(state="disabled")
        self._cancel_btn.config(state="normal")
        self._jump_single.config(state="disabled")
        self._jump_gait.config(state="disabled")
        for k, _l, _s in STAGES:
            self._set_stage_silent(k, "pending" if k in self._plan
                                   else "skipped")
        self._refresh_model()
        self._paws_start()
        self._log_line(f"Pipeline started: {len(vids)} video(s); "
                       f"steps: {', '.join(self._plan)}")
        self._advance(None)

    def _set_stage_silent(self, key, state):
        self._stage_state[key] = state
        dot, color = _DOTS[state]
        self._stage_dots[key].config(text=dot, foreground=color or "#999999")
        if state in ("done", "skipped", "failed", "pending"):
            self._eta_clear(key)

    def _advance(self, finished_key):
        """Mark `finished_key` done and start the next planned stage."""
        if finished_key is not None:
            if self._stage_state.get(finished_key) == "running":
                self._set_stage_silent(finished_key, "done")
                self._stage_dots[finished_key].config(
                    foreground=_DOTS["done"][1])
        if not self._running:
            return
        remaining = [k for k in self._plan
                     if self._stage_state.get(k) == "pending"]
        nxt = remaining[0] if remaining else None
        if nxt in ("transcode", "pose"):
            self._start_pose_stage()
        elif nxt == "features" and "classifiers" in self._plan:
            self._start_batch_stage()
        elif nxt == "classifiers":
            self._start_batch_stage()
        elif nxt == "features":
            self._start_fx_stage()
        elif nxt == "gait":
            self._start_gait_stage()
        else:
            self._finish()

    # ── stage: transcode + pose (one DLCProgressDialog worker) ───────────

    def _start_pose_stage(self):
        want_transcode = "transcode" in self._plan and \
            self._stage_state.get("transcode") == "pending"
        want_pose = "pose" in self._plan and \
            self._stage_state.get("pose") == "pending"
        if want_pose and not self._need_pose and not want_transcode:
            self._log_line("Pose: every selected video is already tracked - "
                           "skipping.")
            self._set_stage_silent("pose", "skipped")
            self._advance(None)
            return

        from dlc_run_dialog import DLCProgressDialog, load_active_bundle
        import shutil as _sh
        bundle = load_active_bundle() if load_active_bundle else None
        if want_pose and self._need_pose and bundle is None:
            self._fail_stage("pose", "No DLC model bundle installed.")
            return
        prov = next((p for p in self._providers
                     if p["display"] == self._device_var.get()),
                    {"name": "cpu"})
        pose_videos = [v["path"] for v in
                       (self._need_pose if want_pose else [])]
        settings = {
            "videos": ([v["path"] for v in self._vids]
                       if want_transcode else pose_videos),
            "device": prov["name"],
            "batch_size": int(self._batch_var.get()),
            "auto_predict": False, "run_gait": False,
            "select_frames": False, "extract_features": False,
            "transcode": bool(want_transcode
                              and _sh.which("ffmpeg") is not None),
            "bundle": bundle,
        }
        if not settings["videos"]:
            for k in ("transcode", "pose"):
                if self._stage_state.get(k) == "pending":
                    self._set_stage_silent(k, "skipped")
            self._advance(None)
            return
        if want_transcode:
            self._set_stage("transcode", "running")
        elif want_pose:
            self._set_stage("pose", "running")
        self._pose_h5s = None
        self._pose_log_index = "1.0"
        dlg = DLCProgressDialog(self.app.root, settings,
                                on_complete=self._on_pose_done)
        try:
            dlg.withdraw()
        except Exception:
            pass
        self._pose_dlg = dlg
        self.after(250, self._poll_pose)

    def _on_pose_done(self, h5_paths):
        self._pose_h5s = list(h5_paths or [])

    def _poll_pose(self):
        dlg = self._pose_dlg
        if dlg is None or not self._running:
            return
        try:
            cur_txt = str(dlg.current_label.cget("text"))
            in_transcode = cur_txt.startswith("transcode ")
            # stage flip transcode -> pose
            if (not in_transcode and cur_txt.strip()
                    and self._stage_state.get("transcode") == "running"):
                self._set_stage_silent("transcode", "done")
                if self._stage_state.get("pose") == "pending":
                    self._set_stage("pose", "running")
            ov, om = dlg.overall_bar["value"], dlg.overall_bar["maximum"]
            cv = dlg.current_bar["value"]
            self._bar.configure(
                value=min(100.0, (ov + cv / 100.0) / max(om, 1) * 100.0))
            if cur_txt != getattr(self, "_pose_prev_label", None):
                self._pose_prev_label = cur_txt
                self._eta_unit = _Eta()
            _ak = ("transcode"
                   if self._stage_state.get("transcode") == "running"
                   else "pose")
            self._eta_show(_ak, (ov + cv / 100.0) / max(om, 1))
            # readable status: stage - video i/n - fps - this-video ETA
            import re as _re
            stats = str(dlg.current_stats.cget("text"))
            fpsm = _re.search(r"([\d.]+)\s*fps", stats)
            u = self._eta_unit.update(cv / 100.0)
            bits = [("Transcoding" if _ak == "transcode"
                     else "Pose tracking"),
                    f"video {min(int(ov) + 1, int(om))}/{int(om)}"]
            if fpsm:
                bits.append(f"{float(fpsm.group(1)):.0f} fps")
            if u is not None:
                bits.append(f"this video ~{_fmt_eta(u)}")
            self._status_lbl.config(text="  ·  ".join(bits))
            # mirror new log lines
            new = dlg.log_text.get(self._pose_log_index, "end-1c")
            if new.strip():
                for ln in new.splitlines():
                    if ln.strip():
                        self._log_line(ln)
            self._pose_log_index = dlg.log_text.index("end-1c")
        except Exception:
            pass
        alive = bool(getattr(dlg, "worker", None)
                     and dlg.worker.is_alive())
        if self._pose_h5s is None and alive:
            self.after(300, self._poll_pose)
            return
        # worker finished
        cancelled = bool(getattr(dlg, "_cancelled", False))
        try:
            dlg.destroy()
        except Exception:
            pass
        self._pose_dlg = None
        if cancelled or self._cancel_requested or not self._running:
            self._abort("Stopped during pose/transcode.")
            return
        for k in ("transcode", "pose"):
            if self._stage_state.get(k) == "running":
                self._set_stage_silent(k, "done")
        if self._pose_h5s is None:
            self._pose_h5s = []
        n_new = len(self._pose_h5s)
        if n_new:
            self._log_line(f"Pose done: {n_new} video(s) tracked.")
        self.on_project_changed()
        self._advance(None)

    # ── stage: features + classifiers (Core-8 batch) ─────────────────────

    def _start_batch_stage(self):
        feat_planned = self._stage_state.get("features") == "pending"
        self._set_stage("classifiers", "running")
        if feat_planned:
            self._set_stage("features", "running")   # bar color: features
        self._batch_log_index = "1.0"
        try:
            ok = self.app.run_default_classifier_set(
                select_tab=False,
                only_videos=[v["name"] for v in self._vids])
        except Exception as e:
            self._fail_stage("classifiers", str(e))
            return
        self.after(400, self._poll_batch)

    def _poll_batch(self):
        if not self._running:
            return
        app = self.app
        try:
            bar = getattr(app, "batch_progress", None)
            frac = None
            if bar is not None:
                frac = float(bar["value"]) / 100.0
                self._bar.configure(value=float(bar["value"]))
            lbl = getattr(app, "batch_progress_label", None)
            t = str(lbl.cget("text")) if lbl is not None else ""
            _ak = ("features"
                   if self._stage_state.get("features") == "running"
                   else "classifiers")
            self._eta_show(_ak, frac)
            import re as _re
            m = _re.search(r"Processing (\d+)/(\d+)", t)
            bits = ["Scoring behaviors"]
            if m:
                cur, tot = int(m.group(1)), int(m.group(2))
                opv = max(len(getattr(app, "batch_classifiers", {}) or {}),
                          1)
                n_vids = max(tot // opv, 1)
                vid_idx = (max(cur, 1) - 1) // opv
                if vid_idx != getattr(self, "_batch_vid_idx", None):
                    self._batch_vid_idx = vid_idx
                    self._eta_unit = _Eta()
                u = self._eta_unit.update(
                    ((max(cur, 1) - 1) % opv + 1) / opv)
                bits.append(f"video {vid_idx + 1}/{n_vids}")
                bits.append(f"classifier {(max(cur, 1) - 1) % opv + 1}"
                            f"/{opv}")
                if u is not None:
                    bits.append(f"this video ~{_fmt_eta(u)}")
            self._status_lbl.config(text="  ·  ".join(bits))
            logw = getattr(app, "batch_log", None)
            if logw is not None:
                new = logw.get(self._batch_log_index, "end-1c")
                self._batch_log_index = logw.index("end-1c")
                tail = [ln for ln in new.splitlines() if ln.strip()]
                for ln in tail[-6:]:
                    self._log_line(ln)
                # flip the bar color between features / classifiers phases
                joined = "\n".join(tail)
                if "Extracting features" in joined and \
                        self._stage_vars["features"].get():
                    self._bar.configure(
                        style="OCfeatures.Horizontal.TProgressbar")
                elif "Running prediction" in joined or "→ Running" in joined:
                    if self._stage_state.get("features") == "running":
                        self._set_stage_silent("features", "done")
                    self._bar.configure(
                        style="OCclassifiers.Horizontal.TProgressbar")
        except Exception:
            pass
        th = getattr(app, "_batch_thread", None)
        if th is not None and th.is_alive():
            self.after(400, self._poll_batch)
            return
        if not self._running:
            return
        if self._cancel_requested or (
                getattr(app, "_batch_cancel_flag", None) is not None
                and app._batch_cancel_flag.is_set()):
            self._abort("Stopped during classifier scoring.")
            return
        if self._stage_state.get("features") == "running":
            self._set_stage_silent("features", "done")
        self._advance("classifiers")

    # ── stage: standalone feature extraction (no classifiers ticked) ─────

    def _start_fx_stage(self):
        self._set_stage("features", "running")
        try:
            self.app.open_feature_extraction_run()
            self._log_line("Feature extraction opened in its own window "
                           "(runs independently).")
        except Exception as e:
            self._fail_stage("features", str(e))
            return
        # the FX window manages its own progress; treat as detached
        self._set_stage_silent("features", "done")
        self._advance(None)

    # ── stage: gait ──────────────────────────────────────────────────────

    def _start_gait_stage(self):
        tab = getattr(self.app, "wb_tab", None)
        if tab is None or not hasattr(tab, "auto_run"):
            self._fail_stage("gait", "Gait tab unavailable.")
            return
        self._set_stage("gait", "running")
        try:
            tab.auto_run()
        except Exception as e:
            self._fail_stage("gait", str(e))
            return
        th = getattr(tab, "_fit_thread", None)
        if th is None or not th.is_alive():
            # auto_run logged why it skipped
            self._log_line("Gait run did not start (see Gait tab log).")
            self._set_stage_silent("gait", "skipped")
            self._advance(None)
            return
        self.after(400, self._poll_gait)

    def _poll_gait(self):
        if not self._running:
            return
        tab = self.app.wb_tab
        try:
            v = float(tab._progress["value"])
            m = float(tab._progress["maximum"]) or 1.0
            self._bar.configure(value=v / m * 100.0)
            sub = str(tab._sub_progress_label.cget("text"))
            self._eta_show("gait", v / m)
            import time as _t
            bits = ["Gait & contour",
                    f"session {min(int(v) + 1, int(m))}/{int(m)}"]
            if v >= 1:
                _per = (_t.time() - self._eta_batch.t0) / v
                bits.append(f"≈{_fmt_eta(_per)}/session")
            if sub:
                bits.append(sub)
            self._status_lbl.config(text="  ·  ".join(bits))
        except Exception:
            pass
        th = getattr(tab, "_fit_thread", None)
        if th is not None and th.is_alive():
            self.after(400, self._poll_gait)
            return
        if self._cancel_requested:
            self._abort("Stopped during gait analysis.")
            return
        self._advance("gait")

    # ------------------------------------------------------------------ end

    def _fail_stage(self, key, msg):
        self._set_stage_silent(key, "failed")
        self._log_line(f"✗ {key}: {msg}")
        self._finish(failed=True)

    def _abort(self, msg):
        self._log_line(msg)
        for k, _l, _s in STAGES:
            if self._stage_state.get(k) == "running":
                self._set_stage_silent(k, "failed")
        self._finish(failed=True)

    def _finish(self, failed=False):
        self._running = False
        self._cancel_requested = False
        self._paws_stop(done=not failed)
        self.app._oneclick_active = False
        self._run_btn.config(state="normal")
        self._cancel_btn.config(state="disabled")
        self._active_stage = None
        if failed:
            self._status_lbl.config(text="Pipeline stopped - see log.")
        else:
            self._bar.configure(value=100)
            done = [k for k, _l, _s in STAGES
                    if self._stage_state.get(k) == "done"]
            self._status_lbl.config(
                text="All done - graphs are ready in the Analyze tabs."
                if done else "Nothing to do.")
            self._log_line("Pipeline finished.")
            self._jump_single.config(state="normal")
            self._jump_gait.config(state="normal")

    def _cancel(self):
        """Stop everything: signal every stage unconditionally (immune to
        stage-tracking drift), log what was signalled, and watchdog-release
        the UI if a poller fails to notice within 10 s."""
        if not self._running or self._cancel_requested:
            return
        self._cancel_requested = True
        self._cancel_btn.config(state="disabled")
        self._status_lbl.config(
            text="Stopping after the current operation…")
        self._log_line("Stop requested.")
        # pose / transcode worker
        dlg = self._pose_dlg
        if dlg is not None:
            try:
                dlg._cancelled = True
                proc = getattr(dlg, "proc", None)
                if proc is not None and proc.poll() is None:
                    import subprocess as _sp
                    try:
                        _sp.run(["taskkill", "/PID", str(proc.pid),
                                 "/T", "/F"], capture_output=True,
                                timeout=10)
                        self._log_line(
                            f"  killed pose/transcode process tree "
                            f"(pid {proc.pid})")
                    except Exception:
                        proc.terminate()
                        self._log_line("  terminated pose/transcode process")
                else:
                    self._log_line("  pose/transcode worker flagged to stop")
            except Exception as e:
                self._log_line(f"  pose stop signal failed: {e}")
        # classifier batch
        flag = getattr(self.app, "_batch_cancel_flag", None)
        if flag is not None:
            flag.set()
            self._log_line("  classifier batch flagged to stop "
                           "(takes effect after the current classifier)")
        # gait
        try:
            tab = getattr(self.app, "wb_tab", None)
            if tab is not None and getattr(tab, "_fit_thread", None)                     is not None and tab._fit_thread.is_alive():
                tab._cancel_analysis()
                self._log_line("  gait analysis flagged to stop")
        except Exception:
            pass
        # watchdog: never leave the button looking dead
        self.after(10000, self._cancel_watchdog)

    def _cancel_watchdog(self):
        if self._running and self._cancel_requested:
            self._log_line("Stop watchdog: releasing the pipeline "
                           "(a background worker may still be finishing "
                           "its current operation).")
            self._abort("Stopped.")

    def _jump(self, label):
        try:
            self.app.notebook.select(label)
        except Exception:
            pass
