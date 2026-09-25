# -*- coding: utf-8 -*-
"""Scoring review - play a scored video with the classifier calls overlaid.

The sibling of pose_preview for the step after Run Classifiers: shows which
behaviors fired on each frame (chips on the frame, a bout timeline under the
transport) so a user can eyeball the labeling before trusting the graphs.
The Keypoints toggle draws the DLC pose on the animal (same dots as the pose
preview), from the .h5 next to the video.
Reads the consolidated per-frame sheets Run Classifiers writes to
results/per_frame/<video>_frames.csv (<Behavior>_pred / <Behavior>_prob
columns) and falls back to results/<Behavior>/<video>_predictions.csv.

    from scoring_preview import open_scoring_preview
    open_scoring_preview(root, project_folder)            # session picker
    open_scoring_preview(root, project_folder, video)     # jump straight in
"""

from __future__ import annotations

import os
import glob
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

from pose_preview import keep_colors, h5_for_video, _load_pose, draw_keypoints

_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
# tab10-ish, readable both as a filled chip and as a timeline bar
_COLORS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
           (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
           (188, 189, 34), (23, 190, 207), (255, 187, 120), (152, 223, 138)]


def _hex(rgb):
    return "#%02x%02x%02x" % rgb


def _find_sessions(project_folder):
    """[(video_path, per_frame_csv | None, {behavior: predictions_csv})] for
    videos that have at least one scoring output."""
    vd = os.path.join(project_folder, "videos")
    if not os.path.isdir(vd):
        vd = project_folder
    rd = os.path.join(project_folder, "results")
    pf_dir = os.path.join(rd, "per_frame")
    out = []
    for vp in sorted(p for e in _VIDEO_EXTS
                     for p in glob.glob(os.path.join(vd, f"*{e}"))):
        base = os.path.basename(vp)
        if "DLC_" in base or "_labeled" in base:
            continue
        stem = os.path.splitext(base)[0]
        pf = os.path.join(pf_dir, f"{stem}_frames.csv")
        pf = pf if os.path.isfile(pf) else None
        per_beh = {}
        if os.path.isdir(rd):
            # results/<Behavior>/<stem>_classifier_<Behavior>_predictions.csv;
            # the "_" after the stem keeps S1 from matching S10's files.
            for pat in (f"{stem}_predictions.csv", f"{stem}_*_predictions.csv"):
                for p in glob.glob(os.path.join(rd, "*", pat)):
                    beh = os.path.basename(os.path.dirname(p))
                    if beh != "per_frame":
                        per_beh.setdefault(beh, p)
        if pf or per_beh:
            out.append((vp, pf, per_beh))
    return out


def _load_calls(pf_csv, per_beh):
    """{behavior: (pred uint8 array, prob float array | None)}; the per-frame
    sheet wins because every behavior in it shares one frame axis."""
    import pandas as pd
    calls = {}
    if pf_csv:
        df = pd.read_csv(pf_csv)
        for c in df.columns:
            if str(c).endswith("_pred"):
                beh = str(c)[:-5]
                pred = df[c].fillna(0).to_numpy().astype(np.uint8)
                pc = f"{beh}_prob"
                prob = (df[pc].to_numpy(float) if pc in df.columns else None)
                calls[beh] = (pred, prob)
    for beh, p in per_beh.items():
        if beh in calls:
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        # Run Classifiers and Predict write frame, probability and a 0/1
        # column named after the behavior; older files used "prediction".
        pcol = next((c for c in df.columns
                     if str(c).lower() in ("prediction", "pred")), None)
        if pcol is None:
            others = [c for c in df.columns if str(c).lower() not in
                      ("frame", "probability", "prob", "unnamed: 0")]
            pcol = (beh if beh in df.columns else
                    others[0] if len(others) == 1 else None)
        if pcol is None:
            continue
        pred = df[pcol].fillna(0).to_numpy().astype(np.uint8)
        prcol = next((c for c in df.columns
                      if str(c).lower() in ("probability", "prob")), None)
        prob = df[prcol].to_numpy(float) if prcol else None
        calls[beh] = (pred, prob)
    return calls


def _bouts(pred):
    """[(start, end_exclusive)] runs of 1s."""
    if pred is None or len(pred) == 0:
        return []
    d = np.diff(np.concatenate(([0], pred.astype(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def open_scoring_preview(root, project_folder, video_path=None):
    import cv2
    sessions = _find_sessions(project_folder or "")
    if not sessions:
        messagebox.showinfo(
            "No scored videos",
            "No classifier output was found for this project's videos - run "
            "the classifiers first (Quick Start or Run Classifiers).",
            parent=root)
        return

    win = tk.Toplevel(root)
    win.title("Scoring review - classifier calls on video")
    frm = ttk.Frame(win, padding=8)
    frm.pack(fill="both", expand=True)

    state = {"cap": None, "calls": {}, "order": [], "bouts": {}, "n": 0,
             "fps": 30.0, "frame": 0, "playing": False, "after": None,
             "photo": None, "setting": False, "shown": {},
             "h5": None, "pose": None}

    # ── top: session pick + display options ────────────────────────────
    top = ttk.Frame(frm)
    top.pack(fill="x")
    ttk.Label(top, text="Session:").pack(side="left")
    names = [os.path.basename(v) for v, _p, _b in sessions]
    sess_var = tk.StringVar()
    sess_cb = ttk.Combobox(top, textvariable=sess_var, state="readonly",
                           values=names, width=32)
    sess_cb.pack(side="left", padx=(4, 12))
    # Nothing else references sess_var, so once this function returns it
    # would be garbage-collected and Tk would unset the variable (blank box).
    win._keep = [sess_var]
    prob_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(top, text="Probabilities", variable=prob_var,
                    command=lambda: _show(state["frame"])).pack(side="left")
    tint_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(top, text="Tint frame when active", variable=tint_var,
                    command=lambda: _show(state["frame"])).pack(side="left",
                                                                padx=(8, 0))
    kp_var = tk.BooleanVar(value=False)
    kp_cb = ttk.Checkbutton(top, text="Keypoints", variable=kp_var,
                            command=lambda: _show(state["frame"]))
    kp_cb.pack(side="left", padx=(8, 0))
    src_lbl = ttk.Label(top, text="", foreground="#666666")
    src_lbl.pack(side="right")

    # behavior toggles (filled per session)
    beh_row = ttk.Frame(frm)
    beh_row.pack(fill="x", pady=(4, 0))

    canvas = keep_colors(tk.Label, frm, background="#111111")
    canvas.pack(fill="both", expand=True, pady=6)

    # ── bottom: transport + bout timeline ──────────────────────────────
    bot = ttk.Frame(frm)
    bot.pack(fill="x")
    ttk.Button(bot, text="⏮ bout", width=7,
               command=lambda: _jump_bout(-1)).pack(side="left")
    play_btn = ttk.Button(bot, text="▶", width=4)
    play_btn.pack(side="left", padx=(4, 0))
    ttk.Button(bot, text="bout ⏭", width=7,
               command=lambda: _jump_bout(+1)).pack(side="left", padx=(4, 0))
    slider = ttk.Scale(bot, from_=0, to=100, orient="horizontal")
    slider.pack(side="left", fill="x", expand=True, padx=8)
    time_lbl = ttk.Label(bot, text="0:00 / 0:00", width=14)
    time_lbl.pack(side="left")
    ttk.Label(bot, text="Speed:").pack(side="left", padx=(8, 2))
    speed_var = tk.StringVar(value="1x")
    ttk.Combobox(bot, textvariable=speed_var, state="readonly", width=5,
                 values=["0.25x", "0.5x", "1x", "2x", "4x"]).pack(side="left")

    ROW_H = 14
    GUTTER = 96          # behavior names live here; bars start to the right
    tl = keep_colors(tk.Canvas, frm, height=ROW_H * 3, background="#1a1a1a",
                     highlightthickness=0, cursor="hand2")
    tl.pack(fill="x", pady=(6, 0))
    tl_lbl = ttk.Label(frm, text="", foreground="#666666", wraplength=900,
                       justify="left")
    tl_lbl.pack(fill="x", pady=(2, 0))

    def _shown_behaviors():
        return [b for b in state["order"] if state["shown"][b].get()]

    def _track_w():
        return max(tl.winfo_width() - GUTTER - 2, 10)

    def _draw_timeline():
        tl.delete("all")
        behs = _shown_behaviors()
        n = max(state["n"], 1)
        w = _track_w()
        tl.configure(height=max(ROW_H * len(behs), ROW_H))
        for r, b in enumerate(behs):
            col = _hex(_COLORS[state["order"].index(b) % len(_COLORS)])
            y0, y1 = r * ROW_H + 1, (r + 1) * ROW_H - 1
            tl.create_rectangle(GUTTER, y0, GUTTER + w, y1, fill="#262626",
                                outline="")
            for s, e in state["bouts"].get(b, []):
                x0 = GUTTER + int(s / n * w)
                x1 = max(GUTTER + int(e / n * w), x0 + 1)
                tl.create_rectangle(x0, y0, x1, y1, fill=col, outline="")
            tl.create_text(GUTTER - 4, (y0 + y1) // 2, text=b[:16],
                           anchor="e", fill=col, font=("TkDefaultFont", 7))
        _draw_playhead()

    def _draw_playhead():
        tl.delete("head")
        n = max(state["n"], 1)
        x = GUTTER + int(state["frame"] / n * _track_w())
        tl.create_line(x, 0, x, tl.winfo_height(), fill="#ffffff",
                       width=1, tags="head")

    def _seek_from_timeline(ev):
        n = max(state["n"], 1)
        _stop()
        _show(int((ev.x - GUTTER) / _track_w() * n))

    tl.bind("<Button-1>", _seek_from_timeline)
    tl.bind("<B1-Motion>", _seek_from_timeline)
    tl.bind("<Configure>", lambda e: _draw_timeline())

    def _load(idx):
        _stop()
        if state["cap"] is not None:
            state["cap"].release()
        vp, pf, per_beh = sessions[idx]
        state["cap"] = cv2.VideoCapture(vp)
        state["fps"] = state["cap"].get(cv2.CAP_PROP_FPS) or 30.0
        state["n"] = int(state["cap"].get(cv2.CAP_PROP_FRAME_COUNT))
        try:
            state["calls"] = _load_calls(pf, per_beh)
        except Exception as e:
            messagebox.showerror("Scoring files",
                                 f"Could not read the predictions for "
                                 f"{os.path.basename(vp)}:\n{e}", parent=win)
            state["calls"] = {}
        state["order"] = list(state["calls"])
        # pose loads on first use, so sessions open as fast as before
        state["h5"], state["pose"] = h5_for_video(vp), None
        kp_cb.configure(state="normal" if state["h5"] else "disabled",
                        text=("Keypoints" if state["h5"]
                              else "Keypoints (no pose file)"))
        state["bouts"] = {b: _bouts(p) for b, (p, _q) in state["calls"].items()}
        for ch in beh_row.winfo_children():
            ch.destroy()
        state["shown"] = {}
        ttk.Label(beh_row, text="Behaviors:").pack(side="left")
        # Opting out of the theme also drops its background, so borrow the
        # ttk frame's to keep the row from showing native grey boxes.
        row_bg = ttk.Style(win).lookup("TFrame", "background")
        bg_kw = ({"background": row_bg, "activebackground": row_bg}
                 if row_bg else {})
        for i, b in enumerate(state["order"]):
            var = tk.BooleanVar(value=True)
            state["shown"][b] = var
            nb = len(state["bouts"][b])
            cb = keep_colors(tk.Checkbutton, beh_row, text=f"{b} ({nb})",
                             variable=var,
                             fg=_hex(_COLORS[i % len(_COLORS)]),
                             activeforeground=_hex(_COLORS[i % len(_COLORS)]),
                             command=lambda: (_draw_timeline(),
                                              _show(state["frame"])),
                             **bg_kw)
            cb.pack(side="left", padx=(6, 0))
        src_lbl.config(text=("per-frame sheet" if pf else
                             f"{len(per_beh)} prediction CSV(s)"))
        n_pred = max((len(p) for p, _q in state["calls"].values()), default=0)
        note = ""
        if n_pred and abs(n_pred - state["n"]) > 2:
            note = (f"Note: predictions cover {n_pred} frames, video has "
                    f"{state['n']} - frames past the shorter one show no calls.")
        tl_lbl.config(text=("Timeline: one row per behavior, bars are predicted "
                            "bouts; click or drag to seek. " + note).strip())
        slider.configure(to=max(state["n"] - 1, 1))
        _draw_timeline()
        _show(0)

    def _active(frame_idx):
        """[(behavior, prob | None)] fired on this frame, in display order."""
        out = []
        for b in _shown_behaviors():
            pred, prob = state["calls"][b]
            if frame_idx < len(pred) and pred[frame_idx]:
                pr = (float(prob[frame_idx]) if prob is not None
                      and frame_idx < len(prob) else None)
                out.append((b, pr))
        return out

    def _pose():
        if state["pose"] is None and state["h5"]:
            try:
                state["pose"] = _load_pose(state["h5"])
            except Exception as e:
                state["pose"] = {}
                kp_var.set(False)
                messagebox.showerror("Pose file",
                                     f"Could not read {state['h5']}:\n{e}",
                                     parent=win)
        return state["pose"] or {}

    def _draw(frame_idx, img):
        # keypoints first so the chips and tint sit on top of them
        if kp_var.get() and state["h5"]:
            draw_keypoints(img, _pose(), frame_idx)
        active = _active(frame_idx)
        h, w = img.shape[:2]
        if active and tint_var.get():
            col = _COLORS[state["order"].index(active[0][0]) % len(_COLORS)][::-1]
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), col, 6)
        y = 28
        for b, pr in active:
            col = _COLORS[state["order"].index(b) % len(_COLORS)][::-1]
            txt = b if (pr is None or not prob_var.get()) else f"{b}  {pr:.2f}"
            (tw, th), _bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX,
                                            0.6, 2)
            cv2.rectangle(img, (8, y - th - 6), (8 + tw + 12, y + 6), col, -1)
            cv2.putText(img, txt, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)
            y += th + 16
        if not active:
            cv2.putText(img, "no behavior", (14, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (170, 170, 170), 1, cv2.LINE_AA)
        return img

    def _show(frame_idx):
        cap = state["cap"]
        if cap is None:
            return
        frame_idx = int(max(0, min(frame_idx, state["n"] - 1)))
        if frame_idx != state["frame"] + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, img = cap.read()
        if not ok:
            return
        state["frame"] = frame_idx
        img = _draw(frame_idx, img)
        h, w = img.shape[:2]
        # 480 px tall keeps the whole window (behavior row, transport and an
        # 8-row timeline) inside a 1080p screen above the taskbar.
        scale = min(900 / w, 480 / h, 1.0)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        from PIL import Image, ImageTk
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        state["photo"] = photo                     # keep a reference
        canvas.configure(image=photo)
        state["setting"] = True
        try:
            slider.set(frame_idx)
        except Exception:
            pass
        state["setting"] = False
        secs, tot = frame_idx / state["fps"], state["n"] / state["fps"]
        time_lbl.config(text=f"{int(secs//60)}:{int(secs%60):02d} / "
                             f"{int(tot//60)}:{int(tot%60):02d}")
        _draw_playhead()

    def _jump_bout(direction):
        """Seek to the next / previous bout start among the shown behaviors."""
        _stop()
        starts = sorted({s for b in _shown_behaviors()
                         for s, _e in state["bouts"].get(b, [])})
        if not starts:
            return
        cur = state["frame"]
        if direction > 0:
            nxt = next((s for s in starts if s > cur), None)
        else:
            nxt = next((s for s in reversed(starts) if s < cur), None)
        if nxt is not None:
            _show(nxt)

    def _tick():
        if not state["playing"]:
            return
        speed = float(speed_var.get().rstrip("x"))
        step = max(1, int(round(speed)))
        nxt = state["frame"] + step
        if nxt >= state["n"]:
            _stop()
            return
        _show(nxt)
        delay = max(10, int(1000 / state["fps"] / max(speed, 0.25) * step))
        state["after"] = win.after(delay, _tick)

    def _stop():
        state["playing"] = False
        play_btn.config(text="▶")
        if state["after"]:
            try:
                win.after_cancel(state["after"])
            except Exception:
                pass
            state["after"] = None

    def _toggle():
        if state["playing"]:
            _stop()
        else:
            state["playing"] = True
            play_btn.config(text="⏸")
            _tick()

    play_btn.config(command=_toggle)
    slider.configure(command=lambda v: (
        None if (state["playing"] or state["setting"])
        else _show(float(v))))
    sess_cb.bind("<<ComboboxSelected>>",
                 lambda e: _load(sess_cb.current()))
    win.bind("<space>", lambda e: _toggle())
    win.bind("<Left>", lambda e: (_stop(), _show(state["frame"] - 1)))
    win.bind("<Right>", lambda e: (_stop(), _show(state["frame"] + 1)))
    win.bind("<Prior>", lambda e: _jump_bout(-1))
    win.bind("<Next>", lambda e: _jump_bout(+1))

    def _close():
        _stop()
        if state["cap"] is not None:
            state["cap"].release()
        win.destroy()
    win.protocol("WM_DELETE_WINDOW", _close)

    start = 0
    if video_path:
        # a full path, a file name, or a bare session stem all work
        vn = os.path.splitext(os.path.basename(str(video_path)))[0]
        start = next((i for i, (v, _p, _b) in enumerate(sessions)
                      if os.path.splitext(os.path.basename(v))[0] == vn), 0)
    sess_cb.current(start)
    _load(start)
