# -*- coding: utf-8 -*-
"""Pose-tracking preview - play a video with the DLC keypoints overlaid.

Lets a user sanity-check tracking quality before trusting downstream
analyses. Reused by the Quick Start tab and the Pose Estimation tab:

    from pose_preview import open_pose_preview
    open_pose_preview(root, project_folder)            # session picker
    open_pose_preview(root, project_folder, video)     # jump straight in
"""

from __future__ import annotations

import os
import glob
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

# tab10-ish, high-contrast on dark video
_COLORS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
           (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
           (188, 189, 34), (23, 190, 207), (255, 187, 120), (152, 223, 138)]


def _find_pairs(project_folder):
    """[(video_path, h5_path)] for videos that have pose files."""
    vd = os.path.join(project_folder, "videos")
    if not os.path.isdir(vd):
        vd = project_folder
    pairs = []
    for vp in sorted(glob.glob(os.path.join(vd, "*.mp4"))):
        base = os.path.basename(vp)
        if "DLC_" in base or "_labeled" in base:
            continue
        stem = os.path.splitext(base)[0]
        h5s = sorted(glob.glob(os.path.join(vd, f"{stem}DLC*.h5")))
        if h5s:
            pairs.append((vp, h5s[0]))
    return pairs


def _load_pose(h5_path):
    """{bodypart: (x, y, p)} float arrays from a DLC .h5."""
    import pandas as pd
    df = pd.read_hdf(h5_path)
    out = {}
    if isinstance(df.columns, pd.MultiIndex):
        bps = list(dict.fromkeys(df.columns.get_level_values(-2)))
        for bp in bps:
            sub = df.xs(bp, axis=1, level=-2)

            def _col(*names):
                for c in sub.columns:
                    last = c[-1] if isinstance(c, tuple) else c
                    if str(last).lower() in names:
                        return c
                return None

            xc, yc = _col("x"), _col("y")
            if xc is None or yc is None:
                continue
            x = sub[xc].to_numpy(float).ravel()
            y = sub[yc].to_numpy(float).ravel()
            pc = _col("likelihood", "prob")
            p = (sub[pc].to_numpy(float).ravel() if pc is not None
                 else np.ones_like(x))
            out[str(bp)] = (x, y, p)
    else:  # flat columns: <scorer>_<bp>_x / _y / _prob
        names = [str(c) for c in df.columns]
        stems = sorted({n[:-2] for n in names if n.endswith("_x")})
        for st in stems:
            bp = st.split("_")[-1]
            x = df[st + "_x"].to_numpy(float)
            y = df[st + "_y"].to_numpy(float)
            pcol = next((c for c in (st + "_prob", st + "_likelihood")
                         if c in names), None)
            p = df[pcol].to_numpy(float) if pcol else np.ones_like(x)
            out[bp] = (x, y, p)
    return out


def open_pose_preview(root, project_folder, video_path=None):
    import cv2
    pairs = _find_pairs(project_folder or "")
    if not pairs:
        messagebox.showinfo(
            "No tracked videos",
            "No videos with pose files (.h5) were found - run pose "
            "tracking first.", parent=root)
        return

    win = tk.Toplevel(root)
    win.title("Pose tracking preview")
    frm = ttk.Frame(win, padding=8)
    frm.pack(fill="both", expand=True)

    state = {"cap": None, "pose": None, "n": 0, "fps": 30.0,
             "frame": 0, "playing": False, "after": None, "photo": None,
             "setting": False}

    # ── top: session pick + options ────────────────────────────────────
    top = ttk.Frame(frm)
    top.pack(fill="x")
    ttk.Label(top, text="Session:").pack(side="left")
    names = [os.path.basename(v) for v, _h in pairs]
    sess_var = tk.StringVar()
    sess_cb = ttk.Combobox(top, textvariable=sess_var, state="readonly",
                           values=names, width=32)
    sess_cb.pack(side="left", padx=(4, 12))
    ttk.Label(top, text="Min likelihood:").pack(side="left")
    thr_var = tk.DoubleVar(value=0.6)
    ttk.Spinbox(top, from_=0.0, to=1.0, increment=0.05, width=5,
                textvariable=thr_var, format="%.2f",
                command=lambda: _show(state["frame"])).pack(side="left",
                                                            padx=(4, 12))
    lbl_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(top, text="Labels", variable=lbl_var,
                    command=lambda: _show(state["frame"])).pack(side="left")
    trail_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(top, text="Trails", variable=trail_var,
                    command=lambda: _show(state["frame"])).pack(side="left",
                                                                padx=(8, 0))

    canvas = tk.Label(frm, background="#111111")
    canvas.pack(fill="both", expand=True, pady=6)

    # ── bottom: transport ──────────────────────────────────────────────
    bot = ttk.Frame(frm)
    bot.pack(fill="x")
    play_btn = ttk.Button(bot, text="▶", width=4)
    play_btn.pack(side="left")
    pos_var = tk.IntVar(value=0)
    slider = ttk.Scale(bot, from_=0, to=100, orient="horizontal")
    slider.pack(side="left", fill="x", expand=True, padx=8)
    time_lbl = ttk.Label(bot, text="0:00 / 0:00", width=14)
    time_lbl.pack(side="left")
    ttk.Label(bot, text="Speed:").pack(side="left", padx=(8, 2))
    speed_var = tk.StringVar(value="1x")
    ttk.Combobox(bot, textvariable=speed_var, state="readonly", width=5,
                 values=["0.25x", "0.5x", "1x", "2x", "4x"]).pack(side="left")
    legend = ttk.Label(frm, text="", foreground="#666666", wraplength=760,
                       justify="left")
    legend.pack(fill="x", pady=(4, 0))

    def _load(idx):
        _stop()
        if state["cap"] is not None:
            state["cap"].release()
        vp, h5 = pairs[idx]
        state["cap"] = cv2.VideoCapture(vp)
        state["fps"] = state["cap"].get(cv2.CAP_PROP_FPS) or 30.0
        state["n"] = int(state["cap"].get(cv2.CAP_PROP_FRAME_COUNT))
        try:
            state["pose"] = _load_pose(h5)
        except Exception as e:
            messagebox.showerror("Pose file", f"Could not read {h5}:\n{e}",
                                 parent=win)
            state["pose"] = {}
        slider.configure(to=max(state["n"] - 1, 1))
        bits = []
        for i, bp in enumerate(state["pose"]):
            r, g, b = _COLORS[i % len(_COLORS)]
            bits.append(bp)
        legend.config(text="Bodyparts: " + ", ".join(bits))
        _show(0)

    def _draw(frame_idx, img):
        thr = float(thr_var.get())
        for i, (bp, (x, y, p)) in enumerate(state["pose"].items()):
            col = _COLORS[i % len(_COLORS)][::-1]      # BGR
            if trail_var.get():
                a = max(0, frame_idx - 30)
                for j in range(a, frame_idx):
                    if j < len(x) and p[j] >= thr and np.isfinite(x[j]):
                        cv2.circle(img, (int(x[j]), int(y[j])), 1, col, -1)
            if frame_idx < len(x) and p[frame_idx] >= thr \
                    and np.isfinite(x[frame_idx]):
                cx, cy = int(x[frame_idx]), int(y[frame_idx])
                cv2.circle(img, (cx, cy), 5, col, -1)
                cv2.circle(img, (cx, cy), 6, (255, 255, 255), 1)
                if lbl_var.get():
                    cv2.putText(img, bp, (cx + 7, cy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1,
                                cv2.LINE_AA)
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
        scale = min(900 / w, 620 / h, 1.0)
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

    def _close():
        _stop()
        if state["cap"] is not None:
            state["cap"].release()
        win.destroy()
    win.protocol("WM_DELETE_WINDOW", _close)

    # initial selection
    start = 0
    if video_path:
        vn = os.path.basename(str(video_path))
        start = next((i for i, (v, _h) in enumerate(pairs)
                      if os.path.basename(v) == vn), 0)
    sess_cb.current(start)
    _load(start)
