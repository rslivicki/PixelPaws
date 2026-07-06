"""
dlc_run_dialog.py — the GUI "Analyze Videos (Pose Tracking)" flow.

Two phases:
    1. DLCRunDialog     — settings (device, batch size, video selection, chaining opts)
    2. DLCProgressDialog — live progress while inference runs

Runs in the GUI's Python (which may lack deeplabcut/torch), so the actual inference
is executed by shelling out to pipeline/dlc_analyze.py under
pp_config.resolve_pose_python() and parsing its stdout progress markers. Model
metadata + bundle discovery use dlc_inference directly (those modules don't import
torch at module load).

On completion the caller can chain into behaviour predictions and/or the
"Extract Problem Frames" (select-more-frames) tool.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, List, Optional

import tkinter as tk
from tkinter import messagebox

try:
    import ttkbootstrap as ttk
    from ttkbootstrap.constants import *  # noqa: F401,F403
except ImportError:
    from tkinter import ttk

_REPO = Path(__file__).resolve().parent
if str(_REPO / "pipeline") not in sys.path:
    sys.path.insert(0, str(_REPO / "pipeline"))

try:
    from dlc_inference import load_active_bundle
except Exception:  # pragma: no cover - dlc_inference must be importable
    load_active_bundle = None  # type: ignore


def _resolve_pose_python() -> str:
    try:
        from pp_config import resolve_pose_python
        return resolve_pose_python()
    except Exception:
        return sys.executable


def _probe_providers() -> List[dict]:
    """Ask the DLC-capable Python what devices are available (the GUI Python has no
    torch). Falls back to a GPU/CPU pair if the probe can't run."""
    py = _resolve_pose_python()
    script = str(_REPO / "pipeline" / "dlc_analyze.py")
    try:
        out = subprocess.run(
            [py, script, "--probe"], capture_output=True, text=True, timeout=120,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        for line in out.stdout.splitlines():
            if line.startswith("PROVIDERS "):
                return json.loads(line[len("PROVIDERS "):])
    except Exception:
        pass
    return [
        {"name": "cuda", "display": "NVIDIA CUDA (if available)", "is_gpu": True,
         "vram_gb": None, "suggested_batch": 16},
        {"name": "cpu", "display": "CPU (slow)", "is_gpu": False,
         "vram_gb": None, "suggested_batch": 4},
    ]


# --------------------------------------------------------------------------- #
# Settings dialog
# --------------------------------------------------------------------------- #

class DLCRunDialog(tk.Toplevel):
    """Collect inference settings + video selection."""

    def __init__(self, root, project_folder: str):
        super().__init__(root)
        self.root = root
        self.project_folder = Path(project_folder)
        self.videos_dir = self.project_folder / "videos"
        if not self.videos_dir.is_dir():
            # Some projects keep videos at the project root.
            self.videos_dir = self.project_folder
        self.bundle = load_active_bundle() if load_active_bundle else None
        if self.bundle is None:
            messagebox.showerror(
                "No model bundle",
                "No DLC model bundle is installed. The pose-tracking model bundle "
                "could not be found or loaded.",
                parent=root,
            )
            self.result = None
            self.destroy()
            return

        self.providers: List[dict] = _probe_providers()
        self.result: Optional[dict] = None

        self.title("Analyze Videos — Pose Tracking")
        self.resizable(True, True)
        self.minsize(660, 560)
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        outer = ttk.Frame(self, padding=16)
        outer.pack(fill="both", expand=True)

        # Model header.
        hdr = ttk.LabelFrame(outer, text="Model", padding=10)
        hdr.pack(fill="x", pady=(0, 10))
        ttk.Label(hdr, text=self.bundle.display_name,
                  font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(hdr, text=(
            f"{len(self.bundle.dlc.keypoints)} keypoints · "
            f"{self.bundle.dlc.net_type} · "
            f"shuffle {self.bundle.dlc.shuffle} · "
            f"{self.bundle.dlc.snapshot}"
        )).pack(anchor="w")

        # Video list.
        vid_frame = ttk.LabelFrame(outer, text="Videos in project", padding=10)
        vid_frame.pack(fill="both", expand=True, pady=(0, 10))

        cols = ("name", "duration", "status")
        self.video_tree = ttk.Treeview(vid_frame, columns=cols, show="headings", height=10)
        self.video_tree.heading("name", text="Video")
        self.video_tree.heading("duration", text="Length")
        self.video_tree.heading("status", text="Status")
        self.video_tree.column("name", width=340, anchor="w")
        self.video_tree.column("duration", width=90, anchor="center")
        self.video_tree.column("status", width=150, anchor="w")
        vsb = ttk.Scrollbar(vid_frame, orient="vertical", command=self.video_tree.yview)
        self.video_tree.configure(yscrollcommand=vsb.set)
        self.video_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self.video_states: List[dict] = self._scan_videos()
        for v in self.video_states:
            self.video_tree.insert("", "end", iid=str(v["index"]),
                                   values=(v["name"], v["duration"], v["status"]))
        self.video_tree.selection_set([
            str(v["index"]) for v in self.video_states if v["select_default"]
        ])

        sel_btns = ttk.Frame(outer)
        sel_btns.pack(fill="x", pady=(0, 10))
        ttk.Button(sel_btns, text="Select all", command=self._select_all).pack(side="left")
        ttk.Button(sel_btns, text="Select unanalyzed only",
                   command=self._select_unanalyzed).pack(side="left", padx=(6, 0))
        ttk.Button(sel_btns, text="Clear", command=self._select_clear).pack(side="left", padx=(6, 0))

        # Inference settings.
        cfg = ttk.LabelFrame(outer, text="Inference settings", padding=10)
        cfg.pack(fill="x", pady=(0, 10))

        ttk.Label(cfg, text="Device:").grid(row=0, column=0, sticky="w")
        self.provider_var = tk.StringVar(value=self.providers[0]["display"])
        provider_combo = ttk.Combobox(
            cfg, textvariable=self.provider_var,
            values=[p["display"] for p in self.providers],
            state="readonly", width=46,
        )
        provider_combo.grid(row=0, column=1, sticky="w", padx=(6, 0))
        provider_combo.bind("<<ComboboxSelected>>", self._on_provider_change)

        ttk.Label(cfg, text="Batch size:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.batch_var = tk.IntVar(value=self.providers[0].get("suggested_batch", 16))
        ttk.Spinbox(cfg, from_=1, to=64, textvariable=self.batch_var, width=6).grid(
            row=1, column=1, sticky="w", padx=(6, 0), pady=(6, 0))
        self.batch_hint = ttk.Label(cfg, text=self._batch_hint_for(self.providers[0]),
                                    foreground="#777")
        self.batch_hint.grid(row=1, column=2, sticky="w", padx=(12, 0), pady=(6, 0))

        self.chain_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            cfg, text="Run behavior predictions after pose tracking",
            variable=self.chain_var,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(10, 0))

        self.frames_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            cfg, text="Select more frames to label after pose tracking (active learning)",
            variable=self.frames_var,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(2, 0))

        self.cpu_warn = ttk.Label(cfg, text="", foreground="#a60")
        self.cpu_warn.grid(row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self._refresh_cpu_warning()

        # Buttons.
        btns = ttk.Frame(outer)
        btns.pack(fill="x")
        ttk.Button(btns, text="Cancel", command=self._on_cancel).pack(side="right", padx=(6, 0))
        ttk.Button(btns, text="Start", command=self._on_start).pack(side="right")

        self.transient(root)
        self.grab_set()
        self.update_idletasks()
        w, h = max(720, self.winfo_reqwidth()), max(600, self.winfo_reqheight())
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")
        self.focus_force()

    def _scan_videos(self) -> List[dict]:
        if not self.videos_dir.is_dir():
            return []
        out = []
        for idx, vp in enumerate(sorted(self.videos_dir.glob("*.mp4"))):
            if "DLC_" in vp.name or "_labeled" in vp.name:
                continue
            already = any(self.videos_dir.glob(f"{vp.stem}*DLC*.h5"))
            duration = "?"
            try:
                import cv2
                cap = cv2.VideoCapture(str(vp))
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS) or 0
                    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if fps > 0:
                        secs = n / fps
                        duration = f"{int(secs // 60)}:{int(secs % 60):02d}"
                cap.release()
            except Exception:
                pass
            out.append({
                "index": idx,
                "name": vp.name,
                "path": vp,
                "duration": duration,
                "status": "already analyzed" if already else "needs analysis",
                "select_default": not already,
            })
        return out

    def _selected_videos(self) -> List[Path]:
        sel = set(int(i) for i in self.video_tree.selection())
        return [v["path"] for v in self.video_states if v["index"] in sel]

    def _select_all(self):
        self.video_tree.selection_set([str(v["index"]) for v in self.video_states])

    def _select_unanalyzed(self):
        self.video_tree.selection_set([
            str(v["index"]) for v in self.video_states if v["select_default"]
        ])

    def _select_clear(self):
        self.video_tree.selection_remove(self.video_tree.selection())

    def _provider_by_display(self, disp: str) -> dict:
        for p in self.providers:
            if p["display"] == disp:
                return p
        return self.providers[0]

    def _on_provider_change(self, _evt=None):
        prov = self._provider_by_display(self.provider_var.get())
        self.batch_var.set(prov.get("suggested_batch", 16))
        self.batch_hint.config(text=self._batch_hint_for(prov))
        self._refresh_cpu_warning()

    def _batch_hint_for(self, prov: dict) -> str:
        if prov.get("vram_gb"):
            return f"Suggested for {prov['vram_gb']:.0f} GB VRAM; lower if you hit CUDA OOM."
        return "Lower this if you run into out-of-memory errors."

    def _refresh_cpu_warning(self):
        prov = self._provider_by_display(self.provider_var.get())
        if not prov.get("is_gpu"):
            self.cpu_warn.config(
                text="⚠ CPU mode is 10–30× slower than GPU.")
        else:
            self.cpu_warn.config(text="")

    def _on_start(self):
        videos = self._selected_videos()
        if not videos:
            messagebox.showwarning("No videos", "Select at least one video.", parent=self)
            return
        prov = self._provider_by_display(self.provider_var.get())
        self.result = {
            "videos": videos,
            "device": prov["name"],
            "batch_size": int(self.batch_var.get()),
            "auto_predict": bool(self.chain_var.get()),
            "select_frames": bool(self.frames_var.get()),
            "bundle": self.bundle,
        }
        self.grab_release()
        self.destroy()

    def _on_cancel(self):
        self.result = None
        self.grab_release()
        self.destroy()

    @classmethod
    def show(cls, root, project_folder: str) -> Optional[dict]:
        dlg = cls(root, project_folder)
        if dlg.result is None and not dlg.winfo_exists():
            return None
        root.wait_window(dlg)
        return dlg.result


# --------------------------------------------------------------------------- #
# Progress dialog
# --------------------------------------------------------------------------- #

class DLCProgressDialog(tk.Toplevel):
    """Live progress while inference runs in a shelled-out DLC process."""

    def __init__(self, root, settings: dict,
                 on_complete: Optional[Callable[[List[Path]], None]] = None):
        super().__init__(root)
        self.root = root
        self.settings = settings
        self.on_complete = on_complete
        self._msg_queue: queue.Queue = queue.Queue()
        self.h5_paths: List[Path] = []
        self.proc: Optional[subprocess.Popen] = None
        self._cancelled = False

        self.title("Running pose tracking")
        self.resizable(True, True)
        self.minsize(560, 420)
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        outer = ttk.Frame(self, padding=16)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="Overall progress", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.overall_bar = ttk.Progressbar(outer, mode="determinate",
                                           maximum=len(settings["videos"]))
        self.overall_bar.pack(fill="x", pady=(2, 4))
        self.overall_label = ttk.Label(outer, text=f"0 / {len(settings['videos'])} videos")
        self.overall_label.pack(anchor="w")

        ttk.Separator(outer).pack(fill="x", pady=10)

        ttk.Label(outer, text="Current video", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.current_label = ttk.Label(outer, text="—")
        self.current_label.pack(anchor="w")
        self.current_bar = ttk.Progressbar(outer, mode="determinate", maximum=100)
        self.current_bar.pack(fill="x", pady=(2, 4))
        self.current_stats = ttk.Label(outer, text="starting…")
        self.current_stats.pack(anchor="w")

        ttk.Separator(outer).pack(fill="x", pady=10)

        ttk.Label(outer, text="Log").pack(anchor="w")
        log_frame = ttk.Frame(outer)
        log_frame.pack(fill="both", expand=True, pady=(2, 8))
        self.log_text = tk.Text(log_frame, height=8, wrap="word",
                                font=("Consolas", 9), state="disabled")
        log_sb = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_sb.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_sb.pack(side="right", fill="y")

        btns = ttk.Frame(outer)
        btns.pack(fill="x")
        self.cancel_btn = ttk.Button(btns, text="Cancel", command=self._on_cancel)
        self.cancel_btn.pack(side="right")

        self.transient(root)
        self.grab_set()
        self.update_idletasks()
        w, h = max(640, self.winfo_reqwidth()), max(480, self.winfo_reqheight())
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()
        self.after(100, self._drain_queue)

    def _worker(self):
        s = self.settings
        videos = [str(p) for p in s["videos"]]
        bundle = s["bundle"]
        py = _resolve_pose_python()
        script = str(_REPO / "pipeline" / "dlc_analyze.py")
        cmd = [py, script, "--videos", json.dumps(videos),
               "--batch", str(s["batch_size"]), "--device", s["device"],
               "--bundle-id", getattr(bundle, "bundle_id", "pixelpaws_v1")]
        self._msg_queue.put(("log", f"Device {s['device']} · batch {s['batch_size']} · "
                                    f"{len(videos)} video(s)"))
        self._msg_queue.put(("log", "python: " + os.path.basename(py)))
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                bufsize=1, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
        except Exception as e:
            self._msg_queue.put(("log", f"failed to launch inference: {e!r}"))
            self._msg_queue.put(("all_done", None))
            return

        for line in self.proc.stdout:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("VIDEO_START "):
                parts = line.split(" ", 4)
                # VIDEO_START i n name total
                try:
                    i, n = parts[1], parts[2]
                    rest = parts[3] if len(parts) > 3 else ""
                    name = rest
                    total = parts[4] if len(parts) > 4 else "0"
                    # name may contain spaces: recover from the trailing int
                    tokens = line.split(" ")
                    total = tokens[-1]
                    name = " ".join(tokens[3:-1])
                    self._msg_queue.put(("video_start", (int(i), int(n), name, int(total))))
                except Exception:
                    self._msg_queue.put(("log", line))
            elif line.startswith("FRAMES "):
                try:
                    _, done, total, fps = line.split(" ")
                    self._msg_queue.put(("frames", (int(done), int(total), float(fps))))
                except Exception:
                    pass
            elif line.startswith("VIDEO_DONE "):
                payload = line[len("VIDEO_DONE "):]
                name, _, h5 = payload.partition(" :: ")
                self._msg_queue.put(("video_done", (name, h5)))
            elif line.startswith("VIDEO_FAIL "):
                payload = line[len("VIDEO_FAIL "):]
                name, _, err = payload.partition(" :: ")
                self._msg_queue.put(("log", f"✗ {name}: {err}"))
            elif line.startswith("ALL_DONE "):
                self._msg_queue.put(("log", line.replace("ALL_DONE ", "Finished: ") + " succeeded"))
            elif line.startswith("TOTAL "):
                pass
            else:
                self._msg_queue.put(("log", line))
        self.proc.wait()
        self._msg_queue.put(("all_done", None))

    def _drain_queue(self):
        try:
            while True:
                kind, payload = self._msg_queue.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "video_start":
                    i, total, name, nframes = payload
                    self.current_label.config(text=f"({i}/{total}) {name}")
                    self.current_bar.config(value=0, maximum=100)
                    self.current_stats.config(text=f"{nframes} frames — starting…")
                    self._append_log(f"⟳ ({i}/{total}) {name}")
                elif kind == "frames":
                    done, total, fps = payload
                    pct = (done / max(total, 1)) * 100
                    self.current_bar.config(value=pct)
                    remaining = (total - done) / max(fps, 1e-3)
                    self.current_stats.config(
                        text=f"{done}/{total}  {fps:.0f} fps  ETA "
                             f"{int(remaining // 60)}m {int(remaining % 60)}s")
                elif kind == "video_done":
                    name, h5 = payload
                    if h5:
                        self.h5_paths.append(Path(h5))
                    self.current_bar.config(value=100)
                    cur = self.overall_bar["value"] + 1
                    self.overall_bar.config(value=cur)
                    self.overall_label.config(
                        text=f"{int(cur)} / {len(self.settings['videos'])} videos")
                    self._append_log(f"✓ {name}")
                elif kind == "all_done":
                    self.cancel_btn.config(text="Close")
                    self.current_stats.config(text="done")
                    if self.on_complete and not self._cancelled:
                        try:
                            self.on_complete(self.h5_paths)
                        except Exception as e:
                            self._append_log(f"on_complete error: {e!r}")
                    return
        except queue.Empty:
            pass
        self.after(100, self._drain_queue)

    def _append_log(self, line: str):
        self.log_text.config(state="normal")
        self.log_text.insert("end", line + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _on_cancel(self):
        if self.cancel_btn["text"] == "Close":
            self.grab_release()
            self.destroy()
            return
        if messagebox.askyesno("Cancel?",
                               "Cancel pose tracking? The in-progress video will stop.",
                               parent=self):
            self._cancelled = True
            if self.proc and self.proc.poll() is None:
                try:
                    self.proc.terminate()
                except Exception:
                    pass
            self._append_log("Cancelled by user.")


def run_dlc_flow(root, project_folder: str,
                 on_predictions_done: Optional[Callable[[], None]] = None,
                 on_select_frames: Optional[Callable[[], None]] = None) -> None:
    """Top-level entry: settings dialog -> progress dialog -> optional chaining."""
    settings = DLCRunDialog.show(root, project_folder)
    if not settings:
        return

    def _after(h5_paths: List[Path]):
        if settings.get("auto_predict") and h5_paths and on_predictions_done:
            on_predictions_done()
        if settings.get("select_frames") and on_select_frames:
            on_select_frames()

    DLCProgressDialog(root, settings, on_complete=_after)
