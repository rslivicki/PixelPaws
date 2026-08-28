"""
dlc_run_dialog.py - the GUI "Analyze Videos (Pose Tracking)" flow.

Two phases:
    1. DLCRunDialog     - settings (device, batch size, video selection, chaining opts)
    2. DLCProgressDialog - live progress while inference runs

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
import shutil
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

def active_bundle_scorer():
    """h5 scorer string of the active DLC bundle, or None."""
    try:
        b = load_active_bundle() if load_active_bundle else None
        return b.dlc.h5_scorer_string if b else None
    except Exception:
        return None


def scan_project_videos(videos_dir, scorer: str = None) -> List[dict]:
    """Scan a project's videos/ folder for analyzable videos.

    Returns row dicts (index/name/path/duration/cal/status/select_default/
    h5_other_model) shared by the pose-tracking dialog and the Quick Start
    tab. When `scorer` (the active bundle's h5 scorer string) is given,
    videos whose pose files came from a different model are flagged."""
    videos_dir = Path(videos_dir)
    if not videos_dir.is_dir():
        return []
    out = []
    for idx, vp in enumerate(sorted(videos_dir.glob("*.mp4"))):
        if "DLC_" in vp.name or "_labeled" in vp.name:
            continue
        h5s = list(videos_dir.glob(f"{vp.stem}DLC*.h5"))
        already = bool(h5s)
        other_model = bool(h5s and scorer
                           and not any(scorer in h.name for h in h5s))
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
        cal = "-"
        try:
            from pawcapture_meta import read_calibration
            c = read_calibration(vp)
            if c and c.get("mm_per_pixel"):
                cal = f"{float(c['mm_per_pixel']):.4f} mm/px"
        except Exception:
            pass
        out.append({
            "index": idx,
            "name": vp.name,
            "path": vp,
            "duration": duration,
            "cal": cal,
            "status": ("analyzed (other model)" if other_model
                       else "already analyzed" if already
                       else "needs analysis"),
            "select_default": not already,
            "h5_other_model": other_model,
        })
    return out


class DLCRunDialog(tk.Toplevel):
    """Collect inference settings + video selection."""

    def __init__(self, root, project_folder: str, on_add_videos=None):
        super().__init__(root)
        self.root = root
        self._on_add_videos = on_add_videos
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

        self.title("Analyze Videos - Pose Tracking")
        self.resizable(True, True)
        self.minsize(660, 560)
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

        outer = ttk.Frame(self, padding=16)
        outer.pack(fill="both", expand=True)

        # Model header.
        hdr = ttk.Labelframe(outer, text="Model", padding=10)
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
        vid_frame = ttk.Labelframe(outer, text="Videos in project", padding=10)
        vid_frame.pack(fill="both", expand=True, pady=(0, 10))

        cols = ("name", "duration", "cal", "status")
        self.video_tree = ttk.Treeview(vid_frame, columns=cols, show="headings", height=10)
        self.video_tree.heading("name", text="Video")
        self.video_tree.heading("duration", text="Length")
        self.video_tree.heading("cal", text="Calibration")
        self.video_tree.heading("status", text="Status")
        self.video_tree.column("name", width=300, anchor="w")
        self.video_tree.column("duration", width=70, anchor="center")
        self.video_tree.column("cal", width=100, anchor="center")
        self.video_tree.column("status", width=140, anchor="w")
        vsb = ttk.Scrollbar(vid_frame, orient="vertical", command=self.video_tree.yview)
        self.video_tree.configure(yscrollcommand=vsb.set)
        self.video_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self.video_states: List[dict] = self._scan_videos()
        for v in self.video_states:
            self.video_tree.insert("", "end", iid=str(v["index"]),
                                   values=(v["name"], v["duration"],
                                           v.get("cal", "-"), v["status"]))
        self.video_tree.selection_set([
            str(v["index"]) for v in self.video_states if v["select_default"]
        ])

        sel_btns = ttk.Frame(outer)
        sel_btns.pack(fill="x", pady=(0, 10))
        ttk.Button(sel_btns, text="Select all", command=self._select_all).pack(side="left")
        ttk.Button(sel_btns, text="Select unanalyzed only",
                   command=self._select_unanalyzed).pack(side="left", padx=(6, 0))
        ttk.Button(sel_btns, text="Clear", command=self._select_clear).pack(side="left", padx=(6, 0))
        if self._on_add_videos is not None:
            ttk.Button(sel_btns, text="➕ Add videos…",
                       command=self._add_videos).pack(side="right")

        # Inference settings.
        cfg = ttk.Labelframe(outer, text="Inference settings", padding=10)
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
            cfg, text="Run the default classifier set (Core 8) after pose tracking",
            variable=self.chain_var,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(10, 0))

        self.features_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            cfg, text="Extract features after pose tracking (pre-caches for "
                      "training; predictions do this automatically)",
            variable=self.features_var,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(2, 0))

        self.gait_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            cfg, text="Run Gait & Limb analysis after the classifiers "
                      "(manuscript paw-contact gate)",
            variable=self.gait_var,
        ).grid(row=8, column=0, columnspan=3, sticky="w", pady=(6, 0))

        self.frames_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            cfg, text="Select more frames to label after pose tracking (active learning)",
            variable=self.frames_var,
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(2, 0))

        self._ffmpeg_ok = shutil.which("ffmpeg") is not None
        # Default ON (when ffmpeg exists): already-H.265 videos are skipped,
        # so the common case costs nothing and raw camera videos shrink
        # ~200-fold before tracking.
        self.transcode_var = tk.BooleanVar(value=self._ffmpeg_ok)
        tr_chk = ttk.Checkbutton(
            cfg, text="Transcode with the intake pipeline first (recommended for "
                      "raw camera videos)",
            variable=self.transcode_var,
        )
        tr_chk.grid(row=5, column=0, columnspan=3, sticky="w", pady=(10, 0))
        if self._ffmpeg_ok:
            _tr_hint = ("Runs the same encode as the transfer-portal pipeline "
                        "(H.265 CRF 23, audio dropped, calibration tags "
                        "preserved) before tracking. CRF 23 shrinks files "
                        "~200-fold at no practical cost: keypoints shift "
                        "~0.5 px and behavioral output is unchanged (validated "
                        "in the manuscript). The transcode keeps the video name; "
                        "the original moves to videos/raw/. Videos already "
                        "H.265 are skipped. Encoding is slow - roughly "
                        "real-time per video.")
        else:
            _tr_hint = ("ffmpeg was not found on PATH; install ffmpeg to enable "
                        "transcoding.")
            tr_chk.config(state="disabled")
        ttk.Label(cfg, text=_tr_hint, foreground="#777", wraplength=600,
                  justify="left").grid(row=6, column=0, columnspan=3,
                                       sticky="w", padx=(24, 0), pady=(2, 0))

        self.cpu_warn = ttk.Label(cfg, text="", foreground="#a60")
        self.cpu_warn.grid(row=7, column=0, columnspan=3, sticky="w", pady=(6, 0))
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
        return scan_project_videos(self.videos_dir,
                                   scorer=active_bundle_scorer())



    def _add_videos(self):
        """Import more videos via the app's add-videos flow, then rescan."""
        self.grab_release()

        def _done():
            try:
                if not self.winfo_exists():
                    return
                self.video_tree.delete(*self.video_tree.get_children())
                self.video_states = self._scan_videos()
                for v in self.video_states:
                    self.video_tree.insert("", "end", iid=str(v["index"]),
                                           values=(v["name"], v["duration"],
                                                   v.get("cal", "-"),
                                                   v["status"]))
                self.video_tree.selection_set([
                    str(v["index"]) for v in self.video_states if v["select_default"]
                ])
            finally:
                try:
                    self.grab_set()
                except Exception:
                    pass

        try:
            self._on_add_videos(on_done=_done)
        except Exception:
            _done()

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
                text="⚠ CPU mode is 10-30× slower than GPU.")
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
            "run_gait": bool(self.gait_var.get()),
            "select_frames": bool(self.frames_var.get()),
            "extract_features": bool(self.features_var.get()),
            "transcode": bool(self.transcode_var.get()) and self._ffmpeg_ok,
            "bundle": self.bundle,
        }
        self.grab_release()
        self.destroy()

    def _on_cancel(self):
        self.result = None
        self.grab_release()
        self.destroy()

    @classmethod
    def show(cls, root, project_folder: str, on_add_videos=None) -> Optional[dict]:
        dlg = cls(root, project_folder, on_add_videos=on_add_videos)
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
        self.current_label = ttk.Label(outer, text="-")
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

    @staticmethod
    def _frame_count(path) -> int:
        try:
            import cv2
            cap = cv2.VideoCapture(str(path))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
            cap.release()
            return max(n, 0)
        except Exception:
            return 0

    @staticmethod
    def _video_codec(path) -> str:
        if shutil.which("ffprobe") is None:
            return ""
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=codec_name",
                 "-of", "default=nk=1:nw=1", str(path)],
                capture_output=True, text=True, timeout=30)
            return r.stdout.strip().lower()
        except Exception:
            return ""

    def _transcode_videos(self, videos):
        """The intake-pipeline encode (pp_pipeline.build_transcode_cmd) applied
        in place: the transcoded file keeps the video's name so downstream
        pairing is unchanged; the original moves to videos/raw/. Returns the
        (possibly updated) video list, or None if cancelled."""
        try:
            from pp_pipeline import build_transcode_cmd, transcode_output_ok
        except Exception as e:
            self._msg_queue.put(("log", f"✗ intake pipeline unavailable ({e!r}) - "
                                        "skipping transcode, tracking originals"))
            return videos
        import time as _time
        out = []
        n = len(videos)
        self._msg_queue.put(("log", f"Transcoding with the intake encode "
                                    f"(H.265 CRF 23) - {n} video(s)"))
        for i, vp in enumerate(videos, 1):
            if self._cancelled:
                return None
            src = Path(vp)
            codec = self._video_codec(src)
            if codec in ("hevc", "h265"):
                self._msg_queue.put(("log", f"↷ {src.name}: already H.265, skipping"))
                out.append(str(src))
                continue
            raw_dir = src.parent / "raw"
            raw = raw_dir / src.name
            if raw.exists():
                self._msg_queue.put(("log", f"↷ {src.name}: raw copy already in "
                                            "videos/raw/, assuming transcoded"))
                out.append(str(src))
                continue
            total = self._frame_count(src)
            self._msg_queue.put(("video_start", (i, n, f"transcode {src.name}", total)))
            raw_dir.mkdir(exist_ok=True)
            try:
                src.rename(raw)
            except OSError as e:
                self._msg_queue.put(("log", f"✗ {src.name}: could not move to "
                                            f"videos/raw/ ({e}) - tracking original"))
                out.append(str(src))
                continue
            dst = src                      # same name, same folder
            cmd = build_transcode_cmd(raw, dst, prog="pipe:1")
            try:
                self.proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1)
            except Exception as e:
                self._msg_queue.put(("log", f"✗ ffmpeg launch failed: {e!r}"))
                raw.rename(src)
                out.append(str(src))
                continue
            t0 = _time.time()
            for line in self.proc.stdout:
                line = line.strip()
                if line.startswith("frame="):
                    try:
                        done = int(line.split("=", 1)[1])
                    except ValueError:
                        continue
                    fps = done / max(_time.time() - t0, 1e-3)
                    if total:
                        self._msg_queue.put(("frames", (min(done, total), total, fps)))
            self.proc.wait()
            ok = (not self._cancelled and self.proc.returncode == 0
                  and transcode_output_ok(dst))
            if ok:
                mb_in = raw.stat().st_size / 1e6
                mb_out = dst.stat().st_size / 1e6
                self._msg_queue.put(
                    ("log", f"✓ {src.name} transcoded "
                            f"({mb_in:.0f} → {mb_out:.0f} MB, "
                            f"{(_time.time() - t0) / 60:.1f} min); "
                            f"original kept in videos/raw/"))
                # PawCapture spatial calibration must survive the encode --
                # the mm-per-pixel path depends on it.
                try:
                    from pawcapture_meta import read_calibration
                    c_src = read_calibration(raw)
                    if c_src and c_src.get("mm_per_pixel"):
                        c_dst = read_calibration(dst)
                        if c_dst and (c_dst.get("mm_per_pixel")
                                      == c_src.get("mm_per_pixel")):
                            self._msg_queue.put(
                                ("log", f"   calibration preserved "
                                        f"({c_dst['mm_per_pixel']:.4f} mm/px)"))
                        else:
                            self._msg_queue.put(
                                ("log", f"   ⚠ calibration tags did NOT "
                                        f"survive the transcode of {src.name} "
                                        f"-- distances will fall back to pixels"))
                except Exception:
                    pass
                out.append(str(dst))
                continue
            # failure or cancel: remove the partial output, restore the original
            try:
                if dst.exists():
                    dst.unlink()
                raw.rename(src)
            except OSError as e:
                self._msg_queue.put(("log", f"✗ {src.name}: restore failed ({e}) - "
                                            f"original is in videos/raw/"))
            if self._cancelled:
                return None
            self._msg_queue.put(("log", f"✗ {src.name}: transcode failed - "
                                        "tracking the original"))
            out.append(str(src))
        return out

    def _kill_proc_tree(self):
        """Force-kill the current subprocess and all its children."""
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        try:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True, timeout=10)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass

    def _worker(self):
        s = self.settings
        videos = [str(p) for p in s["videos"]]
        if s.get("transcode"):
            videos = self._transcode_videos(videos)
            if videos is None:        # cancelled or unrecoverable failure
                self._msg_queue.put(("all_done", None))
                return
            self.proc = None
        if self._cancelled:
            self._msg_queue.put(("all_done", None))
            return
        # The Quick Start wizard can ask for transcode alone ("pose": False)
        # or for pose on a subset of the transcoded videos ("pose_videos").
        if not s.get("pose", True):
            self._msg_queue.put(("log", "Pose tracking not requested - "
                                        "finished after transcode."))
            self._msg_queue.put(("all_done", None))
            return
        pose_videos = s.get("pose_videos")
        if pose_videos is not None:
            keep = {os.path.normcase(os.path.abspath(str(p)))
                    for p in pose_videos}
            videos = [v for v in videos
                      if os.path.normcase(os.path.abspath(v)) in keep]
            if not videos:
                self._msg_queue.put(("log", "No videos need pose tracking."))
                self._msg_queue.put(("all_done", None))
                return
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

        if self._cancelled:
            self._kill_proc_tree()

        for line in self.proc.stdout:
            if self._cancelled:
                self._kill_proc_tree()
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
                    self.current_stats.config(text=f"{nframes} frames - starting…")
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
            self._kill_proc_tree()
            self._append_log("Cancelled by user.")


def run_dlc_flow(root, project_folder: str,
                 on_predictions_done: Optional[Callable[[], None]] = None,
                 on_select_frames: Optional[Callable[[], None]] = None,
                 on_extract_features: Optional[Callable[[], None]] = None,
                 on_add_videos=None,
                 on_gait: Optional[Callable[[bool], None]] = None) -> None:
    """Top-level entry: settings dialog -> progress dialog -> optional chaining."""
    settings = DLCRunDialog.show(root, project_folder, on_add_videos=on_add_videos)
    if not settings:
        return

    def _after(h5_paths: List[Path]):
        # Batch prediction extracts and caches features itself, so a separate
        # feature-extraction pass alongside it is redundant (and could race on
        # the same cache files). Only honour the FX chain when predictions are
        # not also queued.
        if (settings.get("extract_features") and h5_paths and on_extract_features
                and not settings.get("auto_predict")):
            on_extract_features()
        # Gait chains AFTER the classifier batch when one is queued (its
        # licking exclusion needs the predictions); otherwise it runs now.
        if settings.get("run_gait") and h5_paths and on_gait:
            on_gait(not settings.get("auto_predict"))
        if settings.get("auto_predict") and h5_paths and on_predictions_done:
            on_predictions_done()
        if settings.get("select_frames") and on_select_frames:
            on_select_frames()

    DLCProgressDialog(root, settings, on_complete=_after)
