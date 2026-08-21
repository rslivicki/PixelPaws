"""DLC pose-inference backend for the GUI's "Analyze Videos (Pose Tracking)" tool.

RUN UNDER A DLC-CAPABLE PYTHON (the GUI shells out to this via
pp_config.resolve_pose_python(), because the GUI's own interpreter may not have
deeplabcut/torch — see the dlc-env-split note). Two modes:

  --probe            print `PROVIDERS <json>` (device list + VRAM + suggested batch),
                     used to populate the dialog's device dropdown.
  <analyze, default> run the active bundle's model on the given videos and write a
                     DLC-format .h5 next to each, streaming progress markers on stdout.

Progress markers (one per line, parsed by dlc_run_dialog.DLCProgressDialog):
  TOTAL <n>
  VIDEO_START <i> <n> <name> <total_frames>
  FRAMES <done> <total> <fps>
  VIDEO_DONE <name> :: <h5_path>
  VIDEO_FAIL <name> :: <error>
  ALL_DONE <ok> <total>

Inference goes through dlc_inference.run_dlc_on_video (bundle = snapshot.pt +
pytorch_config.yaml, no DLC project tree needed). If the fast runner raises (e.g. a
DLC-version API drift), it falls back to deeplabcut.analyze_videos against
pp_config.POSE_DLC_CONFIG for that one video (dev machines, where the project exists).

Usage:
  dlc_analyze.py --probe
  dlc_analyze.py --videos '["C:/a.mp4","C:/b.mp4"]' [--bundle-id pixelpaws_v1]
                 [--batch 16] [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# dlc_inference + pp_config live at the repo root / beside this file.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_HERE))


def _emit(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _probe() -> int:
    try:
        from dlc_inference import probe_providers
        provs = [
            {
                "name": p.name,
                "display": str(p),
                "is_gpu": p.is_gpu,
                "vram_gb": p.vram_gb,
                "suggested_batch": p.suggested_batch,
            }
            for p in probe_providers()
        ]
    except Exception as e:
        provs = [{"name": "cpu", "display": f"CPU (probe failed: {e})",
                  "is_gpu": False, "vram_gb": None, "suggested_batch": 4}]
    _emit("PROVIDERS " + json.dumps(provs))
    return 0


def _analyze(videos, bundle_id: str, batch: int, device: str) -> int:
    from dlc_inference import load_bundle, load_active_bundle

    bundle = None
    try:
        bundle = load_bundle(bundle_id) if bundle_id else load_active_bundle()
    except Exception:
        bundle = load_active_bundle()
    if bundle is None:
        _emit("VIDEO_FAIL - :: no active model bundle installed")
        _emit("ALL_DONE 0 %d" % len(videos))
        return 2

    _emit(f"TOTAL {len(videos)}")
    ok = 0
    for i, vp in enumerate(videos, 1):
        vp = Path(vp)
        name = vp.name
        try:
            from dlc_inference.video_frame_iter import count_video_frames
            total_frames = count_video_frames(str(vp))
        except Exception:
            total_frames = 0
        _emit(f"VIDEO_START {i} {len(videos)} {name} {total_frames}")

        last = [0.0]

        def _cb(done, total, fps, _last=last):
            now = time.time()
            if now - _last[0] >= 0.5 or done >= total:
                _last[0] = now
                _emit(f"FRAMES {done} {total} {fps:.1f}")

        try:
            from dlc_inference import run_dlc_on_video
            stats = run_dlc_on_video(
                vp, bundle=bundle, batch_size=batch, device=device,
                progress_cb=_cb, overwrite=True,
            )
            _emit(f"VIDEO_DONE {name} :: {stats.out_h5_path}")
            ok += 1
        except Exception as e:
            _emit(f"  fast-runner failed on {name}: {e!r}; trying deeplabcut.analyze_videos")
            h5 = _fallback_analyze(vp, device)
            if h5:
                _emit(f"VIDEO_DONE {name} :: {h5}")
                ok += 1
            else:
                _emit(f"VIDEO_FAIL {name} :: {e}")
                traceback.print_exc()

    _emit(f"ALL_DONE {ok} {len(videos)}")
    return 0 if ok == len(videos) else 1


def _fallback_analyze(vp: Path, device: str):
    """Last-resort path: run DLC's own analyze_videos against POSE_DLC_CONFIG (needs the
    project tree; dev machines only). Returns the produced .h5 path or None."""
    try:
        from pp_config import POSE_DLC_CONFIG, POSE_SHUFFLE, POSE_DLC_BATCH
        import deeplabcut
        kw = dict(shuffle=POSE_SHUFFLE, videotype=".mp4",
                  save_as_csv=False, batchsize=POSE_DLC_BATCH)
        try:
            deeplabcut.analyze_videos(POSE_DLC_CONFIG, [str(vp)], **kw)
        except TypeError:
            kw.pop("batchsize", None)
            deeplabcut.analyze_videos(POSE_DLC_CONFIG, [str(vp)], **kw)
        cands = [p for p in vp.parent.glob(f"{vp.stem}DLC*shuffle{POSE_SHUFFLE}*.h5")   # "DLC" boundary: S1 must not match S10
                 if not p.name.endswith("_filtered.h5")]
        return str(sorted(cands)[-1]) if cands else None
    except Exception:
        traceback.print_exc()
        return None


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--videos", default="[]", help="JSON list of video paths")
    ap.add_argument("--bundle-id", default="pixelpaws_v1")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)

    if a.probe:
        return _probe()
    videos = json.loads(a.videos)
    if not videos:
        _emit("ALL_DONE 0 0")
        return 0
    return _analyze(videos, a.bundle_id, a.batch, a.device)


if __name__ == "__main__":
    raise SystemExit(main())
