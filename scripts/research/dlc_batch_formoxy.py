"""
DLC analyze + filterpredictions for the 2605 FormOxy Left-paws cohort.

Same shuffle 9 / snapshot best-190 / batchsize 42 model as the rim and
THC cohorts (palmreader-500Mar25). Idempotent: scans the cohort
videos/ folder each call and only analyzes videos that lack a raw
.h5; only filters videos that lack a filtered.h5.

Run from DEEPLABCUT conda env:
  C:/Users/Gereau/anaconda3/envs/DEEPLABCUT/python.exe \
      scripts/research/dlc_batch_formoxy.py [--include-baseline] [--include-vehicle]

By default only `*_formalin.mp4` videos are processed. Flags:
  --include-baseline   also analyze formalin-cohort baseline recordings.
  --include-vehicle    also analyze the 2605_FormOxy_Veh_S<N>_{baseline,vehicle}
                       videos (the saline-injected control cohort).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

CONFIG_PATH = r"E:\RSDLC\2511_RSGNKK_Blackbox\config.yaml"
VIDEO_DIR = Path(r"E:\RSVIDS\Blackbox\2605_FormOxy_LeftPaws\videos")
SHUFFLE = 9
BATCHSIZE = 42
BATCHSIZE_FALLBACK = 32


def step(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def main() -> int:
    include_baseline = "--include-baseline" in sys.argv
    include_vehicle = "--include-vehicle" in sys.argv

    if include_baseline:
        videos = sorted(VIDEO_DIR.glob("2605_FormOxy_S*.mp4"))
    else:
        videos = sorted(VIDEO_DIR.glob("2605_FormOxy_S*_formalin.mp4"))
    if include_vehicle:
        veh_videos = sorted(VIDEO_DIR.glob("2605_FormOxy_Veh_S*.mp4"))
        existing = {v.name for v in videos}
        for v in veh_videos:
            if v.name not in existing:
                videos.append(v)
        videos = sorted(videos)
    # Exclude DLC artefacts that happen to end in .mp4 (eg labeled overlays)
    videos = [v for v in videos if "DLC_" not in v.name]
    if not videos:
        step(f"! No matching mp4s in {VIDEO_DIR}")
        return 1

    need_analyze, need_filter = [], []
    for v in videos:
        raw = [p for p in v.parent.glob(f"{v.stem}*shuffle{SHUFFLE}*.h5")
               if not p.name.endswith("_filtered.h5")]
        filt = list(v.parent.glob(f"{v.stem}*shuffle{SHUFFLE}*_filtered.h5"))
        if not raw:
            need_analyze.append(str(v)); need_filter.append(str(v))
            step(f"  pending analyze + filter: {v.name}")
        elif not filt:
            need_filter.append(str(v))
            step(f"  pending filter only: {v.name}")
        else:
            step(f"  skip (done): {v.name}")

    if not need_analyze and not need_filter:
        step("Nothing to do.")
        return 0

    step("Importing DLC...")
    import deeplabcut

    if need_analyze:
        step(f"analyze_videos: {len(need_analyze)} video(s) at batchsize={BATCHSIZE}")
        try:
            deeplabcut.analyze_videos(
                CONFIG_PATH, need_analyze,
                shuffle=SHUFFLE, videotype=".mp4",
                save_as_csv=False, batchsize=BATCHSIZE,
            )
        except (RuntimeError, MemoryError) as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda" in msg:
                step(f"  ! OOM at batchsize={BATCHSIZE}, retrying at {BATCHSIZE_FALLBACK}")
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                still_pending = [
                    v for v in need_analyze
                    if not list(Path(v).parent.glob(f"{Path(v).stem}*shuffle{SHUFFLE}*.h5"))
                ]
                deeplabcut.analyze_videos(
                    CONFIG_PATH, still_pending,
                    shuffle=SHUFFLE, videotype=".mp4",
                    save_as_csv=False, batchsize=BATCHSIZE_FALLBACK,
                )
            else:
                raise

    if need_filter:
        step(f"filterpredictions: {len(need_filter)} video(s)")
        deeplabcut.filterpredictions(
            CONFIG_PATH, need_filter,
            shuffle=SHUFFLE, videotype=".mp4",
        )

    step("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
