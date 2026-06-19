"""
DLC analyze + filterpredictions for the 2604_DV_DSS Pdyn-ChR2 cohort.

Idempotent: scans videos/ at each run, skips videos that already have
both raw and filtered .h5 files. Safe to re-run as new cropped videos
are dropped into the folder.

Run from the DEEPLABCUT conda env:
  C:/Users/Gereau/anaconda3/envs/DEEPLABCUT/python.exe ^
      scripts/research/dlc_batch_dvv.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

CONFIG_PATH = r"E:\RSDLC\2511_RSGNKK_Blackbox\config.yaml"
VIDEO_DIR = Path(r"E:\RSVIDS\Blackbox\2604_DV_DSS\videos")
SHUFFLE = 9
BATCHSIZE = 32


def step(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def main() -> int:
    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    if not videos:
        step(f"! No mp4s in {VIDEO_DIR}")
        return 1

    need_analyze: list[str] = []
    need_filter: list[str] = []
    for v in videos:
        raw = [
            p for p in v.parent.glob(f"{v.stem}*shuffle{SHUFFLE}*.h5")
            if not p.name.endswith("_filtered.h5")
        ]
        filt = list(v.parent.glob(f"{v.stem}*shuffle{SHUFFLE}*_filtered.h5"))
        if not raw:
            need_analyze.append(str(v))
            need_filter.append(str(v))
            step(f"  pending analyze + filter: {v.name}")
        elif not filt:
            need_filter.append(str(v))
            step(f"  pending filter only: {v.name}")
        else:
            step(f"  skip (analyze + filter done): {v.name}")

    if not need_analyze and not need_filter:
        step("All videos already have DLC + filtered output. Nothing to do.")
        return 0

    step("Importing DLC...")
    import deeplabcut

    if need_analyze:
        step(f"analyze_videos: {len(need_analyze)} video(s), "
             f"shuffle={SHUFFLE} batchsize={BATCHSIZE}")
        deeplabcut.analyze_videos(
            CONFIG_PATH,
            need_analyze,
            shuffle=SHUFFLE,
            videotype=".mp4",
            save_as_csv=False,
            batchsize=BATCHSIZE,
        )

    if need_filter:
        step(f"filterpredictions: {len(need_filter)} video(s)")
        deeplabcut.filterpredictions(
            CONFIG_PATH,
            need_filter,
            shuffle=SHUFFLE,
            videotype=".mp4",
        )

    step("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
