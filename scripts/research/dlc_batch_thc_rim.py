"""
DLC analyze + filterpredictions for the 260512 THC rimonabant cohort.

Idempotent: scans videos/ at each run, skips videos that already have
both raw and filtered .h5 files. Safe to re-run as new videos are
dropped into the folder.

Same model as previous THC withdrawal cohort: shuffle 9, snapshot best-190,
batchsize 32 (2080 Ti is power-capped above 32).

Run from the DEEPLABCUT conda env:
  C:/Users/Gereau/anaconda3/envs/DEEPLABCUT/python.exe ^
      scripts/research/dlc_batch_thc_rim.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

CONFIG_PATH = r"E:\RSDLC\2511_RSGNKK_Blackbox\config.yaml"
VIDEO_DIR = Path(r"E:\RSVIDS\Blackbox\260512_THC_Rim_Cohort\videos")
SHUFFLE = 9
# Tried batch=64 -- VRAM thrashed at 95% (10.7/11.3 GB), fps oscillated
# 5-25 it/s. Dropping to 42 for stable headroom on the 2080 Ti.
BATCHSIZE = 42
BATCHSIZE_FALLBACK = 32


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
            CONFIG_PATH,
            need_filter,
            shuffle=SHUFFLE,
            videotype=".mp4",
        )

    step("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
