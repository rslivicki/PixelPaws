"""
Run DLC shuffle 9 on every video in
E:\\RSVIDS\\Blackbox\\260506_RS_THC_Withdrawal\\Videos with batched inference.

Same model + shuffle as the THC1_Baseline test (snapshot best-190).
Default DLC batchsize is 32 — bumped to 128 here since the 2080 Ti
was clearly underutilised in the prior run (variable 25-90 fps
throughput on a constant-load model = CPU/IO stalls).

All videos passed in a single analyze_videos call so the model
weights load once instead of 12 times.

Run from the DEEPLABCUT conda env:
  C:/Users/Gereau/anaconda3/envs/DEEPLABCUT/python.exe scripts/research/dlc_batch_thc_withdrawal.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

CONFIG_PATH = r'E:\RSDLC\2511_RSGNKK_Blackbox\config.yaml'
VIDEO_DIR = Path(r'E:\RSVIDS\Blackbox\260506_RS_THC_Withdrawal\Videos')
SHUFFLE = 9
# batchsize=128 saturated VRAM at 95.6% and ran into memory thrashing.
# batchsize=64 was OK for the first 2 videos in a run but degraded to
# 207 min/video by video 5 (likely VRAM fragmentation + power-cap
# throttling). Dropping to 32 trades peak throughput for stability.
BATCHSIZE = 32


def step(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def main() -> int:
    videos = sorted(VIDEO_DIR.glob('*.mp4'))
    if not videos:
        step(f'! No mp4s in {VIDEO_DIR}')
        return 1

    # Two separate lists: videos that need analyze_videos (no raw .h5) and
    # videos that need filterpredictions (have raw but missing _filtered.h5).
    # If a previous run was interrupted between analyze and filterpredictions,
    # we'd otherwise skip those videos from `pending` and never produce the
    # filtered output — causing the orchestrator to spin in a restart loop.
    need_analyze = []
    need_filter = []
    for v in videos:
        raw = [p for p in v.parent.glob(f'{v.stem}*shuffle{SHUFFLE}*.h5')
               if not p.name.endswith('_filtered.h5')]
        filt = list(v.parent.glob(f'{v.stem}*shuffle{SHUFFLE}*_filtered.h5'))
        if not raw:
            need_analyze.append(str(v))
            need_filter.append(str(v))
            step(f'  pending analyze + filter: {v.name}')
        elif not filt:
            need_filter.append(str(v))
            step(f'  pending filter only: {v.name}')
        else:
            step(f'  skip (analyze + filter done): {v.name}')

    if not need_analyze and not need_filter:
        step('All videos already have DLC + filtered output. Nothing to do.')
        return 0

    step('Importing DLC...')
    import deeplabcut

    if need_analyze:
        step(f'analyze_videos: {len(need_analyze)} video(s), '
             f'shuffle={SHUFFLE} batchsize={BATCHSIZE}')
        deeplabcut.analyze_videos(
            CONFIG_PATH,
            need_analyze,
            shuffle=SHUFFLE,
            videotype='.mp4',
            save_as_csv=False,
            batchsize=BATCHSIZE,
        )

    if need_filter:
        step(f'filterpredictions: {len(need_filter)} video(s)')
        deeplabcut.filterpredictions(
            CONFIG_PATH,
            need_filter,
            shuffle=SHUFFLE,
            videotype='.mp4',
        )

    step('Done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
