"""
DLC shuffle 9 on E:\\RSVIDS\\Blackbox\\2603_SNLT_JG\\Baseline\\videos\\2605_Cohort2.

batchsize=32 (proven stable on 2080 Ti for sustained runs).
"""
from __future__ import annotations
import sys, time
from pathlib import Path

CONFIG_PATH = r'E:\RSDLC\2511_RSGNKK_Blackbox\config.yaml'
VIDEO_DIR = Path(r'E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\videos\2605_Cohort2')
SHUFFLE = 9
BATCHSIZE = 32


def step(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def main():
    videos = sorted(VIDEO_DIR.glob('*.mp4'))
    # Split into analyze-needed and filter-needed. See sister
    # dlc_batch_thc_withdrawal.py for rationale (interrupted runs leave
    # raw .h5 without _filtered.h5; previously got skipped forever).
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
        step('Nothing to do.')
        return 0

    step('Importing DLC...')
    import deeplabcut

    if need_analyze:
        step(f'analyze_videos: {len(need_analyze)} video(s), '
             f'shuffle={SHUFFLE} batchsize={BATCHSIZE}')
        deeplabcut.analyze_videos(
            CONFIG_PATH, need_analyze, shuffle=SHUFFLE,
            videotype='.mp4', save_as_csv=False,
            batchsize=BATCHSIZE,
        )
    if need_filter:
        step(f'filterpredictions: {len(need_filter)} video(s)')
        deeplabcut.filterpredictions(
            CONFIG_PATH, need_filter, shuffle=SHUFFLE, videotype='.mp4',
        )
    step('Done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
