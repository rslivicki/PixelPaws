"""
Run filterpredictions on THC withdrawal videos that have raw shuffle9 .h5
but no _filtered.h5. Caused by the orchestrator killing DLC after analyze
but before filterpredictions completed.

Run:
  C:/Users/Gereau/anaconda3/envs/DEEPLABCUT/python.exe scripts/research/thc_filter_missing.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path

CONFIG_PATH = r'E:\RSDLC\2511_RSGNKK_Blackbox\config.yaml'
VIDEO_DIR = Path(r'E:\RSVIDS\Blackbox\260506_RS_THC_Withdrawal\Videos')
SHUFFLE = 9


def step(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def main():
    missing = []
    for v in sorted(VIDEO_DIR.glob('*.mp4')):
        raw = list(v.parent.glob(f'{v.stem}*shuffle{SHUFFLE}*.h5'))
        raw = [p for p in raw if not p.name.endswith('_filtered.h5')]
        filt = list(v.parent.glob(f'{v.stem}*shuffle{SHUFFLE}*_filtered.h5'))
        if raw and not filt:
            missing.append(str(v))

    if not missing:
        step('All videos already have _filtered.h5. Nothing to do.')
        return 0

    step(f'Need filterpredictions on {len(missing)} video(s):')
    for v in missing:
        step(f'  - {Path(v).name}')

    step('Importing DLC...')
    import deeplabcut

    step('filterpredictions...')
    deeplabcut.filterpredictions(
        CONFIG_PATH, missing,
        shuffle=SHUFFLE, videotype='.mp4',
    )
    step('Done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
