"""Filter the 5 THC videos that DLC analyze finished but never filtered
(killed before filterpredictions ran in the first DLC pass)."""
from __future__ import annotations
from pathlib import Path

CONFIG = r'E:\RSDLC\2511_RSGNKK_Blackbox\config.yaml'
VIDEO_DIR = Path(r'E:\RSVIDS\Blackbox\260506_RS_THC_Withdrawal\Videos')
SHUFFLE = 9

videos = sorted(VIDEO_DIR.glob('*.mp4'))
to_filter = []
for v in videos:
    has_filtered = list(v.parent.glob(f'{v.stem}*shuffle{SHUFFLE}*filtered.h5'))
    if not has_filtered:
        to_filter.append(str(v))

print(f'Need to filter: {len(to_filter)}')
for v in to_filter:
    print(f'  {Path(v).name}')

if to_filter:
    import deeplabcut
    deeplabcut.filterpredictions(CONFIG, to_filter, shuffle=SHUFFLE,
                                 videotype='.mp4')
    print('Filter done.')
