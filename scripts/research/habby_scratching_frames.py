"""
Render frame samples from Habby's longest detected scratching bouts so
the user can visually verify the classifier hits real scratching.

Per session: pick the top 3 longest bouts from the AllFeatures
classifier, snap 4 frames evenly spaced through each bout, lay them
out as a strip with bout info captioned. Post to Discord.
"""
from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = Path(r'E:\RSVIDS\Blackbox\AM_ChR2_stim_SPB_analysis')
VIDEO_DIR = PROJECT / 'Videos'
ANALYSIS_DIR = PROJECT / 'analysis'

WEBHOOK = (
    ""
)


def post_to_discord(text, png_paths):
    import os, subprocess
    cmd = ['curl', '-s', '-o', os.devnull, '-w', '%{http_code}', '-X', 'POST',
           '-F', f'payload_json={{"content":"{text}"}}']
    for i, p in enumerate(png_paths):
        cmd += ['-F', f'file{i}=@{p}']
    cmd.append(WEBHOOK)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f'Discord upload: HTTP {r.stdout}')


def render_session(bouts_df: pd.DataFrame, video_path: Path, out_png: Path,
                   n_top: int = 3, frames_per_bout: int = 4) -> bool:
    """Pick top n_top longest bouts, snap frames_per_bout frames per bout,
    save as a grid."""
    sub = bouts_df.sort_values('length_frames', ascending=False).head(n_top)
    if sub.empty:
        return False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f'  ! cannot open {video_path}')
        return False

    n_rows = len(sub)
    n_cols = frames_per_bout
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.3, n_rows * 3.0))
    if n_rows == 1:
        axes = np.array([axes])

    for r, (_, row) in enumerate(sub.iterrows()):
        s = int(row['start_frame']); e = int(row['end_frame'])
        # Frame indices evenly spaced through bout
        idxs = np.linspace(s, e, n_cols).astype(int)
        for c, fi in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            ax = axes[r, c]
            if ok:
                # BGR -> RGB
                ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ax.set_xticks([]); ax.set_yticks([])
            t_sec = fi / 60.0
            ax.set_xlabel(f'fr {fi}  ({t_sec:.1f}s)', fontsize=8)
            if c == 0:
                ax.set_ylabel(
                    f'bout {r+1}\n{row["length_frames"]}fr  '
                    f'({row["length_sec"]:.2f}s)', fontsize=9)

    cap.release()
    fig.suptitle(out_png.stem, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return True


def main() -> int:
    bouts_csv = ANALYSIS_DIR / 'scratching_bouts.csv'
    if not bouts_csv.is_file():
        print(f'! missing {bouts_csv}')
        return 1
    df = pd.read_csv(bouts_csv)
    # Use the AllFeatures classifier only
    df = df[df['classifier'].str.contains('AllFeatures', case=False)]
    print(f'Total AllFeatures bouts across project: {len(df)}')

    out_dir = ANALYSIS_DIR / 'scratching_frames'
    out_dir.mkdir(parents=True, exist_ok=True)

    pngs = []
    for sess, sess_df in df.groupby('session'):
        # Find matching video
        cands = list(VIDEO_DIR.glob(f'{sess}.mp4'))
        if not cands:
            cands = list(VIDEO_DIR.glob(f'{sess}*.mp4'))
        if not cands:
            print(f'  ! no video for {sess}')
            continue
        video = cands[0]
        cond = sess_df['condition'].iloc[0]
        mouse = sess_df['mouse'].iloc[0]
        out = out_dir / f'{mouse}_{cond}_{sess[:30]}.png'
        if render_session(sess_df, video, out):
            pngs.append(out)
            print(f'  rendered {out.name}')

    if not pngs:
        print('No frames rendered.')
        return 1

    # Discord — post in batches of 10 (Discord cap)
    BATCH = 10
    for i in range(0, len(pngs), BATCH):
        chunk = pngs[i:i + BATCH]
        text = (f'Habby scratching: longest 3 bouts per session, '
                f'4 frames per bout (evenly spaced). '
                f'Sessions {i+1}-{i+len(chunk)} of {len(pngs)}.')
        post_to_discord(text, chunk)
    return 0


if __name__ == '__main__':
    sys.exit(main())
