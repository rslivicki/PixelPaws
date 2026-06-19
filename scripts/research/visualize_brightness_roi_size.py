"""Render frames from a 2511 L_flinching session with sq=20 vs sq=40
brightness ROI overlays at hlpaw, hrpaw, snout. Lets the user visually
decide if sq=20 is too small or sq=40 too big.

Output: PNG montage at E:/PixelPaws/scripts/research/roi_size_compare.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Choose a session — pick 251114_Formalin_S1 (2511 cohort, used for flinching)
H5 = Path(r"E:/RSVIDS/Blackbox/2511_Blackbox_Formalin/251114_Formalin_encoded/"
          r"251114_Formalin_S1DLC_Resnet50_palmreader-500Mar25shuffle1_snapshot_best-110_filtered.h5")
# Fall back to non-filtered if filtered isn't there
if not H5.is_file():
    cand = list(Path(r"E:/RSVIDS/Blackbox/2511_Blackbox_Formalin/251114_Formalin_encoded/").glob(
        "251114_Formalin_S1*filtered.h5"))
    H5 = cand[0] if cand else None
MP4 = Path(r"E:/RSVIDS/Blackbox/2511_Blackbox_Formalin/251114_Formalin_encoded/"
           r"251114_Formalin_S1.mp4")
OUT = Path(r"E:/PixelPaws/scripts/research/roi_size_compare.png")

BPS = ["hlpaw", "hrpaw", "snout"]
COLORS = {"hlpaw": (0, 255, 255),    # cyan
          "hrpaw": (255, 0, 255),    # magenta
          "snout": (255, 255, 0)}    # yellow


def load_dlc(h5: Path) -> pd.DataFrame:
    df = pd.read_hdf(h5)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(c).strip() for c in df.columns.values]
        df.columns = [c.replace('_likelihood', '_prob') for c in df.columns]
    return df


def get_xy_prob(df: pd.DataFrame, bp: str, frame_idx: int) -> tuple[float, float, float]:
    # find the col matching e.g. 'hlpaw_x'
    cols = [c for c in df.columns if c.endswith(f'{bp}_x')]
    if not cols:
        raise KeyError(bp)
    base = cols[0][:-2]
    x = float(df[f'{base}_x'].iloc[frame_idx])
    y = float(df[f'{base}_y'].iloc[frame_idx])
    p = float(df[f'{base}_prob'].iloc[frame_idx])
    return x, y, p


def draw_box(img: np.ndarray, cx: float, cy: float, sq: int, color: tuple) -> None:
    H, W = img.shape[:2]
    x1, y1 = int(round(cx - sq)), int(round(cy - sq))
    x2, y2 = int(round(cx + sq)), int(round(cy + sq))
    cv2.rectangle(img, (max(0, x1), max(0, y1)), (min(W-1, x2), min(H-1, y2)),
                  color, 2)


def main() -> int:
    if not H5 or not H5.is_file():
        print(f"DLC h5 not found"); return 1
    if not MP4.is_file():
        print(f"video not found: {MP4}"); return 1

    df = load_dlc(H5)
    cap = cv2.VideoCapture(str(MP4))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"video: {n_frames} frames @ {fps:.1f} fps; H5 rows: {len(df)}")

    # Pick 4 sample frames spread through the video, prefer frames where all
    # bps have high prob (mouse is visible in good pose).
    n = min(n_frames, len(df))
    # Search for frames with all bps prob > 0.9
    candidates = []
    rng = np.random.default_rng(42)
    for trial in range(200):
        i = int(rng.integers(int(n * 0.1), int(n * 0.9)))
        try:
            probs = [get_xy_prob(df, bp, i)[2] for bp in BPS]
            if all(p > 0.9 for p in probs):
                candidates.append(i)
                if len(candidates) >= 4:
                    break
        except KeyError:
            pass
    if len(candidates) < 4:
        candidates += list(rng.integers(0, n, 4 - len(candidates)))
    candidates = candidates[:4]
    print(f"sampling frames: {candidates}")

    fig, axes = plt.subplots(len(candidates), 2, figsize=(14, 4 * len(candidates)))
    if len(candidates) == 1:
        axes = axes[None, :]

    for row, fi in enumerate(candidates):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        # crop to mouse-relevant region: find bounding box of bps with margin
        xs, ys = [], []
        for bp in BPS:
            x, y, p = get_xy_prob(df, bp, fi)
            xs.append(x); ys.append(y)
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        # Crop ROI 250x250 centred on mouse
        roi_half = 200
        H, W = frame.shape[:2]
        cx_i, cy_i = int(round(cx)), int(round(cy))
        x0, x1 = max(0, cx_i - roi_half), min(W, cx_i + roi_half)
        y0, y1 = max(0, cy_i - roi_half), min(H, cy_i + roi_half)

        for col, sq in enumerate([20, 40]):
            img = frame.copy()
            for bp in BPS:
                x, y, p = get_xy_prob(df, bp, fi)
                if p > 0.5:
                    draw_box(img, x, y, sq, COLORS[bp])
                    # center dot
                    cv2.circle(img, (int(round(x)), int(round(y))), 3,
                               COLORS[bp], -1)
            cropped = img[y0:y1, x0:x1]
            ax = axes[row, col]
            ax.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            ax.set_title(f"frame {fi} (t={fi/fps:.1f}s) — sq={sq} "
                         f"(box = {2*sq+1}×{2*sq+1}px)",
                         fontsize=11)
            ax.axis("off")

    # Legend
    legend = "\n".join([f"{bp}: {tuple(c/255 for c in col[::-1])}"
                       for bp, col in COLORS.items()])
    plt.suptitle(
        f"Brightness ROI: sq=20 (left) vs sq=40 (right)\n"
        f"video: {MP4.name}  |  bps: hlpaw=cyan, hrpaw=magenta, snout=yellow",
        fontsize=12, y=1.005,
    )
    plt.tight_layout()
    plt.savefig(OUT, dpi=120, bbox_inches="tight")
    print(f"wrote: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
