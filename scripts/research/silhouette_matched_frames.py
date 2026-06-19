"""
silhouette_matched_frames.py — sample frames with the SAME
silhouette_frac value from two different sessions / projects, and
display them side by side. If the same numerical value corresponds
to the same posture, the feature scale is cross-rig comparable. If
not, there's a rig artifact.
"""

from __future__ import annotations
import os
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SESSIONS = [
    {
        "label": "2604_DV_DSS",
        "video": r"E:\RSVIDS\Blackbox\2604_DV_DSS\videos\202604_Ai32_61_BL_cropped.mp4",
        "pkl":   r"E:\RSVIDS\Blackbox\2604_DV_DSS\features\202604_Ai32_61_BL_cropped_features_2d46556a.pkl",
    },
    {
        "label": "SNLT_Baseline",
        "video": r"E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\videos\260318_JG_9417_Baseline.mp4",
        "pkl":   r"E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\features\260318_JG_9417_Baseline_features_414c7814.pkl",
    },
]
TARGET_FRACS = [0.07, 0.10, 0.13, 0.16, 0.19]   # spaced across the overlapping range


def grab_frame(video: str, idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    cap.release()
    if not ok or fr is None:
        return None
    return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)


def find_frame_near(sf: pd.Series, target: float) -> int:
    """Return the frame index whose silhouette_frac is closest to target,
    skipping NaN. Avoids picking adjacent frames by sampling from a
    quartile-of-the-session slice if multiple matches exist."""
    diffs = (sf - target).abs()
    return int(diffs.idxmin())


def main():
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "analysis_output",
        f"silhouette_matched_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(out_dir, exist_ok=True)

    cols = TARGET_FRACS
    rows = SESSIONS
    fig, axes = plt.subplots(len(rows), len(cols),
                              figsize=(2.4 * len(cols), 2.6 * len(rows)),
                              constrained_layout=True, squeeze=False)

    for ri, sess in enumerate(rows):
        X = pd.read_pickle(sess["pkl"])
        sf = X["silhouette_frac"].astype(float)
        sa = X["silhouette_aspect"].astype(float)
        # Skip frames > 0.5 (artifacts) and last 5%
        cutoff = int(len(X) * 0.95)
        valid = (sf <= 0.5) & (sf.index < cutoff)
        sf_v = sf[valid]
        for ci, target in enumerate(cols):
            idx = find_frame_near(sf_v, target)
            actual = sf.iloc[idx]
            asp_v  = sa.iloc[idx]
            img = grab_frame(sess["video"], idx)
            ax = axes[ri, ci]
            ax.set_xticks([]); ax.set_yticks([])
            if img is not None:
                ax.imshow(img, cmap='gray', vmin=0, vmax=255)
            ax.set_title(f'frac={actual:.3f}\nasp={asp_v:.2f}\nf{idx}',
                          fontsize=8)
        axes[ri, 0].set_ylabel(sess["label"], fontsize=11)

    fig.suptitle('Same silhouette_frac value → same posture? '
                  '(matched frames across rigs)', fontsize=11)
    fig.savefig(os.path.join(out_dir, 'matched_frames.png'), dpi=120)
    plt.close(fig)
    print(f"Outputs → {out_dir}")


if __name__ == "__main__":
    main()
