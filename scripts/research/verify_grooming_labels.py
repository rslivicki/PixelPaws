"""Pull a representative frame from each of the first 6 body_grooming
bouts in 260318_JG_9433_Baseline (the session with 11.84% pos rate).
Saves a montage so the user can eyeball whether the labels look like
actual grooming.
"""
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

SESSION = "260318_JG_9433_Baseline"
BEHAVIOUR = "body_grooming"

sc_path = Path(r"E:/RS_Boris/per_frame_labels") / f"{SESSION}__{BEHAVIOUR}.json"
arr_path = sc_path.with_suffix(".npy")
meta = json.loads(sc_path.read_text())
arr = np.load(arr_path)
vid = Path(meta["video_path"])
fps = meta["fps"]


def to_bouts(arr):
    bouts = []
    in_b, s = False, 0
    for i, v in enumerate(arr):
        if v and not in_b:
            in_b, s = True, i
        elif not v and in_b:
            in_b = False
            bouts.append((s, i))
    if in_b:
        bouts.append((s, len(arr)))
    return bouts


bouts = to_bouts(arr)
print(f"{SESSION}: {len(bouts)} bouts, video={vid.name}")

# Pick first 6 bouts; midframe of each
sample_bouts = bouts[:6]
cap = cv2.VideoCapture(str(vid))

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
for i, (s, e) in enumerate(sample_bouts):
    mid = (s + e) // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ok, frame = cap.read()
    if not ok:
        continue
    ax = axes[i]
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.set_title(f"bout {i+1}: frame {mid} (t={mid/fps:.1f}s, dur={(e-s)/fps:.1f}s)",
                 fontsize=11)
    ax.axis("off")

# fill remaining axes if fewer bouts
for j in range(len(sample_bouts), 6):
    axes[j].axis("off")

cap.release()
plt.suptitle(f"BORIS label sanity: {SESSION} — body_grooming midframes",
             fontsize=13, y=1.005)
plt.tight_layout()
out = Path(r"E:/PixelPaws/scripts/research/label_sanity_body_grooming.png")
plt.savefig(out, dpi=110, bbox_inches="tight")
print(f"wrote {out}")
