"""Sanity-check L_flinching labels visually. Pick a high-pos-rate session
and dump 6 labeled flinching bout midframes.
"""
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# 251126_Formalin_F_S2_Cut has 2.07% pos rate (highest in the L_flinching set)
SESSION = "251126_Formalin_F_S2_Cut"
BEHAVIOUR = "L_flinching"

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
print(f"{SESSION}: {len(bouts)} flinch bouts, video={vid.name}")

# Pick 6 bouts: 2 short (<5 frames), 2 medium (5-30 frames), 2 longest
short = [b for b in bouts if b[1] - b[0] < 5][:2]
med = [b for b in bouts if 5 <= b[1] - b[0] <= 30][:2]
long = sorted(bouts, key=lambda b: -(b[1] - b[0]))[:2]
sample_bouts = (short + med + long)[:6]

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
    ax.set_title(f"bout: frame {mid} (t={mid/fps:.1f}s, dur={(e-s)/fps*1000:.0f}ms)",
                 fontsize=11)
    ax.axis("off")

for j in range(len(sample_bouts), 6):
    axes[j].axis("off")

cap.release()
plt.suptitle(
    f"BORIS label sanity: {SESSION} — L_flinching midframes\n"
    f"(2 short bouts + 2 medium + 2 longest, of {len(bouts)} total)",
    fontsize=13, y=1.005,
)
plt.tight_layout()
out = Path(r"E:/PixelPaws/scripts/research/label_sanity_L_flinching.png")
plt.savefig(out, dpi=110, bbox_inches="tight")
print(f"wrote {out}")
