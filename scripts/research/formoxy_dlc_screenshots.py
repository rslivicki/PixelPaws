"""
Sample one mid-session frame from each 2605 formalin video, overlay
DLC keypoints (colored + labeled), and post a 3x4 grid to #results so
we can eyeball tracking quality across subjects.

For each subject:
  - pick a frame near the middle of the recording where the snout and
    both hind paws have likelihood >= 0.8 (otherwise scan +/- 1500 frames
    for one that does)
  - draw each body part as a colored dot
  - label the hind paws "HL"/"HR" to make formalin vs control obvious
  - subtitle: subject + dose + frame index + min:sec into session
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
import uuid
from pathlib import Path

COHORT = Path(r"E:\RSVIDS\Blackbox\2605_FormOxy_LeftPaws")
VIDS = COHORT / "videos"
ANALYSIS = COHORT / "analysis"

SHUFFLE = 9
FPS = 60

DOSE_SEQ = ["Veh", "Oxy1", "Oxy3", "Oxy10"]


def dose_for(n: int) -> str:
    return DOSE_SEQ[(n - 1) % 4]


# Body-part color scheme — hind paws are the biology of interest
BP_COLORS = {
    "tailtip":  "#9467bd",
    "tailbase": "#8c564b",
    "centroid": "#7f7f7f",
    "neck":     "#17becf",
    "snout":    "#ffdd00",
    "hlpaw":    "#e41a1c",  # injected paw -> red
    "hrpaw":    "#377eb8",  # control paw  -> blue
    "flpaw":    "#fb9a99",
    "frpaw":    "#a6cee3",
}

WEBHOOK = (
    ""
)


def step(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def discord_upload(content: str, files: list[Path]) -> None:
    boundary = "----PP" + uuid.uuid4().hex
    body = [
        f"--{boundary}".encode(),
        b'Content-Disposition: form-data; name="payload_json"',
        b'Content-Type: application/json', b"",
        json.dumps({"content": content}).encode(),
    ]
    for i, p in enumerate(files):
        body += [
            f"--{boundary}".encode(),
            f'Content-Disposition: form-data; name="file{i}"; filename="{p.name}"'.encode(),
            b"Content-Type: image/png", b"",
            p.read_bytes(),
        ]
    body.append(f"--{boundary}--".encode())
    req = urllib.request.Request(
        WEBHOOK, data=b"\r\n".join(body), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "User-Agent": "PixelPaws-FormOxy-Screenshots/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            step(f"Discord upload: HTTP {r.status}")
    except Exception as e:
        step(f"Discord upload failed: {e}")


def load_dlc(h5_path: Path):
    """Return dict of {bp: (x_arr, y_arr, prob_arr)} and n_frames."""
    import numpy as np
    import pandas as pd
    df = pd.read_hdf(h5_path)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(c).strip() for c in df.columns.values]
    out = {}
    for bp in BP_COLORS:
        x_col = next((c for c in df.columns if c.endswith(f"{bp}_x")), None)
        y_col = next((c for c in df.columns if c.endswith(f"{bp}_y")), None)
        p_col = next((c for c in df.columns
                      if c.endswith(f"{bp}_likelihood") or c.endswith(f"{bp}_prob")), None)
        if x_col is None or y_col is None:
            continue
        out[bp] = (df[x_col].to_numpy().astype(float),
                   df[y_col].to_numpy().astype(float),
                   df[p_col].to_numpy().astype(float) if p_col else np.ones(len(df)))
    return out, len(df)


def pick_frame(bp_data: dict, n_frames: int, target: int, min_prob: float = 0.8,
               scan: int = 1500) -> int:
    """Scan +/- `scan` frames around `target` for one where snout, hlpaw, hrpaw
    all have likelihood >= min_prob; fall back to `target` if none found."""
    import numpy as np
    must_have = ["snout", "hlpaw", "hrpaw"]
    have = [bp for bp in must_have if bp in bp_data]
    if not have:
        return target
    n_frames = min(n_frames, len(next(iter(bp_data.values()))[0]))
    lo = max(0, target - scan); hi = min(n_frames, target + scan)
    for f in range(target, hi):
        if all(bp_data[bp][2][f] >= min_prob for bp in have):
            return f
    for f in range(target - 1, lo - 1, -1):
        if all(bp_data[bp][2][f] >= min_prob for bp in have):
            return f
    return target


def grab_frame(video_path: Path, frame_idx: int):
    """Return (RGB uint8 image, fw, fh)."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, bgr = cap.read()
    cap.release()
    if not ret or bgr is None:
        raise RuntimeError(f"could not read frame {frame_idx} of {video_path.name}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, rgb.shape[1], rgb.shape[0]


def main() -> int:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    videos = sorted(VIDS.glob("2605_FormOxy_S*_formalin.mp4"))
    videos = [v for v in videos if "DLC_" not in v.name and "_labeled" not in v.name]
    # Sort by subject number, not alphabetic (so S2 before S10)
    rx = re.compile(r"S(\d+)_formalin")
    videos.sort(key=lambda v: int(rx.search(v.stem).group(1)))
    step(f"Subjects: {len(videos)}")

    fig, axes = plt.subplots(3, 4, figsize=(15, 12))
    for ax, vid in zip(axes.flat, videos):
        subj = int(rx.search(vid.stem).group(1))
        dose = dose_for(subj)
        h5_candidates = list(VIDS.glob(f"{vid.stem}*shuffle{SHUFFLE}*_filtered.h5"))
        if not h5_candidates:
            step(f"  ! no filtered h5 for S{subj}; skipping")
            ax.set_title(f"S{subj} ({dose}) — no DLC h5")
            ax.axis("off")
            continue
        bp_data, n_frames = load_dlc(h5_candidates[0])
        target = n_frames // 2
        f_idx = pick_frame(bp_data, n_frames, target)
        try:
            img, fw, fh = grab_frame(vid, f_idx)
        except Exception as e:
            step(f"  ! S{subj} grab failed: {e}")
            ax.set_title(f"S{subj} ({dose}) — frame read failed")
            ax.axis("off")
            continue

        ax.imshow(img)
        for bp, (xa, ya, pa) in bp_data.items():
            if f_idx >= len(xa):
                continue
            x, y, p = xa[f_idx], ya[f_idx], pa[f_idx]
            if not (np.isfinite(x) and np.isfinite(y)) or p < 0.3:
                continue
            color = BP_COLORS[bp]
            ax.plot(x, y, marker="o", markersize=6, color=color,
                    markeredgecolor="white", markeredgewidth=0.8,
                    linestyle="none")
            # Label hind paws prominently
            if bp == "hlpaw":
                ax.annotate("HL", xy=(x, y), xytext=(8, -8),
                            textcoords="offset points", color="#e41a1c",
                            fontsize=10, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                      edgecolor="#e41a1c", alpha=0.85))
            elif bp == "hrpaw":
                ax.annotate("HR", xy=(x, y), xytext=(8, -8),
                            textcoords="offset points", color="#377eb8",
                            fontsize=10, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                                      edgecolor="#377eb8", alpha=0.85))

        sec = f_idx / FPS
        mm, ss = int(sec // 60), int(sec % 60)
        ax.set_title(f"S{subj}  ({dose})  —  frame {f_idx:,}  ({mm}:{ss:02d})",
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    # Legend with body-part colors (only meaningful ones)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=BP_COLORS[bp], markeredgecolor="white",
                      markersize=8, label=bp)
               for bp in ["snout", "neck", "centroid", "tailbase", "tailtip",
                          "flpaw", "frpaw", "hlpaw", "hrpaw"]]
    fig.legend(handles=handles, loc="lower center", ncol=9, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("2605 FormOxy — DLC keypoints overlaid (mid-session frame per subject)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    out_png = ANALYSIS / "dlc_keypoints_grid_2605.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    step(f"Figure: {out_png}")

    discord_upload(
        "**2605 FormOxy: DLC keypoints overlaid on a mid-session frame per subject.**\n"
        "HL (red) = formalin-injected hind paw; HR (blue) = contralateral control. "
        "Frame chosen from around the 50%-mark, snapped to the nearest frame where "
        "snout + both hind paws have likelihood >= 0.8.",
        [out_png],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
