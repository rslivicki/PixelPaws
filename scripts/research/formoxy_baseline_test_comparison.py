"""
Comparison panels for the baseline test-pair (S1, S5 Veh + S4, S8 Oxy10).

Produces three figures and posts them to #results:

  1. dlc_overlay_paired.png — 4 rows (one per subject) x 2 cols
     (baseline | formalin) DLC keypoint overlays. Lets us eyeball
     whether the same DLC model places HL/HR correctly on both the
     baseline and formalin recordings for each subject, including
     across the resolution clusters in the 2605 cohort (S1/S4 at
     744x720, S5/S8 at 706x708).

  2. ratio_paired_bars.png — paired baseline vs formalin HL/HR contour
     ratio per subject, with resolution annotated. Red dashed line at
     1.0 (symmetry) and a horizontal band at the test-pair baseline
     mean ± SD as the "no-pain ceiling" reference.

  3. paw_intensity_paired.png — for each subject, mean Pix_HL and
     mean Pix_HR (contour interior) under baseline and formalin. Tells
     us whether the ratio change is driven by HL dropping, HR rising,
     or both -- and whether absolute brightness differs by resolution.

Reads cached data only; runs in seconds.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
import uuid
from pathlib import Path

REPO = Path(r"E:\PixelPaws")
sys.path.insert(0, str(REPO))

COHORT = Path(r"E:\RSVIDS\Blackbox\2605_FormOxy_LeftPaws")
VIDS = COHORT / "videos"
GLAB = COHORT / "gait_limb_analysis"
ANALYSIS = COHORT / "analysis"

SHUFFLE = 9
FPS = 60
SUBJECTS = [
    (1, "Veh"),   # 744x720
    (5, "Veh"),   # 706x708
    (4, "Oxy10"), # 744x720
    (8, "Oxy10"), # 706x708
]

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
        mime = "image/png" if p.suffix == ".png" else "text/csv"
        body += [
            f"--{boundary}".encode(),
            f'Content-Disposition: form-data; name="file{i}"; filename="{p.name}"'.encode(),
            f"Content-Type: {mime}".encode(), b"",
            p.read_bytes(),
        ]
    body.append(f"--{boundary}--".encode())
    req = urllib.request.Request(
        WEBHOOK, data=b"\r\n".join(body), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "User-Agent": "PixelPaws-FormOxy-BaselineCompare/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            step(f"Discord upload: HTTP {r.status}")
    except Exception as e:
        step(f"Discord upload failed: {e}")


def video_dims(v: Path) -> tuple[int, int]:
    import cv2
    cap = cv2.VideoCapture(str(v))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def session_metrics(contour_csv: Path) -> dict:
    """Return mean HL, mean HR, ratio on contour-valid frames."""
    import numpy as np
    import pandas as pd
    df = pd.read_csv(contour_csv, usecols=["intensities_HL", "intensities_HR"])
    hl = df["intensities_HL"].to_numpy(dtype=float)
    hr = df["intensities_HR"].to_numpy(dtype=float)
    valid = (hl > 0) & (hr > 0) & np.isfinite(hl) & np.isfinite(hr)
    if valid.sum() == 0:
        return {"HL": float("nan"), "HR": float("nan"),
                "ratio": float("nan"), "n_valid": 0}
    mh = float(np.mean(hl[valid])); mr = float(np.mean(hr[valid]))
    return {"HL": mh, "HR": mr,
            "ratio": mh / mr if mr > 0 else float("nan"),
            "n_valid": int(valid.sum())}


def make_overlay_paired(out_png: Path) -> None:
    """4-row (subject) x 2-col (baseline|formalin) grid of DLC overlays."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from formoxy_dlc_screenshots import (load_dlc, pick_frame, grab_frame,
                                         BP_COLORS)

    fig, axes = plt.subplots(len(SUBJECTS), 2, figsize=(9, 4.0 * len(SUBJECTS)))
    if len(SUBJECTS) == 1:
        axes = np.array([axes])

    for row, (subj, dose) in enumerate(SUBJECTS):
        for col, cond in enumerate(["baseline", "formalin"]):
            ax = axes[row, col]
            vid = VIDS / f"2605_FormOxy_S{subj}_{cond}.mp4"
            h5_candidates = list(VIDS.glob(f"{vid.stem}*shuffle{SHUFFLE}*_filtered.h5"))
            if not h5_candidates:
                ax.set_title(f"S{subj} {cond} — no h5"); ax.axis("off")
                continue
            try:
                bp_data, n_frames = load_dlc(h5_candidates[0])
                target = n_frames // 2
                f_idx = pick_frame(bp_data, n_frames, target)
                img, fw, fh = grab_frame(vid, f_idx)
            except Exception as e:
                step(f"  ! S{subj} {cond} failed: {e}")
                ax.set_title(f"S{subj} {cond} — read failed"); ax.axis("off")
                continue

            ax.imshow(img)
            for bp, (xa, ya, pa) in bp_data.items():
                if f_idx >= len(xa): continue
                x, y, p = xa[f_idx], ya[f_idx], pa[f_idx]
                if not (np.isfinite(x) and np.isfinite(y)) or p < 0.3: continue
                ax.plot(x, y, marker="o", markersize=6, color=BP_COLORS[bp],
                        markeredgecolor="white", markeredgewidth=0.8,
                        linestyle="none")
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
            ax.set_title(f"S{subj} ({dose}) — {cond}\n{fw}x{fh} • frame {f_idx:,} ({int(sec//60)}:{int(sec%60):02d})",
                         fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("DLC keypoints — baseline vs formalin (test pair)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def gather_metrics():
    """Return list of dicts with per-session ratio + resolution."""
    rows = []
    for subj, dose in SUBJECTS:
        for cond in ["baseline", "formalin"]:
            vid = VIDS / f"2605_FormOxy_S{subj}_{cond}.mp4"
            ctr_csvs = list(GLAB.glob(f"2605_FormOxy_S{subj}_{cond}_contour_*.csv"))
            if not ctr_csvs:
                step(f"  ! no contour CSV for S{subj} {cond}; skip"); continue
            w, h = video_dims(vid)
            m = session_metrics(ctr_csvs[0])
            rows.append({
                "subject": f"S{subj}", "dose": dose, "condition": cond,
                "resolution": f"{w}x{h}",
                **m,
            })
    return rows


def plot_ratio_bars(rows, out_png: Path) -> None:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Order: pair baseline/formalin per subject
    subjects = [f"S{s}" for s, _ in SUBJECTS]
    n_sub = len(subjects)
    x = np.arange(n_sub)
    w = 0.36

    bl_vals = [next((r["ratio"] for r in rows
                     if r["subject"] == s and r["condition"] == "baseline"),
                    float("nan")) for s in subjects]
    fm_vals = [next((r["ratio"] for r in rows
                     if r["subject"] == s and r["condition"] == "formalin"),
                    float("nan")) for s in subjects]
    bl_res = [next((r["resolution"] for r in rows
                    if r["subject"] == s and r["condition"] == "baseline"), "?")
              for s in subjects]
    fm_res = [next((r["resolution"] for r in rows
                    if r["subject"] == s and r["condition"] == "formalin"), "?")
              for s in subjects]

    dose_colors = {"Veh": "#000000", "Oxy10": "#08306b"}
    bar_colors_fm = [dose_colors[d] for _, d in SUBJECTS]

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.bar(x - w/2, bl_vals, width=w, color="#7f7f7f", edgecolor="black",
           linewidth=0.6, label="baseline", alpha=0.7)
    ax.bar(x + w/2, fm_vals, width=w, color=bar_colors_fm, edgecolor="black",
           linewidth=0.6, label="formalin (dose-colored)", alpha=0.8)

    # Numeric labels above each bar
    for i, (bv, fv) in enumerate(zip(bl_vals, fm_vals)):
        ax.text(i - w/2, bv + 0.005, f"{bv:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + w/2, fv + 0.005, f"{fv:.3f}", ha="center", va="bottom", fontsize=9)
        # Resolution under x-axis
        ax.text(i - w/2, -0.02, bl_res[i], ha="center", va="top",
                fontsize=7, color="#555555",
                transform=ax.get_xaxis_transform())
        ax.text(i + w/2, -0.02, fm_res[i], ha="center", va="top",
                fontsize=7, color="#555555",
                transform=ax.get_xaxis_transform())

    # Reference lines: 1.0 symmetry + baseline cohort mean ± SD
    bl_arr = np.array(bl_vals, dtype=float)
    bl_mean = float(np.nanmean(bl_arr))
    bl_sd = float(np.nanstd(bl_arr, ddof=1))
    ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.1, alpha=0.8,
               label="L/R symmetry (1.0)")
    ax.axhspan(bl_mean - bl_sd, bl_mean + bl_sd, color="#2ca02c", alpha=0.12,
               label=f"baseline mean +/- SD  ({bl_mean:.3f} +/- {bl_sd:.3f})")
    ax.axhline(bl_mean, color="#2ca02c", linestyle=":", linewidth=1.2, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n({d})" for s, d in [(f"S{s}", d) for s, d in SUBJECTS]],
                       fontsize=10)
    ax.set_ylabel("Contour intensity ratio HL/HR")
    ax.set_title("Test pair: per-subject baseline vs formalin HL/HR contour ratio",
                 fontweight="bold")
    ax.set_ylim(0.70, 1.08)
    ax.legend(loc="lower left", frameon=True, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_intensity_paired(rows, out_png: Path) -> None:
    """For each subject, mean Pix_HL and Pix_HR under both conditions."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subjects = [f"S{s}" for s, _ in SUBJECTS]
    n_sub = len(subjects)
    x = np.arange(n_sub)
    w = 0.18

    def pick(s, cond, key):
        return next((r[key] for r in rows
                     if r["subject"] == s and r["condition"] == cond),
                    float("nan"))

    bl_hl = [pick(s, "baseline", "HL") for s in subjects]
    bl_hr = [pick(s, "baseline", "HR") for s in subjects]
    fm_hl = [pick(s, "formalin", "HL") for s in subjects]
    fm_hr = [pick(s, "formalin", "HR") for s in subjects]

    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.bar(x - 1.5*w, bl_hl, width=w, color="#e41a1c", alpha=0.45,
           label="baseline HL", edgecolor="black", linewidth=0.5)
    ax.bar(x - 0.5*w, bl_hr, width=w, color="#377eb8", alpha=0.45,
           label="baseline HR", edgecolor="black", linewidth=0.5)
    ax.bar(x + 0.5*w, fm_hl, width=w, color="#e41a1c", alpha=0.95,
           label="formalin HL", edgecolor="black", linewidth=0.5)
    ax.bar(x + 1.5*w, fm_hr, width=w, color="#377eb8", alpha=0.95,
           label="formalin HR", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n({d})" for s, d in [(f"S{s}", d) for s, d in SUBJECTS]],
                       fontsize=10)
    ax.set_ylabel("Mean inside-contour intensity (a.u.)")
    ax.set_title("Test pair: per-paw contour intensity, baseline vs formalin",
                 fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    import pandas as pd
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    step("Building DLC overlay grid...")
    overlay_png = ANALYSIS / "baseline_test_dlc_overlay_paired.png"
    make_overlay_paired(overlay_png)
    step(f"  -> {overlay_png}")

    step("Gathering ratio/intensity metrics...")
    rows = gather_metrics()
    step(f"  rows: {len(rows)}")

    summary_csv = ANALYSIS / "baseline_test_compare_summary.csv"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    step(f"  -> {summary_csv}")

    bars_png = ANALYSIS / "baseline_test_ratio_paired_bars.png"
    plot_ratio_bars(rows, bars_png)
    step(f"  -> {bars_png}")

    intensity_png = ANALYSIS / "baseline_test_paw_intensity_paired.png"
    plot_intensity_paired(rows, intensity_png)
    step(f"  -> {intensity_png}")

    # Headline numbers
    import numpy as np
    bl_vals = [r["ratio"] for r in rows if r["condition"] == "baseline"]
    fm_vals = [r["ratio"] for r in rows if r["condition"] == "formalin"]
    bl_mean = float(np.nanmean(bl_vals)) if bl_vals else float("nan")
    bl_sd   = float(np.nanstd(bl_vals, ddof=1)) if len(bl_vals) > 1 else 0.0
    fm_mean = float(np.nanmean(fm_vals)) if fm_vals else float("nan")

    head = (
        "**Baseline pipeline test comparison (4 subjects, both conditions)**\n"
        f"Subjects: 2 Veh (S1 @ 744x720, S5 @ 706x708) + 2 Oxy10 (S4 @ 744x720, "
        f"S8 @ 706x708) -- covers two of the three resolution clusters in 2605.\n\n"
        f"**Baseline ratio across the 4 subjects:** {bl_mean:.3f} +/- {bl_sd:.3f} "
        f"(SD across mice, not within-session).\n"
        f"**Formalin ratio across the same 4 subjects:** {fm_mean:.3f}.\n\n"
        "Three figures attached:\n"
        "1. DLC overlay grid -- visually verify keypoints land correctly on "
        "both resolutions and both conditions.\n"
        "2. Paired baseline vs formalin ratio bars per subject, with resolution "
        "annotated below each bar. Green band = baseline mean +/- SD.\n"
        "3. Per-paw intensity breakdown so you can see whether the ratio change "
        "is driven by HL dropping (paw guarding), HR staying constant, or both."
    )
    discord_upload(head, [overlay_png, bars_png, intensity_png, summary_csv])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
