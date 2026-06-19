"""
Sanity-check the baseline pipeline on just two subjects (S1 = Veh,
S4 = Oxy10) before committing to the full 12-subject overnight run.

Stage 1: DLC analyze + filterpredictions on the two specific baseline
mp4s, called from the DEEPLABCUT env via subprocess.

Stage 2: brightness extraction on the same two videos (calls a tiny
inline driver so we don't re-trigger the orchestrator's full glob).

Stage 3: contour extraction on the same two videos.

Stage 4: post DLC overlay screenshots for the 2 baseline frames + the
2 corresponding formalin frames (same subject, same paw orientation)
to #results, so we can visually compare baseline vs formalin keypoint
placement.

Run from regular py env:
  PYTHONIOENCODING=utf-8 py -X utf8 -u \
      scripts/research/formoxy_baseline_test_pair.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

REPO = Path(r"E:\PixelPaws")
sys.path.insert(0, str(REPO))

DLC_PYTHON = r"C:\Users\Gereau\anaconda3\envs\DEEPLABCUT\python.exe"
DLC_CONFIG = r"E:\RSDLC\2511_RSGNKK_Blackbox\config.yaml"

COHORT = Path(r"E:\RSVIDS\Blackbox\2605_FormOxy_LeftPaws")
VIDS = COHORT / "videos"
OUT_DIR = COHORT / "gait_limb_analysis"
ANALYSIS = COHORT / "analysis"

SHUFFLE = 9
SUBJECTS = [1, 5, 4, 8]   # 2 Veh (S1, S5) + 2 Oxy10 (S4, S8) — both dose extremes

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
                 "User-Agent": "PixelPaws-FormOxy-BaselineTest/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            step(f"Discord upload: HTTP {r.status}")
    except Exception as e:
        step(f"Discord upload failed: {e}")


def run_dlc_on(videos: list[Path]):
    """Subprocess the DEEPLABCUT env to analyze + filter just these videos."""
    inline = f"""
import deeplabcut
config = r"{DLC_CONFIG}"
vids = {[str(v) for v in videos]}
print('analyze_videos on', len(vids), 'video(s)')
deeplabcut.analyze_videos(config, vids, shuffle={SHUFFLE},
                          videotype='.mp4', save_as_csv=False, batchsize=32)
print('filterpredictions...')
deeplabcut.filterpredictions(config, vids, shuffle={SHUFFLE}, videotype='.mp4')
print('done')
"""
    step(f"DLC analyze + filter on {len(videos)} baseline video(s)...")
    rc = subprocess.run([DLC_PYTHON, "-c", inline],
                        env={**os.environ, "PYTHONIOENCODING": "utf-8"}).returncode
    step(f"DLC exit: {rc}")
    return rc


def run_brt_contour(videos: list[Path]):
    """Inline brightness + contour extraction for just these videos."""
    from formoxy_paw_brightness_extract import (BODYPARTS, ROI_SIZES as BRT_ROI,
                                                PIXEL_THRESHOLD, EXTRACTION_STRIDE,
                                                CROP_X, CROP_Y,
                                                cache_hash as brt_hash)
    from formoxy_paw_contour_extract import (extract_one as contour_extract,
                                             cache_hash as ctr_hash,
                                             ROI_SIZES as CTR_ROI)
    from brightness_features import PixelBrightnessExtractorOptimized
    import json as _json
    import pandas as pd
    import numpy as np

    bh = brt_hash(); ch = ctr_hash()
    step(f"brt hash: {bh},  contour hash: {ch}")

    for v in videos:
        base = v.stem
        h5_candidates = list(VIDS.glob(f"{base}*shuffle{SHUFFLE}*_filtered.h5"))
        if not h5_candidates:
            step(f"  ! no filtered h5 for {base}; skip"); continue
        h5 = h5_candidates[0]

        # Brightness
        out_brt = OUT_DIR / f"{base}_brt_{bh}.csv"
        if out_brt.is_file():
            step(f"  brt cached: {out_brt.name}")
        else:
            step(f"  brt extract {base} ...")
            ex = PixelBrightnessExtractorOptimized(
                BODYPARTS, square_size={k: v_ for k, v_ in BRT_ROI.items()},
                pixel_threshold=float(PIXEL_THRESHOLD),
                crop_offset_x=CROP_X, crop_offset_y=CROP_Y,
            )
            brt_df = ex.extract_brightness_features(str(h5), str(v),
                                                    stride=EXTRACTION_STRIDE)
            cols = {f"Pix_{bp}": brt_df[f"Pix_{bp}"].reset_index(drop=True)
                    for bp in ("hlpaw", "hrpaw", "flpaw", "frpaw")
                    if f"Pix_{bp}" in brt_df.columns}
            pd.DataFrame(cols).to_csv(out_brt, index=False)
            (out_brt.with_suffix(".json")).write_text(_json.dumps({
                "video": str(v), "dlc_h5": str(h5),
                "roi_sizes": [[bp, BRT_ROI[bp]] for bp in BODYPARTS],
                "crop_x": CROP_X, "crop_y": CROP_Y,
                "brt_thresh": PIXEL_THRESHOLD,
                "extraction_stride": EXTRACTION_STRIDE,
                "n_frames": int(len(next(iter(cols.values())))),
            }, indent=2))
            step(f"    -> {out_brt.name}")

        # Contour
        out_csv = OUT_DIR / f"{base}_contour_{ch}.csv"
        out_npz = OUT_DIR / f"{base}_contour_{ch}_shapes.npz"
        if out_csv.is_file() and out_npz.is_file():
            step(f"  contour cached: {out_csv.name}")
        else:
            step(f"  contour extract {base} ...")
            per_frame, shape_lists, n_kept = contour_extract(v, h5)
            cols = {}
            for role, arrays in per_frame.items():
                for metric, arr in arrays.items():
                    cols[f"{metric}_{role}"] = arr
            pd.DataFrame(cols).to_csv(out_csv, index=False)
            shape_dict = {r: np.array(s) for r, s in shape_lists.items() if s}
            if shape_dict:
                np.savez_compressed(out_npz, **shape_dict)
            (out_csv.with_suffix(".json")).write_text(_json.dumps({
                "video": str(v), "dlc_h5": str(h5),
                "contour_roi_sizes": [[bp, CTR_ROI[bp]] for bp in CTR_ROI],
                "n_frames": int(n_kept),
                "n_shapes": {r: int(len(shape_lists[r])) for r in shape_lists},
            }, indent=2))
            step(f"    -> {out_csv.name}")


def make_overlay(videos: list[Path], title: str, out_png: Path):
    """4-panel grid: each pair gets baseline + formalin DLC overlay."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from formoxy_dlc_screenshots import load_dlc, pick_frame, grab_frame, BP_COLORS

    fig, axes = plt.subplots(1, len(videos), figsize=(4.5 * len(videos), 5))
    if len(videos) == 1:
        axes = [axes]
    for ax, vid in zip(axes, videos):
        h5_candidates = list(VIDS.glob(f"{vid.stem}*shuffle{SHUFFLE}*_filtered.h5"))
        if not h5_candidates:
            ax.set_title(f"{vid.stem} — no h5"); ax.axis("off"); continue
        bp_data, n_frames = load_dlc(h5_candidates[0])
        target = n_frames // 2
        f_idx = pick_frame(bp_data, n_frames, target)
        try:
            img, fw, fh = grab_frame(vid, f_idx)
        except Exception as e:
            ax.set_title(f"{vid.stem} — frame read failed"); ax.axis("off"); continue

        ax.imshow(img)
        for bp, (xa, ya, pa) in bp_data.items():
            if f_idx >= len(xa): continue
            x, y, p = xa[f_idx], ya[f_idx], pa[f_idx]
            if not (np.isfinite(x) and np.isfinite(y)) or p < 0.3: continue
            ax.plot(x, y, marker="o", markersize=6, color=BP_COLORS[bp],
                    markeredgecolor="white", markeredgewidth=0.8, linestyle="none")
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
        sec = f_idx / 60
        ax.set_title(f"{vid.stem}\nframe {f_idx:,} ({int(sec//60)}:{int(sec%60):02d})",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def compute_ratio_summary(videos: list[Path]):
    """Per-session HL/HR contour intensity ratio for the test pair."""
    import pandas as pd
    import numpy as np
    from formoxy_paw_contour_extract import cache_hash as ctr_hash
    ch = ctr_hash()
    rows = []
    for v in videos:
        base = v.stem
        csv = OUT_DIR / f"{base}_contour_{ch}.csv"
        if not csv.is_file():
            rows.append({"video": base, "n_valid": 0, "ratio_HL_HR": float("nan")})
            continue
        df = pd.read_csv(csv, usecols=["intensities_HL", "intensities_HR"])
        hl = df["intensities_HL"].to_numpy()
        hr = df["intensities_HR"].to_numpy()
        valid = (hl > 0) & (hr > 0) & np.isfinite(hl) & np.isfinite(hr)
        ratio = float(np.mean(hl[valid]) / np.mean(hr[valid])) if valid.sum() and hr[valid].mean() > 0 else float("nan")
        rows.append({
            "video": base,
            "n_valid_frames": int(valid.sum()),
            "intensities_HL_mean": round(float(np.mean(hl[valid])), 3) if valid.sum() else float("nan"),
            "intensities_HR_mean": round(float(np.mean(hr[valid])), 3) if valid.sum() else float("nan"),
            "ratio_HL_HR": round(ratio, 4),
        })
    return pd.DataFrame(rows)


def main() -> int:
    import pandas as pd
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    bls = [VIDS / f"2605_FormOxy_S{s}_baseline.mp4" for s in SUBJECTS]
    fms = [VIDS / f"2605_FormOxy_S{s}_formalin.mp4" for s in SUBJECTS]
    for v in bls + fms:
        if not v.is_file():
            step(f"! missing video: {v}"); return 1

    # Stage 1: DLC on the 2 baselines
    bls_to_dlc = [v for v in bls
                  if not list(VIDS.glob(f"{v.stem}*shuffle{SHUFFLE}*_filtered.h5"))]
    if bls_to_dlc:
        rc = run_dlc_on(bls_to_dlc)
        if rc != 0:
            step(f"! DLC failed (rc={rc}); abort"); return rc
    else:
        step("All baseline DLC already done; skipping DLC step.")

    # Stage 2 + 3: brightness + contour on the 2 baselines
    run_brt_contour(bls)

    # Stage 4: overlay screenshots (baseline + formalin) + ratio table
    bl_png = ANALYSIS / "dlc_keypoints_baseline_test_pair.png"
    make_overlay(bls, "2605 baselines (test pair) — DLC overlay", bl_png)
    fm_png = ANALYSIS / "dlc_keypoints_formalin_test_pair.png"
    make_overlay(fms, "2605 formalin (same subjects) — DLC overlay for reference", fm_png)

    # Ratio summary
    summary = pd.concat([
        compute_ratio_summary(bls).assign(condition="baseline"),
        compute_ratio_summary(fms).assign(condition="formalin"),
    ])
    summary_csv = ANALYSIS / "baseline_test_pair_summary.csv"
    summary.to_csv(summary_csv, index=False)
    step("\nTest pair summary:\n" + summary.to_string(index=False))

    head = (
        "**Baseline pipeline test (4 subjects: S1, S5 Veh + S4, S8 Oxy10)**\n"
        "End-to-end check before committing to the 12-subject overnight run. "
        "2 from each dose extreme so we can also see whether baseline ratios "
        "are consistent across animals (they should be — drug isn't given "
        "until after baseline).\n"
        "Two overlay PNGs (baseline + matching formalin per subject) + per-session "
        "contour HL/HR ratio in the CSV.\n"
        "If the baseline ratios cluster well below 1.0, that supports the "
        "setup-asymmetry hypothesis and the high-dose formalin ratios are at "
        "the achievable ceiling."
    )
    discord_upload(head, [bl_png, fm_png, summary_csv])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
