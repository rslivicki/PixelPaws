"""
Re-send the PNG versions of the figures we previously uploaded as PDF /
SVG, plus a sample raster-evaluation PNG (the 3-panel layout the user
asked about).
"""
from __future__ import annotations
import json, urllib.request, uuid, time
from pathlib import Path


WEBHOOK = (
    ""
)


def step(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def discord_upload(content, files):
    boundary = "----PP" + uuid.uuid4().hex
    body = [
        f"--{boundary}".encode(),
        b'Content-Disposition: form-data; name="payload_json"',
        b'Content-Type: application/json', b"",
        json.dumps({"content": content}).encode(),
    ]
    for i, p in enumerate(files):
        mime = {"png": "image/png", "csv": "text/csv"}.get(
            p.suffix.lower().lstrip("."), "application/octet-stream"
        )
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
                 "User-Agent": "PixelPaws-PNG-Resend/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            step(f"  HTTP {r.status}")
    except Exception as e:
        step(f"  upload failed: {e}")


# Files grouped by upload (Discord webhook max ~10 files / 25 MB per post)
GROUP_PRIOR = [
    Path(r"E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\transitions\_classifier_f1.png"),
    Path(r"E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\analysis\transitions\temporal_probability_corrected.png"),
    Path(r"E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\analysis\sntx_paw_contour_figure.png"),
    Path(r"E:\RSVIDS\Blackbox\2605_FormOxy_LeftPaws\analysis\contour_intensity_ratio_tc.png"),
    Path(r"E:\RSVIDS\Blackbox\2605_FormOxy_LeftPaws\analysis\mean_contour_HL_HR.png"),
]
GROUP_SCRATCHING = [
    Path(r"E:\RSVIDS\Blackbox\2510_Blackbox_Rimonabant\Blackbox_videos-selected\Classifiers\plots\regen_20260429_140020\PixelPaws_Scratching_panelB_learning.png"),
    Path(r"E:\RSVIDS\Blackbox\2510_Blackbox_Rimonabant\Blackbox_videos-selected\Classifiers\plots\regen_20260429_140020\PixelPaws_Scratching_panelC_threshold.png"),
    Path(r"E:\RSVIDS\Blackbox\2510_Blackbox_Rimonabant\Blackbox_videos-selected\Classifiers\plots\regen_20260429_140020\PixelPaws_Scratching_panelD_shap.png"),
]
# A representative raster-evaluation PNG (matches the 3-panel layout
# the user referenced: ethogram | confusion matrix | binned correlation)
GROUP_RASTER_SAMPLES = [
    Path(r"E:\RSVIDS\Blackbox\2510_Blackbox_Rimonabant\Blackbox_videos-selected\Classifiers\plots\PixelPaws_Scratching_Raster_251030_S1_Rimonabant.png"),
    Path(r"E:\RSVIDS\Blackbox\2510_Blackbox_Rimonabant\Blackbox_videos-selected\Classifiers\plots\PixelPaws_Scratching_Raster_251031_S4_F_Rimonabant.png"),
    Path(r"E:\RSVIDS\Blackbox\2511_Blackbox_Formalin\PixelPaws\Flinching\classifiers\plots\PixelPaws_L_flinching_Raster_260129_Formalin_3128.png"),
    Path(r"E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\classifiers\plots\PixelPaws_body_grooming_Raster_260318_JG_9435_Baseline.png"),
]


def main():
    # 1) prior figures (non-Scratching)
    files = [p for p in GROUP_PRIOR if p.is_file()]
    missing = [p for p in GROUP_PRIOR if not p.is_file()]
    for p in missing:
        step(f"  ! missing: {p}")
    if files:
        step(f"upload 1: prior figures ({len(files)} files)")
        discord_upload(
            "**Prior figures — PNG versions**\n"
            "Same content as the PDFs already posted; just easier-to-preview "
            "PNGs alongside them.",
            files,
        )

    # 2) Scratching panels
    files = [p for p in GROUP_SCRATCHING if p.is_file()]
    if files:
        step(f"upload 2: Scratching panels ({len(files)} files)")
        discord_upload(
            "**Scratching classifier panels — PNG versions**\n"
            "Panel B (learning curve), Panel C (threshold sweep), Panel D "
            "(SHAP beeswarm + bars). All from the cached CV run.",
            files,
        )

    # 3) Raster eval samples
    files = [p for p in GROUP_RASTER_SAMPLES if p.is_file()]
    if files:
        step(f"upload 3: raster-evaluation samples ({len(files)} files)")
        discord_upload(
            "**Raster-evaluation figures** (the 3-panel layout you "
            "asked about) — produced by `_generate_session_raster_plots()` "
            "in `evaluation_tab.py:1879`. Three panels per session: "
            "**raster** (model vs human, per-frame), **confusion matrix** "
            "with F1, **10-s-binned** time series with Pearson R. Sending "
            "a few samples from different cohorts so you can identify the "
            "exact one — let me know which.",
            files,
        )


if __name__ == "__main__":
    main()
