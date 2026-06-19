"""
Re-export the three Scratching-classifier figures as SVG (vector) and
post to #results.

Reuses the cached results from the prior `regen_scratching_plots.py` run
so no retraining / no SHAP recompute is needed:
  - Panel B (learning curve)   <- cached `lc` list-of-tuples
  - Panel C (threshold sweep)  <- cached `oof` + `y`
  - Panel D (SHAP beeswarm)    <- cached `final_model` + cached SHAP-on-sample

The plotting functions live in `regen_scratching_plots.py`; we import
them and just redirect the output to SVG instead of PNG.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
import urllib.request
import uuid
from pathlib import Path

REPO = Path(r"E:\PixelPaws")
sys.path.insert(0, str(REPO / "scripts" / "research"))

# Project / cache locations (from the prior regen run)
PROJECT = Path(r"E:\RSVIDS\Blackbox\2510_Blackbox_Rimonabant\Blackbox_videos-selected")
REGEN_DIR = PROJECT / "Classifiers" / "plots" / "regen_20260429_140020"
CACHE_PKL = REGEN_DIR / "_regen_cache.pkl"

OUT_DIR = PROJECT / "Classifiers" / "plots" / "svg_exports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
        ext = p.suffix.lower()
        mime = {".svg": "image/svg+xml", ".png": "image/png"}.get(
            ext, "application/octet-stream"
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
                 "User-Agent": "PixelPaws-Scratching-SVG/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            step(f"Discord upload: HTTP {r.status}")
    except Exception as e:
        step(f"Discord upload failed: {e}")


def main() -> int:
    if not CACHE_PKL.is_file():
        step(f"! cached pickle not found: {CACHE_PKL}")
        step("  run regen_scratching_plots.py first")
        return 1

    step(f"loading cache: {CACHE_PKL}")
    with open(CACHE_PKL, "rb") as f:
        cache = pickle.load(f)

    step(f"cache keys: {list(cache.keys())}")

    # X and y aren't in the cache — pull them from the training pickle
    if "X" not in cache or "y" not in cache:
        train_pkl = PROJECT / "Classifiers" / "training_data" / "Scratching_train_set.pkl"
        step(f"loading X, y from {train_pkl.name}")
        with open(train_pkl, "rb") as f:
            ts = pickle.load(f)
        # The training pickle is typically {'X': ..., 'y': ..., ...}
        if isinstance(ts, dict):
            cache["X"] = ts.get("X", ts.get("features"))
            cache["y"] = ts.get("y", ts.get("labels"))
        else:
            step(f"! unexpected training-pickle type: {type(ts)}; keys/attrs: "
                 f"{dir(ts)[:20]}")
            return 1
        if cache["X"] is None or cache["y"] is None:
            step(f"! could not pull X/y from training pickle; keys: {list(ts.keys())}")
            return 1
        import numpy as np
        cache["y"] = np.asarray(cache["y"])
        step(f"  X shape: {getattr(cache['X'], 'shape', '?')}  "
             f"y length: {len(cache['y'])}  pos: {int(cache['y'].sum())}")

    from regen_scratching_plots import (
        plot_learning_curve, plot_threshold_curves, plot_shap_panel,
    )

    behavior = "Scratching"

    lc_svg = OUT_DIR / "PixelPaws_Scratching_panelB_learning.svg"
    step("rendering Panel B (learning curve) -> SVG...")
    plot_learning_curve(cache["lc"], behavior, str(lc_svg))
    step(f"  -> {lc_svg}  ({lc_svg.stat().st_size/1024:.0f} KB)")

    thr_svg = OUT_DIR / "PixelPaws_Scratching_panelC_threshold.svg"
    step("rendering Panel C (threshold curves) -> SVG...")
    plot_threshold_curves(cache["y"], cache["oof"], behavior, str(thr_svg))
    step(f"  -> {thr_svg}  ({thr_svg.stat().st_size/1024:.0f} KB)")

    shap_svg = OUT_DIR / "PixelPaws_Scratching_panelD_shap.svg"
    step("rendering Panel D (SHAP beeswarm + bar) -> SVG"
         " — sampling 3500 rows so the SVG stays under Discord's 10 MB limit")
    plot_shap_panel(cache["final_model"], cache["X"], behavior, str(shap_svg),
                    sample_n=3500)
    step(f"  -> {shap_svg}  ({shap_svg.stat().st_size/1024:.0f} KB)")

    head = (
        "**Scratching classifier — vector exports**\n"
        "Re-rendered through matplotlib's SVG backend (every line, dot, "
        "and axis is an editable vector object). Reused the cached CV "
        "results from `regen_20260429_140020/_regen_cache.pkl` so no "
        "re-training was needed.\n"
        "• `panelB_learning.svg` — F1 vs bout-positive frames (5-fold CV)\n"
        "• `panelC_threshold.svg` — Precision / Recall / F1 vs threshold (OOF)\n"
        "• `panelD_shap.svg` — top-12 SHAP beeswarm + mean(|SHAP|) bars"
    )
    discord_upload(head, [lc_svg, thr_svg, shap_svg])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
