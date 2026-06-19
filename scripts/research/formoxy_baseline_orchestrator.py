"""
End-to-end pipeline for the 12 baseline mp4s in the 2605_FormOxy_LeftPaws
cohort (no formalin yet). Runs in three stages:

  1. DLC analyze + filterpredictions on every baseline mp4 missing a
     filtered .h5 (calls `dlc_batch_formoxy.py --include-baseline` in
     the DEEPLABCUT conda env). The script is idempotent and already-done
     formalin sessions are skipped; only baselines remain pending.

  2. Paw brightness extraction (same script we used for the formalin
     pass; its glob has been widened to `*_S*_*.mp4` so baselines are
     picked up, while formalin csv caches are skipped).

  3. Paw contour extraction (likewise).

After all three complete, posts a status line to the general Discord
channel; the baseline-vs-formalin comparison plot is a separate, faster
script (`formoxy_baseline_vs_formalin_contour_ratio.py`) launched once
this orchestrator finishes.

Run from regular py env:
  PYTHONIOENCODING=utf-8 py -X utf8 -u \
    scripts/research/formoxy_baseline_orchestrator.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

REPO = Path(r"E:\PixelPaws")

DLC_PYTHON = r"C:\Users\Gereau\anaconda3\envs\DEEPLABCUT\python.exe"
DLC_BATCH  = REPO / "scripts" / "research" / "dlc_batch_formoxy.py"
BRT_SCRIPT = REPO / "scripts" / "research" / "formoxy_paw_brightness_extract.py"
CTR_SCRIPT = REPO / "scripts" / "research" / "formoxy_paw_contour_extract.py"

COHORT = Path(r"E:\RSVIDS\Blackbox\2605_FormOxy_LeftPaws")
VIDS = COHORT / "videos"

# General chain channel (not #results — that's for finished analyses)
WEBHOOK = (
    ""
)


def step(msg: str) -> None:
    line = f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)


def discord(msg: str) -> None:
    try:
        urllib.request.urlopen(urllib.request.Request(
            WEBHOOK,
            data=json.dumps({"content": msg}).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json",
                     "User-Agent": "PixelPaws-FormOxy-Baseline/1.0"},
        ), timeout=15)
    except Exception as e:
        step(f"  ! discord failed: {e}")


def baseline_video_count() -> int:
    return sum(1 for p in VIDS.glob("2605_FormOxy_S*_baseline.mp4")
               if "DLC_" not in p.name)


def baseline_filtered_h5_count() -> int:
    return sum(1 for p in VIDS.glob("2605_FormOxy_S*_baseline*shuffle9*_filtered.h5"))


def baseline_brt_count() -> int:
    out = COHORT / "gait_limb_analysis"
    return sum(1 for p in out.glob("2605_FormOxy_S*_baseline_brt_*.csv"))


def baseline_contour_count() -> int:
    out = COHORT / "gait_limb_analysis"
    return sum(1 for p in out.glob("2605_FormOxy_S*_baseline_contour_*.csv"))


def run_subprocess(args, env_overrides=None, label="step") -> int:
    step(f"--- {label} ---")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    if env_overrides:
        env.update(env_overrides)
    rc = subprocess.run(args, cwd=str(REPO), env=env).returncode
    step(f"--- {label} exit: {rc} ---")
    return rc


def main() -> int:
    t_start = time.time()
    n_baseline = baseline_video_count()
    step(f"Baselines on disk: {n_baseline}")
    if n_baseline == 0:
        step("No baseline mp4s found; nothing to do."); return 0

    discord(f":robot: FormOxy baseline pipeline armed: "
            f"{n_baseline} baseline videos, DLC -> brightness -> contour.")

    # Stage 1: DLC (analyze + filterpredictions), --include-baseline
    rc = run_subprocess(
        [DLC_PYTHON, str(DLC_BATCH), "--include-baseline"],
        label="DLC (analyze + filter, baseline + already-done formalin skipped)",
    )
    if rc != 0:
        discord(f":warning: FormOxy baseline DLC exited with rc={rc}; "
                f"continuing to brightness pass anyway.")
    n_filt = baseline_filtered_h5_count()
    step(f"Baseline filtered .h5 count: {n_filt}/{n_baseline}")
    discord(f":gear: DLC pass done. Baseline filtered .h5: {n_filt}/{n_baseline}.")

    # Stage 2: Brightness extraction (widened glob -> baseline + formalin;
    # formalin csvs are cached and skipped)
    rc = run_subprocess(
        ["py", "-X", "utf8", "-u", str(BRT_SCRIPT)],
        label="Paw brightness extraction (baseline pass)",
    )
    n_brt = baseline_brt_count()
    step(f"Baseline brt CSVs: {n_brt}/{n_baseline}")
    discord(f":bulb: Brightness pass done. Baseline brt CSVs: {n_brt}/{n_baseline}.")

    # Stage 3: Contour extraction (same idempotency story)
    rc = run_subprocess(
        ["py", "-X", "utf8", "-u", str(CTR_SCRIPT)],
        label="Paw contour extraction (baseline pass)",
    )
    n_ctr = baseline_contour_count()
    step(f"Baseline contour CSVs: {n_ctr}/{n_baseline}")
    discord(f":paw_prints: Contour pass done. Baseline contour CSVs: "
            f"{n_ctr}/{n_baseline}.")

    elapsed_h = (time.time() - t_start) / 3600
    discord(
        f":white_check_mark: FormOxy baseline pipeline complete in "
        f"{elapsed_h:.2f} h. Baselines processed: "
        f"DLC {n_filt}/{n_baseline}, brt {n_brt}/{n_baseline}, "
        f"contour {n_ctr}/{n_baseline}. "
        f"Run `formoxy_baseline_vs_formalin_contour_ratio.py` for the "
        f"comparison plot."
    )
    step(f"All baseline stages done in {elapsed_h*60:.0f} min.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
