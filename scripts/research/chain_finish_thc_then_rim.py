"""
Sequential pipeline:
  1. Re-run THC rim orchestrator -- processes the 13 sessions added
     after the orchestrator's first wave finished (idempotent batch
     script picks them up).
  2. Wait for THC to be fully done (25 filtered .h5 + 25 predictions).
  3. Transcode the 18 new rim videos from the portal at libx264 CRF 26
     (slower preset) and move into the new dose-response cohort folder.
  4. DLC + features + 6-classifier predict (incl Scratching) on the rim
     cohort.
  5. Dose-response analysis + Discord post.

Run in background:
  py -X utf8 -u scripts/research/chain_finish_thc_then_rim.py \
      >> E:/RSVIDS/Blackbox/chain_thc_then_rim.log 2>&1
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

REPO = Path(r"E:\PixelPaws")
DLC_PYTHON = r"C:\Users\Gereau\anaconda3\envs\DEEPLABCUT\python.exe"

# THC follow-up
THC_DIR = Path(r"E:\RSVIDS\Blackbox\260512_THC_Rim_Cohort\videos")
THC_TARGET = 25
THC_DLC_BATCH = r"E:\PixelPaws\scripts\research\dlc_batch_thc_rim.py"
THC_PREDICT = r"E:\PixelPaws\scripts\research\thc_rim_predict_all.py"
THC_GROUP = r"E:\PixelPaws\scripts\research\thc_rim_group_analyze.py"

# Rim dose-response cohort
PORTAL_BASE = Path(r"E:\RSVIDS\Video_transfer_portal\2605_Rimonabant\2026-05-15")
RIM_COHORT = Path(r"E:\RSVIDS\Blackbox\260515_Rim_DoseResp")
RIM_VIDS = RIM_COHORT / "videos"
RIM_CLASSIFIERS = RIM_COHORT / "classifiers"
THC_WITHDRAWAL_CLASSIFIERS = Path(r"E:\RSVIDS\Blackbox\260506_RS_THC_Withdrawal\classifiers")
RIM_DLC_BATCH = r"E:\PixelPaws\scripts\research\dlc_batch_rim_dose.py"
RIM_ANALYZE = r"E:\PixelPaws\scripts\research\rim_dose_response.py"

CRF_TARGET = 26

WEBHOOK = (
    ""
)


def step(msg: str) -> None:
    line = f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    try: print(line, flush=True)
    except UnicodeEncodeError: print(line.encode("ascii","replace").decode(), flush=True)


def discord(msg: str):
    req = urllib.request.Request(
        WEBHOOK,
        data=json.dumps({"content": msg}).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json",
                 "User-Agent": "PixelPaws-Chain/1.0"},
    )
    try:
        urllib.request.urlopen(req, timeout=15)
    except Exception as e:
        step(f"  ! discord failed: {e}")


def count_filtered(folder: Path) -> int:
    return sum(1 for p in folder.iterdir()
               if p.is_file() and "shuffle9" in p.name
               and p.name.endswith("_filtered.h5"))


# --------------------------------------------------------------------------- #
# Phase 1: finish THC
# --------------------------------------------------------------------------- #

def run_thc_dlc_loop() -> int:
    """Run dlc_batch_thc_rim in a loop until target filtered count reached
    (or 5 retries). Each call is idempotent — picks up videos still pending."""
    for attempt in range(1, 6):
        cur = count_filtered(THC_DIR)
        if cur >= THC_TARGET:
            step(f"THC: {cur}/{THC_TARGET} filtered. Done.")
            return 0
        step(f"THC attempt {attempt}: {cur}/{THC_TARGET} done. Launching DLC batch.")
        discord(f"THC batch attempt {attempt} -- {cur}/{THC_TARGET} done so far.")
        rc = subprocess.run(
            [DLC_PYTHON, THC_DLC_BATCH],
            cwd=str(REPO),
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        ).returncode
        step(f"THC DLC batch attempt {attempt} exit: {rc}")
        time.sleep(30)
    step("THC: max retries hit, not at target. Bailing.")
    return 1


def run_thc_predict_and_analyze() -> int:
    """Single-pass features + predict + group analyze on full 25-video set."""
    step("THC: predict_all (idempotent — only new sessions get processed)")
    rc = subprocess.run(
        ["py", "-X", "utf8", "-u", THC_PREDICT],
        cwd=str(REPO),
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    ).returncode
    if rc != 0:
        discord(f":x: THC predict_all failed rc={rc}")
        return rc
    step("THC: group_analyze on full 25-video cohort")
    rc = subprocess.run(
        ["py", "-X", "utf8", "-u", THC_GROUP],
        cwd=str(REPO),
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    ).returncode
    if rc != 0:
        discord(f":x: THC group_analyze failed rc={rc}")
    return rc


# --------------------------------------------------------------------------- #
# Phase 2: transcode + setup rim cohort
# --------------------------------------------------------------------------- #

def setup_rim_cohort() -> int:
    RIM_VIDS.mkdir(parents=True, exist_ok=True)
    (RIM_COHORT / "features").mkdir(exist_ok=True)
    (RIM_COHORT / "results").mkdir(exist_ok=True)
    (RIM_COHORT / "analysis").mkdir(exist_ok=True)
    # Copy classifiers (5 SNLT + Scratching)
    RIM_CLASSIFIERS.mkdir(exist_ok=True)
    for c in THC_WITHDRAWAL_CLASSIFIERS.glob("*.pkl"):
        dst = RIM_CLASSIFIERS / c.name
        if not dst.is_file():
            shutil.copy(c, dst)
    step(f"Cohort folder ready: {RIM_COHORT}")

    # Find new videos in portal (skip session JSONs, only .mp4)
    portal_mp4s = sorted(PORTAL_BASE.rglob("*.mp4"))
    step(f"Found {len(portal_mp4s)} mp4s in portal")
    if not portal_mp4s:
        return 1

    for src in portal_mp4s:
        # Destination: same filename in cohort/videos/
        dst = RIM_VIDS / src.name
        if dst.is_file():
            step(f"  skip (cohort has it): {src.name}")
            continue
        step(f"  transcode -> {dst.name}  (libx264 crf={CRF_TARGET})")
        t0 = time.time()
        cmd = [
            "ffmpeg", "-y", "-i", str(src),
            "-c:v", "libx264",
            "-crf", str(CRF_TARGET),
            "-preset", "slower",
            "-an",
            str(dst),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            step(f"  ! ffmpeg failed: {proc.stderr[-300:]}")
            return 2
        # Remove source from portal once successfully copied (per user
        # workflow: portal -> cohort, like before)
        src.unlink()
        step(f"    {dst.stat().st_size/1e6:.0f} MB (took {time.time()-t0:.0f}s) — removed from portal")

    return 0


# --------------------------------------------------------------------------- #
# Phase 3 + 4: DLC + features + predict + dose response
# --------------------------------------------------------------------------- #

def run_rim_dlc() -> int:
    step("Rim DLC batch starting...")
    discord(":gear: Rim cohort DLC starting (18 sessions, ~25 h).")
    rc = subprocess.run(
        [DLC_PYTHON, RIM_DLC_BATCH],
        cwd=str(REPO),
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    ).returncode
    if rc != 0:
        discord(f":x: Rim DLC batch failed rc={rc}")
    return rc


def run_rim_analysis() -> int:
    step("Rim features + dose-response analysis starting...")
    discord(":bar_chart: Rim features + 6-classifier predict + dose-response analysis.")
    rc = subprocess.run(
        ["py", "-X", "utf8", "-u", RIM_ANALYZE],
        cwd=str(REPO),
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    ).returncode
    if rc != 0:
        discord(f":x: Rim analysis failed rc={rc}")
    return rc


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main():
    t0 = time.time()
    discord("**Chain started**: finish THC (13 unanalyzed sessions) -> "
            "transcode rim videos at CRF 26 -> DLC -> dose-response analysis.")
    step("=== Phase 1: finish THC ===")
    if run_thc_dlc_loop() != 0:
        return 1
    if run_thc_predict_and_analyze() != 0:
        discord(":warning: THC analysis exited non-zero, continuing to rim anyway.")
    else:
        discord(":white_check_mark: THC fully analyzed (25 sessions).")

    step("\n=== Phase 2: setup rim cohort (transcode + move) ===")
    if setup_rim_cohort() != 0:
        discord(":x: Failed to set up rim cohort. Aborting.")
        return 2
    n_videos = len(list(RIM_VIDS.glob("*.mp4")))
    discord(f":white_check_mark: Rim cohort built: {n_videos} videos at CRF {CRF_TARGET}.")

    step("\n=== Phase 3: rim DLC ===")
    if run_rim_dlc() != 0:
        return 3

    step("\n=== Phase 4: rim features + predict + dose response ===")
    if run_rim_analysis() != 0:
        return 4

    elapsed_h = (time.time() - t0) / 3600
    discord(f":tada: **Chain complete** in {elapsed_h:.1f}h. "
            "THC + Rim dose-response figures posted above.")
    step(f"Chain done in {elapsed_h:.1f}h.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
