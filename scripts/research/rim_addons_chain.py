"""
Idempotent chain for adding new rim animals to the 260515 dose-response cohort.

Picks up every mp4 found in the portal at
  E:\\RSVIDS\\Video_transfer_portal\\2605_Rimonabant\\<date>\\(baseline|post-rim)\\
that is NOT still mid-transfer (no ~syncthing~ prefix, no .tmp suffix,
session_*.json companion exists), and:

  1. Transcodes at libx264 CRF 26 slower preset into the cohort
     videos/ folder, renaming to the cohort convention
     fem_(baseline|postrim)_Rim<N>.mp4.
  2. Removes the portal copy (PawCapture rig still holds the source).
  3. Triggers the rest of the existing chain:
       - dlc_batch_rim_dose.py     (DLC + filterpredictions)
       - rim_dose_response.py      (features + 6-classifier predict + dose response)
       - combined_rim_scratching_timecourse.py  (rescored combined cohorts)

Run in the regular PixelPaws env:

  PYTHONIOENCODING=utf-8 py -X utf8 -u scripts/research/rim_addons_chain.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

REPO = Path(r"E:\PixelPaws")
DLC_PYTHON = r"C:\Users\Gereau\anaconda3\envs\DEEPLABCUT\python.exe"

PORTAL_BASE = Path(r"E:\RSVIDS\Video_transfer_portal\2605_Rimonabant")
COHORT = Path(r"E:\RSVIDS\Blackbox\260515_Rim_DoseResp")
COHORT_VIDS = COHORT / "videos"
CRF_TARGET = 26

DLC_BATCH = r"E:\PixelPaws\scripts\research\dlc_batch_rim_dose.py"
ANALYZE = r"E:\PixelPaws\scripts\research\rim_dose_response.py"
TIMECOURSE = r"E:\PixelPaws\scripts\research\combined_rim_scratching_timecourse.py"

WEBHOOK = (
    ""
)


def step(msg: str) -> None:
    line = f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode(), flush=True)


def discord(msg: str) -> None:
    try:
        req = urllib.request.Request(
            WEBHOOK,
            data=json.dumps({"content": msg}).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json",
                     "User-Agent": "PixelPaws-Chain/1.0"},
        )
        urllib.request.urlopen(req, timeout=15)
    except Exception as e:
        step(f"  ! discord failed: {e}")


# --------------------------------------------------------------------------- #
# Portal discovery
# --------------------------------------------------------------------------- #

# Accepts baseline, post-rim, postrim, post-rimonabant, postrimonabant.
_PORTAL_NAME = re.compile(r"^Rim(\d+)_(baseline|post-?rim(?:onabant)?)\.mp4$", re.IGNORECASE)


def find_ready_videos() -> list[tuple[Path, str, int]]:
    """Return list of (portal_mp4_path, condition, mouse_num) for files that
    have finished transferring (no ~syncthing~/.tmp, JSON companion exists).
    """
    ready = []
    if not PORTAL_BASE.is_dir():
        return ready
    for mp4 in PORTAL_BASE.rglob("*.mp4"):
        name = mp4.name
        if name.startswith("~syncthing~") or name.endswith(".tmp"):
            continue
        m = _PORTAL_NAME.match(name)
        if not m:
            continue
        mouse = int(m.group(1))
        cond_portal = m.group(2).lower().replace("-", "")  # baseline / postrim
        cond = "baseline" if cond_portal == "baseline" else "postrim"
        # Require a session_*.json in the same folder so we only act after
        # PawCapture finished writing metadata.
        json_companion = list(mp4.parent.glob("session_*.json"))
        if not json_companion:
            continue
        ready.append((mp4, cond, mouse))
    return ready


def cohort_dst_name(condition: str, mouse: int) -> str:
    return f"fem_{condition}_Rim{mouse}.mp4"


# --------------------------------------------------------------------------- #
# Stage 1: transcode
# --------------------------------------------------------------------------- #

def transcode_new(ready: list[tuple[Path, str, int]]) -> list[str]:
    """Transcode each new mp4 into the cohort folder. Returns the list of
    cohort-side filenames that were added this run."""
    COHORT_VIDS.mkdir(parents=True, exist_ok=True)
    new_files = []
    for src, cond, mouse in ready:
        dst = COHORT_VIDS / cohort_dst_name(cond, mouse)
        if dst.is_file():
            step(f"  skip transcode (already exists): {dst.name}")
            continue
        step(f"  transcode -> {dst.name}  (libx264 crf={CRF_TARGET})")
        t0 = time.time()
        cmd = ["ffmpeg", "-y", "-i", str(src),
               "-c:v", "libx264",
               "-crf", str(CRF_TARGET),
               "-preset", "slower",
               "-an",
               str(dst)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            step(f"  ! ffmpeg failed for {src.name}: {proc.stderr[-300:]}")
            # Remove a partially-written destination so re-runs retry cleanly
            if dst.is_file():
                dst.unlink()
            continue
        dt = time.time() - t0
        sz_mb = dst.stat().st_size / 1e6
        step(f"    done in {dt:.0f}s, {sz_mb:.0f} MB; removing portal copy")
        try:
            src.unlink()
        except OSError as e:
            step(f"  ! could not remove portal source: {e}")
        new_files.append(dst.name)
    return new_files


# --------------------------------------------------------------------------- #
# Subprocess helpers for downstream stages
# --------------------------------------------------------------------------- #

def run_dlc() -> int:
    step("DLC batch starting on rim cohort (will skip any video that already has filtered .h5)")
    rc = subprocess.run(
        [DLC_PYTHON, DLC_BATCH],
        cwd=str(REPO),
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    ).returncode
    step(f"DLC exit: {rc}")
    return rc


def run_analyze() -> int:
    step("Features + 6-classifier predict + dose-response analysis")
    rc = subprocess.run(
        ["py", "-X", "utf8", "-u", ANALYZE],
        cwd=str(REPO),
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    ).returncode
    step(f"rim_dose_response exit: {rc}")
    return rc


def run_timecourse() -> int:
    step("Refreshing combined-cohort scratching bouts timecourse")
    rc = subprocess.run(
        ["py", "-X", "utf8", "-u", TIMECOURSE],
        cwd=str(REPO),
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    ).returncode
    step(f"timecourse exit: {rc}")
    return rc


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def wait_for_portal(max_wait_s: int = 6 * 3600, poll_s: int = 60) -> list[tuple[Path, str, int]]:
    """Poll the portal until at least one rim video has finished transferring.
    Returns the ready list. Times out after max_wait_s with an empty list."""
    t_start = time.time()
    announced_pending = False
    while True:
        ready = find_ready_videos()
        # Also check for any mid-transfer files so we know to keep waiting
        in_flight = [p for p in PORTAL_BASE.rglob("*.mp4*")
                     if p.name.startswith("~syncthing~") or p.name.endswith(".tmp")]
        if ready:
            return ready
        if not in_flight:
            return []  # nothing to wait for
        if not announced_pending:
            step(f"Portal has {len(in_flight)} mp4(s) still mid-transfer; polling every {poll_s}s")
            announced_pending = True
        if time.time() - t_start > max_wait_s:
            step(f"Timed out waiting for portal after {max_wait_s/3600:.1f}h")
            return []
        time.sleep(poll_s)


def cohort_has_pending_work() -> bool:
    """True if any mp4 in the cohort still lacks a filtered .h5 or
    predictions CSV. Lets us catch-up cases where videos got transcoded
    out-of-band (e.g. by rim13_15_pretranscode.py) and the prior chain's
    DLC subscan missed them."""
    if not COHORT_VIDS.is_dir():
        return False
    mp4s = sorted(COHORT_VIDS.glob("fem_*Rim*.mp4"))
    if not mp4s:
        return False
    results_dir = COHORT / "results"
    feats_dir = COHORT / "features"
    for v in mp4s:
        filt_h5 = list(COHORT_VIDS.glob(f"{v.stem}*shuffle9*_filtered.h5"))
        if not filt_h5:
            return True
        pred = results_dir / f"{v.stem}_predictions.csv"
        feat = list(feats_dir.glob(f"{v.stem}_features_*.pkl"))
        if not pred.is_file() or not feat:
            return True
    return False


def main() -> int:
    wait_mode = "--wait" in sys.argv
    t0 = time.time()
    if wait_mode:
        ready = wait_for_portal()
    else:
        ready = find_ready_videos()

    pending_work = cohort_has_pending_work()
    if not ready and not pending_work:
        step("No new fully-transferred rim videos in portal AND no cohort backlog. "
             "Nothing to do.")
        return 0
    if not ready and pending_work:
        step("Portal empty but cohort has pending DLC/features/predictions. "
             "Running catch-up DLC + analyze.")
        discord(":mag: Cohort backlog detected (videos with no h5/predictions) -- "
                "running catch-up pass.")

    if ready:
        ready.sort(key=lambda r: (r[2], r[1]))  # mouse num, condition
        step("Ready to ingest:")
        for src, cond, m in ready:
            sz = src.stat().st_size / 1e9
            step(f"  Rim{m} {cond}: {src.name} ({sz:.2f} GB)")

        discord(f":inbox_tray: Rim cohort addon: {len(ready)} new video(s) ready -> "
                f"{', '.join(f'Rim{m}_{c}' for _, c, m in ready)}. Starting chain.")

        new_files = transcode_new(ready)
        if new_files:
            discord(f":scissors: Transcoded {len(new_files)} new video(s) at CRF {CRF_TARGET}: "
                    f"{', '.join(new_files)}")

    if run_dlc() != 0:
        discord(":x: DLC stage failed. Aborting.")
        return 1
    if run_analyze() != 0:
        discord(":warning: Analyze stage failed; skipping timecourse refresh.")
        return 2
    if run_timecourse() != 0:
        discord(":warning: Timecourse refresh failed.")
        return 3

    elapsed_h = (time.time() - t0) / 3600
    discord(f":white_check_mark: Rim addon chain complete in {elapsed_h:.1f}h. "
            f"Combined cohort now includes {len(new_files)} new subject(s).")
    step(f"Done in {elapsed_h:.1f}h.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
