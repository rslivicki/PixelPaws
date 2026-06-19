"""
Wait for the in-flight DLC batch to finish, then run the final pipeline
stages on Rim13/14/15 postrim:
  - rim_dose_response.py  (features + 6-classifier predict + dose-response analysis)
  - combined_rim_scratching_timecourse.py  (refresh the merged scratching plot)

Detects DLC by looking for python.exe running deeplabcut.analyze_videos
(via env path or script name).
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
ANALYZE = r"E:\PixelPaws\scripts\research\rim_dose_response.py"
TIMECOURSE = r"E:\PixelPaws\scripts\research\combined_rim_scratching_timecourse.py"
POLL_S = 60
MAX_WAIT_H = 8

WEBHOOK = (
    ""
)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def discord(msg: str) -> None:
    try:
        urllib.request.urlopen(urllib.request.Request(
            WEBHOOK,
            data=json.dumps({"content": msg}).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json",
                     "User-Agent": "PixelPaws-Finalize/1.0"},
        ), timeout=15)
    except Exception as e:
        log(f"  ! discord failed: {e}")


def dlc_running() -> bool:
    """True while DLC's worker python (DEEPLABCUT env) is alive OR the
    dlc_batch_rim_dose.py wrapper is running."""
    try:
        out = subprocess.run(
            ["wmic", "process", "where",
             "name='python.exe' or name='py.exe'",
             "get", "ProcessId,CommandLine"],
            capture_output=True, text=True, timeout=15,
        ).stdout
    except Exception:
        return False
    return ("dlc_batch_rim_dose" in out) or ("DEEPLABCUT" in out)


def main() -> int:
    t0 = time.time()
    log("Finalize monitor armed. Waiting for DLC batch to finish...")

    # Settle: give the DLC subprocess a moment to come up if we just launched it
    settle = 0
    while not dlc_running() and settle < 30:
        time.sleep(2); settle += 2

    while dlc_running():
        if time.time() - t0 > MAX_WAIT_H * 3600:
            log(f"Hit {MAX_WAIT_H}h cap; bailing.")
            discord(f":warning: Finalize monitor timed out at {MAX_WAIT_H}h.")
            return 1
        time.sleep(POLL_S)

    log("DLC batch exited. Running rim_dose_response.py")
    discord(":arrow_forward: DLC done on Rim13/14/15 postrim. "
            "Running features + 6-classifier predict + dose-response.")

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    rc = subprocess.run(["py", "-X", "utf8", "-u", ANALYZE], cwd=str(REPO), env=env).returncode
    log(f"rim_dose_response exit: {rc}")
    if rc != 0:
        discord(f":x: rim_dose_response failed rc={rc}")

    log("Running combined scratching timecourse refresh")
    rc = subprocess.run(["py", "-X", "utf8", "-u", TIMECOURSE], cwd=str(REPO), env=env).returncode
    log(f"timecourse exit: {rc}")
    if rc != 0:
        discord(f":x: timecourse failed rc={rc}")
    else:
        discord(":white_check_mark: All 30 rim sessions finalised. "
                "Dose-response + combined scratching figures refreshed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
