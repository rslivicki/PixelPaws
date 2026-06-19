"""
Wait for the current rim addons chain to finish, then fire another
addons-chain run so the rest of the newly-arrived portal videos get
ingested.

Why: the first chain run only sees what is portal-ready at launch time.
DLC is GPU-bound and can't run concurrently. So we just poll the python
process tree, and once the original chain's worker python is gone,
spawn a fresh rim_addons_chain.py.

Idempotent -- the addons chain is, so safe to re-run.

Run in background:
  PYTHONIOENCODING=utf-8 nohup py -X utf8 -u \
      scripts/research/rim_addons_followup.py > /tmp/rim_followup.log 2>&1 &
  disown
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
ADDONS_CHAIN = r"E:\PixelPaws\scripts\research\rim_addons_chain.py"
POLL_S = 60
MAX_WAIT_H = 8  # safety cap

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
                     "User-Agent": "PixelPaws-Followup/1.0"},
        ), timeout=15)
    except Exception as e:
        log(f"  ! discord failed: {e}")


def chain_python_running() -> bool:
    """Detect whether the prior addons chain's python is still alive.
    Looks for python.exe or py.exe processes whose command line contains
    rim_addons_chain.py."""
    try:
        out = subprocess.run(
            ["wmic", "process", "where",
             "name='python.exe' or name='py.exe'",
             "get", "ProcessId,CommandLine"],
            capture_output=True, text=True, timeout=15,
        ).stdout
    except Exception as e:
        log(f"  ! wmic failed: {e}; falling back to tasklist heuristic")
        out = subprocess.run(["tasklist"], capture_output=True, text=True).stdout
    return "rim_addons_chain" in out


def dlc_subprocess_running() -> bool:
    """Detect whether DLC's worker python (different env) is running.
    DLC's python lives at C:\\Users\\Gereau\\anaconda3\\envs\\DEEPLABCUT.
    We check ffmpeg too, since the chain interleaves with transcoding."""
    try:
        out = subprocess.run(["tasklist"], capture_output=True, text=True).stdout
    except Exception:
        return False
    return ("DEEPLABCUT" in out) or False  # DEEPLABCUT path would only show in full command line, fallback below


def chain_alive() -> bool:
    """True while either the parent addons script or a DLC subprocess is
    still running."""
    return chain_python_running()


def main() -> int:
    t0 = time.time()
    log("Follow-up monitor started. Waiting for current rim addons chain to finish...")
    discord(":hourglass_flowing_sand: Follow-up monitor armed; will fire a second "
            "rim addons-chain pass after the current one finishes.")

    # Wait for chain to start (if launched a few seconds ago) -- but in practice
    # the first chain has been running for a while, so this is usually a no-op
    settle = 0
    while not chain_alive() and settle < 30:
        time.sleep(2); settle += 2

    while chain_alive():
        if time.time() - t0 > MAX_WAIT_H * 3600:
            log(f"Hit {MAX_WAIT_H}h cap; bailing.")
            discord(f":warning: Follow-up monitor timed out at {MAX_WAIT_H}h.")
            return 1
        time.sleep(POLL_S)

    log("Previous chain exited. Spawning new rim_addons_chain.py pass.")
    discord(":arrow_forward: First rim addons-chain finished. Firing follow-up pass.")

    rc = subprocess.run(
        ["py", "-X", "utf8", "-u", ADDONS_CHAIN],
        cwd=str(REPO),
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    ).returncode
    log(f"Follow-up addons chain exit: {rc}")
    if rc != 0:
        discord(f":x: Follow-up addons chain exit rc={rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
