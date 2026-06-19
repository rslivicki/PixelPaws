"""
Long-running watcher that repeatedly runs formoxy_addons_chain.py as
new oxy/formalin videos arrive in the portal. Stops when the cohort
has 24 sessions (12 subjects x 2 conditions) OR after a safety cap.

Cadence:
  - Run formoxy_addons_chain.py once.
  - If we already have 24 sessions, exit.
  - Otherwise poll the portal for the next batch of ready files; the
    moment any new ready file appears (or a previously in-flight
    syncthing transfer finishes), run the chain again.
  - If nothing happens for IDLE_LIMIT_MIN minutes, exit (assume
    end-of-day done).

Discord-posts at each pass; the chain itself also posts.

Run in background:
  PYTHONIOENCODING=utf-8 nohup py -X utf8 -u \
      scripts/research/formoxy_addons_loop.py > /tmp/formoxy_loop.log 2>&1 &
  disown
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

REPO = Path(r"E:\PixelPaws")
CHAIN = r"E:\PixelPaws\scripts\research\formoxy_addons_chain.py"

PORTAL = Path(r"E:\RSVIDS\Video_transfer_portal\oxyformalin")
COHORT_VIDS = Path(r"E:\RSVIDS\Blackbox\2605_FormOxy_LeftPaws\videos")

TARGET_SESSIONS = 24  # 12 subjects x (baseline + formalin)
POLL_S = 60
IDLE_LIMIT_MIN = 180   # exit after 3 h of nothing changing
MAX_TOTAL_H = 18

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
                     "User-Agent": "PixelPaws-FormOxy-Loop/1.0"},
        ), timeout=15)
    except Exception as e:
        log(f"  ! discord failed: {e}")


def count_cohort_sessions() -> int:
    if not COHORT_VIDS.is_dir():
        return 0
    return sum(1 for p in COHORT_VIDS.glob("2605_FormOxy_S*_*.mp4"))


def portal_signature() -> tuple[int, int]:
    """(n_ready_mp4, n_in_flight) so we can detect change."""
    if not PORTAL.is_dir():
        return (0, 0)
    ready = 0
    in_flight = 0
    for p in PORTAL.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith("~syncthing~") or p.name.endswith(".tmp"):
            in_flight += 1
        elif p.suffix.lower() == ".mp4":
            ready += 1
    return (ready, in_flight)


def another_chain_running() -> bool:
    """True if a separately-launched formoxy_addons_chain.py is alive."""
    try:
        out = subprocess.run(
            ["wmic", "process", "where",
             "name='python.exe' or name='py.exe'",
             "get", "ProcessId,CommandLine"],
            capture_output=True, text=True, timeout=15,
        ).stdout
    except Exception:
        return False
    # Filter to lines mentioning the chain script and NOT this loop script
    hits = [ln for ln in out.splitlines()
            if "formoxy_addons_chain" in ln
            and "formoxy_addons_loop" not in ln]
    # The loop's own subprocess.run will also show during run_chain(), but
    # those are the chain we just launched -- handled by run_chain() blocking.
    # Here we only care about a SEPARATELY launched chain (e.g. the one
    # the user started by hand before the loop).
    return any(hits)


def run_chain() -> int:
    # If a chain is already running externally, wait for it before launching.
    if another_chain_running():
        log("External formoxy_addons_chain.py detected; waiting for it to finish")
        while another_chain_running():
            time.sleep(POLL_S)
        log("External chain exited.")

    log("Running formoxy_addons_chain.py")
    rc = subprocess.run(
        ["py", "-X", "utf8", "-u", CHAIN],
        cwd=str(REPO),
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    ).returncode
    log(f"chain exit: {rc}")
    return rc


def main() -> int:
    t_start = time.time()
    discord(":eyes: FormOxy ingest loop armed. Target: 24 sessions "
            "(12 subjects x baseline+formalin). Will poll the portal "
            "and run the chain as new videos arrive.")
    log("FormOxy ingest loop started.")

    last_change = time.time()
    last_sig = portal_signature()

    while True:
        if time.time() - t_start > MAX_TOTAL_H * 3600:
            log(f"Hit {MAX_TOTAL_H}h total cap; exiting.")
            discord(f":warning: FormOxy loop hit {MAX_TOTAL_H}h cap, exiting. "
                    f"{count_cohort_sessions()}/{TARGET_SESSIONS} sessions ingested.")
            return 1

        run_chain()
        n_sessions = count_cohort_sessions()
        log(f"After pass: {n_sessions}/{TARGET_SESSIONS} cohort sessions on disk.")

        if n_sessions >= TARGET_SESSIONS:
            log("Target reached; exiting.")
            discord(f":tada: FormOxy ingest complete: "
                    f"{n_sessions}/{TARGET_SESSIONS} sessions on disk.")
            return 0

        # Wait for portal to change (new ready file or in-flight count drops)
        while True:
            if time.time() - t_start > MAX_TOTAL_H * 3600:
                break
            sig = portal_signature()
            if sig != last_sig:
                log(f"Portal change detected: {last_sig} -> {sig}")
                last_change = time.time()
                last_sig = sig
                # Settle: wait a beat in case more files are still being written
                time.sleep(15)
                break
            if (time.time() - last_change) > IDLE_LIMIT_MIN * 60:
                log(f"Idle for {IDLE_LIMIT_MIN} min with no portal change; "
                    f"exiting (assume end of acquisition for the day).")
                discord(f":zzz: FormOxy loop exiting after {IDLE_LIMIT_MIN} min of "
                        f"idle. Ingested {n_sessions}/{TARGET_SESSIONS} sessions. "
                        f"Re-run `formoxy_addons_chain.py` (or this loop) when "
                        f"more arrive.")
                return 0
            time.sleep(POLL_S)


if __name__ == "__main__":
    raise SystemExit(main())
