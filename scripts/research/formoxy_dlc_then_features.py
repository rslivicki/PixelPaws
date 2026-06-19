"""
Orchestrator: run DLC on formoxy formalin videos, then feature
extraction, posting status to Discord.

Loops: every few minutes, scan the cohort for any formalin video that
needs DLC or features. If anything is pending, run the appropriate
stage. Stops when all current formalin videos in the cohort have both
filtered .h5 and feature pickles.

Run in background (regular PixelPaws env):
  PYTHONIOENCODING=utf-8 nohup py -X utf8 -u \
      scripts/research/formoxy_dlc_then_features.py > /tmp/formoxy_pipeline.log 2>&1 &
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
DLC_PYTHON = r"C:\Users\Gereau\anaconda3\envs\DEEPLABCUT\python.exe"
DLC_BATCH = r"E:\PixelPaws\scripts\research\dlc_batch_formoxy.py"
FEATURES = r"E:\PixelPaws\scripts\research\formoxy_features_extract.py"

COHORT = Path(r"E:\RSVIDS\Blackbox\2605_FormOxy_LeftPaws")
VIDS = COHORT / "videos"
FEATS = COHORT / "features"

SHUFFLE = 9
POLL_S = 120
IDLE_LIMIT_MIN = 30
MAX_TOTAL_H = 12

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
                     "User-Agent": "PixelPaws-FormOxy-Pipeline/1.0"},
        ), timeout=15)
    except Exception as e:
        log(f"  ! discord failed: {e}")


def formalin_videos() -> list[Path]:
    return sorted(p for p in VIDS.glob("2605_FormOxy_S*_formalin.mp4")
                  if "DLC_" not in p.name and "_labeled" not in p.name)


def needs_dlc(v: Path) -> bool:
    return not list(v.parent.glob(f"{v.stem}*shuffle{SHUFFLE}*_filtered.h5"))


def needs_features(v: Path) -> bool:
    # Any feature pickle matching the stem
    return not list(FEATS.glob(f"{v.stem}_features_*.pkl"))


def signature() -> tuple[int, int, int]:
    vids = formalin_videos()
    return (
        len(vids),
        sum(0 if needs_dlc(v) else 1 for v in vids),
        sum(0 if needs_features(v) else 1 for v in vids),
    )


def run_dlc() -> int:
    log("DLC pass starting (formalin only)")
    rc = subprocess.run(
        [DLC_PYTHON, DLC_BATCH],
        cwd=str(REPO),
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    ).returncode
    log(f"DLC exit: {rc}")
    return rc


def run_features() -> int:
    log("Feature extraction pass starting (formalin only)")
    rc = subprocess.run(
        ["py", "-X", "utf8", "-u", FEATURES],
        cwd=str(REPO),
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    ).returncode
    log(f"Features exit: {rc}")
    return rc


def main() -> int:
    t_start = time.time()
    discord(":robot: FormOxy formalin pipeline armed: DLC -> features.")
    log("FormOxy pipeline started.")
    last_change = time.time()
    last_sig = signature()

    while True:
        if time.time() - t_start > MAX_TOTAL_H * 3600:
            discord(f":warning: FormOxy pipeline hit {MAX_TOTAL_H}h cap, exiting.")
            return 1

        vids = formalin_videos()
        if not vids:
            log("No formalin videos in cohort yet; waiting.")
            time.sleep(POLL_S)
            continue

        need_dlc = [v for v in vids if needs_dlc(v)]
        need_feat = [v for v in vids if not needs_dlc(v) and needs_features(v)]

        if need_dlc:
            log(f"DLC pending: {len(need_dlc)} formalin video(s) -- "
                f"{', '.join(v.stem for v in need_dlc)}")
            discord(f":gear: DLC pass on {len(need_dlc)} formalin video(s): "
                    f"{', '.join(v.stem for v in need_dlc)}")
            run_dlc()
        if need_feat:
            log(f"Features pending: {len(need_feat)} video(s) -- "
                f"{', '.join(v.stem for v in need_feat)}")
            discord(f":bar_chart: Feature extraction on {len(need_feat)} formalin video(s)")
            run_features()

        sig = signature()
        log(f"After pass: total_formalin={sig[0]} dlc_done={sig[1]} features_done={sig[2]}")
        if sig != last_sig:
            last_sig = sig
            last_change = time.time()

        # Done if every formalin video has both DLC and features
        if sig[0] > 0 and sig[1] == sig[0] and sig[2] == sig[0]:
            log(f"All {sig[0]} formalin videos have DLC + features. Done.")
            discord(f":white_check_mark: FormOxy formalin DLC + features complete: "
                    f"{sig[0]}/{sig[0]} sessions.")
            return 0

        # Idle check: if no progress and no new videos in IDLE_LIMIT_MIN, exit
        if (time.time() - last_change) > IDLE_LIMIT_MIN * 60:
            log(f"Idle for {IDLE_LIMIT_MIN} min with no progress; exiting.")
            discord(f":zzz: FormOxy pipeline exiting after {IDLE_LIMIT_MIN} min idle. "
                    f"State: {sig[1]}/{sig[0]} DLC, {sig[2]}/{sig[0]} features.")
            return 0

        time.sleep(POLL_S)


if __name__ == "__main__":
    raise SystemExit(main())
