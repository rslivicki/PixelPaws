"""
Pre-transcode Rim13/14/15 post-rimonabant videos at CRF 26 into the
cohort folder while the main chain's DLC pass is occupying the GPU.
CPU-only ffmpeg work; safe to run concurrently with DLC.

Behaviour mirrors rim_addons_chain.transcode_new exactly: libx264
crf=26 slower preset, rename to fem_postrim_Rim<N>.mp4, unlink portal
copy after success.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path

PORTAL_DIR = Path(r"E:\RSVIDS\Video_transfer_portal\2605_Rimonabant\2026-05-18\post-rimonabant")
COHORT_VIDS = Path(r"E:\RSVIDS\Blackbox\260515_Rim_DoseResp\videos")
CRF_TARGET = 26

WEBHOOK = (
    ""
)

_PORTAL_NAME = re.compile(
    r"^Rim(\d+)_(baseline|post-?rim(?:onabant)?)\.mp4$", re.IGNORECASE
)


def step(msg: str) -> None:
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', flush=True)


def discord(msg: str) -> None:
    try:
        urllib.request.urlopen(urllib.request.Request(
            WEBHOOK,
            data=json.dumps({"content": msg}).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json",
                     "User-Agent": "PixelPaws-Pretranscode/1.0"},
        ), timeout=15)
    except Exception as e:
        step(f"  ! discord failed: {e}")


def main() -> int:
    if not PORTAL_DIR.is_dir():
        step(f"!! portal dir does not exist: {PORTAL_DIR}")
        return 1
    portal_mp4s = sorted(PORTAL_DIR.glob("Rim*post-rimonabant.mp4"))
    if not portal_mp4s:
        step("No post-rimonabant mp4s in portal. Nothing to do.")
        return 0

    step(f"Found {len(portal_mp4s)} post-rimonabant mp4(s) in portal.")
    discord(f":fast_forward: Pre-transcoding {len(portal_mp4s)} Rim post-rimonabant "
            f"video(s) at CRF {CRF_TARGET} while DLC runs on the prior batch.")

    for src in portal_mp4s:
        m = _PORTAL_NAME.match(src.name)
        if not m:
            step(f"  !! cannot parse {src.name}, skipping")
            continue
        mouse = int(m.group(1))
        dst = COHORT_VIDS / f"fem_postrim_Rim{mouse}.mp4"
        if dst.is_file():
            step(f"  skip (cohort already has it): {dst.name}")
            try: src.unlink()
            except OSError: pass
            continue
        step(f"  transcode -> {dst.name}  (libx264 crf={CRF_TARGET})")
        t0 = time.time()
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", str(src),
             "-c:v", "libx264",
             "-crf", str(CRF_TARGET),
             "-preset", "slower",
             "-an",
             str(dst)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            step(f"  ! ffmpeg failed for {src.name}: {proc.stderr[-300:]}")
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

    discord(":white_check_mark: Pre-transcode done. DLC will pick these up on the "
            "next addons-chain pass (follow-up monitor will fire it).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
