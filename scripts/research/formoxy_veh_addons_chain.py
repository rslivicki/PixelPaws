"""Vehicle-paw addons chain — vehicle-injected controls for the 2605 FormOxy
LeftPaws cohort.

For each ready mp4 in
  E:\\RSVIDS\\Video_transfer_portal\\vehicle_paw\\<date>\\(baseline|post-vehicle)\\
the chain:
  1. Classifies the file by regex (vehpaw subject N, baseline/vehicle).
  2. Transcodes at libx264 CRF 26 slower preset, no audio, no flip.
     (These were captured with the new rig orientation — no horizontal
     flip needed, matching the late-batch formalin sessions.)
  3. Drops it into the cohort `videos/` folder under the canonical name
       2605_FormOxy_Veh_S<N>_baseline.mp4
       2605_FormOxy_Veh_S<N>_vehicle.mp4
  4. Removes the portal copy after a successful transcode.
  5. Appends Vehicle-treatment rows to the FormOxy key CSV (idempotent).

Idempotent: re-runs skip already-transcoded files and clean up portal
duplicates.

Run from PixelPaws env:
  py -X utf8 -u scripts/research/formoxy_veh_addons_chain.py [--wait]

`--wait` polls the portal every 60 s until at least one ready file
appears, useful while the rig is still recording.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

PORTAL_BASE = Path(r"E:\RSVIDS\Video_transfer_portal\vehicle_paw")
COHORT = Path(r"E:\RSVIDS\Blackbox\2605_FormOxy_LeftPaws")
COHORT_VIDS = COHORT / "videos"
KEY_CSV = COHORT / "2605_FormOxy_key.csv"
CRF_TARGET = 26

WEBHOOK = (
    ""
)

PORTAL_NAME = re.compile(
    r"^2605_vehpaw_(?:male|female)_(\d+)_(baseline|post-vehicle)\.mp4$",
    re.IGNORECASE,
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
                     "User-Agent": "PixelPaws-FormOxy-VehChain/1.0"},
        )
        urllib.request.urlopen(req, timeout=15)
    except Exception as e:
        step(f"  ! discord failed: {e}")


def cohort_name(mouse: int, cond_portal: str) -> str:
    cond = "baseline" if cond_portal.lower() == "baseline" else "vehicle"
    return f"2605_FormOxy_Veh_S{mouse}_{cond}.mp4"


def list_portal_files() -> list[Path]:
    if not PORTAL_BASE.is_dir():
        return []
    out = []
    for mp4 in PORTAL_BASE.rglob("*.mp4"):
        if mp4.name.startswith("~syncthing~") or mp4.name.endswith(".tmp"):
            continue
        out.append(mp4)
    return out


def classify(src: Path) -> tuple[int, str, str] | None:
    m = PORTAL_NAME.match(src.name)
    if not m:
        return None
    mouse = int(m.group(1))
    cond_portal = m.group(2).lower()
    return (mouse, cond_portal, cohort_name(mouse, cond_portal))


def find_ready() -> list[tuple[Path, int, str, str]]:
    out = []
    for mp4 in list_portal_files():
        cls = classify(mp4)
        if cls is None:
            continue
        mouse, cond_portal, dst_name = cls
        sz = mp4.stat().st_size
        if sz < 200 * 1024 * 1024:
            step(f"  skip {mp4.name}: too small ({sz/1e6:.0f} MB) — likely a partial")
            continue
        out.append((mp4, mouse, cond_portal, dst_name))
    return out


def wait_for_portal(max_wait_s: int = 6 * 3600, poll_s: int = 60) -> list:
    t0 = time.time()
    while True:
        ready = find_ready()
        in_flight = [p for p in PORTAL_BASE.rglob("*.mp4*")
                     if p.name.startswith("~syncthing~") or p.name.endswith(".tmp")]
        if ready:
            return ready
        if not in_flight:
            return []
        if time.time() - t0 > max_wait_s:
            return []
        time.sleep(poll_s)


def transcode(src: Path, dst: Path) -> bool:
    cmd = ["ffmpeg", "-y", "-i", str(src),
           "-c:v", "libx264",
           "-crf", str(CRF_TARGET),
           "-preset", "slower",
           "-an",
           str(dst)]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        step(f"  ! ffmpeg failed: {proc.stderr[-300:]}")
        if dst.is_file():
            dst.unlink()
        return False
    dt = time.time() - t0
    sz_mb = dst.stat().st_size / 1e6
    step(f"    done in {dt:.0f}s, {sz_mb:.0f} MB")
    return True


def append_veh_to_key() -> None:
    """Make sure every Veh_S<N> subject in the cohort folder has a row in
    2605_FormOxy_key.csv with Treatment=Vehicle. Idempotent."""
    if not COHORT_VIDS.is_dir():
        return
    veh_subjects = set()
    for mp4 in COHORT_VIDS.glob("2605_FormOxy_Veh_S*_*.mp4"):
        m = re.match(r"2605_FormOxy_Veh_S(\d+)_(baseline|vehicle)\.mp4$", mp4.name)
        if m:
            veh_subjects.add(int(m.group(1)))
    if not veh_subjects:
        return

    existing_lines: list[str] = []
    keyed_subjects: set[str] = set()
    header_seen = False
    if KEY_CSV.is_file():
        with open(KEY_CSV, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                if not header_seen:
                    header_seen = True
                    existing_lines.append(line)
                    continue
                existing_lines.append(line)
                keyed_subjects.add(line.split(",", 1)[0])
    if not header_seen:
        existing_lines = ["Subject,Treatment"]

    appended = 0
    for n in sorted(veh_subjects):
        subject_id = f"2605_FormOxy_Veh_S{n}"
        if subject_id in keyed_subjects:
            continue
        existing_lines.append(f"{subject_id},Vehicle")
        appended += 1

    with open(KEY_CSV, "w", encoding="utf-8") as f:
        for line in existing_lines:
            f.write(line + "\n")
    step(f"  key file: {appended} vehicle row(s) appended, "
         f"{len(veh_subjects)} total vehicle subjects on key")


def main() -> int:
    COHORT_VIDS.mkdir(parents=True, exist_ok=True)
    wait_mode = "--wait" in sys.argv
    t0 = time.time()
    ready = wait_for_portal() if wait_mode else find_ready()
    if not ready:
        step("No ready vehicle-paw videos in portal. Nothing to do.")
        append_veh_to_key()
        return 0

    ready.sort(key=lambda r: (r[1], r[2]))  # mouse, condition
    step("Ready to ingest:")
    for src, mouse, cond_portal, dst_name in ready:
        sz_gb = src.stat().st_size / 1e9
        step(f"  Veh_S{mouse:<2} {cond_portal:<13}  {src.name}  ({sz_gb:.2f} GB)")

    discord(f":inbox_tray: FormOxy VehChain: {len(ready)} new video(s) ready.")

    n_new = 0
    for src, mouse, cond_portal, dst_name in ready:
        dst = COHORT_VIDS / dst_name
        if dst.is_file():
            step(f"  skip transcode (cohort already has {dst.name})")
            try:
                src.unlink()
            except OSError:
                pass
            continue
        step(f"  transcode Veh_S{mouse} {cond_portal} -> {dst.name}")
        if not transcode(src, dst):
            continue
        try:
            src.unlink()
            step(f"    removed portal copy: {src.name}")
        except OSError as e:
            step(f"    could not remove portal: {e}")
        n_new += 1

    append_veh_to_key()

    elapsed_min = (time.time() - t0) / 60
    discord(
        f":white_check_mark: FormOxy VehChain done in {elapsed_min:.0f} min. "
        f"Ingested {n_new} new video(s). "
        f"Cohort at `{COHORT_VIDS}`."
    )
    step(f"Done in {elapsed_min:.1f} min ({n_new} new ingested).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
