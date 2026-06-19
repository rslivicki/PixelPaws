"""
Idempotent chain for ingesting the 2605 oxy/formalin Left-paws cohort
into PixelPaws.

For each fully-transferred mp4 in
  E:\\RSVIDS\\Video_transfer_portal\\oxyformalin\\<date>\\(baseline|formalin|formaline)\\
the chain:
  1. Classifies it. Files containing `_unflipped` in the name were
     recorded mirrored and need horizontal flipping; new-format files
     (no `_unflipped`) are kept as-is.
  2. Transcodes at libx264 CRF 26 slower preset, optionally with
     `-vf hflip`, dropping into the cohort `videos/` folder under the
     canonical name `2605_FormOxy_S<N>_<baseline|formalin>.mp4`.
  3. Removes the portal copy (external HDD backups are the durable copy
     per the user).
  4. Writes / refreshes the dose key xlsx with the modular assignment.

Idempotent: any cohort file that already exists is skipped, and the
portal source is still cleaned up if the destination matches.

Run (regular PixelPaws env):
  PYTHONIOENCODING=utf-8 py -X utf8 -u scripts/research/formoxy_addons_chain.py
  (add --wait to poll the portal every 60s until at least one ready file appears)
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

PORTAL_BASE = Path(r"E:\RSVIDS\Video_transfer_portal\oxyformalin")
COHORT = Path(r"E:\RSVIDS\Blackbox\2605_FormOxy_LeftPaws")
COHORT_VIDS = COHORT / "videos"
CRF_TARGET = 26

DOSE_SEQUENCE = ["Veh", "Oxy1", "Oxy3", "Oxy10"]

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
                     "User-Agent": "PixelPaws-FormOxy-Chain/1.0"},
        )
        urllib.request.urlopen(req, timeout=15)
    except Exception as e:
        step(f"  ! discord failed: {e}")


# --------------------------------------------------------------------------- #
# Classifier: portal filename -> (mouse, condition, needs_flip, dst_name)
# --------------------------------------------------------------------------- #

# Order matters -- the _unflipped patterns must run FIRST so that
# `..._merged.mp4` (without _unflipped) does not accidentally match
# a more permissive new-format regex.
PATTERNS_FLIP = [
    # 2605_FormOxy<N>_baseline_unflipped.mp4
    re.compile(r"^2605_FormOxy(\d+)_baseline(?:_merged)?_unflipped\.mp4$", re.IGNORECASE),
    # 2605_FormOxy<N>_formalin_merged_unflipped.mp4
    re.compile(r"^2605_FormOxy(\d+)_formalin(?:_merged)?_unflipped\.mp4$", re.IGNORECASE),
]
PATTERNS_NOFLIP_BASELINE = [
    # baseline_2605_Formoxy<N>.mp4 (new convention)
    re.compile(r"^baseline_2605_Formoxy(\d+)\.mp4$", re.IGNORECASE),
]
PATTERNS_NOFLIP_FORMALIN = [
    # 2605_Formoxy<N>_formaline.mp4 (typo) / formalin.mp4 / formalin_merged.mp4
    # only matched if no `_unflipped` exists for the same (subject, condition)
    re.compile(r"^2605_Formoxy(\d+)_formaline?\.mp4$", re.IGNORECASE),
    # Bare filename `2605_formoxy<N>.mp4` (parent dir, lowercase, no condition
    # suffix) -- these are the merged 1-hour formalin recordings for the
    # later subjects (S7+). Size-filtered to >= 500 MB in find_ready().
    re.compile(r"^2605_formoxy(\d+)\.mp4$", re.IGNORECASE),
]


def cohort_name(mouse: int, cond: str) -> str:
    return f"2605_FormOxy_S{mouse}_{cond}.mp4"


def has_unflipped_partner(portal_files: list[Path], mouse: int, cond: str) -> bool:
    """True if a `<mouse>_<cond>*_unflipped.mp4` exists alongside; used so
    we don't ingest the irrelevant non-flipped version when an
    unflipped (canonical) version is present."""
    needle = re.compile(
        rf"^2605_FormOxy{mouse}_{cond}(?:_merged)?_unflipped\.mp4$", re.IGNORECASE
    )
    return any(needle.match(p.name) for p in portal_files)


def classify(src: Path, all_portal: list[Path]) -> tuple[int, str, bool, str] | None:
    name = src.name
    if name.startswith("~syncthing~") or name.endswith(".tmp"):
        return None
    # `_unflipped` suffix is a SOURCE-SELECTION marker, NOT a "needs flip"
    # marker. The `_unflipped` files are already in correct orientation and
    # should NOT have hflip applied. (Verified empirically by visual check
    # after the first ingest pass produced mirrored output.)
    for rx in PATTERNS_FLIP:
        m = rx.match(name)
        if m:
            mouse = int(m.group(1))
            cond = "baseline" if "baseline" in name.lower() else "formalin"
            return (mouse, cond, False, cohort_name(mouse, cond))
    # No-flip baseline
    for rx in PATTERNS_NOFLIP_BASELINE:
        m = rx.match(name)
        if m:
            mouse = int(m.group(1))
            return (mouse, "baseline", False, cohort_name(mouse, "baseline"))
    # No-flip formalin (only if no _unflipped partner for same subject+condition)
    for rx in PATTERNS_NOFLIP_FORMALIN:
        m = rx.match(name)
        if m:
            mouse = int(m.group(1))
            if has_unflipped_partner(all_portal, mouse, "formalin"):
                step(f"  skip {name}: an _unflipped version exists for "
                     f"S{mouse} formalin -- preferring that")
                return None
            return (mouse, "formalin", False, cohort_name(mouse, "formalin"))
    # OLD-style naming `2605_FormOxy<N>_<cond>(_merged)?.mp4` without an
    # `_unflipped` suffix. If the same (subject, condition) has an
    # `_unflipped` partner present, this non-flipped version is the
    # "irrelevant" mirrored leftover -- skip. Otherwise it's a new-batch
    # recording that happens to reuse the old naming (e.g. S7/8/9) and
    # should be ingested as-is (no flip).
    m = re.match(r"^2605_FormOxy(\d+)_(baseline|formalin)(?:_merged)?\.mp4$",
                 name, re.IGNORECASE)
    if m:
        mouse = int(m.group(1))
        cond = m.group(2).lower()
        if has_unflipped_partner(all_portal, mouse, cond):
            step(f"  skip {name}: an _unflipped version exists for "
                 f"S{mouse} {cond} -- preferring that")
            return None
        return (mouse, cond, False, cohort_name(mouse, cond))
    return None


# --------------------------------------------------------------------------- #
# Portal scan
# --------------------------------------------------------------------------- #

def list_portal_files() -> list[Path]:
    if not PORTAL_BASE.is_dir():
        return []
    out = []
    for mp4 in PORTAL_BASE.rglob("*.mp4"):
        if mp4.name.startswith("~syncthing~") or mp4.name.endswith(".tmp"):
            continue
        out.append(mp4)
    return out


def find_ready() -> list[tuple[Path, int, str, bool, str]]:
    all_files = list_portal_files()
    ready = []
    for mp4 in all_files:
        cls = classify(mp4, all_files)
        if cls is None:
            continue
        mouse, cond, flip, dst_name = cls
        # Skip the 45-110 MB short partials (only the merged 1-hour versions)
        # by requiring the file be >= 500 MB
        if mp4.stat().st_size < 500 * 1024 * 1024:
            step(f"  skip {mp4.name}: too small ({mp4.stat().st_size/1e6:.0f}MB) "
                 f"-- probably a partial, not the merged 1-hour version")
            continue
        ready.append((mp4, mouse, cond, flip, dst_name))
    return ready


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


# --------------------------------------------------------------------------- #
# Transcode
# --------------------------------------------------------------------------- #

def transcode(src: Path, dst: Path, flip: bool) -> bool:
    cmd = ["ffmpeg", "-y", "-i", str(src),
           "-c:v", "libx264",
           "-crf", str(CRF_TARGET),
           "-preset", "slower",
           "-an"]
    if flip:
        cmd += ["-vf", "hflip"]
    cmd.append(str(dst))
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        step(f"  ! ffmpeg failed: {proc.stderr[-300:]}")
        if dst.is_file():
            dst.unlink()
        return False
    dt = time.time() - t0
    sz_mb = dst.stat().st_size / 1e6
    step(f"    {'flipped ' if flip else ''}done in {dt:.0f}s, {sz_mb:.0f} MB")
    return True


# --------------------------------------------------------------------------- #
# Dose key
# --------------------------------------------------------------------------- #

def dose_for_mouse(n: int) -> str:
    return DOSE_SEQUENCE[(n - 1) % len(DOSE_SEQUENCE)]


def refresh_dose_key() -> Path | None:
    """Write a fresh dose key CSV covering every subject currently in the cohort."""
    if not COHORT_VIDS.is_dir():
        return None
    subjects = set()
    for mp4 in COHORT_VIDS.glob("2605_FormOxy_S*_*.mp4"):
        m = re.match(r"2605_FormOxy_S(\d+)_(baseline|formalin)\.mp4$", mp4.name)
        if m:
            subjects.add(int(m.group(1)))
    if not subjects:
        return None
    key_csv = COHORT / "2605_FormOxy_key.csv"
    key_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(key_csv, "w", encoding="utf-8") as f:
        f.write("Subject,Treatment\n")
        for n in sorted(subjects):
            f.write(f"2605_FormOxy_S{n},{dose_for_mouse(n)}\n")
    return key_csv


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main() -> int:
    COHORT_VIDS.mkdir(parents=True, exist_ok=True)
    wait_mode = "--wait" in sys.argv
    t0 = time.time()
    ready = wait_for_portal() if wait_mode else find_ready()
    if not ready:
        step("No ready oxy/formalin videos in portal. Nothing to do.")
        return 0

    ready.sort(key=lambda r: (r[1], r[2]))  # mouse num, condition
    step("Ready to ingest:")
    for src, mouse, cond, flip, dst_name in ready:
        sz = src.stat().st_size / 1e9
        flip_tag = "+FLIP" if flip else "    "
        step(f"  S{mouse:<2} {cond:<8} {flip_tag}  {src.name}  ({sz:.2f} GB)")

    discord(f":inbox_tray: FormOxy chain: {len(ready)} new video(s) ready.")

    n_new = 0
    for src, mouse, cond, flip, dst_name in ready:
        dst = COHORT_VIDS / dst_name
        if dst.is_file():
            step(f"  skip transcode (cohort already has {dst.name})")
            # Still clean up portal copy since we've established this src
            # is the right one (idempotent re-runs hit this branch)
            try: src.unlink()
            except OSError: pass
            continue
        step(f"  transcode S{mouse} {cond} ({'flip' if flip else 'no flip'}) -> {dst.name}")
        ok = transcode(src, dst, flip)
        if not ok:
            continue
        try:
            src.unlink()
            step(f"    removed portal copy: {src.name}")
        except OSError as e:
            step(f"    could not remove portal: {e}")
        n_new += 1

    key_csv = refresh_dose_key()
    if key_csv:
        step(f"Dose key refreshed: {key_csv}")

    elapsed_h = (time.time() - t0) / 3600
    discord(f":white_check_mark: FormOxy chain pass done in "
            f"{elapsed_h*60:.0f} min. Ingested {n_new} new video(s). "
            f"Cohort at {COHORT_VIDS}.")
    step(f"Done in {elapsed_h*60:.1f} min ({n_new} new ingested).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
