"""
Sequential pipeline:
  1. Wait for crop_for_dlc.py to finish (poll for the process to exit).
  2. Run DLC batch on the DV_DSS cohort (idempotent; picks up new crops).
  3. Run locomotor analysis (idempotent; picks up newly analyzed sessions).
  4. Re-launch the THC rim orchestrator so it resumes the suspended cohort.

Posts Discord milestones at each step.

Run in background:
  py -X utf8 -u scripts/research/dvv_then_resume_thc.py \
      >> E:/RSVIDS/Blackbox/2604_DV_DSS/dvv_pipeline.log 2>&1
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
DLC_PYTHON = r"C:\Users\Gereau\anaconda3\envs\DEEPLABCUT\python.exe"
DVV_VIDEO_DIR = Path(r"E:\RSVIDS\Blackbox\2604_DV_DSS\videos")
DVV_DLC_BATCH = r"E:\PixelPaws\scripts\research\dlc_batch_dvv.py"
DVV_LOCOMOTOR = r"E:\PixelPaws\scripts\research\dvv_locomotor.py"
THC_ORCHESTRATOR = r"E:\PixelPaws\scripts\research\thc_rim_orchestrator.py"
DVV_LOG = Path(r"E:\RSVIDS\Blackbox\2604_DV_DSS\dlc_batch.log")
THC_LOG = Path(r"E:\RSVIDS\Blackbox\260512_THC_Rim_Cohort\orchestrator.log")

# Filename → (genotype, mouse, condition). Same rule as dvv_locomotor.py.
_DVV_NAME_RE = re.compile(
    r"^(?:\d{4,6}_)?(?P<geno>(?:pdyn_)?ai32)_(?P<mouse>\d+)_"
    r"(?P<cond>BL|STIM)(?:_cropped)?\.mp4$",
    re.IGNORECASE,
)

WEBHOOK = (
    ""
)

CROP_POLL_S = 30
CROP_MAX_WAIT_HOURS = 6  # bail out if crop_for_dlc somehow never exits


def step(msg: str) -> None:
    line = f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode(), flush=True)


def discord(msg: str):
    req = urllib.request.Request(
        WEBHOOK, data=json.dumps({"content": msg}).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json",
                 "User-Agent": "PixelPaws-DVV-Chain/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15):
            pass
    except Exception as e:
        step(f"  ! discord failed: {e}")


def prune_duplicate_crops() -> int:
    """When two .mp4 exist for the same (genotype, mouse, condition), keep
    the OLDEST and delete the newer copies. Returns count of files removed.

    The user's convention: when crop_for_dlc.py re-runs and produces a
    second crop alongside the original, the older crop is the canonical
    one — delete the newer.
    """
    from collections import defaultdict
    groups: dict[tuple, list[Path]] = defaultdict(list)
    for v in DVV_VIDEO_DIR.glob("*.mp4"):
        m = _DVV_NAME_RE.match(v.name)
        if not m:
            continue
        geno = "pdyn_ai32" if "pdyn" in m["geno"].lower() else "ai32"
        key = (geno, m["mouse"], m["cond"].upper())
        groups[key].append(v)

    removed = 0
    for key, files in groups.items():
        if len(files) < 2:
            continue
        # Sort by mtime ascending; keep first, delete rest.
        files.sort(key=lambda p: p.stat().st_mtime)
        keep = files[0]
        step(f"  duplicate for {key}: keeping {keep.name} "
             f"(mtime {time.strftime('%Y-%m-%d %H:%M', time.localtime(keep.stat().st_mtime))})")
        for f in files[1:]:
            mt = time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime))
            step(f"    removing newer: {f.name} (mtime {mt})")
            # Also remove any DLC outputs that may have been generated for
            # the newer copy (rare but possible if a prior batch saw it).
            for ext in (".mp4", "DLC_Resnet50_palmreader-500Mar25shuffle9_snapshot_best-190.h5",
                        "DLC_Resnet50_palmreader-500Mar25shuffle9_snapshot_best-190_filtered.h5",
                        "DLC_Resnet50_palmreader-500Mar25shuffle9_snapshot_best-190_full.pickle",
                        "DLC_Resnet50_palmreader-500Mar25shuffle9_snapshot_best-190_meta.pickle"):
                related = DVV_VIDEO_DIR / (f.stem + ext) if ext != ".mp4" else f
                if related.is_file():
                    related.unlink()
                    step(f"      also removed: {related.name}")
            removed += 1
    return removed


def crop_running() -> bool:
    """Return True if any python.exe is currently running crop_for_dlc.py."""
    try:
        out = subprocess.check_output(
            ["wmic", "process", "where", "name='python.exe'", "get", "CommandLine"],
            stderr=subprocess.DEVNULL, timeout=15,
        ).decode("utf-8", errors="ignore")
    except Exception:
        return False
    return any("crop_for_dlc.py" in line for line in out.splitlines())


def wait_for_crop():
    t0 = time.time()
    if not crop_running():
        step("crop_for_dlc.py is not running — proceeding immediately.")
        return
    discord(":hourglass: DVV pipeline started. Waiting for crop_for_dlc.py to finish before running DLC.")
    while crop_running():
        elapsed = time.time() - t0
        if elapsed > CROP_MAX_WAIT_HOURS * 3600:
            step(f"crop_for_dlc.py still running after {CROP_MAX_WAIT_HOURS}h — bailing.")
            discord(f":warning: crop_for_dlc.py still running after {CROP_MAX_WAIT_HOURS}h. DVV pipeline aborting.")
            sys.exit(2)
        if int(elapsed) % 300 == 0:
            step(f"  still waiting on crop_for_dlc.py ({elapsed/60:.1f} min elapsed)")
        time.sleep(CROP_POLL_S)
    step(f"crop_for_dlc.py finished after {(time.time() - t0)/60:.1f} min wait.")
    discord(f":white_check_mark: crop_for_dlc.py done after {(time.time() - t0)/60:.1f} min wait. Starting DVV DLC.")


def run_dvv_dlc() -> int:
    DVV_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DVV_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n\n[{datetime.now().isoformat(timespec='seconds')}] "
                f"DVV DLC launch from chain\n")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    with open(DVV_LOG, "a", encoding="utf-8") as f:
        rc = subprocess.run(
            [DLC_PYTHON, DVV_DLC_BATCH],
            cwd=str(REPO), env=env,
            stdout=f, stderr=subprocess.STDOUT,
        ).returncode
    return rc


def run_locomotor() -> int:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [DLC_PYTHON, "-X", "utf8", DVV_LOCOMOTOR],
        cwd=str(REPO), env=env,
    ).returncode


def launch_thc_orchestrator() -> int:
    """Launch the THC rim orchestrator detached so it survives this chain exit."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    THC_LOG.parent.mkdir(parents=True, exist_ok=True)
    fout = open(THC_LOG, "ab")
    proc = subprocess.Popen(
        ["py", "-X", "utf8", "-u", THC_ORCHESTRATOR],
        cwd=str(REPO), env=env,
        stdout=fout, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            if sys.platform == "win32" else 0,
    )
    step(f"THC rim orchestrator relaunched (pid={proc.pid})")
    return proc.pid


def main() -> int:
    step("DVV-then-resume-THC chain starting.")
    discord("**DVV pipeline kicked off.** Will wait for crop_for_dlc, "
            "run DVV DLC, run locomotor analysis, then resume THC rim DLC.")

    wait_for_crop()

    step("--- Pruning duplicate crops (keep oldest) ---")
    removed = prune_duplicate_crops()
    if removed:
        discord(f":scissors: Pruned {removed} duplicate crop(s) before DLC.")
    else:
        step("No duplicates found.")

    step("--- DVV DLC batch starting ---")
    discord(":gear: DVV DLC batch starting.")
    rc = run_dvv_dlc()
    if rc != 0:
        step(f"DVV DLC failed (rc={rc}).")
        discord(f":x: DVV DLC failed with rc={rc}. See {DVV_LOG}.")
        return rc
    discord(":white_check_mark: DVV DLC complete.")

    step("--- Locomotor analysis starting ---")
    discord(":bar_chart: DVV locomotor analysis starting.")
    rc = run_locomotor()
    if rc != 0:
        step(f"Locomotor analysis exited rc={rc}; continuing to THC resume anyway.")
        discord(f":warning: DVV locomotor exited rc={rc}, continuing to THC resume.")
    else:
        discord(":white_check_mark: DVV locomotor analysis complete (figures posted above).")

    step("--- Resuming THC rim orchestrator ---")
    discord(":arrows_counterclockwise: Resuming THC rim DLC orchestrator now.")
    launch_thc_orchestrator()
    step("Chain done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
