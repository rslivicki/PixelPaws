"""
SNTX cohort 2 chain watcher. Runs the analysis pipeline TWICE — first on
the raw (unfiltered) DLC outputs as soon as 29 raw .h5 are present, then
again on the filtered outputs once filterpredictions completes.

Phase A (raw — primary):
  1. Wait for 29 raw shuffle9 .h5 (no _filtered suffix)
  2. Run sntx_cohort2_predict_all.py --h5-source=raw     -> results/, features/
  3. Run sntx_cohort2_group_analyze.py                   -> analysis/ + Discord

Phase B (filtered — comparison):
  4. Wait for 29 *_filtered.h5
  5. Run sntx_cohort2_predict_all.py --h5-source=filtered -> results_filtered/, features_filtered/
  6. Run sntx_cohort2_group_analyze.py --suffix=_filtered -> analysis_filtered/ + Discord

Run in background:
  py -X utf8 scripts/research/sntx_cohort2_watcher.py >> scripts/research/sntx_cohort2_watcher.log 2>&1 &

Idempotent. Safe to re-launch.
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

REPO = Path(r'E:\PixelPaws')
DLC_DIR = Path(r'E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\videos\2605_Cohort2')
TARGET = 29
POLL_SECONDS = 120  # check every 2 min while waiting

WEBHOOK = (
    ""
)


def step(msg: str) -> None:
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}', flush=True)


def discord_post(text: str) -> None:
    req = urllib.request.Request(
        WEBHOOK,
        data=json.dumps({'content': text}).encode('utf-8'),
        headers={'Content-Type': 'application/json',
                 'User-Agent': 'PixelPaws-Watcher/1.0'},
        method='POST',
    )
    try:
        urllib.request.urlopen(req, timeout=15).read()
    except Exception as e:
        step(f'  ! discord_post failed: {e}')


def count_raw() -> int:
    if not DLC_DIR.is_dir():
        return 0
    return sum(
        1 for p in DLC_DIR.iterdir()
        if p.is_file() and 'shuffle9' in p.name
        and p.name.endswith('.h5') and not p.name.endswith('_filtered.h5')
    )


def count_filtered() -> int:
    if not DLC_DIR.is_dir():
        return 0
    return sum(
        1 for p in DLC_DIR.iterdir()
        if p.is_file() and 'shuffle9' in p.name and p.name.endswith('_filtered.h5')
    )


def wait_for(label: str, count_fn, target: int) -> None:
    cur = count_fn()
    if cur >= target:
        step(f'  {label}: already at {cur}/{target}, no wait')
        return
    last = cur
    while cur < target:
        time.sleep(POLL_SECONDS)
        cur = count_fn()
        if cur != last:
            step(f'  {label}: {cur}/{target}')
            last = cur


def run(name: str, script_relpath: str, *extra_args: str) -> int:
    step(f'>>> {name}: starting')
    discord_post(f'SNTX cohort 2: starting *{name}*')
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
    cmd = ['py', '-X', 'utf8', script_relpath, *extra_args]
    proc = subprocess.run(cmd, cwd=str(REPO), env=env)
    rc = proc.returncode
    step(f'<<< {name}: exit {rc}')
    return rc


def main() -> int:
    step(f'Watcher started. Target = {TARGET} raw .h5, then {TARGET} filtered.')
    discord_post(
        f'SNTX cohort 2 watcher armed. Will run pipeline TWICE: '
        f'first on raw (unfiltered) .h5 as soon as DLC analyze completes, '
        f'then on filtered .h5 after filterpredictions runs. '
        f'Currently raw={count_raw()}/{TARGET}, filtered={count_filtered()}/{TARGET}.'
    )

    # ---- Phase A: raw (primary) ----
    step('Waiting for raw shuffle9 .h5 ...')
    wait_for('raw .h5', count_raw, TARGET)
    discord_post(f'SNTX cohort 2: {TARGET}/{TARGET} raw .h5 ready. '
                 f'Starting RAW chain (predict + 3-group analysis).')

    rc = run('predict_all (raw)', 'scripts/research/sntx_cohort2_predict_all.py',
             '--h5-source=raw')
    if rc != 0:
        discord_post(f'SNTX cohort 2 predict_all (raw) FAILED (exit {rc}).')
        return rc

    rc = run('group_analyze (raw)', 'scripts/research/sntx_cohort2_group_analyze.py')
    if rc != 0:
        discord_post(f'SNTX cohort 2 group_analyze (raw) FAILED (exit {rc}).')
        return rc

    discord_post('SNTX cohort 2 RAW chain complete. Now waiting for '
                 'filterpredictions to finish before running the filtered chain.')

    # ---- Phase B: filtered (comparison) ----
    step('Waiting for filterpredictions ...')
    wait_for('filtered .h5', count_filtered, TARGET)
    discord_post(f'SNTX cohort 2: {TARGET}/{TARGET} filtered .h5 ready. '
                 f'Starting FILTERED chain.')

    rc = run('predict_all (filtered)', 'scripts/research/sntx_cohort2_predict_all.py',
             '--h5-source=filtered')
    if rc != 0:
        discord_post(f'SNTX cohort 2 predict_all (filtered) FAILED (exit {rc}).')
        return rc

    rc = run('group_analyze (filtered)', 'scripts/research/sntx_cohort2_group_analyze.py',
             '--suffix=_filtered')
    if rc != 0:
        discord_post(f'SNTX cohort 2 group_analyze (filtered) FAILED (exit {rc}).')
        return rc

    discord_post('SNTX cohort 2 FILTERED chain complete. Both pipelines done. '
                 'Outputs: results/+analysis/ (raw, primary) and '
                 'results_filtered/+analysis_filtered/ (filtered, comparison).')
    step('Watcher done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
