"""
Post-DLC chain for THC withdrawal project after the 18 new videos
finish DLC. Runs sequentially:

  1. Feature extraction with optical flow (skip-existing for the 12
     already done at hash 8aed1c22).
  2. Predict + per-classifier post-processing for all 30 sessions
     (skip-existing for cached + run all 5 classifiers).
  3. Group analysis (THC vs Vehicle, Baseline vs Postdrug, with
     EXCLUDE_NOISE so 'other' frames don't contribute).
  4. Discord push with results.
"""
from __future__ import annotations
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(r'E:\PixelPaws')
WEBHOOK = (
    ""
)


def step(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def discord(msg):
    subprocess.run(
        ['curl', '-s', '-o', os.devnull, '-w', '%{http_code}', '-X', 'POST',
         '-H', 'Content-Type: application/json',
         '-d', f'{{"content":"{msg}"}}', WEBHOOK],
        capture_output=True, text=True,
    )


def run(name, script):
    step(f'>>> {name}: starting')
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
    p = subprocess.Popen(
        ['py', '-X', 'utf8', script],
        cwd=str(REPO), env=env,
        stdout=sys.stdout, stderr=subprocess.STDOUT,
    )
    rc = p.wait()
    step(f'<<< {name}: exit {rc}')
    return rc


def main():
    discord('THC DLC complete. Starting post-DLC chain: features (with flow) -> predict -> group analyze.')

    rc = run('feature-extract', 'scripts/research/thc_withdrawal_extract_with_flow.py')
    if rc != 0:
        discord(f'THC chain: feature extract FAILED (exit {rc}). Stopping.')
        return rc

    rc = run('predict-all', 'scripts/research/thc_withdrawal_predict_all.py')
    if rc != 0:
        discord(f'THC chain: predict-all FAILED (exit {rc}). Stopping.')
        return rc

    rc = run('group-analyze', 'scripts/research/thc_withdrawal_group_analyze.py')
    if rc != 0:
        discord(f'THC chain: group-analyze FAILED (exit {rc}). Stopping.')
        return rc

    discord('THC chain complete (analysis figures already posted by group-analyze). Restarting cohort2 SNLT DLC next.')
    step('Done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
