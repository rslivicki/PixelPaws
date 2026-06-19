"""
Collect per-video timing telemetry for DLC + feature extraction + prediction
runs, paired with video metadata (duration, fps, frame count, resolution,
file size, bitrate).

Outputs a single CSV `scripts/research/telemetry/timing.csv` that grows over
time as more videos are processed. Fields:

  session, project, video_path, file_size_mb, duration_s, fps, frame_count,
  width, height, bitrate_kbps,
  dlc_analyze_s, dlc_analyze_fps, dlc_filter_s,
  feature_extract_s, feature_extract_with_flow,
  predict_s, n_classifiers, run_timestamp

Idempotent: re-running re-parses the same logs and writes/overwrites the same
rows by (session, project, run_timestamp). Safe to invoke after every chain.

Run:
  py -X utf8 scripts/research/collect_timing_telemetry.py
  py -X utf8 scripts/research/collect_timing_telemetry.py --watch  # poll mode
"""
from __future__ import annotations
import argparse
import csv
import re
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(r'E:\PixelPaws')
TELEMETRY_DIR = REPO / 'scripts' / 'research' / 'telemetry'
TELEMETRY_DIR.mkdir(exist_ok=True)
TIMING_CSV = TELEMETRY_DIR / 'timing.csv'

# Logs to parse
LOGS = [
    {
        'project': 'THC_Withdrawal',
        'dlc_log': REPO / 'scripts' / 'research' / 'thc_dlc.log',
        'chain_log': REPO / 'scripts' / 'research' / 'thc_orchestrator.log',
        'video_dir': Path(r'E:\RSVIDS\Blackbox\260506_RS_THC_Withdrawal\Videos'),
    },
    {
        'project': 'SNLT_cohort2',
        'dlc_log': REPO / 'scripts' / 'research' / 'snlt_cohort2_dlc.log',
        'chain_log': None,
        'video_dir': Path(r'E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\videos\2605_Cohort2'),
    },
]

# Patterns
# tqdm: ' 14000/117450 [09:33<25:01, 23.97it/s]'
TQDM_RE = re.compile(r'(\d+)\s*/\s*(\d+)\s*\[(\d+:\d+(?::\d+)?)<(?:\d+:\d+(?::\d+)?|\?),\s*(\d+\.?\d*)\s*it/s')
START_VIDEO_RE = re.compile(r'Starting to analyze\s+(.+?\.mp4)', re.IGNORECASE)
FILTERPRED_RE = re.compile(r'Filtering with median model\s+(.+?\.mp4)', re.IGNORECASE)
# extract_with_flow's `cached (52.1s, X=(108234, 412)) -> ...` line
CACHED_RE = re.compile(r'cached \((\d+\.\d+)s,\s*X=\((\d+),\s*(\d+)\)\)\s*->\s*([A-Za-z0-9_]+)_features_([0-9a-f]+)\.pkl')
# predict_all's `=== <session> ===` marker (start of per-session block) and
# `-> <session>_predictions.csv` (end)
SESSION_HEADER_RE = re.compile(r'^\s*===\s+([A-Za-z0-9_]+)\s+===', re.MULTILINE)
PRED_OUT_RE = re.compile(r'->\s+([A-Za-z0-9_]+)_predictions\.csv')
# step() prefixes give us coarse timestamps
STEP_TS_RE = re.compile(r'^\[(\d{2}:\d{2}:\d{2})\]')


def parse_dlc_log_times(log_path: Path) -> dict:
    """Parse a DLC batch log to recover per-video analyze time + fps and
    filter time. Returns {video_basename: dict}."""
    out: dict = {}
    if not log_path.is_file():
        return out
    try:
        with open(log_path, 'rb') as f:
            data = f.read().decode('utf-8', errors='replace')
    except Exception:
        return out
    text = data.replace('\r', '\n')

    # Find each "Starting to analyze X" → following tqdm ends with elapsed
    # time + final fps when frame count == total. We pair with the final
    # tqdm match before the next "Starting to analyze".
    starts = list(START_VIDEO_RE.finditer(text))
    for i, sm in enumerate(starts):
        video_path = sm.group(1).strip()
        base = Path(video_path.strip('"')).stem
        next_start = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        section = text[sm.end():next_start]
        # Find all tqdm matches; take the LAST (which is the completed one)
        tqdms = list(TQDM_RE.finditer(section))
        if not tqdms:
            continue
        last = tqdms[-1]
        cur, total, elapsed_str, fps = last.group(1), last.group(2), last.group(3), last.group(4)
        try:
            cur_i = int(cur); total_i = int(total); fps_f = float(fps)
        except ValueError:
            continue
        if cur_i < total_i:
            # Did not finish
            continue
        # Parse elapsed (M:SS or H:MM:SS)
        parts = elapsed_str.split(':')
        if len(parts) == 2:
            secs = int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            secs = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        else:
            continue
        out[base] = dict(dlc_analyze_s=secs, dlc_analyze_fps=fps_f,
                         frame_count=total_i)

    # Estimate filter time per video: each filterpredictions log line per
    # video runs in ~5-15s typically. Hard to time individually from log,
    # but we can divide total filter elapsed across the videos that ran.
    # Skip for now — the field stays None.
    return out


def parse_chain_log_times(log_path: Path) -> dict:
    """Parse extract_with_flow / predict_all output for per-session times.
    Returns {session_name: dict}."""
    out: dict = {}
    if not log_path or not log_path.is_file():
        return out
    try:
        with open(log_path, 'rb') as f:
            data = f.read().decode('utf-8', errors='replace')
    except Exception:
        return out
    text = data.replace('\r', '\n')

    # Feature-extract timings (from "cached (X.Xs, ...)")
    for m in CACHED_RE.finditer(text):
        secs = float(m.group(1))
        rows = int(m.group(2)); cols = int(m.group(3))
        session = m.group(4); cfg_hash = m.group(5)
        rec = out.setdefault(session, {})
        rec['feature_extract_s'] = secs
        rec['feature_X_rows'] = rows
        rec['feature_X_cols'] = cols
        rec['feature_hash'] = cfg_hash
        # Hash 8aed1c22 = with flow (per the THC project)
        rec['feature_extract_with_flow'] = (cfg_hash == '8aed1c22')

    # Predict timings: from `=== <session> ===` (start) to `-> <s>_predictions.csv` (end)
    headers = list(SESSION_HEADER_RE.finditer(text))
    for i, hm in enumerate(headers):
        session = hm.group(1)
        # Find timestamp immediately before/after the header
        # Use timestamps on lines after the header until the predictions line
        next_start = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        section = text[hm.start():next_start]
        # Find first "[HH:MM:SS]" inside section
        ts_iter = STEP_TS_RE.finditer(section, re.MULTILINE)
        ts_list = []
        for tsm in ts_iter:
            try:
                t = datetime.strptime(tsm.group(1), '%H:%M:%S').time()
                ts_list.append(t)
            except ValueError:
                pass
        # End timestamp: last STEP_TS_RE match in this section that comes
        # AFTER the `-> ..._predictions.csv` line.
        pred_m = PRED_OUT_RE.search(section)
        if pred_m and len(ts_list) >= 2:
            # Compute seconds between first ts and last ts (rough)
            t0 = ts_list[0]; t1 = ts_list[-1]
            dt0 = t0.hour * 3600 + t0.minute * 60 + t0.second
            dt1 = t1.hour * 3600 + t1.minute * 60 + t1.second
            secs = dt1 - dt0
            if secs < 0:
                secs += 86400  # midnight wrap
            rec = out.setdefault(session, {})
            rec['session_total_s'] = secs

    return out


def probe_video(path: Path) -> dict:
    """File metadata via cv2 + stat. Cached by path."""
    out = dict(file_size_mb=None, duration_s=None, fps=None,
               frame_count=None, width=None, height=None, bitrate_kbps=None)
    if not path.is_file():
        return out
    try:
        size_mb = path.stat().st_size / 1e6
        out['file_size_mb'] = round(size_mb, 1)
    except Exception:
        pass
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        out['fps'] = round(fps, 2) if fps else None
        out['frame_count'] = frames
        out['width'] = w
        out['height'] = h
        if fps and frames:
            duration = frames / fps
            out['duration_s'] = round(duration, 1)
            if out['file_size_mb']:
                out['bitrate_kbps'] = round(out['file_size_mb'] * 8 * 1000 / duration, 0)
    except Exception as e:
        print(f'  ! cv2 probe failed for {path.name}: {e}', flush=True)
    return out


FIELDS = [
    'session', 'project', 'video_path',
    'file_size_mb', 'duration_s', 'fps', 'frame_count',
    'width', 'height', 'bitrate_kbps',
    'dlc_analyze_s', 'dlc_analyze_fps',
    'feature_extract_s', 'feature_extract_with_flow',
    'feature_X_rows', 'feature_X_cols', 'feature_hash',
    'session_total_s',
    'run_timestamp',
]


def collect() -> int:
    """Snapshot semantics: re-parse all logs and rewrite timing.csv with the
    current state. Each (session, project) pair has at most one row. Safe to
    run repeatedly without ballooning the file.
    """
    rows = []
    run_ts = datetime.now().isoformat(timespec='seconds')

    for proj_cfg in LOGS:
        project = proj_cfg['project']
        video_dir = proj_cfg['video_dir']
        dlc_log = proj_cfg['dlc_log']
        chain_log = proj_cfg['chain_log']

        dlc_times = parse_dlc_log_times(dlc_log) if dlc_log else {}
        chain_times = parse_chain_log_times(chain_log) if chain_log else {}

        sessions = set(dlc_times.keys()) | set(chain_times.keys())
        if not sessions:
            print(f'  [{project}] no timing data found in logs')
            continue
        print(f'  [{project}] sessions: {len(sessions)}')

        for session in sorted(sessions):
            video_path = video_dir / f'{session}.mp4'
            meta = probe_video(video_path)
            row = dict(
                session=session,
                project=project,
                video_path=str(video_path),
                run_timestamp=run_ts,
                **meta,
                **dlc_times.get(session, {}),
                **chain_times.get(session, {}),
            )
            row = {k: row.get(k) for k in FIELDS}
            rows.append(row)

    # Overwrite (snapshot mode)
    with open(TIMING_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'\nWrote {len(rows)} row(s) to {TIMING_CSV} (snapshot)')
    return 0


def watch():
    print('Watching for new chain runs (poll every 5 min, Ctrl-C to stop)...')
    while True:
        try:
            collect()
        except Exception as e:
            print(f'  ! collect failed: {e}')
        time.sleep(300)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--watch', action='store_true', help='Poll every 5 min')
    args = ap.parse_args()
    if args.watch:
        watch()
    else:
        sys.exit(collect())
