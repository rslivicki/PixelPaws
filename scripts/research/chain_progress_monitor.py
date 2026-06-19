"""
Live progress monitor for the THC post-DLC chain (extract_with_flow ->
predict_all -> group_analyze). Runs alongside the chain — does not launch
or kill it.

Posts a Discord message and edits it every 60s with:
  - current phase (feature-extract / predict / group-analyze)
  - per-phase progress bar by file count
  - currently processing video (parsed from chain log)
  - elapsed + ETA based on per-video time

Cost: one ~few-MB log read + 3 dir listings per minute. Negligible.

Run in background:
  py -X utf8 scripts/research/chain_progress_monitor.py >> scripts/research/chain_monitor.log 2>&1 &
"""
from __future__ import annotations
import json
import re
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

WEBHOOK = (
    ""
)

PROJECT = Path(r'E:\RSVIDS\Blackbox\260506_RS_THC_Withdrawal')
FEATURES_DIR = PROJECT / 'features'
RESULTS_DIR = PROJECT / 'results'
ANALYSIS_DIR = PROJECT / 'analysis'
CHAIN_LOG = Path(r'E:\PixelPaws\scripts\research\thc_orchestrator.log')

FLOW_HASH = '8aed1c22'    # extract_with_flow features
NOFLOW_HASH = '370c2fb2'  # predict_all features
TARGET = 30
BAR_WIDTH = 24
POLL_SECONDS = 60
MAX_RUNTIME_HOURS = 4  # safety: monitor exits after this

PHASE_RE = re.compile(r'>>> (feature-extract|predict-all|group-analyze): starting')
PHASE_END_RE = re.compile(r'<<< (feature-extract|predict-all|group-analyze): exit (-?\d+)')
VIDEO_RE = re.compile(r'=== ([A-Za-z0-9_]+) ===')
CHAIN_DONE_RE = re.compile(r'THC chain complete|THC orchestrator complete|THC analysis complete')
CHAIN_FAIL_RE = re.compile(r'THC chain: (?:feature extract|predict-all|group-analyze) FAILED')


def step(msg: str) -> None:
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', flush=True)


def _post_json(url: str, payload: dict, method: str = 'POST') -> dict | None:
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={
            'Content-Type': 'application/json',
            'User-Agent': 'PixelPaws-ChainMonitor/1.0',
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read()
            return json.loads(body) if body else None
    except Exception as e:
        step(f'  ! Discord {method} failed: {e}')
        return None


def discord_create(text: str) -> str | None:
    resp = _post_json(WEBHOOK + '?wait=true', {'content': text})
    return resp.get('id') if resp else None


def discord_edit(msg_id: str, text: str) -> None:
    _post_json(f'{WEBHOOK}/messages/{msg_id}', {'content': text}, method='PATCH')


def discord_post(text: str) -> None:
    _post_json(WEBHOOK, {'content': text})


def count_files(folder: Path, pattern: str) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_file() and pattern in p.name)


def read_chain_log() -> str:
    if not CHAIN_LOG.is_file():
        return ''
    try:
        with open(CHAIN_LOG, 'rb') as f:
            return f.read().decode('utf-8', errors='replace')
    except Exception:
        return ''


def detect_phase(text: str) -> str | None:
    """Most recent phase that has 'starting' but no matching 'exit'."""
    starts = list(PHASE_RE.finditer(text))
    ends = list(PHASE_END_RE.finditer(text))
    if not starts:
        return None
    last_start = starts[-1]
    # Is there an end after the last start?
    for end in ends:
        if end.start() > last_start.start() and end.group(1) == last_start.group(1):
            return None  # phase ended
    return last_start.group(1)


def detect_current_video(text: str, phase_start_offset: int = 0) -> str | None:
    """Most recent '=== <name> ===' marker after phase_start_offset."""
    scoped = text[phase_start_offset:] if phase_start_offset else text
    m = list(VIDEO_RE.finditer(scoped))
    if not m:
        return None
    return m[-1].group(1)


def make_bar(cur: int, total: int, width: int = BAR_WIDTH) -> str:
    if total <= 0:
        return '░' * width
    filled = max(0, min(width, cur * width // total))
    return '█' * filled + '░' * (width - filled)


def fmt_dur(seconds: int) -> str:
    if seconds < 60:
        return f'{seconds}s'
    m, s = divmod(seconds, 60)
    if m < 60:
        return f'{m}m{s:02d}s'
    h, m = divmod(m, 60)
    return f'{h}h{m:02d}m'


def progress_text(t_start: float, phase: str | None, cur_video: str | None,
                  done: int, total: int, phase_t_start: float | None) -> str:
    elapsed = int(time.time() - t_start)
    bar = make_bar(done, total)
    pct = 100 * done / total if total else 0

    if phase is None:
        phase_label = 'waiting / between phases'
    elif phase == 'feature-extract':
        phase_label = 'Feature extract (with optical flow)'
    elif phase == 'predict-all':
        phase_label = 'Feature extract (no flow) + predict 5 behaviors'
    elif phase == 'group-analyze':
        phase_label = 'Group analysis (transitions, immobility, ethograms)'
    else:
        phase_label = phase

    # ETA: from per-video time in this phase
    eta_str = '...'
    if phase_t_start and done > 0 and done < total:
        per_vid = (time.time() - phase_t_start) / done
        eta = int(per_vid * (total - done))
        eta_str = f'~{fmt_dur(eta)}'
    elif done >= total:
        eta_str = 'phase done'

    lines = [
        f'**THC post-DLC chain — live progress**',
        f'Phase: *{phase_label}*',
        f'`[{bar}] {done}/{total} sessions  ({pct:5.1f}%)`',
    ]
    if cur_video:
        lines.append(f'Current: `{cur_video}`')
    lines.append(f'Elapsed: {fmt_dur(elapsed)} | ETA (this phase): {eta_str}')
    return '\n'.join(lines)


def find_phase_start_offset(text: str, phase: str) -> int:
    """Byte offset of the most recent 'starting' line for the given phase."""
    starts = list(PHASE_RE.finditer(text))
    for s in reversed(starts):
        if s.group(1) == phase:
            return s.end()
    return 0


def main() -> int:
    t_start = time.time()
    step('chain monitor starting')

    text = read_chain_log()
    phase = detect_phase(text)

    initial_text = (
        f'**THC post-DLC chain — live progress**\n'
        f'Phase: *(starting up)*\n'
        f'`[{"░"*BAR_WIDTH}] 0/{TARGET} sessions  (  0.0%)`\n'
        f'Elapsed: 0s | ETA (this phase): ...'
    )
    msg_id = discord_create(initial_text)
    step(f'progress msg id={msg_id}')

    last_text_hash = None
    last_phase = phase
    phase_starts: dict[str, float] = {}
    if phase:
        phase_starts[phase] = time.time()

    deadline = t_start + MAX_RUNTIME_HOURS * 3600

    while time.time() < deadline:
        time.sleep(POLL_SECONDS)
        text = read_chain_log()
        phase = detect_phase(text)

        # Phase counter
        if phase == 'feature-extract':
            done = count_files(FEATURES_DIR, f'_features_{FLOW_HASH}.pkl')
            total = TARGET
        elif phase == 'predict-all':
            done = count_files(RESULTS_DIR, '_predictions.csv')
            total = TARGET
        elif phase == 'group-analyze':
            done = count_files(ANALYSIS_DIR, '.png')
            total = 6  # 6 figures expected
        else:
            done = 0
            total = TARGET

        # Track when each phase began
        if phase and phase != last_phase:
            phase_starts[phase] = time.time()
            last_phase = phase
            discord_post(f'THC chain phase change: starting *{phase}*')

        phase_t_start = phase_starts.get(phase) if phase else None

        # Find current video name (only look at log after current phase started)
        offset = find_phase_start_offset(text, phase) if phase else 0
        cur_video = detect_current_video(text, offset)

        # Edit the message
        new_text = progress_text(t_start, phase, cur_video, done, total, phase_t_start)
        if msg_id:
            discord_edit(msg_id, new_text)

        # Detect chain completion / failure
        if CHAIN_DONE_RE.search(text):
            step('chain done — final edit')
            if msg_id:
                discord_edit(msg_id, new_text + '\n\n**Chain complete.**')
            return 0
        if CHAIN_FAIL_RE.search(text):
            step('chain failed — final edit')
            if msg_id:
                discord_edit(msg_id, new_text + '\n\n**Chain FAILED — see log.**')
            return 1

    step('hit MAX_RUNTIME — exiting')
    if msg_id:
        discord_edit(msg_id, new_text + '\n\n**Monitor timed out (>4h).**')
    return 0


if __name__ == '__main__':
    sys.exit(main())
