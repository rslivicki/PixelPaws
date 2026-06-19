"""
THC orchestrator with live Discord progress bar, then SNLT cohort2.

Run in background:
  py -X utf8 scripts/research/thc_orchestrator.py >> scripts/research/thc_orchestrator.log 2>&1 &

Behavior:
  1. Posts an initial 'progress' message to Discord, captures its message id.
  2. Polls the Videos/ folder every 60s for *_filtered.h5 count and PATCHes
     the same message with a [████░░░░] bar + ETA. Discord URL is the same
     as the Discord channel — web-accessible from any browser/phone.
  3. Once 30 _filtered.h5 are present, posts a milestone ping and runs the
     post-DLC chain (extract_with_flow -> predict_all -> group_analyze).
  4. After THC analysis pushes to Discord, kicks off SNLT cohort2 DLC
     (29 videos in 2605_Cohort2/) with its own progress message.
  5. On any failure or completion, posts a separate Discord ping.
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

WEBHOOK = (
    ""
)
THC_DIR = Path(r'E:\RSVIDS\Blackbox\260506_RS_THC_Withdrawal\Videos')
SNLT_DIR = Path(r'E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\videos\2605_Cohort2')
REPO = Path(r'E:\PixelPaws')
DLC_PYTHON = r'C:\Users\Gereau\anaconda3\envs\DEEPLABCUT\python.exe'
THC_TARGET = 30
SNLT_TARGET = 29
BAR_WIDTH = 24
POLL_SECONDS = 60
MAX_RESTARTS = 5
KILL_GRACE_SECONDS = 60        # let GPU release VRAM after kill
MIN_HEALTHY_FPS = 8.0          # tqdm it/s; healthy DLC inference is 25-90 fps
FPS_STALL_TIMEOUT_MIN = 10     # >10min sustained low FPS = stall (during analyze)
LOG_MTIME_STALL_MIN = 5        # log file untouched this long = process dead
HARD_STALL_TIMEOUT_MIN = 240   # backstop: >4h with zero new files = stall regardless


def _post_json(url: str, payload: dict, method: str = 'POST') -> dict | None:
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={
            'Content-Type': 'application/json',
            # Discord rejects requests without a User-Agent (returns 403).
            'User-Agent': 'PixelPaws-Orchestrator/1.0 (+thc_orchestrator.py)',
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read()
            if body:
                return json.loads(body)
            return None
    except Exception as e:
        print(f'  ! Discord {method} failed: {e}', flush=True)
        return None


def discord_create_progress_msg(initial_text: str) -> str | None:
    """POST with ?wait=true so we get the message id back for later edits."""
    url = WEBHOOK + '?wait=true'
    resp = _post_json(url, {'content': initial_text})
    if resp and 'id' in resp:
        return resp['id']
    return None


def discord_edit_msg(msg_id: str, text: str) -> None:
    url = f'{WEBHOOK}/messages/{msg_id}'
    _post_json(url, {'content': text}, method='PATCH')


def discord_post(text: str) -> None:
    _post_json(WEBHOOK, {'content': text})


def count_filtered(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(
        1 for p in folder.iterdir()
        if p.is_file() and 'shuffle9' in p.name and p.name.endswith('_filtered.h5')
    )


def count_raw_h5(folder: Path) -> int:
    """Count shuffle-9 raw .h5 (excluding _filtered). Each video's analyze
    step writes one of these; used as the live stall signal."""
    if not folder.is_dir():
        return 0
    return sum(
        1 for p in folder.iterdir()
        if p.is_file() and 'shuffle9' in p.name
        and p.name.endswith('.h5') and not p.name.endswith('_filtered.h5')
    )


def progress_signal(folder: Path) -> int:
    """Combined forward-progress count: raw + filtered .h5. Used only for
    the hard backstop timeout — main stall signal is GPU utilization."""
    return count_raw_h5(folder) + count_filtered(folder)


import re as _re
# Match the tail of a tqdm bar: "14000/117450 [09:33<25:01, 23.97it/s"
_TQDM_RE = _re.compile(r'(\d+)\s*/\s*(\d+)\s*\[[^\]]*?,\s*(\d+\.?\d*)\s*it/s')
_FPS_RE = _re.compile(r'(\d+\.?\d*)\s*it/s')
_VIDEO_RE = _re.compile(r'Starting to analyze\s+(.+?\.mp4)', _re.IGNORECASE)


def parse_dlc_log(log_path: Path) -> dict:
    """Parse the most recent tqdm/video lines from DLC's log file.

    Reads the whole log so the 'Starting to analyze' marker for the current
    video is always found, even after hours of tqdm output have pushed it
    far back. Only tqdm matches AFTER the latest marker count — so the
    per-video bar resets cleanly at every video transition.

    Cost: log stays small (a few MB even after a full day of DLC). Reading
    +regex over that is ~50ms per call, ~once per minute. Trivial.
    """
    out = {'fps': None, 'cur_frames': None, 'total_frames': None,
           'current_video': None}
    if not log_path.is_file():
        return out
    try:
        with open(log_path, 'rb') as f:
            data = f.read()
    except Exception:
        return out
    text = data.decode('utf-8', errors='replace').replace('\r', '\n')

    # Identify the current video first (most recent 'Starting to analyze')
    v_matches = list(_VIDEO_RE.finditer(text))
    if v_matches:
        last_v = v_matches[-1]
        try:
            out['current_video'] = Path(
                last_v.group(1).strip().strip('"')
            ).name
        except Exception:
            pass
        # Only look at log content AFTER the current video's start line.
        # This prevents the previous video's final tqdm bar from leaking
        # into the new video's progress display.
        scan_text = text[last_v.end():]
    else:
        scan_text = text

    tq = list(_TQDM_RE.finditer(scan_text))
    if tq:
        m = tq[-1]
        try:
            out['cur_frames'] = int(m.group(1))
            out['total_frames'] = int(m.group(2))
            out['fps'] = float(m.group(3))
        except ValueError:
            pass
    else:
        # Fallback: bare it/s in the post-marker scope (rare)
        fps_only = _FPS_RE.findall(scan_text)
        if fps_only:
            try:
                out['fps'] = float(fps_only[-1])
            except ValueError:
                pass
    return out


def recent_fps(log_path: Path) -> float | None:
    return parse_dlc_log(log_path).get('fps')


def log_mtime_age_seconds(log_path: Path) -> float:
    if not log_path.is_file():
        return 0.0
    try:
        return time.time() - log_path.stat().st_mtime
    except Exception:
        return 0.0


def kill_proc_tree(proc: subprocess.Popen) -> None:
    """Best-effort kill of the DLC process and any GPU children."""
    if proc.poll() is not None:
        return
    try:
        # On Windows, taskkill /T kills the whole tree including any
        # CUDA worker subprocesses DLC may spawn.
        subprocess.run(
            ['taskkill', '/F', '/T', '/PID', str(proc.pid)],
            capture_output=True, timeout=30,
        )
    except Exception as e:
        step(f'  ! taskkill failed: {e}')
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=30)
    except Exception:
        pass


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


def progress_text(
    cur: int, total: int, t_start: float,
    label: str = 'THC Withdrawal',
    raw: int | None = None, fps: float | None = None,
    cur_video: str | None = None,
    cur_frames: int | None = None, total_frames: int | None = None,
    extra: str = '',
) -> str:
    """Top bar tracks RAW .h5 count (one per video as analyze finishes), not
    filtered. Filterpredictions only runs at end and bumps filtered from 0/12
    -> target in a few minutes — that pass deserves its own short bar."""
    elapsed = int(time.time() - t_start)
    fps_str = f' | FPS: {fps:.1f}' if fps is not None else ''
    raw_val = raw if raw is not None else cur

    # Phase: 'analyze' (raw climbing), 'filter' (raw==total, filtered<target),
    # or 'done'.
    if cur >= total:
        phase = 'done'
    elif raw_val >= total:
        phase = 'filter'
    else:
        phase = 'analyze'

    # Top bar: show whichever number is climbing right now
    if phase == 'filter':
        primary, primary_total = cur, total
        primary_label = 'filtered .h5'
    else:
        primary, primary_total = raw_val, total
        primary_label = 'videos analyzed'
    bar = make_bar(primary, primary_total)
    pct = 100 * primary / primary_total if primary_total else 0

    # ETA
    eta_str = '...'
    if phase == 'done':
        eta_str = 'done'
    elif phase == 'filter':
        eta_str = '~few min (filterpredictions)'
    elif (fps and fps > 1.0 and total_frames and cur_frames is not None):
        in_flight_remaining = max(0, total_frames - cur_frames) / fps
        videos_after_current = max(0, total - raw_val - 1)
        per_video_seconds = total_frames / fps
        eta = int(in_flight_remaining + videos_after_current * per_video_seconds)
        eta_str = f'~{fmt_dur(eta)}'

    lines = [
        f'**{label} -- DLC progress** ({phase})',
        f'`[{bar}] {primary}/{primary_total} {primary_label} ({pct:5.1f}%)`',
        f'`Filtered: {cur}/{total} (filterpredictions runs at end)`',
    ]
    if cur_video and cur_frames is not None and total_frames and phase == 'analyze':
        v_pct = 100 * cur_frames / total_frames
        v_bar = make_bar(cur_frames, total_frames)
        v_remaining = ''
        if fps and fps > 1.0:
            v_eta = int(max(0, total_frames - cur_frames) / fps)
            v_remaining = f' (~{fmt_dur(v_eta)} left)'
        lines.append(
            f'Current: `{cur_video}`{v_remaining}\n'
            f'`[{v_bar}] {cur_frames}/{total_frames} ({v_pct:5.1f}%)`'
        )
    elif cur_video and phase == 'analyze':
        lines.append(f'Current: `{cur_video}` (loading...)')
    lines.append(f'Elapsed: {fmt_dur(elapsed)} | ETA: {eta_str}{fps_str}{extra}')
    return '\n'.join(lines)


def step(msg: str) -> None:
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}', flush=True)


def run_dlc_managed(
    dlc_script_relpath: str,
    folder: Path,
    target: int,
    label: str,
    log_basename: str,
) -> int:
    """Launch DLC, watch live progress, restart on stall. Returns 0 on success.

    Stall = no change in (raw+filtered .h5 count) for STALL_MINUTES. The
    underlying batch scripts are idempotent (skip-existing on shuffle9 .h5),
    so killing and re-running picks up where it left off.
    """
    log_path = REPO / 'scripts' / 'research' / log_basename

    t_start = time.time()
    cur = count_filtered(folder)
    msg_id = discord_create_progress_msg(progress_text(cur, target, t_start, label=label))
    step(f'[{label}] progress msg id={msg_id} (start={cur}/{target})')

    restarts = 0
    last_file_progress = progress_signal(folder)
    last_file_progress_ts = time.time()
    last_fps_healthy_ts = time.time()
    last_edit_ts = 0.0

    while True:
        log_f = open(log_path, 'a', encoding='utf-8')
        log_f.write(f'\n\n[{datetime.now().isoformat(timespec="seconds")}] '
                    f'Launch attempt {restarts + 1}\n')
        log_f.flush()
        env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        proc = subprocess.Popen(
            [DLC_PYTHON, dlc_script_relpath],
            cwd=str(REPO), env=env,
            stdout=log_f, stderr=subprocess.STDOUT,
        )
        step(f'[{label}] DLC launched (pid={proc.pid}, attempt {restarts + 1})')
        if restarts > 0:
            discord_post(
                f'[{label}] DLC restarted (attempt {restarts + 1}) after '
                f'stall recovery. Resuming from {cur}/{target} filtered .h5.'
            )
        # Reset stall timers on restart
        last_fps_healthy_ts = time.time()
        last_file_progress = progress_signal(folder)
        last_file_progress_ts = time.time()

        stalled = False
        last_video_seen = None
        while proc.poll() is None:
            time.sleep(POLL_SECONDS)
            cur = count_filtered(folder)
            raw = count_raw_h5(folder)
            sig = progress_signal(folder)
            log_info = parse_dlc_log(log_path)
            fps = log_info.get('fps')
            cur_video = log_info.get('current_video')
            cur_frames = log_info.get('cur_frames')
            total_frames = log_info.get('total_frames')
            log_age = log_mtime_age_seconds(log_path)
            now = time.time()

            if sig != last_file_progress:
                last_file_progress = sig
                last_file_progress_ts = now
            if fps is not None and fps >= MIN_HEALTHY_FPS:
                last_fps_healthy_ts = now

            # filterpredictions phase: all raw .h5 written, only filtering
            # left. tqdm doesn't run there → exempt from FPS-based stall.
            in_filter_phase = (raw >= target and cur < target)

            extra = f' | restarts: {restarts}' if restarts > 0 else ''
            if in_filter_phase:
                extra += ' | filterpredictions phase'

            # Edit if file count changed, video changed, or 60s passed
            # (frame counter advances every 60s during a long video).
            video_changed = (cur_video != last_video_seen)
            if video_changed:
                last_video_seen = cur_video
            if (cur != last_file_progress or video_changed
                    or (now - last_edit_ts) >= 60):
                if msg_id:
                    discord_edit_msg(
                        msg_id,
                        progress_text(
                            cur, target, t_start, label=label,
                            raw=raw, fps=fps, cur_video=cur_video,
                            cur_frames=cur_frames, total_frames=total_frames,
                            extra=extra,
                        ),
                    )
                last_edit_ts = now

            # Primary stall signal: FPS dropped below MIN_HEALTHY_FPS for too long
            fps_idle_seconds = now - last_fps_healthy_ts
            if (not in_filter_phase
                    and fps_idle_seconds > FPS_STALL_TIMEOUT_MIN * 60):
                step(f'[{label}] STALL (FPS): no it/s >= {MIN_HEALTHY_FPS}'
                     f' for {fps_idle_seconds / 60:.0f}min. Killing DLC.')
                discord_post(
                    f'[{label}] DLC stall: FPS < {MIN_HEALTHY_FPS} '
                    f'(or unparseable) for {fps_idle_seconds / 60:.0f}min '
                    f'during analyze phase. Killing tree, waiting '
                    f'{KILL_GRACE_SECONDS}s for GPU release, then restarting.'
                )
                kill_proc_tree(proc)
                stalled = True
                break

            # Process-died signal: log file untouched
            if (not in_filter_phase
                    and log_age > LOG_MTIME_STALL_MIN * 60):
                step(f'[{label}] STALL (log): log untouched '
                     f'{log_age / 60:.0f}min. Killing DLC.')
                discord_post(
                    f'[{label}] DLC stall: log untouched for '
                    f'{log_age / 60:.0f}min — process likely hung. '
                    f'Killing tree and restarting.'
                )
                kill_proc_tree(proc)
                stalled = True
                break

            # Hard backstop: zero file progress for HARD_STALL_TIMEOUT_MIN
            file_idle_seconds = now - last_file_progress_ts
            if file_idle_seconds > HARD_STALL_TIMEOUT_MIN * 60:
                step(f'[{label}] HARD STALL: no .h5 progress in '
                     f'{file_idle_seconds / 60:.0f}min. Killing DLC.')
                discord_post(
                    f'[{label}] hard stall: no .h5 file in '
                    f'{file_idle_seconds / 60:.0f}min. Killing tree and '
                    f'restarting.'
                )
                kill_proc_tree(proc)
                stalled = True
                break

        log_f.close()

        cur = count_filtered(folder)
        rc = proc.returncode if proc.returncode is not None else -1

        if cur >= target and not stalled:
            step(f'[{label}] reached target {cur}/{target} (rc={rc})')
            if msg_id:
                discord_edit_msg(msg_id, progress_text(cur, target, t_start, label=label))
            return 0

        if stalled:
            # Wait for GPU to release VRAM before restart
            time.sleep(KILL_GRACE_SECONDS)
            last_file_progress = progress_signal(folder)
            last_file_progress_ts = time.time()
            last_fps_healthy_ts = time.time()
            restarts += 1
            if restarts > MAX_RESTARTS:
                discord_post(
                    f'[{label}] hit MAX_RESTARTS={MAX_RESTARTS}, still '
                    f'{cur}/{target}. Giving up.'
                )
                return 1
            continue

        # DLC exited on its own. If we hit target, success; else, restart.
        if rc != 0:
            step(f'[{label}] DLC exited with rc={rc} at {cur}/{target}, '
                 f'restarting.')
            discord_post(
                f'[{label}] DLC exited with rc={rc} at {cur}/{target} '
                f'(not stalled but not done). Restarting.'
            )
            time.sleep(KILL_GRACE_SECONDS)
            restarts += 1
            if restarts > MAX_RESTARTS:
                discord_post(
                    f'[{label}] hit MAX_RESTARTS={MAX_RESTARTS}, still '
                    f'{cur}/{target}. Giving up.'
                )
                return 1
            continue

        # Clean exit but target not hit (rare — shouldn't happen since
        # the batch script processes all pending). Treat as restart.
        step(f'[{label}] DLC exited cleanly but {cur}/{target} done. Restarting.')
        time.sleep(KILL_GRACE_SECONDS)
        restarts += 1
        if restarts > MAX_RESTARTS:
            return 1


def main() -> int:
    # ---- Phase 1: THC DLC (managed: live FPS, stall detection, auto-restart) ----
    discord_post(
        f'THC orchestrator started. Watching `Videos/` for '
        f'{THC_TARGET} `*_filtered.h5`. Live progress bar updates every '
        f'{POLL_SECONDS}s with FPS — stall detector kills + restarts DLC '
        f'if FPS stays below {MIN_HEALTHY_FPS} for {FPS_STALL_TIMEOUT_MIN}min.'
    )
    rc = run_dlc_managed(
        'scripts/research/dlc_batch_thc_withdrawal.py',
        THC_DIR, THC_TARGET, 'THC Withdrawal',
        log_basename='thc_dlc.log',
    )
    if rc != 0:
        discord_post(
            f'THC DLC failed after retries. Not starting post-DLC chain. '
            f'Check thc_dlc.log + thc_orchestrator.log.'
        )
        return rc
    discord_post(
        f'THC DLC complete. Starting post-DLC chain '
        f'(features-with-flow -> predict-all -> group-analyze).'
    )

    # ---- Phase 2: post-DLC chain ----
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
    proc = subprocess.run(
        ['py', '-X', 'utf8', 'scripts/research/thc_post_dlc_chain.py'],
        cwd=str(REPO), env=env,
    )
    rc = proc.returncode
    step(f'Post-DLC chain exit: {rc}')
    if rc != 0:
        discord_post(
            f'THC chain FAILED (exit {rc}). See thc_orchestrator.log. '
            f'NOT starting SNLT cohort2 — fix THC chain first.'
        )
        return rc
    discord_post(
        f'THC analysis complete. Figures pushed above. '
        f'Starting SNLT cohort2 DLC ({SNLT_TARGET} videos) with shuffle 9.'
    )

    # ---- Phase 3: SNLT cohort2 DLC (managed) ----
    rc = run_dlc_managed(
        'scripts/research/dlc_batch_2605_cohort2.py',
        SNLT_DIR, SNLT_TARGET, 'SNLT cohort2',
        log_basename='snlt_cohort2_dlc.log',
    )
    if rc != 0:
        discord_post(f'SNLT cohort2 DLC failed after retries.')
        return rc

    cur_s = count_filtered(SNLT_DIR)
    discord_post(
        f'All done. THC analysis (figures above) + SNLT cohort2 DLC '
        f'({cur_s}/{SNLT_TARGET} filtered .h5).'
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
