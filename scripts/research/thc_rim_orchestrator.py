"""
End-to-end orchestrator for the 260512 THC rimonabant cohort.

Single long-running process that:
  1. Launches DLC analyze + filterpredictions as a managed subprocess.
  2. Posts a live Discord progress bar (silent-edits the same message
     every 60s — no notification spam).
  3. Detects stalls (FPS < 8 it/s for 10 min, log untouched 5 min, or
     no .h5 progress for 4 h) and auto-kills + restarts DLC up to 5
     times. Idempotent batch script picks up from the last completed
     video.
  4. On DLC completion, runs features+predict (single pass with optical
     flow, 5 classifiers) and then group analysis (chronic x challenge
     2x2 design, window caps pre=30 min / post=60 min).
  5. Posts milestone Discord messages at each phase boundary.

Stall thresholds match the documented DLCPP agent settings.

Run in background:
  py -X utf8 -u scripts/research/thc_rim_orchestrator.py \
      >> E:/RSVIDS/Blackbox/260512_THC_Rim_Cohort/orchestrator.log 2>&1
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
PROJECT_ROOT = Path(r"E:\RSVIDS\Blackbox\260512_THC_Rim_Cohort")
VIDEO_DIR = PROJECT_ROOT / "videos"
DLC_LOG = PROJECT_ROOT / "dlc_batch.log"
DLC_PYTHON = r"C:\Users\Gereau\anaconda3\envs\DEEPLABCUT\python.exe"
DLC_BATCH_SCRIPT = r"E:\PixelPaws\scripts\research\dlc_batch_thc_rim.py"
PREDICT_SCRIPT = r"E:\PixelPaws\scripts\research\thc_rim_predict_all.py"
GROUP_ANALYZE_SCRIPT = r"E:\PixelPaws\scripts\research\thc_rim_group_analyze.py"

WEBHOOK = (
    ""
)

BAR_WIDTH = 24
POLL_SECONDS = 60
MAX_RESTARTS = 5
KILL_GRACE_SECONDS = 60
MIN_HEALTHY_FPS = 8.0
FPS_STALL_TIMEOUT_MIN = 10
LOG_MTIME_STALL_MIN = 5
HARD_STALL_TIMEOUT_MIN = 240


# --------------------------------------------------------------------------- #
# Discord helpers
# --------------------------------------------------------------------------- #

def _post_json(url, payload, method="POST"):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method=method,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "PixelPaws-Orchestrator/1.0 (+thc_rim_orchestrator.py)",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read()
            return json.loads(body) if body else None
    except Exception as e:
        step(f"  ! Discord {method} failed: {e}")
        return None


def discord_create(text):
    resp = _post_json(WEBHOOK + "?wait=true", {"content": text})
    return resp.get("id") if resp else None


def discord_edit(msg_id, text):
    if not msg_id:
        return
    _post_json(f"{WEBHOOK}/messages/{msg_id}", {"content": text}, method="PATCH")


def discord_post(text):
    _post_json(WEBHOOK, {"content": text})


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #

def make_bar(cur, total, width=BAR_WIDTH):
    if total <= 0:
        return "-" * width
    filled = max(0, min(width, cur * width // total))
    return "#" * filled + "-" * (width - filled)


def fmt_dur(seconds):
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def step(msg):
    line = f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode(), flush=True)


# --------------------------------------------------------------------------- #
# DLC log parsing
# --------------------------------------------------------------------------- #

_TQDM_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*\[[^\]]*?,\s*(\d+\.?\d*)\s*it/s")
_VIDEO_RE = re.compile(r"Starting to analyze\s+(.+?\.mp4)", re.IGNORECASE)
_SAVE_RE = re.compile(r"Saving results in\s+.+\.h5", re.IGNORECASE)


def parse_dlc_log(path: Path) -> dict:
    out = {"fps": None, "cur_frames": None, "total_frames": None,
           "current_video": None, "completed_saves": 0,
           "filter_started": False}
    if not path.is_file():
        return out
    try:
        text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    except OSError:
        return out

    v_matches = list(_VIDEO_RE.finditer(text))
    if v_matches:
        last_v = v_matches[-1]
        out["current_video"] = Path(last_v.group(1).strip().strip('"')).name
        scan = text[last_v.end():]
    else:
        scan = text

    out["completed_saves"] = len(_SAVE_RE.findall(text))
    out["filter_started"] = "filterpredictions:" in text

    tq = list(_TQDM_RE.finditer(scan))
    if tq:
        m = tq[-1]
        try:
            out["cur_frames"] = int(m.group(1))
            out["total_frames"] = int(m.group(2))
            out["fps"] = float(m.group(3))
        except ValueError:
            pass
    return out


def log_mtime_age(path: Path) -> float:
    if not path.is_file():
        return 0.0
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return 0.0


def count_videos() -> int:
    return len(list(VIDEO_DIR.glob("*.mp4")))


def count_raw_h5() -> int:
    return sum(1 for p in VIDEO_DIR.iterdir()
               if p.is_file() and "shuffle9" in p.name
               and p.name.endswith(".h5") and not p.name.endswith("_filtered.h5"))


def count_filtered_h5() -> int:
    return sum(1 for p in VIDEO_DIR.iterdir()
               if p.is_file() and "shuffle9" in p.name
               and p.name.endswith("_filtered.h5"))


# --------------------------------------------------------------------------- #
# Subprocess management
# --------------------------------------------------------------------------- #

def kill_proc_tree(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    try:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True, timeout=30,
        )
    except Exception as e:
        step(f"  ! taskkill failed: {e}")
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=30)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Progress text builder
# --------------------------------------------------------------------------- #

def progress_text(state, t_start, n_videos, restarts):
    raw = count_raw_h5()
    filt = count_filtered_h5()
    cur_v = state["current_video"] or "(loading...)"
    cur_frames = state["cur_frames"] or 0
    total_frames = state["total_frames"] or 0
    fps = state["fps"] or 0.0
    in_filter = state["filter_started"] and raw >= n_videos and filt < n_videos
    if filt >= n_videos:
        phase = "done"
    elif in_filter:
        phase = "filter"
    else:
        phase = "analyze"

    elapsed = time.time() - t_start

    if phase == "filter":
        primary, primary_total, primary_lbl = filt, n_videos, "filtered .h5"
    else:
        primary, primary_total, primary_lbl = raw, n_videos, "videos analyzed"
    bar = make_bar(primary, primary_total)
    pct = 100 * primary / max(primary_total, 1)

    # ETA
    if phase == "done":
        eta_str = "done"
    elif phase == "filter":
        eta_str = "~few min"
    elif fps > 1.0 and total_frames and cur_frames is not None:
        in_flight = max(0, total_frames - cur_frames) / fps
        videos_after = max(0, n_videos - raw - 1)
        per_video_s = total_frames / fps
        eta_str = f"~{fmt_dur(in_flight + videos_after * per_video_s)}"
    else:
        eta_str = "..."

    lines = [
        f"**THC x rimonabant -- DLC progress** (`{phase}`)",
        f"`[{bar}] {primary}/{primary_total} {primary_lbl}  ({pct:5.1f}%)`",
        f"`Filtered: {filt}/{n_videos}`",
    ]
    if phase == "analyze" and cur_frames is not None and total_frames:
        v_bar = make_bar(cur_frames, total_frames)
        v_pct = 100 * cur_frames / max(total_frames, 1)
        v_eta = ""
        if fps > 1.0:
            v_eta = f" (~{fmt_dur((total_frames - cur_frames) / fps)} left)"
        lines.append(f"Current: `{cur_v}`{v_eta}")
        lines.append(f"`[{v_bar}] {cur_frames:,}/{total_frames:,} ({v_pct:5.1f}%) @ {fps:.1f} it/s`")
    elif phase == "analyze":
        lines.append(f"Current: `{cur_v}` (loading...)")
    extra = f" | restarts: {restarts}" if restarts > 0 else ""
    lines.append(f"Elapsed: {fmt_dur(elapsed)} | ETA: {eta_str}{extra}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Managed DLC run
# --------------------------------------------------------------------------- #

def run_dlc_managed() -> int:
    n_videos = count_videos()
    if n_videos == 0:
        step("No videos in cohort. Nothing to do.")
        return 1

    t_start = time.time()
    msg_id = discord_create(
        f"THC x rimonabant orchestrator started. Watching `videos/` for "
        f"{n_videos} `*_filtered.h5`. Live progress bar updates every "
        f"{POLL_SECONDS}s; stall detector kills + restarts DLC if FPS stays "
        f"below {MIN_HEALTHY_FPS} for {FPS_STALL_TIMEOUT_MIN}min."
    )
    step(f"Discord msg id: {msg_id}")

    restarts = 0
    while True:
        # Truncate prior dlc log so each launch's parse is fresh.
        DLC_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(DLC_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n\n[{datetime.now().isoformat(timespec='seconds')}] "
                    f"Launch attempt {restarts + 1}\n")
        log_f = open(DLC_LOG, "a", encoding="utf-8")

        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.Popen(
            [DLC_PYTHON, DLC_BATCH_SCRIPT],
            cwd=str(REPO), env=env,
            stdout=log_f, stderr=subprocess.STDOUT,
        )
        step(f"DLC launched (pid={proc.pid}, attempt {restarts + 1})")
        if restarts > 0:
            discord_post(
                f"DLC restarted (attempt {restarts + 1}) after stall. "
                f"Resuming from {count_filtered_h5()}/{n_videos} filtered."
            )

        last_file_sig = count_raw_h5() + count_filtered_h5()
        last_file_sig_ts = time.time()
        last_fps_healthy_ts = time.time()
        last_edit_ts = 0.0
        last_video_seen = None
        stalled = False

        while proc.poll() is None:
            time.sleep(POLL_SECONDS)
            n_videos = max(count_videos(), n_videos)  # user may add more
            state = parse_dlc_log(DLC_LOG)
            sig = count_raw_h5() + count_filtered_h5()
            now = time.time()

            if sig != last_file_sig:
                last_file_sig = sig
                last_file_sig_ts = now
            if state["fps"] is not None and state["fps"] >= MIN_HEALTHY_FPS:
                last_fps_healthy_ts = now

            in_filter = state["filter_started"] and count_raw_h5() >= n_videos and count_filtered_h5() < n_videos

            video_changed = state["current_video"] != last_video_seen
            if video_changed:
                last_video_seen = state["current_video"]
            if video_changed or (now - last_edit_ts) >= POLL_SECONDS:
                discord_edit(msg_id, progress_text(state, t_start, n_videos, restarts))
                last_edit_ts = now

            log_age = log_mtime_age(DLC_LOG)
            fps_idle = now - last_fps_healthy_ts
            file_idle = now - last_file_sig_ts

            if not in_filter and fps_idle > FPS_STALL_TIMEOUT_MIN * 60:
                step(f"STALL (FPS): no it/s >= {MIN_HEALTHY_FPS} for "
                     f"{fps_idle/60:.0f}min. Killing.")
                discord_post(
                    f":warning: DLC stall: FPS < {MIN_HEALTHY_FPS} for "
                    f"{fps_idle/60:.0f}min. Killing tree, waiting "
                    f"{KILL_GRACE_SECONDS}s, then restarting."
                )
                kill_proc_tree(proc)
                stalled = True
                break

            if not in_filter and log_age > LOG_MTIME_STALL_MIN * 60:
                step(f"STALL (log): untouched {log_age/60:.0f}min.")
                discord_post(
                    f":warning: DLC log untouched for {log_age/60:.0f}min - "
                    f"process likely hung. Killing tree and restarting."
                )
                kill_proc_tree(proc)
                stalled = True
                break

            if file_idle > HARD_STALL_TIMEOUT_MIN * 60:
                step(f"HARD STALL: no .h5 progress in {file_idle/60:.0f}min.")
                discord_post(
                    f":warning: HARD STALL: no .h5 progress in "
                    f"{file_idle/60:.0f}min. Killing tree and restarting."
                )
                kill_proc_tree(proc)
                stalled = True
                break

        log_f.close()

        n_videos = max(count_videos(), n_videos)
        filt = count_filtered_h5()
        rc = proc.returncode if proc.returncode is not None else -1

        if filt >= n_videos and not stalled:
            step(f"DLC complete: {filt}/{n_videos} filtered (rc={rc})")
            discord_edit(msg_id, progress_text(
                parse_dlc_log(DLC_LOG), t_start, n_videos, restarts,
            ))
            return 0

        if stalled or rc != 0:
            time.sleep(KILL_GRACE_SECONDS)
            restarts += 1
            if restarts > MAX_RESTARTS:
                discord_post(
                    f":x: Hit MAX_RESTARTS={MAX_RESTARTS}, still {filt}/{n_videos}. "
                    f"Giving up. Check `{DLC_LOG}`."
                )
                return 1
            continue

        # Clean exit but target not hit — retry.
        step(f"DLC exited cleanly at {filt}/{n_videos}. Retrying.")
        time.sleep(KILL_GRACE_SECONDS)
        restarts += 1
        if restarts > MAX_RESTARTS:
            return 1


def run_post_dlc_chain() -> int:
    """Features (with flow, single pass) -> predict 5 classifiers -> group analyze."""
    discord_post("DLC complete. Starting features + predict (5 classifiers) -> group analyze.")

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    step("Launching predict_all...")
    rc = subprocess.run(
        ["py", "-X", "utf8", "-u", PREDICT_SCRIPT],
        cwd=str(REPO), env=env,
    ).returncode
    if rc != 0:
        discord_post(f":x: thc_rim_predict_all.py exited rc={rc}. Stopping.")
        return rc

    step("Launching group_analyze...")
    rc = subprocess.run(
        ["py", "-X", "utf8", "-u", GROUP_ANALYZE_SCRIPT],
        cwd=str(REPO), env=env,
    ).returncode
    if rc != 0:
        discord_post(f":x: thc_rim_group_analyze.py exited rc={rc}.")
        return rc

    discord_post("All done. Per-group figures pushed above.")
    return 0


def main() -> int:
    step("orchestrator starting.")
    rc = run_dlc_managed()
    if rc != 0:
        step(f"DLC management exited rc={rc}; not running post-DLC chain.")
        return rc
    return run_post_dlc_chain()


if __name__ == "__main__":
    sys.exit(main())
