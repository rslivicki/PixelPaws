"""
Live Discord progress bar for the 260512 THC rimonabant DLC batch.

Polls dlc_batch.log every 60 seconds, parses tqdm progress + per-video
"Starting to analyze" / "Saving results in" markers, and silently
PATCHes a single Discord message so we get a live progress bar without
notification spam.

Exits when:
  - "Done." marker appears in the log (analyze + filter complete), OR
  - the log hasn't been modified in 30 min (assume orchestrator dead), OR
  - the watcher has been running for 24 h (safety cap).

Run in background — no env requirements beyond stdlib + the running log:
  py scripts/research/thc_rim_dlc_progress_watcher.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

LOG = Path(r"E:\RSVIDS\Blackbox\260512_THC_Rim_Cohort\dlc_batch.log")
VIDEOS_DIR = Path(r"E:\RSVIDS\Blackbox\260512_THC_Rim_Cohort\videos")
WEBHOOK = (
    ""
)
POLL_INTERVAL_S = 60
LOG_STALL_TIMEOUT_S = 30 * 60
WATCHER_MAX_S = 24 * 3600
BAR_WIDTH = 24


def _send(url, payload, method="POST"):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method=method,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "PixelPaws-Chain/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read()
            return json.loads(body) if body else None
    except Exception as e:
        print(f"  ! Discord {method} failed: {e}", flush=True)
        return None


def discord_create(text):
    resp = _send(WEBHOOK + "?wait=true", {"content": text})
    return resp.get("id") if resp else None


def discord_edit(msg_id, text):
    if not msg_id:
        return
    _send(f"{WEBHOOK}/messages/{msg_id}", {"content": text}, method="PATCH")


def discord_post(text):
    _send(WEBHOOK, {"content": text})


def make_bar(cur, total, width=BAR_WIDTH):
    if total <= 0:
        return "░" * width
    filled = max(0, min(width, cur * width // total))
    return "█" * filled + "░" * (width - filled)


def fmt_dur(seconds):
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


# tqdm progress line:  17504/221256 [07:31<1:18:19, 43.35it/s]
PROGRESS_RE = re.compile(
    r"(\d+)/(\d+)\s+\[[^]]*?,\s*([\d.]+)\s*it/s\]"
)
START_RE = re.compile(
    r"Starting to analyze .*?\\([^\\]+\.mp4)", re.IGNORECASE
)
SAVE_RE = re.compile(
    r"Saving results in .*?\\([^\\]+)\.h5", re.IGNORECASE
)
FRAMES_TOTAL_RE = re.compile(r"Overall # of frames:\s+(\d+)")
DURATION_RE = re.compile(r"Duration of video \[s\]:\s+([\d.]+)")


def parse_log(text: str):
    """Pull current state from the DLC batch log."""
    # All "Starting to analyze" entries; latest is current video.
    starts = START_RE.findall(text)
    saves = SAVE_RE.findall(text)
    current_video = starts[-1] if starts else None
    completed_videos = len(saves)
    started_videos = len(starts)

    # Latest tqdm progress
    matches = list(PROGRESS_RE.finditer(text))
    cur_frame = total_frames = 0
    fps = 0.0
    if matches:
        last = matches[-1]
        cur_frame = int(last.group(1))
        total_frames = int(last.group(2))
        fps = float(last.group(3))

    # Look up the current video's total frames specifically (the last
    # "Overall # of frames" before the current video's progress lines).
    return {
        "current_video": current_video,
        "started": started_videos,
        "completed": completed_videos,
        "cur_frame": cur_frame,
        "total_frames": total_frames,
        "fps": fps,
        "done": "[" in text and "] Done." in text,
        "filter_started": "filterpredictions:" in text,
    }


def count_videos() -> int:
    return len(list(VIDEOS_DIR.glob("*.mp4")))


def build_message(state, n_videos, t_start):
    cur_v = state["current_video"] or "(loading...)"
    completed = state["completed"]
    started = state["started"]
    cur_frame = state["cur_frame"]
    total_frames = state["total_frames"]
    fps = state["fps"]

    elapsed = time.time() - t_start
    video_bar = make_bar(completed, n_videos)
    frame_bar = make_bar(cur_frame, total_frames) if total_frames else "░" * BAR_WIDTH

    # ETA for current video
    eta_video_str = "..."
    if total_frames and fps > 0 and cur_frame < total_frames:
        eta_s = (total_frames - cur_frame) / fps
        eta_video_str = fmt_dur(eta_s)
    elif cur_frame >= total_frames and total_frames > 0:
        eta_video_str = "done"

    # ETA overall: assume per-video time roughly proportional to frames
    # (post = ~220k, pre = ~110k). Use observed fps.
    eta_overall_str = "..."
    if fps > 0:
        # Frames remaining = current video remaining + average of remaining videos
        cur_remaining = max(0, total_frames - cur_frame)
        videos_remaining = max(0, n_videos - completed - 1)
        # Use the current video's total as an estimate per remaining video.
        # This is rough; pre videos are smaller, so this is an upper bound.
        eta_overall_s = (cur_remaining + videos_remaining * total_frames) / fps
        eta_overall_str = fmt_dur(eta_overall_s)

    stage = "DLC analyze"
    if state["filter_started"]:
        stage = "DLC filterpredictions"
    if state["done"]:
        stage = "DLC done"

    lines = [
        "**THC x rimonabant -- DLC progress**",
        f"Stage: `{stage}`",
        f"Videos:  `[{video_bar}] {completed}/{n_videos}`",
        f"Current: `{cur_v}` (video {min(started, n_videos)}/{n_videos})",
        f"Frames:  `[{frame_bar}] {cur_frame:,}/{total_frames:,}  @ {fps:.1f} it/s`",
        f"ETA this video: {eta_video_str} | ETA overall: {eta_overall_str} | "
        f"Elapsed: {fmt_dur(elapsed)}",
    ]
    return "\n".join(lines)


def main():
    if not LOG.is_file():
        print(f"Waiting for {LOG} to appear...")
        while not LOG.is_file():
            time.sleep(5)

    n_videos = count_videos()
    print(f"Watching {LOG} -- {n_videos} videos in cohort")

    t_start = time.time()
    initial_text = LOG.read_text(errors="ignore")
    initial_state = parse_log(initial_text)
    msg_id = discord_create(build_message(initial_state, n_videos, t_start))
    print(f"Discord message id: {msg_id}")
    if not msg_id:
        print("Failed to create Discord message; running without live updates.")

    last_text_len = len(initial_text)
    last_change_t = time.time()

    while True:
        if time.time() - t_start > WATCHER_MAX_S:
            print("24h cap hit; exiting watcher.")
            break

        # Wait one poll interval, but be responsive to log growth.
        time.sleep(POLL_INTERVAL_S)

        try:
            text = LOG.read_text(errors="ignore")
        except OSError as e:
            print(f"log read failed: {e}")
            continue

        if len(text) != last_text_len:
            last_text_len = len(text)
            last_change_t = time.time()

        state = parse_log(text)
        # Re-count in case user dropped videos in mid-run.
        n_videos = max(count_videos(), n_videos)

        msg = build_message(state, n_videos, t_start)
        discord_edit(msg_id, msg)

        # Console echo for local visibility.
        print(msg.replace("\n", " | "))

        if state["done"]:
            discord_post(
                f"DLC batch finished after {fmt_dur(time.time() - t_start)}. "
                "Kicking off features + 5-classifier predict next."
            )
            break

        if time.time() - last_change_t > LOG_STALL_TIMEOUT_S:
            discord_post(
                f":warning: DLC log hasn't changed for "
                f"{fmt_dur(time.time() - last_change_t)}. "
                "Orchestrator may be stalled."
            )
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
