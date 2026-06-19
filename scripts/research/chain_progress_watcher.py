"""
Live Discord progress bar for the THC -> Rim dose-response chain.

Polls chain_thc_then_rim.log every 60 s, parses tqdm + per-video
markers, silently PATCHes a single Discord message so the channel
doesn't get spammed.

Tracks both phases:
  Phase 1 = THC (25 filtered.h5 target)
  Phase 2 = Rim (18 filtered.h5 target, only counts after phase 2 starts)

Exits when chain process dies OR phase 2 reaches target.

Run in background (regular py env):
  py -X utf8 -u scripts/research/chain_progress_watcher.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

LOG = Path(r"E:\RSVIDS\Blackbox\chain_thc_then_rim.log")
THC_VIDS = Path(r"E:\RSVIDS\Blackbox\260512_THC_Rim_Cohort\videos")
RIM_VIDS = Path(r"E:\RSVIDS\Blackbox\260515_Rim_DoseResp\videos")
THC_TARGET = 25
RIM_TARGET = 18

WEBHOOK = (
    ""
)
POLL_S = 60
BAR_W = 24
WATCHER_MAX_S = 72 * 3600   # 3 days

_TQDM = re.compile(r"(\d+)\s*/\s*(\d+)\s*\[[^\]]*?,\s*(\d+\.?\d*)\s*it/s")
_VIDEO = re.compile(r"Starting to analyze\s+(.+?\.mp4)", re.IGNORECASE)
_PHASE = re.compile(r"=== Phase (\d): ([^=]+?) ===")


def _post(url, payload, method="POST"):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        method=method,
        headers={"Content-Type": "application/json",
                 "User-Agent": "PixelPaws-Chain/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            body = r.read()
            return json.loads(body) if body else None
    except Exception as e:
        print(f"discord {method} failed: {e}", flush=True)
        return None


def discord_create(text):
    r = _post(WEBHOOK + "?wait=true", {"content": text})
    return r.get("id") if r else None


def discord_edit(msg_id, text):
    if msg_id:
        _post(f"{WEBHOOK}/messages/{msg_id}", {"content": text}, method="PATCH")


def bar(cur, total, width=BAR_W):
    if total <= 0:
        return "-" * width
    f = max(0, min(width, cur * width // total))
    return "#" * f + "-" * (width - f)


def fmt_dur(s):
    s = int(s)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def count_filtered(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir()
               if p.is_file() and "shuffle9" in p.name and p.name.endswith("_filtered.h5"))


def count_raw(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir()
               if p.is_file() and "shuffle9" in p.name
               and p.name.endswith(".h5") and not p.name.endswith("_filtered.h5"))


def parse_log() -> dict:
    out = {"phase_num": None, "phase_name": None,
           "current_video": None, "cur_frames": None,
           "total_frames": None, "fps": None}
    if not LOG.is_file():
        return out
    try:
        text = LOG.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    except OSError:
        return out
    ph = list(_PHASE.finditer(text))
    if ph:
        m = ph[-1]
        out["phase_num"] = int(m.group(1))
        out["phase_name"] = m.group(2).strip()
    v = list(_VIDEO.finditer(text))
    scan = text
    if v:
        last = v[-1]
        out["current_video"] = Path(last.group(1).strip().strip('"')).name
        scan = text[last.end():]
    tq = list(_TQDM.finditer(scan))
    if tq:
        m = tq[-1]
        try:
            out["cur_frames"] = int(m.group(1))
            out["total_frames"] = int(m.group(2))
            out["fps"] = float(m.group(3))
        except ValueError:
            pass
    return out


def build_message(state, t_start):
    thc_filt = count_filtered(THC_VIDS)
    thc_raw = count_raw(THC_VIDS)
    rim_filt = count_filtered(RIM_VIDS)
    rim_raw = count_raw(RIM_VIDS)
    phase_num = state.get("phase_num") or (1 if thc_filt < THC_TARGET else 3)
    phase_name = state.get("phase_name") or "?"
    elapsed = time.time() - t_start
    fps = state.get("fps") or 0.0
    cur_v = state.get("current_video") or "(...)"
    cur_f = state.get("cur_frames") or 0
    tot_f = state.get("total_frames") or 0

    # Per-video ETA
    if fps > 1.0 and tot_f and cur_f < tot_f:
        v_eta = fmt_dur((tot_f - cur_f) / fps)
    elif tot_f and cur_f >= tot_f:
        v_eta = "done"
    else:
        v_eta = "..."

    lines = ["**THC -> Rim dose-response chain -- live progress**",
             f"Phase {phase_num}: `{phase_name}`",
             f"Elapsed: {fmt_dur(elapsed)}",
             ""]

    # THC progress
    lines.append(f"**THC cohort** (target {THC_TARGET} filtered .h5)")
    lines.append(f"  raw:      `[{bar(thc_raw, THC_TARGET)}] {thc_raw}/{THC_TARGET}`")
    lines.append(f"  filtered: `[{bar(thc_filt, THC_TARGET)}] {thc_filt}/{THC_TARGET}`")
    lines.append("")

    # Rim progress (only meaningful after phase 2 starts)
    rim_total = RIM_TARGET if (RIM_VIDS.is_dir() and len(list(RIM_VIDS.glob("*.mp4"))) > 0) else 0
    if rim_total > 0:
        lines.append(f"**Rim cohort** (target {RIM_TARGET} filtered .h5)")
        lines.append(f"  raw:      `[{bar(rim_raw, RIM_TARGET)}] {rim_raw}/{RIM_TARGET}`")
        lines.append(f"  filtered: `[{bar(rim_filt, RIM_TARGET)}] {rim_filt}/{RIM_TARGET}`")
        lines.append("")

    # Current video detail
    if cur_v and tot_f and phase_num in (1, 3):
        v_pct = 100 * cur_f / max(tot_f, 1)
        lines.append(f"Current: `{cur_v}` ({v_pct:.1f}%, {fps:.1f} it/s, ~{v_eta} left)")

    return "\n".join(lines)


def main():
    if not LOG.is_file():
        print(f"Waiting for {LOG} to appear...", flush=True)
        while not LOG.is_file():
            time.sleep(5)

    t_start = time.time()
    msg_id = discord_create(build_message(parse_log(), t_start))
    print(f"Discord msg id: {msg_id}", flush=True)

    while True:
        if time.time() - t_start > WATCHER_MAX_S:
            print("max watcher time hit, exiting", flush=True)
            break
        time.sleep(POLL_S)
        state = parse_log()
        msg = build_message(state, t_start)
        discord_edit(msg_id, msg)
        # Console echo (one liner)
        thc_filt = count_filtered(THC_VIDS)
        rim_filt = count_filtered(RIM_VIDS)
        print(f"[{time.strftime('%H:%M:%S')}] THC {thc_filt}/{THC_TARGET}  "
              f"Rim {rim_filt}/{RIM_TARGET}  fps={state.get('fps')}",
              flush=True)
        # Exit when both phases at target
        if thc_filt >= THC_TARGET and rim_filt >= RIM_TARGET:
            print("both targets reached, exiting", flush=True)
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
