"""
Live Discord progress tracker for the 2605 FormOxy Left-paws cohort.

Edits a single Discord message every 60 s with:
  - cohort ingestion totals (baseline/formalin mp4, raw .h5, filtered .h5,
    feature pickles)
  - DLC current video + tqdm progress + fps (instant + 50-sample rolling
    mean + range)
  - active ffmpeg transcode target (from formoxy chain logs)
  - portal queue (ready and in-flight)

Exits when every formalin video has DLC + features, OR after 12 h cap.

Run in background (regular PixelPaws env):
  PYTHONIOENCODING=utf-8 nohup py -X utf8 -u \
      scripts/research/formoxy_progress_watcher.py > /tmp/formoxy_watcher.log 2>&1 &
  disown
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

COHORT = Path(r"E:\RSVIDS\Blackbox\2605_FormOxy_LeftPaws")
VIDS = COHORT / "videos"
FEATS = COHORT / "features"

PORTAL = Path(r"E:\RSVIDS\Video_transfer_portal\oxyformalin")

import tempfile
_TMP = Path(tempfile.gettempdir())  # resolves to %TEMP% on Windows, which is
                                     # where bash `>` redirects when you use
                                     # the `/tmp/...` shorthand from Git Bash.
CHAIN_LOGS = [
    _TMP / "formoxy_chain.log",
    _TMP / "formoxy_loop.log",
    _TMP / "formoxy_pipeline.log",
]

# Live trackers post to #dlctracker (separate from the general channel
# the chain coordinators use). See docs/private/discord_webhooks.md.
WEBHOOK = (
    ""
)

# 12 subjects target, 2 conditions each
TARGET_PER_COND = 12
POLL_S = 60
BAR_W = 24
WATCHER_MAX_S = 12 * 3600

_TQDM = re.compile(r"(\d+)\s*/\s*(\d+)\s*\[[^\]]*?,\s*(\d+\.?\d*)\s*it/s")
_VIDEO = re.compile(r"Starting to analyze\s+(.+?\.mp4)", re.IGNORECASE)
_TRANSCODE = re.compile(r"transcode\s+S\d+\s+\w+\s*\(.+?\)\s*->\s*(\S+\.mp4)")


def _post(url, payload, method="POST"):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        method=method,
        headers={"Content-Type": "application/json",
                 "User-Agent": "PixelPaws-FormOxy-Watcher/1.0"},
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


def count_files(folder: Path, pred) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_file() and pred(p.name))


def n_baseline_mp4() -> int:
    return count_files(VIDS, lambda n: n.startswith("2605_FormOxy_S") and n.endswith("_baseline.mp4"))


def n_formalin_mp4() -> int:
    return count_files(VIDS, lambda n: n.startswith("2605_FormOxy_S") and n.endswith("_formalin.mp4"))


def n_dlc_raw_formalin() -> int:
    return count_files(VIDS,
        lambda n: ("shuffle9" in n) and n.endswith(".h5")
                  and "_formalin" in n.split("DLC_")[0] and "_filtered.h5" not in n)


def n_dlc_filt_formalin() -> int:
    return count_files(VIDS,
        lambda n: ("shuffle9" in n) and n.endswith("_filtered.h5")
                  and "_formalin" in n.split("DLC_")[0])


def n_features_formalin() -> int:
    return count_files(FEATS,
        lambda n: "_formalin_features_8aed1c22" in n)


def parse_dlc() -> dict:
    out = {"video": None, "cur": None, "tot": None, "fps": None,
           "fps_avg": None, "fps_min": None, "fps_max": None}
    combined = []
    for log in CHAIN_LOGS:
        if log.is_file():
            try:
                combined.append((log.stat().st_mtime, log.read_text(encoding="utf-8", errors="replace")))
            except OSError:
                pass
    if not combined:
        return out
    combined.sort()
    text = "".join(t for _, t in combined).replace("\r", "\n")
    vids = list(_VIDEO.finditer(text))
    if vids:
        last = vids[-1]
        out["video"] = Path(last.group(1).strip().strip('"')).name
        scan = text[last.end():]
    else:
        scan = text
    tqs = list(_TQDM.finditer(scan))
    if tqs:
        last = tqs[-1]
        try:
            out["cur"] = int(last.group(1))
            out["tot"] = int(last.group(2))
            out["fps"] = float(last.group(3))
        except ValueError:
            pass
        recent = tqs[-50:]
        fpss = []
        for m in recent:
            try: fpss.append(float(m.group(3)))
            except ValueError: pass
        if fpss:
            out["fps_avg"] = sum(fpss) / len(fpss)
            out["fps_min"] = min(fpss)
            out["fps_max"] = max(fpss)
    return out


def find_transcode_active() -> str | None:
    log = Path("/tmp/formoxy_loop.log")
    if not log.is_file():
        return None
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    lines = text.splitlines()
    last_idx = None
    for i, line in enumerate(lines):
        if "transcode" in line and "->" in line:
            last_idx = i
    if last_idx is None:
        return None
    for line in lines[last_idx + 1:]:
        if "done in" in line:
            return None
    m = _TRANSCODE.search(lines[last_idx])
    return m.group(1) if m else None


def portal_state() -> dict:
    out = {"ready": [], "in_flight": []}
    if not PORTAL.is_dir():
        return out
    for p in PORTAL.rglob("*.mp4*"):
        if p.name.startswith("~syncthing~") or p.name.endswith(".tmp"):
            out["in_flight"].append(p.name)
        elif p.suffix.lower() == ".mp4":
            out["ready"].append(p.name)
    return out


def build_message(t_start: float) -> str:
    n_b = n_baseline_mp4()
    n_f = n_formalin_mp4()
    n_raw = n_dlc_raw_formalin()
    n_filt = n_dlc_filt_formalin()
    n_feat = n_features_formalin()
    portal = portal_state()
    dlc = parse_dlc()
    transcoding = find_transcode_active()
    elapsed = time.time() - t_start

    lines = [
        "**FormOxy Left-paws cohort -- live progress**",
        f"Elapsed: {fmt_dur(elapsed)}    Cohort: `E:/RSVIDS/Blackbox/2605_FormOxy_LeftPaws/`",
        "",
        f"**Stage progress** (target {TARGET_PER_COND} per condition)",
        f"  mp4 (baseline): `[{bar(n_b, TARGET_PER_COND)}] {n_b}/{TARGET_PER_COND}`",
        f"  mp4 (formalin): `[{bar(n_f, TARGET_PER_COND)}] {n_f}/{TARGET_PER_COND}`",
        f"  DLC raw  (formalin): `[{bar(n_raw, TARGET_PER_COND)}] {n_raw}/{TARGET_PER_COND}`",
        f"  DLC filt (formalin): `[{bar(n_filt, TARGET_PER_COND)}] {n_filt}/{TARGET_PER_COND}`",
        f"  Features (formalin): `[{bar(n_feat, TARGET_PER_COND)}] {n_feat}/{TARGET_PER_COND}`",
        "",
    ]
    if dlc.get("video"):
        cur = dlc.get("cur") or 0
        tot = dlc.get("tot") or 0
        fps = dlc.get("fps") or 0.0
        fps_avg = dlc.get("fps_avg") or 0.0
        fps_min = dlc.get("fps_min") or 0.0
        fps_max = dlc.get("fps_max") or 0.0
        if fps > 1.0 and tot and cur < tot:
            eta = fmt_dur((tot - cur) / fps)
            pct = 100 * cur / max(tot, 1)
        elif tot and cur >= tot:
            eta = "done"; pct = 100.0
        else:
            eta = "..."; pct = 0.0
        lines.append(f"**DLC current**: `{dlc['video']}`")
        lines.append(f"  {pct:5.1f}%  ({cur:,}/{tot:,} frames)  ETA ~{eta}")
        lines.append(f"  fps: latest **{fps:5.1f}**  avg(last 50) {fps_avg:5.1f}  "
                     f"range {fps_min:5.1f}-{fps_max:5.1f} it/s")
    if transcoding:
        lines.append(f"**Transcoding**: `{transcoding}`")
    if portal["ready"]:
        names = sorted(set(portal["ready"]))[:8]
        lines.append(f"Portal ready (not yet ingested): {len(portal['ready'])} "
                     f"({', '.join(names)[:200]})")
    if portal["in_flight"]:
        lines.append(f"Portal in-flight (syncthing): {len(portal['in_flight'])}")
    return "\n".join(lines)


def main() -> int:
    t0 = time.time()
    msg_id = discord_create(build_message(t0))
    print(f"Discord msg id: {msg_id}", flush=True)

    while True:
        if time.time() - t0 > WATCHER_MAX_S:
            print("max watcher time hit", flush=True)
            break
        time.sleep(POLL_S)
        discord_edit(msg_id, build_message(t0))
        n_filt = n_dlc_filt_formalin()
        n_feat = n_features_formalin()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"formalin: dlc_filt={n_filt}/{TARGET_PER_COND}  feats={n_feat}/{TARGET_PER_COND}",
              flush=True)
        if n_filt >= TARGET_PER_COND and n_feat >= TARGET_PER_COND:
            discord_edit(msg_id, build_message(t0) + "\n\n:tada: All formalin sessions complete.")
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
