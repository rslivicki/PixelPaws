"""
Live Discord progress tracker for the rim addons-chain runs.

Edits a single Discord message every 60 s with: cohort ingestion totals
(mp4 / raw .h5 / filtered .h5 / features pickle / predictions CSV),
current DLC tqdm progress, and pre-transcode portal queue. Exits when
all 30 sessions (15 baseline + 15 postrim) reach predictions stage, OR
after a safety cap.

Run from regular PixelPaws env:
  PYTHONIOENCODING=utf-8 nohup py -X utf8 -u \
      scripts/research/rim_addons_progress_watcher.py > /tmp/rim_watcher.log 2>&1 &
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

COHORT = Path(r"E:\RSVIDS\Blackbox\260515_Rim_DoseResp")
VIDS = COHORT / "videos"
FEATS = COHORT / "features"
RESULTS = COHORT / "results"

PORTAL_2605 = Path(r"E:\RSVIDS\Video_transfer_portal\2605_Rimonabant")

# DLC chain logs (both first chain + follow-up write to these)
CHAIN_LOGS = [
    Path("/tmp/rim_addons.log"),
    Path("/tmp/rim_followup.log"),
    Path("/tmp/rim_pretrans.log"),
]

WEBHOOK = (
    ""
)

TOTAL_BASELINE = 15
TOTAL_POSTRIM = 15
TOTAL_SESSIONS = TOTAL_BASELINE + TOTAL_POSTRIM
POLL_S = 60
BAR_W = 24
WATCHER_MAX_S = 12 * 3600

_TQDM = re.compile(r"(\d+)\s*/\s*(\d+)\s*\[[^\]]*?,\s*(\d+\.?\d*)\s*it/s")
_VIDEO = re.compile(r"Starting to analyze\s+(.+?\.mp4)", re.IGNORECASE)


def _post(url, payload, method="POST"):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method=method,
        headers={"Content-Type": "application/json",
                 "User-Agent": "PixelPaws-Watcher/2.0"},
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


def count_files(folder: Path, pattern_fn) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir()
               if p.is_file() and pattern_fn(p.name))


def count_baseline_mp4() -> int:
    return count_files(VIDS, lambda n: n.startswith("fem_baseline_Rim") and n.endswith(".mp4"))


def count_postrim_mp4() -> int:
    return count_files(VIDS, lambda n: n.startswith("fem_postrim_Rim") and n.endswith(".mp4"))


def count_raw_h5() -> int:
    return count_files(VIDS, lambda n: "shuffle9" in n and n.endswith(".h5")
                                       and not n.endswith("_filtered.h5"))


def count_filtered_h5() -> int:
    return count_files(VIDS, lambda n: "shuffle9" in n and n.endswith("_filtered.h5"))


def count_features() -> int:
    return count_files(FEATS, lambda n: "_features_8aed1c22.pkl" in n)


def count_predictions() -> int:
    return count_files(RESULTS, lambda n: n.endswith("_predictions.csv"))


def count_portal_pending() -> dict:
    """Return dict with counts of portal mp4s waiting (ready and in-flight)."""
    out = {"ready": [], "in_flight": []}
    if not PORTAL_2605.is_dir():
        return out
    for mp4 in PORTAL_2605.rglob("*.mp4"):
        if mp4.name.startswith("~syncthing~") or mp4.name.endswith(".tmp"):
            out["in_flight"].append(mp4.name)
        else:
            out["ready"].append(mp4.name)
    for tmp in PORTAL_2605.rglob("~syncthing~*"):
        out["in_flight"].append(tmp.name)
    return out


def parse_dlc_progress() -> dict:
    """Return latest current_video, cur_frames, total_frames, fps, fps_avg
    (mean over last 50 tqdm samples) by reading chain logs."""
    out = {"current_video": None, "cur_frames": None,
           "total_frames": None, "fps": None, "fps_avg": None,
           "fps_min": None, "fps_max": None}
    combined = []
    for log in CHAIN_LOGS:
        if log.is_file():
            try:
                t = log.read_text(encoding="utf-8", errors="replace")
                combined.append((log.stat().st_mtime, t))
            except OSError:
                pass
    if not combined:
        return out
    combined.sort()
    text = "".join(t for _, t in combined).replace("\r", "\n")
    v_matches = list(_VIDEO.finditer(text))
    if v_matches:
        last = v_matches[-1]
        out["current_video"] = Path(last.group(1).strip().strip('"')).name
        scan = text[last.end():]
    else:
        scan = text
    tq_matches = list(_TQDM.finditer(scan))
    if tq_matches:
        # Instantaneous = latest tqdm sample
        last = tq_matches[-1]
        try:
            out["cur_frames"] = int(last.group(1))
            out["total_frames"] = int(last.group(2))
            out["fps"] = float(last.group(3))
        except ValueError:
            pass
        # Rolling average + min/max over the last 50 tqdm samples
        recent = tq_matches[-50:]
        fpss = []
        for m in recent:
            try:
                fpss.append(float(m.group(3)))
            except ValueError:
                pass
        if fpss:
            out["fps_avg"] = sum(fpss) / len(fpss)
            out["fps_min"] = min(fpss)
            out["fps_max"] = max(fpss)
    return out


def find_pretranscode_active() -> str | None:
    """Return current pretranscode target name, if any."""
    log = Path("/tmp/rim_pretrans.log")
    if not log.is_file():
        return None
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    # Find last `transcode -> ...` line that wasn't followed by `done`
    lines = text.splitlines()
    last_transcode_idx = None
    for i, line in enumerate(lines):
        if "transcode ->" in line:
            last_transcode_idx = i
    if last_transcode_idx is None:
        return None
    # Look for a "done in" line after it
    for line in lines[last_transcode_idx + 1:]:
        if "done in" in line:
            return None
    # Still active
    m = re.search(r"transcode\s*->\s*(\S+)", lines[last_transcode_idx])
    return m.group(1) if m else None


def build_message(t_start: float) -> str:
    n_b = count_baseline_mp4()
    n_p = count_postrim_mp4()
    n_raw = count_raw_h5()
    n_filt = count_filtered_h5()
    n_feat = count_features()
    n_pred = count_predictions()
    portal = count_portal_pending()

    dlc = parse_dlc_progress()
    pre = find_pretranscode_active()

    elapsed = time.time() - t_start

    cur_v = dlc.get("current_video") or "(idle)"
    fps = dlc.get("fps") or 0.0
    tot_f = dlc.get("total_frames") or 0
    cur_f = dlc.get("cur_frames") or 0
    if fps > 1.0 and tot_f and cur_f < tot_f:
        v_eta = fmt_dur((tot_f - cur_f) / fps)
        v_pct = 100 * cur_f / max(tot_f, 1)
    elif tot_f and cur_f >= tot_f:
        v_eta = "done"
        v_pct = 100.0
    else:
        v_eta = "..."
        v_pct = 0.0

    lines = [
        "**Rim cohort addons -- live progress**",
        f"Elapsed: {fmt_dur(elapsed)}    Cohort: `E:/RSVIDS/Blackbox/260515_Rim_DoseResp/`",
        "",
        f"**Stage progress** (target {TOTAL_SESSIONS} sessions = {TOTAL_BASELINE} baseline + {TOTAL_POSTRIM} postrim)",
        f"  mp4 (baseline): `[{bar(n_b, TOTAL_BASELINE)}] {n_b}/{TOTAL_BASELINE}`",
        f"  mp4 (postrim):  `[{bar(n_p, TOTAL_POSTRIM)}] {n_p}/{TOTAL_POSTRIM}`",
        f"  DLC raw:        `[{bar(n_raw, TOTAL_SESSIONS)}] {n_raw}/{TOTAL_SESSIONS}`",
        f"  DLC filtered:   `[{bar(n_filt, TOTAL_SESSIONS)}] {n_filt}/{TOTAL_SESSIONS}`",
        f"  Features:       `[{bar(n_feat, TOTAL_SESSIONS)}] {n_feat}/{TOTAL_SESSIONS}`",
        f"  Predictions:    `[{bar(n_pred, TOTAL_SESSIONS)}] {n_pred}/{TOTAL_SESSIONS}`",
        "",
    ]
    if cur_v != "(idle)":
        lines.append(f"**DLC current**: `{cur_v}`")
        lines.append(f"  {v_pct:5.1f}%  ({cur_f:,}/{tot_f:,} frames)  ETA ~{v_eta}")
        fps_avg = dlc.get("fps_avg") or 0.0
        fps_min = dlc.get("fps_min") or 0.0
        fps_max = dlc.get("fps_max") or 0.0
        lines.append(f"  fps: latest **{fps:5.1f}**  avg(last 50) {fps_avg:5.1f}  "
                     f"range {fps_min:5.1f}-{fps_max:5.1f} it/s")
    if pre:
        lines.append(f"**Pre-transcode active**: `{pre}`")
    if portal["ready"] or portal["in_flight"]:
        if portal["ready"]:
            lines.append(f"Portal ready (not yet ingested): {len(portal['ready'])} "
                         f"({', '.join(sorted(set(portal['ready'])))[:200]})")
        if portal["in_flight"]:
            lines.append(f"Portal in-flight (syncthing): {len(portal['in_flight'])}")
    return "\n".join(lines)


def main() -> int:
    t_start = time.time()
    msg_id = discord_create(build_message(t_start))
    print(f"Discord msg id: {msg_id}", flush=True)

    while True:
        if time.time() - t_start > WATCHER_MAX_S:
            print("max watcher time hit, exiting", flush=True)
            break
        time.sleep(POLL_S)
        msg = build_message(t_start)
        discord_edit(msg_id, msg)
        n_pred = count_predictions()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] preds={n_pred}/{TOTAL_SESSIONS}", flush=True)
        if n_pred >= TOTAL_SESSIONS:
            print("all sessions predicted, exiting", flush=True)
            discord_edit(msg_id, msg + "\n\n:tada: All sessions complete.")
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
