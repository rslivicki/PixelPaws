"""Unified Discord pipeline tracker for pp_pipeline.py — any cohort.

Combines:
  - the ROBUST on-disk file-count progress logic from status_tracker.py
    (count <proj>/videos/*shuffle{SHUFFLE}*.h5 for DLC, <proj>/features/*_{HASH}.pkl
     for features — no dependency on stdout), and
  - the single-editable-message bar style from discord_progress.py
    (ONE message edited every 60s; created via webhook ?wait=true then PATCH
     /messages/{id}; message id persisted in <proj>/_discord_msg.json so restarts
     reuse it).

Reads <proj>/_stage.json + <proj>/videos/.ffprog.txt for the live transcode fps.
Stops when <proj>/PIPELINE_DONE exists or after a safety timeout.

All cohort-agnostic constants come from pp_config.

CLI:
  pp_tracker.py --proj DIR --cohort NAME [--nvid N] [--webhook URL]
"""
from __future__ import annotations
import argparse, json, re, subprocess, time, urllib.request
from pathlib import Path
import sys


def _gpu_stats():
    """(temp_C, util_%, power_W) from nvidia-smi, or None on failure."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,power.draw",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=8)
        t, u, p = (x.strip() for x in r.stdout.strip().splitlines()[0].split(","))
        return int(float(t)), int(float(u)), int(float(p))
    except Exception:
        return None


def _cpu_temp():
    """CPU temperature °C — best effort. Tries LibreHardwareMonitor / OpenHardwareMonitor
    WMI, then ACPI thermal zone. Returns None if no source is exposed (the usual case
    on Windows without a hardware-monitor tool running / without admin)."""
    ps = (
        "$ErrorActionPreference='SilentlyContinue';"
        "foreach($ns in 'root/LibreHardwareMonitor','root/OpenHardwareMonitor'){"
        " $s=Get-CimInstance -Namespace $ns -ClassName Sensor 2>$null |"
        "   Where-Object {$_.SensorType -eq 'Temperature' -and $_.Name -match 'CPU'} |"
        "   Sort-Object Value -Descending | Select-Object -First 1;"
        " if($s){ [int]$s.Value; return } }"
        "$z=Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature 2>$null |"
        "  Select-Object -First 1;"
        "if($z){ [int](($z.CurrentTemperature/10)-273.15) }"
    )
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, text=True, timeout=10)
        out = r.stdout.strip()
        return int(out) if out.lstrip("-").isdigit() else None
    except Exception:
        return None

sys.path.insert(0, str(Path(__file__).resolve().parent))      # pp_config beside us
from pp_config import PROCESSING_WEBHOOK, FEATURE_HASH, SHUFFLE  # noqa: E402

PERIOD, MAX_H = 60, 24
# DLC tqdm: e.g. " 20%|## | 44044/219340 [10:24<41:23, 70.59it/s]"
DLC_RE = re.compile(r"(\d+)/(\d+)[^\r\n]*?([\d.]+)it/s")
# feature extractor tqdm: e.g. "... 145326/219214 frames [.. , 536.49frames/s]"
FEAT_RE = re.compile(r"(\d+)/(\d+) frames[^\r\n]*?([\d.]+)frames/s")


def _pid_alive(pid: int) -> bool:
    """True if a process with this pid exists (Windows: tasklist filter; else os.kill 0)."""
    try:
        if sys.platform.startswith("win"):
            r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                               capture_output=True, text=True, timeout=10)
            return str(pid) in r.stdout
        import os
        os.kill(pid, 0)
        return True
    except Exception:
        return True          # can't tell -> keep tracking (safety timeout still applies)


def bar(done, total, w=20):
    frac = max(0.0, min(1.0, done / total if total else 0))
    f = int(round(frac * w))
    return "`" + "█" * f + "░" * (w - f) + f"` {done}/{total} ({frac*100:3.0f}%)"


def fbar(frac, w=18):
    frac = 0.0 if frac is None else max(0.0, min(1.0, frac))
    f = int(round(frac * w))
    return "`" + "█" * f + "░" * (w - f) + f"` {frac*100:3.0f}%"


def hms(s):
    s = int(s)
    return f"{s//3600}h{(s%3600)//60:02d}m" if s >= 3600 else f"{s//60}m{s%60:02d}s"


def ts_to_sec(t):
    try:
        h, m, s = t.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        return 0.0


class Tracker:
    def __init__(self, proj: Path, cohort: str, nvid: int, webhook: str,
                 expected_stems=None, shuffle: int = SHUFFLE,
                 msg_file=None, follow_pid=None):
        self.proj = proj
        self.cohort = cohort
        self.shuffle = int(shuffle)
        self.nvid = max(1, nvid)
        self.webhook = webhook
        self.videos = proj / "videos"
        self.features = proj / "features"
        self.stage_file = proj / "_stage.json"
        self.prog = self.videos / ".ffprog.txt"
        self.done = proj / "PIPELINE_DONE"
        # --msg-file: where this tracker's Discord message id is persisted. Default is the
        # project-wide _discord_msg.json (one live message per project); concurrent runs in
        # one project (e.g. a transcode worker beside a DLC batch) each need their own file
        # or they fight over a single message.
        self.msg_file = Path(msg_file) if msg_file else proj / "_discord_msg.json"
        # --follow-pid: when set, the tracker ends when THAT process exits and ignores the
        # shared PIPELINE_DONE / _stage.json "done" sentinels, which another concurrent
        # pp_pipeline run in the same project may write at any time.
        self.follow_pid = int(follow_pid) if follow_pid else None
        self.start = time.time()
        # Optional set of expected output basenames (sans .mp4). If provided,
        # all on-disk counters are filtered to just these stems — so an
        # incremental run into an existing project doesn't double-count old files.
        self.expected_stems = set(expected_stems) if expected_stems else None

    def _matches(self, name: str) -> bool:
        if not self.expected_stems:
            return True
        # Strict stem boundary: "<stem>.mp4", "<stem>DLC..." (h5), "<stem>_features_..." —
        # a bare prefix test would count S10's outputs for S1.
        for s in self.expected_stems:
            if name == s or name == s + ".mp4" or name.startswith(s + "DLC") or name.startswith(s + "_features_"):
                return True
        return False

    # --- robust on-disk counters ---------------------------------------------
    def count_transcoded(self):
        try:
            return sum(1 for x in self.videos.glob("*.mp4") if self._matches(x.stem))
        except Exception:
            return 0

    def count_h5(self):
        # self.shuffle, not pp_config.SHUFFLE: a run launched with pp_pipeline's
        # --shuffle writes *shuffle<N>*.h5, and counting the default instead would
        # report DLC as 0/N for the whole stage.
        try:
            return sum(1 for x in self.videos.glob(f"*shuffle{self.shuffle}*.h5")
                       if not x.name.endswith("_filtered.h5") and self._matches(x.name))
        except Exception:
            return 0

    def count_feat(self):
        try:
            return sum(1 for x in self.features.glob(f"*_features_{FEATURE_HASH}.pkl")
                       if self._matches(x.name))
        except Exception:
            return 0

    def ffprog(self):
        d = {}
        try:
            for ln in self.prog.read_text().splitlines():
                if "=" in ln:
                    k, v = ln.split("=", 1)
                    d[k.strip()] = v.strip()
        except Exception:
            pass
        return d

    def stage_info(self):
        try:
            return json.loads(self.stage_file.read_text())
        except Exception:
            return {}

    def dlc_rate(self):
        """(cur_frame, total_frames, it_per_s) from the live DLC tqdm capture, or None."""
        try:
            txt = (self.videos / ".dlcprog.txt").read_text(errors="replace")
        except Exception:
            return None
        m = None
        for seg in reversed(re.split(r"[\r\n]", txt)):
            m = DLC_RE.search(seg)
            if m:
                return int(m.group(1)), int(m.group(2)), float(m.group(3))
        return None

    def feat_rate(self):
        """(cur_frame, total_frames, frames_per_s) from the live feature-extractor tqdm, or None."""
        try:
            txt = (self.videos / ".featprog.txt").read_text(errors="replace")
        except Exception:
            return None
        for seg in reversed(re.split(r"[\r\n]", txt)):
            m = FEAT_RE.search(seg)
            if m:
                return int(m.group(1)), int(m.group(2)), float(m.group(3))
        return None

    # --- render the single message body --------------------------------------
    def render(self):
        st = self.stage_info()
        stage = st.get("stage", "?")
        tr, h5, feat = self.count_transcoded(), self.count_h5(), self.count_feat()
        n = self.nvid
        head = (f"**🐭 {self.cohort} — pipeline live** · elapsed {hms(time.time()-self.start)} "
                f"· upd {time.strftime('%H:%M:%S')}\n")
        # hardware line: GPU temp/util/power (always) + CPU temp (best-effort)
        g = _gpu_stats(); ct = _cpu_temp()
        hw = []
        if g:
            hw.append(f"🌡️ GPU **{g[0]}°C** · {g[1]}% · {g[2]}W")
        hw.append(f"CPU {('**'+str(ct)+'°C**') if ct is not None else 'temp n/a'}")
        lines = [head, " · ".join(hw)]

        # Transcode line (+ live fps when actively encoding)
        tline = f"Transcode  {bar(min(tr, n), n)}"
        if stage == "transcode":
            d = self.ffprog()
            fps, spd, ot = d.get("fps", "…"), d.get("speed", "…"), d.get("out_time", "")[:8]
            dur = st.get("duration") or 0
            cur = ts_to_sec(ot) / dur if (dur and ot) else 0.0
            tline += (f"\n  ▸ `{st.get('video','')}` {fbar(cur)} · 🎞️ **{fps} fps** · {spd} · {ot}")
        lines.append(tline)

        # DLC + features lines (robust counts; DLC also shows live it/s + ETA)
        dline = f"DLC pose   {bar(min(h5, n), n)}"
        if stage == "dlc":
            dline += f"\n  ▸ tracking `{st.get('video','')}` ({st.get('idx','?')}/{st.get('total','?')})"
            rate = self.dlc_rate()
            if rate:
                cf, tf, ips = rate
                # ETA over remaining frames of this video + the videos still queued, at current it/s
                rem_vids = max(0, (st.get("total", n) or n) - (st.get("idx", 0) or 0))
                rem = (tf - cf) + rem_vids * tf
                dline += (f"\n  {fbar(cf / tf if tf else 0)} **{cf:,}**/{tf:,} frames "
                          f"({tf - cf:,} left)"
                          + (f" · 🚀 **{ips:.0f} it/s**" if ips else "")
                          + (f" · ETA {hms(rem / ips)}" if ips > 0 else ""))
        lines.append(dline)

        fline = f"Features   {bar(min(feat, n), n)}"
        if stage == "features":
            fline += f"\n  ▸ extracting `{st.get('video','')}` ({st.get('idx','?')}/{st.get('total','?')})"
            fr = self.feat_rate()
            if fr:
                cf, tf, fps = fr
                rem_vids = max(0, (st.get("total", n) or n) - (st.get("idx", 0) or 0))
                rem = (tf - cf) + rem_vids * tf
                fline += (f"\n  {fbar(cf / tf if tf else 0)} **{cf:,}**/{tf:,} frames"
                          + (f" · 🎞️ **{fps:.0f} fps**" if fps else "")
                          + (f" · ETA {hms(rem / fps)}" if fps > 0 else ""))
        lines.append(fline)

        if self.follow_pid:
            finished = not _pid_alive(self.follow_pid)
            if finished:
                lines.append("⏹ **Run finished** (followed process exited).")
        else:
            finished = self.done.exists() or stage == "done"
            if finished:
                lines.append("✅ **Pipeline done** — transcode + DLC + features complete.")
        return "\n".join(lines), finished

    # --- webhook single-message edit -----------------------------------------
    def _post(self, content):
        req = urllib.request.Request(
            self.webhook + "?wait=true", data=json.dumps({"content": content[:1990]}).encode(),
            method="POST", headers={"Content-Type": "application/json", "User-Agent": "PP-Track/1.0"})
        return json.loads(urllib.request.urlopen(req, timeout=20).read().decode())["id"]

    def _patch(self, mid, content):
        req = urllib.request.Request(
            f"{self.webhook}/messages/{mid}", data=json.dumps({"content": content[:1990]}).encode(),
            method="PATCH", headers={"Content-Type": "application/json", "User-Agent": "PP-Track/1.0"})
        try:
            urllib.request.urlopen(req, timeout=20).read()
        except Exception as e:
            print("patch err:", e, flush=True)

    def message_id(self):
        if self.msg_file.exists():
            try:
                mid = json.loads(self.msg_file.read_text()).get("id")
                if mid:
                    return mid
            except Exception:
                pass
        mid = self._post(f"**🐭 {self.cohort} — tracker starting…**")
        try:
            self.msg_file.write_text(json.dumps({"id": mid}))
        except Exception:
            pass
        return mid

    def loop(self):
        mid = self.message_id()
        print("tracker msg id", mid, flush=True)
        while True:
            c, done = self.render()
            self._patch(mid, c)
            if done:
                break
            if time.time() - self.start > MAX_H * 3600:
                self._patch(mid, c + "\n_(tracker timed out)_")
                break
            time.sleep(PERIOD)
        return 0


def _derive_nvid(proj: Path, portal, sources):
    """Expected video count: from sources.json, else by counting portal mp4s."""
    if sources:
        try:
            return len(json.loads(Path(sources).read_text(encoding="utf-8")))
        except Exception:
            pass
    if portal:
        try:
            return len(list(Path(portal).rglob("*.mp4")))
        except Exception:
            pass
    # last resort: count what's already transcoded
    try:
        return len(list((proj / "videos").glob("*.mp4"))) or 1
    except Exception:
        return 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="Unified Discord tracker for pp_pipeline.py.")
    ap.add_argument("--proj", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--nvid", type=int, help="expected video count (else derive from sources/portal)")
    ap.add_argument("--sources", help="sources.json to derive --nvid from")
    ap.add_argument("--portal", help="portal dir to derive --nvid from")
    ap.add_argument("--webhook", default=PROCESSING_WEBHOOK)
    ap.add_argument("--shuffle", type=int, default=SHUFFLE,
                    help="DLC shuffle whose *shuffle<N>*.h5 count as DLC progress. Must match "
                         "the --shuffle the run was launched with, else DLC reads 0/N throughout.")
    ap.add_argument("--msg-file", dest="msg_file",
                    help="JSON file holding this tracker's Discord message id (default "
                         "<proj>/_discord_msg.json). Give concurrent runs in one project "
                         "separate files so each gets its own live message.")
    ap.add_argument("--follow-pid", dest="follow_pid", type=int,
                    help="end when this process exits instead of on PIPELINE_DONE (use when "
                         "another pp_pipeline run in the same project may write that sentinel).")
    a = ap.parse_args(argv)
    proj = Path(a.proj)
    nvid = a.nvid if a.nvid else _derive_nvid(proj, a.portal, a.sources)
    # If sources is given, derive expected output basenames so the tracker
    # only counts THIS batch's files (works for incremental runs).
    expected_stems = None
    if a.sources:
        try:
            sj = json.loads(Path(a.sources).read_text(encoding="utf-8"))
            expected_stems = {Path(s["out"]).stem for s in sj if "out" in s}
        except Exception:
            pass
    return Tracker(proj, a.cohort, nvid, a.webhook, expected_stems=expected_stems,
                   shuffle=a.shuffle, msg_file=a.msg_file, follow_pid=a.follow_pid).loop()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    raise SystemExit(main())
