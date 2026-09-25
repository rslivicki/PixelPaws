"""PawCapture: CameraThread's FFmpeg-crash auto-resume, without cameras.

_open_ffmpeg_capture is replaced with a fake that launches a tiny Python
process standing in for FFmpeg: it writes N raw frames to stdout, then exits
with the access-violation code, like the 2026-09-15 CAM 1 crash. The thread's
real run loop, respawn logic and part-file naming are what's under test.
Ported from the camsync rig's test_crash_autoresume.py (2026-09-15).

Needs PyQt5 (the PawCapture build environment); skipped elsewhere:
    <pawcapture venv>\\python -m pytest tests/pawcapture -q
"""
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("PyQt5")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "pawcapture"))
import pawcapture_app as cp  # noqa: E402

W, H = 4, 4
FB = W * H * 3
PY = sys.executable


class FakeCtrl:
    def __init__(self, *a): self.opens = 0
    def open(self): self.opens += 1; return None
    def close(self): pass
    def is_open(self): return False          # skip COM property sync/probe paths
    def set_via_cv2(self, *a): return True
    def probe_ranges(self): return {}


@pytest.fixture(autouse=True)
def _fakes(monkeypatch):
    monkeypatch.setattr(cp, "_DshowCameraControl", FakeCtrl)
    monkeypatch.setattr(cp, "_FFMPEG_OK", True)
    # scenarios change the class-level guard; restore it afterwards
    monkeypatch.setattr(cp.CameraThread, "RESUME_MIN_UPTIME_S",
                        cp.CameraThread.RESUME_MIN_UPTIME_S)


def fake_proc(frames, exit_code=0xC0000005 & 0x7FFFFFFF, hang=False):
    # hang=True: keep producing frames until killed (a healthy FFmpeg)
    body = ("import sys,time\n"
            f"fb=b'\\x00'*{FB}\n"
            f"n={frames}\n"
            "i=0\n"
            f"while {'True' if hang else 'i<n'}:\n"
            "    sys.stdout.buffer.write(fb); sys.stdout.buffer.flush(); i+=1; time.sleep(0.002)\n"
            f"sys.exit({exit_code})\n")
    return subprocess.Popen([PY, "-c", body], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def make_thread(plan, record_path):
    """plan: list of dicts {frames, hang} consumed per FFmpeg open."""
    t = cp.CameraThread(0, {}, {}, width=W, height=H, fps=60, device_name="fake")
    t.RESUME_SETTLE_S = 0.0
    opened, resumed, errors = [], [], []

    def _open():
        if not plan:
            return None, 0, "no more fake procs"
        step = plan.pop(0)
        opened.append(dict(t._record_args) if t._record_args else None)
        t._eff_w, t._eff_h = W, H
        return fake_proc(step.get("frames", 5), hang=step.get("hang", False)), FB, ""

    t._open_ffmpeg_capture = _open
    t._graceful_stop_ffmpeg = lambda p: (p.kill() if p and p.poll() is None else None)
    # DirectConnection: there is no Qt event loop, so queued signals would
    # never be delivered.
    t.recording_resumed.connect(lambda path, part: resumed.append((path, part)),
                                cp.Qt.DirectConnection)
    t.camera_error.connect(lambda m: errors.append(m), cp.Qt.DirectConnection)
    if record_path:
        t._record_args = {"path": Path(record_path), "width": W, "height": H,
                          "fps": 60, "bitrate": 8.0, "codec": "h264_qsv",
                          "metadata": {"mm_per_pixel": "0.19"}}
    return t, opened, resumed, errors


def run_until(t, cond, timeout=15):
    th = threading.Thread(target=t._run_inner, daemon=True)
    th.start()
    end = time.time() + timeout
    while time.time() < end and not cond() and th.is_alive():
        time.sleep(0.02)
    t.running = False
    th.join(5)
    return not th.is_alive()


def test_crash_mid_recording_resumes_into_pt2(tmp_path):
    base = tmp_path / "post-drug_2609_BE22025_Form_M_S5.mp4"
    base.write_bytes(b"x")                       # the crashed first part
    cp.CameraThread.RESUME_MIN_UPTIME_S = 0.0    # every run counts as long-lived
    t, opened, resumed, errors = make_thread([{"frames": 5}, {"hang": True}], base)
    assert run_until(t, lambda: len(resumed) >= 1), "thread did not stop"
    exp = tmp_path / "post-drug_2609_BE22025_Form_M_S5_pt2.mp4"
    assert resumed == [(str(exp), 2)]
    ra = opened[1]
    assert Path(ra["path"]) == exp
    assert ra["metadata"]["mm_per_pixel"] == "0.19"
    assert ra["metadata"]["pawcapture_part"] == "2"
    assert ra["metadata"]["pawcapture_part_of"] == base.name
    assert not errors


def test_repeated_crashes_skip_existing_parts(tmp_path):
    base = tmp_path / "post-drug_2609_BE22025_Form_M_S5.mp4"
    base.write_bytes(b"x")
    (tmp_path / "post-drug_2609_BE22025_Form_M_S5_pt2.mp4").write_bytes(b"x")
    (tmp_path / "post-drug_2609_BE22025_Form_M_S5_pt3.mp4").write_bytes(b"x")
    cp.CameraThread.RESUME_MIN_UPTIME_S = 0.0
    t, opened, resumed, errors = make_thread(
        [{"frames": 5}, {"frames": 5}, {"hang": True}], base)
    run_until(t, lambda: len(resumed) >= 2)
    assert [(Path(p).name, n) for p, n in resumed] == [
        ("post-drug_2609_BE22025_Form_M_S5_pt4.mp4", 4),
        ("post-drug_2609_BE22025_Form_M_S5_pt5.mp4", 5)]


def test_crash_loop_gives_up_and_reports(tmp_path):
    cp.CameraThread.RESUME_MIN_UPTIME_S = 60.0   # every run is a quick failure
    t, opened, resumed, errors = make_thread([{"frames": 2}] * 10,
                                             tmp_path / "loop_S1.mp4")
    assert run_until(t, lambda: bool(errors))
    assert len(opened) == cp.CameraThread.RESUME_MAX_QUICK_FAILS
    assert "could not be resumed" in errors[0]


def test_preview_only_crash_keeps_old_behavior(tmp_path):
    cp.CameraThread.RESUME_MIN_UPTIME_S = 0.0
    t, opened, resumed, errors = make_thread([{"frames": 5}, {"hang": True}], None)
    run_until(t, lambda: bool(errors))
    assert not resumed and len(opened) == 1
    assert "FFmpeg capture stopped" in errors[0]


def test_stop_racing_the_crash_does_not_resume(tmp_path):
    cp.CameraThread.RESUME_MIN_UPTIME_S = 0.0
    t, opened, resumed, errors = make_thread([{"frames": 5}, {"hang": True}],
                                             tmp_path / "race_S1.mp4")
    t.stop_recording()          # _record_args=None, respawn pending
    run_until(t, lambda: len(opened) >= 2)
    assert not resumed
    assert len(opened) >= 2 and opened[1] is None     # respawned preview-only


def test_next_part_path_never_overwrites(tmp_path):
    base = tmp_path / "a.mp4"
    assert cp.CameraThread._next_part_path(base) == (tmp_path / "a_pt2.mp4", 2)
    (tmp_path / "a_pt2.mp4").write_bytes(b"x")
    assert cp.CameraThread._next_part_path(base) == (tmp_path / "a_pt3.mp4", 3)
