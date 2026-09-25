# -*- coding: utf-8 -*-
"""frame_rate_normalize: S1-vs-S10 pose resolution, per-video content fps,
calibration tags surviving the re-encode, and per-video failure isolation."""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

cv2 = pytest.importorskip("cv2")
pytest.importorskip("tables")

import frame_rate_normalize as frn  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH")

N_UNIQUE = 40          # unique frames; each is held for 2 ticks at 60 fps


def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    return p


def test_find_session_assets_does_not_take_a_longer_stems_h5(tmp_path):
    vids = tmp_path / "videos"
    v1 = _touch(vids / "S1.mp4")
    _touch(vids / "S10.mp4")
    _touch(vids / "S10DLC_resnet50_netshuffle4_100filtered.h5")
    # S1 has no pose file of its own: must not pick up S10's
    assert frn._find_session_assets(v1, tmp_path)["dlc_h5"] is None
    own = _touch(vids / "S1DLC_resnet50_netshuffle4_100.h5")
    assert frn._find_session_assets(v1, tmp_path)["dlc_h5"] == own
    own_f = _touch(vids / "S1DLC_resnet50_netshuffle4_100filtered.h5")
    assert frn._find_session_assets(v1, tmp_path)["dlc_h5"] == own_f


def test_content_fps():
    assert frn.content_fps(60.0, 1 / 3) == 40.0
    assert frn.content_fps(60.0, 0.5) == 30.0
    assert frn.content_fps(30.0, 0.0) == 30.0
    assert frn.content_fps(0.0, 0.5) is None


def _probe(path, entries):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", entries,
                          "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def _make_session(project, stem, n_h5_rows=None, mm_per_pixel="0.25"):
    """A 60 fps video of N_UNIQUE noise frames each held twice (true rate
    30), carrying a PawCapture calibration tag, plus a pose h5 and labels."""
    vids = project / "videos"
    vids.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(stem)) % (2 ** 32))
    raw = vids / f"{stem}_raw.avi"
    w = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"MJPG"), 60.0, (64, 64))
    for _ in range(N_UNIQUE):
        f = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        w.write(f)
        w.write(f)
    w.release()
    video = vids / f"{stem}.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw),
                    "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p",
                    "-metadata", f"mm_per_pixel={mm_per_pixel}",
                    "-metadata", "pixelpaws_calibrated=1",
                    "-movflags", "use_metadata_tags", str(video)], check=True)
    raw.unlink()

    n_rows = 2 * N_UNIQUE if n_h5_rows is None else n_h5_rows
    cols = pd.MultiIndex.from_product([["net"], ["snout"], ["x", "y", "likelihood"]],
                                      names=["scorer", "bodyparts", "coords"])
    h5 = vids / f"{stem}DLC_resnet50_netshuffle1_100.h5"
    pd.DataFrame(np.ones((n_rows, 3)), columns=cols).to_hdf(
        str(h5), key="df_with_missing", mode="w")
    lab_dir = project / "behavior_labels"
    lab_dir.mkdir(exist_ok=True)
    labels = lab_dir / f"{stem}_labels.csv"
    lab = np.zeros(2 * N_UNIQUE, dtype=int)
    lab[10:20] = 1
    pd.DataFrame({"Licking": lab}).to_csv(labels, index=False)
    return video, h5, labels


@needs_ffmpeg
def test_wet_run_keeps_duration_and_calibration_tags(tmp_path):
    video, h5, labels = _make_session(tmp_path, "m1")
    dur_in = float(_probe(video, "format=duration")["format"]["duration"])

    rep = frn.normalize_project(str(tmp_path), dry_run=False,
                                invalidate_features=False, progress=lambda m: None)
    (plan,) = rep.plans
    assert plan.status == "applied", plan.error
    assert plan.output_fps == 30.0 == plan.content_fps
    assert plan.fps_warning is None
    assert plan.video_metadata.startswith("ffmpeg remux")

    cap = cv2.VideoCapture(str(video))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == N_UNIQUE
    assert round(cap.get(cv2.CAP_PROP_FPS)) == 30
    cap.release()
    fmt = _probe(video, "format=duration:format_tags")["format"]
    assert abs(float(fmt["duration"]) - dur_in) < 0.1
    assert fmt["tags"]["mm_per_pixel"] == "0.25"
    assert fmt["tags"]["pixelpaws_calibrated"] == "1"

    assert len(pd.read_hdf(h5)) == N_UNIQUE
    assert len(pd.read_csv(labels)) == N_UNIQUE
    backup = tmp_path / os.path.basename(rep.backup_dir)
    assert {p.name for p in backup.iterdir()} >= {video.name, h5.name, labels.name}
    assert not list(tmp_path.glob("_fps_normalize_stage_*"))
    assert list((tmp_path / "diagnostics").glob("fps_normalize_applied_*.json"))


@needs_ffmpeg
def test_explicit_fps_off_content_rate_is_flagged(tmp_path):
    _make_session(tmp_path, "m1")
    rep = frn.normalize_project(str(tmp_path), target_fps=20.0, dry_run=True,
                                progress=lambda m: None)
    (plan,) = rep.plans
    assert plan.output_fps == 20.0
    assert plan.fps_warning and "1.50x" in plan.fps_warning


@needs_ffmpeg
def test_failed_video_is_left_untouched_and_reported(tmp_path):
    # m1's pose file is shorter than its video, so the h5 trim raises
    bad_video, bad_h5, bad_labels = _make_session(tmp_path, "m1", n_h5_rows=10)
    good_video, good_h5, _ = _make_session(tmp_path, "m2")
    before = {p: p.read_bytes() for p in (bad_video, bad_h5, bad_labels)}

    rep = frn.normalize_project(str(tmp_path), dry_run=False,
                                invalidate_features=False, progress=lambda m: None)
    by = {os.path.basename(p.video): p for p in rep.plans}
    assert by["m1.mp4"].status == "failed" and "ValueError" in by["m1.mp4"].error
    assert by["m2.mp4"].status == "applied"
    for p, data in before.items():
        assert p.read_bytes() == data, f"{p.name} changed despite the failure"
    assert len(pd.read_hdf(good_h5)) == N_UNIQUE
    assert not list(tmp_path.glob("_fps_normalize_stage_*"))

    (rp,) = (tmp_path / "diagnostics").glob("fps_normalize_applied_*.json")
    stat = {os.path.basename(p["video"]): p["status"]
            for p in json.loads(rp.read_text(encoding="utf-8"))["plans"]}
    assert stat == {"m1.mp4": "failed", "m2.mp4": "applied"}
