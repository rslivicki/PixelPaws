# -*- coding: utf-8 -*-
"""Preview/dialog fixes: keep_colors survives plain tkinter; the data-quality
checker no longer compares against an undefined or stale pose table; preview
windows refuse an unreadable video instead of dividing by its zero fps."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tkinter as tk  # noqa: E402


@pytest.fixture(scope="module")
def root():
    try:
        r = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"no display: {e}")
    r.withdraw()
    yield r
    try:
        r.destroy()
    except tk.TclError:
        pass


def test_keep_colors_on_plain_tkinter(root):
    from pose_preview import keep_colors
    lbl = keep_colors(tk.Label, root, background="#111111")
    assert lbl.cget("background") == "#111111"
    assert lbl._tb_no_autostyle is True
    cb = keep_colors(tk.Checkbutton, root, text="x", fg="#1f77b4")
    assert cb.cget("fg") == "#1f77b4"


def test_quality_checker_without_readable_pose(root, tmp_path, monkeypatch):
    cv2 = pytest.importorskip("cv2")
    import dialogs
    monkeypatch.setattr(dialogs.DataQualityChecker, "display_summary", lambda self: None)
    vid = tmp_path / "s1.avi"
    w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (32, 32))
    for _ in range(5):
        w.write(np.zeros((32, 32, 3), np.uint8))
    w.release()
    sessions = [{"session_name": "s1", "pose_path": str(tmp_path / "missing.h5"),
                 "video_path": str(vid)}]
    chk = dialogs.DataQualityChecker(root, sessions)
    try:
        msgs = [str(i) for i in chk.issues]
        assert not any("not defined" in m for m in msgs), msgs
        assert any("Could not read DLC file" in m for m in msgs), msgs
        assert chk.progress["value"] <= 100
    finally:
        chk.window.destroy()


def test_preview_refuses_unreadable_video(root, tmp_path, monkeypatch):
    import dialogs
    errors = []
    monkeypatch.setattr(dialogs.messagebox, "showerror",
                        lambda *a, **k: errors.append(a))
    bogus = tmp_path / "not_a_video.mp4"
    bogus.write_bytes(b"nope")
    before = set(root.winfo_children())
    p = dialogs.SideBySidePreview(root, str(bogus), None, None, "licking", 0.5)
    assert errors and p.cap is None
    assert set(root.winfo_children()) == before   # its Toplevel was destroyed


def test_scoring_review_reads_behavior_named_column(tmp_path):
    """Run Classifiers writes frame, probability and a 0/1 column named after
    the behavior; without a per-frame sheet Review scoring must still read
    the calls from that file (and old "prediction" files)."""
    import pandas as pd
    import scoring_preview as sp
    new = tmp_path / "m1_classifier_Left_licking_predictions.csv"
    pd.DataFrame({"frame": range(6), "probability": [.9, .8, .1, .2, .7, .9],
                  "Left_licking": [1, 1, 0, 0, 1, 1]}).to_csv(new, index=False)
    old = tmp_path / "m1_predictions.csv"
    pd.DataFrame({"frame": range(3), "prediction": [0, 1, 1],
                  "probability": [.1, .8, .9]}).to_csv(old, index=False)
    calls = sp._load_calls(None, {"Left_licking": str(new), "Old": str(old)})
    assert calls["Left_licking"][0].tolist() == [1, 1, 0, 0, 1, 1]
    assert sp._bouts(calls["Left_licking"][0]) == [(0, 2), (4, 6)]
    assert calls["Old"][0].tolist() == [0, 1, 1]
    assert calls["Left_licking"][1] is not None


def test_draw_keypoints_gates_on_likelihood():
    """Shared by Check tracking and the Review scoring Keypoints toggle."""
    pytest.importorskip("cv2")
    from pose_preview import draw_keypoints
    img = np.zeros((40, 40, 3), np.uint8)
    pose = {"good": (np.array([10.0]), np.array([10.0]), np.array([0.9])),
            "weak": (np.array([30.0]), np.array([30.0]), np.array([0.2]))}
    draw_keypoints(img, pose, 0, thr=0.6)
    assert img[10, 10].any()          # above the gate: drawn
    assert not img[30, 30].any()      # below the gate: not drawn
