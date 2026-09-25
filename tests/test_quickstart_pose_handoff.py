# -*- coding: utf-8 -*-
"""Quick Start pose step: a finished tracking thread is not yet delivered results.

The pose dialog hands its results to Quick Start when it reads "all_done" from
its message queue, on a 100 ms timer. Quick Start polls the worker thread. A
thread that ended between two of the dialog's reads used to look like "nothing
was tracked", and the step failed with every pose file on disk."""
import os
import queue
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _DeadWorker:
    def is_alive(self):
        return False


class _Widget:
    """Stands in for the few Tk widgets the dialog touches while draining."""
    def __init__(self, **kw):
        self.kw = dict(kw)

    def config(self, **kw):
        self.kw.update(kw)
    configure = config

    def cget(self, k):
        return self.kw.get(k, "")

    def __getitem__(self, k):
        return self.kw.get(k, 0)

    def get(self, *a):
        return ""

    def index(self, *a):
        return "1.0"


def _dialog(messages, on_complete):
    """A DLCProgressDialog with its real _drain_queue, no window."""
    dlc_run_dialog = pytest.importorskip("dlc_run_dialog")
    d = dlc_run_dialog.DLCProgressDialog.__new__(dlc_run_dialog.DLCProgressDialog)
    d._msg_queue = queue.Queue()
    for m in messages:
        d._msg_queue.put(m)
    d.h5_paths = []
    d.settings = {"videos": ["a.mp4", "b.mp4"]}
    d.on_complete = on_complete
    d._cancelled = False
    d.worker = _DeadWorker()
    d.current_label = _Widget(text="(2/2) b.mp4")
    d.current_bar = _Widget(value=0, maximum=100)
    d.current_stats = _Widget(text="")
    d.overall_bar = _Widget(value=0, maximum=2)
    d.overall_label = _Widget(text="")
    d.cancel_btn = _Widget(text="Cancel")
    d.log_text = _Widget()
    d._append_log = lambda line: None
    d.after = lambda ms, fn: None
    d.destroy = lambda: None
    return d


def _quick_start(tmp_path, asked):
    oneclick_tab = pytest.importorskip("oneclick_tab")
    t = oneclick_tab.OneClickTab.__new__(oneclick_tab.OneClickTab)
    t._running = True
    t._cancel_requested = False
    t._pose_h5s = None
    t._pose_list = asked
    t._plan = ["pose", "features"]
    t._stage_state = {"transcode": "skipped", "pose": "running"}
    t.failed, t.advanced, t.lines = [], [], []
    t._fail_stage = lambda k, why: t.failed.append((k, why))
    t._set_stage_silent = lambda k, v: t._stage_state.__setitem__(k, v)
    t._log_line = t.lines.append
    t.on_project_changed = lambda: None
    t._advance = lambda _: t.advanced.append(True)
    t._abort = lambda why: t.failed.append(("abort", why))
    t.after = lambda ms, fn: None
    t._on_pose_done = types.MethodType(oneclick_tab.OneClickTab._on_pose_done, t)
    return t


def test_results_still_in_the_queue_are_read_before_judging(tmp_path):
    """Thread dead, "all_done" unread: the step must finish, not fail."""
    vids = [{"path": tmp_path / "a.mp4"}, {"path": tmp_path / "b.mp4"}]
    t = _quick_start(tmp_path, vids)
    t._pose_dlg = _dialog(
        [("video_done", ("a.mp4", str(tmp_path / "aDLC_x.h5"))),
         ("video_done", ("b.mp4", str(tmp_path / "bDLC_x.h5"))),
         ("all_done", None)],
        on_complete=t._on_pose_done)
    t._poll_pose()
    assert t.failed == []
    assert t._stage_state["pose"] == "done"
    assert len(t._pose_h5s) == 2 and t.advanced == [True]


def test_pose_files_on_disk_count_even_without_a_callback(tmp_path, monkeypatch):
    """No results list at all, but the pose files exist: not a failure."""
    import dlc_run_dialog
    monkeypatch.setattr(dlc_run_dialog, "active_bundle_scorer", lambda: "DLC_net")
    vids = [{"path": tmp_path / "a.mp4"}, {"path": tmp_path / "b.mp4"}]
    for stem in ("a", "b"):
        (tmp_path / f"{stem}DLC_net_shuffle1.h5").write_bytes(b"x")
    t = _quick_start(tmp_path, vids)
    t._pose_dlg = _dialog([("all_done", None)], on_complete=None)
    t._poll_pose()
    assert t.failed == [] and t._stage_state["pose"] == "done"


def test_nothing_tracked_still_fails(tmp_path, monkeypatch):
    """The check this guards: tracking ran and produced nothing."""
    import dlc_run_dialog
    monkeypatch.setattr(dlc_run_dialog, "active_bundle_scorer", lambda: "DLC_net")
    vids = [{"path": tmp_path / "a.mp4"}, {"path": tmp_path / "b.mp4"}]
    (tmp_path / "aDLC_othernet_shuffle1.h5").write_bytes(b"x")   # another model's file
    t = _quick_start(tmp_path, vids)
    t._pose_dlg = _dialog([("all_done", None)], on_complete=t._on_pose_done)
    t._poll_pose()
    assert t.failed and t.failed[0][0] == "pose"
    assert "0 of 2" in t.failed[0][1]
    assert t.advanced == []


def test_drain_is_safe_to_call_twice():
    got = []
    d = _dialog([("video_done", ("a.mp4", "aDLC.h5")), ("all_done", None)], on_complete=got.append)
    d._drain_queue()
    d._drain_queue()
    assert len(got) == 1 and len(got[0]) == 1
