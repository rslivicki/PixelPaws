# -*- coding: utf-8 -*-
"""The shipped model bundle reaches the per-user bundles folder: on first run, and again
when the shipped manifest changes (a newer network). Other bundles are left alone."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def roots(tmp_path, monkeypatch):
    monkeypatch.setenv("PIXELPAWS_BUNDLES_ROOT", str(tmp_path / "user"))
    src = tmp_path / "shipped" / "pixelpaws_v1"
    src.mkdir(parents=True)
    (src / "snapshot.pt").write_bytes(b"net-v1")
    (src / "manifest.json").write_text(json.dumps({"bundle_id": "pixelpaws_v1", "bundle_version": "1.1.0"}))
    return src, tmp_path / "user" / "bundles" / "pixelpaws_v1"


def test_first_run_copies_the_bundle(roots):
    from dlc_inference import bundle_manager as bm
    src, dest = roots
    assert bm.install_default_bundle(src) == "pixelpaws_v1"
    assert (dest / "snapshot.pt").read_bytes() == b"net-v1"


def test_same_manifest_leaves_the_copy_alone(roots):
    from dlc_inference import bundle_manager as bm
    src, dest = roots
    bm.install_default_bundle(src)
    (dest / "user_note.txt").write_text("mine")
    bm.install_default_bundle(src)
    assert (dest / "user_note.txt").exists()


def test_newer_manifest_replaces_the_copy(roots):
    from dlc_inference import bundle_manager as bm
    src, dest = roots
    bm.install_default_bundle(src)
    other = dest.parent / "my_own_net"
    other.mkdir()
    (other / "manifest.json").write_text(json.dumps({"bundle_id": "my_own_net", "bundle_version": "0.1"}))
    (src / "snapshot.pt").write_bytes(b"net-v2")
    (src / "manifest.json").write_text(json.dumps({"bundle_id": "pixelpaws_v1", "bundle_version": "1.2.0"}))
    bm.install_default_bundle(src)
    assert (dest / "snapshot.pt").read_bytes() == b"net-v2"
    assert json.loads((dest / "manifest.json").read_text())["bundle_version"] == "1.2.0"
    assert (other / "manifest.json").exists()
