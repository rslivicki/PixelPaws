"""Per-user bundle storage under %LOCALAPPDATA%\\PixelPaws\\bundles\\.

Bundles are self-contained directories: each contains its own model.onnx,
manifest.json, classifiers/, and postprocess.json. The active bundle id is
recorded in active_bundle.txt next to the bundles dir.
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


# Allow tests / dev to override via env var.
_ENV_OVERRIDE = "PIXELPAWS_BUNDLES_ROOT"


@dataclass
class BundleSummary:
    bundle_id: str
    display_name: str
    bundle_version: str
    release_date: str
    path: Path
    has_model: bool


def bundles_root() -> Path:
    """Resolve the per-user bundles directory; create it if missing."""
    override = os.environ.get(_ENV_OVERRIDE)
    if override:
        base = Path(override)
    elif os.name == "nt":
        local = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/AppData/Local")
        base = Path(local) / "PixelPaws"
    else:
        base = Path(os.path.expanduser("~/.local/share/PixelPaws"))
    (base / "bundles").mkdir(parents=True, exist_ok=True)
    return base / "bundles"


def _active_pointer_path() -> Path:
    return bundles_root().parent / "active_bundle.txt"


def list_bundles() -> List[BundleSummary]:
    out: List[BundleSummary] = []
    root = bundles_root()
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        manifest = child / "manifest.json"
        if not manifest.is_file():
            continue
        try:
            m = json.loads(manifest.read_text())
        except json.JSONDecodeError:
            continue
        out.append(BundleSummary(
            bundle_id=m.get("bundle_id", child.name),
            display_name=m.get("display_name", child.name),
            bundle_version=m.get("bundle_version", "?"),
            release_date=m.get("release_date", "?"),
            path=child,
            has_model=(child / "model.onnx").is_file(),
        ))
    return out


def active_bundle_id() -> Optional[str]:
    ptr = _active_pointer_path()
    if not ptr.is_file():
        # If exactly one bundle is installed, treat it as active by default.
        bundles = list_bundles()
        if len(bundles) == 1:
            return bundles[0].bundle_id
        return None
    try:
        return ptr.read_text().strip() or None
    except OSError:
        return None


def set_active_bundle(bundle_id: str) -> None:
    if not (bundles_root() / bundle_id).is_dir():
        raise FileNotFoundError(f"No bundle named {bundle_id!r} installed.")
    _active_pointer_path().write_text(bundle_id)


def install_default_bundle(source_dir: Path) -> str:
    """Copy a bundle dir from `source_dir` into bundles_root() if not already present.

    Used on first-run to populate %LOCALAPPDATA% from the install-time
    default_bundle/ folder shipped with the app.
    """
    src = Path(source_dir)
    if not src.is_dir():
        raise FileNotFoundError(source_dir)
    manifest = src / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest.json missing in {source_dir}")
    bundle_id = json.loads(manifest.read_text())["bundle_id"]
    dest = bundles_root() / bundle_id
    if dest.is_dir():
        return bundle_id
    shutil.copytree(src, dest)
    return bundle_id


def import_bundle_zip(zip_path: Path) -> str:
    """Extract a bundle .zip into bundles_root(). The zip's top-level dir
    must contain a manifest.json. Returns the installed bundle id.
    """
    zp = Path(zip_path)
    with zipfile.ZipFile(zp) as zf:
        # Find a manifest.json near the top.
        manifest_member = None
        for name in zf.namelist():
            if name.endswith("/manifest.json") and name.count("/") == 1:
                manifest_member = name
                break
            if name == "manifest.json":
                manifest_member = name
                break
        if manifest_member is None:
            raise ValueError(f"No top-level manifest.json found in {zip_path}")
        with zf.open(manifest_member) as f:
            m = json.loads(f.read().decode("utf-8"))
        bundle_id = m["bundle_id"]
        dest = bundles_root() / bundle_id
        if dest.is_dir():
            raise FileExistsError(
                f"Bundle {bundle_id!r} already installed at {dest}. "
                "Delete it first or use a different version."
            )
        dest.mkdir(parents=True)
        zf.extractall(dest.parent if manifest_member.count("/") == 1 else dest)
    return bundle_id
