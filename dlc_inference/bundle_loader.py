"""Load a bundle from disk, verify integrity, expose a typed view of its contents."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .bundle_manager import bundles_root


@dataclass
class ClassifierEntry:
    name: str
    display_name: str
    path: Path
    sha256: Optional[str]
    trained_on_feature_hash: Optional[str]
    trained_on_dlc_model: Optional[str]
    best_thresh: float
    min_bout: int
    min_after_bout: int
    max_gap: int


@dataclass
class DlcInfo:
    model_name: str
    shuffle: int
    snapshot: str
    snapshot_epoch: int
    net_type: str
    backbone: str
    snapshot_path: Path
    snapshot_sha256: Optional[str]
    pytorch_config_path: Path
    pytorch_config_sha256: Optional[str]
    keypoints: List[str]
    h5_scorer_string: str
    deeplabcut_version_min: str


@dataclass
class Bundle:
    bundle_id: str
    display_name: str
    bundle_version: str
    release_date: str
    pixelpaws_min_version: str
    inference_backend: str
    path: Path
    dlc: DlcInfo
    classifiers: List[ClassifierEntry]
    feature_config: Dict
    group_analysis_defaults: Dict
    manifest_raw: Dict = field(repr=False)

    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.json"

    def classifier_by_name(self, name: str) -> Optional[ClassifierEntry]:
        for c in self.classifiers:
            if c.name == name:
                return c
        return None


class BundleIntegrityError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bundle(bundle_id: str, verify_sha: bool = False) -> Bundle:
    """Load and parse a bundle. If `verify_sha`, hash large files and compare against manifest."""
    root = bundles_root()
    bdir = root / bundle_id
    if not bdir.is_dir():
        raise FileNotFoundError(f"Bundle not installed: {bundle_id}")
    manifest_path = bdir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest missing: {manifest_path}")
    m = json.loads(manifest_path.read_text())

    backend = m.get("inference_backend", "dlc_pytorch")
    dlc_m = m["dlc"]
    snapshot_path = bdir / dlc_m.get("snapshot_file", "snapshot.pt")
    pytorch_config_path = bdir / dlc_m.get("pytorch_config_file", "pytorch_config.yaml")

    if not snapshot_path.is_file():
        raise FileNotFoundError(f"snapshot missing in bundle {bundle_id}: {snapshot_path}")
    if not pytorch_config_path.is_file():
        raise FileNotFoundError(
            f"pytorch_config.yaml missing in bundle {bundle_id}: {pytorch_config_path}"
        )

    if verify_sha:
        if dlc_m.get("snapshot_sha256"):
            actual = _sha256(snapshot_path)
            if actual != dlc_m["snapshot_sha256"]:
                raise BundleIntegrityError(
                    f"snapshot hash mismatch for {bundle_id}: "
                    f"expected {dlc_m['snapshot_sha256']}, got {actual}"
                )
        if dlc_m.get("pytorch_config_sha256"):
            actual = _sha256(pytorch_config_path)
            if actual != dlc_m["pytorch_config_sha256"]:
                raise BundleIntegrityError(
                    f"pytorch_config hash mismatch for {bundle_id}"
                )

    dlc = DlcInfo(
        model_name=dlc_m["model_name"],
        shuffle=int(dlc_m["shuffle"]),
        snapshot=dlc_m["snapshot"],
        snapshot_epoch=int(dlc_m.get("snapshot_epoch", 0)),
        net_type=dlc_m["net_type"],
        backbone=dlc_m["backbone"],
        snapshot_path=snapshot_path,
        snapshot_sha256=dlc_m.get("snapshot_sha256"),
        pytorch_config_path=pytorch_config_path,
        pytorch_config_sha256=dlc_m.get("pytorch_config_sha256"),
        keypoints=list(dlc_m["keypoints"]),
        h5_scorer_string=dlc_m["dlc_h5_scorer_string"],
        deeplabcut_version_min=dlc_m.get("deeplabcut_version_min", "3.0.0"),
    )

    classifiers: List[ClassifierEntry] = []
    for name, c in m.get("classifiers", {}).items():
        cls_path = bdir / c["file"]
        if not cls_path.is_file():
            raise FileNotFoundError(
                f"Classifier file missing in bundle {bundle_id}: {cls_path}"
            )
        if verify_sha and c.get("sha256"):
            actual = _sha256(cls_path)
            if actual != c["sha256"]:
                raise BundleIntegrityError(
                    f"Classifier hash mismatch for {name}: expected {c['sha256']}, got {actual}"
                )
        pp = c.get("postprocess", {})
        classifiers.append(ClassifierEntry(
            name=name,
            display_name=c.get("display_name", name),
            path=cls_path,
            sha256=c.get("sha256"),
            trained_on_feature_hash=c.get("trained_on_feature_hash"),
            trained_on_dlc_model=c.get("trained_on_dlc_model"),
            best_thresh=float(pp.get("best_thresh", 0.5)),
            min_bout=int(pp.get("min_bout", 1)),
            min_after_bout=int(pp.get("min_after_bout", 0)),
            max_gap=int(pp.get("max_gap", 0)),
        ))

    return Bundle(
        bundle_id=m["bundle_id"],
        display_name=m.get("display_name", m["bundle_id"]),
        bundle_version=m.get("bundle_version", "?"),
        release_date=m.get("release_date", "?"),
        pixelpaws_min_version=m.get("pixelpaws_min_version", "0.0.0"),
        inference_backend=backend,
        path=bdir,
        dlc=dlc,
        classifiers=classifiers,
        feature_config=m.get("features", {}),
        group_analysis_defaults=m.get("group_analysis_defaults", {}),
        manifest_raw=m,
    )
