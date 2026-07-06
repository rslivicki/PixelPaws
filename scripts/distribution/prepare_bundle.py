"""
prepare_bundle.py — Copy the DLC snapshot + pytorch_config into the default bundle
and stamp the manifest sha256s.

Run this in any env with Python (no DLC needed): it just copies two files and
updates manifest.json. The GUI's pose-tracking tool loads the resulting bundle
from default_bundle/pixelpaws_v1 (seeded into %LOCALAPPDATA%\\PixelPaws\\bundles on
first run).

Usage:
    python scripts/distribution/prepare_bundle.py [--dlc-root <path>] [--bundle-dir <path>]
                                                  [--shuffle N] [--snapshot-stem snapshot-best-930]

Defaults target the current SlivickiR_WangH network (task pixelpaws / date Jul26),
iteration-0 / shuffle 1 / snapshot-best-930.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


# Current SlivickiR_WangH network (see pipeline/pp_config.POSE_DLC_CONFIG).
DEFAULT_DLC_ROOT = Path(r"D:\PixelPaws_Active\PixelPaws_SlivickiR_WangH")
DEFAULT_BUNDLE_DIR = Path(__file__).resolve().parents[2] / "default_bundle" / "pixelpaws_v1"
DEFAULT_SHUFFLE = 1
DEFAULT_SNAPSHOT_STEM = "snapshot-best-930"
DEFAULT_TRAIN_FRACTION = 0.95


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_train_folder(dlc_root: Path, shuffle: int, train_fraction: float) -> Path:
    """Walk the DLC project dir to find the pytorch model train/ folder for the
    given shuffle/train-fraction, preferring the latest iteration."""
    pytorch_root = dlc_root / "dlc-models-pytorch"
    if not pytorch_root.is_dir():
        raise FileNotFoundError(pytorch_root)
    iters = sorted(
        (p for p in pytorch_root.iterdir() if p.name.startswith("iteration-")),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    if not iters:
        raise FileNotFoundError(f"No iteration-* under {pytorch_root}")
    pct = int(round(train_fraction * 100))
    for it in reversed(iters):  # prefer the latest iteration
        for model_dir in sorted(it.iterdir()):
            train_dir = model_dir / "train"
            if train_dir.is_dir() and model_dir.name.endswith(f"trainset{pct}shuffle{shuffle}"):
                return train_dir
    raise FileNotFoundError(
        f"No train folder matching shuffle={shuffle} trainset={pct} under {pytorch_root}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dlc-root", type=Path, default=DEFAULT_DLC_ROOT)
    ap.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE_DIR)
    ap.add_argument("--shuffle", type=int, default=DEFAULT_SHUFFLE)
    ap.add_argument("--snapshot-stem", default=DEFAULT_SNAPSHOT_STEM)
    ap.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    args = ap.parse_args()

    train_dir = _resolve_train_folder(args.dlc_root, args.shuffle, args.train_fraction)
    snap_src = train_dir / f"{args.snapshot_stem}.pt"
    cfg_src = train_dir / "pytorch_config.yaml"
    for p in (snap_src, cfg_src):
        if not p.is_file():
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 1

    args.bundle_dir.mkdir(parents=True, exist_ok=True)
    snap_dst = args.bundle_dir / "snapshot.pt"
    cfg_dst = args.bundle_dir / "pytorch_config.yaml"
    manifest_path = args.bundle_dir / "manifest.json"

    print(f"[prepare] DLC train folder:   {train_dir}")
    print(f"[prepare] Bundle dir:         {args.bundle_dir}")
    print(f"[prepare] Copying snapshot:   {snap_src.name}  ({snap_src.stat().st_size / 1e6:.1f} MB)")
    shutil.copyfile(snap_src, snap_dst)
    print(f"[prepare] Copying pyt config: {cfg_src.name}")
    shutil.copyfile(cfg_src, cfg_dst)

    snap_sha = _sha256(snap_dst)
    cfg_sha = _sha256(cfg_dst)
    print(f"[prepare] snapshot sha256:    {snap_sha}")
    print(f"[prepare] config sha256:      {cfg_sha}")

    if manifest_path.is_file():
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        m.setdefault("dlc", {})
        m["dlc"]["snapshot_sha256"] = snap_sha
        m["dlc"]["pytorch_config_sha256"] = cfg_sha
        m["dlc"]["snapshot_file"] = "snapshot.pt"
        m["dlc"]["pytorch_config_file"] = "pytorch_config.yaml"
        manifest_path.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[prepare] Updated manifest:   {manifest_path}")
    else:
        print(f"[prepare] No manifest found at {manifest_path}; skipping update.")

    print("[prepare] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
