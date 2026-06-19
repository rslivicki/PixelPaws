"""Extract pose + brightness + optical-flow features for one session to
match the canonical 8aed1c22 hash (sq=40, OF=True, bp=hrpaw/hlpaw/snout).

Used by `batch_extract_8aed1c22.py` to extract one session per child
process so multiple sessions can run in parallel.

Usage:
  python extract_session_8aed1c22.py <video_stem> <h5_path> <mp4_path> <out_dir>

Skips if the target pkl already exists.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pose_features import PoseFeatureExtractor  # noqa: E402
from brightness_features import (  # noqa: E402
    PixelBrightnessExtractorOptimized,
)
from optical_flow_features import OpticalFlowExtractor  # noqa: E402
from feature_cache import FeatureCacheManager  # noqa: E402


BODY_PARTS = [
    "snout", "neck", "tailbase", "tailtip",
    "hlpaw", "hrpaw", "flpaw", "frpaw", "centroid",
]
BRT_BPS = ["hrpaw", "hlpaw", "snout"]  # this exact order makes hash 8aed1c22
OF_BPS = ["hrpaw", "hlpaw", "snout"]
SQUARE = 40
PIX_THRESHOLD = 0.3
EXPECTED_HASH = "8aed1c22"


def step(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video_stem")
    ap.add_argument("h5_path")
    ap.add_argument("mp4_path")
    ap.add_argument("out_dir")
    args = ap.parse_args(argv)

    h5 = Path(args.h5_path)
    mp4 = Path(args.mp4_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pkl = out_dir / f"{args.video_stem}_features_{EXPECTED_HASH}.pkl"

    # Skip if already extracted
    if out_pkl.is_file():
        step(f"[{args.video_stem}] already extracted: {out_pkl.name}")
        return 0

    # Hash sanity
    cfg = dict(
        bp_include_list=None,
        bp_pixbrt_list=BRT_BPS,
        square_size=[SQUARE, SQUARE, SQUARE],
        pix_threshold=PIX_THRESHOLD,
        include_optical_flow=True,
        bp_optflow_list=OF_BPS,
    )
    actual_hash = FeatureCacheManager.compute_hash(cfg)
    if actual_hash != EXPECTED_HASH:
        step(f"!! hash mismatch: cfg produced {actual_hash}, expected {EXPECTED_HASH}")
        return 1

    if not h5.is_file():
        step(f"!! missing h5: {h5}")
        return 1
    if not mp4.is_file():
        step(f"!! missing mp4: {mp4}")
        return 1

    step(f"[{args.video_stem}] start; h5={h5.name} mp4={mp4.name}")

    # 1) Pose features
    t0 = time.time()
    pose_ext = PoseFeatureExtractor(
        bodyparts=BODY_PARTS,
        likelihood_threshold=0.8,
        velocity_delta=2,
    )
    pose_df = pose_ext.extract_all_features(str(h5))
    step(f"[{args.video_stem}] pose: {pose_df.shape} in {time.time()-t0:.1f}s")

    # 2) OF extractor (preloaded for in-pass extraction)
    of_ext = OpticalFlowExtractor(bodyparts=OF_BPS).preload(str(h5))

    # 3) Brightness + optical flow (single pass through video)
    t1 = time.time()
    brt_ext = PixelBrightnessExtractorOptimized(
        bodyparts_to_track=BRT_BPS,
        square_size=SQUARE,
        pixel_threshold=PIX_THRESHOLD,
        min_prob=0.8,
    )
    brt_df = brt_ext.extract_brightness_features(
        dlc_file=str(h5),
        video_file=str(mp4),
        dt_vel=2,
        create_video=False,
        optical_flow_extractor=of_ext,
    )
    step(f"[{args.video_stem}] brt+OF: {brt_df.shape} in {time.time()-t1:.1f}s")

    # 4) Align + concat
    n = min(len(pose_df), len(brt_df))
    if len(pose_df) != len(brt_df):
        step(f"[{args.video_stem}] length mismatch pose={len(pose_df)} "
             f"brt={len(brt_df)} — truncating to {n}")
    df = pd.concat(
        [pose_df.iloc[:n].reset_index(drop=True),
         brt_df.iloc[:n].reset_index(drop=True)],
        axis=1,
    )
    df = df.loc[:, ~df.columns.duplicated()]
    step(f"[{args.video_stem}] final: {df.shape}")

    # 5) Save
    try:
        joblib.dump(df, out_pkl, compress=("lz4", 1))
    except ValueError:
        joblib.dump(df, out_pkl, compress=("zlib", 3))

    # Sidecar
    sidecar = out_pkl.with_suffix(".pkl.version.json")
    sidecar.write_text(json.dumps({
        "pose_feature_version": 5,
        "brightness_feature_version": 1,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "feature_hash": EXPECTED_HASH,
        "config": cfg,
    }, indent=2))

    step(f"[{args.video_stem}] saved: {out_pkl.name} "
         f"({out_pkl.stat().st_size/1e6:.0f} MB)  TOTAL {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
