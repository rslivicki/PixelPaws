"""Extract pose + brightness features for one missing session to match
the canonical 0b787aea feature schema used across the L_flinching
training cohort.

Target session: 260129_Formalin_3127 (only had old 375-col 82cc6538 in
FeatureCache; the L_flinching training cohort needs the 599-col
0b787aea pkl to participate).

Config (matches `2511_Blackbox_Formalin/PixelPaws/Flinching/PixelPaws_project.json`,
the project that produced the rest of the cohort's 0b787aea pkls):
  bp_include_list  = None  (all 9 body parts)
  bp_pixbrt_list   = ['hlpaw','hrpaw','snout']
  square_size      = [20]
  pix_threshold    = 0.3
  include_optical_flow = False

The extraction script does NOT add post-cache augmentations (Brightness
Cat-B, normalized distances) — those are added at training time by
`prediction_pipeline._add_post_cache_features` and the trainer.
"""
from __future__ import annotations

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
from feature_cache import FeatureCacheManager  # noqa: E402


# ----- inputs -----
H5 = Path(r"E:/RSVIDS/Blackbox/2601_JDR_videos/John-A657T- Formalin 2%/Videos/"
          r"260129_Formalin_3127DLC_Resnet50_palmreader-500Mar25shuffle1_snapshot_best-110_filtered.h5")
MP4 = Path(r"E:/RSVIDS/Blackbox/2601_JDR_videos/John-A657T- Formalin 2%/Videos/"
           r"260129_Formalin_3127.mp4")
SESSION = "260129_Formalin_3127"

BODY_PARTS = [
    "snout", "neck", "tailbase", "tailtip",
    "hlpaw", "hrpaw", "flpaw", "frpaw", "centroid",
]
BRT_BPS = ["hlpaw", "hrpaw", "snout"]
SQUARE = 20
PIX_THRESHOLD = 0.3

# Canonical hash for this config (verified by reverse-engineering)
EXPECTED_HASH = "0b787aea"

OUT_DIR = Path(r"E:/RSVIDS/Blackbox/2601_JDR_videos/John-A657T- Formalin 2%/features")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PKL = OUT_DIR / f"{SESSION}_features_{EXPECTED_HASH}.pkl"


def step(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def main() -> int:
    # Sanity check hash
    cfg = dict(
        bp_include_list=None,
        bp_pixbrt_list=BRT_BPS,
        square_size=[SQUARE],
        pix_threshold=PIX_THRESHOLD,
        include_optical_flow=False,
        bp_optflow_list=[],
    )
    actual_hash = FeatureCacheManager.compute_hash(cfg)
    if actual_hash != EXPECTED_HASH:
        step(f"!! Hash mismatch: cfg produced {actual_hash}, expected {EXPECTED_HASH}")
        return 1
    step(f"config hash OK: {actual_hash}")

    if not H5.is_file():
        step(f"!! missing DLC h5: {H5}")
        return 1
    if not MP4.is_file():
        step(f"!! missing video: {MP4}")
        return 1

    step("=== Pose features ===")
    t0 = time.time()
    pose_ext = PoseFeatureExtractor(
        bodyparts=BODY_PARTS,
        likelihood_threshold=0.8,
        velocity_delta=2,
    )
    pose_df = pose_ext.extract_all_features(str(H5))
    step(f"pose: {pose_df.shape} in {time.time()-t0:.1f}s")

    step("=== Brightness features ===")
    t1 = time.time()
    brt_ext = PixelBrightnessExtractorOptimized(
        bodyparts_to_track=BRT_BPS,
        square_size=SQUARE,
        pixel_threshold=PIX_THRESHOLD,
        min_prob=0.8,
    )
    brt_df = brt_ext.extract_brightness_features(
        dlc_file=str(H5),
        video_file=str(MP4),
        dt_vel=2,
        create_video=False,
        optical_flow_extractor=None,
    )
    step(f"brightness: {brt_df.shape} in {time.time()-t1:.1f}s")

    # Align & concat
    step("=== Aligning + concat ===")
    n = min(len(pose_df), len(brt_df))
    if len(pose_df) != len(brt_df):
        step(f"  length mismatch pose={len(pose_df)} brt={len(brt_df)} — truncating to {n}")
    pose_df = pose_df.iloc[:n].reset_index(drop=True)
    brt_df = brt_df.iloc[:n].reset_index(drop=True)
    df = pd.concat([pose_df, brt_df], axis=1)
    # drop dupe columns if any
    df = df.loc[:, ~df.columns.duplicated()]
    step(f"final shape: {df.shape}")

    # Confirm column count matches 0b787aea schema (599)
    if df.shape[1] != 599:
        step(f"!! WARNING: expected 599 cols (0b787aea schema), got {df.shape[1]}")
        step("    Inspect mismatch before saving. Skipping save.")
        # Print extra/missing columns vs a reference
        import joblib as _jl
        ref = _jl.load(r"E:/RSVIDS/Blackbox/2511_Blackbox_Formalin/PixelPaws/Flinching/features/"
                       r"260129_Formalin_2802_features_0b787aea.pkl")
        ref_cols = set(ref.columns) if hasattr(ref, "columns") else set()
        new_cols = set(df.columns)
        step(f"    extra: {sorted(new_cols - ref_cols)[:20]}")
        step(f"    missing: {sorted(ref_cols - new_cols)[:20]}")
        return 2

    # Try lz4 (matches existing pkl format); fall back to zlib if missing.
    try:
        joblib.dump(df, OUT_PKL, compress=("lz4", 1))
    except ValueError:
        joblib.dump(df, OUT_PKL, compress=("zlib", 3))
    step(f"saved: {OUT_PKL}  ({OUT_PKL.stat().st_size/1e6:.1f} MB)")

    # Sidecar
    sidecar = OUT_PKL.with_suffix(".pkl.version.json")
    import json
    sidecar.write_text(json.dumps({
        "pose_feature_version": 5,
        "brightness_feature_version": 1,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "feature_hash": EXPECTED_HASH,
        "config": cfg,
    }, indent=2))
    step(f"sidecar: {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
