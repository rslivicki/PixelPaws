"""
Feature extraction for the 2605 FormOxy Left-paws cohort, formalin
recordings only. No classifier predictions yet -- just builds the
canonical feature pickles so any XGBoost classifier (current
Left_licking or future one that wants optical flow) can be run later.

Feature config: same brightness setup as the existing 2512 Left_paws
classifier (PixelPaws_Left_licking.pkl), plus single-pass sparse-LK
optical flow on the same body parts so the cache is future-proof for
flow-aware classifiers.

  bp_pixbrt_list      = ['hrpaw', 'hlpaw', 'snout']
  square_size         = [40, 40, 40]
  pix_threshold       = 0.3
  include_optical_flow = True
  bp_optflow_list     = ['hrpaw', 'hlpaw', 'snout']

The Left_licking classifier (no-flow) can still be applied -- its
`predict_with_xgboost` only consumes the feature columns it was
trained on, ignoring the extra flow columns sitting alongside.

Run in the regular PixelPaws env:
  PYTHONIOENCODING=utf-8 py -X utf8 -u \
      scripts/research/formoxy_features_extract.py [--include-baseline]
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

REPO = Path(r"E:\PixelPaws")
sys.path.insert(0, str(REPO))

COHORT = Path(r"E:\RSVIDS\Blackbox\2605_FormOxy_LeftPaws")
VIDS = COHORT / "videos"
FEATS = COHORT / "features"
FEATS.mkdir(parents=True, exist_ok=True)

SHUFFLE = 9
BP_PIXBRT = ["hrpaw", "hlpaw", "snout"]
SQUARE_SIZE = [40, 40, 40]
PIX_THRESHOLD = 0.3
INCLUDE_OPTICAL_FLOW = True
BP_OPTFLOW = ["hrpaw", "hlpaw", "snout"]


def step(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def feature_hash() -> str:
    from feature_cache import FeatureCacheManager
    cfg = {
        "bp_include_list": None,
        "bp_pixbrt_list": BP_PIXBRT,
        "square_size": SQUARE_SIZE,
        "pix_threshold": PIX_THRESHOLD,
        "include_optical_flow": INCLUDE_OPTICAL_FLOW,
        "bp_optflow_list": BP_OPTFLOW,
    }
    return FeatureCacheManager.compute_hash(cfg)


def main() -> int:
    from prediction_pipeline import PixelPaws_ExtractFeatures

    include_baseline = "--include-baseline" in sys.argv

    if include_baseline:
        videos = sorted(VIDS.glob("2605_FormOxy_S*.mp4"))
    else:
        videos = sorted(VIDS.glob("2605_FormOxy_S*_formalin.mp4"))
    videos = [v for v in videos if "DLC_" not in v.name and "_labeled" not in v.name]
    if not videos:
        step(f"! No videos found in {VIDS}")
        return 1

    hsh = feature_hash()
    step(f"Feature-cache hash: {hsh}")
    step(f"Targets: {len(videos)} video(s)")

    pairs = []
    for v in videos:
        h5s = list(v.parent.glob(f"{v.stem}*shuffle{SHUFFLE}*_filtered.h5"))
        if not h5s:
            step(f"  ! skip {v.name}: no filtered .h5 yet")
            continue
        pairs.append((v, h5s[0]))

    if not pairs:
        step("Nothing to extract (no filtered .h5 ready yet). Run DLC first.")
        return 1

    for v, h5 in pairs:
        out = FEATS / f"{v.stem}_features_{hsh}.pkl"
        if out.is_file():
            step(f"  skip (cached): {out.name}")
            continue
        step(f"  extracting {v.name} ...")
        t0 = time.time()
        X = PixelPaws_ExtractFeatures(
            pose_data_file=str(h5),
            video_file_path=str(v),
            bp_include_list=None,
            bp_pixbrt_list=BP_PIXBRT,
            square_size=SQUARE_SIZE,
            pix_threshold=PIX_THRESHOLD,
            config_yaml_path=None,
            include_optical_flow=INCLUDE_OPTICAL_FLOW,
            bp_optflow_list=BP_OPTFLOW,
        )
        with open(out, "wb") as f:
            pickle.dump(X, f)
        step(f"    {X.shape} in {time.time()-t0:.0f}s -> {out.name}")

    step("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
