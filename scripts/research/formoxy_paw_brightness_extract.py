"""
Extract per-frame paw brightness (mean pixel intensity in fixed ROI) for
the 12 formalin sessions in the 2605_FormOxy_LeftPaws cohort.

Replicates the GUI Gait & Limb tab's brightness extraction with the same
ROI configuration used for the 2512 cohort:
  hlpaw / hrpaw : 20-px ROI
  flpaw / frpaw : 15-px ROI
  pixel_threshold : 101  (only count pixels above this intensity)
  extraction_stride : 1   (every frame)

The 2605 videos are 744x720 (already cropped to the test area) so
crop_offset_x/y default to 0 (the 2512 cohort used 250/0 because those
videos were wider).

Output: `2605_FormOxy_LeftPaws/gait_limb_analysis/2605_FormOxy_S<N>_formalin_brt_<hash>.csv`
with columns Pix_hlpaw, Pix_hrpaw, Pix_flpaw, Pix_frpaw, plus a sidecar
.json with the extraction settings.

Run from regular py:
  PYTHONIOENCODING=utf-8 py -X utf8 -u scripts/research/formoxy_paw_brightness_extract.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(r"E:\PixelPaws")
sys.path.insert(0, str(REPO))

COHORT = Path(r"E:\RSVIDS\Blackbox\2605_FormOxy_LeftPaws")
VIDS = COHORT / "videos"
OUT_DIR = COHORT / "gait_limb_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHUFFLE = 9
BODYPARTS = ["flpaw", "frpaw", "hlpaw", "hrpaw"]
ROI_SIZES = {"flpaw": 15, "frpaw": 15, "hlpaw": 20, "hrpaw": 20}
PIXEL_THRESHOLD = 101
EXTRACTION_STRIDE = 1
CROP_X = 0
CROP_Y = 0


def step(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def cache_hash() -> str:
    """Same key format the GUI uses for brightness cache filenames."""
    cache_key = {
        "bodyparts":      BODYPARTS,
        "roi_sizes":      sorted(ROI_SIZES.items()),
        "pixel_threshold": PIXEL_THRESHOLD,
        "stride":         EXTRACTION_STRIDE,
        "crop_x":         CROP_X,
        "crop_y":         CROP_Y,
    }
    return hashlib.md5(json.dumps(cache_key, sort_keys=True).encode()).hexdigest()[:8]


def main() -> int:
    import pandas as pd
    from brightness_features import PixelBrightnessExtractorOptimized

    h = cache_hash()
    step(f"Cache hash: {h}")

    # Walk ALL cohort sessions (baseline + formalin); idempotent so
    # already-cached files are skipped.
    videos = sorted(set(
        list(VIDS.glob("2605_FormOxy_S*_*.mp4"))
        + list(VIDS.glob("2605_FormOxy_Veh_S*_*.mp4"))
    ))
    videos = [v for v in videos if "DLC_" not in v.name and "_labeled" not in v.name]
    step(f"Videos: {len(videos)}")

    n_done = 0
    n_skip = 0
    n_err = 0
    t_total = time.time()
    for v in videos:
        base = v.stem
        out_csv = OUT_DIR / f"{base}_brt_{h}.csv"
        sidecar = out_csv.with_suffix(".json")
        if out_csv.is_file():
            step(f"  skip (cached): {out_csv.name}")
            n_skip += 1
            continue

        h5_candidates = list(VIDS.glob(f"{base}*shuffle{SHUFFLE}*_filtered.h5"))
        if not h5_candidates:
            step(f"  ! no filtered .h5 for {base}; skipping")
            n_err += 1
            continue
        h5 = h5_candidates[0]

        step(f"  extract {base} ...")
        t0 = time.time()
        try:
            ex = PixelBrightnessExtractorOptimized(
                BODYPARTS,
                square_size={k: v_ for k, v_ in ROI_SIZES.items()},
                pixel_threshold=float(PIXEL_THRESHOLD),
                crop_offset_x=CROP_X,
                crop_offset_y=CROP_Y,
            )
            brt_df = ex.extract_brightness_features(
                str(h5), str(v),
                stride=EXTRACTION_STRIDE,
            )
        except Exception as e:
            step(f"    ! extraction failed: {type(e).__name__}: {e}")
            n_err += 1
            continue

        # Keep only the four paw columns; align names to GUI's convention
        out_cols = {}
        for bp in ["hlpaw", "hrpaw", "flpaw", "frpaw"]:
            col = f"Pix_{bp}"
            if col in brt_df.columns:
                out_cols[col] = brt_df[col].reset_index(drop=True)
            else:
                step(f"    ! missing column {col} in extractor output")

        if not out_cols:
            step(f"    ! no usable columns; skipping write")
            n_err += 1
            continue

        out_df = pd.DataFrame(out_cols)
        out_df.to_csv(out_csv, index=False)
        sidecar.write_text(json.dumps({
            "video":               str(v),
            "dlc_h5":              str(h5),
            "video_mtime":         v.stat().st_mtime,
            "dlc_mtime":           h5.stat().st_mtime,
            "roi_sizes":           [[bp, ROI_SIZES[bp]] for bp in BODYPARTS],
            "crop_x":              CROP_X,
            "crop_y":              CROP_Y,
            "brt_thresh":          PIXEL_THRESHOLD,
            "extraction_stride":   EXTRACTION_STRIDE,
            "n_frames":            int(len(out_df)),
        }, indent=2))
        dt = time.time() - t0
        step(f"    -> {out_csv.name} ({len(out_df)} frames, {dt:.0f}s)")
        n_done += 1

    elapsed = (time.time() - t_total) / 60
    step(f"\nDone in {elapsed:.1f} min. New: {n_done}; cached: {n_skip}; errors: {n_err}.")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
