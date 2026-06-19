"""
Standalone paw-contour extraction for the 2605_FormOxy_LeftPaws cohort
formalin sessions.

Mirrors the GUI Gait & Limb tab's contour pass (gait_limb_tab.py:3863-3960,
8898-8932): per frame, around each hind paw DLC keypoint:
  1. crop a (ROI x ROI) grayscale ROI
  2. GaussianBlur(3,3)
  3. Otsu binary threshold
  4. findContours RETR_EXTERNAL, pick largest
  5. area filter (>4 px^2)
  6. compute area / bounding-rect width-height / solidity / aspect / circularity
  7. mean intensity inside the filled contour
  8. every Nth frame, resample to 64 evenly-spaced points and normalise
     (centroid-zeroed, scaled by sqrt(area)) -> stored as shape sample

Per-paw config (matching the 2512 cohort's settings):
  contour_roi_sizes : hlpaw=50, hrpaw=50
  crop_x / crop_y   : 0  (2512 used 250 because those mp4s were wider; the
                          2605 videos are already 744-wide so no offset)
  extraction_stride : 1
  max stored shapes : 500 per paw

Output per session:
  gait_limb_analysis/2605_FormOxy_S<N>_formalin_contour_<hash>.csv
  gait_limb_analysis/2605_FormOxy_S<N>_formalin_contour_<hash>_shapes.npz
  gait_limb_analysis/2605_FormOxy_S<N>_formalin_contour_<hash>.json
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
CONTOUR_PAWS = {"HL": "hlpaw", "HR": "hrpaw"}
ROI_SIZES = {"hlpaw": 50, "hrpaw": 50}
CROP_X = 0
CROP_Y = 0
STRIDE = 1
MAX_SHAPES = 500
MIN_AREA = 4


def step(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def cache_hash() -> str:
    cache_key = {
        "contour_roi_sizes": sorted(ROI_SIZES.items()),
        "crop_x": CROP_X, "crop_y": CROP_Y,
        "stride": STRIDE,
        "min_area": MIN_AREA,
    }
    return hashlib.md5(json.dumps(cache_key, sort_keys=True).encode()).hexdigest()[:8]


def resample_contour(pts, n_points: int = 64):
    """Same as GaitLimbTab._resample_contour."""
    import numpy as np
    if len(pts) < 3:
        return None
    closed = np.vstack([pts, pts[0:1]])
    diffs = np.diff(closed, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cum_len = np.concatenate([[0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]
    if total_len <= 0:
        return None
    t_new = np.linspace(0, total_len, n_points, endpoint=False)
    x_new = np.interp(t_new, cum_len, closed[:, 0])
    y_new = np.interp(t_new, cum_len, closed[:, 1])
    return np.column_stack([x_new, y_new])


def normalize_contour(pts, area: float):
    """Same as GaitLimbTab._normalize_contour."""
    import numpy as np
    if pts is None or area <= 0:
        return None
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    scale = np.sqrt(area)
    if scale > 0:
        centered = centered / scale
    return centered


def load_paw_xy(h5_path: Path):
    """Return {'hlpaw': (x_arr, y_arr), 'hrpaw': (x_arr, y_arr)}."""
    import numpy as np
    import pandas as pd
    df = pd.read_hdf(h5_path)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(c).strip() for c in df.columns.values]
    out = {}
    for bp in ("hlpaw", "hrpaw"):
        x_col = next((c for c in df.columns if c.endswith(f"{bp}_x")), None)
        y_col = next((c for c in df.columns if c.endswith(f"{bp}_y")), None)
        if x_col is None or y_col is None:
            raise RuntimeError(f"no x/y columns for {bp} in {h5_path.name}")
        out[bp] = (df[x_col].to_numpy().astype(float),
                   df[y_col].to_numpy().astype(float))
    return out


def extract_one(video_path: Path, h5_path: Path):
    """Run contour extraction. Returns (per_frame_dict, shape_dict)."""
    import cv2
    import numpy as np

    paw_xy = load_paw_xy(h5_path)
    n_frames_dlc = len(next(iter(paw_xy.values()))[0])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {video_path}")
    n_frames_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = min(n_frames_dlc, n_frames_vid)
    step(f"    {video_path.name}: {n_frames} frames ({fw}x{fh})")

    per_frame = {role: {
        "areas":         np.zeros(n_frames, dtype=float),
        "spreads":       np.zeros(n_frames, dtype=float),
        "intensities":   np.zeros(n_frames, dtype=float),
        "widths":        np.zeros(n_frames, dtype=float),
        "solidities":    np.zeros(n_frames, dtype=float),
        "aspect_ratios": np.zeros(n_frames, dtype=float),
        "circularities": np.zeros(n_frames, dtype=float),
    } for role in CONTOUR_PAWS}
    shape_lists = {role: [] for role in CONTOUR_PAWS}

    shape_every = max(1, n_frames // (MAX_SHAPES * STRIDE))
    t0 = time.time()
    last_log = t0

    i = 0
    while i < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if i % STRIDE == 0:
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            for role, bp in CONTOUR_PAWS.items():
                px = paw_xy[bp][0][i]
                py = paw_xy[bp][1][i]
                if not (np.isfinite(px) and np.isfinite(py)):
                    continue
                bx = int(px) + CROP_X
                by = int(py) + CROP_Y
                rh = ROI_SIZES[bp]
                x1 = max(0, bx - rh); x2 = min(fw, bx + rh)
                y1 = max(0, by - rh); y2 = min(fh, by + rh)
                if x2 <= x1 or y2 <= y1:
                    continue
                roi = gray[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                blurred = cv2.GaussianBlur(roi, (3, 3), 0)
                _, th = cv2.threshold(blurred, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                best = max(contours, key=lambda c: cv2.contourArea(c))
                area = cv2.contourArea(best)
                if area < MIN_AREA:
                    continue

                x_b, y_b, w_b, h_b = cv2.boundingRect(best)
                per_frame[role]["areas"][i] = area
                per_frame[role]["spreads"][i] = max(w_b, h_b)
                mask_c = np.zeros(roi.shape, dtype=np.uint8)
                cv2.drawContours(mask_c, [best], -1, 255, -1)
                per_frame[role]["intensities"][i] = cv2.mean(roi, mask=mask_c)[0]
                per_frame[role]["widths"][i] = min(w_b, h_b)
                hull = cv2.convexHull(best)
                hull_area = cv2.contourArea(hull)
                per_frame[role]["solidities"][i] = area / hull_area if hull_area > 0 else 0.0
                dim_max = max(w_b, h_b); dim_min = min(w_b, h_b)
                per_frame[role]["aspect_ratios"][i] = dim_max / dim_min if dim_min > 0 else 0.0
                perim = cv2.arcLength(best, True)
                per_frame[role]["circularities"][i] = (4 * np.pi * area) / (perim ** 2) if perim > 0 else 0.0

                if (i % shape_every == 0
                        and len(shape_lists[role]) < MAX_SHAPES):
                    pts = best.squeeze()
                    if pts.ndim == 2 and len(pts) >= 3:
                        resampled = resample_contour(pts, 64)
                        normed = normalize_contour(resampled, area)
                        if normed is not None:
                            shape_lists[role].append(normed)

        i += 1
        if time.time() - last_log > 30:
            elapsed = time.time() - t0
            fps_now = i / elapsed if elapsed else 0
            eta_s = (n_frames - i) / fps_now if fps_now else 0
            step(f"      {i}/{n_frames}  ({100*i/n_frames:.1f}%)  "
                 f"{fps_now:.0f} f/s  ETA ~{eta_s/60:.1f} min")
            last_log = time.time()

    cap.release()
    dt = time.time() - t0
    step(f"      done in {dt:.0f}s ({i/dt:.0f} f/s)")
    return per_frame, shape_lists, i


def main() -> int:
    import numpy as np
    import pandas as pd

    h = cache_hash()
    step(f"Cache hash: {h}")
    # Walk ALL cohort sessions: formalin-paw (S1..S12, baseline + formalin)
    # AND vehicle-paw controls (Veh_S1..S6, baseline + vehicle). Idempotent.
    videos = sorted(set(
        list(VIDS.glob("2605_FormOxy_S*_*.mp4"))
        + list(VIDS.glob("2605_FormOxy_Veh_S*_*.mp4"))
    ))
    videos = [v for v in videos if "DLC_" not in v.name and "_labeled" not in v.name]
    step(f"Videos: {len(videos)}")

    n_done = n_skip = n_err = 0
    t_total = time.time()
    for v in videos:
        base = v.stem
        out_csv = OUT_DIR / f"{base}_contour_{h}.csv"
        out_npz = OUT_DIR / f"{base}_contour_{h}_shapes.npz"
        sidecar = out_csv.with_suffix(".json")
        if out_csv.is_file() and out_npz.is_file():
            step(f"  skip (cached): {out_csv.name}")
            n_skip += 1; continue

        h5_candidates = list(VIDS.glob(f"{base}*shuffle{SHUFFLE}*_filtered.h5"))
        if not h5_candidates:
            step(f"  ! no filtered .h5 for {base}"); n_err += 1; continue

        step(f"  extract {base}")
        try:
            per_frame, shape_lists, n_kept = extract_one(v, h5_candidates[0])
        except Exception as e:
            step(f"    ! failed: {type(e).__name__}: {e}")
            n_err += 1; continue

        # Per-frame CSV (matches GUI column naming: metric_ROLE)
        cols = {}
        for role, arrays in per_frame.items():
            for metric, arr in arrays.items():
                cols[f"{metric}_{role}"] = arr
        pd.DataFrame(cols).to_csv(out_csv, index=False)

        shape_dict = {}
        for role, lst in shape_lists.items():
            if lst:
                shape_dict[role] = np.array(lst)
        if shape_dict:
            np.savez_compressed(out_npz, **shape_dict)

        sidecar.write_text(json.dumps({
            "video": str(v), "dlc_h5": str(h5_candidates[0]),
            "video_mtime": v.stat().st_mtime,
            "dlc_mtime":   h5_candidates[0].stat().st_mtime,
            "contour_roi_sizes": [[bp, ROI_SIZES[bp]] for bp in ROI_SIZES],
            "crop_x": CROP_X, "crop_y": CROP_Y,
            "extraction_stride": STRIDE,
            "min_area": MIN_AREA,
            "n_frames": int(n_kept),
            "n_shapes": {r: int(len(shape_lists[r])) for r in shape_lists},
        }, indent=2))
        step(f"    -> {out_csv.name}  ({n_kept} frames, "
             f"HL shapes={len(shape_lists['HL'])}, HR shapes={len(shape_lists['HR'])})")
        n_done += 1

    step(f"\nDone in {(time.time()-t_total)/60:.1f} min. "
         f"New: {n_done}; cached: {n_skip}; errors: {n_err}.")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
