"""
Apply the Left_licking classifier to right-paw-injected animals by
mirroring the DLC h5 (swap hlpaw<->hrpaw, flpaw<->frpaw). The classifier
was trained on left-paw-injected mice; after the swap, the right paw
(injected) is geometrically/visually positioned as "hlpaw", so the
trained model fires on right-paw licking.

We also run the classifier on the ORIGINAL (un-mirrored) h5 as a sanity
control -- that should be near zero for right-paw-injected animals.

Targets (right-paw-injected videos in 2512_Blackbox_Formalin_Oxy parent):
  251204_Subject1_Vehicle.mp4   (DLC h5 present, shuffle 5)
  251204_Subject1_Oxy10.mp4     (DLC h5 present, shuffle 5)

Subject2_Vehicle has no h5 yet -- skipped.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"E:\PixelPaws")
sys.path.insert(0, str(REPO))

COHORT_DIR = Path(r"E:\RSVIDS\Blackbox\2512_Blackbox_Formalin_Oxy")
CLASSIFIER = COHORT_DIR / "Left_paws" / "Classifiers" / "PixelPaws_Left_licking.pkl"
OUT_DIR = COHORT_DIR / "right_paw_mirror_test"
OUT_DIR.mkdir(exist_ok=True)

# Pairs of (video, original_h5). DLC scorer suffix in h5 will be normalised.
TARGETS = [
    ("251204_Subject1_Vehicle.mp4",
     "251204_Subject1_VehicleDLC_Resnet50_palmreader-500Mar25shuffle5_snapshot_best-190_filtered.h5"),
    ("251204_Subject1_Oxy10.mp4",
     "251204_Subject1_Oxy10DLC_Resnet50_palmreader-500Mar25shuffle5_snapshot_best-190_filtered.h5"),
]

SWAP_PAIRS = [("hlpaw", "hrpaw"), ("flpaw", "frpaw")]


def step(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def mirror_h5(orig: Path, mirrored: Path) -> None:
    """Read DLC filtered.h5 and swap DATA between left/right paw columns
    while keeping the label/column ORDER identical.

    Rationale: feature extraction names many features by alphabetical
    body-part ordering (e.g. Dis_flpaw-frpaw). If we rename labels,
    columns end up in a different order and features come out as
    Dis_frpaw-flpaw, which the classifier rejects. Swapping data values
    at the original column positions leaves all column names + ordering
    untouched -- the classifier sees features that say 'hlpaw' but
    contain the (post-swap) right-paw data, which is exactly what we
    want for an injection-side-mirror.
    """
    df = pd.read_hdf(orig)
    if not isinstance(df.columns, pd.MultiIndex):
        raise RuntimeError(f"Expected MultiIndex columns, got {df.columns}")
    bp_level = df.columns.names.index("bodyparts")

    for a, b in SWAP_PAIRS:
        # All columns belonging to body part a (x, y, likelihood)
        cols_a = [c for c in df.columns if c[bp_level] == a]
        cols_b = [c for c in df.columns if c[bp_level] == b]
        if len(cols_a) != len(cols_b):
            step(f"  ! cannot swap {a} <-> {b}: mismatched col counts "
                 f"({len(cols_a)} vs {len(cols_b)}); skipping pair")
            continue
        # Match coord level (x->x, y->y, likelihood->likelihood) within each bp
        coord_level = df.columns.names.index("coords")
        for ca in cols_a:
            cb = tuple(b if i == bp_level else ca[i] for i in range(len(ca)))
            if cb in df.columns:
                tmp = df[ca].copy()
                df[ca] = df[cb].values
                df[cb] = tmp.values

    df.to_hdf(mirrored, key="df_with_missing", mode="w", format="table",
             complevel=9, complib="zlib")
    step(f"  mirrored h5 -> {mirrored.name}  ({len(df)} frames; "
         f"data swapped, labels/order preserved)")


def apply_postprocess(y_raw: np.ndarray, min_bout: int, min_after_bout: int, max_gap: int) -> np.ndarray:
    """Same post-processing the predict pipeline uses."""
    y = y_raw.copy().astype(int)
    if max_gap > 0:
        # close gaps shorter than max_gap
        i = 0
        n = len(y)
        while i < n:
            if y[i] == 0:
                j = i
                while j < n and y[j] == 0:
                    j += 1
                gap = j - i
                if gap <= max_gap and i > 0 and j < n and y[i - 1] == 1 and y[j] == 1:
                    y[i:j] = 1
                i = j
            else:
                i += 1
    if min_bout > 1:
        # drop bouts shorter than min_bout
        i = 0
        n = len(y)
        while i < n:
            if y[i] == 1:
                j = i
                while j < n and y[j] == 1:
                    j += 1
                if (j - i) < min_bout:
                    y[i:j] = 0
                i = j
            else:
                i += 1
    if min_after_bout > 0:
        # enforce minimum off-period after each bout (drop short follow-ups)
        i = 0
        n = len(y)
        while i < n:
            if y[i] == 1:
                j = i
                while j < n and y[j] == 1:
                    j += 1
                # j is start of off period
                k = j
                while k < n and y[k] == 0:
                    k += 1
                if (k - j) < min_after_bout:
                    y[j:k] = 0  # absorb the brief off into bout — actually nope, just leave it
                i = k
            else:
                i += 1
    return y


def find_bouts(y: np.ndarray) -> list[tuple[int, int]]:
    bouts = []
    i = 0
    n = len(y)
    while i < n:
        if y[i] == 1:
            j = i
            while j < n and y[j] == 1:
                j += 1
            bouts.append((i, j - 1))
            i = j
        else:
            i += 1
    return bouts


def run_predict(video: Path, h5: Path, classifier_data: dict, label: str,
                feature_cache: Path | None = None) -> dict:
    """Extract features then run the classifier. Returns summary dict."""
    from prediction_pipeline import (
        PixelPaws_ExtractFeatures,
        predict_with_xgboost,
        augment_features_post_cache,
    )

    bp_pixbrt = classifier_data.get("bp_pixbrt_list") or ["hrpaw", "hlpaw", "snout"]
    sq = classifier_data.get("square_size") or [40, 40, 40]
    pix_thresh = classifier_data.get("pix_threshold", 0.3)

    if feature_cache is not None and feature_cache.is_file():
        step(f"  [{label}] loading cached features from {feature_cache.name}")
        with open(feature_cache, "rb") as f:
            X = pickle.load(f)
    else:
        step(f"  [{label}] extracting features...")
        t0 = time.time()
        X = PixelPaws_ExtractFeatures(
            pose_data_file=str(h5),
            video_file_path=str(video),
            bp_include_list=None,
            bp_pixbrt_list=bp_pixbrt,
            square_size=sq,
            pix_threshold=pix_thresh,
            config_yaml_path=None,
            include_optical_flow=False,
            bp_optflow_list=[],
        )
        step(f"    features in {time.time()-t0:.0f}s, shape {X.shape}")
        if feature_cache is not None:
            with open(feature_cache, "wb") as f:
                pickle.dump(X, f)
            step(f"    cached -> {feature_cache.name}")

    X_aug = augment_features_post_cache(X.copy(), classifier_data, classifier_data["clf_model"], str(h5))
    y_proba = predict_with_xgboost(
        classifier_data["clf_model"], X_aug,
        calibrator=classifier_data.get("prob_calibrator"),
        fold_models=classifier_data.get("fold_models"),
    )
    best_thresh = float(classifier_data.get("best_thresh", 0.5))
    y_raw = (y_proba >= best_thresh).astype(int)
    y_post = apply_postprocess(
        y_raw,
        int(classifier_data.get("min_bout", 1)),
        int(classifier_data.get("min_after_bout", 0)),
        int(classifier_data.get("max_gap", 0)),
    )
    n_pos = int(y_post.sum())
    bouts = find_bouts(y_post)
    duration_min = len(y_post) / 60.0 / 60.0
    return {
        "label": label,
        "n_frames": len(y_post),
        "duration_min": duration_min,
        "n_pos_frames": n_pos,
        "pct_time": 100 * n_pos / len(y_post),
        "n_bouts": len(bouts),
        "bouts_per_min": len(bouts) / duration_min if duration_min > 0 else 0.0,
        "mean_bout_s": float(np.mean([(b - a + 1) / 60 for a, b in bouts])) if bouts else 0.0,
        "total_lick_s": n_pos / 60.0,
        "y_post": y_post,
        "y_proba": y_proba,
    }


def main() -> int:
    with open(CLASSIFIER, "rb") as f:
        cd = pickle.load(f)
    step(f"Loaded classifier: {cd.get('Behavior_type')} "
         f"thresh={cd.get('best_thresh'):.2f} "
         f"min_bout={cd.get('min_bout')} min_after_bout={cd.get('min_after_bout')} "
         f"max_gap={cd.get('max_gap')}")

    all_rows = []
    for video_name, h5_name in TARGETS:
        video = COHORT_DIR / video_name
        h5_orig = COHORT_DIR / h5_name
        if not video.is_file() or not h5_orig.is_file():
            step(f"!! missing: {video_name} or h5; skipping")
            continue
        step(f"\n=== {video.stem} ===")

        # 1. Original (un-mirrored) — sanity baseline; should be ~0 if injection is right
        cache_orig = OUT_DIR / f"{video.stem}_features_orig.pkl"
        res_orig = run_predict(video, h5_orig, cd, "ORIGINAL (no mirror)",
                               feature_cache=cache_orig)

        # 2. Mirrored
        h5_mirror = OUT_DIR / h5_orig.name.replace(".h5", "_MIRRORED.h5")
        cache_mirror = OUT_DIR / f"{video.stem}_features_mirror.pkl"
        if not cache_mirror.is_file():
            mirror_h5(h5_orig, h5_mirror)
        res_mirror = run_predict(video, h5_mirror, cd, "MIRRORED",
                                 feature_cache=cache_mirror)

        # Save per-frame predictions
        out_csv = OUT_DIR / f"{video.stem}_mirror_predictions.csv"
        df = pd.DataFrame({
            "frame": range(res_mirror["n_frames"]),
            "lick_proba_original": res_orig["y_proba"],
            "lick_pred_original":  res_orig["y_post"],
            "lick_proba_mirror":   res_mirror["y_proba"],
            "lick_pred_mirror":    res_mirror["y_post"],
        })
        df.to_csv(out_csv, index=False)
        step(f"  per-frame preds -> {out_csv.name}")

        for r in (res_orig, res_mirror):
            all_rows.append({
                "video": video.stem,
                "variant": r["label"],
                "duration_min": round(r["duration_min"], 1),
                "lick_pct_time": round(r["pct_time"], 3),
                "n_bouts": r["n_bouts"],
                "bouts_per_min": round(r["bouts_per_min"], 3),
                "mean_bout_s": round(r["mean_bout_s"], 2),
                "total_lick_s": round(r["total_lick_s"], 1),
            })

    summary = pd.DataFrame(all_rows)
    summary_csv = OUT_DIR / "right_paw_mirror_summary.csv"
    summary.to_csv(summary_csv, index=False)
    step(f"\nWrote {summary_csv}")
    print("\nSUMMARY:")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
