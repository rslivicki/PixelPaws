"""
One-off: run DLC shuffle 9 on THC1_Baseline.mp4, then a grooming
classifier from the 2603_SNLT_JG/Baseline project, and write a
prediction CSV.

Run from the DEEPLABCUT conda env (DLC 3.x, pytorch). The PixelPaws
imports work because the modules are pure-Python; only DLC's
analyze_videos call requires the env's torch + dlc-models.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Make sure PixelPaws repo is importable.
REPO = Path(r'E:\PixelPaws')
sys.path.insert(0, str(REPO))

CONFIG_PATH = r'E:\RSDLC\2511_RSGNKK_Blackbox\config.yaml'
VIDEO_PATH = r'E:\RSVIDS\Video_transfer_portal\THC1_Baseline.mp4'
SHUFFLE = 9
CLF_PATH = (r'E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\classifiers'
            r'\PixelPaws_body_grooming_pruned_100.pkl')
OUT_DIR = Path(r'E:\RSVIDS\Video_transfer_portal\thc1_baseline_predict')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def step(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def main() -> int:
    # ── 1. DLC inference ───────────────────────────────────────────────
    # If the filtered .h5 already exists (e.g. previous run wrote it),
    # skip the DLC import + analyze step. This lets us re-run the
    # prediction half in any env with xgboost without dragging the
    # DEEPLABCUT conda env along.
    video_dir = Path(VIDEO_PATH).parent
    base = Path(VIDEO_PATH).stem
    h5_files = sorted(video_dir.glob(f'{base}*shuffle{SHUFFLE}*filtered.h5'))
    if not h5_files:
        h5_files = sorted(video_dir.glob(f'{base}*shuffle{SHUFFLE}*.h5'))

    if h5_files:
        dlc_path = str(h5_files[0])
        step(f'Reusing existing DLC output: {dlc_path}')
    else:
        step('Importing DLC...')
        import deeplabcut

        step(f'analyze_videos shuffle={SHUFFLE} on {Path(VIDEO_PATH).name}')
        deeplabcut.analyze_videos(
            CONFIG_PATH,
            [VIDEO_PATH],
            shuffle=SHUFFLE,
            videotype='.mp4',
            save_as_csv=False,
        )
        step('filterpredictions...')
        deeplabcut.filterpredictions(
            CONFIG_PATH,
            [VIDEO_PATH],
            shuffle=SHUFFLE,
            videotype='.mp4',
        )
        h5_files = sorted(video_dir.glob(f'{base}*shuffle{SHUFFLE}*filtered.h5'))
        if not h5_files:
            h5_files = sorted(video_dir.glob(f'{base}*shuffle{SHUFFLE}*.h5'))
        if not h5_files:
            step('! Could not find DLC h5 output')
            return 1
        dlc_path = str(h5_files[0])
        step(f'DLC output: {dlc_path}')

    # ── 2. Load classifier ─────────────────────────────────────────────
    import pickle
    step(f'Loading classifier: {Path(CLF_PATH).name}')
    with open(CLF_PATH, 'rb') as f:
        clf_data = pickle.load(f)

    behavior = clf_data.get('Behavior_type') or 'body_grooming'
    best_thresh = float(clf_data.get('best_thresh', 0.5))
    step(f'  behavior={behavior}    best_thresh={best_thresh:.3f}')

    # ── 3. Read PawCapture calibration (for mm-aware features if trained that way)
    try:
        from pawcapture_meta import read_calibration
        cal = read_calibration(VIDEO_PATH)
        mm_per_pixel = cal['mm_per_pixel'] if cal and 'mm_per_pixel' in cal else None
    except Exception:
        mm_per_pixel = None
    step(f'  PawCapture calibration: mm_per_pixel={mm_per_pixel}')
    step(f'  Classifier training_calibration_mode='
         f'{clf_data.get("training_calibration_mode", "off")}    '
         f'training_mm_per_pixel={clf_data.get("training_mm_per_pixel")}')

    # ── 4. Extract features ────────────────────────────────────────────
    step('Extracting features (may take a few minutes)...')
    from prediction_pipeline import (
        PixelPaws_ExtractFeatures,
        predict_with_xgboost,
        augment_features_post_cache,
    )
    X = PixelPaws_ExtractFeatures(
        pose_data_file=dlc_path,
        video_file_path=VIDEO_PATH,
        bp_include_list=clf_data.get('bp_include_list'),
        bp_pixbrt_list=clf_data.get('bp_pixbrt_list', []),
        square_size=clf_data.get('square_size', [40]),
        pix_threshold=clf_data.get('pix_threshold', 0.3),
        config_yaml_path=None,  # No crop detection needed (fresh PawCapture video)
        clf_data=clf_data,
    )
    step(f'  X shape: {X.shape}')

    # ── 5. Augment + predict ──────────────────────────────────────────
    step('Augmenting + predicting...')
    model = clf_data['clf_model']
    if mm_per_pixel is not None:
        clf_data['mm_per_pixel_at_runtime'] = float(mm_per_pixel)
    X_aug = augment_features_post_cache(X, clf_data, model, dlc_path)
    y_proba = predict_with_xgboost(
        model, X_aug,
        calibrator=clf_data.get('prob_calibrator'),
        fold_models=clf_data.get('fold_models'),
    )
    y_pred = (y_proba >= best_thresh).astype(int)

    # ── 6. Write outputs ──────────────────────────────────────────────
    import pandas as pd
    out_csv = OUT_DIR / f'{base}_{behavior}_predictions.csv'
    pd.DataFrame({
        'frame': range(len(y_proba)),
        f'{behavior}_proba': y_proba,
        behavior: y_pred,
    }).to_csv(out_csv, index=False)
    step(f'Predictions: {out_csv}')

    # Headline metrics
    n = len(y_pred)
    n_pos = int(y_pred.sum())
    bouts = 0
    in_bout = False
    for v in y_pred:
        if v and not in_bout:
            bouts += 1; in_bout = True
        elif not v:
            in_bout = False
    pct = 100 * n_pos / max(1, n)
    step(f'\nSummary:')
    step(f'  total frames:        {n}')
    step(f'  positive frames:     {n_pos}  ({pct:.2f}%)')
    step(f'  inferred bouts:      {bouts}')
    step(f'  proba p10/p50/p90:   {float(pd.Series(y_proba).quantile(0.10)):.3f} / '
         f'{float(pd.Series(y_proba).quantile(0.50)):.3f} / '
         f'{float(pd.Series(y_proba).quantile(0.90)):.3f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
