"""
Re-use the existing THC1_Baseline DLC h5 and run multiple
classifiers from the SNLT/Baseline project. Extracts features once,
then loops over classifiers and writes a per-classifier prediction CSV.

Run with:
  PYTHONIOENCODING=utf-8 py -X utf8 scripts/research/thc1_baseline_all_classifiers.py
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

REPO = Path(r'E:\PixelPaws')
sys.path.insert(0, str(REPO))

VIDEO_PATH = r'E:\RSVIDS\Video_transfer_portal\THC1_Baseline.mp4'
DLC_H5 = (r'E:\RSVIDS\Video_transfer_portal'
          r'\THC1_BaselineDLC_Resnet50_palmreader-500Mar25shuffle9_snapshot_best-190_filtered.h5')
CLF_DIR = Path(r'E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\classifiers')
OUT_DIR = Path(r'E:\RSVIDS\Video_transfer_portal\thc1_baseline_predict')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Classifiers to run. pruned variants are typically the ones with
# tuned thresholds. Reads behavior name from each pkl.
CLASSIFIERS = [
    'PixelPaws_body_grooming_pruned_100.pkl',
    'PixelPaws_body_grooming_AllFeatures.pkl',
    'PixelPaws_Facial_grooming_pruned_100.pkl',
    'PixelPaws_Facial_grooming_AllFeatures.pkl',
    'PixelPaws_rearing_pruned_100.pkl',
    'PixelPaws_rearing_AllFeatures.pkl',
]


def step(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def find_bouts(y):
    bouts = []
    in_b = False; start = 0
    for i, v in enumerate(y):
        if v and not in_b:
            in_b = True; start = i
        elif not v and in_b:
            in_b = False; bouts.append((start, i - 1))
    if in_b:
        bouts.append((start, len(y) - 1))
    return bouts


def main() -> int:
    import numpy as np
    import pandas as pd
    from prediction_pipeline import (
        PixelPaws_ExtractFeatures,
        predict_with_xgboost,
        augment_features_post_cache,
    )

    # ── 1. Extract features once (with the union of bp_pixbrt across clfs)
    union_bp_pixbrt = []
    union_square_size = [40]
    union_pix_threshold = 0.3
    for fname in CLASSIFIERS:
        with open(CLF_DIR / fname, 'rb') as f:
            cd = pickle.load(f)
        for bp in cd.get('bp_pixbrt_list', []):
            if bp not in union_bp_pixbrt:
                union_bp_pixbrt.append(bp)
        union_square_size = cd.get('square_size', union_square_size)
        union_pix_threshold = cd.get('pix_threshold', union_pix_threshold)
    step(f'union bp_pixbrt = {union_bp_pixbrt}')

    step('Extracting features (single pass)...')
    X = PixelPaws_ExtractFeatures(
        pose_data_file=DLC_H5,
        video_file_path=VIDEO_PATH,
        bp_include_list=None,  # all body parts
        bp_pixbrt_list=union_bp_pixbrt,
        square_size=union_square_size,
        pix_threshold=union_pix_threshold,
        config_yaml_path=None,
    )
    step(f'  X shape: {X.shape}')

    # ── 2. Loop over classifiers
    summary_rows = []
    for fname in CLASSIFIERS:
        step(f'\n=== {fname} ===')
        with open(CLF_DIR / fname, 'rb') as f:
            clf_data = pickle.load(f)

        behavior = clf_data.get('Behavior_type') or fname.replace('PixelPaws_', '').replace('.pkl', '')
        best_thresh = float(clf_data.get('best_thresh', 0.5))
        min_bout = int(clf_data.get('min_bout', 2))
        model = clf_data['clf_model']
        step(f'  behavior={behavior}    best_thresh={best_thresh:.3f}    saved min_bout={min_bout}')

        try:
            X_aug = augment_features_post_cache(X.copy(), clf_data, model, DLC_H5)
        except Exception as e:
            step(f'  ! augment failed: {e}')
            continue

        try:
            y_proba = predict_with_xgboost(
                model, X_aug,
                calibrator=clf_data.get('prob_calibrator'),
                fold_models=clf_data.get('fold_models'),
            )
        except Exception as e:
            step(f'  ! predict failed: {e}')
            continue

        y_pred = (y_proba >= best_thresh).astype(int)
        n = len(y_pred)
        n_pos = int(y_pred.sum())

        # raw bouts
        raw = find_bouts(y_pred)
        # apply saved min_bout
        y_filt_saved = y_pred.copy()
        for s, e in raw:
            if e - s + 1 < min_bout:
                y_filt_saved[s:e + 1] = 0
        # apply min_bout=30 (0.5s)
        y_filt_30 = y_pred.copy()
        for s, e in raw:
            if e - s + 1 < 30:
                y_filt_30[s:e + 1] = 0
        # apply min_bout=60 (1.0s)
        y_filt_60 = y_pred.copy()
        for s, e in raw:
            if e - s + 1 < 60:
                y_filt_60[s:e + 1] = 0

        bouts_saved = find_bouts(y_filt_saved)
        bouts_30 = find_bouts(y_filt_30)
        bouts_60 = find_bouts(y_filt_60)

        out_csv = OUT_DIR / f'THC1_Baseline_{behavior}_predictions.csv'
        pd.DataFrame({
            'frame': range(n),
            f'{behavior}_proba': y_proba,
            behavior: y_pred,
        }).to_csv(out_csv, index=False)
        step(f'  csv: {out_csv.name}')

        step(f'  raw:                {len(raw):4d} bouts  {n_pos:6d} pos ({100*n_pos/n:5.2f}%)')
        step(f'  min_bout=saved({min_bout:2d}): {len(bouts_saved):4d} bouts  '
             f'{int(y_filt_saved.sum()):6d} pos ({100*int(y_filt_saved.sum())/n:5.2f}%)')
        step(f'  min_bout=30 (0.5s): {len(bouts_30):4d} bouts  '
             f'{int(y_filt_30.sum()):6d} pos ({100*int(y_filt_30.sum())/n:5.2f}%)')
        step(f'  min_bout=60 (1.0s): {len(bouts_60):4d} bouts  '
             f'{int(y_filt_60.sum()):6d} pos ({100*int(y_filt_60.sum())/n:5.2f}%)')
        step(f'  proba p50/p90/p99 = {np.percentile(y_proba,50):.3f}/'
             f'{np.percentile(y_proba,90):.3f}/{np.percentile(y_proba,99):.3f}')

        summary_rows.append({
            'classifier': fname,
            'behavior': behavior,
            'best_thresh': best_thresh,
            'saved_min_bout': min_bout,
            'raw_pos_pct': round(100 * n_pos / n, 3),
            'raw_bouts': len(raw),
            'saved_bouts': len(bouts_saved),
            'saved_pos_pct': round(100 * int(y_filt_saved.sum()) / n, 3),
            'min30_bouts': len(bouts_30),
            'min30_pos_pct': round(100 * int(y_filt_30.sum()) / n, 3),
            'min60_bouts': len(bouts_60),
            'min60_pos_pct': round(100 * int(y_filt_60.sum()) / n, 3),
            'proba_p90': round(float(np.percentile(y_proba, 90)), 4),
            'proba_p99': round(float(np.percentile(y_proba, 99)), 4),
        })

    # ── 3. Summary table
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = OUT_DIR / 'THC1_Baseline_classifier_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    step(f'\nSummary: {summary_csv}')
    step('\n' + summary_df.to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
