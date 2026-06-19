"""regen_flinching_augmented.py — Same flinching pipeline as
``regen_flinching_plots.py`` but with the GUI-style post-cache feature
augmentation enabled:

  • Brightness Category B (derived from Pix_ cols)
  • Normalized pairwise distances (ARBEL-parity Dis_norm_*)
  • Egocentric distances + velocities (Ego_*) — requires DLC h5
  • Multi-timescale rolling stats (std + max @ 100ms, 700ms)
  • Contact state features (from *_Height cols)
  • Lag/lead features (top-N by variance, ±1/±2 frame shifts)

Caches the augmented X and re-runs OOF + LOVO + plots. Heavier than
the un-augmented baseline but the augmentations are the same ones
the GUI _real_training adds, so this is what 'PixelPaws-trained'
flinching actually looks like.
"""

from __future__ import annotations

import os
import sys
import time
import glob
import re
from datetime import datetime

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                # peer scripts in this folder
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # project root (E:/PixelPaws)
from regen_scratching_plots import (
    cv_oof_proba, train_full_model, learning_curve, lovo_oof,
    plot_threshold_curves, plot_shap_panel, plot_learning_curve,
    plot_per_video_metrics, plot_per_session_diagnostics,
    PLASMA, GRID_GREY, TEXT_GREY,
)
import regen_scratching_plots as _rsp

from prediction_pipeline import (
    compute_brightness_category_b, compute_normalized_distances,
)
from pose_features import PoseFeatureExtractor


PROJECT      = r'E:/RSVIDS/Blackbox/2511_Blackbox_Formalin/PixelPaws/Flinching'
BEHAVIOR     = 'L_flinching'
LABEL_COL    = 'L_flinching'
FEATURE_HASH = '0b787aea'
PLOTS_BASE   = os.path.join(PROJECT, 'classifiers', 'plots',
                              f'regen_aug_{datetime.now():%Y%m%d_%H%M%S}')
N_FOLDS      = 5
RANDOM_SEED  = 42
FPS          = 60.0


def _find_dlc_h5(session_name: str) -> str | None:
    """Locate the DLC h5 file for a session inside the project's
    `Videos/` folder. Prefers the unfiltered file; falls back to
    `_filtered.h5` if necessary."""
    videos_dir = os.path.join(PROJECT, 'Videos')
    pat = os.path.join(videos_dir, f'{session_name}DLC*.h5')
    cands = sorted(glob.glob(pat))
    raw = [c for c in cands if '_filtered' not in c]
    chosen = raw or cands
    return chosen[0] if chosen else None


def augment_session(X: pd.DataFrame, dlc_path: str | None) -> pd.DataFrame:
    """Apply the full GUI augmentation suite to a single session's
    feature cache. Cheaper transforms first; egocentric (DLC h5
    parse) last because it's the most expensive step."""
    n0 = X.shape[1]

    # 1. Brightness Category B (always — derives from Pix_ cols)
    X = compute_brightness_category_b(X, log_fn=None)

    # 2. Normalized pairwise distances (always — derives from Dis_ cols)
    X = compute_normalized_distances(X, log_fn=None)

    # 3. Multi-timescale rolling stats (std + max @ 100/700ms)
    try:
        ext_ms = PoseFeatureExtractor(bodyparts=[])
        ms_df = ext_ms.calculate_multiscale_features(
            X, fps=FPS, windows_ms=(100, 700), stats=('std', 'max'))
        if not ms_df.empty:
            X = pd.concat([X.reset_index(drop=True),
                              ms_df.reset_index(drop=True)], axis=1)
    except Exception as e:
        print(f'    ⚠ multi-timescale failed: {e}')

    # 4. Contact state features (from *_Height cols)
    try:
        ct_ext = PoseFeatureExtractor(bodyparts=[], contact_threshold=15.0)
        ct_df = ct_ext.calculate_contact_features(X)
        if not ct_df.empty:
            X = pd.concat([X.reset_index(drop=True),
                              ct_df.reset_index(drop=True)], axis=1)
    except Exception as e:
        print(f'    ⚠ contact features failed: {e}')

    # 5. Lag/lead features (variance-selected top 10 base cols)
    try:
        lag_ext = PoseFeatureExtractor(bodyparts=[])
        lag_df = lag_ext.calculate_lag_features(
            X, lags=(-2, -1, 1, 2), top_n=10)
        if not lag_df.empty:
            X = pd.concat([X.reset_index(drop=True),
                              lag_df.reset_index(drop=True)], axis=1)
    except Exception as e:
        print(f'    ⚠ lag features failed: {e}')

    # 6. Egocentric features — requires DLC h5
    if dlc_path is not None and os.path.isfile(dlc_path):
        try:
            ego_ext = PoseFeatureExtractor(bodyparts=[])
            ego_dlc = ego_ext.load_dlc_data(dlc_path)
            ego_xc, ego_yc, _ = ego_ext.get_bodypart_coords(ego_dlc)
            ego_x, ego_y = ego_ext.normalize_egocentric(ego_xc, ego_yc)
            ego_dist = ego_ext.calculate_distances(ego_x, ego_y)
            ego_dist.columns = [f'Ego_{c}' for c in ego_dist.columns]
            ego_vel  = ego_ext.calculate_velocities(ego_x, ego_y, t=1)
            ego_vel.columns  = [f'Ego_{c}' for c in ego_vel.columns]
            ego_df = pd.concat([ego_dist, ego_vel], axis=1).fillna(0)
            ego_df = ego_df.iloc[:len(X)].reset_index(drop=True)
            X = pd.concat([X.reset_index(drop=True), ego_df], axis=1)
        except Exception as e:
            print(f'    ⚠ egocentric failed: {e}')

    print(f'    augment: {n0} → {X.shape[1]} cols (+{X.shape[1] - n0})')
    return X


def build_augmented_training_set(sessions: list[tuple[str, str, str]]
                                   ) -> tuple[pd.DataFrame, np.ndarray,
                                                list[tuple[str, int]]]:
    X_parts, y_parts, splits = [], [], []
    common_cols = None
    for sname, label_path, feat_path in sessions:
        feat = pd.read_pickle(feat_path)
        lab  = pd.read_csv(label_path)
        if LABEL_COL not in lab.columns:
            print(f'  ⚠ {sname}: column {LABEL_COL!r} missing — skip')
            continue
        n = min(len(feat), len(lab))
        X_s = feat.iloc[:n].reset_index(drop=True)
        y_s = lab[LABEL_COL].iloc[:n].astype(int).values

        # Augment per-session (egocentric needs the matching DLC file)
        dlc = _find_dlc_h5(sname)
        if dlc is None:
            print(f'  {sname}: no DLC h5 — egocentric features skipped')
        t0 = time.time()
        X_aug = augment_session(X_s, dlc)
        print(f'  {sname}: aug took {time.time() - t0:.1f}s, '
              f'shape={X_aug.shape}, pos={int(y_s.sum())}')

        # Trim to length n (some augmentations can append/cut a row)
        n = min(n, len(X_aug))
        X_aug = X_aug.iloc[:n]
        y_s   = y_s[:n]

        if common_cols is None:
            common_cols = list(X_aug.columns)
        else:
            common_cols = [c for c in common_cols if c in X_aug.columns]

        X_parts.append(X_aug)
        y_parts.append(y_s)
        splits.append((sname, n))

    X_parts = [df[common_cols] for df in X_parts]
    X = pd.concat(X_parts, ignore_index=True)
    y = np.concatenate(y_parts)
    return X, y, splits


def main():
    os.makedirs(PLOTS_BASE, exist_ok=True)
    print(f'[1/6] scanning sessions…')
    labels_dir   = os.path.join(PROJECT, 'behavior_labels')
    features_dir = os.path.join(PROJECT, 'features')
    labels = {f.replace('_labels.csv', ''):
                  os.path.join(labels_dir, f)
                for f in os.listdir(labels_dir)
                if f.endswith('_labels.csv')}
    feats = {}
    for f in os.listdir(features_dir):
        if f.endswith(f'_features_{FEATURE_HASH}.pkl'):
            sname = f.replace(f'_features_{FEATURE_HASH}.pkl', '')
            feats[sname] = os.path.join(features_dir, f)
    sessions = []
    for s in sorted(set(labels) & set(feats)):
        sessions.append((s, labels[s], feats[s]))
    print(f'      {len(sessions)} sessions usable')

    print(f'[2/6] building AUGMENTED training set (per-session)…')
    t0 = time.time()
    X, y, splits = build_augmented_training_set(sessions)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    print(f'      total aug build time: {(time.time()-t0)/60:.1f} min')
    print(f'      X={X.shape}  y_pos={y.sum()} ({y.mean()*100:.2f}%)')

    pd.to_pickle({'X': X, 'y': y, 'splits': splits},
                   os.path.join(PLOTS_BASE, '_training_set.pkl'))

    # Patch SESSION_SPLITS for plot helpers
    _rsp.SESSION_SPLITS = [(s.replace('260129_Formalin_', 'F_')
                                  .replace('251126_Formalin_F_', 'FS_')
                                  .replace('_Cut', ''), n)
                              for s, n in splits]

    print(f'[3/6] {N_FOLDS}-fold OOF CV…')
    oof, fold_f1 = cv_oof_proba(X, y, n_folds=N_FOLDS, n_estimators=250)
    print(f'      mean F1 (per-fold) = {np.mean(fold_f1):.3f}±'
          f'{np.std(fold_f1):.3f}')

    print(f'[4/6] panel C — threshold curves…')
    pc_path = os.path.join(PLOTS_BASE,
                              f'PixelPaws_{BEHAVIOR}_panelC_threshold.png')
    plot_threshold_curves(y, oof, BEHAVIOR, pc_path)

    from sklearn.metrics import f1_score
    ths = np.linspace(0.05, 0.99, 95)
    f1s = [f1_score(y, (oof >= t).astype(int), zero_division=0)
              for t in ths]
    best_t = float(ths[int(np.argmax(f1s))])
    print(f'      best threshold = {best_t:.2f}  F1 = {max(f1s):.3f}')

    print(f'[5/6] panel D + per-video bars + panel B…')
    final = train_full_model(X, y)
    plot_shap_panel(final, X, BEHAVIOR,
                       os.path.join(PLOTS_BASE,
                                       f'PixelPaws_{BEHAVIOR}_panelD_shap.png'),
                       n_top=12, sample_n=8000)
    plot_per_video_metrics(y, oof, threshold=best_t,
                              behavior=BEHAVIOR,
                              out_path=os.path.join(PLOTS_BASE,
                                                       f'PixelPaws_{BEHAVIOR}_perVideo_metrics.png'))
    lc = learning_curve(X, y,
                          fractions=[0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00],
                          n_folds=3, n_estimators=150)
    plot_learning_curve(lc, BEHAVIOR,
                          os.path.join(PLOTS_BASE,
                                          f'PixelPaws_{BEHAVIOR}_panelB_learning.png'))

    print(f'[6/6] LOVO + per-session diagnostics…')
    oof_lovo, per_sess_f1 = lovo_oof(X, y, n_estimators=150)
    print(f'      LOVO per-sess F1@0.5: {per_sess_f1}')
    f1s_lovo = [f1_score(y, (oof_lovo >= t).astype(int), zero_division=0)
                   for t in ths]
    best_t_lovo = float(ths[int(np.argmax(f1s_lovo))])
    print(f'      LOVO best threshold = {best_t_lovo:.2f}  '
          f'F1 = {max(f1s_lovo):.3f}')

    plot_per_session_diagnostics(y, oof_lovo, threshold=best_t_lovo,
                                    behavior=BEHAVIOR,
                                    out_path=os.path.join(PLOTS_BASE,
                                                             f'PixelPaws_{BEHAVIOR}_perSession_diagnostics.png'))
    plot_per_video_metrics(y, oof_lovo, threshold=best_t_lovo,
                              behavior=BEHAVIOR,
                              out_path=os.path.join(PLOTS_BASE,
                                                       f'PixelPaws_{BEHAVIOR}_perVideo_metrics_LOVO.png'))

    pd.to_pickle({'oof': oof, 'oof_lovo': oof_lovo, 'fold_f1': fold_f1,
                    'lc': lc, 'best_t': best_t, 'best_t_lovo': best_t_lovo,
                    'per_sess_f1': per_sess_f1, 'splits': splits,
                    'final_model': final},
                   os.path.join(PLOTS_BASE, '_regen_cache.pkl'))

    print(f'\n✓ done. Outputs → {PLOTS_BASE}')


if __name__ == '__main__':
    main()
