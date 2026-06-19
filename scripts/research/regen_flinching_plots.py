"""regen_flinching_plots.py — train an XGBoost flinching classifier
on the 2511_Blackbox_Formalin/PixelPaws/Flinching project (17 sessions
with both labels and feature caches after the BORIS auto-conversion)
and produce the plasma-coloured panel suite:

  Panel C : Threshold curves (Precision / Recall / F1 vs threshold)
  Panel D : Bulbous SHAP beeswarm + importance bar chart
  Panel B : Learning curve (F1 vs bout-positive frames)
  + Per-video bar plot (P / R / F1 by session, in-fold)
  + Per-session diagnostic panel (raster + confusion + sec/bin heatmap, LOVO)

No Optuna. Uses the raw feature cache columns directly (skips the
post-augmentation pipeline that the GUI's `_real_training` adds —
simpler / faster and still a solid baseline).

Run:
    py regen_flinching_plots.py
"""

from __future__ import annotations

import os
import sys
import time
import glob
from datetime import datetime

import numpy as np
import pandas as pd

# Reuse plotting helpers from the scratching script — they're behavior-agnostic
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                # peer scripts in this folder
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # project root (E:/PixelPaws)
from regen_scratching_plots import (
    cv_oof_proba, train_full_model, learning_curve, lovo_oof,
    plot_threshold_curves, plot_shap_panel, plot_learning_curve,
    plot_per_video_metrics, plot_per_session_diagnostics,
    _make_xgb,   # exposed for any custom training
    PLASMA, GRID_GREY, TEXT_GREY,
)
from regen_scratching_plots import SESSION_SPLITS as _SCRATCHING_SPLITS  # not used; just in case
import regen_scratching_plots as _rsp


PROJECT     = r'E:/RSVIDS/Blackbox/2511_Blackbox_Formalin/PixelPaws/Flinching'
BEHAVIOR    = 'L_flinching'
LABEL_COL   = 'L_flinching'           # column name inside *_labels.csv
FEATURE_HASH = '0b787aea'              # newer cache hash (599 raw cols)
PLOTS_BASE  = os.path.join(PROJECT, 'classifiers', 'plots',
                           f'regen_{datetime.now():%Y%m%d_%H%M%S}')
N_FOLDS     = 5
RANDOM_SEED = 42


def _scan_sessions() -> list[tuple[str, str, str]]:
    """Return ``[(session_name, labels_csv_path, feature_cache_path)…]``
    for every session in the project that has BOTH a `*_labels.csv` and
    a feature-cache `*_features_<FEATURE_HASH>.pkl`. Sessions sorted by
    name."""
    labels_dir   = os.path.join(PROJECT, 'behavior_labels')
    features_dir = os.path.join(PROJECT, 'features')
    labels = {f.replace('_labels.csv', ''): os.path.join(labels_dir, f)
                for f in os.listdir(labels_dir) if f.endswith('_labels.csv')}
    feats  = {}
    for f in os.listdir(features_dir):
        if f.endswith(f'_features_{FEATURE_HASH}.pkl'):
            sname = f.replace(f'_features_{FEATURE_HASH}.pkl', '')
            feats[sname] = os.path.join(features_dir, f)
    out = []
    for s in sorted(set(labels) & set(feats)):
        out.append((s, labels[s], feats[s]))
    return out


def build_training_set(sessions: list[tuple[str, str, str]]
                          ) -> tuple[pd.DataFrame, np.ndarray, list[tuple[str, int]]]:
    """Concat features + labels across sessions. Returns (X, y, splits)
    where splits = [(session_name, n_rows), …] in concat order."""
    X_parts = []
    y_parts = []
    splits = []
    common_cols = None

    for sname, label_path, feat_path in sessions:
        feat = pd.read_pickle(feat_path)
        lab  = pd.read_csv(label_path)
        if LABEL_COL not in lab.columns:
            print(f'  ⚠ {sname}: column {LABEL_COL!r} not in labels — '
                  f'skipping (cols={list(lab.columns)[:5]})')
            continue
        n = min(len(feat), len(lab))
        X_s = feat.iloc[:n].reset_index(drop=True)
        y_s = lab[LABEL_COL].iloc[:n].astype(int).values
        if common_cols is None:
            common_cols = list(X_s.columns)
        else:
            # Intersect; reorder to the master list
            common_cols = [c for c in common_cols if c in X_s.columns]
        X_parts.append(X_s)
        y_parts.append(y_s)
        splits.append((sname, n))
        print(f'  {sname:30s}  features={X_s.shape}  labels n={n}  '
              f'pos={y_s.sum()} ({y_s.mean()*100:.2f}%)')

    # Apply the consensus column intersection so every session contributes
    # the same feature set
    X_parts = [df[common_cols] for df in X_parts]
    X = pd.concat(X_parts, ignore_index=True)
    y = np.concatenate(y_parts)
    return X, y, splits


def main():
    os.makedirs(PLOTS_BASE, exist_ok=True)
    print(f'[1/6] scanning for sessions with labels + features…')
    sessions = _scan_sessions()
    print(f'      {len(sessions)} sessions usable')
    if not sessions:
        sys.exit('no sessions to train on')

    print(f'[2/6] building training set from feature caches…')
    X, y, splits = build_training_set(sessions)
    # Drop any non-finite columns (rare — usually NaN from edge frames)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    print(f'      X={X.shape}   y_pos={y.sum()} ({y.mean()*100:.2f}%)')
    print(f'      session splits sum check: '
          f'{sum(n for _, n in splits)} == {len(y)}')

    # Persist training set so future plot iterations can skip the build
    train_pkl = os.path.join(PLOTS_BASE, '_training_set.pkl')
    pd.to_pickle({'X': X, 'y': y, 'splits': splits}, train_pkl)
    print(f'      cached → {train_pkl}')

    # Patch the scratching-script's SESSION_SPLITS so its per-session
    # helpers use OUR splits instead of the Rimonabant ones.
    _rsp.SESSION_SPLITS = [(s.replace('260129_Formalin_', 'F_')
                                  .replace('251126_Formalin_F_', 'FS_')
                                  .replace('_Cut', ''), n)
                              for s, n in splits]
    print(f'      patched SESSION_SPLITS for plot helpers')

    print(f'[3/6] {N_FOLDS}-fold OOF CV (re-training)…')
    oof, fold_f1 = cv_oof_proba(X, y, n_folds=N_FOLDS, n_estimators=250)
    print(f'      mean F1 (per-fold) = {np.mean(fold_f1):.3f}±'
          f'{np.std(fold_f1):.3f}')

    print(f'[4/6] panel C — threshold curves…')
    pc_path = os.path.join(PLOTS_BASE,
                              f'PixelPaws_{BEHAVIOR}_panelC_threshold.png')
    plot_threshold_curves(y, oof, BEHAVIOR, pc_path)
    print(f'      → {pc_path}')

    # Best threshold from full OOF (used by per-video + diagnostics)
    from sklearn.metrics import f1_score
    ths = np.linspace(0.05, 0.99, 95)
    f1s = [f1_score(y, (oof >= t).astype(int), zero_division=0)
              for t in ths]
    best_t = float(ths[int(np.argmax(f1s))])
    print(f'      best threshold = {best_t:.2f}  F1 = {max(f1s):.3f}')

    print(f'[5a/6] training final model on full set for SHAP…')
    final = train_full_model(X, y)
    pd_path = os.path.join(PLOTS_BASE,
                              f'PixelPaws_{BEHAVIOR}_panelD_shap.png')
    plot_shap_panel(final, X, BEHAVIOR, pd_path,
                       n_top=12, sample_n=8000)
    print(f'      → {pd_path}')

    print(f'[5b/6] per-video bar plot (in-fold OOF)…')
    perv_path = os.path.join(PLOTS_BASE,
                                f'PixelPaws_{BEHAVIOR}_perVideo_metrics.png')
    plot_per_video_metrics(y, oof, threshold=best_t,
                              behavior=BEHAVIOR, out_path=perv_path)
    print(f'      → {perv_path}')

    print(f'[5c/6] panel B — learning curve…')
    lc = learning_curve(X, y,
                          fractions=[0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00],
                          n_folds=3, n_estimators=150)
    pb_path = os.path.join(PLOTS_BASE,
                              f'PixelPaws_{BEHAVIOR}_panelB_learning.png')
    plot_learning_curve(lc, BEHAVIOR, pb_path)
    print(f'      → {pb_path}')

    print(f'[6/6] LOVO CV + per-session diagnostics…')
    oof_lovo, per_sess_f1 = lovo_oof(X, y, n_estimators=150)
    print(f'      per-session LOVO F1@0.5: {per_sess_f1}')

    # LOVO best threshold
    f1s_lovo = [f1_score(y, (oof_lovo >= t).astype(int), zero_division=0)
                   for t in ths]
    best_t_lovo = float(ths[int(np.argmax(f1s_lovo))])
    print(f'      LOVO best threshold = {best_t_lovo:.2f}  F1 = '
          f'{max(f1s_lovo):.3f}')

    diag_path = os.path.join(PLOTS_BASE,
                                f'PixelPaws_{BEHAVIOR}_perSession_diagnostics.png')
    plot_per_session_diagnostics(y, oof_lovo, threshold=best_t_lovo,
                                    behavior=BEHAVIOR,
                                    out_path=diag_path)
    print(f'      → {diag_path}')

    perv_lovo_path = os.path.join(PLOTS_BASE,
                                      f'PixelPaws_{BEHAVIOR}_perVideo_metrics_LOVO.png')
    plot_per_video_metrics(y, oof_lovo, threshold=best_t_lovo,
                              behavior=BEHAVIOR, out_path=perv_lovo_path)
    print(f'      → {perv_lovo_path}')

    # Cache OOFs for future iteration
    cache_path = os.path.join(PLOTS_BASE, '_regen_cache.pkl')
    pd.to_pickle({'oof': oof, 'oof_lovo': oof_lovo, 'fold_f1': fold_f1,
                    'lc': lc, 'best_t': best_t, 'best_t_lovo': best_t_lovo,
                    'per_sess_f1': per_sess_f1, 'splits': splits,
                    'final_model': final}, cache_path)
    print(f'      cache → {cache_path}')

    print(f'\n✓ done. Outputs → {PLOTS_BASE}')


if __name__ == '__main__':
    main()
