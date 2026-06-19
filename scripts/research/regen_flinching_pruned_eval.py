"""regen_flinching_pruned_eval.py — Run the same OOF + LOVO + plot
suite as the augmented pipeline but using only the SHAP-pruned top-120
feature subset. Verifies whether the pruned classifier matches the
AllFeatures one on cross-animal generalization (often it does — and
predict-time is ~7× faster)."""

from __future__ import annotations

import os
import sys
import json
import time
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
    prune_and_save_classifier,
)
import regen_scratching_plots as _rsp

PROJECT     = r'E:/RSVIDS/Blackbox/2511_Blackbox_Formalin/PixelPaws/Flinching'
BEHAVIOR    = 'L_flinching'
PLOTS_BASE  = os.path.join(PROJECT, 'classifiers', 'plots',
                           f'regen_aug_pruned_{datetime.now():%Y%m%d_%H%M%S}')
PRIOR_RUN   = os.path.join(PROJECT, 'classifiers', 'plots',
                           'regen_aug_20260429_153027')
JSON_PATH   = os.path.join(PROJECT, 'classifiers',
                           'PixelPaws_L_flinching_pruned_120_20260429_165631.json')
N_FOLDS     = 5


def main():
    os.makedirs(PLOTS_BASE, exist_ok=True)
    print(f'[1/4] loading prior augmented training set + top-120 list…')
    td = pd.read_pickle(os.path.join(PRIOR_RUN, '_training_set.pkl'))
    X_full, y, splits = td['X'], np.asarray(td['y'], dtype=np.int64), td['splits']
    print(f'      X_full={X_full.shape}  y_pos={y.sum()}  n_sessions={len(splits)}')

    j = json.load(open(JSON_PATH))
    top120 = j['top_features']
    # Filter X to top-120 (preserving training column order)
    keep = [c for c in top120 if c in X_full.columns]
    X = X_full[keep]
    print(f'      X_pruned={X.shape}  (kept {len(keep)} of {len(top120)} top features)')

    # Patch SESSION_SPLITS so plot helpers split by our 17 sessions
    _rsp.SESSION_SPLITS = [(s.replace('260129_Formalin_', 'F_')
                                  .replace('251126_Formalin_F_', 'FS_')
                                  .replace('_Cut', ''), n)
                              for s, n in splits]

    print(f'\n[2/4] {N_FOLDS}-fold OOF on pruned features…')
    t0 = time.time()
    oof, fold_f1 = cv_oof_proba(X, y, n_folds=N_FOLDS, n_estimators=300)
    print(f'      mean F1@0.5 (per-fold) = {np.mean(fold_f1):.3f}±'
          f'{np.std(fold_f1):.3f}   (took {(time.time()-t0)/60:.1f} min)')

    # Best threshold
    from sklearn.metrics import f1_score
    ths = np.linspace(0.05, 0.99, 95)
    f1s = [f1_score(y, (oof >= t).astype(int), zero_division=0) for t in ths]
    best_t = float(ths[int(np.argmax(f1s))])
    best_f1 = max(f1s)
    print(f'      pruned OOF best threshold = {best_t:.2f}  F1 = {best_f1:.3f}')

    plot_threshold_curves(y, oof, BEHAVIOR,
                              os.path.join(PLOTS_BASE,
                                              f'PixelPaws_{BEHAVIOR}_pruned_panelC_threshold.png'))
    plot_per_video_metrics(y, oof, threshold=best_t,
                              behavior=f'{BEHAVIOR}_pruned',
                              out_path=os.path.join(PLOTS_BASE,
                                                       f'PixelPaws_{BEHAVIOR}_pruned_perVideo_metrics.png'))

    print(f'\n[3/4] LOVO on pruned features…')
    t0 = time.time()
    oof_lovo, per_sess_f1 = lovo_oof(X, y, n_estimators=200)
    f1s_lovo = [f1_score(y, (oof_lovo >= t).astype(int), zero_division=0) for t in ths]
    best_t_lovo = float(ths[int(np.argmax(f1s_lovo))])
    best_f1_lovo = max(f1s_lovo)
    print(f'      pruned LOVO best threshold = {best_t_lovo:.2f}  '
          f'F1 = {best_f1_lovo:.3f}   (took {(time.time()-t0)/60:.1f} min)')

    plot_per_session_diagnostics(y, oof_lovo, threshold=best_t_lovo,
                                    behavior=f'{BEHAVIOR}_pruned',
                                    out_path=os.path.join(PLOTS_BASE,
                                                             f'PixelPaws_{BEHAVIOR}_pruned_perSession_diagnostics.png'))
    plot_per_video_metrics(y, oof_lovo, threshold=best_t_lovo,
                              behavior=f'{BEHAVIOR}_pruned',
                              out_path=os.path.join(PLOTS_BASE,
                                                       f'PixelPaws_{BEHAVIOR}_pruned_perVideo_metrics_LOVO.png'))

    print(f'\n[4/4] panel B (learning curve) on pruned features…')
    lc = learning_curve(X, y,
                          fractions=[0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00],
                          n_folds=3, n_estimators=200)
    plot_learning_curve(lc, BEHAVIOR,
                          os.path.join(PLOTS_BASE,
                                          f'PixelPaws_{BEHAVIOR}_pruned_panelB_learning.png'))

    # Final model + SHAP panel on the pruned subset
    print(f'\n[5/4] final pruned model + SHAP panel…')
    final = train_full_model(X, y)
    plot_shap_panel(final, X, f'{BEHAVIOR}_pruned',
                       os.path.join(PLOTS_BASE,
                                       f'PixelPaws_{BEHAVIOR}_pruned_panelD_shap.png'),
                       n_top=12, sample_n=8000)

    # Cache for re-iteration
    pd.to_pickle({'oof': oof, 'oof_lovo': oof_lovo,
                    'fold_f1': fold_f1, 'lc': lc,
                    'best_t': best_t, 'best_t_lovo': best_t_lovo,
                    'best_f1': best_f1, 'best_f1_lovo': best_f1_lovo,
                    'per_sess_f1': per_sess_f1, 'splits': splits,
                    'final_model': final},
                   os.path.join(PLOTS_BASE, '_regen_cache.pkl'))

    # Summary
    print('\n' + '=' * 60)
    print(f'Pruned (120 features) summary:')
    print(f'  OOF best F1   = {best_f1:.3f} @ thresh={best_t:.2f}')
    print(f'  LOVO best F1  = {best_f1_lovo:.3f} @ thresh={best_t_lovo:.2f}')
    print('Compare AllFeatures (899 features):')
    print(f'  OOF best F1   = 0.852 @ thresh=0.86')
    print(f'  LOVO best F1  = 0.597 @ thresh=0.82')
    print('=' * 60)
    print(f'\n✓ done. Outputs → {PLOTS_BASE}')


if __name__ == '__main__':
    main()
