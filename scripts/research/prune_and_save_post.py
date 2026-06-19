"""prune_and_save_post.py — Post-processing step: load a regen run's
cache + training set, compute SHAP-based feature importance, prune to
top-N, and save both AllFeatures + pruned classifiers in
PixelPaws-compatible .pkl format.

Usage:
    py prune_and_save_post.py <cache_dir> <behavior> [<classifier_dir>]

Examples:
    py prune_and_save_post.py \
        E:/RSVIDS/.../regen_20260429_140020 Scratching
    py prune_and_save_post.py \
        E:/RSVIDS/.../regen_aug_20260429_154500 L_flinching
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                # peer scripts in this folder
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # project root (E:/PixelPaws)
from regen_scratching_plots import prune_and_save_classifier


def main():
    if len(sys.argv) < 3:
        sys.exit('usage: prune_and_save_post.py <cache_dir> '
                  '<behavior> [<classifier_dir>]')
    cache_dir = sys.argv[1]
    behavior  = sys.argv[2]
    # Default classifier dir = parent of cache_dir's "plots/" → "classifiers"
    if len(sys.argv) >= 4:
        classifier_dir = sys.argv[3]
    else:
        # cache_dir is .../classifiers/plots/regen_xxx → up two levels
        classifier_dir = os.path.dirname(os.path.dirname(cache_dir))

    train_set_pkl = os.path.join(cache_dir, '_training_set.pkl')
    regen_cache_pkl = os.path.join(cache_dir, '_regen_cache.pkl')

    if not os.path.isfile(train_set_pkl):
        # Some scratching runs use the project's training_data
        # location instead of caching it inside plots/regen_*
        train_set_pkl = None

    if not os.path.isfile(regen_cache_pkl):
        sys.exit(f'no regen cache at {regen_cache_pkl}')

    print(f'[load] regen cache: {regen_cache_pkl}')
    cache = pd.read_pickle(regen_cache_pkl)
    final = cache.get('final_model')
    oof   = np.asarray(cache.get('oof'))
    fold_f1 = cache.get('fold_f1', [])
    splits  = cache.get('splits', [])
    if final is None:
        sys.exit('no final_model in cache')

    if train_set_pkl:
        print(f'[load] training set: {train_set_pkl}')
        td = pd.read_pickle(train_set_pkl)
        X, y = td['X'], np.asarray(td['y'], dtype=np.int64)
    else:
        # Fallback: scratching project keeps training data elsewhere
        scratching_pkl = (
            r'E:/RSVIDS/Blackbox/2510_Blackbox_Rimonabant/'
            r'Blackbox_videos-selected/Classifiers/training_data/'
            r'Scratching_train_set.pkl')
        td = pd.read_pickle(scratching_pkl)
        X, y = td['X'], np.asarray(td['y'], dtype=np.int64)

    training_sessions = [s for s, _ in splits] if splits else []
    if not training_sessions:
        # If splits weren't saved, try to recover from project metadata
        training_sessions = ['(unknown)']

    print(f'[shapes] X={X.shape}  y_pos={y.sum()}  oof={oof.shape}')
    print(f'[sessions] {len(training_sessions)} sessions')

    result = prune_and_save_classifier(
        final, X, y, oof, fold_f1, training_sessions,
        behavior=behavior,
        classifier_dir=classifier_dir,
        n_top=120, sample_n=8000)

    print(f'\n✓ done.')
    print(f'  AllFeatures : {result["all_features_path"]}')
    print(f'  pruned_120  : {result["pruned_path"]}')
    print(f'  json        : {result["json_path"]}')
    print(f'  best F1     : {result["best_f1"]:.3f} '
          f'@ thresh={result["best_thresh"]:.2f}')


if __name__ == '__main__':
    main()
