"""Empirical sanity check: load a real classifier's saved OOF stream
and confirm the new min_after_bout pass actually changes the resulting
binary prediction sequence vs min_after_bout=0.

Picks the SNLT walking classifier because it has stored OOF probs,
and walking has a high event rate so the refractory pass should bite.
"""

import os
import sys
import joblib
import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from evaluation_tab import _apply_bout_filtering


CLF_PATHS = [
    r'E:/RSVIDS/Blackbox/2603_SNLT_JG/Baseline/classifiers/PixelPaws_walking_AllFeatures.pkl',
    r'E:/RSVIDS/Blackbox/2603_SNLT_JG/Baseline/classifiers/PixelPaws_still_AllFeatures.pkl',
    r'E:/RSVIDS/Blackbox/2603_SNLT_JG/Baseline/classifiers/PixelPaws_Facial_grooming_AllFeatures.pkl',
]


def main():
    any_diff = False
    for path in CLF_PATHS:
        if not os.path.isfile(path):
            print(f'  [skip] {os.path.basename(path)} not found')
            continue
        d = joblib.load(path)
        oof = d.get('oof_proba')
        if oof is None:
            print(f'  [skip] {os.path.basename(path)} has no oof_proba')
            continue
        oof = np.asarray(oof, dtype=float)
        thresh = float(d.get('best_thresh', 0.5))
        min_bout = int(d.get('min_bout', 1))
        max_gap = int(d.get('max_gap', 0))
        ma_stored = int(d.get('min_after_bout', 0))

        y_raw = (oof >= thresh).astype(np.int8)

        # 0: noop branch (matches pre-2026-05-01 behaviour)
        y_noop = _apply_bout_filtering(
            y_raw.copy(), min_bout, 0, max_gap)
        # stored value: what the classifier was tuned with
        y_stored = _apply_bout_filtering(
            y_raw.copy(), min_bout, ma_stored, max_gap)
        # extreme: large refractory to force a difference
        y_strict = _apply_bout_filtering(
            y_raw.copy(), min_bout, max(ma_stored, 30), max_gap)

        diff_stored = int((y_noop != y_stored).sum())
        diff_strict = int((y_noop != y_strict).sum())
        n = len(y_raw)

        print(f'\n{os.path.basename(path)}')
        print(f'  n_frames={n}  thresh={thresh:.2f}  min_bout={min_bout}  '
              f'max_gap={max_gap}  stored min_after={ma_stored}')
        print(f'  positives  raw={y_raw.sum():>7d}  '
              f'after_noop={y_noop.sum():>7d}  '
              f'after_stored={y_stored.sum():>7d}  '
              f'after_strict(30)={y_strict.sum():>7d}')
        print(f'  delta vs noop: stored={diff_stored} frames, '
              f'strict={diff_strict} frames')
        if diff_stored > 0 or diff_strict > 0:
            any_diff = True

    if any_diff:
        print('\n  PASS  min_after_bout produces non-zero deltas on real data.')
        print('         Pre-2026-05-01 these deltas were silently dropped.')
    else:
        print('\n  WARN  no observed differences. Either every classifier has '
              'min_after_bout=0 stored AND no events close enough to bite, '
              'or the implementation is wrong.')


if __name__ == '__main__':
    main()
