"""
Export per-frame Habby predictions:
  - <session>_probabilities.csv : frame + 6 *_proba columns
  - <session>_states.csv        : frame + 6 binary state columns + final_state

States are post-processed per each classifier's saved settings
(best_thresh, min_bout, min_after_bout, max_gap). final_state is the
argmax assignment over the 6 binary columns using the same priority
order as the transitions analysis. Frames where no behavior is on get
labeled 'other'.

Reuses Habby's already-cached features (hash 8aed1c22).
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r'E:\PixelPaws')
sys.path.insert(0, str(REPO))

PROJECT = Path(r'E:\RSVIDS\Blackbox\AM_ChR2_stim_SPB_analysis')
VIDEO_DIR = PROJECT / 'Videos'
FEATURES_DIR = PROJECT / 'features'
OUT_DIR = PROJECT / 'csv_exports'
OUT_DIR.mkdir(parents=True, exist_ok=True)

JSON_PATH = Path(r'E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\transitions\for_claude.json')
SNLT_CLF_DIR = Path(r'E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\classifiers')
SCRATCH_CLF = PROJECT / 'classifiers' / 'PixelPaws_Scratching_AllFeatures.pkl'

PRIORITY = ['Facial_grooming', 'Left_licking', 'Scratching',
            'rearing', 'still', 'walking']
OTHER_LABEL = 'other'


def step(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def find_bouts(y):
    bouts = []
    in_b = False; start = 0
    for i, v in enumerate(y):
        if v and not in_b: in_b = True; start = i
        elif not v and in_b: in_b = False; bouts.append((start, i - 1))
    if in_b: bouts.append((start, len(y) - 1))
    return bouts


def apply_postprocess(y_pred, min_bout=1, min_after_bout=0, max_gap=0):
    y = y_pred.astype(int).copy()
    if max_gap > 0:
        for i in range(len(find_bouts(y)) - 1):
            bouts = find_bouts(y)
            if i >= len(bouts) - 1:
                break
            gap = bouts[i + 1][0] - bouts[i][1] - 1
            if 0 < gap <= max_gap:
                y[bouts[i][1] + 1: bouts[i + 1][0]] = 1
    if min_bout > 1:
        for s, e in find_bouts(y):
            if e - s + 1 < min_bout:
                y[s: e + 1] = 0
    if min_after_bout > 0:
        bouts = find_bouts(y)
        for i in range(1, len(bouts)):
            gap = bouts[i][0] - bouts[i - 1][1] - 1
            if gap < min_after_bout:
                s, e = bouts[i]
                y[s: e + 1] = 0
    return y


def parse_session(name):
    m = re.search(r'_(nostim|stim)_FED2_(F\d|M\d)_', name)
    if not m:
        return None
    return m.group(2), ('nostim' if m.group(1) == 'nostim' else 'stim')


def main():
    from prediction_pipeline import predict_with_xgboost, augment_features_post_cache

    # Load classifiers (5 SNLT + Scratching)
    with open(JSON_PATH, 'r') as f:
        spec = json.load(f)
    classifiers = []
    for c in spec['classifiers']:
        local = SNLT_CLF_DIR / Path(c['path']).name
        with open(local, 'rb') as f:
            cd = pickle.load(f)
        classifiers.append({**c, 'clf_data': cd, 'name': Path(c['path']).stem})
    with open(SCRATCH_CLF, 'rb') as f:
        scr_cd = pickle.load(f)
    classifiers.append({
        'best_thresh': float(scr_cd.get('best_thresh', 0.6)),
        'min_bout': int(scr_cd.get('min_bout', 12)),
        'min_after_bout': int(scr_cd.get('min_after_bout', 1)),
        'max_gap': 0,
        'clf_data': scr_cd,
        'name': 'Scratching',
    })
    step(f'Loaded {len(classifiers)} classifiers')

    feat_files = sorted(FEATURES_DIR.glob('*_features_*.pkl'))
    step(f'Sessions: {len(feat_files)}')

    for fc in feat_files:
        m = re.match(r'(.+)_features_[0-9a-f]+\.pkl$', fc.name)
        if not m:
            continue
        sess_base = m.group(1)
        parsed = parse_session(sess_base)
        if not parsed:
            continue
        mouse, cond = parsed
        step(f'\n=== {mouse} / {cond}: {sess_base} ===')

        with open(fc, 'rb') as f:
            X = pickle.load(f)
        n = len(X)

        # Resolve dlc h5 path for augment_features_post_cache
        vids = list(VIDEO_DIR.glob(f'{sess_base}*.mp4'))
        h5s = list(VIDEO_DIR.glob(f'{sess_base}*shuffle9*filtered.h5')) if vids else []
        if not h5s and vids:
            h5s = list(VIDEO_DIR.glob(f'{sess_base}*shuffle9*.h5'))
        dlc_path = str(h5s[0]) if h5s else ''

        proba_cols = {'frame': np.arange(n)}
        state_cols = {'frame': np.arange(n)}

        for L in classifiers:
            cd = L['clf_data']
            model = cd['clf_model']
            best_thresh = float(L.get('best_thresh', cd.get('best_thresh', 0.5)))
            min_bout = int(L.get('min_bout', cd.get('min_bout', 1)))
            min_after_bout = int(L.get('min_after_bout', 0))
            max_gap = int(L.get('max_gap', 0))

            X_aug = augment_features_post_cache(X.copy(), cd, model, dlc_path)
            y_proba = predict_with_xgboost(
                model, X_aug,
                calibrator=cd.get('prob_calibrator'),
                fold_models=cd.get('fold_models'),
            )
            y_raw = (y_proba >= best_thresh).astype(int)
            y_post = apply_postprocess(y_raw, min_bout, min_after_bout, max_gap)

            behavior = cd.get('Behavior_type') or L['name']
            proba_cols[f'{behavior}_proba'] = np.asarray(y_proba, dtype=np.float32)
            state_cols[behavior] = y_post.astype(np.uint8)
            step(f'  {behavior:18}  thresh={best_thresh:.2f} min_bout={min_bout:>2}  '
                 f'pos={int(y_post.sum())} ({100*int(y_post.sum())/n:.2f}%)')

        # Build final_state column via argmax with priority order.
        # Higher priority wins; frames with no behavior on -> OTHER_LABEL.
        final = np.full(n, OTHER_LABEL, dtype=object)
        for behavior in reversed(PRIORITY):
            if behavior in state_cols:
                mask = state_cols[behavior] == 1
                final[mask] = behavior
        state_cols['final_state'] = final

        # Order proba columns by priority
        proba_df = pd.DataFrame({
            'frame': proba_cols['frame'],
            **{f'{b}_proba': proba_cols.get(f'{b}_proba') for b in PRIORITY
               if f'{b}_proba' in proba_cols},
        })
        # Order state columns: priority binaries then final_state
        state_df = pd.DataFrame({
            'frame': state_cols['frame'],
            **{b: state_cols[b] for b in PRIORITY if b in state_cols},
            'final_state': state_cols['final_state'],
        })

        proba_csv = OUT_DIR / f'{sess_base}_probabilities.csv'
        state_csv = OUT_DIR / f'{sess_base}_states.csv'
        proba_df.to_csv(proba_csv, index=False)
        state_df.to_csv(state_csv, index=False)
        step(f'  -> {proba_csv.name}')
        step(f'  -> {state_csv.name}')

    step(f'\nAll exports under: {OUT_DIR}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
