"""
Post-DLC predict for SNTX cohort 2. Per-frame predictions for the 5 SNLT
behaviors using the same JSON spec as the THC withdrawal pipeline.

Single feature-extraction pass per video WITH optical flow.

H5 source preference (--h5-source): 'raw' (default) uses the unfiltered
shuffle9 .h5; 'filtered' uses *_filtered.h5 (filterpredictions output).
Different sources produce different brightness/optical-flow features
(crop centers come from keypoints, so smoothed vs raw keypoints differ),
so outputs are written to separate directories:

  --h5-source=raw       results/, features/
  --h5-source=filtered  results_filtered/, features_filtered/

Run:
  py -X utf8 scripts/research/sntx_cohort2_predict_all.py
  py -X utf8 scripts/research/sntx_cohort2_predict_all.py --h5-source=filtered
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(r'E:\PixelPaws')
sys.path.insert(0, str(REPO))

PROJECT_ROOT = Path(r'E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\videos\2605_Cohort2')
VIDEO_DIR = PROJECT_ROOT

JSON_PATH = Path(r'E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\transitions\for_claude.json')
SHUFFLE = 9

# Use the same classifier set as the THC project (those were trained on the
# same DLC model and feature schema).
LOCAL_CLF_DIR = Path(r'E:\RSVIDS\Blackbox\260506_RS_THC_Withdrawal\classifiers')

WEBHOOK = (
    ""
)
BAR_WIDTH = 24


def step(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def _send(url, payload, method='POST'):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode('utf-8'),
        method=method,
        headers={'Content-Type': 'application/json',
                 'User-Agent': 'PixelPaws-Chain/1.0'},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read()
            return json.loads(body) if body else None
    except Exception as e:
        print(f'  ! Discord {method} failed: {e}', flush=True)
        return None


def discord_create(text):
    resp = _send(WEBHOOK + '?wait=true', {'content': text})
    return resp.get('id') if resp else None


def discord_edit(msg_id, text):
    if not msg_id:
        return
    _send(f'{WEBHOOK}/messages/{msg_id}', {'content': text}, method='PATCH')


def make_bar(cur, total, width=BAR_WIDTH):
    if total <= 0:
        return '░' * width
    filled = max(0, min(width, cur * width // total))
    return '█' * filled + '░' * (width - filled)


def fmt_dur(seconds):
    s = int(seconds)
    if s < 60:
        return f'{s}s'
    m, s = divmod(s, 60)
    if m < 60:
        return f'{m}m{s:02d}s'
    h, m = divmod(m, 60)
    return f'{h}h{m:02d}m'


def progress_text(done, total, t_start, h5_source, current=None, status=None):
    bar = make_bar(done, total)
    pct = 100 * done / total if total else 0
    elapsed = time.time() - t_start
    if done > 0 and done < total:
        rate = elapsed / done
        eta = rate * (total - done)
        eta_str = f'~{fmt_dur(eta)}'
    elif done >= total:
        eta_str = 'done'
    else:
        eta_str = '...'
    lines = [
        f'**SNTX cohort 2 predictions ({h5_source} h5) -- progress**',
        f'`[{bar}] {done}/{total} sessions  ({pct:5.1f}%)`',
    ]
    if current:
        lines.append(f'Current: `{current}`{(" -- " + status) if status else ""}')
    lines.append(f'Elapsed: {fmt_dur(elapsed)} | ETA: {eta_str}')
    return '\n'.join(lines)


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


def apply_postprocess(y_pred, min_bout=1, min_after_bout=0, max_gap=0):
    import numpy as np
    y = y_pred.astype(int).copy()
    if max_gap > 0:
        bouts = find_bouts(y)
        for i in range(len(bouts) - 1):
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


def main() -> int:
    import numpy as np
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument('--h5-source', choices=['raw', 'filtered'], default='raw',
                        help='Which DLC .h5 to use; raw is unfiltered, filtered '
                             'is the filterpredictions output. Default: raw.')
    args = parser.parse_args()
    h5_source = args.h5_source

    if h5_source == 'raw':
        results_dir = PROJECT_ROOT / 'results'
        features_dir = PROJECT_ROOT / 'features'
    else:
        results_dir = PROJECT_ROOT / 'results_filtered'
        features_dir = PROJECT_ROOT / 'features_filtered'
    results_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    with open(JSON_PATH, 'r') as f:
        spec = json.load(f)
    classifiers = spec['classifiers']
    step(f'JSON spec: {len(classifiers)} classifiers, fps={spec.get("fps")}')
    step(f'h5 source: {h5_source}  ->  results={results_dir.name}/, features={features_dir.name}/')

    # Discover videos with the selected DLC h5 type
    pairs = []
    for v in sorted(VIDEO_DIR.glob('*.mp4')):
        if h5_source == 'filtered':
            h5 = list(v.parent.glob(f'{v.stem}*shuffle{SHUFFLE}*_filtered.h5'))
        else:  # raw: shuffle9 .h5 that is NOT _filtered.h5
            h5 = [p for p in v.parent.glob(f'{v.stem}*shuffle{SHUFFLE}*.h5')
                  if not p.name.endswith('_filtered.h5')]
        if h5:
            pairs.append((v, h5[0]))
        else:
            step(f'  ! no DLC {h5_source} h5 for {v.name} — skip')
    if not pairs:
        step(f'No (video, {h5_source} h5) pairs found.')
        return 1
    step(f'Sessions to process: {len(pairs)}')

    # Pre-load classifiers
    loaded = []
    union_bp_pixbrt = []
    union_square_size = [40]
    union_pix_threshold = 0.3
    for c in classifiers:
        local = LOCAL_CLF_DIR / Path(c['path']).name
        clf_path = local if local.is_file() else Path(c['path'])
        with open(clf_path, 'rb') as f:
            cd = pickle.load(f)
        for bp in cd.get('bp_pixbrt_list', []):
            if bp not in union_bp_pixbrt:
                union_bp_pixbrt.append(bp)
        union_square_size = cd.get('square_size', union_square_size)
        union_pix_threshold = cd.get('pix_threshold', union_pix_threshold)
        loaded.append({**c, 'clf_data': cd, 'name': Path(c['path']).stem})
    step(f'Union bp_pixbrt: {union_bp_pixbrt}')

    # Single-pass with-flow extraction (matches THC fix)
    from feature_cache import FeatureCacheManager
    cfg_for_hash = {
        'bp_include_list': None,
        'bp_pixbrt_list': union_bp_pixbrt,
        'square_size': union_square_size,
        'pix_threshold': union_pix_threshold,
        'include_optical_flow': True,
        'bp_optflow_list': ['hrpaw', 'hlpaw', 'snout'],
    }
    cfg_hash = FeatureCacheManager.compute_hash(cfg_for_hash)
    step(f'feature-cache hash (with flow): {cfg_hash}')

    from prediction_pipeline import (
        PixelPaws_ExtractFeatures,
        predict_with_xgboost,
        augment_features_post_cache,
    )

    summary_rows = []
    total_sessions = len(pairs)
    t0_all = time.time()
    msg_id = discord_create(progress_text(0, total_sessions, t0_all, h5_source))
    step(f'Discord progress msg id: {msg_id}')

    for i, (v, h5) in enumerate(pairs):
        base = v.stem
        step(f'\n=== {base} ===')

        discord_edit(msg_id, progress_text(
            i, total_sessions, t0_all, h5_source,
            current=base, status='loading features...',
        ))

        feat_cache = features_dir / f'{base}_features_{cfg_hash}.pkl'
        if feat_cache.is_file():
            step(f'  loading cached features (with flow)...')
            with open(feat_cache, 'rb') as f:
                X = pickle.load(f)
        else:
            step('  extracting features (with optical flow, single pass)...')
            discord_edit(msg_id, progress_text(
                i, total_sessions, t0_all, h5_source,
                current=base, status='extracting features (with flow)...',
            ))
            X = PixelPaws_ExtractFeatures(
                pose_data_file=str(h5),
                video_file_path=str(v),
                bp_include_list=None,
                bp_pixbrt_list=union_bp_pixbrt,
                square_size=union_square_size,
                pix_threshold=union_pix_threshold,
                config_yaml_path=None,
                include_optical_flow=True,
                bp_optflow_list=['hrpaw', 'hlpaw', 'snout'],
            )
            with open(feat_cache, 'wb') as f:
                pickle.dump(X, f)
            step(f'  cached -> {feat_cache.name}')
        step(f'  X shape: {X.shape}')

        discord_edit(msg_id, progress_text(
            i, total_sessions, t0_all, h5_source,
            current=base, status='running 5 classifiers...',
        ))

        out = pd.DataFrame({'frame': range(len(X))})
        n = len(X)
        for L in loaded:
            cd = L['clf_data']
            model = cd['clf_model']
            behavior = cd.get('Behavior_type') or L['name']
            best_thresh = float(L.get('best_thresh', cd.get('best_thresh', 0.5)))
            min_bout = int(L.get('min_bout', cd.get('min_bout', 1)))
            min_after_bout = int(L.get('min_after_bout', 0))
            max_gap = int(L.get('max_gap', 0))

            X_aug = augment_features_post_cache(X.copy(), cd, model, str(h5))
            y_proba = predict_with_xgboost(
                model, X_aug,
                calibrator=cd.get('prob_calibrator'),
                fold_models=cd.get('fold_models'),
            )
            y_raw = (y_proba >= best_thresh).astype(int)
            y_post = apply_postprocess(y_raw, min_bout, min_after_bout, max_gap)
            out[f'{behavior}_proba'] = y_proba
            out[behavior] = y_post
            n_pos = int(y_post.sum())
            bouts = len(find_bouts(y_post))
            step(f'  {behavior:18}  thresh={best_thresh:.2f}  '
                 f'pos={n_pos:>6} ({100*n_pos/n:>5.2f}%)  bouts={bouts}')
            summary_rows.append({
                'session': base, 'behavior': behavior,
                'best_thresh': best_thresh, 'min_bout': min_bout,
                'max_gap': max_gap,
                'n_pos': n_pos, 'pct': round(100 * n_pos / n, 3),
                'bouts': bouts,
            })

        out_csv = results_dir / f'{base}_predictions.csv'
        out.to_csv(out_csv, index=False)
        step(f'  -> {out_csv.name}')
        discord_edit(msg_id, progress_text(
            i + 1, total_sessions, t0_all, h5_source,
            current=base, status='done',
        ))

    summary_df = pd.DataFrame(summary_rows)
    summary_suffix = '' if h5_source == 'raw' else '_filtered'
    summary_csv = PROJECT_ROOT / f'sntx_cohort2_predict_summary{summary_suffix}.csv'
    summary_df.to_csv(summary_csv, index=False)
    step(f'\nSummary: {summary_csv}')
    discord_edit(msg_id, progress_text(total_sessions, total_sessions, t0_all, h5_source))
    return 0


if __name__ == '__main__':
    sys.exit(main())
