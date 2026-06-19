"""
Habby Wang's AM_ChR2_stim_SPB scratching analysis pipeline.

Within-subject design: 4 mice (FED2-F4, F5, M4, M5) x 2 conditions
(nostim baseline Mar-10 vs 20Hz stim Mar-12). All videos 60 fps,
1280x720, DLC palmreader shuffle 9 (snapshot best-110, vs our
best-190 - same body parts).

Single feature extraction per video (cached to project features/),
then runs both Scratching classifiers (pruned + AllFeatures).
Apply each classifier's saved post-processing
(best_thresh=0.6, min_bout=12, min_after_bout=1).

Outputs:
  results/<session>_predictions.csv      per-frame proba + 0/1
  features/<session>_features_<hash>.pkl GUI-compatible cache
  analysis/scratching_summary.csv        per-session bout/time stats
  analysis/scratching_*.png              per-session + group plots
  Discord uploads of plots when done.

Run:
  PYTHONIOENCODING=utf-8 py -X utf8 scripts/research/habby_scratching_pipeline.py
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

REPO = Path(r'E:\PixelPaws')
sys.path.insert(0, str(REPO))

PROJECT = Path(r'E:\RSVIDS\Blackbox\AM_ChR2_stim_SPB_analysis')
VIDEO_DIR = PROJECT / 'Videos'
CLF_DIR = PROJECT / 'classifiers'
RESULTS_DIR = PROJECT / 'results'
FEATURES_DIR = PROJECT / 'features'
ANALYSIS_DIR = PROJECT / 'analysis'
for d in (RESULTS_DIR, FEATURES_DIR, ANALYSIS_DIR):
    d.mkdir(parents=True, exist_ok=True)

WEBHOOK = (
    ""
)

CLASSIFIERS = [
    'PixelPaws_Scratching_AllFeatures.pkl',
    'PixelPaws_Scratching.pkl',  # pruned variant (sometimes has different post-proc)
]


def step(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def parse_session(name: str):
    """Pull mouse_id (e.g. F4, M5), condition (nostim|stim), date from
    filenames like 20260310_RTPP2_4wCCI_nostim_FED2_F4_19-36-25.

    Note one file has '20263012_' (typo Habby's date), still parses cond
    + mouse correctly.
    """
    m = re.search(r'_(nostim|stim)_FED2_(F\d|M\d)_', name)
    if not m:
        return None
    cond = 'nostim' if m.group(1) == 'nostim' else 'stim'
    mouse = m.group(2)
    return mouse, cond


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


def post_to_discord(text: str, png_paths) -> None:
    import subprocess
    cmd = ['curl', '-s', '-o', os.devnull, '-w', '%{http_code}', '-X', 'POST',
           '-F', f'payload_json={{"content":"{text}"}}']
    for i, p in enumerate(png_paths):
        cmd += ['-F', f'file{i}=@{p}']
    cmd.append(WEBHOOK)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f'Discord upload: HTTP {r.stdout}')


def main() -> int:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from prediction_pipeline import (
        PixelPaws_ExtractFeatures,
        predict_with_xgboost,
        augment_features_post_cache,
    )

    pairs = []
    for v in sorted(VIDEO_DIR.glob('*.mp4')):
        h5 = list(v.parent.glob(f'{v.stem}*shuffle9*.h5'))
        if h5:
            pairs.append((v, h5[0]))
        else:
            step(f'  ! no h5 for {v.name}')
    if not pairs:
        step('No (video, h5) pairs found')
        return 1
    step(f'Sessions: {len(pairs)}')

    # Feature-cache hash (GUI-compatible). Scratching classifier was
    # trained WITH optical flow, so include those features here.
    try:
        from feature_cache import FeatureCacheManager
        cfg = {
            'bp_include_list': None,
            'bp_pixbrt_list': ['hrpaw', 'hlpaw', 'snout'],
            'square_size': [40, 40, 40],
            'pix_threshold': 0.3,
            'include_optical_flow': True,
            'bp_optflow_list': ['hrpaw', 'hlpaw', 'snout'],
        }
        cfg_hash = FeatureCacheManager.compute_hash(cfg)
    except Exception:
        cfg_hash = 'manual'
    step(f'feature-cache hash: {cfg_hash}')

    # Load classifiers
    loaded = []
    for fname in CLASSIFIERS:
        p = CLF_DIR / fname
        if not p.is_file():
            step(f'  ! missing classifier: {fname}')
            continue
        with open(p, 'rb') as f:
            cd = pickle.load(f)
        loaded.append({'name': fname.replace('.pkl', ''), 'clf_data': cd})
    step(f'Classifiers: {len(loaded)}')

    summary_rows = []
    bout_rows = []
    for v, h5 in pairs:
        base = v.stem
        parsed = parse_session(base)
        if not parsed:
            step(f'  ! cannot parse session: {base}'); continue
        mouse, cond = parsed
        step(f'\n=== {mouse} / {cond} ({base}) ===')

        feat_cache = FEATURES_DIR / f'{base}_features_{cfg_hash}.pkl'
        if feat_cache.is_file():
            step('  loading cached features')
            with open(feat_cache, 'rb') as f:
                X = pickle.load(f)
        else:
            step('  extracting features')
            t0 = time.time()
            X = PixelPaws_ExtractFeatures(
                pose_data_file=str(h5),
                video_file_path=str(v),
                bp_include_list=None,
                bp_pixbrt_list=['hrpaw', 'hlpaw', 'snout'],
                square_size=[40, 40, 40],
                pix_threshold=0.3,
                config_yaml_path=None,
                include_optical_flow=True,
                bp_optflow_list=['hrpaw', 'hlpaw', 'snout'],
            )
            with open(feat_cache, 'wb') as f:
                pickle.dump(X, f)
            step(f'  cached ({time.time()-t0:.1f}s, X={X.shape})')

        # Run each classifier
        per_video = {'frame': range(len(X))}
        for L in loaded:
            cd = L['clf_data']
            model = cd['clf_model']
            best_thresh = float(cd.get('best_thresh', 0.5))
            min_bout = int(cd.get('min_bout', 1))
            min_after_bout = int(cd.get('min_after_bout', 0))

            X_aug = augment_features_post_cache(X.copy(), cd, model, str(h5))
            y_proba = predict_with_xgboost(
                model, X_aug,
                calibrator=cd.get('prob_calibrator'),
                fold_models=cd.get('fold_models'),
            )
            y_raw = (y_proba >= best_thresh).astype(int)
            y_post = apply_postprocess(y_raw, min_bout, min_after_bout, 0)
            per_video[f'{L["name"]}_proba'] = y_proba
            per_video[f'{L["name"]}'] = y_post

            n = len(y_post)
            n_pos = int(y_post.sum())
            bouts = find_bouts(y_post)
            mean_bout = (np.mean([b[1]-b[0]+1 for b in bouts]) / 60
                         if bouts else 0)
            step(f'  {L["name"]:42}  thresh={best_thresh:.2f} '
                 f'min_bout={min_bout:>2}  pos={n_pos:>5} ({100*n_pos/n:5.2f}%)  '
                 f'bouts={len(bouts):>3}  mean_bout_s={mean_bout:.2f}')

            summary_rows.append({
                'session': base, 'mouse': mouse, 'condition': cond,
                'classifier': L['name'],
                'best_thresh': best_thresh, 'min_bout': min_bout,
                'n_frames': n, 'n_pos': n_pos,
                'pct_time': round(100 * n_pos / n, 3),
                'time_sec': round(n_pos / 60.0, 2),
                'n_bouts': len(bouts),
                'mean_bout_sec': round(mean_bout, 3),
                'median_bout_sec': round(float(np.median([b[1]-b[0]+1 for b in bouts]) / 60)
                                          if bouts else 0, 3),
            })
            for s, e in bouts:
                bout_rows.append({
                    'session': base, 'mouse': mouse, 'condition': cond,
                    'classifier': L['name'],
                    'start_frame': s, 'end_frame': e,
                    'length_frames': e - s + 1,
                    'length_sec': round((e - s + 1) / 60, 3),
                    'start_sec': round(s / 60, 2),
                })

        out_csv = RESULTS_DIR / f'{base}_predictions.csv'
        pd.DataFrame(per_video).to_csv(out_csv, index=False)
        step(f'  -> {out_csv.name}')

    # Save summaries
    summary_df = pd.DataFrame(summary_rows)
    bout_df = pd.DataFrame(bout_rows)
    summary_df.to_csv(ANALYSIS_DIR / 'scratching_summary.csv', index=False)
    bout_df.to_csv(ANALYSIS_DIR / 'scratching_bouts.csv', index=False)
    step('\nSaved scratching_summary.csv + scratching_bouts.csv')

    # Plots: focus on the AllFeatures classifier (more features = generally
    # more confident). Loop over both for completeness.
    pngs = []
    palette = {'nostim': '#1f77b4', 'stim': '#d62728'}
    for clf_name in summary_df['classifier'].unique():
        sub = summary_df[summary_df['classifier'] == clf_name]
        if sub.empty:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        # 1. % time scratching paired
        ax = axes[0]
        for mouse, mdf in sub.groupby('mouse'):
            no = mdf[mdf['condition'] == 'nostim']['pct_time']
            st = mdf[mdf['condition'] == 'stim']['pct_time']
            if len(no) and len(st):
                ax.plot([0, 1], [no.iloc[0], st.iloc[0]], 'o-',
                        color='gray', alpha=0.7)
                ax.text(1.02, st.iloc[0], f'{mouse}', fontsize=9, va='center')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['nostim', 'stim'])
        ax.set_ylabel('% time scratching')
        ax.set_title('Within-subject: % time scratching')
        ax.grid(axis='y', alpha=0.3)
        # 2. # bouts paired
        ax = axes[1]
        for mouse, mdf in sub.groupby('mouse'):
            no = mdf[mdf['condition'] == 'nostim']['n_bouts']
            st = mdf[mdf['condition'] == 'stim']['n_bouts']
            if len(no) and len(st):
                ax.plot([0, 1], [no.iloc[0], st.iloc[0]], 'o-',
                        color='gray', alpha=0.7)
                ax.text(1.02, st.iloc[0], f'{mouse}', fontsize=9, va='center')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['nostim', 'stim'])
        ax.set_ylabel('scratching bouts')
        ax.set_title('Within-subject: bout count')
        ax.grid(axis='y', alpha=0.3)
        # 3. mean bout length paired
        ax = axes[2]
        for mouse, mdf in sub.groupby('mouse'):
            no = mdf[mdf['condition'] == 'nostim']['mean_bout_sec']
            st = mdf[mdf['condition'] == 'stim']['mean_bout_sec']
            if len(no) and len(st):
                ax.plot([0, 1], [no.iloc[0], st.iloc[0]], 'o-',
                        color='gray', alpha=0.7)
                ax.text(1.02, st.iloc[0], f'{mouse}', fontsize=9, va='center')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['nostim', 'stim'])
        ax.set_ylabel('mean bout length (s)')
        ax.set_title('Within-subject: mean bout length')
        ax.grid(axis='y', alpha=0.3)
        fig.suptitle(f'Scratching - {clf_name}', fontsize=12, fontweight='bold')
        fig.tight_layout()
        png = ANALYSIS_DIR / f'scratching_{clf_name}.png'
        fig.savefig(png, dpi=140); plt.close(fig)
        pngs.append(png)

    # Headline summary text
    af = summary_df[summary_df['classifier'].str.contains('AllFeatures', case=False)]
    lines = ['Scratching analysis (AM_ChR2_stim) complete.']
    lines.append('Per mouse (AllFeatures classifier):')
    for mouse, mdf in af.groupby('mouse'):
        no = mdf[mdf['condition'] == 'nostim']
        st = mdf[mdf['condition'] == 'stim']
        if len(no) and len(st):
            lines.append(
                f'  {mouse}:  nostim={no.iloc[0]["pct_time"]:.2f}% '
                f'({no.iloc[0]["n_bouts"]} bouts) -> stim='
                f'{st.iloc[0]["pct_time"]:.2f}% ({st.iloc[0]["n_bouts"]} bouts)'
            )
    summary_text = '\\n'.join(lines).replace('"', '')
    post_to_discord(summary_text, pngs)
    step(f'Posted to Discord: {len(pngs)} figures')
    return 0


if __name__ == '__main__':
    sys.exit(main())
