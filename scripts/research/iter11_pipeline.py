"""
iter11_pipeline.py — Scratching + 6-behavior transition analysis for the
RTPP_Cohort2_pixelpaws_iter11 videos added 2026-05-19 to
``E:\\Sync\\Blackbox\\AM_ChR2_stim_SPB_analysis\\Videos``.

This is the iter11 sibling of ``habby_scratching_pipeline.py`` +
``habby_transitions.py``. Differences vs. those:

  - Reads session metadata (mouse / condition / cohort / laser side)
    from ``<project>/key_file.csv`` instead of parsing the filename, so
    the iter11 naming (``..._F_LR_stim_...``, ``..._F5_Null_nostim_...``)
    and the two single-mouse ``C57_M_..._PawCapture_*`` videos are all
    handled by the same path.
  - Matches ``*shuffle11*.h5`` DLC files alongside the iter9
    ``*shuffle9*.h5`` ones already in the folder.
  - Restricts to ``cohort == 'iter11'`` so existing iter9 outputs
    (``analysis/scratching_*``, ``analysis/transitions_*``) are not
    overwritten.
  - Writes all outputs under ``analysis/iter11/`` with the same naming
    used by the Habby pipelines so downstream readers stay simple.

Run:
    PYTHONIOENCODING=utf-8 py -X utf8 scripts/research/iter11_pipeline.py
    PYTHONIOENCODING=utf-8 py -X utf8 scripts/research/iter11_pipeline.py --features-only

With --features-only: extracts features for every iter11 + pawcapture
session in the key file and writes them to ``<project>/features/`` —
nothing else. Useful when you just want the cache populated.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

PROJECT = Path(r'E:\Sync\Blackbox\AM_ChR2_stim_SPB_analysis')
VIDEO_DIR = PROJECT / 'Videos'
CLF_DIR = PROJECT / 'classifiers'
FEATURES_DIR = PROJECT / 'features'
RESULTS_DIR = PROJECT / 'results'
ANALYSIS_DIR = PROJECT / 'analysis' / 'iter11'
KEY_FILE = PROJECT / 'key_file.csv'

# SNLT 5-behavior classifier set + local Scratching classifier (already
# copied into this project's classifiers/).
SNLT_CLF_DIR = Path(r'E:\Sync\Blackbox\2603_SNLT_JG\Baseline\classifiers')
SCRATCH_CLF = CLF_DIR / 'PixelPaws_Scratching_AllFeatures.pkl'

# Post-processing per classifier — matches the values used in
# ``habby_transitions.py`` (sourced from SNLT ``for_claude.json``).
SNLT_CLF_SPEC = [
    {'name': 'Facial_grooming', 'pkl': 'PixelPaws_Facial_grooming_AllFeatures.pkl',
     'best_thresh': 0.85, 'min_bout': 15, 'min_after_bout': 0, 'max_gap': 0},
    {'name': 'Left_licking',    'pkl': 'PixelPaws_Left_licking.pkl',
     'best_thresh': 0.53, 'min_bout': 3,  'min_after_bout': 1, 'max_gap': 5},
    {'name': 'rearing',         'pkl': 'PixelPaws_rearing_AllFeatures.pkl',
     'best_thresh': 0.54, 'min_bout': 5,  'min_after_bout': 0, 'max_gap': 0},
    {'name': 'still',           'pkl': 'PixelPaws_still_AllFeatures.pkl',
     'best_thresh': 0.58, 'min_bout': 2,  'min_after_bout': 0, 'max_gap': 10},
    {'name': 'walking',         'pkl': 'PixelPaws_walking_AllFeatures.pkl',
     'best_thresh': 0.56, 'min_bout': 2,  'min_after_bout': 0, 'max_gap': 4},
]

# argmax priority — Scratching slotted after Left_licking, same as
# ``habby_transitions.PRIORITY``.
PRIORITY = ['Facial_grooming', 'Left_licking', 'Scratching',
            'rearing', 'still', 'walking']
NOISE_LABEL = 'noise'
EXCLUDE_NOISE = True
FPS = 60

# Feature config — must match the project's PixelPaws_project.json so
# the cache hash collides with the GUI's cache.
FEAT_CFG = {
    'bp_include_list': None,
    'bp_pixbrt_list': ['hrpaw', 'hlpaw', 'snout'],
    'square_size': [40, 40, 40],
    'pix_threshold': 0.3,
    'include_optical_flow': True,
    'bp_optflow_list': ['hrpaw', 'hlpaw', 'snout'],
}


def step(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ---------- bout / postproc helpers (mirrored from habby_*.py) ----------

def find_bouts(y):
    bouts = []; in_b = False; start = 0
    for i, v in enumerate(y):
        if v and not in_b:
            in_b = True; start = i
        elif not v and in_b:
            in_b = False; bouts.append((start, i - 1))
    if in_b:
        bouts.append((start, len(y) - 1))
    return bouts


def apply_postprocess(y_pred, min_bout=1, min_after_bout=0, max_gap=0):
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


def assign_states(df, priority):
    n = len(df)
    states = np.full(n, NOISE_LABEL, dtype=object)
    for behavior in reversed(priority):
        if behavior in df.columns:
            mask = df[behavior].astype(int).to_numpy() == 1
            states[mask] = behavior
    return states


def state_occupancy(states, all_states):
    n_total = len(states)
    out = {s: 0.0 for s in all_states}
    out[NOISE_LABEL] = 0.0
    vals, counts = np.unique(states, return_counts=True)
    raw = dict(zip(vals, counts))
    n_noise = int(raw.get(NOISE_LABEL, 0))
    n_labeled = n_total - n_noise
    out[NOISE_LABEL] = n_noise / max(1, n_total)
    denom = n_labeled if (EXCLUDE_NOISE and n_labeled > 0) else max(1, n_total)
    for s in all_states:
        out[s] = raw.get(s, 0) / denom
    return out


def transition_matrix(states, all_states):
    if len(states) == 0:
        return pd.DataFrame(0, index=all_states, columns=all_states)
    bouts = [states[0]]
    for s in states[1:]:
        if s != bouts[-1]:
            bouts.append(s)
    if EXCLUDE_NOISE:
        bouts = [b for b in bouts if b != NOISE_LABEL]
    M = pd.DataFrame(0, index=all_states, columns=all_states)
    for a, b in zip(bouts[:-1], bouts[1:]):
        if a in M.index and b in M.columns:
            M.loc[a, b] += 1
    return M


def normalise_rows(M):
    rs = M.sum(axis=1).replace(0, 1)
    return M.div(rs, axis=0)


# ---------- session discovery ----------

def find_h5_for(video: Path) -> Path | None:
    """Locate the DLC h5 for a video. iter11 = shuffle 11, iter9 =
    shuffle 9; try both so this helper works for either cohort."""
    for pat in (f'{video.stem}*shuffle11*filtered.h5',
                f'{video.stem}*shuffle11*.h5',
                f'{video.stem}*shuffle9*filtered.h5',
                f'{video.stem}*shuffle9*.h5'):
        hits = list(video.parent.glob(pat))
        # Skip the unfiltered when a filtered exists at the same path
        hits = [h for h in hits if '_meta.pickle' not in h.name]
        if hits:
            return hits[0]
    return None


def load_key_file() -> pd.DataFrame:
    if not KEY_FILE.is_file():
        raise FileNotFoundError(f'key_file.csv not found at {KEY_FILE}')
    df = pd.read_csv(KEY_FILE)
    df.columns = [c.strip() for c in df.columns]
    # Required columns
    for col in ('Subject', 'Cohort', 'Animal', 'Condition'):
        if col not in df.columns:
            raise KeyError(f'key_file.csv missing required column: {col}')
    return df


# ---------- main ----------

def _extract_one(row, v, h5, cfg_hash):
    """Run feature extraction for a single (row, video, h5) and write the
    cache pkl. Returns the loaded feature matrix X (or None on failure).

    If a fresh cache already exists at the expected path it's loaded
    instead of re-extracting.
    """
    from prediction_pipeline import PixelPaws_ExtractFeatures
    base = v.stem
    feat_cache = FEATURES_DIR / f'{base}_features_{cfg_hash}.pkl'
    if feat_cache.is_file():
        step(f'  cached: {feat_cache.name}')
        with open(feat_cache, 'rb') as f:
            return pickle.load(f)
    step(f'  extracting features for {base}')
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
    step(f'  -> {feat_cache.name} ({time.time()-t0:.1f}s, X={X.shape})')
    return X


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-only', action='store_true',
                        help='extract + cache features and exit (no classifiers, '
                             'no analysis). Includes pawcapture cohort.')
    args = parser.parse_args()

    from prediction_pipeline import (
        predict_with_xgboost, augment_features_post_cache,
    )
    try:
        from feature_cache import FeatureCacheManager
        cfg_hash = FeatureCacheManager.compute_hash(FEAT_CFG)
    except Exception:
        cfg_hash = 'manual'
    step(f'feature-cache hash: {cfg_hash}')

    for d in (FEATURES_DIR, RESULTS_DIR, ANALYSIS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    key_df = load_key_file()
    iter11 = key_df[key_df['Cohort'].astype(str).str.lower() == 'iter11'].copy()
    pawcap = key_df[key_df['Cohort'].astype(str).str.lower() == 'pawcapture'].copy()
    step(f'iter11 sessions: {len(iter11)}   pawcapture sessions: {len(pawcap)}')

    def _pairs_for(df):
        out = []
        for _, row in df.iterrows():
            subj = str(row['Subject'])
            vids = [p for p in VIDEO_DIR.glob(f'{subj}*.mp4') if p.stem == subj]
            if not vids:
                step(f'  ! no video for {subj}'); continue
            v = vids[0]
            h5 = find_h5_for(v)
            if h5 is None:
                step(f'  ! no DLC h5 for {subj}'); continue
            out.append((row, v, h5))
        return out

    # ---- features-only fast path ---------------------------------------
    if args.features_only:
        all_pairs = _pairs_for(iter11) + _pairs_for(pawcap)
        step(f'\nFeature-only mode: {len(all_pairs)} session(s)')
        for row, v, h5 in all_pairs:
            step(f'\n=== {row["Subject"]} ({row["Cohort"]}/{row["Condition"]}) ===')
            try:
                _extract_one(row, v, h5, cfg_hash)
            except Exception as e:
                step(f'  ! extraction failed: {e}')
        step('\nDone (features-only).')
        return 0

    # ---- full pipeline path --------------------------------------------
    pairs = _pairs_for(iter11)
    if not pairs:
        step('No iter11 (video, h5) pairs found'); return 1
    step(f'Sessions to process for classifiers: {len(pairs)}')

    # Load classifiers
    scratch_with_cd = None
    if SCRATCH_CLF.is_file():
        with open(SCRATCH_CLF, 'rb') as f:
            scratch_with_cd = pickle.load(f)
    else:
        step(f'  ! missing Scratching classifier: {SCRATCH_CLF}'); return 1

    snlt_loaded = []
    for spec in SNLT_CLF_SPEC:
        p = SNLT_CLF_DIR / spec['pkl']
        if not p.is_file():
            step(f'  ! missing SNLT classifier: {p}'); continue
        with open(p, 'rb') as f:
            cd = pickle.load(f)
        snlt_loaded.append({**spec, 'clf_data': cd})
    step(f'Classifiers loaded: {len(snlt_loaded)} SNLT + Scratching')

    # Scratching postproc from its own pkl (same as habby_*.py).
    scr_thresh = float(scratch_with_cd.get('best_thresh', 0.6))
    scr_min_bout = int(scratch_with_cd.get('min_bout', 12))
    scr_min_after = int(scratch_with_cd.get('min_after_bout', 1))

    scratching_summary = []
    scratching_bouts = []
    occ_rows = []
    cond_mats = {}                # condition -> sum-of-matrices
    per_session_states = {}

    for row, v, h5 in pairs:
        base = v.stem
        animal = str(row['Animal'])
        cond = str(row['Condition'])
        laser = str(row.get('LaserSide', '') or '')
        # within-cohort grouping for paired plots
        mouse_key = f'{animal}_{laser}' if laser else animal
        step(f'\n=== {animal}/{cond} [{base}] ===')

        feat_cache = FEATURES_DIR / f'{base}_features_{cfg_hash}.pkl'
        if feat_cache.is_file():
            step('  loading cached features')
            with open(feat_cache, 'rb') as f:
                X = pickle.load(f)
        else:
            step('  extracting features')
            from prediction_pipeline import PixelPaws_ExtractFeatures
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

        # Per-classifier predictions (Scratching + 5 SNLT)
        per_video = pd.DataFrame({'frame': range(len(X))})

        # 1. Scratching
        X_aug = augment_features_post_cache(X.copy(), scratch_with_cd,
                                            scratch_with_cd['clf_model'], str(h5))
        y_proba = predict_with_xgboost(
            scratch_with_cd['clf_model'], X_aug,
            calibrator=scratch_with_cd.get('prob_calibrator'),
            fold_models=scratch_with_cd.get('fold_models'),
        )
        y_raw = (y_proba >= scr_thresh).astype(int)
        y_scratch = apply_postprocess(y_raw, scr_min_bout, scr_min_after, 0)
        per_video['Scratching_proba'] = y_proba
        per_video['Scratching'] = y_scratch
        n_pos = int(y_scratch.sum()); n = len(y_scratch)
        bouts = find_bouts(y_scratch)
        mean_bout = (np.mean([b[1]-b[0]+1 for b in bouts]) / FPS) if bouts else 0
        step(f'  Scratching       thresh={scr_thresh:.2f} min_bout={scr_min_bout:>2}  '
             f'pos={n_pos:>5} ({100*n_pos/n:5.2f}%)  bouts={len(bouts):>3}  '
             f'mean_bout_s={mean_bout:.2f}')
        scratching_summary.append({
            'session': base, 'mouse': animal, 'condition': cond, 'laser': laser,
            'best_thresh': scr_thresh, 'min_bout': scr_min_bout,
            'n_frames': n, 'n_pos': n_pos,
            'pct_time': round(100 * n_pos / n, 3),
            'time_sec': round(n_pos / FPS, 2),
            'n_bouts': len(bouts),
            'mean_bout_sec': round(mean_bout, 3),
            'median_bout_sec': round(float(np.median([b[1]-b[0]+1 for b in bouts]) / FPS)
                                      if bouts else 0, 3),
        })
        for s, e in bouts:
            scratching_bouts.append({
                'session': base, 'mouse': animal, 'condition': cond, 'laser': laser,
                'start_frame': s, 'end_frame': e,
                'length_frames': e - s + 1,
                'length_sec': round((e - s + 1) / FPS, 3),
                'start_sec': round(s / FPS, 2),
            })

        # 2. SNLT 5 classifiers
        for L in snlt_loaded:
            cd = L['clf_data']; model = cd['clf_model']
            try:
                X_aug = augment_features_post_cache(X.copy(), cd, model, str(h5))
                yp = predict_with_xgboost(
                    model, X_aug,
                    calibrator=cd.get('prob_calibrator'),
                    fold_models=cd.get('fold_models'),
                )
                yr = (yp >= L['best_thresh']).astype(int)
                yp_post = apply_postprocess(
                    yr, L['min_bout'], L['min_after_bout'], L['max_gap'])
            except Exception as e:
                step(f'  ! {L["name"]} failed: {e}'); continue
            per_video[L['name']] = yp_post
            n_pos = int(yp_post.sum())
            step(f'  {L["name"]:18}  pos={n_pos:>5} ({100*n_pos/n:5.2f}%)')

        out_csv = RESULTS_DIR / f'{base}_predictions.csv'
        per_video.to_csv(out_csv, index=False)
        step(f'  -> {out_csv.name}')

        # Argmax + transitions
        states = assign_states(per_video, PRIORITY)
        per_session_states[base] = {
            'states': states, 'mouse': mouse_key, 'condition': cond,
            'animal': animal, 'laser': laser,
        }
        occ = state_occupancy(states, PRIORITY)
        occ.update({'session': base, 'mouse': mouse_key, 'animal': animal,
                    'condition': cond, 'laser': laser})
        occ_rows.append(occ)
        M = transition_matrix(states, PRIORITY)
        cond_mats[cond] = (cond_mats[cond].add(M, fill_value=0)
                            if cond in cond_mats else M)

    # ---- save tabular outputs ----
    pd.DataFrame(scratching_summary).to_csv(
        ANALYSIS_DIR / 'scratching_summary.csv', index=False)
    pd.DataFrame(scratching_bouts).to_csv(
        ANALYSIS_DIR / 'scratching_bouts.csv', index=False)
    occ_df = pd.DataFrame(occ_rows)
    occ_df.to_csv(ANALYSIS_DIR / 'transitions_state_occupancy.csv', index=False)
    for cond, M in cond_mats.items():
        M.to_csv(ANALYSIS_DIR / f'transitions_{cond}.csv')
    step('\nSaved scratching_summary.csv, scratching_bouts.csv, '
         'transitions_state_occupancy.csv, transitions_{cond}.csv')

    # ---- plots: scratching paired stim vs nostim (per LaserSide) ----
    if len(occ_df) and {'nostim', 'stim'}.issubset(occ_df['condition'].unique()):
        _plot_state_occupancy(occ_df, ANALYSIS_DIR / 'transitions_state_occupancy.png')
        _plot_heatmaps(cond_mats, ANALYSIS_DIR / 'transitions_heatmaps.png')
        _plot_diff(cond_mats, ANALYSIS_DIR / 'transitions_diffs.png')
        _plot_ethograms(per_session_states, PRIORITY,
                        ANALYSIS_DIR / 'transitions_ethograms.png', fps=FPS)
        step('Saved transition plots')
    else:
        step('Note: needs both nostim AND stim sessions in key_file '
             '(Condition column) to draw paired/diff plots')

    return 0


# ---------- plot helpers (lifted/trimmed from habby_transitions.py) ----------

def _plot_state_occupancy(occ_df, png):
    states = PRIORITY if EXCLUDE_NOISE else PRIORITY + [NOISE_LABEL]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(states)); w = 0.35
    palette = {'nostim': '#1f77b4', 'stim': '#d62728'}
    for i, cond in enumerate(['nostim', 'stim']):
        sub = occ_df[occ_df['condition'] == cond]
        means = [sub[s].mean() * 100 if s in sub.columns and len(sub) else 0
                 for s in states]
        sems = [sub[s].sem() * 100 if (s in sub.columns and len(sub) > 1) else 0
                for s in states]
        ax.bar(x + (i - 0.5) * w, means, w, yerr=sems, label=cond,
               color=palette[cond], alpha=0.7, capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(states, rotation=30, ha='right')
    ax.set_ylabel('% time'); ax.legend(); ax.grid(axis='y', alpha=0.3)
    ax.set_title('iter11 — per-state occupancy (mean ± SEM)')
    fig.tight_layout(); fig.savefig(png, dpi=140); plt.close(fig)


def _plot_heatmaps(group_mats, png):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for c, cond in enumerate(['nostim', 'stim']):
        ax = axes[c]
        M = group_mats.get(cond)
        if M is None or M.sum().sum() == 0:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center'); ax.set_title(cond); continue
        P = normalise_rows(M)
        im = ax.imshow(P.to_numpy(), cmap='inferno', vmin=0, vmax=1)
        ax.set_xticks(range(len(P.columns)))
        ax.set_xticklabels(P.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(P.index))); ax.set_yticklabels(P.index)
        ax.set_title(cond); ax.set_xlabel('to'); ax.set_ylabel('from')
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                v = P.iloc[i, j]
                if v > 0.01:
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                            color='white' if v < 0.5 else 'black', fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle('iter11 — transition probabilities (row-normalised)',
                 fontweight='bold')
    fig.tight_layout(); fig.savefig(png, dpi=140); plt.close(fig)


def _plot_diff(group_mats, png):
    A = group_mats.get('stim'); B = group_mats.get('nostim')
    fig, ax = plt.subplots(figsize=(8, 7))
    if A is None or B is None or A.sum().sum() == 0 or B.sum().sum() == 0:
        ax.text(0.5, 0.5, 'no data', ha='center', va='center')
    else:
        PA = normalise_rows(A); PB = normalise_rows(B)
        idx = PA.index.union(PB.index); cols = PA.columns.union(PB.columns)
        PA = PA.reindex(index=idx, columns=cols, fill_value=0)
        PB = PB.reindex(index=idx, columns=cols, fill_value=0)
        D = (PA - PB).to_numpy()
        im = ax.imshow(D, cmap='RdBu_r', vmin=-0.4, vmax=0.4)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha='right')
        ax.set_yticks(range(len(idx))); ax.set_yticklabels(idx)
        ax.set_title('STIM − NOSTIM (transition probabilities)')
        ax.set_xlabel('to'); ax.set_ylabel('from')
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                v = D[i, j]
                if abs(v) > 0.03:
                    ax.text(j, i, f'{v:+.2f}', ha='center', va='center',
                            color='black', fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle('iter11 — within-cohort difference (stim − nostim)',
                 fontweight='bold')
    fig.tight_layout(); fig.savefig(png, dpi=140); plt.close(fig)


def _plot_ethograms(per_session_states, all_states, png, fps=60):
    items = sorted(per_session_states.items(),
                   key=lambda kv: (kv[1]['condition'], kv[1]['mouse']))
    n_sess = len(items)
    state_to_int = {s: i for i, s in enumerate(all_states + [NOISE_LABEL])}
    max_len = max(len(info['states']) for _, info in items)
    grid = np.full((n_sess, max_len), state_to_int[NOISE_LABEL], dtype=int)
    labels = []
    for r, (sess, info) in enumerate(items):
        s = info['states']
        ints = np.array([state_to_int.get(x, state_to_int[NOISE_LABEL]) for x in s])
        grid[r, :len(ints)] = ints
        labels.append(f'{info["condition"][:5]}  {info["mouse"]}')
    n_states = len(all_states) + 1
    cmap = plt.get_cmap('tab10', n_states)
    fig, ax = plt.subplots(figsize=(14, max(3, n_sess * 0.5)))
    im = ax.imshow(grid, aspect='auto', interpolation='nearest', cmap=cmap,
                   vmin=-0.5, vmax=n_states - 0.5)
    ax.set_yticks(range(n_sess)); ax.set_yticklabels(labels, fontsize=9)
    xt = np.linspace(0, max_len, 11)
    ax.set_xticks(xt); ax.set_xticklabels([f'{int(t/fps/60)}' for t in xt])
    ax.set_xlabel('Time (min)')
    ax.set_title('iter11 — per-session ethograms')
    cbar = plt.colorbar(im, ax=ax, ticks=range(n_states),
                        fraction=0.02, pad=0.01)
    cbar.ax.set_yticklabels(all_states + [NOISE_LABEL], fontsize=9)
    fig.tight_layout(); fig.savefig(png, dpi=140); plt.close(fig)


if __name__ == '__main__':
    sys.exit(main())
