"""
Habby project - transition / state-occupancy analysis using the same
6-behavior argmax framework as the THC analysis.

Behaviors (priority order, matches SNLT for_claude.json + Scratching):
  Facial_grooming, Left_licking, Scratching, rearing, still, walking

Reuses already-cached Habby features (extracted with optical flow for
Scratching). Runs the 5 SNLT classifiers + already-computed Scratching
predictions, applies each classifier's saved post-processing, then:

  - argmax assignment per frame (priority order, EXCLUDE_NOISE for
    occupancy + transitions)
  - per-session state-occupancy CSV
  - per-condition (nostim / stim) transition matrices
  - PNGs: immobility (paired), state_occupancy bars, transitions abs +
    diffs, bout structure, ethograms
  - Discord uploads
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(r'E:\PixelPaws')
sys.path.insert(0, str(REPO))

PROJECT = Path(r'E:\RSVIDS\Blackbox\AM_ChR2_stim_SPB_analysis')
VIDEO_DIR = PROJECT / 'Videos'
FEATURES_DIR = PROJECT / 'features'
RESULTS_DIR = PROJECT / 'results'
ANALYSIS_DIR = PROJECT / 'analysis'

WEBHOOK = (
    ""
)

JSON_PATH = Path(r'E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\transitions\for_claude.json')
SNLT_CLF_DIR = Path(r'E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\classifiers')
SCRATCH_CLF = PROJECT / 'classifiers' / 'PixelPaws_Scratching_AllFeatures.pkl'

# Priority order for argmax (highest priority wins on ties).
# Scratching inserted between Left_licking and rearing.
PRIORITY = ['Facial_grooming', 'Left_licking', 'Scratching',
            'rearing', 'still', 'walking']
NOISE_LABEL = 'noise'
EXCLUDE_NOISE = True


def step(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def parse_session(name):
    m = re.search(r'_(nostim|stim)_FED2_(F\d|M\d)_', name)
    if not m:
        return None
    return m.group(2), ('nostim' if m.group(1) == 'nostim' else 'stim')


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
        idx = all_states + ([] if EXCLUDE_NOISE else [NOISE_LABEL])
        return pd.DataFrame(0, index=idx, columns=idx)
    bouts = [states[0]]
    for s in states[1:]:
        if s != bouts[-1]:
            bouts.append(s)
    if EXCLUDE_NOISE:
        bouts = [b for b in bouts if b != NOISE_LABEL]
        idx = all_states
    else:
        idx = all_states + [NOISE_LABEL]
    M = pd.DataFrame(0, index=idx, columns=idx)
    for a, b in zip(bouts[:-1], bouts[1:]):
        if a in M.index and b in M.columns:
            M.loc[a, b] += 1
    return M


def normalise_rows(M):
    rs = M.sum(axis=1).replace(0, 1)
    return M.div(rs, axis=0)


def post_to_discord(text, png_paths):
    import subprocess
    cmd = ['curl', '-s', '-o', os.devnull, '-w', '%{http_code}', '-X', 'POST',
           '-F', f'payload_json={{"content":"{text}"}}']
    for i, p in enumerate(png_paths):
        cmd += ['-F', f'file{i}=@{p}']
    cmd.append(WEBHOOK)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f'Discord upload: HTTP {r.stdout}')


# ---------- Plot helpers ----------

def plot_immobility(occ_df, png):
    palette = {'nostim': '#1f77b4', 'stim': '#d62728'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Panel 1: nostim vs stim boxplot
    ax = axes[0]
    data = []
    labels = []
    for cond in ['nostim', 'stim']:
        sub = occ_df[occ_df['condition'] == cond]
        data.append((sub['still'].to_numpy() * 100).tolist())
        labels.append(cond)
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55,
                    medianprops=dict(color='black'))
    for patch, c in zip(bp['boxes'], ['#1f77b4', '#d62728']):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    for i, ys in enumerate(data, 1):
        if ys:
            xs = np.random.normal(i, 0.05, len(ys))
            ax.scatter(xs, ys, color='black', s=24, zorder=3)
    ax.set_ylabel('% time still (excl. other)')
    ax.set_title('Habby: % time still')
    ax.grid(axis='y', alpha=0.3)

    # Panel 2: paired within-subject
    ax = axes[1]
    for mouse, mdf in occ_df.groupby('mouse'):
        if len(mdf) >= 2:
            no = mdf[mdf['condition'] == 'nostim']['still']
            st = mdf[mdf['condition'] == 'stim']['still']
            if len(no) and len(st):
                ax.plot([0, 1], [no.iloc[0] * 100, st.iloc[0] * 100],
                        'o-', color='gray', alpha=0.7)
                ax.text(1.02, st.iloc[0] * 100, mouse, fontsize=9, va='center')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['nostim', 'stim'])
    ax.set_ylabel('% time still')
    ax.set_title('Within-subject: nostim -> stim')
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Habby - immobility (still)', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(png, dpi=140); plt.close(fig)


def plot_state_occupancy(occ_df, png):
    states = PRIORITY if EXCLUDE_NOISE else PRIORITY + [NOISE_LABEL]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(states))
    w = 0.35
    palette = {'nostim': '#1f77b4', 'stim': '#d62728'}
    for i, cond in enumerate(['nostim', 'stim']):
        sub = occ_df[occ_df['condition'] == cond]
        means = [sub[s].mean() * 100 if s in sub.columns else 0 for s in states]
        sems = [sub[s].sem() * 100 if (s in sub.columns and len(sub) > 1) else 0
                for s in states]
        ax.bar(x + (i - 0.5) * w, means, w, yerr=sems, label=cond,
               color=palette[cond], alpha=0.7, capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(states, rotation=30, ha='right')
    ax.set_ylabel('% time'); ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_title('Habby: per-state occupancy (mean +/- SEM)')
    fig.tight_layout(); fig.savefig(png, dpi=140); plt.close(fig)


def plot_transition_heatmaps(group_mats, png):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for c, cond in enumerate(['nostim', 'stim']):
        ax = axes[c]
        M = group_mats.get(cond)
        if M is None or M.sum().sum() == 0:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center')
            ax.set_title(cond); continue
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
    fig.suptitle('Habby - transition probabilities (row-normalised)',
                 fontweight='bold')
    fig.tight_layout(); fig.savefig(png, dpi=140); plt.close(fig)


def plot_transition_diffs(group_mats, png):
    """Difference matrix: stim - nostim (within-subject between-cond)."""
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
        ax.set_title('STIM - NOSTIM (transition probabilities)')
        ax.set_xlabel('to'); ax.set_ylabel('from')
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                v = D[i, j]
                if abs(v) > 0.03:
                    ax.text(j, i, f'{v:+.2f}', ha='center', va='center',
                            color='black', fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle('Habby - within-subject difference (stim - nostim)',
                 fontweight='bold')
    fig.tight_layout(); fig.savefig(png, dpi=140); plt.close(fig)


def collect_bout_stats(per_session_states, all_states):
    rows = []
    for sess, info in per_session_states.items():
        states = info['states']
        bouts = []
        if len(states):
            cur = states[0]; cur_s = 0
            for i in range(1, len(states)):
                if states[i] != cur:
                    bouts.append((cur, cur_s, i - 1))
                    cur = states[i]; cur_s = i
            bouts.append((cur, cur_s, len(states) - 1))
        bout_lens = {s: [] for s in all_states + [NOISE_LABEL]}
        for s, a, b in bouts:
            if s in bout_lens:
                bout_lens[s].append(b - a + 1)
        row = {'session': sess, 'mouse': info['mouse'], 'condition': info['condition']}
        for s in all_states + [NOISE_LABEL]:
            row[f'{s}_bouts'] = len(bout_lens[s])
            row[f'{s}_mean_len'] = float(np.mean(bout_lens[s])) if bout_lens[s] else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def plot_bout_structure(bout_df, png):
    states = PRIORITY
    palette = {'nostim': '#1f77b4', 'stim': '#d62728'}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(states)); w = 0.35
    for ax, metric, ylabel, scale in [
            (axes[0], '_bouts', 'bout count', 1.0),
            (axes[1], '_mean_len', 'mean bout length (s)', 1.0 / 60.0)]:
        for i, cond in enumerate(['nostim', 'stim']):
            sub = bout_df[bout_df['condition'] == cond]
            means = [sub[f'{s}{metric}'].mean() * scale if len(sub) else 0 for s in states]
            sems = [sub[f'{s}{metric}'].sem() * scale if len(sub) > 1 else 0 for s in states]
            ax.bar(x + (i - 0.5) * w, means, w, yerr=sems, label=cond,
                   color=palette[cond], alpha=0.7, capsize=3)
        ax.set_xticks(x); ax.set_xticklabels(states, rotation=30, ha='right')
        ax.set_ylabel(ylabel); ax.legend()
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('Habby - bout structure', fontweight='bold')
    fig.tight_layout(); fig.savefig(png, dpi=140); plt.close(fig)


def plot_ethograms(per_session_states, all_states, png, fps=60):
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
    ax.set_title('Habby - per-session ethograms')
    cbar = plt.colorbar(im, ax=ax, ticks=range(n_states), fraction=0.02, pad=0.01)
    cbar.ax.set_yticklabels(all_states + [NOISE_LABEL], fontsize=9)
    fig.tight_layout(); fig.savefig(png, dpi=140); plt.close(fig)


# ---------- Main ----------

def main():
    from prediction_pipeline import (
        predict_with_xgboost, augment_features_post_cache,
    )

    # Load 5 SNLT classifiers from for_claude.json + Scratching local
    with open(JSON_PATH, 'r') as f:
        spec = json.load(f)
    snlt_clfs = []
    for c in spec['classifiers']:
        local = SNLT_CLF_DIR / Path(c['path']).name
        with open(local, 'rb') as f:
            cd = pickle.load(f)
        snlt_clfs.append({**c, 'clf_data': cd, 'name': Path(c['path']).stem})
    # Add scratching with its saved postprocessing
    with open(SCRATCH_CLF, 'rb') as f:
        scr_cd = pickle.load(f)
    snlt_clfs.append({
        'path': str(SCRATCH_CLF),
        'best_thresh': float(scr_cd.get('best_thresh', 0.6)),
        'min_bout': int(scr_cd.get('min_bout', 12)),
        'min_after_bout': int(scr_cd.get('min_after_bout', 1)),
        'max_gap': 0,
        'clf_data': scr_cd,
        'name': SCRATCH_CLF.stem,
    })

    rows = []
    cond_mats = {}  # condition -> sum-of-matrices
    per_session_states = {}

    feat_files = sorted(FEATURES_DIR.glob('*_features_*.pkl'))
    for fc in feat_files:
        # Reconstruct base name (drop _features_<hash>.pkl)
        base = fc.name
        m = re.match(r'(.+)_features_[0-9a-f]+\.pkl$', base)
        if not m:
            continue
        sess_base = m.group(1)
        parsed = parse_session(sess_base)
        if not parsed:
            continue
        mouse, cond = parsed
        step(f'\n=== {mouse} / {cond} ===')

        with open(fc, 'rb') as f:
            X = pickle.load(f)
        step(f'  X shape: {X.shape}')

        # Find dlc h5 next to video
        vids = list(VIDEO_DIR.glob(f'{sess_base}*.mp4'))
        if not vids:
            step(f'  ! no video for {sess_base}'); continue
        h5s = list(vids[0].parent.glob(f'{vids[0].stem}*shuffle9*filtered.h5'))
        if not h5s:
            h5s = list(vids[0].parent.glob(f'{vids[0].stem}*shuffle9*.h5'))
        dlc_path = str(h5s[0]) if h5s else ''

        # Run all 6 classifiers, collect binary state columns
        per_video = pd.DataFrame({'frame': range(len(X))})
        for L in snlt_clfs:
            cd = L['clf_data']
            model = cd['clf_model']
            best_thresh = float(L.get('best_thresh', cd.get('best_thresh', 0.5)))
            min_bout = int(L.get('min_bout', cd.get('min_bout', 1)))
            min_after_bout = int(L.get('min_after_bout', 0))
            max_gap = int(L.get('max_gap', 0))
            try:
                X_aug = augment_features_post_cache(X.copy(), cd, model, dlc_path)
                y_proba = predict_with_xgboost(
                    model, X_aug,
                    calibrator=cd.get('prob_calibrator'),
                    fold_models=cd.get('fold_models'),
                )
                y_raw = (y_proba >= best_thresh).astype(int)
                y_post = apply_postprocess(y_raw, min_bout, min_after_bout, max_gap)
            except Exception as e:
                step(f'  ! {L["name"]} failed: {e}')
                continue

            behavior = cd.get('Behavior_type') or L['name']
            per_video[behavior] = y_post
            n_pos = int(y_post.sum())
            step(f'  {behavior:18}  thresh={best_thresh:.2f} min_bout={min_bout:>2}  '
                 f'pos={n_pos} ({100*n_pos/len(X):.2f}%)')

        # Argmax assignment
        states = assign_states(per_video, PRIORITY)
        per_session_states[sess_base] = {
            'states': states, 'mouse': mouse, 'condition': cond,
        }
        occ = state_occupancy(states, PRIORITY)
        occ.update({'session': sess_base, 'mouse': mouse, 'condition': cond})
        rows.append(occ)

        M = transition_matrix(states, PRIORITY)
        if cond in cond_mats:
            cond_mats[cond] = cond_mats[cond].add(M, fill_value=0)
        else:
            cond_mats[cond] = M

    occ_df = pd.DataFrame(rows)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    occ_df.to_csv(ANALYSIS_DIR / 'transitions_state_occupancy.csv', index=False)
    for cond, M in cond_mats.items():
        M.to_csv(ANALYSIS_DIR / f'transitions_{cond}.csv')

    # Plots
    p_immob = ANALYSIS_DIR / 'transitions_immobility.png'
    p_states = ANALYSIS_DIR / 'transitions_state_occupancy.png'
    p_trans = ANALYSIS_DIR / 'transitions_heatmaps.png'
    p_diffs = ANALYSIS_DIR / 'transitions_diffs.png'
    p_bouts = ANALYSIS_DIR / 'transitions_bout_structure.png'
    p_etho = ANALYSIS_DIR / 'transitions_ethograms.png'

    plot_immobility(occ_df, p_immob)
    plot_state_occupancy(occ_df, p_states)
    plot_transition_heatmaps(cond_mats, p_trans)
    plot_transition_diffs(cond_mats, p_diffs)

    bout_df = collect_bout_stats(per_session_states, PRIORITY)
    bout_df.to_csv(ANALYSIS_DIR / 'transitions_bout_stats.csv', index=False)
    plot_bout_structure(bout_df, p_bouts)
    plot_ethograms(per_session_states, PRIORITY, p_etho)

    # Headline summary text
    lines = ['Habby transition analysis complete.']
    for cond in ['nostim', 'stim']:
        sub = occ_df[occ_df['condition'] == cond]
        if len(sub):
            lines.append(f'{cond}:')
            for s in PRIORITY:
                lines.append(f'  {s:18}  {sub[s].mean()*100:5.1f}% (n={len(sub)})')
    summary = '\\n'.join(lines).replace('"', '')

    post_to_discord(summary, [p_immob, p_states, p_trans, p_diffs])
    post_to_discord('Habby transitions: bout structure + ethograms.',
                    [p_bouts, p_etho])
    return 0


if __name__ == '__main__':
    sys.exit(main())
