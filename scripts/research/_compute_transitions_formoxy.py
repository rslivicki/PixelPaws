"""Standalone "compute transitions" runner for the FormalinOxy/Left_paws
project, side-stepping the GUI's cache-upgrade chain.

Logs to BOTH stdout AND a tail-able log file so we can watch progress.

Strategy:
  - For each session, find the existing brightness columns (Pix_*) in
    whatever feature pkl is on disk (any hash).
  - Re-extract pose features fresh from the DLC h5 (via PoseFeatureExtractor;
    fast and DLC-only — no video re-read).
  - Concat pose + brightness, then run augment_features_post_cache to add
    ego/contact/lag features.
  - For each classifier in the JSON config: predict_proba → threshold →
    bout filter (min_bout, min_after_bout, max_gap as specified in JSON).
  - Argmax across the K classifiers; if no classifier crosses its
    threshold the frame is "Other".
  - Compute per-session occupancy fractions (per-class frame share).
  - Group sessions by `Treatment` from the key xlsx and plot a grouped
    fractional-occupancy bar chart (inferno palette).

Outputs PNG + CSV to the project's transitions/ folder.
"""

from __future__ import annotations

import os
import re
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

LOG_PATH = None  # set in main()
def log(msg: str):
    sys.stdout.write(msg + '\n')
    sys.stdout.flush()
    if LOG_PATH:
        with open(LOG_PATH, 'a', encoding='utf-8') as _f:
            _f.write(msg + '\n')

PROJECT = r'E:/RSVIDS/Blackbox/2512_Blackbox_Formalin_Oxy/Left_paws'
JSON_PATH = os.path.join(PROJECT, 'transitions', 'Config1.json')
OUT_DIR = os.path.join(PROJECT, 'transitions')

sys.path.insert(0, r'E:/PixelPaws')
from pose_features import PoseFeatureExtractor
from prediction_pipeline import (
    augment_features_post_cache, predict_with_xgboost,
)
from evaluation_tab import _apply_bout_filtering


# ── locate per-session DLC h5 + best brightness cache ────────────────
def find_dlc_h5(session: str) -> str | None:
    """Find DLC h5 for *exact* session prefix — guard against S1 picking
    up S10's file by requiring a non-digit boundary after the session
    name."""
    sess_re = re.compile(rf'^{re.escape(session)}(?:DLC|_)')
    for sub in ('videos', 'Cut_videos'):
        d = os.path.join(PROJECT, sub)
        if not os.path.isdir(d):
            continue
        # Prefer _filtered.h5
        for f in os.listdir(d):
            if (sess_re.match(f) and f.endswith('_filtered.h5')):
                return os.path.join(d, f)
        for f in os.listdir(d):
            if (sess_re.match(f) and f.endswith('.h5')
                    and '_filtered' not in f):
                return os.path.join(d, f)
    return None


def find_existing_cache(session: str) -> str | None:
    """Pick the largest pkl that matches `<session>_features_*.pkl`
    exactly (boundary check guards against S1 grabbing S10's cache)."""
    fdir = os.path.join(PROJECT, 'features')
    candidates = []
    pat = re.compile(rf'^{re.escape(session)}_features_[a-f0-9]+(?:_corrected)?\.pkl$')
    for f in os.listdir(fdir):
        if pat.match(f) and '.version' not in f:
            candidates.append(os.path.join(fdir, f))
    if not candidates:
        return None
    candidates.sort(key=os.path.getsize, reverse=True)
    return candidates[0]


def derive_brightness_features(pix_df: pd.DataFrame, dt_vel: int = 2) -> pd.DataFrame:
    """Re-derive the |d/dt(Pix_*)|, Log10 ratios, etc. from a small Pix_*
    cache slice — matches PixelBrightnessExtractor behaviour but DOESN'T
    re-read video. Lets us run inference without re-extraction.
    """
    base = pix_df.copy().reset_index(drop=True)
    pix_cols = [c for c in base.columns if c.startswith('Pix_')
                and '/' not in c]

    # Pairwise log ratios — must come BEFORE derivatives so we can also
    # take the |d/dt| of each ratio.
    ratios = pd.DataFrame(index=base.index)
    for i in range(len(pix_cols)):
        for j in range(i + 1, len(pix_cols)):
            bp1 = pix_cols[i][len('Pix_'):]
            bp2 = pix_cols[j][len('Pix_'):]
            num = base[pix_cols[i]].astype(float)
            den = base[pix_cols[j]].astype(float).replace(0, 1e-10).abs() + 1e-10
            r = (num / den).clip(lower=1e-10)
            ratios[f'Log10(Pix_{bp1}/Pix_{bp2})'] = np.log10(r).fillna(0)

    full_brt = pd.concat([base[pix_cols], ratios], axis=1)

    # |d/dt(...)| at dt_vel (the canonical one) and at dt=1, dt=5
    deriv_dfs = []
    diff_dt = full_brt.diff(periods=dt_vel).abs().fillna(0)
    diff_dt.columns = [f'|d/dt({c})|' for c in full_brt.columns]
    deriv_dfs.append(diff_dt)
    for dt_x in (1, 5):
        if dt_x == dt_vel:
            continue
        d_x = full_brt.diff(periods=dt_x).abs().fillna(0)
        d_x.columns = [f'|d/dt{dt_x}({c})|' for c in full_brt.columns]
        deriv_dfs.append(d_x)

    # Per-bp derived: BrightOnsetPeak / BrightAccel / BrightAsymmetry / SurfaceZ / SurfaceZVel
    extra_cols = []
    for col in pix_cols:
        raw = base[col].astype(float)
        d1 = raw.diff(1).fillna(0)
        onset_peak = d1.abs().rolling(5, center=True, min_periods=1).max()
        bright_accel = d1.diff(1).fillna(0)
        pos_d1 = d1.clip(lower=0)
        neg_d1_abs = (-d1).clip(lower=0)
        roll_max_pos = pos_d1.rolling(30, min_periods=1).max()
        roll_mean_neg = neg_d1_abs.rolling(30, min_periods=1).mean()
        bright_asym = (roll_max_pos / (roll_mean_neg + 1e-6)).fillna(0)
        roll_med = raw.rolling(500, min_periods=1).median()
        roll_std = raw.rolling(500, min_periods=1).std().fillna(1)
        surface_z = ((raw - roll_med) / (roll_std + 1e-6)).fillna(0)
        surface_z_vel = surface_z.diff(1).fillna(0)
        extra_cols.extend([
            onset_peak.rename(f'{col}_BrightOnsetPeak'),
            bright_accel.rename(f'{col}_BrightAccel'),
            bright_asym.rename(f'{col}_BrightAsymmetry'),
            surface_z.rename(f'{col}_SurfaceZ'),
            surface_z_vel.rename(f'{col}_SurfaceZVel'),
        ])

    out = pd.concat([full_brt] + deriv_dfs + [pd.concat(extra_cols, axis=1)],
                    axis=1)
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out = out.clip(lower=-1e9, upper=1e9)
    return out


def load_brightness_cols(cache_path: str) -> pd.DataFrame:
    """Load Pix_<bp> raw columns from cache, then derive the full
    brightness feature suite from them (no video needed)."""
    with open(cache_path, 'rb') as f:
        X = pickle.load(f)
    if isinstance(X, dict) and 'X' in X:
        X = X['X']
    pix_cols = [c for c in X.columns if c.startswith('Pix_')
                and '/' not in c]
    if not pix_cols:
        return pd.DataFrame()
    return derive_brightness_features(X[pix_cols].reset_index(drop=True))


# ── feature build per session ────────────────────────────────────────
def build_features_for_session(session: str, dlc_h5: str,
                               brightness_df: pd.DataFrame) -> pd.DataFrame:
    """Pose features fresh from DLC + brightness from cache."""
    pe = PoseFeatureExtractor(bodyparts=[], likelihood_threshold=0.8,
                              velocity_delta=2, contact_threshold=15.0)
    X_pose = pe.extract_all_features(
        dlc_h5,
        include_angles=True,
        include_velocities=True,
        include_distance_velocities=True,
        include_in_frame=True,
        include_new_pose=True,
        include_new_kinematics=True,
        include_flinch_features=True,
        include_temporal_context_features=True,
        include_lag_features=False,  # added later by augment_features_post_cache
    ).reset_index(drop=True)

    if not brightness_df.empty:
        m = min(len(X_pose), len(brightness_df))
        X = pd.concat([X_pose.iloc[:m].reset_index(drop=True),
                       brightness_df.iloc[:m].reset_index(drop=True)], axis=1)
    else:
        X = X_pose
    return X


# ── main pipeline ────────────────────────────────────────────────────
def main():
    global LOG_PATH
    LOG_PATH = os.path.join(OUT_DIR, '_compute_transitions_run.log')
    open(LOG_PATH, 'w').close()  # truncate
    cfg = json.load(open(JSON_PATH))
    classifiers = cfg['classifiers']
    sessions = cfg['session_selection']
    key_path = cfg['key_file']

    key_df = pd.read_excel(key_path)
    treat_map = dict(zip(key_df['Subject'], key_df['Treatment']))
    log(f'[setup] {len(sessions)} sessions, {len(classifiers)} classifiers, '
          f'key file {os.path.basename(key_path)}')

    # Load classifiers
    clf_data_list = []
    behaviors = []
    for c in classifiers:
        d = joblib.load(c['path'])
        # JSON-overrides for threshold + bout filter
        d['best_thresh'] = float(c['best_thresh'])
        d['min_bout'] = int(c['min_bout'])
        d['min_after_bout'] = int(c['min_after_bout'])
        d['max_gap'] = int(c['max_gap'])
        clf_data_list.append(d)
        behaviors.append(d.get('Behavior_type', os.path.basename(c['path']).replace('.pkl', '')))
    log(f'[clfs ] {behaviors}')

    occupancy_rows = []  # (session, treatment, behavior, fraction)
    state_labels = ['Other'] + behaviors  # index 0 = Other
    states_dir = os.path.join(OUT_DIR, '_states')
    os.makedirs(states_dir, exist_ok=True)

    for sess in sessions:
        dlc = find_dlc_h5(sess)
        cache = find_existing_cache(sess)
        if not dlc:
            log(f'[skip ] {sess}: DLC h5 not found')
            continue
        if not cache:
            log(f'[warn ] {sess}: no existing cache for brightness')
            brt = pd.DataFrame()
        else:
            brt = load_brightness_cols(cache)
        log(f'[build] {sess}  dlc={os.path.basename(dlc)}  '
              f'cache={os.path.basename(cache) if cache else "—"}  '
              f'brt_cols={brt.shape[1]}')
        try:
            X = build_features_for_session(sess, dlc, brt)
        except Exception as e:
            log(f'[fail ] {sess}: feature build failed: {e}')
            continue
        n_frames = len(X)

        # Per-classifier predict + threshold + bout filter
        prob_matrix = np.zeros((n_frames, len(clf_data_list)))
        binary_matrix = np.zeros((n_frames, len(clf_data_list)), dtype=int)
        for ci, cd in enumerate(clf_data_list):
            try:
                m = cd['clf_model']
                X_aug = augment_features_post_cache(
                    X.copy(), cd, m, dlc, log_fn=None)
                proba = predict_with_xgboost(
                    m, X_aug,
                    calibrator=cd.get('prob_calibrator'),
                    fold_models=cd.get('fold_models'))
                t = float(cd['best_thresh'])
                raw = (proba >= t).astype(int)
                filt = _apply_bout_filtering(
                    raw, cd['min_bout'], cd['min_after_bout'], cd['max_gap'])
                prob_matrix[:len(proba), ci] = proba
                binary_matrix[:len(filt), ci] = filt
            except Exception as e:
                log(f'  ! {behaviors[ci]}: {e}')

        # Argmax assignment: whichever classifier has highest proba AND
        # crosses its own threshold; else 'Other' (state 0). This matches
        # the GUI's argmax assign_mode behaviour.
        best_idx = np.argmax(prob_matrix, axis=1)
        threshes = np.array([cd['best_thresh'] for cd in clf_data_list])
        crossed = prob_matrix[np.arange(n_frames), best_idx] >= threshes[best_idx]
        # AND require the binary (post bout-filter) to also be 1, so the
        # min_bout / max_gap parameters are honoured per-class.
        bin_at_best = binary_matrix[np.arange(n_frames), best_idx] == 1
        states = np.zeros(n_frames, dtype=int)
        active = crossed & bin_at_best
        states[active] = best_idx[active] + 1

        # Persist state sequence for downstream plots
        np.save(os.path.join(states_dir, f'{sess}_states.npy'), states)

        # Occupancy fractions
        treat = treat_map.get(sess, 'Unknown')
        for sid, lbl in enumerate(state_labels):
            frac = float(np.mean(states == sid))
            occupancy_rows.append({
                'session': sess, 'treatment': treat,
                'behavior': lbl, 'fraction': frac})
        non_other = 1.0 - float(np.mean(states == 0))
        log(f'  -> non-Other occupancy: {non_other:.1%}  '
              f'({sess} treat={treat})')

    occ_df = pd.DataFrame(occupancy_rows)
    csv_out = os.path.join(OUT_DIR, '_occupancy_fractions.csv')
    occ_df.to_csv(csv_out, index=False)
    log(f'\nwrote {csv_out}')

    # ── Plot: fractional occupancy by treatment group, faceted by behavior ─
    treat_order = ['Veh', 'Oxy1', 'Oxy3', 'Oxy10']
    # Drop "Other" from the bars (it's just the complement)
    behaviors_to_plot = [b for b in state_labels if b != 'Other']

    n_b = len(behaviors_to_plot)
    INFERNO = plt.cm.inferno
    treat_colors = {t: INFERNO(0.20 + 0.62 * (i / max(len(treat_order) - 1, 1)))
                    for i, t in enumerate(treat_order)}

    fig, axes = plt.subplots(1, n_b, figsize=(3.4 * n_b, 4.2),
                             sharey=True, constrained_layout=True)
    if n_b == 1:
        axes = [axes]
    for ax, bname in zip(axes, behaviors_to_plot):
        sub = occ_df[occ_df['behavior'] == bname]
        means, stds, ns = [], [], []
        for t in treat_order:
            vals = sub[sub['treatment'] == t]['fraction'].values
            means.append(vals.mean() if len(vals) else 0.0)
            stds.append(vals.std() if len(vals) > 1 else 0.0)
            ns.append(len(vals))
        x = np.arange(len(treat_order))
        bars = ax.bar(x, means, 0.65, yerr=stds, capsize=4,
                      color=[treat_colors[t] for t in treat_order],
                      edgecolor='black', linewidth=0.6,
                      error_kw={'ecolor': '#222', 'elinewidth': 1.2,
                                'capthick': 1.2}, zorder=2)
        # Per-subject dots
        for ti, t in enumerate(treat_order):
            vals = sub[sub['treatment'] == t]['fraction'].values
            jitter = (np.random.RandomState(ti).rand(len(vals)) - 0.5) * 0.20
            ax.scatter(np.full(len(vals), ti) + jitter, vals,
                       s=22, color='white', edgecolor='black',
                       linewidth=0.6, zorder=3)
        for xi, mv in zip(x, means):
            ax.text(xi, mv + 0.012, f'{mv:.2f}',
                    ha='center', va='bottom', fontsize=8.5,
                    color='#222', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{t}\n(n={n})' for t, n in zip(treat_order, ns)],
                           fontsize=9)
        ax.set_title(bname, fontsize=11, fontweight='bold')
        ax.grid(axis='y', linewidth=0.4, color='#bbb', zorder=1)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(0.05, occ_df['fraction'].max() * 1.15))
    axes[0].set_ylabel('Fractional occupancy', fontsize=11)
    fig.suptitle('FormOxy / Left_paws — behavior occupancy by treatment',
                 fontsize=12, fontweight='bold')

    out_png = os.path.join(OUT_DIR, '_occupancy_fractions.png')
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    log(f'wrote {out_png}')


if __name__ == '__main__':
    main()
