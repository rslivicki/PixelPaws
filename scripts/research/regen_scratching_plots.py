"""regen_scratching_plots.py — produce three plasma-coloured figures
for the existing Scratching classifier in the 2510_Blackbox_Rimonabant
project, matching the BAREfoot/Panel-B-C-D reference style:

  Panel B : Learning curve — F1-score (CV) vs bout-positive frames
  Panel C : Threshold curves — Precision / Recall / F1 vs threshold
  Panel D : SHAP feature impact (beeswarm) + importance bar chart

Reuses what's already cached so the run is fast:
  - Panel C uses the cached `oof_proba` from ``PixelPaws_Scratching.pkl``
    — no retraining at all (≈ 1 s).
  - Panel D loads the cached `clf_model` and runs TreeSHAP on a small
    sample (≈ 30–60 s).
  - Panel B does need a CV sweep at several training-set sizes, so
    that's the slow step (≈ 3–6 min with reduced n_estimators).

No Optuna / no hyper-parameter sweep. Outputs land in
``<project>/Classifiers/plots/regen_<timestamp>/``.

Usage:
    py regen_scratching_plots.py
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_recall_curve, precision_score, recall_score, f1_score,
)


PROJECT     = r'E:/RSVIDS/Blackbox/2510_Blackbox_Rimonabant/' \
              r'Blackbox_videos-selected'
BEHAVIOR    = 'Scratching'
TRAIN_PKL   = os.path.join(PROJECT, 'Classifiers', 'training_data',
                           f'{BEHAVIOR}_train_set.pkl')
CLF_PKL     = os.path.join(PROJECT, 'Classifiers',
                           f'PixelPaws_{BEHAVIOR}.pkl')
PLOTS_BASE  = os.path.join(PROJECT, 'Classifiers', 'plots',
                           f'regen_{datetime.now():%Y%m%d_%H%M%S}')
N_FOLDS     = 5
RANDOM_SEED = 42

# Plasma anchors used across all three panels.
PLASMA            = plt.get_cmap('plasma')
PLASMA_DARK       = PLASMA(0.10)
PLASMA_MID        = PLASMA(0.50)
PLASMA_LIGHT      = PLASMA(0.80)
PLASMA_BAR        = PLASMA(0.30)
PLASMA_THRESHOLD  = PLASMA(0.70)
GRID_GREY         = '#dddddd'
TEXT_GREY         = '#444'

# Plot defaults — match the reference's clean sans look
mpl.rcParams.update({
    'font.family':      ['Arial', 'DejaVu Sans'],
    'font.size':        9.0,
    'axes.titlesize':   10.0,
    'axes.labelsize':   9.5,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         False,
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
})


# ─────────────────────────────────────────────────────────────────────
# Train + OOF
# ─────────────────────────────────────────────────────────────────────

def _make_xgb(scale_pos_weight: float, n_estimators: int = 300):
    """XGBoost with sensible defaults for binary behaviour
    classification — no tuning, just a fast baseline that lands in
    the ballpark of the existing classifier."""
    import xgboost as xgb
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method='hist',
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbosity=0,
    )


def cv_oof_proba(X: pd.DataFrame, y: np.ndarray,
                  n_folds: int = N_FOLDS, n_estimators: int = 300
                  ) -> tuple[np.ndarray, list[float]]:
    """Stratified-KFold OOF probabilities + per-fold F1 at threshold 0.5."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                            random_state=RANDOM_SEED)
    oof = np.zeros(len(y), dtype=np.float32)
    fold_f1 = []
    pos_w = (y == 0).sum() / max((y == 1).sum(), 1)
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        t0 = time.time()
        clf = _make_xgb(scale_pos_weight=pos_w, n_estimators=n_estimators)
        clf.fit(X.iloc[tr] if isinstance(X, pd.DataFrame) else X[tr],
                  y[tr])
        proba = clf.predict_proba(
            X.iloc[te] if isinstance(X, pd.DataFrame) else X[te])[:, 1]
        oof[te] = proba
        f1 = f1_score(y[te], (proba >= 0.5).astype(int), zero_division=0)
        fold_f1.append(float(f1))
        print(f'    fold {fold}/{n_folds}  F1@0.5={f1:.3f}  '
              f'({time.time()-t0:.1f}s)')
    return oof, fold_f1


def train_full_model(X: pd.DataFrame, y: np.ndarray):
    """Train on the entire cached set — used for SHAP + final classifier."""
    pos_w = (y == 0).sum() / max((y == 1).sum(), 1)
    clf = _make_xgb(scale_pos_weight=pos_w, n_estimators=400)
    clf.fit(X, y)
    return clf


# ─────────────────────────────────────────────────────────────────────
# Panel B — Learning curve
# ─────────────────────────────────────────────────────────────────────

def _bout_positive_subsample(X: pd.DataFrame, y: np.ndarray,
                              frac_pos: float, rng: np.random.Generator):
    """Sub-sample preserving ALL negatives but only `frac_pos` of the
    positives (mirrors how learning curves are typically swept against
    bout-positive frame counts).
    """
    pos_idx = np.where(y == 1)[0]
    keep_n = max(50, int(round(frac_pos * len(pos_idx))))
    keep_pos = rng.choice(pos_idx, size=min(keep_n, len(pos_idx)),
                            replace=False)
    neg_idx = np.where(y == 0)[0]
    final = np.concatenate([keep_pos, neg_idx])
    final.sort()
    return final


def learning_curve(X: pd.DataFrame, y: np.ndarray,
                    fractions: list[float] | None = None,
                    n_folds: int = 3, n_estimators: int = 200):
    """For each fraction f∈fractions: sample `f` of the positives,
    keep all negatives, run `n_folds` stratified-CV, record mean F1.
    Returns ``[(bout_pos_count, mean_f1, std_f1)…]``.
    """
    if fractions is None:
        fractions = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
    rng = np.random.default_rng(RANDOM_SEED)
    out = []
    for frac in fractions:
        idx = _bout_positive_subsample(X, y, frac, rng)
        X_sub = X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]
        y_sub = y[idx]
        n_pos = int(y_sub.sum())
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                random_state=RANDOM_SEED)
        f1s = []
        for fold, (tr, te) in enumerate(skf.split(X_sub, y_sub), 1):
            t0 = time.time()
            pos_w = (y_sub[tr] == 0).sum() / max((y_sub[tr] == 1).sum(), 1)
            clf = _make_xgb(scale_pos_weight=pos_w,
                              n_estimators=n_estimators)
            clf.fit(X_sub.iloc[tr] if isinstance(X_sub, pd.DataFrame)
                      else X_sub[tr], y_sub[tr])
            p = clf.predict_proba(
                X_sub.iloc[te] if isinstance(X_sub, pd.DataFrame)
                else X_sub[te])[:, 1]
            f1s.append(f1_score(y_sub[te], (p >= 0.5).astype(int),
                                  zero_division=0))
            print(f'    LC frac={frac:.2f}  fold {fold}/{n_folds}  '
                  f'F1={f1s[-1]:.3f}  ({time.time()-t0:.1f}s)')
        out.append((n_pos, float(np.mean(f1s)), float(np.std(f1s))))
        print(f'  → frac={frac:.2f}  n_pos={n_pos}  '
              f'F1={out[-1][1]:.3f}±{out[-1][2]:.3f}')
    return out


def plot_learning_curve(lc_results, behavior: str, out_path: str):
    """Panel B style — F1-score (5-fold CV) vs bout-positive frames.
    Single plasma-coloured line + markers + ±1σ shaded band."""
    xs    = np.array([r[0] for r in lc_results])
    means = np.array([r[1] for r in lc_results])
    stds  = np.array([r[2] for r in lc_results])

    fig, ax = plt.subplots(figsize=(4.0, 3.2),
                             constrained_layout=True)
    ax.plot(xs, means, marker='o', markersize=5, linewidth=1.6,
              color=PLASMA_MID, label=behavior, zorder=3)
    ax.fill_between(xs, means - stds, means + stds,
                      color=PLASMA_MID, alpha=0.18, zorder=2)
    ax.set_xlabel('Bout-positive frames')
    ax.set_ylabel(f'F1-score ({N_FOLDS}-fold CV)')
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.grid(axis='y', color=GRID_GREY, linewidth=0.6, zorder=1)
    ax.legend(frameon=False, loc='lower right', fontsize=8)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Panel C — Threshold curves
# ─────────────────────────────────────────────────────────────────────

def plot_threshold_curves(y: np.ndarray, oof_proba: np.ndarray,
                            behavior: str, out_path: str,
                            show_grid: bool = True):
    """Panel C style — Precision / Recall / F1 vs threshold from OOF
    predictions, plasma palette, dashed vertical line at the best-F1
    threshold.

    If ``show_grid`` is False, the horizontal y-grid is omitted (cleaner
    look for combined multi-panel figures)."""
    thresholds = np.linspace(0.01, 0.99, 99)
    prec, rec, f1 = [], [], []
    for t in thresholds:
        pred = (oof_proba >= t).astype(int)
        prec.append(precision_score(y, pred, zero_division=0))
        rec.append(recall_score(y, pred, zero_division=0))
        f1.append(f1_score(y, pred, zero_division=0))
    prec, rec, f1 = map(np.asarray, (prec, rec, f1))
    best_idx = int(np.argmax(f1))
    best_t   = float(thresholds[best_idx])
    best_f1  = float(f1[best_idx])

    fig, ax = plt.subplots(figsize=(4.0, 3.2),
                             constrained_layout=True)
    ax.plot(thresholds, rec,  color=PLASMA_LIGHT,    linewidth=1.6,
              label='Recall')
    ax.plot(thresholds, prec, color=PLASMA_MID,      linewidth=1.6,
              label='Precision')
    ax.plot(thresholds, f1,   color=PLASMA_DARK,     linewidth=1.8,
              label='F1 Score')
    ax.axvline(best_t, color=TEXT_GREY, linestyle=':', linewidth=0.9,
                 zorder=1)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1, 11))
    if show_grid:
        ax.grid(axis='y', color=GRID_GREY, linewidth=0.6, zorder=1)
    else:
        ax.grid(False)
    ax.legend(frameon=True, loc='lower left', fontsize=8,
                framealpha=0.95, edgecolor='#cccccc')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'    best threshold = {best_t:.2f}  F1 = {best_f1:.3f}')


# ─────────────────────────────────────────────────────────────────────
# Panel D — SHAP beeswarm + importance bar
# ─────────────────────────────────────────────────────────────────────

def _plasma_yellow_brown_cmap():
    """The reference uses a yellow→brown gradient on the SHAP beeswarm
    (Low → High feature value). For plasma consistency, build a tighter
    plasma slice (yellow at high → dark purple at low)."""
    return LinearSegmentedColormap.from_list(
        'plasma_lh',
        [PLASMA(0.05), PLASMA(0.50), PLASMA(0.90)],
        N=256)


def _density_jitter(x_values: np.ndarray,
                       n_bins: int = 80,
                       max_amp: float = 0.42,
                       seed: int = RANDOM_SEED) -> np.ndarray:
    """Vertical jitter scaled by local density along the x-axis. Where
    many SHAP values pile up at similar x, the jitter is wider — giving
    the row a bulbous beeswarm shape. Where the row is sparse, jitter
    stays small. Uses a histogram pass (O(n) instead of the n² KDE
    that the SHAP library does) so it scales to 10k+ points cheaply."""
    if len(x_values) == 0:
        return np.zeros(0)
    counts, edges = np.histogram(x_values, bins=n_bins)
    if counts.max() == 0:
        return np.zeros(len(x_values))
    bin_idx = np.clip(np.digitize(x_values, edges) - 1, 0, n_bins - 1)
    density_per_bin = counts / counts.max()         # 0..1
    point_density   = density_per_bin[bin_idx]
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, len(x_values)) * max_amp \
              * np.sqrt(point_density)


def plot_shap_panel(model, X: pd.DataFrame, behavior: str, out_path: str,
                     n_top: int = 12, sample_n: int = 12000):
    """Side-by-side SHAP beeswarm (left) + mean(|SHAP|) bar chart
    (right). Beeswarm dots colour-coded by feature value via a plasma
    slice and laid out with density-aware vertical jitter so each row
    forms a bulbous violin/teardrop shape. Importance bars are flat
    plasma so they read as magnitude only."""
    import shap

    # SHAP on a larger stratified sample so the beeswarm density reads
    # at the bulbous level the reference figure has
    rng = np.random.default_rng(RANDOM_SEED)
    if len(X) > sample_n:
        # Stratified sample on the binary label-free feature set: take
        # equal-ish counts across the index (proxy for time / session)
        sample_idx = rng.choice(len(X), size=sample_n, replace=False)
        X_s = X.iloc[sample_idx] if isinstance(X, pd.DataFrame) \
                  else X[sample_idx]
    else:
        X_s = X
    print(f'    computing SHAP on {len(X_s)} rows…')
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_s)
    if isinstance(sv, list):
        sv = sv[1]   # binary classifier — positive-class SHAP

    feat_names = list(X.columns) if isinstance(X, pd.DataFrame) \
                    else [f'f{i}' for i in range(X.shape[1])]
    importance = np.abs(sv).mean(axis=0)
    order      = np.argsort(importance)[::-1][:n_top]
    top_names  = [feat_names[i] for i in order]
    top_imp    = importance[order]

    # Larger figure so the beeswarm has room to spread + the bar chart
    # sits cleanly on the right.
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 6.5),
                                       gridspec_kw={'width_ratios':
                                                       [2.0, 1.0]},
                                       constrained_layout=True)
    cmap = _plasma_yellow_brown_cmap()

    # ── LEFT: bulbous SHAP beeswarm ───────────────────────────────
    y_pos = np.arange(n_top)[::-1]
    for j, idx in enumerate(order):
        col_vals = (X_s.iloc[:, idx] if isinstance(X_s, pd.DataFrame)
                      else X_s[:, idx])
        col_vals = col_vals.to_numpy() if hasattr(col_vals, 'to_numpy') \
                      else np.asarray(col_vals)
        # Robust 2-98 % min-max for the colour gradient (outliers don't
        # crush the visual)
        lo, hi = np.nanpercentile(col_vals, [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        norm = np.clip((col_vals - lo) / (hi - lo), 0, 1)
        x_sv = sv[:, idx]
        # Density-aware vertical jitter — gives the bulbous shape
        jitter = _density_jitter(x_sv, n_bins=80, max_amp=0.42,
                                    seed=RANDOM_SEED + j)
        axL.scatter(x_sv, y_pos[j] + jitter,
                      c=norm, cmap=cmap,
                      s=8, alpha=0.55, linewidths=0,
                      vmin=0, vmax=1, zorder=3)
        # Faint baseline grid so each row reads
        axL.axhline(y_pos[j], color=GRID_GREY, linewidth=0.4,
                      linestyle=':', alpha=0.5, zorder=1)
    axL.axvline(0, color='#888', linewidth=0.7, zorder=2)
    axL.set_yticks(y_pos)
    axL.set_yticklabels(top_names, fontsize=10)
    axL.set_xlabel('SHAP value (impact on model output)')
    axL.set_title(f'Feature Importance — {behavior}',
                     fontsize=11, fontweight='bold', loc='center')
    axL.tick_params(left=False, pad=2)
    axL.spines['left'].set_visible(False)
    axL.set_ylim(-0.7, n_top - 0.3)

    # Vertical "Low → High" plasma colour-bar at the right edge of axL
    cbar_ax = fig.add_axes([0.642, 0.18, 0.011, 0.55])
    cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap,
                                       orientation='vertical')
    cb.set_ticks([0.02, 0.98])
    cb.set_ticklabels(['Low', 'High'])
    cb.outline.set_visible(False)
    cbar_ax.tick_params(length=0, labelsize=8)
    cbar_ax.set_ylabel('Feature value', fontsize=9, rotation=90,
                          labelpad=6)

    # ── RIGHT: importance bar chart ──────────────────────────────
    axR.barh(y_pos, top_imp, color=PLASMA_BAR, edgecolor='black',
                linewidth=0.4, height=0.85)
    axR.set_yticks(y_pos)
    axR.set_yticklabels([])   # names already on the left axis
    axR.set_xlabel('Importance (mean(|SHAP value|))')
    axR.text(0.97, 0.04, behavior, transform=axR.transAxes,
                ha='right', va='bottom', fontsize=10, color=TEXT_GREY,
                fontstyle='italic')
    axR.grid(axis='x', color=GRID_GREY, linewidth=0.6, zorder=1)
    axR.set_axisbelow(True)
    axR.tick_params(left=False, pad=2)
    axR.spines['left'].set_visible(False)
    axR.set_ylim(-0.7, n_top - 0.3)

    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# SHAP-based pruning + classifier persistence (PixelPaws .pkl format)
# ─────────────────────────────────────────────────────────────────────

def prune_and_save_classifier(final_model, X: pd.DataFrame, y: np.ndarray,
                                 oof_proba: np.ndarray,
                                 fold_f1: list,
                                 training_sessions: list,
                                 behavior: str,
                                 classifier_dir: str,
                                 n_top: int = 120,
                                 sample_n: int = 8000):
    """Compute SHAP feature importance from the trained final model,
    keep the top-N most-impactful columns, retrain a pruned model on
    that subset, and save both classifiers (AllFeatures + pruned) as
    PixelPaws-compatible .pkl bundles. Mirrors the GUI's training
    output so these classifiers can be loaded by the existing
    prediction/evaluation tabs."""
    import shap
    import joblib
    import json

    os.makedirs(classifier_dir, exist_ok=True)

    # ── Compute SHAP on a sample ───────────────────────────────────
    rng = np.random.default_rng(RANDOM_SEED)
    if len(X) > sample_n:
        sample_idx = rng.choice(len(X), size=sample_n, replace=False)
        X_s = X.iloc[sample_idx]
    else:
        X_s = X
    print(f'    [prune] SHAP on {len(X_s)} rows…')
    explainer = shap.TreeExplainer(final_model)
    sv = explainer.shap_values(X_s)
    if isinstance(sv, list):
        sv = sv[1]
    importance = np.abs(sv).mean(axis=0)
    feat_names = list(X.columns)
    order = np.argsort(importance)[::-1]
    top_features = [feat_names[i] for i in order[:n_top]]
    top_importance = importance[order[:n_top]].astype(float)

    # ── Retrain pruned model on top-N ──────────────────────────────
    print(f'    [prune] retraining on top {n_top} of {X.shape[1]} '
          f'features…')
    X_pruned = X[top_features]
    pos_w = (y == 0).sum() / max((y == 1).sum(), 1)
    pruned_model = _make_xgb(scale_pos_weight=pos_w, n_estimators=400)
    pruned_model.fit(X_pruned, y)

    # ── Best threshold from OOF ────────────────────────────────────
    from sklearn.metrics import f1_score
    ths = np.linspace(0.05, 0.99, 95)
    f1s = [f1_score(y, (oof_proba >= t).astype(int), zero_division=0)
              for t in ths]
    best_idx = int(np.argmax(f1s))
    best_thresh = float(ths[best_idx])
    best_f1 = float(f1s[best_idx])

    def _bundle(model, n_estimators):
        cols = (list(model.feature_names_in_)
                  if hasattr(model, 'feature_names_in_') else None)
        return {
            'clf_model': model,
            'Behavior_type': behavior,
            'selected_feature_cols': cols,
            'best_thresh': best_thresh,
            'min_bout': 1,
            'min_after_bout': 0,
            'max_gap': 0,
            'ui_min_bout': 1,
            'ui_min_after_bout': 0,
            'ui_max_gap': 0,
            'bp_pixbrt_list': ['hrpaw', 'hlpaw', 'snout'],
            'square_size': [40, 40, 15],
            'pix_threshold': 0.4,
            'include_optical_flow': False,
            'bp_optflow_list': [],
            'training_sessions': list(training_sessions),
            'cv_f1_scores': list(fold_f1) if fold_f1 else [],
            'mean_cv_f1': float(np.mean(fold_f1)) if fold_f1 else 0.0,
            'std_cv_f1': float(np.std(fold_f1)) if fold_f1 else 0.0,
            'oof_best_f1': best_f1,
            'final_n_estimators': int(n_estimators),
            'scale_pos_weight': float(pos_w),
            'optuna_best_params': None,
            'oof_proba': np.asarray(oof_proba),
            'oof_best_params': {
                'thresh': best_thresh, 'min_bout': 1,
                'min_after_bout': 0, 'max_gap': 0, 'f1': best_f1,
            },
            'use_egocentric': any(c.startswith('Ego_')
                                       for c in (cols or feat_names)),
            'use_lag_features': any('_lag' in c
                                          for c in (cols or feat_names)),
            'use_contact_features': any(c.endswith('_ContactState')
                                              for c in (cols or feat_names)),
            'use_multiscale_features': any(
                bool(re.search(r'_(std|max)_\d+ms$', c))
                for c in (cols or feat_names)),
            'contact_threshold': 15.0,
        }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    all_path = os.path.join(
        classifier_dir,
        f'PixelPaws_{behavior}_AllFeatures_{timestamp}.pkl')
    joblib.dump(_bundle(final_model, n_estimators=400), all_path)
    print(f'    [prune] AllFeatures → {all_path}')

    pruned_path = os.path.join(
        classifier_dir,
        f'PixelPaws_{behavior}_pruned_{n_top}_{timestamp}.pkl')
    joblib.dump(_bundle(pruned_model, n_estimators=400), pruned_path)
    print(f'    [prune] pruned_{n_top} → {pruned_path}')

    # SHAP importance JSON sidecar
    json_path = os.path.join(
        classifier_dir,
        f'PixelPaws_{behavior}_pruned_{n_top}_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump({
            'behavior': behavior,
            'all_features_count': int(X.shape[1]),
            'pruned_to': int(n_top),
            'top_features': top_features,
            'top_importance': top_importance.tolist(),
            'best_thresh': best_thresh,
            'oof_best_f1': best_f1,
            'training_sessions': list(training_sessions),
            'cv_f1_scores': list(fold_f1) if fold_f1 else [],
            'mean_cv_f1': float(np.mean(fold_f1)) if fold_f1 else 0.0,
        }, f, indent=2)
    print(f'    [prune] importance JSON → {json_path}')

    return {
        'all_features_path': all_path,
        'pruned_path': pruned_path,
        'json_path': json_path,
        'top_features': top_features,
        'top_importance': top_importance,
        'pruned_model': pruned_model,
        'best_thresh': best_thresh,
        'best_f1': best_f1,
    }


# Imports needed by prune_and_save_classifier
import re


# ─────────────────────────────────────────────────────────────────────
# Per-video metrics — F1 / Precision / Recall grouped bar plot
# ─────────────────────────────────────────────────────────────────────

# Order matches `training_sessions` from the cached classifier pkl;
# row counts are the per-session feature counts that go into the
# concatenated training set (sum = 495 594 = X.shape[0]). Verified
# against the per-frame label files: S1 matches exactly; the others
# drift by ≤ 21 positives due to edge-frame trimming during feature
# extraction, which doesn't affect the session boundary in `y`.
SESSION_SPLITS = [
    ('251030_S1_Rimonabant',    100362),
    ('251030_S2_Rimonabant',    105439),
    ('251030_S3_Rimonabant',    117349),
    ('251031_S4_F_Rimonabant',   72232),
    ('251031_S5_F_Rimonabant',  100212),
]


def _session_ids_array() -> np.ndarray:
    """Per-row session index in the concatenated training set order."""
    ids = np.empty(sum(n for _, n in SESSION_SPLITS), dtype=np.int32)
    s = 0
    for i, (_, n) in enumerate(SESSION_SPLITS):
        ids[s:s + n] = i
        s += n
    return ids


def lovo_oof(X, y: np.ndarray, n_estimators: int = 200
              ) -> tuple[np.ndarray, dict[str, float]]:
    """Leave-one-video-out CV — each fold holds out an entire session.
    Returns LOVO OOF probabilities + per-fold F1@0.5."""
    from sklearn.model_selection import GroupKFold
    sids = _session_ids_array()
    n_sess = len(SESSION_SPLITS)
    gkf = GroupKFold(n_splits=n_sess)
    oof = np.zeros(len(y), dtype=np.float32)
    per_session = {}
    pos_w = (y == 0).sum() / max((y == 1).sum(), 1)
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=sids), 1):
        held = int(sids[te[0]])
        held_name = SESSION_SPLITS[held][0]
        t0 = time.time()
        clf = _make_xgb(scale_pos_weight=pos_w, n_estimators=n_estimators)
        clf.fit(X.iloc[tr] if isinstance(X, pd.DataFrame) else X[tr],
                  y[tr])
        proba = clf.predict_proba(
            X.iloc[te] if isinstance(X, pd.DataFrame) else X[te])[:, 1]
        oof[te] = proba
        f1 = f1_score(y[te], (proba >= 0.5).astype(int), zero_division=0)
        per_session[held_name] = float(f1)
        print(f'    LOVO fold {fold}/{n_sess}  held out {held_name}  '
              f'F1@0.5={f1:.3f}  ({time.time()-t0:.1f}s)')
    return oof, per_session


def plot_per_video_metrics(y: np.ndarray, oof: np.ndarray,
                              threshold: float, behavior: str,
                              out_path: str):
    """Grouped bar plot — three bars per session (F1, Precision,
    Recall), plasma palette, threshold applied uniformly across
    sessions. Sessions ordered as in the training set."""
    starts = np.cumsum([0] + [n for _, n in SESSION_SPLITS])
    rows = []
    for (name, _), s, e in zip(SESSION_SPLITS, starts[:-1], starts[1:]):
        y_s, p_s = y[s:e], oof[s:e]
        pred = (p_s >= threshold).astype(int)
        if y_s.sum() == 0:
            f1 = prec = rec = float('nan')
        else:
            f1   = f1_score(y_s, pred, zero_division=0)
            prec = precision_score(y_s, pred, zero_division=0)
            rec  = recall_score(y_s, pred, zero_division=0)
        rows.append({'session': name.replace('_Rimonabant', ''),
                       'F1': f1, 'Precision': prec, 'Recall': rec,
                       'n_pos': int(y_s.sum()),
                       'n_total': int(len(y_s))})

    metrics = ['Precision', 'Recall', 'F1']
    metric_colors = [PLASMA(0.30), PLASMA(0.55), PLASMA(0.80)]
    sessions = [r['session'] for r in rows]
    n_sess   = len(sessions)
    n_metric = len(metrics)
    bar_w    = 0.26
    x_base   = np.arange(n_sess)

    fig, ax = plt.subplots(figsize=(7.5, 4.0),
                              constrained_layout=True)
    for i, m in enumerate(metrics):
        vals = np.array([r[m] for r in rows])
        offsets = (i - (n_metric - 1) / 2) * bar_w
        bars = ax.bar(x_base + offsets, vals, bar_w,
                         color=metric_colors[i], edgecolor='black',
                         linewidth=0.4, label=m, zorder=3)
        # Value label on top of each bar
        for xb, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(xb.get_x() + xb.get_width() / 2,
                          v + 0.012, f'{v:.2f}',
                          ha='center', va='bottom', fontsize=7.5,
                          color=TEXT_GREY)
    ax.set_xticks(x_base)
    ax.set_xticklabels(sessions, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_ylabel('Score')
    ax.set_title(f'Per-video metrics — {behavior}  '
                    f'(threshold = {threshold:.2f})',
                    fontsize=11, fontweight='bold')
    ax.grid(axis='y', color=GRID_GREY, linewidth=0.6, zorder=1)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=n_metric, loc='lower center',
                bbox_to_anchor=(0.5, -0.18), fontsize=9)

    # Annotate sessions with 0 positives so the empty bars aren't a
    # mystery (S3 in this project — silver-truth label source differs)
    for j, r in enumerate(rows):
        if r['n_pos'] == 0:
            ax.text(j, 0.05, '(no labels)', ha='center', va='bottom',
                      fontsize=7.5, color='#888', fontstyle='italic')

    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    # Console summary
    print('    per-video metrics:')
    for r in rows:
        if r['n_pos'] == 0:
            print(f'      {r["session"]:14s}  (no labels — {r["n_total"]} fr)')
        else:
            print(f'      {r["session"]:14s}  '
                    f'P={r["Precision"]:.3f}  R={r["Recall"]:.3f}  '
                    f'F1={r["F1"]:.3f}   '
                    f'(n_pos={r["n_pos"]}/{r["n_total"]})')


def plot_per_session_diagnostics(y: np.ndarray, oof: np.ndarray,
                                    threshold: float, behavior: str,
                                    out_path: str,
                                    fps: float = 60.0,
                                    bin_sec: int = 30):
    """Multi-row figure mirroring the BAREfoot per-behavior diagnostic
    layout, but rows = sessions (since this project has one behaviour).
    Each row: raster (Human top / Model bottom) + 2×2 confusion matrix
    + time-binned `sec/bin` heatmap (Human top / Model bottom) with
    Pearson R between the two."""
    from sklearn.metrics import confusion_matrix

    accent     = PLASMA(0.55)         # plasma magenta — single behaviour
    # Gradient from a faint tint (≈92 % toward white) → accent. Pure
    # white at the low end made the heatmap / confusion-matrix cells
    # feel disconnected from the solid-accent raster bars; this keeps
    # all three panels reading as the same colour family even where
    # the data is sparse.
    pale = tuple(c + (1 - c) * 0.92 for c in accent[:3]) + (1.0,)
    cmap_acc = LinearSegmentedColormap.from_list(
                    'plasma_tint', [pale, accent])
    starts = np.cumsum([0] + [n for _, n in SESSION_SPLITS])
    n_sess = len(SESSION_SPLITS)

    fig = plt.figure(figsize=(15, 2.6 * n_sess))
    gs = fig.add_gridspec(n_sess, 3, width_ratios=[5, 1.2, 3.5],
                              hspace=0.55, wspace=0.30,
                              left=0.06, right=0.98,
                              top=0.96, bottom=0.05)

    bin_size = int(round(fps * bin_sec))

    for i, (name, _) in enumerate(SESSION_SPLITS):
        s, e = int(starts[i]), int(starts[i + 1])
        y_s, p_s = y[s:e], oof[s:e]
        pred = (p_s >= threshold).astype(int)
        n_frames = len(y_s)

        # ── (i, 0) Raster: Human (top) vs Model (bottom) ───────────
        ax_r = fig.add_subplot(gs[i, 0])
        human_events = np.where(y_s == 1)[0]
        model_events = np.where(pred == 1)[0]
        ax_r.eventplot([model_events, human_events],
                          lineoffsets=[0.25, 0.75], linelengths=0.40,
                          colors=accent, linewidths=0.5)
        ax_r.axhline(0.50, color='black', lw=0.4)
        ax_r.set_yticks([0.25, 0.75])
        ax_r.set_yticklabels(['Model', 'Human'], fontsize=9)
        ax_r.set_xlim(0, n_frames)
        ax_r.set_ylim(0, 1)
        ax_r.set_xlabel('Frame', fontsize=9)
        ax_r.set_ylabel(name.replace('_Rimonabant', ''),
                          fontsize=10, fontweight='bold', labelpad=10)
        ax_r.tick_params(labelsize=8)
        ax_r.spines['top'].set_visible(False)
        ax_r.spines['right'].set_visible(False)

        # ── (i, 1) Confusion matrix ─────────────────────────────────
        ax_c = fig.add_subplot(gs[i, 1])
        cm = confusion_matrix(y_s, pred, normalize='true',
                                  labels=[0, 1])
        ax_c.imshow(cm, cmap=cmap_acc, vmin=0, vmax=1, aspect='equal')
        ax_c.set_xticks([0, 1]); ax_c.set_yticks([0, 1])
        ax_c.set_xticklabels(['0', '1'], fontsize=9)
        ax_c.set_yticklabels(['0', '1'], fontsize=9)
        ax_c.set_xlabel('Predicted label', fontsize=9)
        ax_c.set_ylabel('True label', fontsize=9)
        for r_ in range(2):
            for c_ in range(2):
                txt_color = 'white' if cm[r_, c_] > 0.55 else 'black'
                ax_c.text(c_, r_, f'{cm[r_, c_]:.2f}',
                            ha='center', va='center',
                            color=txt_color, fontsize=9)
        f1 = f1_score(y_s, pred, zero_division=0)
        ax_c.set_title(f'F1-Score: {f1:.2f}', fontsize=9,
                          fontweight='bold')
        ax_c.tick_params(length=0)
        for s_ in ('top', 'right', 'bottom', 'left'):
            ax_c.spines[s_].set_visible(False)

        # ── (i, 2) Time-binned heatmap ──────────────────────────────
        ax_h = fig.add_subplot(gs[i, 2])
        n_bins = (n_frames + bin_size - 1) // bin_size
        human_b = np.zeros(n_bins)
        model_b = np.zeros(n_bins)
        for b in range(n_bins):
            bs, be = b * bin_size, min((b + 1) * bin_size, n_frames)
            human_b[b] = y_s[bs:be].sum() / fps
            model_b[b] = pred[bs:be].sum() / fps
        heatmap = np.vstack([human_b, model_b])
        vmax = max(human_b.max(), model_b.max(), 1e-6)
        im = ax_h.imshow(heatmap, cmap=cmap_acc, aspect='auto',
                            interpolation='nearest', vmin=0, vmax=vmax)
        ax_h.set_yticks([0, 1])
        ax_h.set_yticklabels(['Human', 'Model'], fontsize=9)
        ax_h.set_xlabel('Time (sec)', fontsize=9)
        # X ticks every ~30 sec at bin centres
        n_xticks = min(7, n_bins)
        x_idx = np.linspace(0, n_bins - 1, n_xticks).astype(int)
        ax_h.set_xticks(x_idx)
        ax_h.set_xticklabels([f'{int(t * bin_sec)}' for t in x_idx],
                                  fontsize=8)
        ax_h.tick_params(left=False)
        # Pearson R top-right
        if human_b.std() > 0 and model_b.std() > 0:
            r_corr = float(np.corrcoef(human_b, model_b)[0, 1])
            ax_h.text(0.98, 1.06, f'R = {r_corr:.2f}',
                         transform=ax_h.transAxes, ha='right',
                         va='bottom', fontsize=9, fontweight='bold',
                         color=TEXT_GREY)
        # Colorbar to the right of the heatmap
        cax = ax_h.inset_axes([1.02, 0.10, 0.025, 0.80])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(f'{behavior} (sec/bin)', fontsize=7.5, rotation=90,
                       labelpad=4)
        cb.ax.tick_params(labelsize=7)

    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    if not os.path.isfile(TRAIN_PKL):
        sys.exit(f'training set not found: {TRAIN_PKL}')
    os.makedirs(PLOTS_BASE, exist_ok=True)

    print(f'[1/5] loading training set…')
    td = pd.read_pickle(TRAIN_PKL)
    X, y = td['X'], np.asarray(td['y'], dtype=np.int64)
    print(f'      X: {X.shape}   y pos: {y.sum()} '
          f'({y.mean()*100:.2f}%)')

    print(f'[2/5] {N_FOLDS}-fold OOF CV (re-training)…')
    oof, fold_f1 = cv_oof_proba(X, y, n_folds=N_FOLDS,
                                  n_estimators=300)
    print(f'      mean F1 (per-fold) = {np.mean(fold_f1):.3f}±'
          f'{np.std(fold_f1):.3f}')

    print(f'[3/5] panel C — threshold curves (from fresh OOF)…')
    pc_path = os.path.join(PLOTS_BASE,
                              f'PixelPaws_{BEHAVIOR}_panelC_threshold.png')
    plot_threshold_curves(y, oof, BEHAVIOR, pc_path)
    print(f'      → {pc_path}')

    print(f'[4/5] training final model on full set for SHAP…')
    final = train_full_model(X, y)
    print(f'[5a/5] panel D — bulbous SHAP beeswarm + importance…')
    pd_path = os.path.join(PLOTS_BASE,
                              f'PixelPaws_{BEHAVIOR}_panelD_shap.png')
    plot_shap_panel(final, X, BEHAVIOR, pd_path)
    print(f'      → {pd_path}')

    print(f'[5b/5] panel B — learning curve (CV sweep)…')
    lc = learning_curve(X, y,
                          fractions=[0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00],
                          n_folds=3, n_estimators=150)
    pb_path = os.path.join(PLOTS_BASE,
                              f'PixelPaws_{BEHAVIOR}_panelB_learning.png')
    plot_learning_curve(lc, BEHAVIOR, pb_path)
    print(f'      → {pb_path}')

    # Persist the fresh OOF + final model so we don't have to retrain
    # next time we tweak the plots.
    cache_path = os.path.join(PLOTS_BASE, '_regen_cache.pkl')
    pd.to_pickle({'oof': oof, 'fold_f1': fold_f1,
                    'lc': lc, 'final_model': final}, cache_path)
    print(f'      cache → {cache_path}')

    print(f'\n✓ done. Outputs → {PLOTS_BASE}')


if __name__ == '__main__':
    main()
