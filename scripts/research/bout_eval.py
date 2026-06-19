"""bout_eval.py — Bout-level Precision/Recall/F1 for the regen
LOVO/OOF predictions. Pairs frame-level metrics with the more
biologically meaningful bout-level matching (a predicted bout is a TP
if it overlaps any true bout; a true bout is recalled if any predicted
bout overlaps it).

Applies standard PixelPaws post-processing first:
  threshold → min_bout → max_gap

then extracts bout intervals from both y_true and y_pred and counts
overlap matches. Per-session + aggregate metrics, plus a bar plot.

Usage:
    py bout_eval.py <cache_folder> <behavior> [<min_bout=3> <max_gap=5>]
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                # peer scripts in this folder
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # project root (E:/PixelPaws)
from evaluation_tab import _apply_bout_filtering
from regen_scratching_plots import PLASMA, GRID_GREY, TEXT_GREY


def find_bouts(y_binary: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start, end_exclusive) bout intervals."""
    bouts = []
    in_bout = False
    start = 0
    for i, v in enumerate(y_binary):
        if v == 1 and not in_bout:
            start = i
            in_bout = True
        elif v == 0 and in_bout:
            bouts.append((start, i))
            in_bout = False
    if in_bout:
        bouts.append((start, len(y_binary)))
    return bouts


def bout_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Bout-level P/R/F1 with any-overlap matching."""
    true_bouts = find_bouts(y_true)
    pred_bouts = find_bouts(y_pred)

    def overlaps(a, b):
        return max(a[0], b[0]) < min(a[1], b[1])

    tp_pred = sum(1 for p in pred_bouts if any(overlaps(p, t) for t in true_bouts))
    matched_true = sum(1 for t in true_bouts if any(overlaps(p, t) for p in pred_bouts))

    n_pred = len(pred_bouts)
    n_true = len(true_bouts)
    precision = tp_pred / max(n_pred, 1)
    recall    = matched_true / max(n_true, 1)
    f1 = (2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0)
    return {
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'n_true_bouts':  n_true,
        'n_pred_bouts':  n_pred,
        'matched_true':  matched_true,
        'tp_pred':       tp_pred,
    }


def main():
    if len(sys.argv) < 3:
        sys.exit('usage: bout_eval.py <cache_folder> <behavior> '
                  '[min_bout=3 max_gap=5 use_lovo=1]')
    cache_dir = sys.argv[1]
    behavior  = sys.argv[2]
    min_bout  = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    max_gap   = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    use_lovo  = int(sys.argv[5]) if len(sys.argv) > 5 else 1

    cache = pd.read_pickle(os.path.join(cache_dir, '_regen_cache.pkl'))
    oof_key = 'oof_lovo' if (use_lovo and 'oof_lovo' in cache) else 'oof'
    oof = np.asarray(cache[oof_key], dtype=np.float64)
    splits = cache.get('splits', [])
    best_t = cache.get(
        'best_t_lovo' if oof_key == 'oof_lovo' else 'best_t', 0.5)

    # Load training y from same folder if cached, else fall back to a
    # sibling regen run that has it, else the project's training_data.
    train_pkl = os.path.join(cache_dir, '_training_set.pkl')
    if not os.path.isfile(train_pkl):
        # Try sibling regen folders (e.g. regen_aug_pruned_X falls back
        # to regen_aug_X). Look for any peer with a _training_set.pkl.
        plots_dir = os.path.dirname(cache_dir)
        for sib in sorted(os.listdir(plots_dir), reverse=True):
            cand = os.path.join(plots_dir, sib, '_training_set.pkl')
            if os.path.isfile(cand):
                train_pkl = cand
                print(f'[setup] sibling training set: {train_pkl}')
                break
    if os.path.isfile(train_pkl):
        td = pd.read_pickle(train_pkl)
    else:
        # cache_dir lives at .../classifiers/plots/regen_X → up TWO
        # levels lands on .../classifiers (sibling of plots/), so append
        # only training_data.
        train_dir = os.path.join(os.path.dirname(plots_dir),
                                     'training_data')
        if not os.path.isdir(train_dir):
            # Older project layout: training_data sits next to plots
            train_dir = os.path.join(os.path.dirname(plots_dir),
                                         'training_data')
        cands = [f for f in os.listdir(train_dir)
                    if f.endswith('.pkl') and behavior.lower() in f.lower()]
        if not cands:
            sys.exit(f'no training set found in {train_dir}')
        td = pd.read_pickle(os.path.join(train_dir, cands[0]))
    y = np.asarray(td['y'], dtype=np.int64)
    if len(y) != len(oof):
        sys.exit(f'oof/y length mismatch: oof={len(oof)} y={len(y)}')

    print(f'[setup] oof_key={oof_key}  best_t={best_t:.2f}  '
          f'n={len(y)}  pos={int(y.sum())}  '
          f'min_bout={min_bout}  max_gap={max_gap}')

    # Apply threshold + bout filter on the FULL prediction stream
    y_pred_raw = (oof >= best_t).astype(int)
    y_pred = _apply_bout_filtering(y_pred_raw.copy(), min_bout=min_bout,
                                       min_after_bout=0, max_gap=max_gap)

    # Aggregate frame-level (post-filter) for sanity
    from sklearn.metrics import precision_score, recall_score, f1_score
    frame_p = precision_score(y, y_pred, zero_division=0)
    frame_r = recall_score(y, y_pred, zero_division=0)
    frame_f = f1_score(y, y_pred, zero_division=0)
    print(f'[frame] P={frame_p:.3f}  R={frame_r:.3f}  F1={frame_f:.3f}  '
          '(after bout filtering)')

    # Aggregate bout-level
    agg = bout_metrics(y, y_pred)
    print(f'[bout ] P={agg["precision"]:.3f}  R={agg["recall"]:.3f}  '
          f'F1={agg["f1"]:.3f}  '
          f'(true={agg["n_true_bouts"]}, pred={agg["n_pred_bouts"]}, '
          f'matched={agg["matched_true"]}, tp_pred={agg["tp_pred"]})')

    # Per-session
    rows = []
    starts = np.cumsum([0] + [n for _, n in splits])
    for (sname, _), s, e in zip(splits, starts[:-1], starts[1:]):
        y_s, p_s = y[s:e], y_pred[s:e]
        m = bout_metrics(y_s, p_s)
        m['session'] = sname
        m['frame_f1'] = f1_score(y_s, p_s, zero_division=0)
        rows.append(m)
        print(f'  {sname:30s}  bout P={m["precision"]:.3f}  '
              f'R={m["recall"]:.3f}  F1={m["f1"]:.3f}   '
              f'(true={m["n_true_bouts"]}, pred={m["n_pred_bouts"]}, '
              f'matched={m["matched_true"]})  '
              f'frame_F1={m["frame_f1"]:.3f}')

    # ── Plot: per-session bout P/R/F1 (with frame F1 line for compare) ─
    accent = PLASMA(0.55)
    metric_colors = [PLASMA(0.30), PLASMA(0.55), PLASMA(0.80)]
    sessions = [r['session'].replace('_Rimonabant', '')
                                    .replace('260129_Formalin_', 'F_')
                                    .replace('251126_Formalin_F_', 'FS_')
                                    .replace('_Cut', '') for r in rows]
    n = len(rows)
    bar_w = 0.26
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(8.0, 0.6 * n + 4), 4.2),
                              constrained_layout=True)
    for i, m in enumerate(['precision', 'recall', 'f1']):
        vals = np.array([r[m] for r in rows])
        offs = (i - 1) * bar_w
        bars = ax.bar(x + offs, vals, bar_w,
                         color=metric_colors[i], edgecolor='black',
                         linewidth=0.4, label=m.capitalize(), zorder=3)
        for xb, v in zip(bars, vals):
            ax.text(xb.get_x() + xb.get_width() / 2, v + 0.012,
                      f'{v:.2f}', ha='center', va='bottom',
                      fontsize=7.5, color=TEXT_GREY)
    # frame-F1 ghost markers for direct compare
    frame_vals = np.array([r['frame_f1'] for r in rows])
    ax.plot(x, frame_vals, marker='x', color='#444', linestyle='None',
              markersize=8, label='Frame F1 (×)', zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(sessions, fontsize=9, rotation=0)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_ylabel('Score')
    ax.set_title(
        f'Bout-level vs Frame-level — {behavior}  '
        f'(thresh={best_t:.2f}, min_bout={min_bout}, max_gap={max_gap}, '
        f'{"LOVO" if oof_key == "oof_lovo" else "OOF"})',
        fontsize=11, fontweight='bold')
    ax.grid(axis='y', color=GRID_GREY, linewidth=0.6, zorder=1)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=4, loc='lower center',
                bbox_to_anchor=(0.5, -0.18), fontsize=9)

    out_path = os.path.join(
        cache_dir,
        f'PixelPaws_{behavior}_boutVsFrame_'
        f'{"LOVO" if oof_key == "oof_lovo" else "OOF"}.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\n→ {out_path}')

    # Save numeric summary
    summary_path = os.path.join(
        cache_dir,
        f'bout_metrics_{"LOVO" if oof_key == "oof_lovo" else "OOF"}.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'behavior': behavior,
            'oof_key': oof_key,
            'best_threshold': float(best_t),
            'min_bout': min_bout,
            'max_gap': max_gap,
            'aggregate': {
                'frame_precision': float(frame_p),
                'frame_recall':    float(frame_r),
                'frame_f1':        float(frame_f),
                'bout_precision':  float(agg['precision']),
                'bout_recall':     float(agg['recall']),
                'bout_f1':         float(agg['f1']),
                'n_true_bouts':    int(agg['n_true_bouts']),
                'n_pred_bouts':    int(agg['n_pred_bouts']),
            },
            'per_session': [{
                'session': r['session'],
                'bout_precision': float(r['precision']),
                'bout_recall':    float(r['recall']),
                'bout_f1':        float(r['f1']),
                'frame_f1':       float(r['frame_f1']),
                'n_true_bouts':   int(r['n_true_bouts']),
                'n_pred_bouts':   int(r['n_pred_bouts']),
            } for r in rows],
        }, f, indent=2)
    print(f'→ {summary_path}')


if __name__ == '__main__':
    main()
