"""
Locomotor analysis for THC withdrawal cohort (first 25 min only):
- Total distance traveled (centroid Euclidean displacement, pixels)
- Mean velocity (pixels/s)
- % time moving (velocity above threshold)
- % time immobile (from `still` classifier — for comparison with the kinematic measure)

Output: <project>/analysis_25min/locomotor.csv + 4 PNGs + Discord push.

Run:
  PYTHONIOENCODING=utf-8 py -X utf8 scripts/research/thc_25min_locomotor.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(r'E:\RSVIDS\Blackbox\260506_RS_THC_Withdrawal')
VIDEO_DIR = PROJECT_ROOT / 'Videos'
RESULTS_DIR = PROJECT_ROOT / 'results'
ANALYSIS_DIR = PROJECT_ROOT / 'analysis_25min'
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

WEBHOOK = (
    ""
)

FPS = 60
MAX_MINUTES = 25
MAX_FRAMES = MAX_MINUTES * 60 * FPS  # 90000

# Movement threshold: centroid moved > MOVE_THRESHOLD_PX between consecutive
# frames counts as "moving". 60 fps → 1 px/frame = 60 px/s.
# A typical resting mouse has < 1 px/frame jitter from DLC noise; walking
# is ~5-15 px/frame.
MOVE_THRESHOLD_PX = 1.0

# Pose likelihood floor for centroid frames (drop frames with poor tracking)
LIKELIHOOD_FLOOR = 0.5


def parse_session(name: str):
    m = re.match(r'^THC(\d+)_(Baseline|Postdrug)$', name)
    if not m:
        return None
    n = int(m.group(1))
    cond = m.group(2)
    group = 'THC' if (n % 2 == 1) else 'Vehicle'
    return n, cond, group


def find_h5(base: str) -> Path | None:
    """Find the filtered shuffle9 .h5 next to the video for this session."""
    candidates = list(VIDEO_DIR.glob(f'{base}*shuffle9*_filtered.h5'))
    if not candidates:
        candidates = list(VIDEO_DIR.glob(f'{base}*shuffle9*.h5'))
    return candidates[0] if candidates else None


def load_centroid(h5_path: Path):
    """Returns (x, y, likelihood) numpy arrays for the centroid bodypart."""
    df = pd.read_hdf(h5_path)
    # MultiIndex columns: (scorer, bodypart, coord)
    scorer = df.columns.get_level_values(0)[0]
    x = df[(scorer, 'centroid', 'x')].to_numpy()
    y = df[(scorer, 'centroid', 'y')].to_numpy()
    lik = df[(scorer, 'centroid', 'likelihood')].to_numpy()
    return x, y, lik


def compute_metrics(x, y, lik, max_frames: int = MAX_FRAMES, fps: int = FPS):
    """Restrict to first max_frames, mask low-likelihood frames, then compute
    distance, velocity, and % time moving."""
    n = min(len(x), max_frames)
    x = x[:n]; y = y[:n]; lik = lik[:n]

    valid = lik >= LIKELIHOOD_FLOOR
    # Frame-to-frame displacement; only when both endpoints are valid
    dx = np.diff(x)
    dy = np.diff(y)
    dist = np.sqrt(dx * dx + dy * dy)
    valid_pair = valid[:-1] & valid[1:]
    dist_valid = dist[valid_pair]
    n_valid_pairs = int(valid_pair.sum())

    if n_valid_pairs == 0:
        return dict(
            total_distance_px=0.0, mean_velocity_px_s=0.0,
            pct_moving=0.0, n_frames=n, n_valid_pairs=0,
        )

    total_dist = float(dist_valid.sum())
    duration_s = n_valid_pairs / fps
    mean_vel = total_dist / duration_s if duration_s > 0 else 0.0
    moving_mask = dist_valid > MOVE_THRESHOLD_PX
    pct_moving = 100.0 * moving_mask.sum() / n_valid_pairs

    return dict(
        total_distance_px=round(total_dist, 1),
        mean_velocity_px_s=round(mean_vel, 2),
        pct_moving=round(pct_moving, 2),
        n_frames=n,
        n_valid_pairs=n_valid_pairs,
    )


def plot_box_by_group(df, value_col, ylabel, title, out_path):
    """Two-panel: Postdrug (primary) and Baseline, THC vs Vehicle box."""
    palette = {'THC': '#d62728', 'Vehicle': '#1f77b4'}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def _box(ax, sub_df, group_col, panel_title):
        groups = sorted(sub_df[group_col].unique())
        data = [sub_df[sub_df[group_col] == g][value_col].to_numpy().tolist()
                for g in groups]
        bp = ax.boxplot(data, tick_labels=groups, patch_artist=True, widths=0.55,
                        medianprops=dict(color='black'))
        for patch, g in zip(bp['boxes'], groups):
            patch.set_facecolor(palette.get(g, '#888'))
            patch.set_alpha(0.6)
        for i, (g, ys) in enumerate(zip(groups, data), 1):
            if ys:
                xs = np.random.normal(i, 0.05, len(ys))
                ax.scatter(xs, ys, color='black', s=20, zorder=3)
                ax.text(i, max(ys) * 1.02, f'n={len(ys)}',
                        ha='center', fontsize=9, color='gray')
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)
        ax.grid(axis='y', alpha=0.3)

    pd_sub = df[df['condition'] == 'Postdrug']
    _box(axes[0], pd_sub, 'group', 'POSTDRUG: THC vs Vehicle (primary)')
    bl_sub = df[df['condition'] == 'Baseline']
    _box(axes[1], bl_sub, 'group', 'Baseline: THC vs Vehicle')

    # Within-subject paired
    ax = axes[2]
    for grp in ['THC', 'Vehicle']:
        sub = df[df['group'] == grp]
        for mouse, mdf in sub.groupby('mouse'):
            bl = mdf[mdf['condition'] == 'Baseline'][value_col]
            po = mdf[mdf['condition'] == 'Postdrug'][value_col]
            if len(bl) and len(po):
                ax.plot([0, 1], [bl.iloc[0], po.iloc[0]],
                        'o-', color=palette[grp], alpha=0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Baseline', 'Postdrug'])
    ax.set_ylabel(ylabel)
    ax.set_title('Within-subject: Baseline -> Postdrug')
    ax.grid(axis='y', alpha=0.3)
    handles = [plt.Line2D([0], [0], color=palette[g], marker='o',
                          linestyle='-', label=g) for g in ['THC', 'Vehicle']]
    ax.legend(handles=handles, loc='best')

    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def post_to_discord(text: str, png_paths: list) -> None:
    """Multipart upload via curl."""
    cmd = ['curl', '-s', '-o', os.devnull, '-w', '%{http_code}', '-X', 'POST',
           '-F', f'payload_json={{"content":"{text}"}}']
    for i, p in enumerate(png_paths):
        cmd += ['-F', f'file{i}=@{p}']
    cmd.append(WEBHOOK)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f'Discord upload: HTTP {r.stdout}')


def main() -> int:
    rows = []

    # Build list of sessions from prediction CSVs (we want to align with the
    # group-analysis cohort and pull the `still` % from the predictions).
    pred_csvs = sorted(RESULTS_DIR.glob('*_predictions.csv'))
    for csv in pred_csvs:
        base = csv.name.replace('_predictions.csv', '')
        parsed = parse_session(base)
        if not parsed:
            continue
        n, cond, grp = parsed

        h5 = find_h5(base)
        if not h5:
            print(f'  ! no DLC h5 for {base}')
            continue

        try:
            x, y, lik = load_centroid(h5)
        except Exception as e:
            print(f'  ! failed to read {h5.name}: {e}')
            continue

        m = compute_metrics(x, y, lik)

        # Pull `still` % from the predictions CSV (truncated to 25 min)
        pred = pd.read_csv(csv).head(MAX_FRAMES)
        if 'still' in pred.columns:
            pct_still = 100.0 * (pred['still'].astype(int) == 1).mean()
        else:
            pct_still = float('nan')
        m['pct_still_classifier'] = round(pct_still, 2)

        m.update({'session': base, 'mouse': n, 'condition': cond, 'group': grp})
        rows.append(m)
        print(f'  {base:>20}  dist={m["total_distance_px"]:>10.1f}px  '
              f'vel={m["mean_velocity_px_s"]:>6.2f}px/s  '
              f'%moving={m["pct_moving"]:>5.2f}  '
              f'%still(clf)={m["pct_still_classifier"]:>5.2f}')

    if not rows:
        print('No sessions processed.')
        return 1

    df = pd.DataFrame(rows)
    out_csv = ANALYSIS_DIR / 'locomotor.csv'
    df.to_csv(out_csv, index=False)
    print(f'Wrote: {out_csv}')

    p_dist = ANALYSIS_DIR / 'locomotor_distance.png'
    p_vel = ANALYSIS_DIR / 'locomotor_velocity.png'
    p_move = ANALYSIS_DIR / 'locomotor_pct_moving.png'
    p_still = ANALYSIS_DIR / 'locomotor_pct_still_clf.png'

    plot_box_by_group(df, 'total_distance_px',
                      'Total distance (pixels, first 25 min)',
                      'Distance traveled — THC withdrawal (25-min window)',
                      p_dist)
    plot_box_by_group(df, 'mean_velocity_px_s',
                      'Mean velocity (px/s)',
                      'Mean velocity — THC withdrawal (25-min window)',
                      p_vel)
    plot_box_by_group(df, 'pct_moving',
                      f'% time moving (>{MOVE_THRESHOLD_PX} px/frame)',
                      'Percent time moving — THC withdrawal (25-min window)',
                      p_move)
    plot_box_by_group(df, 'pct_still_classifier',
                      '% time still (classifier)',
                      'Classifier-based immobility (25-min window)',
                      p_still)

    text = (
        f'THC withdrawal -- locomotor analysis (first 25 min, '
        f'{len(df)} sessions). Centroid keypoint, '
        f'likelihood floor {LIKELIHOOD_FLOOR}, move threshold '
        f'{MOVE_THRESHOLD_PX} px/frame at {FPS} fps. '
        f'Units in pixels (no calibration applied).'
    )
    post_to_discord(text, [p_dist, p_vel, p_move, p_still])
    return 0


if __name__ == '__main__':
    sys.exit(main())
