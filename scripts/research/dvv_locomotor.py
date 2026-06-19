"""
Locomotor analysis for the 2604_DV_DSS Pdyn-ChR2 cohort.

Compares BL vs STIM kinematic measures, separated by genotype:
  - ai32         : ChR2-only controls (no Cre)
  - pdyn_ai32    : Pdyn-Cre x Ai32 (ChR2 expressed in Pdyn+ neurons)

Metrics from centroid keypoint, first 25 min of each session:
  - Total distance traveled (pixels)
  - Mean velocity (px/s)
  - % time moving (centroid >MOVE_THRESHOLD_PX between consecutive frames)

Reads ALL existing DLC .h5 in videos/ (works for whatever subset is already
analyzed; safe to re-run as new .h5 land).

Output: <project>/analysis/locomotor.csv + 3 PNGs + Discord push.

Run:
  PYTHONIOENCODING=utf-8 py -X utf8 scripts/research/dvv_locomotor.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(r"E:\RSVIDS\Blackbox\2604_DV_DSS")
VIDEO_DIR = PROJECT_ROOT / "videos"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

WEBHOOK = (
    ""
)

FPS = 60
MAX_MINUTES = 25
MAX_FRAMES = MAX_MINUTES * 60 * FPS  # 90,000
BIN_MINUTES = 5
BIN_FRAMES = BIN_MINUTES * 60 * FPS   # 18,000

# Movement detection parameters.
#   - MOVE_THRESHOLD_PX: per-frame tailbase displacement above this counts
#     as "moving". 1 px/frame = 60 px/s ~= 0.9 cm/s with ~0.15 mm/px (right
#     at the DLC jitter floor). 2 px/frame ~= 1.8 cm/s is firmly above
#     jitter for stable keypoints like tailbase.
#   - SMOOTH_WINDOW_FRAMES: rolling-mean window applied to the displacement
#     signal before thresholding. Removes single-frame DLC blips while
#     preserving multi-frame movement onset. 10 frames @ 60 fps = 167 ms.
#   - MIN_BOUT_FRAMES: contiguous "moving" frames shorter than this get
#     re-labeled as still. Filters out brief transient blips. 0 disables.
MOVE_THRESHOLD_PX = 2.0
SMOOTH_WINDOW_FRAMES = 10
MIN_BOUT_FRAMES = 0
LIKELIHOOD_FLOOR = 0.5

# When True, restrict analysis to mice that have BOTH BL and STIM
# sessions. Unmatched singletons are dropped from the per-group plots
# and summary stats (they still appear in the raw CSV, just not in the
# group analysis).
MATCHED_ONLY = True

# Reference keypoint(s) for whole-animal position. tailbase has 93-97% valid
# frames on this cohort vs centroid's 47-71%; it's also the standard anchor
# for mouse locomotion (high contrast, low body-relative motion).
#
# When BODY_KEYPOINTS has >1 entry, the body position is the mean across the
# listed keypoints (using only frames where ALL of them clear LIKELIHOOD_FLOOR
# — a per-keypoint masked mean is too noisy when one drops out). Single
# keypoint = use that keypoint directly.
BODY_KEYPOINTS = ["tailbase"]   # set to ["tailbase","hrpaw","hlpaw","snout"] for multi-kp body centroid

PALETTE = {"ai32": "#7f7f7f", "pdyn_ai32": "#d62728"}
LINESTYLE = {"BL": "--", "STIM": "-"}


def parse_session(name: str):
    """Returns (mouse:int, condition:str, genotype:str) or None.

    Filename patterns seen in this cohort:
        202604_Ai32_61_BL_cropped
        202604_Pdyn_AI32_21_BL_cropped
        202604_Pdyn_ai32_19_STIM_cropped
        2604_ai32_60_BL_cropped
        2604_pdyn_ai32_18_BL_cropped
    Optional `2026?604_` prefix, optional `Pdyn_` prefix (case-insensitive)
    before the `ai32` token.
    """
    # Optional year-month prefix: 2604_ or 202604_ (or other 4-6 digit run).
    pat = re.compile(
        r"^(?:\d{4,6}_)?(?P<genotype>(?:pdyn_)?ai32)_(?P<mouse>\d+)_"
        r"(?P<cond>BL|STIM)(?:_cropped)?$",
        re.IGNORECASE,
    )
    m = pat.match(name)
    if not m:
        return None
    genotype_raw = m.group("genotype").lower()
    genotype = "pdyn_ai32" if "pdyn" in genotype_raw else "ai32"
    return int(m.group("mouse")), m.group("cond").upper(), genotype


def load_centroid(h5_path: Path):
    """Return per-frame (x, y, likelihood) for the configured body keypoint(s).

    Single keypoint -> direct passthrough.
    Multi-keypoint -> per-frame mean of the listed keypoints. The output
    `likelihood` is the MINIMUM across keypoints (a frame counts only when
    ALL keypoints exceed the floor — averaging position over a partial
    set otherwise introduces large jumps when one keypoint drops out).
    """
    df = pd.read_hdf(h5_path)
    scorer = df.columns.get_level_values(0)[0]
    xs, ys, lks = [], [], []
    for bp in BODY_KEYPOINTS:
        xs.append(df[(scorer, bp, "x")].to_numpy())
        ys.append(df[(scorer, bp, "y")].to_numpy())
        lks.append(df[(scorer, bp, "likelihood")].to_numpy())
    xs = np.stack(xs, axis=0)
    ys = np.stack(ys, axis=0)
    lks = np.stack(lks, axis=0)
    if xs.shape[0] == 1:
        return xs[0], ys[0], lks[0]
    # Multi-keypoint: mean position, min likelihood
    return xs.mean(axis=0), ys.mean(axis=0), lks.min(axis=0)


def _smoothed_displacement(x, y, valid_pair, smooth_w):
    """Per-frame displacement (length matches valid_pair), with rolling-mean
    smoothing over `smooth_w` frames. Invalid pairs contribute 0 to the
    displacement (already masked elsewhere)."""
    dx = np.diff(x); dy = np.diff(y)
    dist = np.sqrt(dx * dx + dy * dy)
    if smooth_w <= 1:
        return dist
    # Centered rolling mean, edges handled by uniform-weight convolution.
    kernel = np.ones(smooth_w) / smooth_w
    smoothed = np.convolve(dist, kernel, mode="same")
    return smoothed


def _apply_min_bout(moving: np.ndarray, min_bout: int) -> np.ndarray:
    """Drop contiguous moving-runs shorter than min_bout frames."""
    if min_bout <= 1 or moving.sum() == 0:
        return moving
    out = moving.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i]:
            j = i
            while j < n and out[j]:
                j += 1
            if j - i < min_bout:
                out[i:j] = False
            i = j
        else:
            i += 1
    return out


def _slice_metrics(x, y, lik, start, end,
                   threshold=None, smooth_w=None, min_bout=None):
    """Compute kinematic metrics for frames[start:end].

    Overrides for threshold / smooth_w / min_bout are used by the sensitivity
    sweep; defaults fall back to module-level constants for the main run.
    """
    if threshold is None: threshold = MOVE_THRESHOLD_PX
    if smooth_w  is None: smooth_w  = SMOOTH_WINDOW_FRAMES
    if min_bout  is None: min_bout  = MIN_BOUT_FRAMES

    x = x[start:end]; y = y[start:end]; lik = lik[start:end]
    n = len(x)
    valid = lik >= LIKELIHOOD_FLOOR
    if n < 2:
        return dict(total_distance_px=0.0, mean_velocity_px_s=0.0,
                    pct_moving=0.0, total_moving_s=0.0,
                    n_frames=n, n_valid_pairs=0,
                    frac_valid_centroid=float(valid.mean()) if n else 0.0)
    valid_pair = valid[:-1] & valid[1:]
    if valid_pair.sum() == 0:
        return dict(total_distance_px=0.0, mean_velocity_px_s=0.0,
                    pct_moving=0.0, total_moving_s=0.0,
                    n_frames=n, n_valid_pairs=0,
                    frac_valid_centroid=float(valid.mean()))
    # Raw displacement for distance metrics (no smoothing — distance is
    # additive, smoothing would underestimate it). Smoothed displacement
    # for the moving/not-moving decision.
    dx = np.diff(x); dy = np.diff(y)
    raw_dist = np.sqrt(dx * dx + dy * dy)
    smooth_dist = _smoothed_displacement(x, y, valid_pair, smooth_w)

    raw_valid = raw_dist[valid_pair]
    smooth_valid = smooth_dist[valid_pair]
    total_dist = float(raw_valid.sum())
    n_valid_pairs = int(valid_pair.sum())
    duration_s = n_valid_pairs / FPS
    mean_vel = total_dist / duration_s if duration_s > 0 else 0.0

    moving = smooth_valid > threshold
    moving = _apply_min_bout(moving, min_bout)
    pct_moving = 100.0 * moving.sum() / len(moving)
    total_moving_s = float(moving.sum()) / FPS
    return dict(
        total_distance_px=round(total_dist, 1),
        mean_velocity_px_s=round(mean_vel, 2),
        pct_moving=round(pct_moving, 2),
        total_moving_s=round(total_moving_s, 2),
        n_frames=n,
        n_valid_pairs=n_valid_pairs,
        frac_valid_centroid=round(float(valid.mean()), 3),
    )


def compute_metrics(x, y, lik):
    """Whole-session metrics over first MAX_MINUTES."""
    end = min(len(x), MAX_FRAMES)
    return _slice_metrics(x, y, lik, 0, end)


def compute_bin_metrics(x, y, lik):
    """Per-bin metrics (BIN_MINUTES each). Returns list of dicts with bin_idx."""
    out = []
    n = min(len(x), MAX_FRAMES)
    for bi, start in enumerate(range(0, n, BIN_FRAMES)):
        end = min(start + BIN_FRAMES, n)
        if end - start < FPS * 5:  # ignore final fragment <5s
            continue
        m = _slice_metrics(x, y, lik, start, end)
        m["bin_idx"] = bi
        m["bin_start_min"] = start / FPS / 60
        m["bin_end_min"] = end / FPS / 60
        out.append(m)
    return out


def _box_panel(ax, df, group_key, value_col, panel_title, ylabel):
    groups = sorted(df[group_key].unique())
    data = [df[df[group_key] == g][value_col].to_numpy().tolist() for g in groups]
    bp = ax.boxplot(data, labels=groups, patch_artist=True, widths=0.55,
                    medianprops=dict(color="black"))
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(PALETTE.get(g, "#888"))
        patch.set_alpha(0.6)
    for i, (g, ys) in enumerate(zip(groups, data), 1):
        if ys:
            xs = np.random.normal(i, 0.05, len(ys))
            ax.scatter(xs, ys, color="black", s=20, zorder=3)
            ymax = max(ys) if max(ys) > 0 else 1
            ax.text(i, ymax * 1.02, f"n={len(ys)}",
                    ha="center", fontsize=9, color="gray")
    ax.set_ylabel(ylabel)
    ax.set_title(panel_title)
    ax.grid(axis="y", alpha=0.3)


def plot_metric(df, value_col, ylabel, title, out_path):
    """4-panel: STIM genotype contrast, BL genotype contrast,
    BL->STIM paired per genotype."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    stim_sub = df[df["condition"] == "STIM"]
    _box_panel(axes[0], stim_sub, "genotype", value_col,
               "STIM: ai32 vs pdyn_ai32 (primary)", ylabel)
    bl_sub = df[df["condition"] == "BL"]
    _box_panel(axes[1], bl_sub, "genotype", value_col,
               "BL: ai32 vs pdyn_ai32", ylabel)

    ax = axes[2]
    for geno in ["ai32", "pdyn_ai32"]:
        sub = df[df["genotype"] == geno]
        for mouse, mdf in sub.groupby("mouse"):
            bl = mdf[mdf["condition"] == "BL"][value_col]
            st = mdf[mdf["condition"] == "STIM"][value_col]
            if len(bl) and len(st):
                ax.plot([0, 1], [bl.iloc[0], st.iloc[0]],
                        "o-", color=PALETTE[geno], alpha=0.7,
                        label=f"m{mouse} ({geno})")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["BL", "STIM"])
    ax.set_ylabel(ylabel)
    ax.set_title("Within-subject: BL -> STIM")
    ax.grid(axis="y", alpha=0.3)
    handles = [plt.Line2D([0], [0], color=PALETTE[g], marker="o",
                          linestyle="-", label=g) for g in ["ai32", "pdyn_ai32"]]
    ax.legend(handles=handles, loc="best")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def discord_upload(content: str, file_paths: list[Path]):
    """Upload up to 10 files (Discord's limit). Mime type guessed from
    extension — png stays inline, csv/txt show as downloadable attachment."""
    boundary = "----PixelPaws" + uuid.uuid4().hex
    body = []
    body.append(f"--{boundary}".encode())
    body.append(b'Content-Disposition: form-data; name="payload_json"')
    body.append(b'Content-Type: application/json')
    body.append(b"")
    body.append(json.dumps({"content": content}).encode())
    for i, p in enumerate(file_paths):
        if p.suffix.lower() == ".png":
            mime = "image/png"
        elif p.suffix.lower() in (".csv", ".tsv"):
            mime = "text/csv"
        else:
            mime = "application/octet-stream"
        body.append(f"--{boundary}".encode())
        body.append(
            f'Content-Disposition: form-data; name="file{i}"; filename="{p.name}"'
            .encode()
        )
        body.append(f"Content-Type: {mime}".encode())
        body.append(b"")
        body.append(p.read_bytes())
    body.append(f"--{boundary}--".encode())
    data = b"\r\n".join(body)
    req = urllib.request.Request(
        WEBHOOK, data=data, method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "PixelPaws-Chain/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            print(f"Discord HTTP {resp.status}")
    except Exception as e:
        print(f"Discord upload failed: {e}")


def plot_bin_timecourse(bin_df, value_col, ylabel, title, out_path):
    """One panel per (genotype, condition) combo with per-mouse traces +
    group mean line. Side-by-side: 4 panels (2 genotypes x 2 conditions)."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    for ri, geno in enumerate(["ai32", "pdyn_ai32"]):
        for ci, cond in enumerate(["BL", "STIM"]):
            ax = axes[ri, ci]
            sub = bin_df[(bin_df["genotype"] == geno) & (bin_df["condition"] == cond)]
            for mouse, mdf in sub.groupby("mouse"):
                mdf = mdf.sort_values("bin_idx")
                ax.plot(mdf["bin_start_min"], mdf[value_col],
                        marker="o", color=PALETTE[geno], alpha=0.4,
                        linewidth=1, label=f"m{mouse}")
            # Group mean
            if not sub.empty:
                mean_per_bin = sub.groupby("bin_start_min")[value_col].mean()
                ax.plot(mean_per_bin.index, mean_per_bin.values,
                        color=PALETTE[geno], linewidth=3,
                        linestyle=LINESTYLE[cond], label="mean")
            ax.set_xlabel("bin start (min)")
            ax.set_ylabel(ylabel if ci == 0 else "")
            n = sub["mouse"].nunique()
            ax.set_title(f"{geno} / {cond}  (n={n})")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="best")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_bin_by_condition(bin_df, value_col, ylabel, title, out_path):
    """1x2 panels: BL (left), STIM (right). Each panel overlays both
    genotypes (gray=ai32, red=pdyn_ai32) with per-mouse faint traces +
    group mean +/- SEM."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, cond in zip(axes, ["BL", "STIM"]):
        for geno in ["ai32", "pdyn_ai32"]:
            color = PALETTE[geno]
            sub = bin_df[(bin_df["genotype"] == geno) & (bin_df["condition"] == cond)]
            if sub.empty:
                continue
            for mouse, mdf in sub.groupby("mouse"):
                mdf = mdf.sort_values("bin_idx")
                ax.plot(mdf["bin_start_min"], mdf[value_col],
                        color=color, alpha=0.25, linewidth=1,
                        marker="o", markersize=3)
            agg = sub.groupby("bin_start_min")[value_col].agg(["mean", "sem"]).reset_index()
            ax.errorbar(agg["bin_start_min"], agg["mean"], yerr=agg["sem"],
                        marker="o", color=color, linewidth=2.5, capsize=3,
                        label=f"{geno} (n={sub['mouse'].nunique()})")
        ax.set_xlabel(f"Bin start (min)  -- bin width {BIN_MINUTES} min")
        if ax is axes[0]:
            ax.set_ylabel(ylabel)
        ax.set_title(f"{'PRE-stim (BL)' if cond=='BL' else 'POST-stim (STIM)'}",
                     fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=10)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_bin_split(bin_df, value_col, ylabel, title, out_path):
    """1x2 panels: ai32 (left, gray), pdyn_ai32 (right, red). Each panel
    overlays BL (dashed) vs STIM (solid) with mean +/- SEM. Per-mouse
    traces shown in faint background for context."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, geno in zip(axes, ["ai32", "pdyn_ai32"]):
        color = PALETTE[geno]
        for cond in ["BL", "STIM"]:
            sub = bin_df[(bin_df["genotype"] == geno) & (bin_df["condition"] == cond)]
            if sub.empty:
                continue
            # Faint per-mouse traces — same color as group, alpha low
            for mouse, mdf in sub.groupby("mouse"):
                mdf = mdf.sort_values("bin_idx")
                ax.plot(mdf["bin_start_min"], mdf[value_col],
                        color=color, alpha=0.25, linewidth=1,
                        linestyle=LINESTYLE[cond],
                        marker="o", markersize=3)
            # Group mean +/- SEM
            agg = sub.groupby("bin_start_min")[value_col].agg(["mean", "sem"]).reset_index()
            ax.errorbar(agg["bin_start_min"], agg["mean"], yerr=agg["sem"],
                        marker="o", color=color, linewidth=2.5,
                        capsize=3, linestyle=LINESTYLE[cond],
                        label=f"{cond} (n={sub['mouse'].nunique()})")
        ax.set_xlabel(f"Bin start (min)  -- bin width {BIN_MINUTES} min")
        if ax is axes[0]:
            ax.set_ylabel(ylabel)
        ax.set_title(f"{geno}", fontsize=12, fontweight="bold",
                     color=color)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=10)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    rows = []
    bin_rows = []
    h5_files = sorted(
        p for p in VIDEO_DIR.glob("*shuffle9*.h5")
        if not p.name.endswith("_filtered.h5")
    )
    print(f"Found {len(h5_files)} DLC .h5 files in {VIDEO_DIR}")

    for h5 in h5_files:
        base = h5.stem.split("DLC")[0]
        parsed = parse_session(base)
        if not parsed:
            print(f"  skip (cannot parse): {base}")
            continue
        mouse, cond, geno = parsed
        try:
            x, y, lik = load_centroid(h5)
        except Exception as e:
            print(f"  ! failed to read {h5.name}: {e}")
            continue

        # Whole-window
        m = compute_metrics(x, y, lik)
        m.update({"session": base, "mouse": mouse,
                  "condition": cond, "genotype": geno})
        rows.append(m)
        print(f"  {base:>42}  geno={geno:>9} m={mouse:<3} {cond:>4}  "
              f"dist={m['total_distance_px']:>9.1f}px  "
              f"vel={m['mean_velocity_px_s']:>5.2f}px/s  "
              f"%mov={m['pct_moving']:>5.2f}  "
              f"time_mov={m['total_moving_s']/60:>5.2f}min  "
              f"valid={m['frac_valid_centroid']:.2f}")

        # 5-min bins
        for bm in compute_bin_metrics(x, y, lik):
            bm.update({"session": base, "mouse": mouse,
                       "condition": cond, "genotype": geno})
            bin_rows.append(bm)

    if not rows:
        print("Nothing analyzable.")
        return 1
    df = pd.DataFrame(rows)
    bin_df = pd.DataFrame(bin_rows)
    out_csv = ANALYSIS_DIR / "locomotor.csv"
    out_bin_csv = ANALYSIS_DIR / "locomotor_5min_bins.csv"
    df.to_csv(out_csv, index=False)
    bin_df.to_csv(out_bin_csv, index=False)
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_bin_csv}")

    # Drop unmatched mice (those that don't have BOTH BL and STIM) from
    # all downstream group analysis. CSVs still contain everyone.
    if MATCHED_ONLY:
        paired = (df.groupby(["genotype", "mouse"])["condition"].nunique() == 2)
        keep = paired[paired].index  # MultiIndex of (genotype, mouse) tuples
        keep_set = set(keep.tolist())
        before = df["mouse"].nunique()
        df = df[df.apply(lambda r: (r["genotype"], r["mouse"]) in keep_set, axis=1)].copy()
        bin_df = bin_df[bin_df.apply(lambda r: (r["genotype"], r["mouse"]) in keep_set, axis=1)].copy()
        dropped = before - df["mouse"].nunique()
        if dropped > 0:
            print(f"MATCHED_ONLY: dropped {dropped} unmatched mouse/mice from group analysis.")
            for (g, m) in [(g, m) for g, m in zip(df["genotype"], df["mouse"])] + []: pass  # no-op
            # Explicit list:
            all_pairs = set((r["genotype"], r["mouse"]) for _, r in pd.read_csv(out_csv).iterrows())
            for g, m in sorted(all_pairs - keep_set):
                print(f"  dropped: {g} m{m}")

    p_dist = ANALYSIS_DIR / "locomotor_distance.png"
    p_vel = ANALYSIS_DIR / "locomotor_velocity.png"
    p_move = ANALYSIS_DIR / "locomotor_pct_moving.png"
    p_tmov = ANALYSIS_DIR / "locomotor_total_moving_min.png"
    plot_metric(df, "total_distance_px",
                f"Total distance (pixels, first {MAX_MINUTES} min)",
                f"Distance traveled - DV_DSS Pdyn-ChR2 ({MAX_MINUTES}-min window)",
                p_dist)
    plot_metric(df, "mean_velocity_px_s",
                "Mean velocity (px/s)",
                f"Mean velocity - DV_DSS Pdyn-ChR2 ({MAX_MINUTES}-min window)",
                p_vel)
    plot_metric(df, "pct_moving",
                f"% time moving (>{MOVE_THRESHOLD_PX} px/frame)",
                f"% time moving - DV_DSS Pdyn-ChR2 ({MAX_MINUTES}-min window)",
                p_move)
    # New: total moving in minutes
    df_min = df.copy()
    df_min["total_moving_min"] = df_min["total_moving_s"] / 60.0
    plot_metric(df_min, "total_moving_min",
                "Total time moving (minutes, of 25-min window)",
                "Total time moving - DV_DSS Pdyn-ChR2",
                p_tmov)

    # 5-min bin time-course plots -- view A: split by genotype, BL vs STIM in each panel
    p_bin_dist = ANALYSIS_DIR / "locomotor_5min_distance.png"
    p_bin_vel = ANALYSIS_DIR / "locomotor_5min_velocity.png"
    p_bin_pct = ANALYSIS_DIR / "locomotor_5min_pct_moving.png"
    p_bin_tmov = ANALYSIS_DIR / "locomotor_5min_total_moving.png"

    bin_df["total_moving_min"] = bin_df["total_moving_s"] / 60.0
    plot_bin_split(bin_df, "total_distance_px",
                     "Distance per 5-min bin (px)",
                     f"Per-bin distance (by genotype, {BIN_MINUTES}-min bins)",
                     p_bin_dist)
    plot_bin_split(bin_df, "mean_velocity_px_s",
                     "Mean velocity per bin (px/s)",
                     f"Per-bin velocity (by genotype, {BIN_MINUTES}-min bins)",
                     p_bin_vel)
    plot_bin_split(bin_df, "pct_moving",
                     "% time moving per bin",
                     f"Per-bin % moving (by genotype, {BIN_MINUTES}-min bins)",
                     p_bin_pct)
    plot_bin_split(bin_df, "total_moving_min",
                     "Time moving per bin (min, max 5)",
                     f"Per-bin total moving time (by genotype)",
                     p_bin_tmov)

    # View B: split by condition (BL/STIM), genotypes overlaid in each panel
    p_bin_dist_c = ANALYSIS_DIR / "locomotor_5min_distance_by_cond.png"
    p_bin_vel_c = ANALYSIS_DIR / "locomotor_5min_velocity_by_cond.png"
    p_bin_pct_c = ANALYSIS_DIR / "locomotor_5min_pct_moving_by_cond.png"
    p_bin_tmov_c = ANALYSIS_DIR / "locomotor_5min_total_moving_by_cond.png"
    plot_bin_by_condition(bin_df, "total_distance_px",
                          "Distance per 5-min bin (px)",
                          f"Per-bin distance (PRE vs POST, {BIN_MINUTES}-min bins)",
                          p_bin_dist_c)
    plot_bin_by_condition(bin_df, "mean_velocity_px_s",
                          "Mean velocity per bin (px/s)",
                          f"Per-bin velocity (PRE vs POST, {BIN_MINUTES}-min bins)",
                          p_bin_vel_c)
    plot_bin_by_condition(bin_df, "pct_moving",
                          "% time moving per bin",
                          f"Per-bin % moving (PRE vs POST, {BIN_MINUTES}-min bins)",
                          p_bin_pct_c)
    plot_bin_by_condition(bin_df, "total_moving_min",
                          "Time moving per bin (min, max 5)",
                          f"Per-bin total moving time (PRE vs POST)",
                          p_bin_tmov_c)

    # Per-session text summary for Discord
    def _mean_sd(arr):
        a = np.asarray(arr, dtype=float)
        if len(a) == 0:
            return "n/a"
        sd = a.std(ddof=1) if len(a) > 1 else 0.0
        return f"{a.mean():.1f}+/-{sd:.1f}"

    kp_label = "+".join(BODY_KEYPOINTS) if len(BODY_KEYPOINTS) > 1 else BODY_KEYPOINTS[0]
    lines = [
        f"**DV_DSS Pdyn-ChR2 -- locomotor ({MAX_MINUTES}-min window)**",
        f"Sessions: {len(df)} ({df['mouse'].nunique()} mice, "
        f"{df['genotype'].nunique()} genotypes).",
        f"Body keypoint: `{kp_label}`  |  likelihood floor {LIKELIHOOD_FLOOR}  |  "
        f"move threshold {MOVE_THRESHOLD_PX} px/frame at {FPS} fps.\n",
        "Total time spent moving (min, mean+/-SD):",
    ]
    df["total_moving_min"] = df["total_moving_s"] / 60.0
    for geno in ["ai32", "pdyn_ai32"]:
        for cond in ["BL", "STIM"]:
            sub = df[(df["genotype"] == geno) & (df["condition"] == cond)]["total_moving_min"]
            lines.append(
                f"  {geno:>9} / {cond:<4}: {_mean_sd(sub.to_numpy()):>13}  (n={len(sub)})"
            )

    lines.append("\nTotal distance (px, mean+/-SD):")
    for geno in ["ai32", "pdyn_ai32"]:
        for cond in ["BL", "STIM"]:
            sub = df[(df["genotype"] == geno) & (df["condition"] == cond)]["total_distance_px"]
            lines.append(
                f"  {geno:>9} / {cond:<4}: {_mean_sd(sub.to_numpy()):>15}  (n={len(sub)})"
            )

    # Within-subject delta (STIM - BL) in TOTAL moving time
    deltas = {"ai32": [], "pdyn_ai32": []}
    for geno in ["ai32", "pdyn_ai32"]:
        sub = df[df["genotype"] == geno]
        for mouse, mdf in sub.groupby("mouse"):
            bl = mdf[mdf["condition"] == "BL"]["total_moving_min"]
            st = mdf[mdf["condition"] == "STIM"]["total_moving_min"]
            if len(bl) and len(st):
                deltas[geno].append(st.iloc[0] - bl.iloc[0])
    lines.append("\nWithin-subject delta in total moving time (STIM - BL, min):")
    for geno in ["ai32", "pdyn_ai32"]:
        lines.append(f"  {geno:>9}: {_mean_sd(deltas[geno])}  (n={len(deltas[geno])})")

    content = "\n".join(lines)
    # Whole-window summary + the two split-by-genotype timecourse plots
    discord_upload(content, [p_dist, p_tmov, p_bin_dist, p_bin_tmov])
    discord_upload(
        "Per-bin time-course (by GENOTYPE panel) -- velocity + %moving:",
        [p_bin_vel, p_bin_pct],
    )
    # New view: by CONDITION (BL left, STIM right; genotypes overlaid)
    discord_upload(
        "Per-bin time-course (by CONDITION panel, genotypes overlaid):\n"
        "Distance + total moving time",
        [p_bin_dist_c, p_bin_tmov_c],
    )
    discord_upload(
        "Per-bin time-course (by CONDITION panel): velocity + %moving",
        [p_bin_vel_c, p_bin_pct_c],
    )
    # Send the raw data CSVs as a separate post
    discord_upload(
        "Raw locomotor data (CSV downloads):",
        [out_csv, out_bin_csv],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
