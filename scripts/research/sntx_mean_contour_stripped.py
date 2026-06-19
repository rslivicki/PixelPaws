"""
Render the SNTX vs Naive mean-contour overlay panels (HL, HR) with NO
chart chrome: no title, no axes, no axis labels, no gridlines. Keeps
the SD envelope + mean outline per group and a small in-panel legend.

Coordinate frame is shared across both panels and matched to the
normalized [-1.1, 1.1] envelope that the paw-print PDFs render in, so
the visual size of a paw is comparable to the paw-print rows above
when composited.

Reads `_shapes.npz` from
  E:\\RSVIDS\\Blackbox\\2603_SNLT_JG\\Baseline\\gait_limb_analysis
Writes per-role transparent PNGs into the same `analysis` folder used
by the publication-figure script.
"""
from __future__ import annotations
import sys, time
from pathlib import Path

REPO = Path(r"E:\PixelPaws"); sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sntx_paw_contour_publication_figure import (
    shape_metrics, SOLIDITY_MAX, ASPECT_MAX, CIRCULARITY_MIN,
)

COHORT = Path(r"E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline")
GLAB   = COHORT / "gait_limb_analysis"
KEY    = COHORT / "key_file.csv"
OUT    = COHORT / "analysis"

GROUP_ORDER  = ["SNTX", "Naive"]
GROUP_COLORS = {"SNTX": "#d62728", "Naive": "#000000"}


def step(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def load_paw_like():
    df = pd.read_csv(KEY).dropna()
    mt = {r["Subject"]: r["Treatment"] for _, r in df.iterrows()
          if r["Treatment"] in GROUP_ORDER}
    pooled = {r: {g: [] for g in GROUP_ORDER} for r in ("HL", "HR")}
    for p in sorted(GLAB.glob("*_contour_*_shapes.npz")):
        match = next((mid for mid in mt if mid in p.name), None)
        if not match: continue
        g = mt[match]
        with np.load(p) as data:
            for role in ("HL", "HR"):
                if role in data:
                    pooled[role][g].append(data[role])
    out = {"HL": {}, "HR": {}}
    for role in ("HL", "HR"):
        for g in GROUP_ORDER:
            chunks = pooled[role][g]
            if not chunks: continue
            all_shapes = np.concatenate(chunks, axis=0)
            sol, ar, circ = shape_metrics(all_shapes)
            mask = ((sol <= SOLIDITY_MAX) & (ar <= ASPECT_MAX)
                    & (circ >= CIRCULARITY_MIN))
            out[role][g] = all_shapes[mask]
            step(f"  pooled {role} {g}: paw-like {int(mask.sum())} / "
                 f"{len(all_shapes)}")
    return out


def plot_mean_panel(ax, shapes_by_group, x_lim, y_lim, show_legend=True):
    for group in GROUP_ORDER:
        stacked = shapes_by_group.get(group)
        if stacked is None or len(stacked) == 0:
            continue
        mean_shape = stacked.mean(axis=0)
        sd_shape = (stacked.std(axis=0, ddof=1) if len(stacked) > 1
                    else np.zeros_like(mean_shape))
        mean_closed = np.vstack([mean_shape, mean_shape[0:1]])
        sd_closed = np.vstack([sd_shape, sd_shape[0:1]])
        radial_sd = np.sqrt(sd_closed[:, 0] ** 2 + sd_closed[:, 1] ** 2)
        centroid = mean_closed[:-1].mean(axis=0)
        directions = mean_closed - centroid
        norms = np.sqrt((directions ** 2).sum(axis=1, keepdims=True))
        norms[norms == 0] = 1
        unit_dirs = directions / norms
        outer = mean_closed + unit_dirs * radial_sd[:, np.newaxis]
        inner = mean_closed - unit_dirs * radial_sd[:, np.newaxis]
        ring_x = np.concatenate([outer[:, 0], inner[::-1, 0]])
        ring_y = np.concatenate([outer[:, 1], inner[::-1, 1]])

        color = GROUP_COLORS[group]
        ax.fill(ring_x, ring_y, alpha=0.20, color=color, linewidth=0)
        ax.plot(mean_closed[:, 0], mean_closed[:, 1],
                color=color, linewidth=2.4,
                label=f"{group} (n={len(stacked)})")

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlim(*x_lim); ax.set_ylim(y_lim[1], y_lim[0])
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    if show_legend:
        ax.legend(loc="upper left", frameon=False, fontsize=7,
                  handlelength=0.9, borderaxespad=0.1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pooled = load_paw_like()

    # Compute shared coordinate range across both panels + both groups,
    # tightened to the mean+SD envelope (not the outlier extremes) so the
    # silhouette fills the cell when composited at paw-print scale.
    mean_pts = []
    for role in ("HL", "HR"):
        for g in GROUP_ORDER:
            s = pooled[role].get(g)
            if s is None or len(s) == 0: continue
            mu = s.mean(axis=0)
            sd = s.std(axis=0, ddof=1) if len(s) > 1 else np.zeros_like(mu)
            envelope_outer = mu + sd * 1.3
            envelope_inner = mu - sd * 1.3
            mean_pts.append(envelope_outer)
            mean_pts.append(envelope_inner)
    mean_pts = np.concatenate(mean_pts, axis=0)
    pad = 0.05
    x_lim = (mean_pts[:, 0].min() - pad, mean_pts[:, 0].max() + pad)
    y_lim = (mean_pts[:, 1].min() - pad, mean_pts[:, 1].max() + pad)
    step(f"shared limits: x [{x_lim[0]:.2f}, {x_lim[1]:.2f}]  "
         f"y [{y_lim[0]:.2f}, {y_lim[1]:.2f}]")

    paths = []
    for role in ("HL", "HR"):
        fig, ax = plt.subplots(figsize=(3.2, 4.3))
        plot_mean_panel(ax, pooled[role], x_lim, y_lim, show_legend=True)
        out_path = OUT / f"sntx_mean_contour_stripped_{role}.png"
        fig.savefig(out_path, dpi=240, bbox_inches="tight",
                    transparent=True, pad_inches=0.05)
        plt.close(fig)
        step(f"  -> {out_path.name}")
        paths.append(out_path)

    # Also drop a local copy next to the export folder so the
    # combine script can find them without crossing drives.
    EXPORT = Path(r"C:\Users\Gereau\Desktop\Toexport")
    for p in paths:
        dst = EXPORT / p.name
        dst.write_bytes(p.read_bytes())
        step(f"  copied -> {dst}")


if __name__ == "__main__":
    main()
