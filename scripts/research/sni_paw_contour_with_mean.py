"""
SHAM vs SNI paw-print figure WITH averaged-contour overlay.

For each role (HL, HR) and each group (SHAM, SNI), render:
  * the representative paw-print outline (filled, group colour) used
    for the existing desktop figures (chosen by closest-to-template
    after canonical PCA alignment + cyclic shift), AND
  * the group-mean contour (averaged across every paw-like contour in
    the group's pool) overlaid on top as a thick dashed outline so the
    averaged shape is visible without obscuring the representative.

No axes / no gridlines / no title -- just the two shapes on a white
background.  Saves a 2x2 grid (PNG + SVG + transparent PNG) and four
individual single-panel PNGs.

Reads `_shapes.npz` from:
  E:\\RSVIDS\\Blackbox\\2603_SNI_PG\\weight_bearing_analysis
"""
from __future__ import annotations
import json, sys, time, urllib.request, uuid
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

# Reuse the shape-metrics + alignment + closest-to-template helpers
from sntx_paw_contour_publication_figure import (
    shape_metrics, pca_canonical_align,
    SOLIDITY_MAX, ASPECT_MAX, CIRCULARITY_MIN,
    STRICT_SOLID_MAX, STRICT_ASPECT_MAX, STRICT_CIRC_MAX,
)

COHORT = Path(r"E:\RSVIDS\Blackbox\2603_SNI_PG")
GLAB   = COHORT / "weight_bearing_analysis"
KEY    = COHORT / "key_file.csv"
OUT    = COHORT / "analysis"

GROUP_ORDER = ["SHAM", "SNI"]
GROUP_COLORS = {
    "SHAM": "#2ca02c",   # green (matches the desktop SHAM panel)
    "SNI":  "#ff7f0e",   # orange (matches the desktop SNI panel)
}
MEAN_COLOR = "#222222"   # dark grey dashed mean outline

WEBHOOK = ("")


def step(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def discord_upload(content, files):
    boundary = "----PP" + uuid.uuid4().hex
    body = [f"--{boundary}".encode(),
            b'Content-Disposition: form-data; name="payload_json"',
            b'Content-Type: application/json', b"",
            json.dumps({"content": content}).encode()]
    for i, p in enumerate(files):
        mime = {".png": "image/png", ".svg": "image/svg+xml",
                ".pdf": "application/pdf", ".csv": "text/csv"
                }.get(p.suffix.lower(), "application/octet-stream")
        body += [f"--{boundary}".encode(),
                 f'Content-Disposition: form-data; name="file{i}"; filename="{p.name}"'.encode(),
                 f"Content-Type: {mime}".encode(), b"",
                 p.read_bytes()]
    body.append(f"--{boundary}--".encode())
    req = urllib.request.Request(
        WEBHOOK, data=b"\r\n".join(body), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "User-Agent": "PixelPaws-SNI-PawPrintMean/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            step(f"  HTTP {r.status}")
    except Exception as e:
        step(f"  upload failed: {e}")


def load_groups():
    df = pd.read_csv(KEY).dropna()
    mouse_to_group = {row["Subject"]: row["Treatment"]
                      for _, row in df.iterrows()
                      if row["Treatment"] in GROUP_ORDER}
    step(f"key file: SHAM={sum(v=='SHAM' for v in mouse_to_group.values())}, "
         f"SNI={sum(v=='SNI' for v in mouse_to_group.values())}")

    pooled = {r: {g: [] for g in GROUP_ORDER} for r in ("HL", "HR")}
    for p in sorted(GLAB.glob("*_contour_*_shapes.npz")):
        match = next((mid for mid in mouse_to_group if mid in p.name), None)
        if not match:
            step(f"  ! no key match for {p.name}"); continue
        g = mouse_to_group[match]
        with np.load(p) as data:
            for role in ("HL", "HR"):
                if role in data:
                    pooled[role][g].append(data[role])
        step(f"  {match} -> {g}")

    out = {"HL": {}, "HR": {}}
    for role in ("HL", "HR"):
        for g in GROUP_ORDER:
            chunks = pooled[role][g]
            if not chunks: continue
            all_shapes = np.concatenate(chunks, axis=0)
            sol, ar, circ = shape_metrics(all_shapes)
            mask = (sol <= SOLIDITY_MAX) & (ar <= ASPECT_MAX) & (circ >= CIRCULARITY_MIN)
            paw_like = all_shapes[mask]
            out[role][g] = (paw_like, int(mask.sum()), int(len(all_shapes)))
            step(f"  pooled {role} {g}: paw-like {int(mask.sum())} / {len(all_shapes)}")
    return out


def cyclic_align_to(s, template):
    n = s.shape[0]
    best_sse = np.inf; best_shift = 0
    for k in range(n):
        sse = float(np.sum((np.roll(s, k, axis=0) - template) ** 2))
        if sse < best_sse:
            best_sse = sse; best_shift = k
    return np.roll(s, best_shift, axis=0), best_sse


def aligned_pool(stacked):
    """Apply PCA canonical alignment to every contour in a pool."""
    out = np.empty_like(stacked)
    for i in range(len(stacked)):
        out[i] = pca_canonical_align(stacked[i])
    return out


def strict_subset(stacked):
    sol, ar, circ = shape_metrics(stacked)
    mask = ((sol <= STRICT_SOLID_MAX)
            & (ar  <= STRICT_ASPECT_MAX)
            & (circ <= STRICT_CIRC_MAX)
            & (circ >= CIRCULARITY_MIN))
    return stacked[mask]


def pick_representative_and_mean(paw_like, partner_paw_like):
    """Return (rep_shape, mean_shape).

    * Align both groups' paw-like pools, build a JOINT strict-filtered
      template (mean across both groups' clean-toed contours), and
      return the strict-subset shape closest to that template as the
      representative.  Using the joint template here mirrors the
      existing publication-figure logic so the rep matches what's
      already on the desktop.
    * The mean shape is the centroid of the FULL paw-like pool after
      canonical alignment + cyclic-shift to a shared starting point.
    """
    aligned = aligned_pool(paw_like)
    partner_aligned = aligned_pool(partner_paw_like)

    # Joint strict template = mean of both groups' strict-filtered pools
    a_strict = strict_subset(aligned)
    b_strict = strict_subset(partner_aligned)
    if len(a_strict) == 0 or len(b_strict) == 0:
        template = aligned.mean(axis=0)
        rep_pool = aligned
    else:
        joint = np.concatenate([a_strict, b_strict], axis=0)
        # cyclic-align each strict shape to the joint mean, then take mean
        jmean = joint.mean(axis=0)
        aligned_joint = np.empty_like(joint)
        for i in range(len(joint)):
            aligned_joint[i], _ = cyclic_align_to(joint[i], jmean)
        template = aligned_joint.mean(axis=0)
        rep_pool = a_strict

    # Representative = strict-pool shape closest (cyclic-aligned) to template
    best_sse = np.inf; rep = rep_pool[0]
    for s in rep_pool:
        shifted, sse = cyclic_align_to(s, template)
        if sse < best_sse:
            best_sse = sse; rep = shifted

    # Mean = centroid of the FULL paw-like pool (after cyclic alignment to
    # the joint template) -- this is the "averaged version" of the contour
    # across the entire group.
    aligned_full = np.empty_like(aligned)
    for i in range(len(aligned)):
        aligned_full[i], _ = cyclic_align_to(aligned[i], template)
    mean_shape = aligned_full.mean(axis=0)
    return rep, mean_shape


def plot_panel(ax, rep, mean, color):
    rep_closed  = np.vstack([rep,  rep[0:1]])
    mean_closed = np.vstack([mean, mean[0:1]])
    # filled representative
    ax.fill(rep_closed[:, 0], rep_closed[:, 1],
            color=color, alpha=0.85, edgecolor=color, linewidth=1.4,
            zorder=1)
    # averaged outline overlay (dashed dark grey, thick)
    ax.plot(mean_closed[:, 0], mean_closed[:, 1],
            color=MEAN_COLOR, linewidth=2.2, linestyle="--", zorder=2,
            dash_capstyle="round", solid_capstyle="round")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    # Strip every chart artifact
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pooled = load_groups()

    # Pick representative + mean per role-group
    reps = {"HL": {}, "HR": {}}
    means = {"HL": {}, "HR": {}}
    for role in ("HL", "HR"):
        for g in GROUP_ORDER:
            paw_like = pooled[role][g][0]
            partner = next(gg for gg in GROUP_ORDER if gg != g)
            partner_paw_like = pooled[role][partner][0]
            step(f"computing rep + mean for {role} {g} (n={len(paw_like)})")
            rep, mean_shape = pick_representative_and_mean(paw_like, partner_paw_like)
            reps[role][g] = rep
            means[role][g] = mean_shape

    # 2x2 combined figure
    fig, axes = plt.subplots(2, 2, figsize=(7, 9.5),
                             gridspec_kw=dict(wspace=0.05, hspace=0.05))
    for row, role in enumerate(("HL", "HR")):
        for col, group in enumerate(GROUP_ORDER):
            plot_panel(axes[row, col], reps[role][group], means[role][group],
                       GROUP_COLORS[group])

    # Save -- transparent so the background is truly removed
    out_png = OUT / "sni_paw_print_with_mean.png"
    out_svg = OUT / "sni_paw_print_with_mean.svg"
    out_pdf = OUT / "sni_paw_print_with_mean.pdf"
    fig.savefig(out_png, format="png", dpi=240, bbox_inches="tight",
                facecolor="white")
    fig.savefig(out_svg, format="svg", bbox_inches="tight",
                facecolor="white")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight",
                facecolor="white")
    out_png_t = OUT / "sni_paw_print_with_mean_transparent.png"
    fig.savefig(out_png_t, format="png", dpi=240, bbox_inches="tight",
                transparent=True)
    plt.close(fig)

    # Individual panels (PNG only)
    individual_files = []
    for role in ("HL", "HR"):
        for g in GROUP_ORDER:
            fig2, ax2 = plt.subplots(figsize=(3.2, 4.3))
            plot_panel(ax2, reps[role][g], means[role][g], GROUP_COLORS[g])
            ip = OUT / f"sni_paw_print_with_mean_{role}_{g}.png"
            fig2.savefig(ip, format="png", dpi=240, bbox_inches="tight",
                         transparent=True)
            plt.close(fig2)
            individual_files.append(ip)

    head = ("**SNI vs SHAM paw contours -- representative + group mean overlay**\n"
            "Filled paw print = the strict-toed representative used in the prior "
            "desktop figure. Dashed dark line = the group's averaged contour "
            "(centroid of every paw-like contour after canonical alignment + "
            "cyclic-shift to a shared starting point). No background -- white "
            "fill in PNG/SVG/PDF, transparent in the `_transparent.png`. "
            "Layout: rows = HL/HR, cols = SHAM/SNI.")
    discord_upload(head, [out_png, out_png_t, out_svg, out_pdf, *individual_files])


if __name__ == "__main__":
    main()
