"""
Clean, aligned SNTX vs Naive paw-print panels -- no axes, no gridlines,
no labels, no titles. Same representative-shape selection as
`sntx_paw_contour_publication_figure.py` (closest to the joint
strict-template after PCA canonical alignment + cyclic shift), but
stripped of all chart chrome and rendered with shared x/y data limits
across every panel so the four paws are visually aligned at consistent
scale (panel-to-panel comparable).
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

from sntx_paw_contour_publication_figure import (
    shape_metrics, pca_canonical_align,
    SOLIDITY_MAX, ASPECT_MAX, CIRCULARITY_MIN,
    STRICT_SOLID_MAX, STRICT_ASPECT_MAX, STRICT_CIRC_MAX,
)
from sntx_paw_contour_with_mean import (
    cyclic_align_to, aligned_pool, strict_subset,
    pick_representative_and_mean,
)

COHORT = Path(r"E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline")
GLAB   = COHORT / "gait_limb_analysis"
KEY    = COHORT / "key_file.csv"
OUT    = COHORT / "analysis"

GROUP_ORDER = ["SNTX", "Naive"]
GROUP_COLORS = {"SNTX": "#d62728", "Naive": "#000000"}

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
                ".pdf": "application/pdf"}.get(p.suffix.lower(),
                                                "application/octet-stream")
        body += [f"--{boundary}".encode(),
                 f'Content-Disposition: form-data; name="file{i}"; filename="{p.name}"'.encode(),
                 f"Content-Type: {mime}".encode(), b"",
                 p.read_bytes()]
    body.append(f"--{boundary}--".encode())
    req = urllib.request.Request(
        WEBHOOK, data=b"\r\n".join(body), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "User-Agent": "PixelPaws-SNTX-CleanAligned/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        step(f"  HTTP {r.status}")


def load_groups():
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
        step(f"  {match} -> {g}")
    out = {"HL": {}, "HR": {}}
    for role in ("HL", "HR"):
        for g in GROUP_ORDER:
            chunks = pooled[role][g]
            if not chunks: continue
            all_shapes = np.concatenate(chunks, axis=0)
            sol, ar, circ = shape_metrics(all_shapes)
            mask = (sol <= SOLIDITY_MAX) & (ar <= ASPECT_MAX) & (circ >= CIRCULARITY_MIN)
            out[role][g] = all_shapes[mask]
            step(f"  pooled {role} {g}: paw-like {int(mask.sum())} / {len(all_shapes)}")
    return out


def plot_shape(ax, pts, color):
    closed = np.vstack([pts, pts[0:1]])
    ax.fill(closed[:, 0], closed[:, 1],
            color=color, alpha=0.92, edgecolor=color, linewidth=1.4)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pooled = load_groups()

    reps = {"HL": {}, "HR": {}}
    for role in ("HL", "HR"):
        for g in GROUP_ORDER:
            paw_like = pooled[role][g]
            partner = next(gg for gg in GROUP_ORDER if gg != g)
            partner_paw_like = pooled[role][partner]
            step(f"selecting rep: {role} {g} (n={len(paw_like)})")
            rep, _ = pick_representative_and_mean(paw_like, partner_paw_like)
            reps[role][g] = rep

    # Compute shared axis limits across every panel so the four paws
    # render at the same scale.
    all_pts = np.concatenate([reps[r][g] for r in ("HL", "HR")
                              for g in GROUP_ORDER], axis=0)
    pad = 0.05
    x_lo, x_hi = all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad
    y_lo, y_hi = all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad
    step(f"shared limits: x [{x_lo:.2f}, {x_hi:.2f}]  y [{y_lo:.2f}, {y_hi:.2f}]")

    # 2x2 combined: rows = HL/HR, cols = SNTX/Naive
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 9),
                             gridspec_kw=dict(wspace=0.04, hspace=0.04))
    for row, role in enumerate(("HL", "HR")):
        for col, group in enumerate(GROUP_ORDER):
            ax = axes[row, col]
            plot_shape(ax, reps[role][group], GROUP_COLORS[group])
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_hi, y_lo)  # already inverted; keep range consistent

    out_png  = OUT / "sntx_paw_print_clean_aligned.png"
    out_pngT = OUT / "sntx_paw_print_clean_aligned_transparent.png"
    out_svg  = OUT / "sntx_paw_print_clean_aligned.svg"
    out_pdf  = OUT / "sntx_paw_print_clean_aligned.pdf"
    fig.savefig(out_png, format="png", dpi=240, bbox_inches="tight",
                facecolor="white", pad_inches=0.04)
    fig.savefig(out_pngT, format="png", dpi=240, bbox_inches="tight",
                transparent=True, pad_inches=0.04)
    fig.savefig(out_svg, format="svg", bbox_inches="tight",
                facecolor="white", pad_inches=0.04)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight",
                facecolor="white", pad_inches=0.04)
    plt.close(fig)

    # Individual single-panel PNGs (also aligned to shared limits)
    individual = []
    for role in ("HL", "HR"):
        for g in GROUP_ORDER:
            fig2, ax2 = plt.subplots(figsize=(3.0, 4.2))
            plot_shape(ax2, reps[role][g], GROUP_COLORS[g])
            ax2.set_xlim(x_lo, x_hi); ax2.set_ylim(y_hi, y_lo)
            ip = OUT / f"sntx_paw_print_clean_aligned_{role}_{g}.png"
            fig2.savefig(ip, format="png", dpi=240, bbox_inches="tight",
                         transparent=True, pad_inches=0.04)
            plt.close(fig2)
            individual.append(ip)

    head = ("**SNTX vs Naive paw prints -- background lines removed, "
            "panels aligned to a shared coordinate range**\n"
            "Same representative selection as the publication-figure script "
            "(closest to joint strict-template after PCA canonical alignment "
            "+ cyclic shift). All four panels share x/y data limits so the "
            "paws render at the same visual scale. 2x2 layout: rows = HL/HR, "
            "cols = SNTX (red) / Naive (black). PNG + transparent PNG + SVG "
            "+ PDF + four single-panel transparent PNGs.")
    discord_upload(head, [out_png, out_pngT, out_svg, out_pdf, *individual])


if __name__ == "__main__":
    main()
