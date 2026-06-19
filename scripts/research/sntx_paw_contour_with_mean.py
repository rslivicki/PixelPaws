"""
SNTX vs Naive paw-print figure WITH averaged-contour overlay -- no
background, no axes, no labels.

For each role (HL, HR) and each group (SNTX, Naive):
  * filled representative paw-print outline (group colour), chosen by
    closest-to-joint-template after PCA canonical alignment + cyclic
    shift -- same selection logic as `sntx_paw_contour_publication_figure.py`
  * group-mean contour overlaid on top as a thick dashed dark-grey
    line (centroid of every paw-like contour after canonical
    alignment + cyclic shift to a shared starting point).

Reads 7 `_shapes.npz` files from
  E:\\RSVIDS\\Blackbox\\2603_SNLT_JG\\Baseline\\gait_limb_analysis

SNTX rendered red (`#d62728`) to match the user's PDFs;
Naive rendered black (`#000000`).
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

COHORT = Path(r"E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline")
GLAB   = COHORT / "gait_limb_analysis"
KEY    = COHORT / "key_file.csv"
OUT    = COHORT / "analysis"

GROUP_ORDER = ["SNTX", "Naive"]
GROUP_COLORS = {
    "SNTX":  "#d62728",
    "Naive": "#000000",
}
MEAN_COLOR = "#222222"

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
                 "User-Agent": "PixelPaws-SNTX-PawPrintMean/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            step(f"  HTTP {r.status}")
    except Exception as e:
        step(f"  upload failed: {e}")


def load_groups():
    df = pd.read_csv(KEY).dropna()
    mt = {r["Subject"]: r["Treatment"] for _, r in df.iterrows()
          if r["Treatment"] in GROUP_ORDER}
    step(f"key file: SNTX={sum(v=='SNTX' for v in mt.values())}, "
         f"Naive={sum(v=='Naive' for v in mt.values())}")

    pooled = {r: {g: [] for g in GROUP_ORDER} for r in ("HL", "HR")}
    for p in sorted(GLAB.glob("*_contour_*_shapes.npz")):
        match = next((mid for mid in mt if mid in p.name), None)
        if not match:
            continue
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
    aligned = aligned_pool(paw_like)
    partner_aligned = aligned_pool(partner_paw_like)
    a_strict = strict_subset(aligned)
    b_strict = strict_subset(partner_aligned)
    if len(a_strict) == 0 or len(b_strict) == 0:
        template = aligned.mean(axis=0)
        rep_pool = aligned
    else:
        joint = np.concatenate([a_strict, b_strict], axis=0)
        jmean = joint.mean(axis=0)
        aligned_joint = np.empty_like(joint)
        for i in range(len(joint)):
            aligned_joint[i], _ = cyclic_align_to(joint[i], jmean)
        template = aligned_joint.mean(axis=0)
        rep_pool = a_strict

    best_sse = np.inf; rep = rep_pool[0]
    for s in rep_pool:
        shifted, sse = cyclic_align_to(s, template)
        if sse < best_sse:
            best_sse = sse; rep = shifted

    aligned_full = np.empty_like(aligned)
    for i in range(len(aligned)):
        aligned_full[i], _ = cyclic_align_to(aligned[i], template)
    mean_shape = aligned_full.mean(axis=0)
    return rep, mean_shape


def plot_panel(ax, rep, mean, color):
    rep_closed  = np.vstack([rep,  rep[0:1]])
    mean_closed = np.vstack([mean, mean[0:1]])
    ax.fill(rep_closed[:, 0], rep_closed[:, 1],
            color=color, alpha=0.85, edgecolor=color, linewidth=1.4,
            zorder=1)
    ax.plot(mean_closed[:, 0], mean_closed[:, 1],
            color=MEAN_COLOR, linewidth=2.4, linestyle="--", zorder=2,
            dash_capstyle="round", solid_capstyle="round")
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
    means = {"HL": {}, "HR": {}}
    for role in ("HL", "HR"):
        for g in GROUP_ORDER:
            paw_like = pooled[role][g][0]
            partner = next(gg for gg in GROUP_ORDER if gg != g)
            partner_paw_like = pooled[role][partner][0]
            step(f"rep + mean: {role} {g} (n={len(paw_like)})")
            rep, mean_shape = pick_representative_and_mean(paw_like, partner_paw_like)
            reps[role][g] = rep
            means[role][g] = mean_shape

    fig, axes = plt.subplots(2, 2, figsize=(7, 9.5),
                             gridspec_kw=dict(wspace=0.05, hspace=0.05))
    for row, role in enumerate(("HL", "HR")):
        for col, group in enumerate(GROUP_ORDER):
            plot_panel(axes[row, col], reps[role][group], means[role][group],
                       GROUP_COLORS[group])

    out_png  = OUT / "sntx_paw_print_with_mean.png"
    out_svg  = OUT / "sntx_paw_print_with_mean.svg"
    out_pdf  = OUT / "sntx_paw_print_with_mean.pdf"
    out_pngT = OUT / "sntx_paw_print_with_mean_transparent.png"
    fig.savefig(out_png, format="png", dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, format="svg", bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out_pngT, format="png", dpi=240, bbox_inches="tight", transparent=True)
    plt.close(fig)

    individual_files = []
    for role in ("HL", "HR"):
        for g in GROUP_ORDER:
            fig2, ax2 = plt.subplots(figsize=(3.2, 4.3))
            plot_panel(ax2, reps[role][g], means[role][g], GROUP_COLORS[g])
            ip = OUT / f"sntx_paw_print_with_mean_{role}_{g}.png"
            fig2.savefig(ip, format="png", dpi=240, bbox_inches="tight", transparent=True)
            plt.close(fig2)
            individual_files.append(ip)

    head = ("**SNTX vs Naive paw contours -- representative + group-mean overlay**\n"
            "Filled paw print = strict-toed representative shape (closest to joint "
            "template after PCA canonical alignment + cyclic shift). Dashed dark "
            "line = group-mean contour (centroid of every paw-like contour after "
            "canonical alignment + cyclic shift to a shared starting point). "
            "No background -- 2x2 grid (rows = HL/HR, cols = SNTX/Naive), with "
            "individual single-panel PNGs (transparent) attached. SNTX red, Naive black.")
    discord_upload(head, [out_png, out_pngT, out_svg, out_pdf, *individual_files])


if __name__ == "__main__":
    main()
