"""
Batch-export everything we've posted to #results as Illustrator-friendly
PDF (editable text, vector strokes / fills).

Re-uses the existing generator scripts and just adds a PDF save with
`matplotlib.rcParams['pdf.fonttype'] = 42` (TrueType embedded as fonts,
not paths) so axis labels / legends / titles all stay editable in
Illustrator.

Coverage:
  1. SNLT CV-F1 summary (`_classifier_f1.pdf`)
  2. SNTX vs Naïve paw contour figure (`sntx_paw_contour_figure.pdf`)
  3. Scratching panels B / C / D (learning curve, threshold, SHAP)
  4. FormOxy contour intensity ratio TC (the cleaner one)
  5. FormOxy mean paw contour (HL + HR)
  6. FormOxy Left_licking dose-response + 5-min timecourse

(The temporal-probability PDF is handled directly inside
`snlt_temporal_probability_corrected.py`.)
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
import urllib.request
import uuid
from pathlib import Path

REPO = Path(r"E:\PixelPaws")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "research"))

import matplotlib
matplotlib.use("Agg")
# Editable text in Illustrator from PDF / EPS / SVG
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np

WEBHOOK = (
    ""
)


def step(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def discord_upload(content: str, files: list[Path]) -> None:
    boundary = "----PP" + uuid.uuid4().hex
    body = [
        f"--{boundary}".encode(),
        b'Content-Disposition: form-data; name="payload_json"',
        b'Content-Type: application/json', b"",
        json.dumps({"content": content}).encode(),
    ]
    for i, p in enumerate(files):
        ext = p.suffix.lower()
        mime = {".pdf": "application/pdf",
                ".svg": "image/svg+xml",
                ".png": "image/png"}.get(ext, "application/octet-stream")
        body += [
            f"--{boundary}".encode(),
            f'Content-Disposition: form-data; name="file{i}"; filename="{p.name}"'.encode(),
            f"Content-Type: {mime}".encode(), b"",
            p.read_bytes(),
        ]
    body.append(f"--{boundary}--".encode())
    req = urllib.request.Request(
        WEBHOOK, data=b"\r\n".join(body), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "User-Agent": "PixelPaws-PDF-Reexport/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            step(f"  Discord upload: HTTP {r.status}")
    except Exception as e:
        step(f"  Discord upload failed: {e}")


# --------------------------------------------------------------------- #
# 1) SNLT CV-F1 summary — reuse render_f1_svg() and swap output to PDF
# --------------------------------------------------------------------- #
def export_classifier_f1_pdf():
    import joblib
    PROJECT = Path(r"E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline")
    JSON_PATH = PROJECT / "transitions" / "for_claude.json"
    OUT_PDF = PROJECT / "transitions" / "_classifier_f1.pdf"
    cfg = json.load(open(JSON_PATH))
    names, mean_f1s, std_f1s, fold_lists = [], [], [], []
    for c in cfg["classifiers"]:
        p = c["path"]
        base = (os.path.basename(p).replace(".pkl", "")
                .replace("PixelPaws_", "").replace("_AllFeatures", ""))
        d = joblib.load(p)
        names.append(base)
        mean_f1s.append(float(d.get("mean_cv_f1", np.nan)))
        std_f1s.append(float(d.get("std_cv_f1", 0.0)))
        fold_lists.append(list(d.get("cv_f1_scores", []) or []))
    n = len(names)
    colors = [plt.cm.inferno(0.20 + 0.55 * (i / max(n - 1, 1))) for i in range(n)]
    fig, ax = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
    x = np.arange(n); bar_w = 0.62
    ax.bar(x, mean_f1s, bar_w, yerr=std_f1s, capsize=5,
           color=colors, edgecolor="black", linewidth=0.6,
           error_kw={"ecolor": "#222", "elinewidth": 1.2, "capthick": 1.2})
    for xi, folds in zip(x, fold_lists):
        if folds:
            jitter = (np.random.RandomState(0).rand(len(folds)) - 0.5) * 0.18
            ax.scatter(np.full(len(folds), xi) + jitter, folds, s=28,
                       color="white", edgecolor="black", linewidth=0.8)
    for xi, mf in zip(x, mean_f1s):
        ax.text(xi, mf + 0.025, f"{mf:.3f}", ha="center", va="bottom",
                fontsize=9.5, color="#222", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.05); ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_ylabel("F1", fontsize=11)
    ax.set_title("SNLT Baseline classifiers — CV-F1 summary",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", linewidth=0.5, color="#bbb")
    ax.set_axisbelow(True)
    fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return OUT_PDF


# --------------------------------------------------------------------- #
# 2) SNTX paw-contour figure → reuse generator + save PDF
# --------------------------------------------------------------------- #
def export_sntx_contour_pdf():
    import sntx_paw_contour_publication_figure as snm

    pooled = snm.load_groups()
    fig = plt.figure(figsize=(9, 13))
    gs = fig.add_gridspec(3, 2, hspace=0.40, wspace=0.28,
                          left=0.08, right=0.97, top=0.95, bottom=0.05)

    role = "HL"; idx = 11
    for col, group in enumerate(snm.GROUP_ORDER):
        ax = fig.add_subplot(gs[0, col])
        shapes_all, shapes_pawlike, n_pl, n_total = pooled[role][group]
        pick = max(0, min(idx - 1, len(shapes_pawlike) - 1))
        snm.plot_paw_print(ax, shapes_pawlike[pick],
                           color=snm.GROUP_COLORS[group],
                           title=f"{group} (#{idx} of {n_pl} paw-like / {n_total} total)",
                           alpha_fill=1.0)
    role = "HR"; idx = 13
    for col, group in enumerate(snm.GROUP_ORDER):
        ax = fig.add_subplot(gs[1, col])
        shapes_all, shapes_pawlike, n_pl, n_total = pooled[role][group]
        pick = max(0, min(idx - 1, len(shapes_pawlike) - 1))
        snm.plot_paw_print(ax, shapes_pawlike[pick],
                           color=snm.GROUP_COLORS[group],
                           title=f"{group} (#{idx} of {n_pl} paw-like / {n_total} total)",
                           alpha_fill=1.0)
    ax_hl = fig.add_subplot(gs[2, 0])
    snm.plot_mean_contour(ax_hl, {g: pooled["HL"][g][1] for g in snm.GROUP_ORDER if g in pooled["HL"]},
                          "Mean Contour — Hind Left")
    ax_hr = fig.add_subplot(gs[2, 1])
    snm.plot_mean_contour(ax_hr, {g: pooled["HR"][g][1] for g in snm.GROUP_ORDER if g in pooled["HR"]},
                          "Mean Contour — Hind Right")

    OUT_PDF = Path(r"E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\analysis"
                   r"\sntx_paw_contour_figure.pdf")
    fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return OUT_PDF


# --------------------------------------------------------------------- #
# 3) Scratching panels B/C/D — pull from cache
# --------------------------------------------------------------------- #
def export_scratching_pdfs():
    import regen_scratching_plots as rsp
    PROJECT = Path(r"E:\RSVIDS\Blackbox\2510_Blackbox_Rimonabant\Blackbox_videos-selected")
    REGEN_DIR = PROJECT / "Classifiers" / "plots" / "regen_20260429_140020"
    CACHE_PKL = REGEN_DIR / "_regen_cache.pkl"
    OUT_DIR = PROJECT / "Classifiers" / "plots" / "pdf_exports"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CACHE_PKL, "rb") as f:
        cache = pickle.load(f)
    train_pkl = PROJECT / "Classifiers" / "training_data" / "Scratching_train_set.pkl"
    with open(train_pkl, "rb") as f:
        ts = pickle.load(f)
    cache["X"] = ts.get("X", ts.get("features"))
    cache["y"] = np.asarray(ts.get("y", ts.get("labels")))

    behavior = "Scratching"
    panelB = OUT_DIR / "PixelPaws_Scratching_panelB_learning.pdf"
    panelC = OUT_DIR / "PixelPaws_Scratching_panelC_threshold.pdf"
    panelD = OUT_DIR / "PixelPaws_Scratching_panelD_shap.pdf"
    rsp.plot_learning_curve(cache["lc"], behavior, str(panelB))
    rsp.plot_threshold_curves(cache["y"], cache["oof"], behavior, str(panelC))
    rsp.plot_shap_panel(cache["final_model"], cache["X"], behavior,
                        str(panelD), sample_n=3500)
    return [panelB, panelC, panelD]


# --------------------------------------------------------------------- #
# 4) FormOxy contour intensity ratio TC — re-render directly
# --------------------------------------------------------------------- #
def export_formoxy_contour_tc_pdf():
    import formoxy_contour_intensity_ratio_tc as fxc
    df = fxc.collect()
    anova = fxc.two_way_anova(df, "ratio_HL_HR")
    perbin_p = fxc.per_bin_kruskal(df, "ratio_HL_HR")
    OUT_PDF = fxc.ANALYSIS / "contour_intensity_ratio_tc.pdf"
    # plot_ratio_tc currently only writes one path; do the work inline
    import matplotlib.pyplot as plt2
    x_min = [i * fxc.BIN_MIN for i in range(fxc.N_BINS)]
    fig, ax = plt2.subplots(figsize=(10.5, 5.8))
    for dose in fxc.DOSE_SEQ:
        sub = df[df["dose"] == dose]
        means, sems = [], []
        for b in range(fxc.N_BINS):
            vals = sub.loc[sub["bin_idx"] == b, "ratio_HL_HR"].dropna().to_numpy()
            means.append(np.mean(vals) if len(vals) else np.nan)
            sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
        c = fxc.DOSE_COLORS[dose]
        ax.errorbar(x_min, means, yerr=sems, marker="o", color=c,
                    linewidth=1.6, markersize=5.5, capsize=3,
                    markerfacecolor=c, markeredgecolor=c, label=dose)
    ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.8)
    anova_txt = (f"Two-way ANOVA: Treatment p={anova['p_treatment']:.3g}, "
                 f"Time p={anova['p_time']:.3g}, "
                 f"Interaction p={anova['p_interaction']:.3g}")
    ax.annotate(anova_txt, xy=(0.01, 0.985), xycoords="axes fraction",
                ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="lightgray"))
    y_top = ax.get_ylim()[1]
    for i, p in enumerate(perbin_p):
        m = fxc.sig_marker(p)
        if m:
            ax.annotate(m, xy=(x_min[i], y_top * 0.985),
                        ha="center", va="top", fontsize=10, color="black")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Contour intensity ratio HL/HR")
    ax.set_title("Contour Intensity Ratio TC (HL formalin / HR control)",
                 fontweight="bold")
    ax.set_xticks(x_min)
    ax.set_xticklabels([str(i * fxc.BIN_MIN) for i in range(fxc.N_BINS)])
    ax.grid(alpha=0.3)
    ax.legend(loc="best", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
    plt2.close(fig)
    return OUT_PDF


# --------------------------------------------------------------------- #
# 5) FormOxy mean paw contour — re-render via existing helpers
# --------------------------------------------------------------------- #
def export_formoxy_mean_contour_pdf():
    import formoxy_mean_contour_analysis as fmc
    per_role = fmc.collect()
    OUT_PDF = fmc.ANALYSIS / "mean_contour_HL_HR.pdf"
    # plot_mean_contours writes PNG; reuse but redirect to PDF via savefig
    import matplotlib.pyplot as plt2

    paw_label = {"HL": "Hind Left", "HR": "Hind Right"}
    fig, axes = plt2.subplots(1, 2, figsize=(11, 5.5), constrained_layout=True)
    all_pts = []
    for ax, role in zip(axes, ("HL", "HR")):
        ax.set_aspect("equal")
        for dose in fmc.DOSE_SEQ:
            if dose not in per_role[role]:
                continue
            stacked = per_role[role][dose]
            if len(stacked) == 0:
                continue
            mean_shape = stacked.mean(axis=0)
            sd_shape = stacked.std(axis=0, ddof=1) if len(stacked) > 1 else np.zeros_like(mean_shape)
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
            color = fmc.DOSE_COLORS[dose]
            ax.fill(ring_x, ring_y, alpha=0.15, color=color, linewidth=0)
            ax.plot(mean_closed[:, 0], mean_closed[:, 1],
                    color=color, linewidth=2.0,
                    label=f"{dose} (n={len(stacked)})")
            all_pts.extend([outer, inner])
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
        ax.axvline(0, color="gray", linewidth=0.5, alpha=0.3)
        ax.set_title(f"Mean Contour — {paw_label[role]}", fontweight="bold")
        ax.set_xlabel("Normalized X"); ax.set_ylabel("Normalized Y")
        ax.legend(fontsize=9, loc="upper right")
        ax.invert_yaxis()
        ax.grid(alpha=0.3)
    if all_pts:
        cat = np.concatenate(all_pts, axis=0)
        x_min, y_min = cat.min(axis=0); x_max, y_max = cat.max(axis=0)
        pad = 0.05 * max(x_max - x_min, y_max - y_min)
        for ax in axes:
            ax.set_xlim(x_min - pad, x_max + pad)
            ax.set_ylim(y_max + pad, y_min - pad)
    fig.suptitle(
        "FormOxy: mean paw contour by dose (combined 2512 + 2605)\n"
        "centred + scaled by sqrt(area); ribbon = +/- 1 SD radial",
        fontweight="bold", fontsize=12,
    )
    fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
    plt2.close(fig)
    return OUT_PDF


# --------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------- #
def main() -> int:
    step("=== 1) SNLT CV-F1 summary ===")
    f1_pdf = export_classifier_f1_pdf()
    step(f"   -> {f1_pdf}  ({f1_pdf.stat().st_size/1024:.0f} KB)")

    step("=== 2) SNTX paw-contour figure ===")
    contour_pdf = export_sntx_contour_pdf()
    step(f"   -> {contour_pdf}  ({contour_pdf.stat().st_size/1024:.0f} KB)")

    step("=== 3) Scratching panels B/C/D ===")
    scratching_pdfs = export_scratching_pdfs()
    for p in scratching_pdfs:
        step(f"   -> {p}  ({p.stat().st_size/1024:.0f} KB)")

    step("=== 4) FormOxy contour intensity ratio TC ===")
    fx_tc_pdf = export_formoxy_contour_tc_pdf()
    step(f"   -> {fx_tc_pdf}  ({fx_tc_pdf.stat().st_size/1024:.0f} KB)")

    step("=== 5) FormOxy mean paw contour ===")
    fx_mc_pdf = export_formoxy_mean_contour_pdf()
    step(f"   -> {fx_mc_pdf}  ({fx_mc_pdf.stat().st_size/1024:.0f} KB)")

    files = [f1_pdf, contour_pdf, fx_tc_pdf, fx_mc_pdf] + scratching_pdfs
    head = (
        "**Illustrator-friendly PDF re-exports**\n"
        "All rendered with `matplotlib.rcParams['pdf.fonttype'] = 42` so "
        "every axis label / legend / title stays as editable text in "
        "Illustrator (no path-rasterised text).\n"
        "Coverage: SNLT CV-F1 summary, SNTX vs Naïve paw contour, "
        "Scratching panels B/C/D, FormOxy contour-intensity TC, FormOxy "
        "mean paw contour. (Temporal probability PDF is in the separate "
        "corrected-analysis upload.)"
    )
    discord_upload(head, files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
