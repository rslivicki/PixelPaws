"""
Re-export the two SNLT-baseline figures as SVG and post to #results:

  1. SNLT classifier CV-F1 summary bar chart — re-renders directly with
     matplotlib's svg backend, reading the same `for_claude.json` +
     classifier pickle metadata as `_render_snlt_classifier_f1.py`.

  2. Naïve vs SNTX temporal-probability stacked-area chart — converts
     the existing vector PDF at
     `2603_SNLT_JG\Baseline\analysis\transitions\temporal_probability.pdf`
     to SVG via PyMuPDF (`fitz`). PDF is already vector so the SVG is
     a faithful re-encoding.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import uuid
from pathlib import Path

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(r"E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline")
JSON_PATH = PROJECT / "transitions" / "for_claude.json"
F1_SVG = PROJECT / "transitions" / "_classifier_f1.svg"
F1_PNG = PROJECT / "transitions" / "_classifier_f1.png"
TP_PDF = PROJECT / "analysis" / "transitions" / "temporal_probability.pdf"
TP_SVG = PROJECT / "analysis" / "transitions" / "temporal_probability.svg"

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
        mime = {".svg": "image/svg+xml", ".png": "image/png",
                ".pdf": "application/pdf"}.get(ext, "application/octet-stream")
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
                 "User-Agent": "PixelPaws-SNLT-SVG/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            step(f"Discord upload: HTTP {r.status}")
    except Exception as e:
        step(f"Discord upload failed: {e}")


# --------------------------------------------------------------------- #
# Figure 1: CV-F1 summary, SVG export (same code path as the PNG renderer)
# --------------------------------------------------------------------- #
def render_f1_svg() -> Path:
    cfg = json.load(open(JSON_PATH))
    clfs = cfg["classifiers"]
    names, mean_f1s, std_f1s, fold_lists = [], [], [], []
    for c in clfs:
        p = c["path"]
        base = (os.path.basename(p)
                .replace(".pkl", "")
                .replace("PixelPaws_", "")
                .replace("_AllFeatures", ""))
        d = joblib.load(p)
        names.append(base)
        mean_f1s.append(float(d.get("mean_cv_f1", np.nan)))
        std_f1s.append(float(d.get("std_cv_f1", 0.0)))
        fold_lists.append(list(d.get("cv_f1_scores", []) or []))
        step(f"  {base:25s}  mean={mean_f1s[-1]:.3f}+/-{std_f1s[-1]:.3f}  "
             f"folds={fold_lists[-1]}")

    n = len(names)
    colors = [plt.cm.inferno(0.20 + 0.55 * (i / max(n - 1, 1))) for i in range(n)]

    fig, ax = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
    x = np.arange(n)
    bar_w = 0.62

    ax.bar(x, mean_f1s, bar_w, yerr=std_f1s, capsize=5,
           color=colors, edgecolor="black", linewidth=0.6,
           error_kw={"ecolor": "#222", "elinewidth": 1.2, "capthick": 1.2},
           zorder=2, label="Mean CV-F1 (±std)")
    for xi, folds in zip(x, fold_lists):
        if folds:
            jitter = (np.random.RandomState(0).rand(len(folds)) - 0.5) * 0.18
            ax.scatter(np.full(len(folds), xi) + jitter, folds, s=28,
                       color="white", edgecolor="black", linewidth=0.8,
                       zorder=3, label="Per-fold" if xi == 0 else None)
    for xi, mf in zip(x, mean_f1s):
        ax.text(xi, mf + 0.025, f"{mf:.3f}", ha="center", va="bottom",
                fontsize=9.5, color="#222", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.05); ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_ylabel("F1", fontsize=11)
    ax.set_title("SNLT Baseline classifiers — CV-F1 summary",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", linewidth=0.5, color="#bbb", zorder=1)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    seen = set(); h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    ax.legend(h2, l2, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9)

    F1_SVG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(F1_SVG, format="svg", bbox_inches="tight")
    fig.savefig(F1_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)
    step(f"  -> {F1_SVG}  ({F1_SVG.stat().st_size/1024:.0f} KB)")
    return F1_SVG


# --------------------------------------------------------------------- #
# Figure 2: temporal-probability SVG via PDF -> SVG (PDF is vector already)
# --------------------------------------------------------------------- #
def convert_pdf_to_svg(pdf_path: Path, svg_path: Path) -> Path:
    import fitz   # PyMuPDF
    doc = fitz.open(str(pdf_path))
    if len(doc) == 0:
        raise RuntimeError(f"empty PDF: {pdf_path}")
    page = doc[0]
    svg_text = page.get_svg_image()
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(svg_text, encoding="utf-8")
    doc.close()
    step(f"  -> {svg_path}  ({svg_path.stat().st_size/1024:.0f} KB)")
    return svg_path


def main() -> int:
    step("Rendering CV-F1 summary SVG...")
    f1_svg = render_f1_svg()

    step("Converting temporal-probability PDF -> SVG...")
    if TP_PDF.is_file():
        tp_svg = convert_pdf_to_svg(TP_PDF, TP_SVG)
    else:
        step(f"  ! PDF not found: {TP_PDF}")
        tp_svg = None

    files = [f1_svg, F1_PNG]
    if tp_svg is not None:
        files.append(tp_svg)
        files.append(TP_PDF)   # include the source PDF too for cross-check

    head = (
        "**SNLT Baseline — vector exports**\n"
        "• `_classifier_f1.svg` — CV-F1 summary bar chart (re-rendered "
        "via matplotlib SVG backend; PNG companion for thumbnail).\n"
        + ("• `temporal_probability.svg` — Naïve vs SNTX stacked-area "
           "state-occupancy chart, converted from the existing vector "
           "PDF via PyMuPDF (source PDF also attached for cross-check).\n"
           if tp_svg is not None else
           "• `temporal_probability` not found at expected path.\n")
    )
    discord_upload(head, files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
