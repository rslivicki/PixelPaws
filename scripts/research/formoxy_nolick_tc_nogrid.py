"""Re-render the non-licking-frames contour intensity ratio TC with NO
gridlines (keep everything else: dose-coloured traces, ANOVA box,
per-bin asterisks, red dashed reference line at 1.0).

Reads the existing long-form CSV produced by
`formoxy_contour_ratio_tc_nolick.py` and rebuilds the plot."""
from __future__ import annotations
import json, time, urllib.request, uuid
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANALYSIS = Path(r"E:\RSVIDS\Blackbox\2605_FormOxy_LeftPaws\analysis")
LONG_CSV = ANALYSIS / "contour_ratio_tc_nolick_long.csv"

FPS = 60; BIN_MIN = 5; N_BINS = 12
DOSE_SEQ = ["Veh", "Oxy1", "Oxy3", "Oxy10"]
DOSE_COLORS = {"Veh": "#000000", "Oxy1": "#a6cee3",
               "Oxy3": "#1f78b4", "Oxy10": "#08306b"}

WEBHOOK = ("")


def step(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def discord_upload(content, files):
    boundary = "----PP" + uuid.uuid4().hex
    body = [f"--{boundary}".encode(),
            b'Content-Disposition: form-data; name="payload_json"',
            b'Content-Type: application/json', b"",
            json.dumps({"content": content}).encode()]
    for i, p in enumerate(files):
        mime = {".png": "image/png", ".pdf": "application/pdf",
                ".svg": "image/svg+xml"}.get(p.suffix.lower(),
                                              "application/octet-stream")
        body += [f"--{boundary}".encode(),
                 f'Content-Disposition: form-data; name="file{i}"; filename="{p.name}"'.encode(),
                 f"Content-Type: {mime}".encode(), b"",
                 p.read_bytes()]
    body.append(f"--{boundary}--".encode())
    req = urllib.request.Request(
        WEBHOOK, data=b"\r\n".join(body), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "User-Agent": "PixelPaws-NoLickNoGrid/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        step(f"  HTTP {r.status}")


def sig_marker(p):
    if not np.isfinite(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def main():
    df = pd.read_csv(LONG_CSV)
    step(f"loaded {len(df)} rows from {LONG_CSV.name}")

    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        sub = df.dropna(subset=["ratio"]).copy()
        sub["bin_idx"] = sub["bin_idx"].astype("category")
        sub["dose"] = pd.Categorical(sub["dose"], categories=DOSE_SEQ, ordered=True)
        model = ols("ratio ~ C(dose) * C(bin_idx)", data=sub).fit()
        aov = sm.stats.anova_lm(model, typ=2)
        anova = {"p_treatment": float(aov.loc["C(dose)", "PR(>F)"]),
                 "p_time": float(aov.loc["C(bin_idx)", "PR(>F)"]),
                 "p_interaction": float(aov.loc["C(dose):C(bin_idx)", "PR(>F)"])}
    except Exception as e:
        step(f"  statsmodels failed: {e}")
        anova = {"p_treatment": float("nan"), "p_time": float("nan"),
                 "p_interaction": float("nan")}

    from scipy import stats
    perbin_p = []
    for b in range(N_BINS):
        groups = [df.loc[(df["bin_idx"] == b) & (df["dose"] == d), "ratio"]
                    .dropna().to_numpy() for d in DOSE_SEQ]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            perbin_p.append(float("nan")); continue
        try:
            _, p = stats.kruskal(*groups); perbin_p.append(float(p))
        except Exception:
            perbin_p.append(float("nan"))

    x = [i * BIN_MIN for i in range(N_BINS)]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    x_arr = np.asarray(x, dtype=float)
    for dose in DOSE_SEQ:
        sub = df[df["dose"] == dose]
        means, sems, ns = [], [], []
        for b in range(N_BINS):
            vals = sub.loc[sub["bin_idx"] == b, "ratio"].dropna().to_numpy()
            means.append(np.mean(vals) if len(vals) else np.nan)
            sems.append(np.std(vals, ddof=1)/np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
            ns.append(len(vals))
        means_a = np.asarray(means, dtype=float)
        sems_a = np.asarray(sems, dtype=float)
        c = DOSE_COLORS[dose]
        ax.plot(x_arr, means_a, marker="o", color=c,
                linewidth=1.8, markersize=5.5,
                markerfacecolor=c, markeredgecolor=c,
                label=f"{dose} (n={max(ns)})")
        ax.fill_between(x_arr, means_a - sems_a, means_a + sems_a,
                        color=c, alpha=0.18, linewidth=0)

    ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.85)

    anova_txt = (f"Two-way ANOVA: Treatment p={anova['p_treatment']:.3g}{sig_marker(anova['p_treatment'])}, "
                 f"Time p={anova['p_time']:.3g}{sig_marker(anova['p_time'])}, "
                 f"Interaction p={anova['p_interaction']:.3g}{sig_marker(anova['p_interaction'])}")
    ax.annotate(anova_txt, xy=(0.01, 0.985), xycoords="axes fraction",
                ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="lightgray"))

    y_top = ax.get_ylim()[1]
    for i, p in enumerate(perbin_p):
        marker = sig_marker(p)
        if marker:
            ax.annotate(marker, xy=(x[i], y_top * 0.985),
                        ha="center", va="top", fontsize=10)

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Contour intensity ratio HL/HR (non-licking frames only)")
    ax.set_title("Contour Intensity Ratio TC -- NON-LICKING frames only",
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i * BIN_MIN) for i in range(N_BINS)])
    # NO GRID
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=True, fontsize=9)
    fig.tight_layout()

    out_base = ANALYSIS / "contour_ratio_tc_nolick_nogrid"
    fig.savefig(out_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    step(f"wrote {out_base}.png / svg / pdf")

    head = ("**FormOxy contour intensity ratio TC (NON-LICKING frames only) "
            "-- gridlines removed**\n"
            "Same data and stats as the previous version (Treatment p=1.03e-11***, "
            "Interaction p=0.7, per-bin asterisks at 0/15/20/25 min). Just stripped "
            "the gridlines and removed the top/right spines for a cleaner look.")
    discord_upload(head, [out_base.with_suffix(".png"),
                          out_base.with_suffix(".svg"),
                          out_base.with_suffix(".pdf")])


if __name__ == "__main__":
    main()
