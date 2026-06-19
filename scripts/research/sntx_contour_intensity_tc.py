"""
SNTX vs Naive — 5-min bin contour intensity timecourse (baseline cohort).

Reads `*_Baseline_contour_<hash>.csv` from
  E:\\RSVIDS\\Blackbox\\2603_SNLT_JG\\Baseline\\gait_limb_analysis\\
and the Subject->Treatment mapping from `key_file.csv` (4 SNTX + 3 Naive).

Each contour CSV has columns `intensities_HL` / `intensities_HR` — mean
brightness *inside* the Otsu-thresholded paw outline (NOT the fixed-ROI
mean), which is the same metric used by the formoxy contour-intensity
TC scripts.

Recordings are 30 min @ 60 fps -> 6 bins of 5 min.

Per-bin reduction matches the GUI / formoxy convention: aggregate
per-bin mean(HL) and mean(HR) over frames where both contours were
detected; the ratio is mean(HL)/mean(HR).

Renders a 3-panel figure (PNG + PDF + SVG):
  1. HL contour intensity TC (SNTX vs Naive, mean ± SEM)
  2. HR contour intensity TC
  3. HL/HR ratio TC

Stats per panel:
  - Two-way ANOVA (Treatment × Bin)
  - Per-bin Mann-Whitney U (SNTX vs Naive) with significance asterisks

Posts figure + per-session CSV + per-bin summary CSV to #results.
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
import uuid
from pathlib import Path

PROJECT = Path(r"E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline")
GLAB = PROJECT / "gait_limb_analysis"
ANALYSIS = PROJECT / "analysis"
KEY_FILE = PROJECT / "key_file.csv"

FPS = 60
BIN_MIN = 5
BIN_FRAMES = BIN_MIN * 60 * FPS   # 18 000
N_BINS = 6                        # 30-min session window

GROUP_ORDER = ["Naive", "SNTX"]
GROUP_COLORS = {
    "Naive": "#000000",
    "SNTX":  "#d62728",
}

WEBHOOK = (
    ""
)

SESSION_RE = re.compile(
    r"^(?P<subj>\d+_JG_\d+)_Baseline_contour_[0-9a-f]+\.csv$"
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
        mime = {".png": "image/png", ".pdf": "application/pdf",
                ".svg": "image/svg+xml", ".csv": "text/csv"}.get(
            p.suffix.lower(), "application/octet-stream"
        )
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
                 "User-Agent": "PixelPaws-SNTX-ContourIntensityTC/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            step(f"Discord upload: HTTP {r.status}")
    except Exception as e:
        step(f"Discord upload failed: {e}")


def load_key():
    import pandas as pd
    df = pd.read_csv(KEY_FILE).dropna(how="all")
    return {row["Subject"].strip(): row["Treatment"].strip()
            for _, row in df.iterrows()}


def bin_session(df):
    """Per-bin mean(HL) / mean(HR) and ratio, using only frames where
    both contours were detected (intensity > 0)."""
    import numpy as np
    hl_full = df["intensities_HL"].astype(float).to_numpy()
    hr_full = df["intensities_HR"].astype(float).to_numpy()
    out = []
    for i in range(N_BINS):
        s = i * BIN_FRAMES
        e = s + BIN_FRAMES
        hl = hl_full[s:e]
        hr = hr_full[s:e]
        valid = (hl > 0) & (hr > 0) & np.isfinite(hl) & np.isfinite(hr)
        n_valid = int(valid.sum())
        if n_valid == 0:
            mean_hl = mean_hr = ratio = float("nan")
        else:
            mean_hl = float(np.mean(hl[valid]))
            mean_hr = float(np.mean(hr[valid]))
            ratio = mean_hl / mean_hr if mean_hr > 0 else float("nan")
        out.append({
            "bin_idx": i,
            "bin_min_start": i * BIN_MIN,
            "n_valid_frames": n_valid,
            "contour_HL_mean": mean_hl,
            "contour_HR_mean": mean_hr,
            "ratio_HL_HR": ratio,
        })
    return out


def collect():
    import pandas as pd
    key = load_key()
    step(f"Loaded key with {len(key)} subjects")

    rows = []
    n_files = 0
    for p in sorted(GLAB.glob("*_Baseline_contour_*.csv")):
        m = SESSION_RE.match(p.name)
        if not m:
            continue
        subj = m.group("subj")
        treatment = key.get(subj)
        if treatment is None:
            step(f"  ? subject {subj} not in key; skipping")
            continue
        n_files += 1
        df = pd.read_csv(p, usecols=["intensities_HL", "intensities_HR"])
        for b in bin_session(df):
            rows.append({"subject_id": subj, "treatment": treatment, **b})
    step(f"Collected {n_files} contour CSVs")
    return pd.DataFrame(rows)


def sig_marker(p: float) -> str:
    if p is None or p != p:  # NaN
        return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def two_way_anova(df, value_col: str):
    import pandas as pd
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        sub = df.dropna(subset=[value_col]).copy()
        sub["bin_idx"] = sub["bin_idx"].astype("category")
        sub["treatment"] = pd.Categorical(
            sub["treatment"], categories=GROUP_ORDER, ordered=True)
        model = ols(f"{value_col} ~ C(treatment) * C(bin_idx)", data=sub).fit()
        aov = sm.stats.anova_lm(model, typ=2)
        return {
            "p_treatment":   float(aov.loc["C(treatment)", "PR(>F)"]),
            "p_time":        float(aov.loc["C(bin_idx)", "PR(>F)"]),
            "p_interaction": float(aov.loc["C(treatment):C(bin_idx)", "PR(>F)"]),
        }
    except Exception as e:
        step(f"  statsmodels failed ({e}); ANOVA unavailable")
        return {"p_treatment": float("nan"), "p_time": float("nan"),
                "p_interaction": float("nan")}


def per_bin_mwu(df, value_col: str) -> list[float]:
    from scipy import stats
    pvals = []
    for b in range(N_BINS):
        a = df.loc[(df["bin_idx"] == b) & (df["treatment"] == "SNTX"),
                   value_col].dropna().to_numpy()
        c = df.loc[(df["bin_idx"] == b) & (df["treatment"] == "Naive"),
                   value_col].dropna().to_numpy()
        if len(a) < 2 or len(c) < 2:
            pvals.append(float("nan")); continue
        try:
            _, p = stats.mannwhitneyu(a, c, alternative="two-sided")
            pvals.append(float(p))
        except Exception:
            pvals.append(float("nan"))
    return pvals


def plot_three_panel(df, out_stem: Path):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"]  = 42
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt

    x_min = [(i + 0.5) * BIN_MIN for i in range(N_BINS)]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6),
                             constrained_layout=True)

    panel_defs = [
        ("contour_HL_mean", "HL contour intensity (a.u.)",
         "HL (operated side, by convention)"),
        ("contour_HR_mean", "HR contour intensity (a.u.)",
         "HR (contralateral)"),
        ("ratio_HL_HR",     "Contour intensity ratio HL/HR",
         "HL/HR ratio"),
    ]

    summary_rows = []
    all_anova = {}
    all_perbin_p = {}

    n_per_group = {g: df.loc[df["treatment"] == g, "subject_id"].nunique()
                   for g in GROUP_ORDER}

    for ax, (col, ylabel, title) in zip(axes, panel_defs):
        anova = two_way_anova(df, col)
        all_anova[col] = anova
        perbin_p = per_bin_mwu(df, col)
        all_perbin_p[col] = perbin_p

        for group in GROUP_ORDER:
            sub = df[df["treatment"] == group]
            means, sems = [], []
            for b in range(N_BINS):
                vals = sub.loc[sub["bin_idx"] == b, col].dropna().to_numpy()
                means.append(float(np.mean(vals)) if len(vals) else np.nan)
                sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                            if len(vals) > 1 else 0.0)
                summary_rows.append({
                    "panel": col, "treatment": group,
                    "bin_min_start": b * BIN_MIN, "n": int(len(vals)),
                    "mean": round(means[-1], 4) if not np.isnan(means[-1]) else float("nan"),
                    "sem":  round(sems[-1], 4),
                    "p_mwu_at_bin": round(perbin_p[b], 4) if not np.isnan(perbin_p[b]) else float("nan"),
                    "sig_at_bin": sig_marker(perbin_p[b]),
                })
            c = GROUP_COLORS[group]
            means = np.array(means); sems = np.array(sems)
            ax.plot(x_min, means, marker="o", color=c, linewidth=1.8,
                    markersize=6,
                    label=f"{group} (n={n_per_group[group]})")
            ax.fill_between(x_min, means - sems, means + sems,
                            color=c, alpha=0.18)

        if col == "ratio_HL_HR":
            ax.axhline(1.0, color="#888", linestyle="--", linewidth=1.0,
                       alpha=0.6)

        # Per-bin significance markers
        y_top = ax.get_ylim()[1]
        y_low = ax.get_ylim()[0]
        bump = (y_top - y_low) * 0.04
        for i, p in enumerate(perbin_p):
            marker = sig_marker(p)
            if marker:
                ax.annotate(marker, xy=(x_min[i], y_top - bump),
                            ha="center", va="top", fontsize=11, color="black")

        ax.set_xlabel("Time post-baseline (min)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x_min)
        ax.set_xticklabels([f"{i*BIN_MIN}–{(i+1)*BIN_MIN}" for i in range(N_BINS)],
                           rotation=35, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        anova_txt = (
            f"Treatment p={anova['p_treatment']:.2g}{sig_marker(anova['p_treatment'])}\n"
            f"Time p={anova['p_time']:.2g}{sig_marker(anova['p_time'])}\n"
            f"Interaction p={anova['p_interaction']:.2g}{sig_marker(anova['p_interaction'])}"
        )
        ax.annotate(anova_txt, xy=(0.02, 0.02), xycoords="axes fraction",
                    ha="left", va="bottom", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="lightgray", alpha=0.85))
        ax.legend(loc="upper right", frameon=False, fontsize=9)

    out_png = out_stem.with_suffix(".png")
    out_pdf = out_stem.with_suffix(".pdf")
    out_svg = out_stem.with_suffix(".svg")
    fig.savefig(out_png, format="png", dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    import pandas as pd
    return (out_png, out_pdf, out_svg,
            pd.DataFrame(summary_rows), all_anova, all_perbin_p)


def main() -> int:
    import pandas as pd
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    df = collect()
    if df.empty:
        step("! no contour CSVs collected"); return 1

    n_per_group = df.groupby("treatment")["subject_id"].nunique().to_dict()
    step(f"Per-group N: {n_per_group}")
    step(f"Long-form rows: {len(df):,}")

    long_csv = ANALYSIS / "sntx_contour_intensity_tc_long.csv"
    df.to_csv(long_csv, index=False)
    step(f"Long-form: {long_csv}")

    out_stem = ANALYSIS / "sntx_contour_intensity_tc"
    png, pdf, svg, summary_df, all_anova, all_perbin_p = \
        plot_three_panel(df, out_stem)
    summary_csv = ANALYSIS / "sntx_contour_intensity_tc_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    step(f"Figure: {png}")
    step(f"Summary: {summary_csv}")

    # ── Build Discord header with headline numbers ──
    head_lines = [
        "**SNTX vs Naive — contour intensity TC (5-min bins, 30-min "
        "baseline session)**",
        f"Pooled baseline cohort: **Naive n={n_per_group.get('Naive',0)}, "
        f"SNTX n={n_per_group.get('SNTX',0)}**, "
        "mean ± SEM, contour intensity = mean brightness *inside* the "
        "Otsu-thresholded paw outline.",
    ]
    for col, label in [("contour_HL_mean", "HL intensity"),
                       ("contour_HR_mean", "HR intensity"),
                       ("ratio_HL_HR",     "HL/HR ratio")]:
        a = all_anova[col]
        head_lines.append(
            f"• {label}: 2-way ANOVA — Treatment "
            f"p={a['p_treatment']:.3g}{sig_marker(a['p_treatment'])}, "
            f"Time p={a['p_time']:.3g}{sig_marker(a['p_time'])}, "
            f"Interaction p={a['p_interaction']:.3g}{sig_marker(a['p_interaction'])}"
        )
    head_lines.append(
        "Per-bin asterisks above each line = Mann-Whitney U "
        "(SNTX vs Naive at that bin)."
    )
    head = "\n".join(head_lines)
    discord_upload(head, [png, pdf, svg, summary_csv, long_csv])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
