"""
Intensity-ratio HL/HR timecourse computed from CONTOUR intensities (mean
brightness inside the Otsu-thresholded paw outline) rather than the
fixed-ROI mean.

Why contour > ROI for this metric:
  - The ROI mean (Pix_hlpaw / Pix_hrpaw in the brightness pass) includes
    floor pixels around the paw, so when the paw is briefly lifted the
    ROI fills with dark floor and the per-frame HL/HR ratio explodes
    (we saw single bins hit 7+ in the ROI plot).
  - The contour intensity is computed only inside the detected paw
    contour, ignoring background. Distribution is tight (std ~ 3-8
    pixel-intensity units, range ~ 25-125) and the L/R ratio is a clean
    biological signal: brighter contour = paw more illuminated / more
    exposed; dimmer contour = paw tucked / partially occluded.

Per-bin reduction matches the GUI's convention: aggregate per-bin
mean(HL) and mean(HR) FIRST, then take the ratio (NOT per-frame ratio
averaged), so a few low-HR frames can't blow up the bin mean.

Reads `*_contour_<hash>.csv` from both cohorts and posts the GUI-style
TC plot + per-bin summary CSV to #results.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
import uuid
from pathlib import Path

COHORT_NEW = Path(r"E:\RSVIDS\Blackbox\2605_FormOxy_LeftPaws")
NEW_DIR = COHORT_NEW / "gait_limb_analysis"
ANALYSIS = COHORT_NEW / "analysis"

COHORT_OLD = Path(r"E:\RSVIDS\Blackbox\2512_Blackbox_Formalin_Oxy\Left_paws")
OLD_DIR = COHORT_OLD / "gait_limb_analysis"

FPS = 60
BIN_MIN = 5
BIN_FRAMES = BIN_MIN * 60 * FPS
N_BINS = 12

DOSE_2512 = {1: "Veh", 2: "Veh", 3: "Veh",
             4: "Oxy3", 5: "Oxy3", 6: "Oxy3",
             7: "Oxy10", 8: "Oxy10", 9: "Oxy10",
             10: "Oxy1", 11: "Oxy1", 12: "Oxy1"}
DOSE_SEQ = ["Veh", "Oxy1", "Oxy3", "Oxy10"]


def dose_2605(n: int) -> str:
    return DOSE_SEQ[(n - 1) % 4]


DOSE_COLORS = {
    "Veh":   "#000000",
    "Oxy1":  "#a6cee3",
    "Oxy3":  "#1f78b4",
    "Oxy10": "#08306b",
}

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
        mime = "image/png" if p.suffix == ".png" else "text/csv"
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
                 "User-Agent": "PixelPaws-FormOxy-ContourRatioTC/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            step(f"Discord upload: HTTP {r.status}")
    except Exception as e:
        step(f"Discord upload failed: {e}")


def bin_session(df):
    """Per-bin mean(intensities_HL) / mean(intensities_HR).

    Drops frames where either contour was not detected (intensity == 0,
    the array's initial fill value — contour code only writes non-zero
    when area > MIN_AREA contour was found).
    """
    import numpy as np
    out = []
    hl_full = df["intensities_HL"].astype(float).to_numpy()
    hr_full = df["intensities_HR"].astype(float).to_numpy()
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
    rows = []

    rx_new = re.compile(r"^2605_FormOxy_S(\d+)_formalin_contour_[0-9a-f]+\.csv$")
    for p in sorted(NEW_DIR.glob("2605_FormOxy_S*_formalin_contour_*.csv")):
        m = rx_new.match(p.name)
        if not m:
            continue
        subj = int(m.group(1))
        df = pd.read_csv(p, usecols=["intensities_HL", "intensities_HR"])
        for b in bin_session(df):
            rows.append({"cohort": "2605", "subject_id": f"2605_S{subj}",
                         "dose": dose_2605(subj), **b})

    rx_old = re.compile(r"^2512_FormOxy_S(\d+)_contour_[0-9a-f]+\.csv$")
    for p in sorted(OLD_DIR.glob("2512_FormOxy_S*_contour_*.csv")):
        m = rx_old.match(p.name)
        if not m:
            continue
        subj = int(m.group(1))
        df = pd.read_csv(p, usecols=["intensities_HL", "intensities_HR"])
        for b in bin_session(df):
            rows.append({"cohort": "2512", "subject_id": f"2512_S{subj}",
                         "dose": DOSE_2512[subj], **b})

    return pd.DataFrame(rows)


def sig_marker(p: float) -> str:
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
        sub["dose"]    = pd.Categorical(sub["dose"], categories=DOSE_SEQ, ordered=True)
        model = ols(f"{value_col} ~ C(dose) * C(bin_idx)", data=sub).fit()
        aov = sm.stats.anova_lm(model, typ=2)
        return {
            "p_treatment":   float(aov.loc["C(dose)", "PR(>F)"]),
            "p_time":        float(aov.loc["C(bin_idx)", "PR(>F)"]),
            "p_interaction": float(aov.loc["C(dose):C(bin_idx)", "PR(>F)"]),
        }
    except Exception as e:
        step(f"  statsmodels failed ({e}); ANOVA unavailable")
        return {"p_treatment": float("nan"), "p_time": float("nan"),
                "p_interaction": float("nan")}


def per_bin_kruskal(df, value_col: str) -> list[float]:
    from scipy import stats
    pvals = []
    for b in range(N_BINS):
        groups = [df.loc[(df["bin_idx"] == b) & (df["dose"] == d), value_col].dropna().to_numpy()
                  for d in DOSE_SEQ]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            pvals.append(float("nan")); continue
        try:
            _, p = stats.kruskal(*groups)
            pvals.append(float(p))
        except Exception:
            pvals.append(float("nan"))
    return pvals


def plot_ratio_tc(df, out_png: Path, anova: dict, perbin_p: list[float]):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_min = [i * BIN_MIN for i in range(N_BINS)]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    for dose in DOSE_SEQ:
        sub = df[df["dose"] == dose]
        means, sems = [], []
        for b in range(N_BINS):
            vals = sub.loc[sub["bin_idx"] == b, "ratio_HL_HR"].dropna().to_numpy()
            means.append(np.mean(vals) if len(vals) else np.nan)
            sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
        c = DOSE_COLORS[dose]
        ax.errorbar(x_min, means, yerr=sems, marker="o", color=c,
                    linewidth=1.6, markersize=5.5, capsize=3,
                    markerfacecolor=c, markeredgecolor=c, label=dose)

    ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.8)

    anova_txt = (
        f"Two-way ANOVA: Treatment p={anova['p_treatment']:.3g}{sig_marker(anova['p_treatment'])}, "
        f"Time p={anova['p_time']:.3g}{sig_marker(anova['p_time'])}, "
        f"Interaction p={anova['p_interaction']:.3g}{sig_marker(anova['p_interaction'])}"
    )
    ax.annotate(anova_txt, xy=(0.01, 0.985), xycoords="axes fraction",
                ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="lightgray"))

    y_top = ax.get_ylim()[1]
    for i, p in enumerate(perbin_p):
        marker = sig_marker(p)
        if marker:
            ax.annotate(marker, xy=(x_min[i], y_top * 0.985),
                        ha="center", va="top", fontsize=10, color="black")

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Contour intensity ratio HL/HR")
    ax.set_title("Contour Intensity Ratio -- TC (HL formalin / HR control)",
                 fontweight="bold")
    ax.set_xticks(x_min)
    ax.set_xticklabels([str(i * BIN_MIN) for i in range(N_BINS)])
    ax.grid(alpha=0.3)
    ax.legend(loc="best", frameon=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main() -> int:
    import pandas as pd
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    df = collect()
    if df.empty:
        step("! no contour CSVs collected"); return 1
    step(f"Collected {len(df)} bin-rows across {df['subject_id'].nunique()} sessions")

    long_csv = ANALYSIS / "contour_intensity_ratio_tc_long.csv"
    df.to_csv(long_csv, index=False)
    step(f"Long-form: {long_csv}")
    step(f"ratio stats: min={df['ratio_HL_HR'].min():.3f}  "
         f"median={df['ratio_HL_HR'].median():.3f}  max={df['ratio_HL_HR'].max():.3f}")

    anova = two_way_anova(df, "ratio_HL_HR")
    step(f"ANOVA: {anova}")
    perbin_p = per_bin_kruskal(df, "ratio_HL_HR")
    step("Per-bin Kruskal-Wallis (4 doses):")
    for i, p in enumerate(perbin_p):
        step(f"  bin {i*BIN_MIN:>2}-{(i+1)*BIN_MIN:>2} min: p={p:.3g}  {sig_marker(p)}")

    out_png = ANALYSIS / "contour_intensity_ratio_tc.png"
    plot_ratio_tc(df, out_png, anova, perbin_p)
    step(f"Figure: {out_png}")

    # Compact per-dose-per-bin summary CSV
    import numpy as np
    rows = []
    for dose in DOSE_SEQ:
        sub = df[df["dose"] == dose]
        for b in range(N_BINS):
            vals = sub.loc[sub["bin_idx"] == b, "ratio_HL_HR"].dropna().to_numpy()
            rows.append({
                "dose": dose, "bin_min_start": b * BIN_MIN, "n": int(len(vals)),
                "mean": round(float(np.mean(vals)), 4) if len(vals) else float("nan"),
                "sem":  round(float(np.std(vals, ddof=1) / np.sqrt(len(vals))), 4) if len(vals) > 1 else 0.0,
                "p_omnibus_at_bin": round(perbin_p[b], 4),
                "sig_at_bin": sig_marker(perbin_p[b]),
            })
    summary_csv = ANALYSIS / "contour_intensity_ratio_tc_summary.csv"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)

    head = (
        "**FormOxy contour intensity ratio HL/HR -- TC (combined 2512 + 2605)**\n"
        "Mean intensity *inside* the Otsu-thresholded paw contour (cleaner than "
        "the fixed-ROI mean, which included floor pixels around the paw and gave "
        "single-bin ratios up to 7.3 on rare frames where the paw was lifted).\n"
        "Per-bin reduction: `mean(intensities_HL) / mean(intensities_HR)` over "
        "frames where both contours were detected. 5-min bins, first 60 min, "
        "n=6 per dose, mean +/- SEM.\n"
        f"Two-way ANOVA: Treatment p={anova['p_treatment']:.3g}{sig_marker(anova['p_treatment'])}, "
        f"Time p={anova['p_time']:.3g}{sig_marker(anova['p_time'])}, "
        f"Interaction p={anova['p_interaction']:.3g}{sig_marker(anova['p_interaction'])}.\n"
        f"Per-bin asterisks = omnibus Kruskal-Wallis across the 4 doses."
    )
    discord_upload(head, [out_png, summary_csv])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
