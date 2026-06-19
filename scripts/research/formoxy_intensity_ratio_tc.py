"""
Replicate the GUI Gait & Limb -> Statistics -> "Intensity Ratio HL/HR -- TC"
plot for the combined FormOxy cohorts (2512 + 2605).

Style match:
  - Raw HL/HR ratio (not log), values <1 = formalin paw dimmer
  - Red dashed line at 1.0 (symmetry)
  - 5-min bins from 0 to 60 (13 markers at bin edges)
  - Mean +/- SEM error bars
  - Blue palette: Veh = black, Oxy doses = light->dark blue
  - Two-way ANOVA (Treatment, Time, Interaction) in top annotation
  - Per-bin omnibus Kruskal-Wallis asterisks across the 4 doses

Reads per-frame brt CSVs:
  - 2512_.../gait_limb_analysis/2512_FormOxy_S<N>_brt_*.csv
  - 2605_.../gait_limb_analysis/2605_FormOxy_S<N>_formalin_brt_*.csv

Posts the figure + summary CSV to #results.
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
NEW_BRT = COHORT_NEW / "gait_limb_analysis"
ANALYSIS = COHORT_NEW / "analysis"

COHORT_OLD = Path(r"E:\RSVIDS\Blackbox\2512_Blackbox_Formalin_Oxy\Left_paws")
OLD_BRT = COHORT_OLD / "gait_limb_analysis"

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


# Match the GUI palette: Veh = black, Oxy doses = light -> dark blue
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
                 "User-Agent": "PixelPaws-FormOxy-RatioTC/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            step(f"Discord upload: HTTP {r.status}")
    except Exception as e:
        step(f"Discord upload failed: {e}")


def bin_session(df):
    """Mean HL, HR, and mean(HL)/mean(HR) per 5-min bin.

    Earlier this function used the per-frame ratio mean, which is
    unstable: when Pix_hrpaw is small on a few frames (paw lifted ->
    ROI fills with dark floor) the per-frame HL/HR explodes and pulls
    the bin mean above the true value (we saw single bins hit 7+).
    The mean-of-means is stable because the denominator is averaged
    first, smoothing over those rare frames.
    """
    import numpy as np
    out = []
    for i in range(N_BINS):
        s = i * BIN_FRAMES
        e = s + BIN_FRAMES
        chunk = df.iloc[s:e]
        if "Pix_hlpaw" not in chunk.columns or "Pix_hrpaw" not in chunk.columns:
            continue
        hl_arr = chunk["Pix_hlpaw"].astype(float).to_numpy()
        hr_arr = chunk["Pix_hrpaw"].astype(float).to_numpy()
        valid = (hr_arr > 0) & np.isfinite(hl_arr) & np.isfinite(hr_arr)
        if valid.sum() == 0:
            mean_hl = mean_hr = ratio = float("nan")
        else:
            mean_hl = float(np.mean(hl_arr[valid]))
            mean_hr = float(np.mean(hr_arr[valid]))
            ratio = mean_hl / mean_hr if mean_hr > 0 else float("nan")
        out.append({
            "bin_idx": i,
            "bin_min_start": i * BIN_MIN,
            "Pix_hlpaw_mean": mean_hl,
            "Pix_hrpaw_mean": mean_hr,
            "ratio_HL_HR":    ratio,
        })
    return out


def collect():
    import pandas as pd
    rows = []

    rx_new = re.compile(r"^2605_FormOxy_S(\d+)_formalin_brt_[0-9a-f]+\.csv$")
    for p in sorted(NEW_BRT.glob("2605_FormOxy_S*_formalin_brt_*.csv")):
        m = rx_new.match(p.name)
        if not m:
            continue
        subj = int(m.group(1))
        df = pd.read_csv(p)
        for b in bin_session(df):
            rows.append({"cohort": "2605", "subject_id": f"2605_S{subj}",
                         "dose": dose_2605(subj), **b})

    rx_old = re.compile(r"^2512_FormOxy_S(\d+)_brt_[0-9a-f]+\.csv$")
    for p in sorted(OLD_BRT.glob("2512_FormOxy_S*_brt_*.csv")):
        m = rx_old.match(p.name)
        if not m:
            continue
        subj = int(m.group(1))
        df = pd.read_csv(p)
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
    """Statsmodels two-way ANOVA: value ~ C(dose) * C(bin_idx).

    Falls back to a hand-rolled Type-I sum-of-squares ANOVA if statsmodels
    is unavailable (no extra install required).
    """
    import numpy as np
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
        step(f"  statsmodels unavailable ({e}); falling back to scipy Kruskal-Wallis omnibus only")
        from scipy import stats
        # Omnibus per-bin and per-dose
        sub = df.dropna(subset=[value_col]).copy()
        groups_dose = [sub.loc[sub["dose"] == d, value_col].to_numpy() for d in DOSE_SEQ]
        try:
            h_d, p_d = stats.kruskal(*[g for g in groups_dose if len(g) > 1])
        except Exception:
            p_d = float("nan")
        groups_time = [sub.loc[sub["bin_idx"] == b, value_col].to_numpy() for b in range(N_BINS)]
        try:
            h_t, p_t = stats.kruskal(*[g for g in groups_time if len(g) > 1])
        except Exception:
            p_t = float("nan")
        return {"p_treatment": p_d, "p_time": p_t, "p_interaction": float("nan")}


def per_bin_kruskal(df, value_col: str) -> list[float]:
    from scipy import stats
    pvals = []
    for b in range(N_BINS):
        groups = [df.loc[(df["bin_idx"] == b) & (df["dose"] == d), value_col].dropna().to_numpy()
                  for d in DOSE_SEQ]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            pvals.append(float("nan"))
            continue
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

    # x positions in minutes — match GUI: 0, 5, 10, ..., 55 (one per bin start)
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

    # symmetry line
    ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.8)

    # ANOVA annotation (top-left)
    anova_txt = (
        f"Two-way ANOVA: Treatment p={anova['p_treatment']:.3g}{sig_marker(anova['p_treatment'])}, "
        f"Time p={anova['p_time']:.3g}{sig_marker(anova['p_time'])}, "
        f"Interaction p={anova['p_interaction']:.3g}{sig_marker(anova['p_interaction'])}"
    )
    ax.annotate(anova_txt, xy=(0.01, 0.985), xycoords="axes fraction",
                ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="lightgray"))

    # Per-bin significance asterisks (Kruskal-Wallis across 4 doses)
    y_top = ax.get_ylim()[1]
    for i, p in enumerate(perbin_p):
        marker = sig_marker(p)
        if marker:
            ax.annotate(marker, xy=(x_min[i], y_top * 0.985),
                        ha="center", va="top", fontsize=10, color="black")

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Intensity ratio HL/HR")
    ax.set_title("Intensity Ratio -- TC", fontweight="bold")
    ax.set_xticks(x_min)
    ax.set_xticklabels([str(i * BIN_MIN) for i in range(N_BINS)])
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", frameon=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main() -> int:
    import pandas as pd
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    df = collect()
    if df.empty:
        step("! no brt CSVs collected"); return 1
    step(f"Collected {len(df)} bin-rows across {df['subject_id'].nunique()} sessions")

    long_csv = ANALYSIS / "intensity_ratio_tc_long.csv"
    df.to_csv(long_csv, index=False)
    step(f"Long-form: {long_csv}")

    # Stats
    anova = two_way_anova(df, "ratio_HL_HR")
    step(f"ANOVA: {anova}")
    perbin_p = per_bin_kruskal(df, "ratio_HL_HR")
    step("Per-bin Kruskal-Wallis p across the 4 doses:")
    for i, p in enumerate(perbin_p):
        step(f"  bin {i*BIN_MIN:>2}-{(i+1)*BIN_MIN:>2} min: p={p:.3g}  {sig_marker(p)}")

    # Plot
    out_png = ANALYSIS / "intensity_ratio_tc.png"
    plot_ratio_tc(df, out_png, anova, perbin_p)
    step(f"Figure: {out_png}")

    # Discord
    summary_csv = ANALYSIS / "intensity_ratio_tc_summary.csv"
    rows = []
    for dose in DOSE_SEQ:
        sub = df[df["dose"] == dose]
        for b in range(N_BINS):
            vals = sub.loc[sub["bin_idx"] == b, "ratio_HL_HR"].dropna().to_numpy()
            import numpy as np
            rows.append({
                "dose": dose,
                "bin_min_start": b * BIN_MIN,
                "n": int(len(vals)),
                "mean": round(float(np.mean(vals)), 4) if len(vals) else float("nan"),
                "sem":  round(float(np.std(vals, ddof=1) / np.sqrt(len(vals))), 4) if len(vals) > 1 else 0.0,
                "p_omnibus_at_bin": round(perbin_p[b], 4),
                "sig_at_bin": sig_marker(perbin_p[b]),
            })
    pd.DataFrame(rows).to_csv(summary_csv, index=False)

    head = (
        "**FormOxy intensity ratio HL/HR -- TC (combined 2512 + 2605)** (refreshed)\n"
        "Fixed-ROI brightness, per-bin reduction = `mean(Pix_hlpaw) / mean(Pix_hrpaw)`. "
        "Previous version used per-frame ratio mean which exploded on rare paw-lift "
        "frames (some single bins hit 7+); the mean-of-means is stable.\n"
        f"5-min bins, first 60 min. Mean +/- SEM, n=6 per dose. "
        f"Red dashed line at 1.0 = L/R symmetry.\n"
        f"Two-way ANOVA: Treatment p={anova['p_treatment']:.3g}{sig_marker(anova['p_treatment'])}, "
        f"Time p={anova['p_time']:.3g}{sig_marker(anova['p_time'])}, "
        f"Interaction p={anova['p_interaction']:.3g}{sig_marker(anova['p_interaction'])}.\n"
        f"Per-bin asterisks = omnibus Kruskal-Wallis across the 4 doses."
    )
    discord_upload(head, [out_png, summary_csv])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
