"""
Contour HL/HR intensity ratio timecourse computed on NON-LICKING
frames only. Same GUI-style plot as formoxy_contour_intensity_ratio_tc.py
but with `Left_licking == 0` masking applied per bin, so the ratio is
no longer contaminated by the mechanical paw-lift / occlusion that
happens during licking events.

The decomposition analysis showed ~72% of the dose effect on the
all-frame ratio survives when you exclude licking frames — this plot
shows that residual dose-response across time.

Reads per-frame licking predictions + contour intensities for both
cohorts, joins per-frame, drops licking frames + invalid contour
frames, then per 5-min bin reduces via mean(HL)/mean(HR).
Posts a single-panel TC + per-bin summary CSV to #results.
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
NEW_LICK = COHORT_NEW / "results"
NEW_CTR = COHORT_NEW / "gait_limb_analysis"
ANALYSIS = COHORT_NEW / "analysis"

COHORT_OLD = Path(r"E:\RSVIDS\Blackbox\2512_Blackbox_Formalin_Oxy\Left_paws")
OLD_LICK = COHORT_OLD / "Results" / "Left_licking"
OLD_CTR = COHORT_OLD / "gait_limb_analysis"

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
                 "User-Agent": "PixelPaws-FormOxy-RatioTCNoLick/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            step(f"Discord upload: HTTP {r.status}")
    except Exception as e:
        step(f"Discord upload failed: {e}")


def per_session_bins(lick_csv: Path, ctr_csv: Path):
    """Per-bin (HL_mean, HR_mean, ratio) on NON-LICKING frames only."""
    import numpy as np
    import pandas as pd
    df_l = pd.read_csv(lick_csv, usecols=["Left_licking"])
    df_c = pd.read_csv(ctr_csv,  usecols=["intensities_HL", "intensities_HR"])
    n = min(len(df_l), len(df_c), N_BINS * BIN_FRAMES)
    y  = df_l["Left_licking"].astype(int).to_numpy()[:n]
    hl = df_c["intensities_HL"].astype(float).to_numpy()[:n]
    hr = df_c["intensities_HR"].astype(float).to_numpy()[:n]

    out = []
    for b in range(N_BINS):
        s = b * BIN_FRAMES; e = min(s + BIN_FRAMES, n)
        if e <= s:
            out.append({"bin_idx": b, "ratio": float("nan"), "n_valid": 0,
                        "HL_mean": float("nan"), "HR_mean": float("nan")})
            continue
        y_b = y[s:e]; hl_b = hl[s:e]; hr_b = hr[s:e]
        valid = (y_b == 0) & (hl_b > 0) & (hr_b > 0) & np.isfinite(hl_b) & np.isfinite(hr_b)
        n_v = int(valid.sum())
        if n_v == 0:
            out.append({"bin_idx": b, "ratio": float("nan"), "n_valid": 0,
                        "HL_mean": float("nan"), "HR_mean": float("nan")})
            continue
        mh = float(np.mean(hl_b[valid])); mr = float(np.mean(hr_b[valid]))
        out.append({"bin_idx": b,
                    "ratio": mh / mr if mr > 0 else float("nan"),
                    "n_valid": n_v, "HL_mean": mh, "HR_mean": mr})
    return out


def collect():
    import pandas as pd
    rows = []

    # 2605
    rx_l = re.compile(r"^2605_FormOxy_S(\d+)_formalin_Left_licking_predictions\.csv$")
    rx_c = re.compile(r"^2605_FormOxy_S(\d+)_formalin_contour_[0-9a-f]+\.csv$")
    ctr_lookup = {int(rx_c.match(p.name).group(1)): p
                  for p in NEW_CTR.glob("2605_FormOxy_S*_formalin_contour_*.csv")
                  if rx_c.match(p.name)}
    for p in sorted(NEW_LICK.glob("2605_FormOxy_S*_formalin_Left_licking_predictions.csv")):
        m = rx_l.match(p.name); subj = int(m.group(1))
        ctr = ctr_lookup.get(subj)
        if not ctr: continue
        for b in per_session_bins(p, ctr):
            rows.append({"cohort": "2605", "subject_id": f"2605_S{subj}",
                         "dose": dose_2605(subj), **b})

    # 2512
    rx_l = re.compile(r"^2512_FormOxy_S(\d+)_PixelPaws_Left_licking_predictions\.csv$")
    rx_c = re.compile(r"^2512_FormOxy_S(\d+)_contour_[0-9a-f]+\.csv$")
    ctr_lookup = {int(rx_c.match(p.name).group(1)): p
                  for p in OLD_CTR.glob("2512_FormOxy_S*_contour_*.csv")
                  if rx_c.match(p.name)}
    for p in sorted(OLD_LICK.glob("2512_FormOxy_S*_PixelPaws_Left_licking_predictions.csv")):
        m = rx_l.match(p.name); subj = int(m.group(1))
        ctr = ctr_lookup.get(subj)
        if not ctr: continue
        for b in per_session_bins(p, ctr):
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
        step(f"  statsmodels failed ({e})")
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


def plot_tc(df, out_png: Path, anova: dict, perbin_p: list[float]):
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
            vals = sub.loc[sub["bin_idx"] == b, "ratio"].dropna().to_numpy()
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
    ax.set_ylabel("Contour intensity ratio HL/HR (non-licking frames only)")
    ax.set_title("Contour Intensity Ratio TC -- NON-LICKING frames only",
                 fontweight="bold")
    ax.set_xticks(x_min)
    ax.set_xticklabels([str(i * BIN_MIN) for i in range(N_BINS)])
    ax.grid(alpha=0.3)
    ax.legend(loc="best", frameon=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_intensity_panels(df, out_png: Path):
    """Two extra panels: mean Pix_HL and mean Pix_HR per dose, non-licking frames."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_min = [i * BIN_MIN for i in range(N_BINS)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)

    for ax, col, title in [(axes[0], "HL_mean", "Pix_HL  (formalin-injected hind)"),
                           (axes[1], "HR_mean", "Pix_HR  (contralateral control)")]:
        for dose in DOSE_SEQ:
            sub = df[df["dose"] == dose]
            means, sems = [], []
            for b in range(N_BINS):
                vals = sub.loc[sub["bin_idx"] == b, col].dropna().to_numpy()
                means.append(np.mean(vals) if len(vals) else np.nan)
                sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
            c = DOSE_COLORS[dose]
            ax.errorbar(x_min, means, yerr=sems, marker="o", color=c,
                        linewidth=1.6, markersize=5.5, capsize=3,
                        markerfacecolor=c, markeredgecolor=c, label=dose)
        ax.set_xlabel("Time (min)")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x_min)
        ax.set_xticklabels([str(i * BIN_MIN) for i in range(N_BINS)])
        ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8)
    axes[0].set_ylabel("Mean inside-contour intensity (non-licking frames)")
    fig.suptitle("Per-paw contour intensity TC (non-licking frames)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main() -> int:
    import numpy as np
    import pandas as pd
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    df = collect()
    if df.empty:
        step("! no data collected"); return 1
    step(f"Bin rows: {len(df)}  across {df['subject_id'].nunique()} sessions")

    long_csv = ANALYSIS / "contour_ratio_tc_nolick_long.csv"
    df.to_csv(long_csv, index=False)
    step(f"Long-form: {long_csv}")
    step(f"ratio stats: min={df['ratio'].min():.3f}  median={df['ratio'].median():.3f}  "
         f"max={df['ratio'].max():.3f}")

    anova = two_way_anova(df, "ratio")
    step(f"ANOVA: {anova}")
    perbin_p = per_bin_kruskal(df, "ratio")
    step("Per-bin Kruskal-Wallis (4 doses):")
    for i, p in enumerate(perbin_p):
        step(f"  bin {i*BIN_MIN:>2}-{(i+1)*BIN_MIN:>2} min: p={p:.3g}  {sig_marker(p)}")

    tc_png = ANALYSIS / "contour_ratio_tc_nolick.png"
    plot_tc(df, tc_png, anova, perbin_p)
    step(f"Figure: {tc_png}")

    paw_png = ANALYSIS / "contour_intensity_per_paw_tc_nolick.png"
    plot_intensity_panels(df, paw_png)
    step(f"Per-paw figure: {paw_png}")

    # Compact summary CSV
    rows = []
    for dose in DOSE_SEQ:
        sub = df[df["dose"] == dose]
        for b in range(N_BINS):
            vals = sub.loc[sub["bin_idx"] == b, "ratio"].dropna().to_numpy()
            rows.append({
                "dose": dose, "bin_min_start": b * BIN_MIN, "n": int(len(vals)),
                "mean": round(float(np.mean(vals)), 4) if len(vals) else float("nan"),
                "sem":  round(float(np.std(vals, ddof=1) / np.sqrt(len(vals))), 4) if len(vals) > 1 else 0.0,
                "p_omnibus_at_bin": round(perbin_p[b], 4),
                "sig_at_bin": sig_marker(perbin_p[b]),
            })
    summary_csv = ANALYSIS / "contour_ratio_tc_nolick_summary.csv"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)

    head = (
        "**FormOxy contour HL/HR ratio TC -- NON-LICKING frames only**\n"
        "Per-bin reduction `mean(HL) / mean(HR)` on frames where "
        "`Left_licking == 0` AND both contours are valid. Removes the "
        "mechanical paw-lift / mouth-occlusion contribution to the ratio.\n"
        "Combined cohorts 2512 + 2605, n=6 per dose, mean +/- SEM.\n"
        f"Two-way ANOVA: Treatment p={anova['p_treatment']:.3g}{sig_marker(anova['p_treatment'])}, "
        f"Time p={anova['p_time']:.3g}{sig_marker(anova['p_time'])}, "
        f"Interaction p={anova['p_interaction']:.3g}{sig_marker(anova['p_interaction'])}.\n"
        f"Per-bin asterisks = Kruskal-Wallis across 4 doses.\n"
        "Second figure shows the underlying per-paw Pix_HL and Pix_HR "
        "timecourses (non-licking frames)."
    )
    discord_upload(head, [tc_png, paw_png, summary_csv])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
