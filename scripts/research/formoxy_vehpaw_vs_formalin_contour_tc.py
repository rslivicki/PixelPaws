"""Vehicle-paw vs Formalin-paw contour intensity time-course comparison.

Compares Otsu paw-contour brightness between vehicle-paw-injected and
formalin-paw-injected mice over 60 min post-injection, in 12 × 5-min bins.

Design:
  - VehPaw   : 6 subjects (2605_FormOxy_Veh_S1..S6, 3M + 3F).
               Paw was injected with VEHICLE (saline). Control cohort.
  - Formalin : 12 subjects (2605_FormOxy_S1..S12).
               Paw was injected with FORMALIN. IP injection was Veh/Oxy1/3/10,
               which doesn't matter for this brightness comparison.

  Both groups: HL = injected paw, HR = uninjected contralateral paw.

Plot: 3-panel TC (HL, HR, HL/HR ratio), 12 × 5-min bins over 60 min.
  - VehPaw   : grey (#666666)
  - Formalin : red  (#d62728)
  - Lines + circle markers + translucent SEM bands (matches the dose-response
    plots in the poster).

Stats: per-panel two-way ANOVA (Treatment × Bin) + per-bin Mann–Whitney U.

If CORRECT_TO_BASELINE = True, subtracts each subject's own pre-injection
baseline mean (from `*_baseline` recording) so the y-axis is Δ-from-baseline.
"""
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
from scipy import stats

COHORT = Path(r"E:\RSVIDS\Blackbox\2605_FormOxy_LeftPaws")
CONTOUR_DIR = COHORT / "gait_limb_analysis"
ANALYSIS = COHORT / "analysis"
KEY_CSV = COHORT / "2605_FormOxy_key.csv"

CONTOUR_HASH = "7d4120f9"

FPS = 60
BIN_MIN = 5
N_BINS = 12
BIN_FRAMES = BIN_MIN * 60 * FPS

GROUP_COLORS = {"VehPaw": "#666666", "Formalin": "#d62728"}
GROUP_ORDER = ["VehPaw", "Formalin"]

VEH_SUBJECTS = [f"2605_FormOxy_Veh_S{i}" for i in range(1, 7)]
# Only Veh-IP-injected formalin animals from the oxy cohort: S1, S5, S9.
# Isolates paw-injection contents (vehicle vs formalin) without oxy confound.
FORM_SUBJECTS = ["2605_FormOxy_S1", "2605_FormOxy_S5", "2605_FormOxy_S9"]
FORM_LABEL = "Formalin (IP Veh)"

CORRECT_TO_BASELINE = False

WEBHOOK = ("")


def step(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def discord_upload(content, files):
    boundary = "----PP" + uuid.uuid4().hex
    body = [f"--{boundary}".encode(),
            b'Content-Disposition: form-data; name="payload_json"',
            b'Content-Type: application/json', b"",
            json.dumps({"content": content}).encode()]
    for i, p in enumerate(files):
        mime = {".png": "image/png", ".pdf": "application/pdf",
                ".svg": "image/svg+xml", ".csv": "text/csv"}.get(
                    p.suffix.lower(), "application/octet-stream")
        body += [f"--{boundary}".encode(),
                 f'Content-Disposition: form-data; name="file{i}"; '
                 f'filename="{p.name}"'.encode(),
                 f"Content-Type: {mime}".encode(), b"",
                 p.read_bytes()]
    body.append(f"--{boundary}--".encode())
    req = urllib.request.Request(
        WEBHOOK, data=b"\r\n".join(body), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "User-Agent": "PixelPaws-VehPaw-vs-Formalin/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        step(f"  HTTP {r.status}")


def sig_marker(p):
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def session_csv(subject: str, cond: str) -> Path:
    """Path to the contour CSV for `<subject>_<cond>`."""
    stem = f"{subject}_{cond}"
    return CONTOUR_DIR / f"{stem}_contour_{CONTOUR_HASH}.csv"


def load_session(subject: str, cond: str) -> pd.DataFrame | None:
    """Load per-frame contour CSV and add HL/HR + ratio columns."""
    p = session_csv(subject, cond)
    if not p.is_file():
        step(f"  missing: {p.name}")
        return None
    df = pd.read_csv(p)
    needed = ("intensities_HL", "intensities_HR")
    if not all(c in df.columns for c in needed):
        step(f"  missing intensity cols in {p.name}")
        return None
    df = df.reset_index().rename(columns={"index": "frame"})
    df["HL"] = df["intensities_HL"]
    df["HR"] = df["intensities_HR"]
    return df[["frame", "HL", "HR"]]


def bin_session(df: pd.DataFrame) -> pd.DataFrame:
    """Cap to 60 min, mask invalid (zero or non-finite), per-bin means."""
    df = df[df["frame"] < N_BINS * BIN_FRAMES].copy()
    df["bin_idx"] = (df["frame"] // BIN_FRAMES).astype(int)
    valid = (df["HL"] > 0) & (df["HR"] > 0) & \
            np.isfinite(df["HL"]) & np.isfinite(df["HR"])
    df = df[valid].copy()
    g = df.groupby("bin_idx", as_index=False).agg(
        HL=("HL", "mean"),
        HR=("HR", "mean"),
    )
    g["ratio"] = g["HL"] / g["HR"]
    return g


def baseline_means(subject: str) -> dict | None:
    df = load_session(subject, "baseline")
    if df is None:
        return None
    binned = bin_session(df)
    return {"HL": binned["HL"].mean(),
            "HR": binned["HR"].mean(),
            "ratio": binned["ratio"].mean()}


def collect_long_df(correct_to_baseline: bool) -> pd.DataFrame:
    rows = []
    # VehPaw cohort: condition for the post-injection recording is "vehicle"
    for subj in VEH_SUBJECTS:
        post = load_session(subj, "vehicle")
        if post is None:
            continue
        binned = bin_session(post)
        if correct_to_baseline:
            base = baseline_means(subj)
            if base is None:
                step(f"  skip {subj}: no baseline")
                continue
            for col in ("HL", "HR", "ratio"):
                binned[col] = binned[col] - base[col]
        for _, row in binned.iterrows():
            rows.append({
                "subject":   subj,
                "treatment": "VehPaw",
                "bin_idx":   int(row["bin_idx"]),
                "HL":        row["HL"],
                "HR":        row["HR"],
                "ratio":     row["ratio"],
            })

    # Formalin cohort: condition for the post-injection recording is "formalin"
    for subj in FORM_SUBJECTS:
        post = load_session(subj, "formalin")
        if post is None:
            continue
        binned = bin_session(post)
        if correct_to_baseline:
            base = baseline_means(subj)
            if base is None:
                step(f"  skip {subj}: no baseline")
                continue
            for col in ("HL", "HR", "ratio"):
                binned[col] = binned[col] - base[col]
        for _, row in binned.iterrows():
            rows.append({
                "subject":   subj,
                "treatment": "Formalin",
                "bin_idx":   int(row["bin_idx"]),
                "HL":        row["HL"],
                "HR":        row["HR"],
                "ratio":     row["ratio"],
            })

    return pd.DataFrame(rows)


def per_bin_mw(df: pd.DataFrame, metric: str) -> list[float]:
    out = []
    for b in range(N_BINS):
        a = df.loc[(df["bin_idx"] == b) & (df["treatment"] == "VehPaw"),
                   metric].dropna().to_numpy()
        c = df.loc[(df["bin_idx"] == b) & (df["treatment"] == "Formalin"),
                   metric].dropna().to_numpy()
        if len(a) < 2 or len(c) < 2:
            out.append(float("nan"))
            continue
        try:
            _, p = stats.mannwhitneyu(a, c, alternative="two-sided")
            out.append(float(p))
        except Exception:
            out.append(float("nan"))
    return out


def twoway_anova(df: pd.DataFrame, metric: str) -> dict:
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        sub = df.dropna(subset=[metric]).copy()
        sub["bin_idx"] = sub["bin_idx"].astype("category")
        sub["treatment"] = pd.Categorical(
            sub["treatment"], categories=GROUP_ORDER, ordered=True)
        m = ols(f"{metric} ~ C(treatment) * C(bin_idx)", data=sub).fit()
        aov = sm.stats.anova_lm(m, typ=2)
        return {
            "p_treatment":   float(aov.loc["C(treatment)",            "PR(>F)"]),
            "p_time":        float(aov.loc["C(bin_idx)",              "PR(>F)"]),
            "p_interaction": float(aov.loc["C(treatment):C(bin_idx)", "PR(>F)"]),
        }
    except Exception as e:
        step(f"  ANOVA failed for {metric}: {e}")
        return {"p_treatment": float("nan"), "p_time": float("nan"),
                "p_interaction": float("nan")}


def plot_panel(ax, df, metric, ylabel, panel_title, add_unity_line: bool):
    x = np.array([i * BIN_MIN for i in range(N_BINS)], dtype=float)
    for grp in GROUP_ORDER:
        sub = df[df["treatment"] == grp]
        if len(sub) == 0:
            continue
        means, sems, ns = [], [], []
        for b in range(N_BINS):
            vals = sub.loc[sub["bin_idx"] == b, metric].dropna().to_numpy()
            means.append(np.mean(vals) if len(vals) else np.nan)
            sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals))
                        if len(vals) > 1 else 0.0)
            ns.append(len(vals))
        means = np.asarray(means, dtype=float)
        sems = np.asarray(sems, dtype=float)
        c = GROUP_COLORS[grp]
        n_max = max(ns) if ns else 0
        ax.plot(x, means, marker="o", color=c, linewidth=1.8, markersize=5.5,
                markerfacecolor=c, markeredgecolor=c,
                label=f"{grp} (n={n_max})")
        ax.fill_between(x, means - sems, means + sems,
                        color=c, alpha=0.18, linewidth=0)

    if add_unity_line:
        ax.axhline(1.0, color="#d62728", linestyle="--",
                   linewidth=1.2, alpha=0.6)

    anova = twoway_anova(df, metric)
    anova_txt = (
        f"Two-way ANOVA: Treatment p={anova['p_treatment']:.3g}"
        f"{sig_marker(anova['p_treatment'])}, "
        f"Time p={anova['p_time']:.3g}{sig_marker(anova['p_time'])}, "
        f"Interaction p={anova['p_interaction']:.3g}"
        f"{sig_marker(anova['p_interaction'])}"
    )
    ax.annotate(anova_txt, xy=(0.01, 0.985), xycoords="axes fraction",
                ha="left", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="lightgray"))

    pbins = per_bin_mw(df, metric)
    y_top = ax.get_ylim()[1]
    for i, p in enumerate(pbins):
        marker = sig_marker(p)
        if marker:
            ax.annotate(marker, xy=(x[i], y_top * 0.985),
                        ha="center", va="top", fontsize=10)

    ax.set_xlabel("Time post-injection (min)")
    ax.set_ylabel(ylabel)
    ax.set_title(panel_title, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i * BIN_MIN) for i in range(N_BINS)])
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=True, fontsize=9)


def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    df = collect_long_df(CORRECT_TO_BASELINE)
    if df.empty:
        step("no sessions loaded; aborting")
        return 1

    suffix = ("_ipveh" + ("_baseline_corrected" if CORRECT_TO_BASELINE else "")
              if len(FORM_SUBJECTS) < 12
              else ("_baseline_corrected" if CORRECT_TO_BASELINE else ""))
    long_csv = ANALYSIS / f"vehpaw_vs_formalin_contour_tc{suffix}_long.csv"
    df.to_csv(long_csv, index=False)
    step(f"wrote {long_csv.name} ({len(df)} rows)")
    n_veh = df.loc[df["treatment"] == "VehPaw", "subject"].nunique()
    n_form = df.loc[df["treatment"] == "Formalin", "subject"].nunique()
    step(f"  VehPaw subjects: {n_veh}/{len(VEH_SUBJECTS)}")
    step(f"  Formalin subjects: {n_form}/{len(FORM_SUBJECTS)}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    plot_panel(
        axes[0], df, "HL",
        ylabel="HL contour intensity (a.u.)"
               + (" Δ from baseline" if CORRECT_TO_BASELINE else ""),
        panel_title="HL (injected paw)",
        add_unity_line=False,
    )
    plot_panel(
        axes[1], df, "HR",
        ylabel="HR contour intensity (a.u.)"
               + (" Δ from baseline" if CORRECT_TO_BASELINE else ""),
        panel_title="HR (contralateral)",
        add_unity_line=False,
    )
    plot_panel(
        axes[2], df, "ratio",
        ylabel="HL/HR ratio"
               + (" Δ from baseline" if CORRECT_TO_BASELINE else ""),
        panel_title="HL/HR ratio",
        add_unity_line=not CORRECT_TO_BASELINE,
    )
    cohort_tag = f" — Formalin = IP-Veh only (S1/S5/S9)" if len(FORM_SUBJECTS) < 12 else ""
    fig.suptitle(
        f"Vehicle-paw vs Formalin-paw contour intensity{cohort_tag}"
        f"{' (baseline-corrected)' if CORRECT_TO_BASELINE else ''}",
        fontweight="bold", fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_base = ANALYSIS / f"vehpaw_vs_formalin_contour_tc{suffix}"
    fig.savefig(out_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    step(f"wrote {out_base}.png / svg / pdf")

    formalin_tag = (
        " (IP-Veh only — S1/S5/S9, isolates paw-injection contents w/o oxy confound)"
        if len(FORM_SUBJECTS) < 12 else ""
    )
    head = (
        "**Vehicle-paw vs Formalin-paw contour intensity TC (12×5 min)**"
        + formalin_tag
        + ("  — baseline-corrected (Δ post − baseline)"
           if CORRECT_TO_BASELINE else "")
        + f"\nVehPaw: {n_veh}/{len(VEH_SUBJECTS)} subjects (3M+3F). "
        + f"Formalin: {n_form}/{len(FORM_SUBJECTS)} subjects. "
        + "3-panel: HL (injected paw) / HR (contralateral) / HL-HR ratio. "
        + "Lines + translucent SEM. Two-way ANOVA + per-bin Mann–Whitney U asterisks."
    )
    discord_upload(head, [out_base.with_suffix(".png"),
                          out_base.with_suffix(".svg"),
                          out_base.with_suffix(".pdf"),
                          long_csv])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
