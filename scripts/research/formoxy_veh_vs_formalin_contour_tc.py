"""Vehicle vs Formalin contour intensity time-course comparison.

Compares Otsu paw-contour brightness between vehicle-injected and
formalin-injected paws over 60 min post-injection, in 12 × 5-min bins.

Design:
  - 6 vehicle subjects (2605_FormOxy_Veh_S1..S6), each with `_baseline`
    + `_vehicle` recordings.
  - Existing formalin subjects (2605_FormOxy_S1..S12), each with
    `_baseline` + `_formalin` recordings.
  - Post-injection trace plotted: 3 panels (HL, HR, HL/HR ratio), each
    Vehicle vs Formalin.
  - Optional baseline-corrected ("post - baseline") variant for the
    same 3 panels — uncomment the CORRECT_TO_BASELINE block.

Plot style (matches the dose-response TCs in the poster):
  - Solid line + circle markers (markersize 5.5, linewidth 1.8)
  - Translucent SEM bands via `fill_between(±SEM, alpha=0.18)`
  - No error bar caps, no grid, top/right spines hidden
  - HL/HR panel: red dashed reference at ratio = 1.0

Stats (annotated on each panel):
  - Two-way ANOVA: Treatment × Bin
  - Per-bin Mann–Whitney U (2 groups), with asterisks at the top
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

PROJECT = Path(r"E:\RSVIDS\Blackbox\2605_FormOxy_LeftPaws")
CONTOUR_DIR = PROJECT / "gait_limb_analysis"
ANALYSIS = PROJECT / "analysis"
KEY_CSV = PROJECT / "2605_FormOxy_key.csv"

FPS = 60
BIN_MIN = 5
N_BINS = 12
BIN_FRAMES = BIN_MIN * 60 * FPS

GROUP_COLORS = {"Vehicle": "#666666", "Formalin": "#d62728"}
GROUP_ORDER = ["Vehicle", "Formalin"]

VEH_PATTERN = "2605_FormOxy_Veh_S{n}_vehicle"
FORM_PATTERN = "2605_FormOxy_S{n}_formalin"
VEH_SUBJECTS = list(range(1, 7))
FORM_SUBJECTS = list(range(1, 13))

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
                 "User-Agent": "PixelPaws-VehVsFormalin/1.0"})
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


def load_contour_session(stem: str) -> pd.DataFrame | None:
    """Load HL + HR contour CSVs for one session stem and return per-frame
    dataframe with columns: frame, intensities_HL, intensities_HR.
    Returns None if either contour file is missing."""
    hl = CONTOUR_DIR / f"{stem}_contour_HL.csv"
    hr = CONTOUR_DIR / f"{stem}_contour_HR.csv"
    if not hl.exists() or not hr.exists():
        step(f"  missing contour CSV for {stem}: HL={hl.exists()} HR={hr.exists()}")
        return None
    df_hl = pd.read_csv(hl).rename(columns={"intensity": "intensities_HL"})
    df_hr = pd.read_csv(hr).rename(columns={"intensity": "intensities_HR"})
    merged = df_hl[["frame", "intensities_HL"]].merge(
        df_hr[["frame", "intensities_HR"]], on="frame", how="inner")
    merged = merged.replace([np.inf, -np.inf], np.nan)
    return merged


def bin_session(df: pd.DataFrame) -> pd.DataFrame:
    """Cap to first 60 min and reduce to per-bin means of HL, HR, ratio."""
    df = df[df["frame"] < N_BINS * BIN_FRAMES].copy()
    df["bin_idx"] = (df["frame"] // BIN_FRAMES).astype(int)

    valid = (df["intensities_HL"] > 0) & (df["intensities_HR"] > 0) & \
            np.isfinite(df["intensities_HL"]) & np.isfinite(df["intensities_HR"])
    df = df[valid].copy()

    g = df.groupby("bin_idx", as_index=False).agg(
        HL=("intensities_HL", "mean"),
        HR=("intensities_HR", "mean"),
    )
    g["ratio"] = g["HL"] / g["HR"]
    return g


def collect_long_df(correct_to_baseline: bool) -> pd.DataFrame:
    rows = []
    for n in VEH_SUBJECTS:
        stem_post = VEH_PATTERN.format(n=n)
        df = load_contour_session(stem_post)
        if df is None:
            continue
        binned_post = bin_session(df)
        if correct_to_baseline:
            stem_base = stem_post.replace("_vehicle", "_baseline")
            df_b = load_contour_session(stem_base)
            if df_b is None:
                step(f"  skipping {stem_post}: baseline missing")
                continue
            binned_base = bin_session(df_b)
            base_mean = {"HL": binned_base["HL"].mean(),
                         "HR": binned_base["HR"].mean(),
                         "ratio": binned_base["ratio"].mean()}
            for col in ("HL", "HR", "ratio"):
                binned_post[col] = binned_post[col] - base_mean[col]
        for _, row in binned_post.iterrows():
            rows.append({
                "subject": stem_post,
                "treatment": "Vehicle",
                "bin_idx": int(row["bin_idx"]),
                "HL": row["HL"],
                "HR": row["HR"],
                "ratio": row["ratio"],
            })

    for n in FORM_SUBJECTS:
        stem_post = FORM_PATTERN.format(n=n)
        df = load_contour_session(stem_post)
        if df is None:
            continue
        binned_post = bin_session(df)
        if correct_to_baseline:
            stem_base = stem_post.replace("_formalin", "_baseline")
            df_b = load_contour_session(stem_base)
            if df_b is None:
                step(f"  skipping {stem_post}: baseline missing")
                continue
            binned_base = bin_session(df_b)
            base_mean = {"HL": binned_base["HL"].mean(),
                         "HR": binned_base["HR"].mean(),
                         "ratio": binned_base["ratio"].mean()}
            for col in ("HL", "HR", "ratio"):
                binned_post[col] = binned_post[col] - base_mean[col]
        for _, row in binned_post.iterrows():
            rows.append({
                "subject": stem_post,
                "treatment": "Formalin",
                "bin_idx": int(row["bin_idx"]),
                "HL": row["HL"],
                "HR": row["HR"],
                "ratio": row["ratio"],
            })

    return pd.DataFrame(rows)


def per_bin_mw(df: pd.DataFrame, metric: str) -> list[float]:
    out = []
    for b in range(N_BINS):
        veh = df.loc[(df["bin_idx"] == b) & (df["treatment"] == "Vehicle"),
                     metric].dropna().to_numpy()
        form = df.loc[(df["bin_idx"] == b) & (df["treatment"] == "Formalin"),
                      metric].dropna().to_numpy()
        if len(veh) < 2 or len(form) < 2:
            out.append(float("nan"))
            continue
        try:
            _, p = stats.mannwhitneyu(veh, form, alternative="two-sided")
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
        sub["treatment"] = pd.Categorical(sub["treatment"],
                                          categories=GROUP_ORDER, ordered=True)
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
    n_by_grp: dict[str, int] = {}
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
        n_by_grp[grp] = int(max(ns)) if ns else 0
        c = GROUP_COLORS[grp]
        ax.plot(x, means, marker="o", color=c, linewidth=1.8, markersize=5.5,
                markerfacecolor=c, markeredgecolor=c,
                label=f"{grp} (n={n_by_grp[grp]})")
        ax.fill_between(x, means - sems, means + sems,
                        color=c, alpha=0.18, linewidth=0)

    if add_unity_line:
        ax.axhline(1.0, color="#d62728", linestyle="--",
                   linewidth=1.2, alpha=0.7)

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
        m = sig_marker(p)
        if m:
            ax.annotate(m, xy=(x[i], y_top * 0.985),
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

    suffix = "_baseline_corrected" if CORRECT_TO_BASELINE else ""
    long_csv = ANALYSIS / f"veh_vs_formalin_contour_tc{suffix}_long.csv"
    df.to_csv(long_csv, index=False)
    step(f"wrote {long_csv.name} ({len(df)} rows)")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    plot_panel(
        axes[0], df, "HL",
        ylabel="HL contour intensity (a.u.)"
               + (" Δ from baseline" if CORRECT_TO_BASELINE else ""),
        panel_title="HL (injected side)",
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
    fig.suptitle(
        f"Vehicle vs Formalin paw contour intensity"
        f"{' (baseline-corrected)' if CORRECT_TO_BASELINE else ''}",
        fontweight="bold", fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_base = ANALYSIS / f"veh_vs_formalin_contour_tc{suffix}"
    fig.savefig(out_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    step(f"wrote {out_base}.png / svg / pdf")

    head = (
        "**FormOxy paw contour intensity: Vehicle vs Formalin (TC, 12×5 min)**"
        + ("  — baseline-corrected (Δ post − baseline)" if CORRECT_TO_BASELINE else "")
        + f"\nVehicle subjects loaded: {df.loc[df['treatment']=='Vehicle','subject'].nunique()} / {len(VEH_SUBJECTS)}"
        + f"; Formalin subjects loaded: {df.loc[df['treatment']=='Formalin','subject'].nunique()} / {len(FORM_SUBJECTS)}"
        + ". 3-panel: HL, HR, HL/HR ratio. Lines + translucent SEM. Two-way ANOVA box + per-bin Mann–Whitney U asterisks."
    )
    discord_upload(head, [out_base.with_suffix(".png"),
                          out_base.with_suffix(".svg"),
                          out_base.with_suffix(".pdf"),
                          long_csv])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
