"""Dose-response contour intensity TC (Veh / Oxy1 / Oxy3 / Oxy10) PLUS the
vehicle-paw control cohort as a 5th trace.

Pools 2512 + 2605 formalin-paw cohorts (n=6/dose), reading
`*_contour_<hash>.csv` from each cohort's gait_limb_analysis folder.
Adds the 2605 vehicle-paw cohort (Veh_S1..Veh_S6, n=6) as a 5th
group named "VehPaw".

Output: 3-panel TC (HL, HR, HL/HR ratio), 12 × 5-min bins over 60 min,
matching the dose-response style in the poster (lines + circle markers
+ translucent SEM bands).

Stats per panel:
  - two-way ANOVA (Treatment × Bin) across the 4 oxy doses ONLY
    (matches the original dose-response stats — VehPaw is the
     visual control trace).
  - second two-way ANOVA (Treatment × Bin) across the 5 groups
    (Veh, Oxy1, Oxy3, Oxy10, VehPaw) for the cross-cohort comparison.
  - per-bin Kruskal–Wallis across the 5 groups, asterisks at the top.
"""
from __future__ import annotations
import json, re, time, urllib.request, uuid
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

COHORT_NEW = Path(r"E:\RSVIDS\Blackbox\2605_FormOxy_LeftPaws")
NEW_DIR = COHORT_NEW / "gait_limb_analysis"
ANALYSIS = COHORT_NEW / "analysis"

COHORT_OLD = Path(r"E:\RSVIDS\Blackbox\2512_Blackbox_Formalin_Oxy\Left_paws")
OLD_DIR = COHORT_OLD / "gait_limb_analysis"

FPS = 60
BIN_MIN = 5
N_BINS = 12
BIN_FRAMES = BIN_MIN * 60 * FPS

DOSE_2512 = {1: "Veh", 2: "Veh", 3: "Veh",
             4: "Oxy3", 5: "Oxy3", 6: "Oxy3",
             7: "Oxy10", 8: "Oxy10", 9: "Oxy10",
             10: "Oxy1", 11: "Oxy1", 12: "Oxy1"}
DOSE_SEQ = ["Veh", "Oxy1", "Oxy3", "Oxy10"]
ALL_GROUPS = DOSE_SEQ + ["VehPaw"]

GROUP_COLORS = {
    "Veh":    "#000000",
    "Oxy1":   "#a6cee3",
    "Oxy3":   "#1f78b4",
    "Oxy10":  "#08306b",
    "VehPaw": "#7f7f7f",
}


def dose_2605(n: int) -> str:
    return DOSE_SEQ[(n - 1) % 4]


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
                 "User-Agent": "PixelPaws-DoseResponse-w-VehPaw/1.0"})
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


def bin_session(df: pd.DataFrame) -> list[dict]:
    """Per-bin mean(HL), mean(HR), ratio."""
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
            "HL": mean_hl,
            "HR": mean_hr,
            "ratio": ratio,
        })
    return out


def collect() -> pd.DataFrame:
    rows = []

    # 2605 formalin (12 subjects)
    rx_new_form = re.compile(r"^2605_FormOxy_S(\d+)_formalin_contour_[0-9a-f]+\.csv$")
    for p in sorted(NEW_DIR.glob("2605_FormOxy_S*_formalin_contour_*.csv")):
        m = rx_new_form.match(p.name)
        if not m:
            continue
        subj = int(m.group(1))
        df = pd.read_csv(p, usecols=["intensities_HL", "intensities_HR"])
        for b in bin_session(df):
            rows.append({"cohort": "2605", "subject_id": f"2605_S{subj}",
                         "treatment": dose_2605(subj), **b})

    # 2512 formalin (12 subjects)
    rx_old = re.compile(r"^2512_FormOxy_S(\d+)_contour_[0-9a-f]+\.csv$")
    for p in sorted(OLD_DIR.glob("2512_FormOxy_S*_contour_*.csv")):
        m = rx_old.match(p.name)
        if not m:
            continue
        subj = int(m.group(1))
        df = pd.read_csv(p, usecols=["intensities_HL", "intensities_HR"])
        for b in bin_session(df):
            rows.append({"cohort": "2512", "subject_id": f"2512_S{subj}",
                         "treatment": DOSE_2512[subj], **b})

    # 2605 vehpaw (6 subjects, post-vehicle recording)
    rx_vehpaw = re.compile(r"^2605_FormOxy_Veh_S(\d+)_vehicle_contour_[0-9a-f]+\.csv$")
    for p in sorted(NEW_DIR.glob("2605_FormOxy_Veh_S*_vehicle_contour_*.csv")):
        m = rx_vehpaw.match(p.name)
        if not m:
            continue
        subj = int(m.group(1))
        df = pd.read_csv(p, usecols=["intensities_HL", "intensities_HR"])
        for b in bin_session(df):
            rows.append({"cohort": "vehpaw", "subject_id": f"VehPaw_S{subj}",
                         "treatment": "VehPaw", **b})

    return pd.DataFrame(rows)


def per_bin_kruskal(df: pd.DataFrame, metric: str, groups: list[str]) -> list[float]:
    out = []
    for b in range(N_BINS):
        gs = []
        for grp in groups:
            v = df.loc[(df["bin_idx"] == b) & (df["treatment"] == grp),
                       metric].dropna().to_numpy()
            if len(v) > 0:
                gs.append(v)
        gs = [g for g in gs if len(g) > 1]
        if len(gs) < 2:
            out.append(float("nan"))
            continue
        try:
            _, p = stats.kruskal(*gs)
            out.append(float(p))
        except Exception:
            out.append(float("nan"))
    return out


def two_way_anova(df: pd.DataFrame, metric: str, groups: list[str]) -> dict:
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        sub = df[df["treatment"].isin(groups)].dropna(subset=[metric]).copy()
        sub["bin_idx"] = sub["bin_idx"].astype("category")
        sub["treatment"] = pd.Categorical(sub["treatment"], categories=groups, ordered=True)
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
    for grp in ALL_GROUPS:
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
                        color=c, alpha=0.16, linewidth=0)

    if add_unity_line:
        ax.axhline(1.0, color="#d62728", linestyle="--",
                   linewidth=1.2, alpha=0.6)

    anova_dose = two_way_anova(df, metric, DOSE_SEQ)
    anova_5 = two_way_anova(df, metric, ALL_GROUPS)
    anova_txt = (
        f"4-dose ANOVA: Treatment p={anova_dose['p_treatment']:.3g}"
        f"{sig_marker(anova_dose['p_treatment'])}\n"
        f"5-group ANOVA (+VehPaw): Treatment p={anova_5['p_treatment']:.3g}"
        f"{sig_marker(anova_5['p_treatment'])}"
    )
    ax.annotate(anova_txt, xy=(0.01, 0.985), xycoords="axes fraction",
                ha="left", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="lightgray"))

    pbins = per_bin_kruskal(df, metric, ALL_GROUPS)
    y_top = ax.get_ylim()[1]
    for i, p in enumerate(pbins):
        m = sig_marker(p)
        if m:
            ax.annotate(m, xy=(x[i], y_top * 0.985),
                        ha="center", va="top", fontsize=9)

    ax.set_xlabel("Time post-injection (min)")
    ax.set_ylabel(ylabel)
    ax.set_title(panel_title, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i * BIN_MIN) for i in range(N_BINS)])
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=True, fontsize=8)


def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    df = collect()
    if df.empty:
        step("no sessions loaded; aborting")
        return 1
    out_csv = ANALYSIS / "contour_dose_response_with_vehpaw_long.csv"
    df.to_csv(out_csv, index=False)
    n_per = (df.drop_duplicates(["subject_id"])
               .groupby("treatment").size().to_dict())
    step(f"  per-group n: {n_per}")
    step(f"  wrote {out_csv.name} ({len(df)} rows)")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    plot_panel(axes[0], df, "HL",
               ylabel="HL contour intensity (a.u.)",
               panel_title="HL (injected paw)",
               add_unity_line=False)
    plot_panel(axes[1], df, "HR",
               ylabel="HR contour intensity (a.u.)",
               panel_title="HR (contralateral)",
               add_unity_line=False)
    plot_panel(axes[2], df, "ratio",
               ylabel="HL/HR ratio",
               panel_title="HL/HR ratio",
               add_unity_line=True)
    fig.suptitle(
        "Paw contour intensity — oxy dose-response + vehicle-paw control",
        fontweight="bold", fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_base = ANALYSIS / "contour_dose_response_with_vehpaw"
    fig.savefig(out_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    step(f"  wrote {out_base}.png / svg / pdf")

    head = (
        "**Paw contour intensity dose-response + vehicle-paw control**\n"
        + "Cohorts: 2512 + 2605 formalin (n=6/dose: Veh/Oxy1/Oxy3/Oxy10) + "
        + "2605 vehicle-paw (n=6, grey trace). "
        + "3 panels: HL (injected paw) / HR (contralateral) / HL-HR ratio. "
        + "Lines + translucent SEM. Box: 4-dose ANOVA (existing analysis) and "
        + "5-group ANOVA including VehPaw. Per-bin Kruskal–Wallis asterisks."
    )
    discord_upload(head, [out_base.with_suffix(".png"),
                          out_base.with_suffix(".svg"),
                          out_base.with_suffix(".pdf"),
                          out_csv])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
