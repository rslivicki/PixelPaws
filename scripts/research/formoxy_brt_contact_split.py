"""Paw BRIGHTNESS (fixed-ROI) contact-vs-non-contact split — proper
contact-state classification using the brt CSVs.

Why brt instead of contour for this analysis:
  The contour CSV only has data where a paw contour was found in the
  Otsu-thresholded ROI. When the paw is fully lifted off the IR floor,
  no contour is detected → row stays at zero → those frames get
  excluded. That makes "non-contact" in the contour view actually mean
  "small contour", not "paw off floor".

  The brt CSV is different: it computes mean(pixel intensity) inside a
  fixed 20-px box around the DLC paw keypoint, every frame, regardless
  of whether anything was detected. When the paw is on the IR floor →
  box is bright. When the paw lifts → box fills with dark floor pixels
  → brightness collapses.

  So brt is a real contact-state signal.

Method:
  Per session, per paw:
    - Compute the per-session median of Pix_*paw across all 60 min.
    - "in contact" = Pix_*paw > MEDIAN * CONTACT_FRAC
    - "non-contact" = Pix_*paw <= MEDIAN * CONTACT_FRAC

  For the HL/HR ratio, use frame-wise joint condition (both paws in
  contact, or both not in contact).

Output: 2 × 3 figure — top row contact-only, bottom row non-contact.
Same 5 traces (Veh, Oxy1, Oxy3, Oxy10, VehPaw, n=6 each).
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

# Threshold for "in contact": brt > CONTACT_FRAC × per-session median brt.
# The median is roughly the average brightness across the recording —
# when the paw is on the floor it's well above median, when lifted it
# falls below.
CONTACT_FRAC = 0.7

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
                 "User-Agent": "PixelPaws-BrtContactSplit/1.0"})
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


def bin_session_brt(df: pd.DataFrame) -> list[dict]:
    """Per-bin contact / non-contact aggregates using brt as both
    the contact classifier and the brightness metric."""
    hl = df["Pix_hlpaw"].astype(float).to_numpy()
    hr = df["Pix_hrpaw"].astype(float).to_numpy()

    # Per-session median brightness as contact reference
    median_hl = float(np.median(hl[hl > 0])) if (hl > 0).any() else 1.0
    median_hr = float(np.median(hr[hr > 0])) if (hr > 0).any() else 1.0
    th_hl = CONTACT_FRAC * median_hl
    th_hr = CONTACT_FRAC * median_hr

    valid_hl = (hl > 0) & np.isfinite(hl)
    valid_hr = (hr > 0) & np.isfinite(hr)
    contact_hl = valid_hl & (hl > th_hl)
    contact_hr = valid_hr & (hr > th_hr)
    noncon_hl  = valid_hl & ~contact_hl
    noncon_hr  = valid_hr & ~contact_hr

    out = []
    for i in range(N_BINS):
        s = i * BIN_FRAMES
        e = s + BIN_FRAMES

        c_hl = contact_hl[s:e]; c_hr = contact_hr[s:e]
        n_hl = noncon_hl[s:e];  n_hr = noncon_hr[s:e]
        hlb = hl[s:e]; hrb = hr[s:e]

        hl_c = float(np.mean(hlb[c_hl])) if c_hl.sum() else float("nan")
        hr_c = float(np.mean(hrb[c_hr])) if c_hr.sum() else float("nan")
        both_c = c_hl & c_hr
        r_c = float(np.mean(hlb[both_c] / hrb[both_c])) if both_c.sum() else float("nan")

        hl_n = float(np.mean(hlb[n_hl])) if n_hl.sum() else float("nan")
        hr_n = float(np.mean(hrb[n_hr])) if n_hr.sum() else float("nan")
        both_n = n_hl & n_hr
        r_n = float(np.mean(hlb[both_n] / hrb[both_n])) if both_n.sum() else float("nan")

        out.append({
            "bin_idx": i,
            "bin_min_start": i * BIN_MIN,
            "n_contact_hl": int(c_hl.sum()),
            "n_contact_hr": int(c_hr.sum()),
            "n_noncon_hl":  int(n_hl.sum()),
            "n_noncon_hr":  int(n_hr.sum()),
            "HL_contact": hl_c,
            "HR_contact": hr_c,
            "ratio_contact": r_c,
            "HL_noncon": hl_n,
            "HR_noncon": hr_n,
            "ratio_noncon": r_n,
            "median_hl": median_hl,
            "median_hr": median_hr,
        })
    return out


def collect() -> pd.DataFrame:
    rows = []
    cols = ["Pix_hlpaw", "Pix_hrpaw"]

    # 2605 formalin
    rx = re.compile(r"^2605_FormOxy_S(\d+)_formalin_brt_[0-9a-f]+\.csv$")
    for p in sorted(NEW_DIR.glob("2605_FormOxy_S*_formalin_brt_*.csv")):
        m = rx.match(p.name)
        if not m: continue
        subj = int(m.group(1))
        df = pd.read_csv(p, usecols=cols)
        for b in bin_session_brt(df):
            rows.append({"cohort": "2605", "subject_id": f"2605_S{subj}",
                         "treatment": dose_2605(subj), **b})

    # 2512 formalin
    rx = re.compile(r"^2512_FormOxy_S(\d+)_brt_[0-9a-f]+\.csv$")
    for p in sorted(OLD_DIR.glob("2512_FormOxy_S*_brt_*.csv")):
        m = rx.match(p.name)
        if not m: continue
        subj = int(m.group(1))
        df = pd.read_csv(p, usecols=cols)
        for b in bin_session_brt(df):
            rows.append({"cohort": "2512", "subject_id": f"2512_S{subj}",
                         "treatment": DOSE_2512[subj], **b})

    # 2605 vehpaw
    rx = re.compile(r"^2605_FormOxy_Veh_S(\d+)_vehicle_brt_[0-9a-f]+\.csv$")
    for p in sorted(NEW_DIR.glob("2605_FormOxy_Veh_S*_vehicle_brt_*.csv")):
        m = rx.match(p.name)
        if not m: continue
        subj = int(m.group(1))
        df = pd.read_csv(p, usecols=cols)
        for b in bin_session_brt(df):
            rows.append({"cohort": "vehpaw", "subject_id": f"VehPaw_S{subj}",
                         "treatment": "VehPaw", **b})

    return pd.DataFrame(rows)


def per_bin_kruskal(df, metric, groups):
    out = []
    for b in range(N_BINS):
        gs = [df.loc[(df["bin_idx"] == b) & (df["treatment"] == g), metric]
                .dropna().to_numpy() for g in groups]
        gs = [g for g in gs if len(g) > 1]
        if len(gs) < 2:
            out.append(float("nan")); continue
        try:
            _, p = stats.kruskal(*gs); out.append(float(p))
        except Exception:
            out.append(float("nan"))
    return out


def two_way_anova(df, metric, groups):
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
    except Exception:
        return {"p_treatment": float("nan"), "p_time": float("nan"),
                "p_interaction": float("nan")}


def plot_panel(ax, df, metric, ylabel, panel_title, add_unity_line,
               show_legend):
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
        ax.plot(x, means, marker="o", color=c, linewidth=1.7, markersize=4.5,
                markerfacecolor=c, markeredgecolor=c,
                label=f"{grp} (n={n_max})")
        ax.fill_between(x, means - sems, means + sems,
                        color=c, alpha=0.14, linewidth=0)

    if add_unity_line:
        ax.axhline(1.0, color="#d62728", linestyle="--",
                   linewidth=1.0, alpha=0.6)

    anova_dose = two_way_anova(df, metric, DOSE_SEQ)
    anova_5 = two_way_anova(df, metric, ALL_GROUPS)
    anova_txt = (
        f"4-dose ANOVA: p={anova_dose['p_treatment']:.2g}"
        f"{sig_marker(anova_dose['p_treatment'])}\n"
        f"5-group: p={anova_5['p_treatment']:.2g}"
        f"{sig_marker(anova_5['p_treatment'])}"
    )
    ax.annotate(anova_txt, xy=(0.01, 0.985), xycoords="axes fraction",
                ha="left", va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="lightgray"))

    pbins = per_bin_kruskal(df, metric, ALL_GROUPS)
    y_top = ax.get_ylim()[1]
    for i, p in enumerate(pbins):
        m = sig_marker(p)
        if m:
            ax.annotate(m, xy=(x[i], y_top * 0.985),
                        ha="center", va="top", fontsize=8)

    ax.set_xlabel("Time post-injection (min)")
    ax.set_ylabel(ylabel)
    ax.set_title(panel_title, fontweight="bold", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i * BIN_MIN) for i in range(N_BINS)], fontsize=8)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_legend:
        ax.legend(loc="best", frameon=True, fontsize=7)


def prep_subdf(df, mode):
    if mode == "contact":
        return df.rename(columns={"HL_contact": "HL",
                                  "HR_contact": "HR",
                                  "ratio_contact": "ratio"})[
            ["cohort", "subject_id", "treatment", "bin_idx",
             "HL", "HR", "ratio"]]
    elif mode == "noncon":
        return df.rename(columns={"HL_noncon": "HL",
                                  "HR_noncon": "HR",
                                  "ratio_noncon": "ratio"})[
            ["cohort", "subject_id", "treatment", "bin_idx",
             "HL", "HR", "ratio"]]
    raise ValueError(mode)


def main():
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    df = collect()
    if df.empty:
        step("no sessions loaded; aborting")
        return 1
    n_per = (df.drop_duplicates(["subject_id"])
               .groupby("treatment").size().to_dict())
    step(f"  per-group n: {n_per}")
    out_csv = ANALYSIS / "brt_contact_split_long.csv"
    df.to_csv(out_csv, index=False)
    step(f"  wrote {out_csv.name} ({len(df)} rows)")

    df_c = prep_subdf(df, "contact")
    df_n = prep_subdf(df, "noncon")
    g_c = (df_c.groupby(["treatment","bin_idx"])["ratio"].mean()
              .groupby("treatment").mean()).round(3)
    g_n = (df_n.groupby(["treatment","bin_idx"])["ratio"].mean()
              .groupby("treatment").mean()).round(3)
    step(f"  mean ratio (CONTACT)     : {g_c.to_dict()}")
    step(f"  mean ratio (NON-CONTACT) : {g_n.to_dict()}")

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    plot_panel(axes[0, 0], df_c, "HL",
               ylabel="Pix_hlpaw (a.u.)",
               panel_title="HL (injected paw) — CONTACT frames",
               add_unity_line=False, show_legend=True)
    plot_panel(axes[0, 1], df_c, "HR",
               ylabel="Pix_hrpaw (a.u.)",
               panel_title="HR (contralateral) — CONTACT frames",
               add_unity_line=False, show_legend=False)
    plot_panel(axes[0, 2], df_c, "ratio",
               ylabel="HL/HR ratio",
               panel_title="HL/HR ratio — CONTACT frames",
               add_unity_line=True, show_legend=False)

    plot_panel(axes[1, 0], df_n, "HL",
               ylabel="Pix_hlpaw (a.u.)",
               panel_title="HL (injected paw) — NON-CONTACT frames",
               add_unity_line=False, show_legend=True)
    plot_panel(axes[1, 1], df_n, "HR",
               ylabel="Pix_hrpaw (a.u.)",
               panel_title="HR (contralateral) — NON-CONTACT frames",
               add_unity_line=False, show_legend=False)
    plot_panel(axes[1, 2], df_n, "ratio",
               ylabel="HL/HR ratio",
               panel_title="HL/HR ratio — NON-CONTACT frames",
               add_unity_line=True, show_legend=False)

    fig.suptitle(
        f"Paw fixed-ROI brightness (brt) — contact vs non-contact frames "
        f"(threshold = {CONTACT_FRAC:.0%} of per-session median)",
        fontweight="bold", fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_base = ANALYSIS / "brt_contact_split"
    fig.savefig(out_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    step(f"  wrote {out_base}.png / svg / pdf")

    head = (
        "**Paw brightness (brt) — contact vs non-contact, side-by-side**\n"
        "Top row: paw IS in contact with the IR floor "
        f"(brt > {CONTACT_FRAC:.0%} of per-session median).\n"
        "Bottom row: paw NOT in contact (brt below threshold — paw lifted).\n"
        "**This is the proper contact split** — brt uses a fixed ROI that "
        "measures the box around the paw keypoint every frame, regardless of "
        "whether a contour was found. When the paw lifts off the floor, brt "
        "collapses (box fills with dark floor); the contour CSV by contrast "
        "drops those frames entirely.\n"
        "If the formalin-vs-VehPaw asymmetry persists in the CONTACT row, "
        "it's tissue (or weight-bearing without lifting). If only NON-CONTACT "
        "shows the gap, it's behavioural lifting."
    )
    discord_upload(head, [out_base.with_suffix(".png"),
                          out_base.with_suffix(".svg"),
                          out_base.with_suffix(".pdf"),
                          out_csv])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
