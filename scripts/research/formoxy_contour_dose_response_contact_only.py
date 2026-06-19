"""Contact-only contour intensity TC: filters frames to only those where
the hind paw is in contact with the floor (proxied by contour area being
near the per-session typical "open paw" area).

Tests whether the formalin-vs-VehPaw HL/HR asymmetry is driven by
behaviour (guarding → less floor contact → darker pixels) vs by
tissue-level changes (formalin → swelling/blood pooling → darker
pixels even on contact).

Filter rule, per session per paw:
  "in contact" = (area > AREA_FRAC × per-session 75th percentile of area)
                  AND (intensity > 0)

Per bin: aggregate frames where the relevant paw was in contact.
For the ratio panel, both HL and HR must be in contact in the same
frame (per-frame ratio averaged in the bin, with valid-mask).

Compares against the standard (all-valid-frames) script — if the
asymmetry persists in this contact-restricted view, formalin is causing
real tissue darkening, not just behavioral guarding.

5 traces: Veh, Oxy1, Oxy3, Oxy10, VehPaw (same palette as the standard
dose-response figure).
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

AREA_FRAC = 0.7   # frame counts as "in contact" if area > AREA_FRAC * P75
AREA_PCTL = 75    # the reference percentile for "open paw" area

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
                 "User-Agent": "PixelPaws-ContactOnly/1.0"})
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


def session_contact_thresholds(df: pd.DataFrame) -> tuple[float, float]:
    """Return (hl_thresh, hr_thresh) = AREA_FRAC × P75 of each paw's area."""
    areas_hl = df["areas_HL"].astype(float).to_numpy()
    areas_hr = df["areas_HR"].astype(float).to_numpy()
    nz_hl = areas_hl[areas_hl > 0]
    nz_hr = areas_hr[areas_hr > 0]
    p75_hl = float(np.percentile(nz_hl, AREA_PCTL)) if len(nz_hl) else 0.0
    p75_hr = float(np.percentile(nz_hr, AREA_PCTL)) if len(nz_hr) else 0.0
    return AREA_FRAC * p75_hl, AREA_FRAC * p75_hr


def bin_session(df: pd.DataFrame) -> list[dict]:
    """Per-bin mean of HL / HR / ratio, ONLY on frames where the
    corresponding paw is in contact."""
    th_hl, th_hr = session_contact_thresholds(df)
    out = []
    hl_a = df["areas_HL"].astype(float).to_numpy()
    hr_a = df["areas_HR"].astype(float).to_numpy()
    hl_i = df["intensities_HL"].astype(float).to_numpy()
    hr_i = df["intensities_HR"].astype(float).to_numpy()

    contact_hl_full = (hl_a > th_hl) & (hl_i > 0) & np.isfinite(hl_i)
    contact_hr_full = (hr_a > th_hr) & (hr_i > 0) & np.isfinite(hr_i)

    for i in range(N_BINS):
        s = i * BIN_FRAMES
        e = s + BIN_FRAMES
        ch_hl = contact_hl_full[s:e]
        ch_hr = contact_hr_full[s:e]
        hl_bin = hl_i[s:e]
        hr_bin = hr_i[s:e]

        mean_hl = float(np.mean(hl_bin[ch_hl])) if ch_hl.sum() else float("nan")
        mean_hr = float(np.mean(hr_bin[ch_hr])) if ch_hr.sum() else float("nan")
        both = ch_hl & ch_hr
        if both.sum():
            ratio = float(np.mean(hl_bin[both] / hr_bin[both]))
        else:
            ratio = float("nan")
        out.append({
            "bin_idx": i,
            "bin_min_start": i * BIN_MIN,
            "n_contact_hl_frames": int(ch_hl.sum()),
            "n_contact_hr_frames": int(ch_hr.sum()),
            "n_both_frames":       int(both.sum()),
            "HL":    mean_hl,
            "HR":    mean_hr,
            "ratio": ratio,
        })
    return out


def collect() -> pd.DataFrame:
    rows = []
    cols = ["areas_HL", "areas_HR", "intensities_HL", "intensities_HR"]

    # 2605 formalin
    rx = re.compile(r"^2605_FormOxy_S(\d+)_formalin_contour_[0-9a-f]+\.csv$")
    for p in sorted(NEW_DIR.glob("2605_FormOxy_S*_formalin_contour_*.csv")):
        m = rx.match(p.name)
        if not m:
            continue
        subj = int(m.group(1))
        df = pd.read_csv(p, usecols=cols)
        for b in bin_session(df):
            rows.append({"cohort": "2605", "subject_id": f"2605_S{subj}",
                         "treatment": dose_2605(subj), **b})

    # 2512 formalin
    rx = re.compile(r"^2512_FormOxy_S(\d+)_contour_[0-9a-f]+\.csv$")
    for p in sorted(OLD_DIR.glob("2512_FormOxy_S*_contour_*.csv")):
        m = rx.match(p.name)
        if not m:
            continue
        subj = int(m.group(1))
        df = pd.read_csv(p, usecols=cols)
        for b in bin_session(df):
            rows.append({"cohort": "2512", "subject_id": f"2512_S{subj}",
                         "treatment": DOSE_2512[subj], **b})

    # 2605 vehpaw
    rx = re.compile(r"^2605_FormOxy_Veh_S(\d+)_vehicle_contour_[0-9a-f]+\.csv$")
    for p in sorted(NEW_DIR.glob("2605_FormOxy_Veh_S*_vehicle_contour_*.csv")):
        m = rx.match(p.name)
        if not m:
            continue
        subj = int(m.group(1))
        df = pd.read_csv(p, usecols=cols)
        for b in bin_session(df):
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
    out_csv = ANALYSIS / "contour_dose_response_contact_only_long.csv"
    df.to_csv(out_csv, index=False)
    n_per = (df.drop_duplicates(["subject_id"])
               .groupby("treatment").size().to_dict())
    step(f"  per-group n: {n_per}")
    step(f"  wrote {out_csv.name} ({len(df)} rows)")

    # Quick per-group ratio summary for the message
    g = (df.groupby(["treatment", "bin_idx"])["ratio"].mean()
           .groupby("treatment").mean()).round(3)
    step(f"  mean ratio per group across bins: {g.to_dict()}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    plot_panel(axes[0], df, "HL",
               ylabel="HL contour intensity (contact frames only) (a.u.)",
               panel_title="HL (injected paw) — contact frames only",
               add_unity_line=False)
    plot_panel(axes[1], df, "HR",
               ylabel="HR contour intensity (contact frames only) (a.u.)",
               panel_title="HR (contralateral) — contact frames only",
               add_unity_line=False)
    plot_panel(axes[2], df, "ratio",
               ylabel="HL/HR ratio (contact frames only)",
               panel_title="HL/HR ratio — contact frames only",
               add_unity_line=True)
    fig.suptitle(
        f"Paw contour intensity — contact-only frames "
        f"(area > {AREA_FRAC:.0%} of P{AREA_PCTL})",
        fontweight="bold", fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_base = ANALYSIS / "contour_dose_response_contact_only"
    fig.savefig(out_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    step(f"  wrote {out_base}.png / svg / pdf")

    head = (
        "**Paw contour intensity — contact-only frames** "
        f"(per-session area > {AREA_FRAC:.0%} of P{AREA_PCTL})\n"
        "Same 5-group dose-response, but restricted to frames where each "
        "paw's contour area is near its per-session 'open paw' typical area. "
        "If formalin-induced asymmetry was purely a guarding/contact artifact, "
        "this view should collapse the Veh-vs-VehPaw gap. If the asymmetry "
        "persists, it's a tissue-level brightness change (swelling/blood pooling). "
        "Each subject's contact threshold is auto-calibrated from its own paw "
        "area distribution, so swelling is controlled out."
    )
    discord_upload(head, [out_base.with_suffix(".png"),
                          out_base.with_suffix(".svg"),
                          out_base.with_suffix(".pdf"),
                          out_csv])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
