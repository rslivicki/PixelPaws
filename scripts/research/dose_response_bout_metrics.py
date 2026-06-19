"""
Dose-response TIME metrics for two cohorts:
  - Scratching  · Rimonabant   (260515_Rim_DoseResp)
  - Left_licking · Oxycodone   (FormOxy 2512 + 2605 combined)

For each cohort, render three panels (PNG + PDF + SVG):
  1. Total time in behavior (s) per session, mean +/- SEM by dose, with
     per-mouse dots
  2. Mean bout duration (s) per session, mean +/- SEM by dose
  3. Cumulative time in behavior (s) in 5-min bins (timecourse) —
     line per dose, shaded SEM band

All exports use editable text (pdf.fonttype=42, svg.fonttype=none) so
they're Illustrator-ready.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
import uuid
from pathlib import Path

REPO = Path(r"E:\PixelPaws")
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

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
                 "User-Agent": "PixelPaws-DoseResp-BoutMetrics/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            step(f"  HTTP {r.status}")
    except Exception as e:
        step(f"  upload failed: {e}")


def find_bouts(y):
    bouts = []; in_b = False; start = 0
    for i, v in enumerate(y):
        if v and not in_b:
            in_b = True; start = i
        elif not v and in_b:
            in_b = False; bouts.append((start, i - 1))
    if in_b:
        bouts.append((start, len(y) - 1))
    return bouts


FPS = 60
BIN_MIN = 5
BIN_FRAMES = BIN_MIN * 60 * FPS   # 18000
N_BINS = 12                       # 60-min cap

# Rimonabant: black/grey vehicle, green scale for active doses
# Oxycodone : black/grey vehicle, blue  scale for active doses
GREEN_3 = ["#A5D6A7", "#43A047", "#1B5E20"]   # light -> dark
BLUE_3  = ["#90CAF9", "#1976D2", "#0D47A1"]   # light -> dark
VEH_GREY = "#4A4A4A"

RIM_DOSE_SEQ = ["VEH", "1 mg/kg", "3 mg/kg", "10 mg/kg"]
RIM_DOSE_COLORS = {
    "VEH":      VEH_GREY,
    "1 mg/kg":  GREEN_3[0],
    "3 mg/kg":  GREEN_3[1],
    "10 mg/kg": GREEN_3[2],
}
def rim_dose_for(n: int) -> str:
    return RIM_DOSE_SEQ[(n - 1) % 4]

# Male 2510 cohort key (from 2602_Rimonabant_Key.xlsx) — dose-grouped.
DOSE_2510_M = {
    1:  "10 mg/kg", 2:  "10 mg/kg", 3:  "10 mg/kg",
    4:  "3 mg/kg",  5:  "3 mg/kg",  6:  "3 mg/kg",
    7:  "1 mg/kg",  8:  "1 mg/kg",  9:  "1 mg/kg",
    10: "VEH",      11: "VEH",      12: "VEH",
}


OXY_DOSE_SEQ = ["Veh", "Oxy1", "Oxy3", "Oxy10"]
OXY_DOSE_COLORS = {
    "Veh":   VEH_GREY,
    "Oxy1":  BLUE_3[0],
    "Oxy3":  BLUE_3[1],
    "Oxy10": BLUE_3[2],
}
DOSE_2512 = {1: "Veh", 2: "Veh", 3: "Veh",
             4: "Oxy3", 5: "Oxy3", 6: "Oxy3",
             7: "Oxy10", 8: "Oxy10", 9: "Oxy10",
             10: "Oxy1", 11: "Oxy1", 12: "Oxy1"}
def oxy_2605_dose(n: int) -> str:
    return OXY_DOSE_SEQ[(n - 1) % 4]


# --------------------------------------------------------------------- #
# Generic plotting (3-panel) — given a per-session table + bin matrix
# --------------------------------------------------------------------- #
def plot_dose_response_triple(per_session, cum_bins, dose_seq, dose_colors,
                              behavior_name, cohort_name, out_stem: Path):
    """per_session = list of dicts {dose, total_time_s, mean_bout_s,
       n_frames_capped, subject_id, cohort}
       cum_bins = dict {dose: np.array (n_subjects, N_BINS) — cumulative TIME (s)}
    """
    df = pd.DataFrame(per_session)
    fig = plt.figure(figsize=(14, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3)
    ax_tot, ax_dur, ax_cum = gs.subplots()

    x = np.arange(len(dose_seq))
    colors = [dose_colors[d] for d in dose_seq]

    cohorts_present = set(df["cohort"].unique())
    has_sex_split = any("(F)" in c or "(M)" in c for c in cohorts_present)

    def _bar_with_dots(ax, value_col, ylabel, title, show_legend=False):
        means, sems = [], []
        for d in dose_seq:
            v = df.loc[df["dose"] == d, value_col].astype(float).dropna().to_numpy()
            means.append(np.mean(v) if len(v) else np.nan)
            sems.append(np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
        ax.bar(x, means, yerr=sems, capsize=4,
               color=colors, alpha=0.55, edgecolor="black", linewidth=0.6)
        for i, d in enumerate(dose_seq):
            sub = df[(df["dose"] == d) & df[value_col].notna()]
            c = dose_colors[d]
            if has_sex_split:
                males   = sub[sub["cohort"].str.contains(r"\(M\)", regex=True)][value_col].astype(float).to_numpy()
                females = sub[sub["cohort"].str.contains(r"\(F\)", regex=True)][value_col].astype(float).to_numpy()
                if len(males):
                    xs = np.random.normal(i, 0.06, len(males))
                    ax.scatter(xs, males, c=c, s=28,
                               edgecolors="black", linewidths=0.7, zorder=3)
                if len(females):
                    xs = np.random.normal(i, 0.06, len(females))
                    ax.scatter(xs, females, facecolors="none",
                               edgecolors=c, s=28, linewidths=1.2, zorder=3)
            else:
                pts = sub[value_col].astype(float).to_numpy()
                if len(pts):
                    xs = np.random.normal(i, 0.06, len(pts))
                    ax.scatter(xs, pts, c=c, s=28,
                               edgecolors="black", linewidths=0.7, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(dose_seq, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if show_legend:
            handles = [
                plt.matplotlib.patches.Patch(
                    facecolor=dose_colors[d], edgecolor="black",
                    linewidth=0.5, alpha=0.55,
                    label=f"{d} (n={int((df['dose']==d).sum())})")
                for d in dose_seq
            ]
            if has_sex_split:
                handles += [
                    plt.Line2D([0], [0], marker="o", color="none",
                               markerfacecolor="0.4", markeredgecolor="black",
                               markeredgewidth=0.7, markersize=7,
                               label="Male"),
                    plt.Line2D([0], [0], marker="o", color="none",
                               markerfacecolor="none", markeredgecolor="0.4",
                               markeredgewidth=1.2, markersize=7,
                               label="Female"),
                ]
            ax.legend(handles=handles, fontsize=7, loc="upper right",
                      frameon=False)

    verb = behavior_name.lower()
    # Panel 1 — total time spent (legend lives here)
    _bar_with_dots(ax_tot, "total_time_s",
                   f"Total time {verb} (s)",
                   "",
                   show_legend=True)
    # Panel 2 — mean bout duration
    _bar_with_dots(ax_dur, "mean_bout_s",
                   "Mean bout duration (s)", "")

    # Panel 3 — cumulative TIME (5-min bins)
    bin_centres_min = [(i + 0.5) * BIN_MIN for i in range(N_BINS)]
    for d in dose_seq:
        mat = cum_bins.get(d)
        if mat is None or len(mat) == 0:
            continue
        means_c = np.nanmean(mat, axis=0)
        sems_c = (np.nanstd(mat, axis=0, ddof=1) /
                  np.sqrt(np.sum(~np.isnan(mat), axis=0))) \
                  if mat.shape[0] > 1 else np.zeros(N_BINS)
        c = dose_colors[d]
        ax_cum.plot(bin_centres_min, means_c, marker="o", color=c,
                    linewidth=1.6, markersize=4.5,
                    label=f"{d} (n={mat.shape[0]})")
        ax_cum.fill_between(bin_centres_min,
                            means_c - sems_c, means_c + sems_c,
                            color=c, alpha=0.18)
    ax_cum.set_xlabel("Time (min)")
    ax_cum.set_ylabel(f"Cumulative time {verb} (s)")
    ax_cum.spines["top"].set_visible(False)
    ax_cum.spines["right"].set_visible(False)

    out_png = out_stem.with_suffix(".png")
    out_pdf = out_stem.with_suffix(".pdf")
    out_svg = out_stem.with_suffix(".svg")
    fig.savefig(out_png, format="png", dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_png, out_pdf, out_svg


# --------------------------------------------------------------------- #
# Scratching — Rimonabant dose-response
# --------------------------------------------------------------------- #
def _process_scratching_series(y, dose, cohort_tag, subject_id,
                               per_session, cum_bins):
    n_frames = min(len(y), N_BINS * BIN_FRAMES)
    y = y[:n_frames]
    bouts = find_bouts(y)
    n_bouts = len(bouts)
    mean_bout_s = (np.mean([e - s + 1 for s, e in bouts]) / FPS
                   if bouts else float("nan"))
    total_time_s = float(y.sum()) / FPS
    per_session.append({
        "cohort": cohort_tag,
        "subject_id": subject_id,
        "dose": dose,
        "n_bouts": n_bouts,
        "mean_bout_s": mean_bout_s,
        "total_time_s": total_time_s,
        "n_frames_capped": n_frames,
    })
    cum = np.zeros(N_BINS); running = 0.0
    for b_idx in range(N_BINS):
        s = b_idx * BIN_FRAMES
        e = min(s + BIN_FRAMES, n_frames)
        if s >= n_frames:
            cum[b_idx] = running
            continue
        running += float(y[s:e].sum()) / FPS
        cum[b_idx] = running
    cum_bins[dose].append(cum)


def build_rim_scratching():
    PROJ_NEW = Path(r"E:\RSVIDS\Blackbox\260515_Rim_DoseResp")
    PROJ_OLD = Path(r"E:\RSVIDS\Blackbox\2510_Blackbox_Rimonabant"
                    r"\Blackbox_videos-selected\2602_Rimonabant_cropped")
    ANALYSIS = PROJ_NEW / "analysis"
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    OUT_STEM = ANALYSIS / "rim_scratching_time_metrics"

    per_session = []
    cum_bins = {d: [] for d in RIM_DOSE_SEQ}

    # ─ cohort A: 260515_Rim_DoseResp (postrim sessions only) ─
    rx_new = re.compile(r"^fem_(baseline|postrim)_Rim(\d+).*_predictions\.csv$")
    n_new = 0
    for p in sorted((PROJ_NEW / "results").glob("fem_*_predictions.csv")):
        m = rx_new.match(p.name)
        if not m:
            continue
        cond = m.group(1)
        if cond != "postrim":
            continue
        mouse = int(m.group(2))
        dose = rim_dose_for(mouse)
        df = pd.read_csv(p, usecols=["Scratching"])
        y = df["Scratching"].astype(int).to_numpy()
        _process_scratching_series(y, dose, "2605 (F)",
                                   f"2605_Rim{mouse}",
                                   per_session, cum_bins)
        n_new += 1

    # ─ cohort B: 260219 Rim_DoseResp (S1-S12, same 4-dose cycle) ─
    rx_old = re.compile(r"^260\d{3}_Rim_S(\d+)_cropped_PixelPaws_Scratching_"
                        r"AllFeatures_predictions\.csv$")
    n_old = 0
    old_root = PROJ_OLD / "Results" / "Scratching"
    for p in sorted(old_root.glob("260*_Rim_S*_cropped_PixelPaws_Scratching_AllFeatures_predictions.csv")):
        m = rx_old.match(p.name)
        if not m:
            continue
        mouse = int(m.group(1))
        dose = DOSE_2510_M[mouse]
        df = pd.read_csv(p, usecols=["Scratching"])
        y = df["Scratching"].astype(int).to_numpy()
        _process_scratching_series(y, dose, "2510 (M)",
                                   f"2510_Rim_S{mouse}",
                                   per_session, cum_bins)
        n_old += 1

    for d in cum_bins:
        cum_bins[d] = np.array(cum_bins[d]) if cum_bins[d] else np.empty((0, N_BINS))

    step(f"  rim scratching sessions: {len(per_session)} "
         f"(2605 F={n_new}, 2510 M={n_old})")
    df_out = pd.DataFrame(per_session)
    summary_csv = ANALYSIS / "rim_scratching_time_metrics.csv"
    df_out.to_csv(summary_csv, index=False)
    return per_session, cum_bins, OUT_STEM, summary_csv


# --------------------------------------------------------------------- #
# Left_licking — FormOxy combined cohort
# --------------------------------------------------------------------- #
def build_oxy_licking():
    NEW = Path(r"E:\RSVIDS\Blackbox\2605_FormOxy_LeftPaws")
    OLD = Path(r"E:\RSVIDS\Blackbox\2512_Blackbox_Formalin_Oxy\Left_paws")
    ANALYSIS = NEW / "analysis"
    OUT_STEM = ANALYSIS / "oxy_licking_time_metrics"

    per_session = []
    cum_bins = {d: [] for d in OXY_DOSE_SEQ}

    # 2605 (new)
    rx_new = re.compile(r"^2605_FormOxy_S(\d+)_formalin_Left_licking_predictions\.csv$")
    for p in sorted((NEW / "results").glob("2605_FormOxy_S*_formalin_Left_licking_predictions.csv")):
        m = rx_new.match(p.name)
        if not m:
            continue
        subj = int(m.group(1))
        dose = oxy_2605_dose(subj)
        df = pd.read_csv(p, usecols=["Left_licking"])
        y = df["Left_licking"].astype(int).to_numpy()
        n_frames = min(len(y), N_BINS * BIN_FRAMES)
        y = y[:n_frames]
        bouts = find_bouts(y)
        per_session.append({
            "cohort": "2605 (F)", "subject_id": f"2605_S{subj}", "dose": dose,
            "n_bouts": len(bouts),
            "mean_bout_s": (np.mean([e - s + 1 for s, e in bouts]) / FPS
                            if bouts else float("nan")),
            "total_time_s": float(y.sum()) / FPS,
            "n_frames_capped": n_frames,
        })
        cum = np.zeros(N_BINS); running = 0.0
        for b_idx in range(N_BINS):
            s = b_idx * BIN_FRAMES
            e = min(s + BIN_FRAMES, n_frames)
            if s >= n_frames:
                cum[b_idx] = running; continue
            running += float(y[s:e].sum()) / FPS
            cum[b_idx] = running
        cum_bins[dose].append(cum)

    # 2512 (existing)
    rx_old = re.compile(r"^2512_FormOxy_S(\d+)_PixelPaws_Left_licking_predictions\.csv$")
    for p in sorted((OLD / "Results" / "Left_licking").glob("2512_FormOxy_S*_PixelPaws_Left_licking_predictions.csv")):
        m = rx_old.match(p.name)
        if not m:
            continue
        subj = int(m.group(1))
        dose = DOSE_2512[subj]
        df = pd.read_csv(p, usecols=["Left_licking"])
        y = df["Left_licking"].astype(int).to_numpy()
        n_frames = min(len(y), N_BINS * BIN_FRAMES)
        y = y[:n_frames]
        bouts = find_bouts(y)
        per_session.append({
            "cohort": "2512 (M)", "subject_id": f"2512_S{subj}", "dose": dose,
            "n_bouts": len(bouts),
            "mean_bout_s": (np.mean([e - s + 1 for s, e in bouts]) / FPS
                            if bouts else float("nan")),
            "total_time_s": float(y.sum()) / FPS,
            "n_frames_capped": n_frames,
        })
        cum = np.zeros(N_BINS); running = 0.0
        for b_idx in range(N_BINS):
            s = b_idx * BIN_FRAMES
            e = min(s + BIN_FRAMES, n_frames)
            if s >= n_frames:
                cum[b_idx] = running; continue
            running += float(y[s:e].sum()) / FPS
            cum[b_idx] = running
        cum_bins[dose].append(cum)

    for d in cum_bins:
        cum_bins[d] = np.array(cum_bins[d]) if cum_bins[d] else np.empty((0, N_BINS))

    step(f"  oxy licking sessions: {len(per_session)}")
    df_out = pd.DataFrame(per_session)
    summary_csv = ANALYSIS / "oxy_licking_time_metrics.csv"
    df_out.to_csv(summary_csv, index=False)
    return per_session, cum_bins, OUT_STEM, summary_csv


def main() -> int:
    # ── 1) Scratching · Rimonabant ──
    step("=== Scratching · Rimonabant ===")
    rim_sess, rim_cum, rim_stem, rim_csv = build_rim_scratching()
    rim_png, rim_pdf, rim_svg = plot_dose_response_triple(
        rim_sess, rim_cum, RIM_DOSE_SEQ, RIM_DOSE_COLORS,
        behavior_name="Scratching",
        cohort_name="Rimonabant — 2510 (M) + 2605 (F), postrim",
        out_stem=rim_stem,
    )
    step(f"  -> {rim_png}")

    discord_upload(
        "**Scratching · Rimonabant dose-response — time metrics (combined cohorts)**\n"
        "Three panels: total time scratching (s) per session, mean bout "
        "duration (s), cumulative time scratching (s) in 5-min bins. "
        "Pooled cohorts **2510 (M, n=12) + 2605 (F, n=15)** = n=27, "
        "postrim. Dose key: 2510 males are dose-grouped (S1-3=10 mg/kg, "
        "S4-6=3 mg/kg, S7-9=1 mg/kg, S10-12=VEH); 2605 females cycle "
        "VEH/1/3/10 per Rim<N>. PNG + PDF + SVG + CSV.",
        [rim_png, rim_pdf, rim_svg, rim_csv],
    )

    # ── 2) Left_licking · Oxycodone (FormOxy combined) ──
    step("=== Left_licking · Oxycodone ===")
    oxy_sess, oxy_cum, oxy_stem, oxy_csv = build_oxy_licking()
    oxy_png, oxy_pdf, oxy_svg = plot_dose_response_triple(
        oxy_sess, oxy_cum, OXY_DOSE_SEQ, OXY_DOSE_COLORS,
        behavior_name="Left_licking",
        cohort_name="Oxycodone × formalin (2512 + 2605 combined)",
        out_stem=oxy_stem,
    )
    step(f"  -> {oxy_png}")

    discord_upload(
        "**Left_licking · Oxycodone × formalin dose-response — time metrics**\n"
        "Three panels: total time licking (s) per session, mean bout "
        "duration (s), cumulative time licking (s) in 5-min bins. "
        "Pooled cohorts 2512 + 2605, n = 6 per dose, first 60 min capped. "
        "PNG + PDF (editable text) + SVG + per-session CSV.",
        [oxy_png, oxy_pdf, oxy_svg, oxy_csv],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
