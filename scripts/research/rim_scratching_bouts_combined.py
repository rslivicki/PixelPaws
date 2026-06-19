"""
Bout-based dose-response panels for Scratching · Rimonabant, using the
combined cohort (260515 + 260219, n=27, postrim sessions).

Three panels:
  1. Total bouts per session, mean +/- SEM by dose, per-mouse dots
  2. Mean bout duration (s) per session
  3. Cumulative bouts in 5-min bins (timecourse)
"""
from __future__ import annotations

import json, re, sys, time, urllib.request, uuid
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

FPS = 60
BIN_MIN = 5
BIN_FRAMES = BIN_MIN * 60 * FPS
N_BINS = 12

VEH_GREY = "#4A4A4A"
GREEN_3 = ["#A5D6A7", "#43A047", "#1B5E20"]
RIM_DOSE_SEQ = ["VEH", "1 mg/kg", "3 mg/kg", "10 mg/kg"]
RIM_DOSE_COLORS = {
    "VEH": VEH_GREY,
    "1 mg/kg": GREEN_3[0],
    "3 mg/kg": GREEN_3[1],
    "10 mg/kg": GREEN_3[2],
}


def step(msg): print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def rim_dose_for(n): return RIM_DOSE_SEQ[(n - 1) % 4]

# Male 2510 cohort key (from 2602_Rimonabant_Key.xlsx) — dose-grouped,
# not cycled.
DOSE_2510_M = {
    1:  "10 mg/kg", 2:  "10 mg/kg", 3:  "10 mg/kg",
    4:  "3 mg/kg",  5:  "3 mg/kg",  6:  "3 mg/kg",
    7:  "1 mg/kg",  8:  "1 mg/kg",  9:  "1 mg/kg",
    10: "VEH",      11: "VEH",      12: "VEH",
}


def find_bouts(y):
    bouts = []; in_b = False; start = 0
    for i, v in enumerate(y):
        if v and not in_b: in_b = True; start = i
        elif not v and in_b: in_b = False; bouts.append((start, i - 1))
    if in_b: bouts.append((start, len(y) - 1))
    return bouts


def discord_upload(content, files):
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
                 "User-Agent": "PixelPaws-Rim-Bouts/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            step(f"  HTTP {r.status}")
    except Exception as e:
        step(f"  upload failed: {e}")


def add_session(y, dose, cohort, subject, per_session, cum_bins):
    n_frames = min(len(y), N_BINS * BIN_FRAMES)
    y = y[:n_frames]
    bouts = find_bouts(y)
    per_session.append({
        "cohort": cohort, "subject_id": subject, "dose": dose,
        "n_bouts": len(bouts),
        "mean_bout_s": (np.mean([e - s + 1 for s, e in bouts]) / FPS
                        if bouts else float("nan")),
        "total_time_s": float(y.sum()) / FPS,
        "n_frames_capped": n_frames,
    })
    cum = np.zeros(N_BINS); running = 0
    for b_idx in range(N_BINS):
        s = b_idx * BIN_FRAMES; e = s + BIN_FRAMES
        if s >= n_frames:
            cum[b_idx] = running; continue
        running += sum(1 for bs, _ in bouts if s <= bs < e)
        cum[b_idx] = running
    cum_bins[dose].append(cum)


def build():
    PROJ_NEW = Path(r"E:\RSVIDS\Blackbox\260515_Rim_DoseResp")
    PROJ_OLD = Path(r"E:\RSVIDS\Blackbox\2510_Blackbox_Rimonabant"
                    r"\Blackbox_videos-selected\2602_Rimonabant_cropped")
    ANALYSIS = PROJ_NEW / "analysis"
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    OUT_STEM = ANALYSIS / "rim_scratching_bout_metrics_combined"

    per_session = []
    cum_bins = {d: [] for d in RIM_DOSE_SEQ}

    rx_new = re.compile(r"^fem_(baseline|postrim)_Rim(\d+).*_predictions\.csv$")
    n_new = 0
    for p in sorted((PROJ_NEW / "results").glob("fem_*_predictions.csv")):
        m = rx_new.match(p.name)
        if not m or m.group(1) != "postrim":
            continue
        mouse = int(m.group(2))
        y = pd.read_csv(p, usecols=["Scratching"])["Scratching"].astype(int).to_numpy()
        add_session(y, rim_dose_for(mouse), "2605 (F)",
                    f"2605_Rim{mouse}", per_session, cum_bins)
        n_new += 1

    rx_old = re.compile(r"^260\d{3}_Rim_S(\d+)_cropped_PixelPaws_Scratching_"
                        r"AllFeatures_predictions\.csv$")
    n_old = 0
    for p in sorted((PROJ_OLD / "Results" / "Scratching").glob(
            "260*_Rim_S*_cropped_PixelPaws_Scratching_AllFeatures_predictions.csv")):
        m = rx_old.match(p.name)
        if not m: continue
        mouse = int(m.group(1))
        y = pd.read_csv(p, usecols=["Scratching"])["Scratching"].astype(int).to_numpy()
        add_session(y, DOSE_2510_M[mouse], "2510 (M)",
                    f"2510_Rim_S{mouse}", per_session, cum_bins)
        n_old += 1

    for d in cum_bins:
        cum_bins[d] = np.array(cum_bins[d]) if cum_bins[d] else np.empty((0, N_BINS))

    step(f"  rim scratching sessions: {len(per_session)} "
         f"(2605 F={n_new}, 2510 M={n_old})")
    csv_out = OUT_STEM.with_suffix(".csv")
    pd.DataFrame(per_session).to_csv(csv_out, index=False)
    return per_session, cum_bins, OUT_STEM, csv_out


def plot(per_session, cum_bins, out_stem: Path):
    df = pd.DataFrame(per_session)
    fig = plt.figure(figsize=(14, 4.6), constrained_layout=True)
    ax_tot, ax_dur, ax_cum = fig.add_gridspec(1, 3).subplots()
    x = np.arange(len(RIM_DOSE_SEQ))

    def bars(ax, col, ylabel, title, show_legend=False):
        means, sems = [], []
        colors = [RIM_DOSE_COLORS[d] for d in RIM_DOSE_SEQ]
        for d in RIM_DOSE_SEQ:
            v = df.loc[df["dose"] == d, col].astype(float).dropna().to_numpy()
            means.append(np.mean(v) if len(v) else np.nan)
            sems.append(np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
        ax.bar(x, means, yerr=sems, capsize=4,
               color=colors, alpha=0.55, edgecolor="black", linewidth=0.6)
        for i, d in enumerate(RIM_DOSE_SEQ):
            sub = df[(df["dose"] == d) & df[col].notna()]
            c = RIM_DOSE_COLORS[d]
            males   = sub[sub["cohort"] == "2510 (M)"][col].astype(float).to_numpy()
            females = sub[sub["cohort"] == "2605 (F)"][col].astype(float).to_numpy()
            if len(males):
                xs = np.random.normal(i, 0.06, len(males))
                ax.scatter(xs, males, c=c, s=28,
                           edgecolors="black", linewidths=0.7, zorder=3)
            if len(females):
                xs = np.random.normal(i, 0.06, len(females))
                ax.scatter(xs, females, facecolors="none",
                           edgecolors=c, s=28, linewidths=1.2, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(RIM_DOSE_SEQ, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if show_legend:
            handles = [
                plt.matplotlib.patches.Patch(
                    facecolor=RIM_DOSE_COLORS[d], edgecolor="black",
                    linewidth=0.5, alpha=0.55,
                    label=f"{d} (n={int((df['dose']==d).sum())})")
                for d in RIM_DOSE_SEQ
            ]
            handles += [
                plt.Line2D([0], [0], marker="o", color="none",
                           markerfacecolor="0.4", markeredgecolor="black",
                           markeredgewidth=0.7, markersize=7,
                           label="Male (2510)"),
                plt.Line2D([0], [0], marker="o", color="none",
                           markerfacecolor="none", markeredgecolor="0.4",
                           markeredgewidth=1.2, markersize=7,
                           label="Female (2605)"),
            ]
            ax.legend(handles=handles, fontsize=7, loc="upper left",
                      frameon=False)

    bars(ax_tot, "n_bouts", "Total bouts", "", show_legend=True)
    bars(ax_dur, "mean_bout_s", "Mean bout duration (s)", "")

    bin_centres = [(i + 0.5) * BIN_MIN for i in range(N_BINS)]
    for d in RIM_DOSE_SEQ:
        mat = cum_bins.get(d)
        if mat is None or len(mat) == 0:
            continue
        means_c = np.nanmean(mat, axis=0)
        sems_c = (np.nanstd(mat, axis=0, ddof=1) /
                  np.sqrt(np.sum(~np.isnan(mat), axis=0))) \
                  if mat.shape[0] > 1 else np.zeros(N_BINS)
        c = RIM_DOSE_COLORS[d]
        ax_cum.plot(bin_centres, means_c, marker="o", color=c,
                    linewidth=1.6, markersize=4.5,
                    label=f"{d} (n={mat.shape[0]})")
        ax_cum.fill_between(bin_centres, means_c - sems_c, means_c + sems_c,
                            color=c, alpha=0.18)
    ax_cum.set_xlabel("Time (min)")
    ax_cum.set_ylabel("Cumulative bouts")
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


def main():
    step("=== Scratching · Rimonabant (BOUTS, combined cohorts) ===")
    sess, cum, stem, csv_out = build()
    png, pdf, svg = plot(sess, cum, stem)
    step(f"  -> {png}")
    discord_upload(
        "**Scratching · Rimonabant dose-response — BOUT metrics (combined)**\n"
        "Three panels: total bouts per session, mean bout duration (s), "
        "cumulative bouts in 5-min bins. Pooled cohorts **2510 (M, n=12) "
        "+ 2605 (F, n=15)** = n=27, postrim, 4-dose cycle VEH / 1 / 3 / "
        "10 mg/kg. PNG + PDF (editable text) + SVG + per-session CSV.",
        [png, pdf, svg, csv_out],
    )


if __name__ == "__main__":
    main()
