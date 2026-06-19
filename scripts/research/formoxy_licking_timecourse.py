"""
5-minute-bin timecourse of Left_licking across the combined FormOxy cohorts.

Reads the per-frame prediction CSVs already on disk for both cohorts:
  - 2512_Blackbox_Formalin_Oxy/Left_paws/Results/Left_licking/*_predictions.csv
  - 2605_FormOxy_LeftPaws/results/*_Left_licking_predictions.csv

Bins each session into 5-minute windows (18000 frames at 60 fps), capped
at the first 60 min (12 bins). Plots mean +/- SEM percent-licking per
bin per dose (4-line timecourse). Posts to the #results Discord channel.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
import uuid
from pathlib import Path

# --------------------------------------------------------------------- #
# Paths / params
# --------------------------------------------------------------------- #
COHORT_NEW = Path(r"E:\RSVIDS\Blackbox\2605_FormOxy_LeftPaws")
NEW_RESULTS = COHORT_NEW / "results"
ANALYSIS = COHORT_NEW / "analysis"

COHORT_OLD = Path(r"E:\RSVIDS\Blackbox\2512_Blackbox_Formalin_Oxy\Left_paws")
OLD_PRED_DIR = COHORT_OLD / "Results" / "Left_licking"

FPS = 60
BIN_MIN = 5
BIN_FRAMES = BIN_MIN * 60 * FPS         # 18 000
N_BINS = 12                             # 60-min window

DOSE_2512 = {
    1: "Veh", 2: "Veh", 3: "Veh",
    4: "Oxy3", 5: "Oxy3", 6: "Oxy3",
    7: "Oxy10", 8: "Oxy10", 9: "Oxy10",
    10: "Oxy1", 11: "Oxy1", 12: "Oxy1",
}
DOSE_SEQ = ["Veh", "Oxy1", "Oxy3", "Oxy10"]


def dose_2605(n: int) -> str:
    return DOSE_SEQ[(n - 1) % 4]


DOSE_COLORS = {
    "Veh":   "#4A4A4A",
    "Oxy1":  "#90CAF9",
    "Oxy3":  "#1976D2",
    "Oxy10": "#0D47A1",
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
                 "User-Agent": "PixelPaws-FormOxy-Timecourse/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            step(f"Discord upload: HTTP {r.status}")
    except Exception as e:
        step(f"Discord upload failed: {e}")


# --------------------------------------------------------------------- #
# Binning
# --------------------------------------------------------------------- #
def bin_session(y, n_bins: int = N_BINS, bin_frames: int = BIN_FRAMES):
    """Return (pct_licking_list, total_s_licking_list), one per bin."""
    pct, tot = [], []
    for i in range(n_bins):
        s = i * bin_frames
        e = s + bin_frames
        chunk = y[s:e]
        if len(chunk) == 0:
            pct.append(float("nan")); tot.append(float("nan"))
        else:
            pct.append(100.0 * float(chunk.sum()) / len(chunk))
            tot.append(float(chunk.sum()) / FPS)
    return pct, tot


def collect():
    import pandas as pd
    rows = []

    # 2605 (12 sessions, first 60 min)
    rx_new = re.compile(r"2605_FormOxy_S(\d+)_formalin_Left_licking_predictions\.csv$")
    csvs_new = sorted(NEW_RESULTS.glob("2605_FormOxy_S*_formalin_Left_licking_predictions.csv"))
    step(f"2605: {len(csvs_new)} prediction CSVs")
    for p in csvs_new:
        m = rx_new.search(p.name)
        if not m:
            continue
        subj = int(m.group(1))
        df = pd.read_csv(p, usecols=["Left_licking"])
        y = df["Left_licking"].astype(int).to_numpy()
        pct_vals, tot_vals = bin_session(y)
        for b_idx, (pv, tv) in enumerate(zip(pct_vals, tot_vals)):
            rows.append({
                "cohort": "2605 (F)", "subject_id": f"2605_S{subj}",
                "dose": dose_2605(subj),
                "bin_idx": b_idx,
                "bin_min_start": b_idx * BIN_MIN,
                "bin_min_end":   (b_idx + 1) * BIN_MIN,
                "pct_licking": pv,
                "total_s_licking": tv,
            })

    # 2512 (12 sessions, first 60 min)
    rx_old = re.compile(r"2512_FormOxy_S(\d+)_PixelPaws_Left_licking_predictions\.csv$")
    csvs_old = sorted(OLD_PRED_DIR.glob("2512_FormOxy_S*_PixelPaws_Left_licking_predictions.csv"))
    step(f"2512: {len(csvs_old)} prediction CSVs")
    for p in csvs_old:
        m = rx_old.search(p.name)
        if not m:
            continue
        subj = int(m.group(1))
        df = pd.read_csv(p, usecols=["Left_licking"])
        y = df["Left_licking"].astype(int).to_numpy()
        pct_vals, tot_vals = bin_session(y)
        for b_idx, (pv, tv) in enumerate(zip(pct_vals, tot_vals)):
            rows.append({
                "cohort": "2512 (M)", "subject_id": f"2512_S{subj}",
                "dose": DOSE_2512[subj],
                "bin_idx": b_idx,
                "bin_min_start": b_idx * BIN_MIN,
                "bin_min_end":   (b_idx + 1) * BIN_MIN,
                "pct_licking": pv,
                "total_s_licking": tv,
            })

    return pd.DataFrame(rows)


def _plot_one(df, col, ylabel, out_stem: Path) -> tuple[Path, Path, Path]:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"]  = 42
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt

    x_idx = np.arange(N_BINS)
    bin_labels = [f"{i*BIN_MIN}–{(i+1)*BIN_MIN}" for i in range(N_BINS)]

    fig, ax = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
    for dose in DOSE_SEQ:
        sub = df[df["dose"] == dose]
        means, sems, ns = [], [], []
        for b_idx in range(N_BINS):
            vals = sub.loc[sub["bin_idx"] == b_idx, col].dropna().to_numpy()
            ns.append(len(vals))
            means.append(np.mean(vals) if len(vals) else np.nan)
            sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals))
                        if len(vals) > 1 else 0.0)
        means = np.array(means); sems = np.array(sems)
        c = DOSE_COLORS[dose]
        ax.plot(x_idx, means, marker="o", color=c,
                label=f"{dose} (n={max(ns)})",
                linewidth=1.8, markersize=5.5)
        ax.fill_between(x_idx, means - sems, means + sems,
                        color=c, alpha=0.18)
    ax.set_xlabel("Time post-formalin (min)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(bin_labels, rotation=35, ha="right", fontsize=8)
    ax.set_xlim(-0.4, N_BINS - 0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False)

    out_png = out_stem.with_suffix(".png")
    out_pdf = out_stem.with_suffix(".pdf")
    out_svg = out_stem.with_suffix(".svg")
    fig.savefig(out_png, format="png", dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    return out_png, out_pdf, out_svg


def plot_timecourse(df, base_stem: Path):
    pct = _plot_one(df, "pct_licking", "% time licking",
                    base_stem.with_name(base_stem.name + "_pct"))
    tot = _plot_one(df, "total_s_licking", "Total time licking (s)",
                    base_stem.with_name(base_stem.name + "_total_s"))
    return pct, tot


def main() -> int:
    import pandas as pd
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    df = collect()
    long_csv = ANALYSIS / "leftlicking_timecourse_5min.csv"
    df.to_csv(long_csv, index=False)
    step(f"Long-form: {long_csv}")

    # Wide pivot: bin x dose mean (pct + total-s)
    import numpy as np
    bins = [f"{i*BIN_MIN}-{(i+1)*BIN_MIN}" for i in range(N_BINS)]
    pivot_rows = []
    for dose in DOSE_SEQ:
        sub = df[df["dose"] == dose]
        row = {"dose": dose, "n_per_bin": sub.groupby("bin_idx").size().max()}
        for i in range(N_BINS):
            pv = sub.loc[sub["bin_idx"] == i, "pct_licking"].dropna().to_numpy()
            tv = sub.loc[sub["bin_idx"] == i, "total_s_licking"].dropna().to_numpy()
            row[f"pct_mean_{bins[i]}_min"] = round(float(np.mean(pv)), 2) if len(pv) else np.nan
            row[f"pct_sem_{bins[i]}_min"]  = round(float(np.std(pv, ddof=1) / np.sqrt(len(pv))), 2) if len(pv) > 1 else 0.0
            row[f"tot_s_mean_{bins[i]}_min"] = round(float(np.mean(tv)), 2) if len(tv) else np.nan
            row[f"tot_s_sem_{bins[i]}_min"]  = round(float(np.std(tv, ddof=1) / np.sqrt(len(tv))), 2) if len(tv) > 1 else 0.0
        pivot_rows.append(row)
    pivot_df = pd.DataFrame(pivot_rows)
    pivot_csv = ANALYSIS / "leftlicking_timecourse_5min_summary.csv"
    pivot_df.to_csv(pivot_csv, index=False)
    step(f"Summary:   {pivot_csv}")

    base_stem = ANALYSIS / "leftlicking_timecourse_5min"
    pct_files, tot_files = plot_timecourse(df, base_stem)
    step(f"Figures:   {pct_files[0].name},  {tot_files[0].name}")

    discord_upload(
        f"**FormOxy Left_licking timecourse — % time licking** "
        f"({BIN_MIN}-min bins, 0–60 min). Pooled **2512 (M, n=12) + "
        f"2605 (F, n=12)** = n=6/dose, mean ± SEM. "
        f"PNG + PDF + SVG.",
        [*pct_files],
    )
    discord_upload(
        f"**FormOxy Left_licking timecourse — total time licking (s)** "
        f"({BIN_MIN}-min bins, 0–60 min). Same pooling. "
        f"PNG + PDF + SVG + summary CSV + long-form CSV.",
        [*tot_files, pivot_csv, long_csv],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
