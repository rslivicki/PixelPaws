"""
Combined paw-intensity (brightness) analysis for the FormOxy left-paws study.

Reads per-frame paw brightness CSVs from both cohorts:
  - 2512_Blackbox_Formalin_Oxy/Left_paws/gait_limb_analysis/*_brt_*.csv
  - 2605_FormOxy_LeftPaws/gait_limb_analysis/*_brt_*.csv

Columns: Pix_hlpaw, Pix_hrpaw, Pix_flpaw, Pix_frpaw (mean ROI pixel
intensity per frame). Both cohorts used the same Gait & Limb ROI config
(hind 20 px, fore 15 px, threshold 101).

Outputs:
  - 5-minute timecourse plot of Pix_hlpaw (formalin-injected paw) and the
    L-R hind asymmetry index Log10(Pix_hlpaw/Pix_hrpaw), one line per dose
  - Per-dose mean-intensity bar plot (first 60 min) for all 4 paws
  - Combined per-session long-form CSV + per-dose summary CSV
  - Posts to the #results Discord channel.
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
NEW_BRT = COHORT_NEW / "gait_limb_analysis"
ANALYSIS = COHORT_NEW / "analysis"

COHORT_OLD = Path(r"E:\RSVIDS\Blackbox\2512_Blackbox_Formalin_Oxy\Left_paws")
OLD_BRT = COHORT_OLD / "gait_limb_analysis"

FPS = 60
BIN_MIN = 5
BIN_FRAMES = BIN_MIN * 60 * FPS
N_BINS = 12
WINDOW_FRAMES = N_BINS * BIN_FRAMES

PAW_COLS = ["Pix_hlpaw", "Pix_hrpaw", "Pix_flpaw", "Pix_frpaw"]

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
    "Veh":   "#7f7f7f",
    "Oxy1":  "#fbbf24",
    "Oxy3":  "#f97316",
    "Oxy10": "#dc2626",
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
                 "User-Agent": "PixelPaws-FormOxy-PawIntensity/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            step(f"Discord upload: HTTP {r.status}")
    except Exception as e:
        step(f"Discord upload failed: {e}")


# --------------------------------------------------------------------- #
# Per-session binning
# --------------------------------------------------------------------- #
def bin_session(df, n_bins: int = N_BINS, bin_frames: int = BIN_FRAMES):
    """Mean each Pix_* column per 5-min bin."""
    import numpy as np
    out = []
    for i in range(n_bins):
        s = i * bin_frames
        e = s + bin_frames
        chunk = df.iloc[s:e]
        row = {"bin_idx": i, "bin_min_start": i * BIN_MIN, "bin_min_end": (i + 1) * BIN_MIN}
        for c in PAW_COLS:
            if c in chunk.columns and len(chunk):
                row[c] = float(chunk[c].mean())
            else:
                row[c] = float("nan")
        # Hind asymmetry: log10(hlpaw / hrpaw). +ve = injected paw brighter.
        if not np.isnan(row.get("Pix_hlpaw", float("nan"))) and not np.isnan(row.get("Pix_hrpaw", float("nan"))) and row["Pix_hrpaw"] > 0:
            row["LR_log_ratio"] = float(np.log10(max(row["Pix_hlpaw"], 1e-6) / max(row["Pix_hrpaw"], 1e-6)))
        else:
            row["LR_log_ratio"] = float("nan")
        out.append(row)
    return out


def collect():
    import pandas as pd
    rows = []

    # 2605: filename '2605_FormOxy_S<N>_formalin_brt_<hash>.csv'
    rx_new = re.compile(r"^2605_FormOxy_S(\d+)_formalin_brt_[0-9a-f]+\.csv$")
    csvs_new = sorted(NEW_BRT.glob("2605_FormOxy_S*_formalin_brt_*.csv"))
    step(f"2605: {len(csvs_new)} brt CSVs")
    for p in csvs_new:
        m = rx_new.match(p.name)
        if not m:
            continue
        subj = int(m.group(1))
        df = pd.read_csv(p)
        for b in bin_session(df):
            rows.append({"cohort": "2605", "subject_id": f"2605_S{subj}",
                         "dose": dose_2605(subj), **b})

    # 2512: filename '2512_FormOxy_S<N>_brt_<hash>.csv'
    rx_old = re.compile(r"^2512_FormOxy_S(\d+)_brt_[0-9a-f]+\.csv$")
    csvs_old = sorted(OLD_BRT.glob("2512_FormOxy_S*_brt_*.csv"))
    step(f"2512: {len(csvs_old)} brt CSVs")
    for p in csvs_old:
        m = rx_old.match(p.name)
        if not m:
            continue
        subj = int(m.group(1))
        df = pd.read_csv(p)
        for b in bin_session(df):
            rows.append({"cohort": "2512", "subject_id": f"2512_S{subj}",
                         "dose": DOSE_2512[subj], **b})

    return pd.DataFrame(rows)


def plot_timecourse(df, out_png: Path):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_min = [(i + 0.5) * BIN_MIN for i in range(N_BINS)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    panels = [
        ("Pix_hlpaw", "Pix_hlpaw (left hind, formalin-injected)"),
        ("LR_log_ratio", "Log10(Pix_hlpaw / Pix_hrpaw) — L vs R hind asymmetry"),
    ]

    for ax, (col, title) in zip(axes, panels):
        for dose in DOSE_SEQ:
            sub = df[df["dose"] == dose]
            means, sems, ns = [], [], []
            for b_idx in range(N_BINS):
                vals = sub.loc[sub["bin_idx"] == b_idx, col].dropna().to_numpy()
                ns.append(len(vals))
                means.append(np.mean(vals) if len(vals) else np.nan)
                sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
            means = np.array(means)
            sems = np.array(sems)
            c = DOSE_COLORS[dose]
            ax.plot(x_min, means, marker="o", color=c, label=f"{dose} (n={max(ns)})",
                    linewidth=2.0, markersize=5)
            ax.fill_between(x_min, means - sems, means + sems, color=c, alpha=0.2)
        ax.set_xlabel("Time post-formalin (min)")
        ax.set_title(title)
        ax.set_xticks([(i + 0.5) * BIN_MIN for i in range(N_BINS)])
        ax.set_xticklabels([f"{i * BIN_MIN}-{(i + 1) * BIN_MIN}" for i in range(N_BINS)],
                           rotation=35, fontsize=8, ha="right")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", title="Dose", fontsize=8)
        if col == "LR_log_ratio":
            ax.axhline(0, color="black", linestyle="--", linewidth=0.6, alpha=0.6)
            ax.set_ylabel("log10 (formalin / control)")
        else:
            ax.set_ylabel("Mean ROI brightness (a.u.)")

    fig.suptitle("FormOxy paw intensity — 5-min timecourse, combined 2512 + 2605",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_dose_bars(df, out_png: Path):
    """First-60-min mean for each paw, bar plot per dose."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Mean across the 12 bins per session, then per dose
    per_sess = (df.groupby(["cohort", "subject_id", "dose"])[PAW_COLS]
                  .mean()
                  .reset_index())
    per_sess["LR_log_ratio"] = np.log10(
        np.clip(per_sess["Pix_hlpaw"], 1e-6, None) /
        np.clip(per_sess["Pix_hrpaw"], 1e-6, None)
    )

    metrics = [("Pix_hlpaw", "L hind (injected)"),
               ("Pix_hrpaw", "R hind (control)"),
               ("Pix_flpaw", "L fore"),
               ("Pix_frpaw", "R fore"),
               ("LR_log_ratio", "Log10(hl/hr) asymmetry")]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5))
    for ax, (col, title) in zip(axes, metrics):
        means, sems = [], []
        for d in DOSE_SEQ:
            vals = per_sess.loc[per_sess["dose"] == d, col].astype(float).to_numpy()
            means.append(np.mean(vals) if len(vals) else np.nan)
            sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
        x = np.arange(len(DOSE_SEQ))
        colors = [DOSE_COLORS[d] for d in DOSE_SEQ]
        ax.bar(x, means, yerr=sems, capsize=4, color=colors, alpha=0.55,
               edgecolor="black", linewidth=0.6)
        for i, d in enumerate(DOSE_SEQ):
            sub = per_sess[per_sess["dose"] == d]
            for cohort, marker in [("2512", "o"), ("2605", "s")]:
                vals = sub.loc[sub["cohort"] == cohort, col].astype(float).to_numpy()
                if len(vals) == 0:
                    continue
                xs = np.random.normal(i, 0.06, len(vals))
                ax.scatter(xs, vals, color="black", marker=marker, s=22,
                           edgecolors="white", linewidths=0.5, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(DOSE_SEQ, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        if col == "LR_log_ratio":
            ax.axhline(0, color="black", linestyle="--", linewidth=0.6, alpha=0.6)
    fig.suptitle("FormOxy paw intensity — first 60 min mean per dose (combined 2512 + 2605)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return per_sess


def main() -> int:
    import pandas as pd
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    df = collect()
    if df.empty:
        step("! no brightness data collected"); return 1

    long_csv = ANALYSIS / "paw_intensity_timecourse_5min.csv"
    df.to_csv(long_csv, index=False)
    step(f"Long-form: {long_csv}")

    tc_png = ANALYSIS / "paw_intensity_timecourse_5min.png"
    plot_timecourse(df, tc_png)
    step(f"Timecourse plot: {tc_png}")

    bar_png = ANALYSIS / "paw_intensity_dose_bars.png"
    per_sess = plot_dose_bars(df, bar_png)
    step(f"Dose bars: {bar_png}")

    sess_csv = ANALYSIS / "paw_intensity_per_session.csv"
    per_sess.to_csv(sess_csv, index=False)
    step(f"Per-session means: {sess_csv}")

    head = (
        "**FormOxy paw-intensity analysis (mean brightness in paw ROI)**\n"
        f"5-min timecourse + first-60-min dose bars. Combined cohorts 2512 + 2605, "
        f"n=6 per dose (3 from each cohort). "
        "ROI 20 px hind / 15 px fore, pixel threshold 101.\n"
        "L hind is the formalin-injected paw; R hind is the contralateral control. "
        "Higher brightness => paw lifted/exposed; the L-R log-ratio is the asymmetry biomarker."
    )
    discord_upload(head, [tc_png, bar_png, sess_csv])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
