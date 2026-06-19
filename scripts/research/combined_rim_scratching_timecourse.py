"""
Per-subject scratching bout timecourse (5-min bins + cumulative) for the
combined rimonabant dose-response cohorts:

  2510 cohort (Oct 2025, males, postrim only)
     bouts.csv has start_frame -> use that
     S12 lacks bouts.csv -> derive from predictions.csv
     Doses from key xlsx (S1-3=10mg, S4-6=3mg, S7-9=1mg, S10-12=Veh)

  2605 cohort (May 2026, females, postrim window)
     predictions.csv has per-frame Scratching column -> derive bouts
     Doses hardcoded from rim_dose_response.py (Rim1/5/9=VEH,
     Rim2/6=1mg, Rim3/7=3mg, Rim4/8=10mg)

Outputs:
  rim_scratching_bouts_timecourse.csv  (subject x bin counts + cumulative)
  rim_scratching_bouts_per_subject.csv (total bouts per subject)
  rim_scratching_bouts_timecourse.png  (per-bin + cumulative, by dose)
  Posts to Discord.
"""
from __future__ import annotations

import glob
import io
import json
import mimetypes
import os
import re
import secrets
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WEBHOOK = (
    ""
)

FPS = 60
BIN_MIN = 5

# Window cap for the analysis -- option 1: target 60 min.
# Subjects shorter than this lose only their last partial bin.
WINDOW_MIN = 60

# Uniform post-processing applied to BOTH cohorts so bout counts are comparable.
# (The two cohorts' classifiers ship with different threshold/gap defaults --
# 2605 = 0.60 thresh, max_gap 100ms; 2510 = 0.90 thresh, max_gap 33ms -- which
# makes absolute counts incomparable. Apply a uniform rescore from probabilities.)
RESCORE_THRESH    = 0.80   # was 0.60 (2605) / 0.90 (2510)
RESCORE_MIN_BOUT  = 18     # 300 ms -- at least 3 scratch cycles at ~10 Hz
RESCORE_MAX_GAP   = 30     # 500 ms -- merge intra-event pauses

# 2510 cohort
COH2510_DIR = Path(r"E:\RSVIDS\Blackbox\2510_Blackbox_Rimonabant"
                   r"\Blackbox_videos-selected\2602_Rimonabant_cropped"
                   r"\Results\Scratching")
COH2510_DOSES = {
    1: "10 mg/kg", 2: "10 mg/kg", 3: "10 mg/kg",
    4: "3 mg/kg",  5: "3 mg/kg",  6: "3 mg/kg",
    7: "1 mg/kg",  8: "1 mg/kg",  9: "1 mg/kg",
    10: "VEH", 11: "VEH", 12: "VEH",
}

# 2605 cohort -- only the postrim recordings.
# Dose follows the modular formula used by rim_dose_response.py:
#   ["VEH", "1 mg/kg", "3 mg/kg", "10 mg/kg"][(n - 1) % 4]
COH2605_RESULTS = Path(r"E:\RSVIDS\Blackbox\260515_Rim_DoseResp\results")
_COH2605_SEQ = ["VEH", "1 mg/kg", "3 mg/kg", "10 mg/kg"]
COH2605_DOSES = {n: _COH2605_SEQ[(n - 1) % 4] for n in range(1, 100)}

DOSE_ORDER = ["VEH", "1 mg/kg", "3 mg/kg", "10 mg/kg"]
DOSE_COLOR = {"VEH": "#999999", "1 mg/kg": "#1f77b4",
              "3 mg/kg": "#ff7f0e", "10 mg/kg": "#d62728"}

OUT_DIR = Path(r"E:\RSVIDS\Blackbox\260515_Rim_DoseResp\analysis")
OUT_CSV_BINS = OUT_DIR / "rim_scratching_bouts_timecourse.csv"
OUT_CSV_TOTALS = OUT_DIR / "rim_scratching_bouts_per_subject.csv"
OUT_PNG = OUT_DIR / "rim_scratching_bouts_timecourse.png"
OUT_PNG_RATE = OUT_DIR / "rim_scratching_bouts_per_minute.png"


def _post_process(y_raw: np.ndarray, min_bout: int, max_gap: int) -> np.ndarray:
    """Merge gaps <= max_gap (between 1-bouts), drop bouts < min_bout."""
    y = y_raw.copy().astype(int)
    n = len(y)
    # 1) Close short gaps
    if max_gap > 0:
        i = 0
        while i < n:
            if y[i] == 0:
                j = i
                while j < n and y[j] == 0:
                    j += 1
                gap = j - i
                if 0 < gap <= max_gap and i > 0 and j < n and y[i - 1] == 1 and y[j] == 1:
                    y[i:j] = 1
                i = j
            else:
                i += 1
    # 2) Drop short bouts
    if min_bout > 1:
        i = 0
        while i < n:
            if y[i] == 1:
                j = i
                while j < n and y[j] == 1:
                    j += 1
                if (j - i) < min_bout:
                    y[i:j] = 0
                i = j
            else:
                i += 1
    return y


def derive_bouts_from_predictions(pred_csv: Path,
                                  thresh: float = RESCORE_THRESH,
                                  min_bout: int = RESCORE_MIN_BOUT,
                                  max_gap: int = RESCORE_MAX_GAP) -> np.ndarray:
    """Rescore per-frame probabilities at uniform (thresh, min_bout, max_gap)
    and return bout-start frames."""
    df = pd.read_csv(pred_csv)
    # The 2605 cohort columns are 'Scratching_proba' + 'Scratching'.
    # The 2510 cohort columns are 'frame', 'probability', '<Behavior>'.
    if "Scratching_proba" in df.columns:
        proba = df["Scratching_proba"].to_numpy()
    elif "probability" in df.columns:
        proba = df["probability"].to_numpy()
    else:
        raise RuntimeError(f"No probability column in {pred_csv}; "
                           f"columns={list(df.columns)}")
    y_raw = (proba >= thresh).astype(int)
    y = _post_process(y_raw, min_bout, max_gap)
    diff = np.diff(np.concatenate(([0], y, [0])))
    starts = np.where(diff == 1)[0]
    return starts


def load_2510_subjects() -> list[dict]:
    """Return per-subject dicts. Always rescores from probabilities so the
    post-processing matches the 2605 cohort exactly."""
    subjects = []
    for pred in sorted(COH2510_DIR.glob("*_predictions.csv")):
        m = re.match(r"(\d{6})_Rim_S(\d+)_cropped", pred.stem)
        if not m:
            continue
        subj = int(m.group(2))
        if subj not in COH2510_DOSES:
            continue
        n_frames = len(pd.read_csv(pred))
        starts = derive_bouts_from_predictions(pred)
        subjects.append({
            "cohort": "2510 (M)",
            "subject": f"S{subj}",
            "dose": COH2510_DOSES[subj],
            "n_frames": n_frames,
            "duration_min": n_frames / FPS / 60.0,
            "bout_start_frames": starts.astype(int),
        })
    return subjects


def load_2605_subjects() -> list[dict]:
    subjects = []
    for pred in sorted(COH2605_RESULTS.glob("fem_postrim_Rim*_predictions.csv")):
        m = re.match(r"fem_postrim_Rim(\d+)", pred.stem)
        if not m:
            continue
        subj = int(m.group(1))
        if subj not in COH2605_DOSES:
            continue
        df = pd.read_csv(pred)
        n_frames = len(df)
        # The combined predictions CSV has Scratching binary column
        starts = derive_bouts_from_predictions(pred)
        subjects.append({
            "cohort": "2605 (F)",
            "subject": f"Rim{subj}",
            "dose": COH2605_DOSES[subj],
            "n_frames": n_frames,
            "duration_min": n_frames / FPS / 60.0,
            "bout_start_frames": starts.astype(int),
        })
    return subjects


def bin_counts(start_frames: np.ndarray, n_bins: int,
               bin_frames: int) -> np.ndarray:
    """Number of bouts whose START falls in each 5-min bin."""
    counts = np.zeros(n_bins, dtype=int)
    if len(start_frames):
        idx = (start_frames // bin_frames).astype(int)
        idx = idx[idx < n_bins]
        vals, cnt = np.unique(idx, return_counts=True)
        counts[vals] = cnt
    return counts


def main() -> int:
    subjects = load_2510_subjects() + load_2605_subjects()

    # Target window: 60 min. Subjects that fall a few seconds short still
    # contribute to bin 11 (they just have a slightly shorter denominator).
    n_bins = WINDOW_MIN // BIN_MIN  # 12
    bin_frames = BIN_MIN * 60 * FPS
    target_frames = WINDOW_MIN * 60 * FPS
    durations = [s["duration_min"] for s in subjects]
    print(f"Subjects: {len(subjects)}  durations(min): "
          f"min={min(durations):.1f}, max={max(durations):.1f}")
    print(f"Target window: {WINDOW_MIN} min ({n_bins} x {BIN_MIN}-min bins). "
          f"Rescore: thresh={RESCORE_THRESH} min_bout={RESCORE_MIN_BOUT}fr "
          f"max_gap={RESCORE_MAX_GAP}fr")

    rows_bins = []
    rows_totals = []
    for s in subjects:
        # Cap bouts to first 60 min (or end of recording if shorter)
        cap_frames = min(target_frames, s["n_frames"])
        starts_cap = s["bout_start_frames"][s["bout_start_frames"] < cap_frames]
        counts = bin_counts(starts_cap, n_bins, bin_frames)
        cum = np.cumsum(counts)
        for b, (c, cm) in enumerate(zip(counts, cum)):
            # Mark the last bin as partial if the subject's recording cut short
            bin_start_fr = b * bin_frames
            bin_end_fr = min((b + 1) * bin_frames, cap_frames)
            partial = (bin_end_fr - bin_start_fr) < bin_frames
            rows_bins.append({
                "cohort": s["cohort"],
                "subject": s["subject"],
                "dose": s["dose"],
                "bin_idx": b,
                "bin_start_min": b * BIN_MIN,
                "bin_end_min": (b + 1) * BIN_MIN,
                "n_bouts_in_bin": int(c),
                "cumulative_bouts": int(cm),
                "bin_duration_min": round((bin_end_fr - bin_start_fr) / FPS / 60.0, 2),
                "partial_bin": partial,
            })
        eff_min = cap_frames / FPS / 60.0
        rows_totals.append({
            "cohort": s["cohort"],
            "subject": s["subject"],
            "dose": s["dose"],
            "duration_min": round(s["duration_min"], 2),
            "effective_window_min": round(eff_min, 2),
            "n_bouts_full": int(len(s["bout_start_frames"])),
            "n_bouts_capped": int(len(starts_cap)),
            "bouts_per_min_capped": round(len(starts_cap) / eff_min, 3) if eff_min > 0 else 0.0,
        })

    df_bins = pd.DataFrame(rows_bins)
    df_totals = pd.DataFrame(rows_totals)
    df_bins.to_csv(OUT_CSV_BINS, index=False)
    df_totals.to_csv(OUT_CSV_TOTALS, index=False)
    print(f"Wrote {OUT_CSV_BINS}")
    print(f"Wrote {OUT_CSV_TOTALS}")
    print("\nPer-subject totals (capped to common window):")
    print(df_totals.sort_values(["dose", "cohort", "subject"]).to_string(index=False))

    # ------ Plot --------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    bin_centers = np.arange(n_bins) * BIN_MIN + BIN_MIN / 2.0

    # Panel A: per-bin bouts, mean +/- SEM by dose, lines
    ax = axes[0, 0]
    for dose in DOSE_ORDER:
        sub = df_bins[df_bins["dose"] == dose]
        if sub.empty:
            continue
        per_bin = sub.groupby("bin_idx")["n_bouts_in_bin"].agg(["mean", "sem", "count"])
        per_bin = per_bin.reindex(range(n_bins)).fillna(0)
        ax.plot(bin_centers, per_bin["mean"], "-o", color=DOSE_COLOR[dose],
                label=f"{dose} (n={int(per_bin['count'].max())})",
                linewidth=2, markersize=5)
        ax.fill_between(bin_centers,
                        per_bin["mean"] - per_bin["sem"],
                        per_bin["mean"] + per_bin["sem"],
                        color=DOSE_COLOR[dose], alpha=0.2)
    ax.set_xlabel("Time post-rim (min)")
    ax.set_ylabel("Scratching bouts per 5-min bin")
    ax.set_title("Per-bin bout rate (mean ± SEM)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    # Panel B: cumulative bouts, mean +/- SEM by dose
    ax = axes[0, 1]
    for dose in DOSE_ORDER:
        sub = df_bins[df_bins["dose"] == dose]
        if sub.empty:
            continue
        per_bin = sub.groupby("bin_idx")["cumulative_bouts"].agg(["mean", "sem", "count"])
        per_bin = per_bin.reindex(range(n_bins)).fillna(0)
        x_edge = np.arange(n_bins + 1) * BIN_MIN  # cumulative shown at bin END
        y = np.concatenate(([0], per_bin["mean"].values))
        sem = np.concatenate(([0], per_bin["sem"].values))
        ax.plot(x_edge, y, "-o", color=DOSE_COLOR[dose],
                label=f"{dose} (n={int(per_bin['count'].max())})",
                linewidth=2, markersize=5)
        ax.fill_between(x_edge, y - sem, y + sem, color=DOSE_COLOR[dose], alpha=0.2)
    ax.set_xlabel("Time post-rim (min)")
    ax.set_ylabel("Cumulative scratching bouts")
    ax.set_title("Cumulative bouts (mean ± SEM)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    # Panel C: per-subject cumulative curves (faceted by cohort marker)
    ax = axes[1, 0]
    for s in subjects:
        cap_frames = n_bins * bin_frames
        starts_cap = s["bout_start_frames"][s["bout_start_frames"] < cap_frames]
        counts = bin_counts(starts_cap, n_bins, bin_frames)
        x_edge = np.arange(n_bins + 1) * BIN_MIN
        y = np.concatenate(([0], np.cumsum(counts)))
        ls = "-" if "2605" in s["cohort"] else "--"
        ax.plot(x_edge, y, ls, color=DOSE_COLOR[s["dose"]], alpha=0.7,
                linewidth=1.4, label=None)
    # Custom legend entries
    handles = [plt.Line2D([0], [0], color=DOSE_COLOR[d], linewidth=2, label=d)
               for d in DOSE_ORDER]
    handles += [plt.Line2D([0], [0], color="black", linestyle="-",
                            label="2605 (F)"),
                plt.Line2D([0], [0], color="black", linestyle="--",
                            label="2510 (M)")]
    ax.set_xlabel("Time post-rim (min)")
    ax.set_ylabel("Cumulative scratching bouts")
    ax.set_title("Per-subject cumulative curves")
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel D: total bouts per subject bar
    ax = axes[1, 1]
    df_sorted = df_totals.sort_values(["dose", "cohort", "subject"]).reset_index(drop=True)
    df_sorted["dose_rank"] = df_sorted["dose"].map({d: i for i, d in enumerate(DOSE_ORDER)})
    df_sorted = df_sorted.sort_values(["dose_rank", "cohort", "subject"]).reset_index(drop=True)
    xs = np.arange(len(df_sorted))
    colors = [DOSE_COLOR[d] for d in df_sorted["dose"]]
    ax.bar(xs, df_sorted["n_bouts_capped"], color=colors, edgecolor="black",
           linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(df_sorted["subject"], rotation=60, ha="right", fontsize=8)
    # Mark cohort under labels with secondary text
    for i, row in df_sorted.iterrows():
        ax.text(i, -0.06 * df_sorted["n_bouts_capped"].max(),
                row["cohort"].split()[1].strip("()"),
                ha="center", va="top", fontsize=7, color="dimgray",
                transform=ax.transData)
    ax.set_ylabel(f"Total bouts in {n_bins*BIN_MIN}-min window")
    ax.set_title(f"Total scratching bouts per subject (first {n_bins*BIN_MIN} min)")
    ax.grid(axis="y", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=DOSE_COLOR[d], label=d) for d in DOSE_ORDER]
    ax.legend(handles=handles, loc="upper left", fontsize=8)

    fig.suptitle(f"Rim dose-response on scratching bouts — combined cohorts "
                 f"(2510 M + 2605 F, first {n_bins*BIN_MIN} min postrim, "
                 f"rescored thresh={RESCORE_THRESH} min_bout={RESCORE_MIN_BOUT}fr "
                 f"max_gap={RESCORE_MAX_GAP}fr)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print(f"\nWrote {OUT_PNG}")

    # ------ Bouts-per-minute focused plot -------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    # Panel A: bouts/min as a function of time, mean +/- SEM by dose
    ax = axes2[0]
    for dose in DOSE_ORDER:
        sub = df_bins[df_bins["dose"] == dose]
        if sub.empty:
            continue
        # Rate = bouts_in_bin / 5 min
        sub = sub.copy()
        sub["rate"] = sub["n_bouts_in_bin"] / BIN_MIN
        per_bin = sub.groupby("bin_idx")["rate"].agg(["mean", "sem", "count"])
        per_bin = per_bin.reindex(range(n_bins)).fillna(0)
        ax.plot(bin_centers, per_bin["mean"], "-o", color=DOSE_COLOR[dose],
                label=f"{dose} (n={int(per_bin['count'].max())})",
                linewidth=2, markersize=6)
        ax.fill_between(bin_centers,
                        per_bin["mean"] - per_bin["sem"],
                        per_bin["mean"] + per_bin["sem"],
                        color=DOSE_COLOR[dose], alpha=0.2)
    ax.set_xlabel("Time post-rim (min)")
    ax.set_ylabel("Scratching bouts per minute")
    ax.set_title("Scratching rate over time (mean ± SEM by dose)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    # Panel B: window-averaged bouts/min by dose, pooled
    ax = axes2[1]
    pooled = df_totals.groupby("dose")["bouts_per_min_capped"].agg(["mean", "sem", "count"])
    pooled = pooled.reindex(DOSE_ORDER)
    xs = np.arange(len(DOSE_ORDER))
    ax.bar(xs, pooled["mean"], yerr=pooled["sem"], capsize=4,
           color=[DOSE_COLOR[d] for d in DOSE_ORDER],
           alpha=0.85, edgecolor="black")
    for i, d in enumerate(DOSE_ORDER):
        ys = df_totals[df_totals["dose"] == d]["bouts_per_min_capped"].to_numpy()
        ax.scatter(np.full(len(ys), i) + np.random.normal(0, 0.06, len(ys)),
                   ys, color="black", s=25, zorder=3)
        ax.text(i, pooled["mean"].iloc[i] + pooled["sem"].iloc[i] + 0.08,
                f"n={int(pooled['count'].iloc[i])}",
                ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(DOSE_ORDER)
    ax.set_xlabel("Rimonabant dose")
    ax.set_ylabel(f"Mean scratching bouts/min over {n_bins*BIN_MIN} min")
    ax.set_title("Window-averaged scratching rate by dose (pooled)")
    ax.grid(axis="y", alpha=0.3)

    fig2.suptitle(f"Scratching rate (bouts/min) — combined cohorts "
                  f"(2510 M + 2605 F, postrim, rescored "
                  f"thresh={RESCORE_THRESH} min_bout={RESCORE_MIN_BOUT}fr "
                  f"max_gap={RESCORE_MAX_GAP}fr)",
                  fontsize=11, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(OUT_PNG_RATE, dpi=140)
    plt.close(fig2)
    print(f"Wrote {OUT_PNG_RATE}")

    # ------ Pooled summary for Discord ----------------------------------------
    pooled_totals = df_totals.groupby("dose")["n_bouts_capped"].agg(["mean", "sem", "count"])
    pooled_totals = pooled_totals.reindex(DOSE_ORDER)
    pooled_per_min = df_totals.groupby("dose")["bouts_per_min_capped"].agg(["mean", "sem"])
    pooled_per_min = pooled_per_min.reindex(DOSE_ORDER)

    lines = [f"**Rim scratching bouts -- timecourse + totals (rescored)**",
             f"Window: first {n_bins*BIN_MIN} min postrim",
             f"Uniform post-processing: thresh={RESCORE_THRESH}, "
             f"min_bout={RESCORE_MIN_BOUT}fr ({RESCORE_MIN_BOUT*1000//FPS}ms), "
             f"max_gap={RESCORE_MAX_GAP}fr ({RESCORE_MAX_GAP*1000//FPS}ms)",
             f"Subjects: {len(df_totals)} total "
             f"({(df_totals['cohort']=='2510 (M)').sum()} M from 2510, "
             f"{(df_totals['cohort']=='2605 (F)').sum()} F from 2605)",
             "",
             f"Total bouts in {n_bins*BIN_MIN}-min window (mean +/- SEM, pooled):"]
    for d in DOSE_ORDER:
        m = pooled_totals.loc[d, "mean"]
        s = pooled_totals.loc[d, "sem"]
        n = int(pooled_totals.loc[d, "count"])
        rate = pooled_per_min.loc[d, "mean"]
        rate_se = pooled_per_min.loc[d, "sem"]
        lines.append(f"  {d:8} {m:6.1f} +/- {s:5.1f}  ({rate:.2f} +/- {rate_se:.2f} bouts/min, n={n})")

    summary = "\n".join(lines)
    post_discord(summary, [OUT_PNG, OUT_PNG_RATE])
    return 0


def post_discord(text: str, pngs: list[Path]) -> None:
    boundary = secrets.token_hex(16)
    body = io.BytesIO()

    def add_field(name: str, value: str) -> None:
        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.write(value.encode("utf-8"))
        body.write(b"\r\n")

    def add_file(name: str, path: Path) -> None:
        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'.encode())
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body.write(f"Content-Type: {ctype}\r\n\r\n".encode())
        body.write(path.read_bytes())
        body.write(b"\r\n")

    add_field("payload_json", json.dumps({"content": text}))
    for i, p in enumerate(pngs):
        add_file(f"files[{i}]", p)
    body.write(f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        WEBHOOK,
        data=body.getvalue(),
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "User-Agent": "PixelPaws-Chain/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            print(f"Discord upload: HTTP {r.status}")
    except Exception as e:
        print(f"Discord upload failed: {e}")


if __name__ == "__main__":
    raise SystemExit(main())
