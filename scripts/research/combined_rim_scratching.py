"""
Combine two rimonabant cohorts on scratching bouts:
  - 2510 cohort (Oct 2025): 12 males, post-rim only, doses from xlsx key
    (S1-3 = 10mg, S4-6 = 3mg, S7-9 = 1mg, S10-12 = Veh)
  - 2605 cohort (May 2026): 9 females, baseline + postrim, doses hardcoded
    in rim_dose_response.py (Rim1/5/9=VEH, 2/6=1mg, 3/7=3mg, 4/8=10mg)

Output:
  - combined_rim_scratching_summary.csv  (per-subject scratching summary)
  - combined_rim_scratching.png          (dose-response, both cohorts overlaid)
  - posts both to Discord
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WEBHOOK = (
    ""
)

# ----- 2510 cohort (males, post-rim only) ---------------------------------
COH2510_DIR = Path(r"E:\RSVIDS\Blackbox\2510_Blackbox_Rimonabant"
                   r"\Blackbox_videos-selected\2602_Rimonabant_cropped"
                   r"\Results\Scratching")
COH2510_DOSES = {
    1: "10 mg/kg", 2: "10 mg/kg", 3: "10 mg/kg",
    4: "3 mg/kg",  5: "3 mg/kg",  6: "3 mg/kg",
    7: "1 mg/kg",  8: "1 mg/kg",  9: "1 mg/kg",
    10: "VEH", 11: "VEH", 12: "VEH",
}

# ----- 2605 cohort (females, baseline + postrim) --------------------------
COH2605_CSV = Path(r"E:\RSVIDS\Blackbox\260515_Rim_DoseResp\analysis\rim_dose_response_summary.csv")

OUT_DIR = Path(r"E:\RSVIDS\Blackbox\260515_Rim_DoseResp\analysis")
OUT_CSV = OUT_DIR / "combined_rim_scratching_summary.csv"
OUT_PNG = OUT_DIR / "combined_rim_scratching.png"

DOSE_ORDER = ["VEH", "1 mg/kg", "3 mg/kg", "10 mg/kg"]
DOSE_X = {"VEH": 0, "1 mg/kg": 1, "3 mg/kg": 3, "10 mg/kg": 10}


def _bouts_from_predictions(pred_csv: Path, fps: int = 60) -> tuple[int, float, float]:
    """Return (n_bouts, total_scratch_s, mean_bout_s) from a per-frame
    predictions.csv (cols: frame, probability, <behavior>)."""
    df = pd.read_csv(pred_csv)
    behavior_col = [c for c in df.columns if c not in ("frame", "probability")][0]
    pred = df[behavior_col].to_numpy().astype(int)
    if len(pred) == 0:
        return 0, 0.0, 0.0
    # find transitions
    diff = np.diff(np.concatenate(([0], pred, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    bout_lens = (ends - starts).astype(float) / fps
    return int(len(starts)), float(bout_lens.sum()), float(bout_lens.mean() if len(bout_lens) else 0.0)


def load_2510() -> pd.DataFrame:
    rows = []
    pred_files = sorted(COH2510_DIR.glob("*_predictions.csv"))
    for pred in pred_files:
        m = re.match(r"(\d{6})_Rim_S(\d+)_cropped", pred.stem)
        if not m:
            continue
        subj = int(m.group(2))
        if subj not in COH2510_DOSES:
            continue
        # Frame count from predictions
        n_frames = len(pd.read_csv(pred))
        duration_min = n_frames / 60.0 / 60.0
        # Bouts: prefer existing bouts.csv, fall back to derive from predictions
        bouts_csv = pred.parent / pred.name.replace("_predictions.csv", "_bouts.csv")
        if bouts_csv.exists():
            df = pd.read_csv(bouts_csv)
            n_bouts = len(df)
            total_s = float(df["duration_sec"].sum()) if "duration_sec" in df.columns else 0.0
            mean_bout_s = float(df["duration_sec"].mean()) if n_bouts > 0 else 0.0
        else:
            n_bouts, total_s, mean_bout_s = _bouts_from_predictions(pred)
        rows.append({
            "cohort": "2510 (M)",
            "subject": f"S{subj}",
            "dose": COH2510_DOSES[subj],
            "n_bouts": n_bouts,
            "total_scratch_s": total_s,
            "mean_bout_s": mean_bout_s,
            "duration_min": duration_min,
            "bouts_per_min": (n_bouts / duration_min) if duration_min > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def load_2605_postrim() -> pd.DataFrame:
    df = pd.read_csv(COH2605_CSV)
    df = df[(df["behavior"] == "Scratching") & (df["condition"] == "postrim")]
    rows = []
    for _, r in df.iterrows():
        duration_min = r["n_frames"] / 60.0 / 60.0
        rows.append({
            "cohort": "2605 (F)",
            "subject": f"Rim{int(r['mouse'])}",
            "dose": r["dose"],
            "n_bouts": int(r["n_bouts"]),
            "total_scratch_s": float(r["total_time_s"]),
            "mean_bout_s": float(r["mean_bout_s"]),
            "duration_min": float(duration_min),
            "bouts_per_min": float(r["n_bouts"] / duration_min) if duration_min > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def plot_combined(df: pd.DataFrame, png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: scatter — bouts/min vs dose, by cohort
    ax = axes[0]
    palette = {"2510 (M)": "#1f77b4", "2605 (F)": "#d62728"}
    for cohort, sub in df.groupby("cohort"):
        for dose in DOSE_ORDER:
            ys = sub[sub["dose"] == dose]["bouts_per_min"].to_numpy()
            if len(ys) == 0:
                continue
            xs = DOSE_X[dose] + np.random.normal(0, 0.15, len(ys))
            ax.scatter(xs, ys, color=palette[cohort], alpha=0.75, s=55,
                       edgecolor="black", linewidth=0.5,
                       label=cohort if dose == "VEH" else None)
    # mean lines per cohort
    for cohort, sub in df.groupby("cohort"):
        means = [sub[sub["dose"] == d]["bouts_per_min"].mean() for d in DOSE_ORDER]
        xs = [DOSE_X[d] for d in DOSE_ORDER]
        ax.plot(xs, means, color=palette[cohort], linewidth=2, marker="o",
                alpha=0.7, markersize=8)
    ax.set_xticks([DOSE_X[d] for d in DOSE_ORDER])
    ax.set_xticklabels(DOSE_ORDER)
    ax.set_xlabel("Rimonabant dose")
    ax.set_ylabel("Scratching bouts / min")
    ax.set_title("Scratching bouts vs dose (per-subject + cohort mean)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best")

    # Panel 2: pooled mean +/- SEM by dose
    ax = axes[1]
    pooled = df.groupby("dose")["bouts_per_min"].agg(["mean", "sem", "count"])
    pooled = pooled.reindex(DOSE_ORDER)
    xs = np.arange(len(DOSE_ORDER))
    ax.bar(xs, pooled["mean"], yerr=pooled["sem"], capsize=4,
           color="#9467bd", alpha=0.8, edgecolor="black")
    for i, d in enumerate(DOSE_ORDER):
        ys = df[df["dose"] == d]["bouts_per_min"].to_numpy()
        ax.scatter(np.full(len(ys), i) + np.random.normal(0, 0.06, len(ys)),
                   ys, color="black", s=22, zorder=3)
        ax.text(i, pooled["mean"].iloc[i] + pooled["sem"].iloc[i] + 0.05,
                f"n={pooled['count'].iloc[i]:.0f}",
                ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(DOSE_ORDER)
    ax.set_xlabel("Rimonabant dose")
    ax.set_ylabel("Scratching bouts / min (pooled)")
    ax.set_title("Pooled across cohorts")
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: total scratch time
    ax = axes[2]
    pooled2 = df.groupby("dose")["total_scratch_s"].agg(["mean", "sem", "count"])
    pooled2 = pooled2.reindex(DOSE_ORDER)
    ax.bar(xs, pooled2["mean"], yerr=pooled2["sem"], capsize=4,
           color="#2ca02c", alpha=0.8, edgecolor="black")
    for i, d in enumerate(DOSE_ORDER):
        ys = df[df["dose"] == d]["total_scratch_s"].to_numpy()
        ax.scatter(np.full(len(ys), i) + np.random.normal(0, 0.06, len(ys)),
                   ys, color="black", s=22, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(DOSE_ORDER)
    ax.set_xlabel("Rimonabant dose")
    ax.set_ylabel("Total scratching time (s)")
    ax.set_title("Total scratching duration")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Rimonabant dose-response on scratching — 2510 (M, n=12) + 2605 (F, n=9)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    plt.close(fig)


def post_to_discord(text: str, png_paths: list[Path]) -> None:
    """Multipart upload with proper JSON encoding to avoid curl quoting woes."""
    import io
    import mimetypes
    import secrets
    import urllib.request

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
    for i, p in enumerate(png_paths):
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


def main() -> int:
    df_2510 = load_2510()
    df_2605 = load_2605_postrim()
    df = pd.concat([df_2510, df_2605], ignore_index=True)
    df = df.sort_values(["cohort", "dose", "subject"]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} ({len(df)} subjects)")

    print("\nPer-subject summary:")
    print(df.to_string(index=False))

    print("\nPooled by dose (bouts/min):")
    print(df.groupby("dose")["bouts_per_min"].agg(["mean", "sem", "count"])
          .reindex(DOSE_ORDER).to_string())

    plot_combined(df, OUT_PNG)
    print(f"Wrote {OUT_PNG}")

    # Discord summary
    pooled = df.groupby("dose")["bouts_per_min"].agg(["mean", "sem", "count"]).reindex(DOSE_ORDER)
    pooled_total = df.groupby("dose")["total_scratch_s"].agg(["mean", "sem"]).reindex(DOSE_ORDER)
    lines = ["**Combined Rim dose-response on scratching -- 2510 (M, n=12) + 2605 (F, n=9)**",
             "Cohort 2510 = males, postrim only; Cohort 2605 = females, postrim window cap 60 min",
             "",
             "Bouts/min (pooled across cohorts):"]
    for d in DOSE_ORDER:
        m = pooled.loc[d, "mean"]
        s = pooled.loc[d, "sem"]
        n = int(pooled.loc[d, "count"])
        lines.append(f"  {d:8} {m:5.2f} +/- {s:4.2f}  (n={n})")
    lines.append("")
    lines.append("Total scratching time (s):")
    for d in DOSE_ORDER:
        m = pooled_total.loc[d, "mean"]
        s = pooled_total.loc[d, "sem"]
        lines.append(f"  {d:8} {m:6.1f} +/- {s:5.1f}")
    summary = "\n".join(lines)
    post_to_discord(summary, [OUT_PNG])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
