"""
Baseline vs formalin HL/HR contour-intensity comparison for the 2605
cohort, plus the matching reference band added to the combined-cohort
formalin TC.

Question: oxycodone-treated animals never reach ratio=1.0 (true L/R
symmetry) in the formalin TC — even Oxy3 / Oxy10 hover at ~0.95-0.97.
Is the ~0.05 gap residual pain, or is it a constant setup asymmetry
(lighting / Otsu bias / camera angle) that we'd see even in healthy
no-pain animals?

The 2605 cohort has 12 pre-formalin baseline mp4s for the same animals.
This script reads their contour CSVs (produced by the baseline pipeline
orchestrator), computes per-session mean(intensities_HL) /
mean(intensities_HR), and presents three views:

  1. Per-dose paired bars: baseline (gray) vs formalin (dose color),
     within-animal lines. Tells you whether each dose's formalin ratio
     reached the animal's own baseline.

  2. Cohort-wide baseline distribution (n=12, dose-blind) -> a horizontal
     "no-pain ceiling" reference. If the median baseline ratio is also
     ~0.95, the 1.0 we'd expected was never the right reference and
     Oxy3/Oxy10 are at effective full reversal.

  3. Refreshed contour-intensity TC plot (combined 2512 + 2605 formalin)
     with the cohort baseline mean +/- 1 SD shaded as a horizontal band.

Posts both figures + the per-session CSV to #results.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
import uuid
from pathlib import Path

COHORT_NEW = Path(r"E:\RSVIDS\Blackbox\2605_FormOxy_LeftPaws")
NEW_DIR = COHORT_NEW / "gait_limb_analysis"
ANALYSIS = COHORT_NEW / "analysis"

COHORT_OLD = Path(r"E:\RSVIDS\Blackbox\2512_Blackbox_Formalin_Oxy\Left_paws")
OLD_DIR = COHORT_OLD / "gait_limb_analysis"

FPS = 60
BIN_MIN = 5
BIN_FRAMES = BIN_MIN * 60 * FPS
N_BINS = 12

DOSE_2512 = {1: "Veh", 2: "Veh", 3: "Veh",
             4: "Oxy3", 5: "Oxy3", 6: "Oxy3",
             7: "Oxy10", 8: "Oxy10", 9: "Oxy10",
             10: "Oxy1", 11: "Oxy1", 12: "Oxy1"}
DOSE_SEQ = ["Veh", "Oxy1", "Oxy3", "Oxy10"]


def dose_2605(n: int) -> str:
    return DOSE_SEQ[(n - 1) % 4]


DOSE_COLORS = {
    "Veh":   "#000000",
    "Oxy1":  "#a6cee3",
    "Oxy3":  "#1f78b4",
    "Oxy10": "#08306b",
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
                 "User-Agent": "PixelPaws-FormOxy-BaselineRatio/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            step(f"Discord upload: HTTP {r.status}")
    except Exception as e:
        step(f"Discord upload failed: {e}")


def session_ratio(df, cap_frames: int | None = None) -> tuple[float, int]:
    """Return (mean(HL)/mean(HR), n_valid_frames)."""
    import numpy as np
    hl = df["intensities_HL"].astype(float).to_numpy()
    hr = df["intensities_HR"].astype(float).to_numpy()
    if cap_frames is not None:
        hl = hl[:cap_frames]; hr = hr[:cap_frames]
    valid = (hl > 0) & (hr > 0) & np.isfinite(hl) & np.isfinite(hr)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return float("nan"), 0
    mh = float(np.mean(hl[valid])); mr = float(np.mean(hr[valid]))
    return (mh / mr if mr > 0 else float("nan")), n_valid


def collect_2605():
    """Per-session contour ratios for both conditions in the 2605 cohort."""
    import pandas as pd
    rx = re.compile(r"^2605_FormOxy_S(\d+)_(baseline|formalin)_contour_[0-9a-f]+\.csv$")
    rows = []
    for p in sorted(NEW_DIR.glob("2605_FormOxy_S*_contour_*.csv")):
        m = rx.match(p.name)
        if not m:
            continue
        subj = int(m.group(1)); cond = m.group(2)
        df = pd.read_csv(p, usecols=["intensities_HL", "intensities_HR"])
        cap = N_BINS * BIN_FRAMES if cond == "formalin" else None
        ratio, n = session_ratio(df, cap_frames=cap)
        rows.append({
            "cohort": "2605", "subject_id": f"2605_S{subj}",
            "mouse_num": subj, "dose": dose_2605(subj),
            "condition": cond, "ratio_HL_HR": ratio, "n_valid_frames": n,
        })
    return pd.DataFrame(rows)


def collect_2512_formalin():
    """Per-session formalin contour ratios for the 2512 cohort (no baselines)."""
    import pandas as pd
    rx = re.compile(r"^2512_FormOxy_S(\d+)_contour_[0-9a-f]+\.csv$")
    rows = []
    for p in sorted(OLD_DIR.glob("2512_FormOxy_S*_contour_*.csv")):
        m = rx.match(p.name)
        if not m:
            continue
        subj = int(m.group(1))
        df = pd.read_csv(p, usecols=["intensities_HL", "intensities_HR"])
        cap = N_BINS * BIN_FRAMES
        ratio, n = session_ratio(df, cap_frames=cap)
        rows.append({
            "cohort": "2512", "subject_id": f"2512_S{subj}",
            "mouse_num": subj, "dose": DOSE_2512[subj],
            "condition": "formalin", "ratio_HL_HR": ratio, "n_valid_frames": n,
        })
    return pd.DataFrame(rows)


def plot_baseline_vs_formalin(df_paired, baseline_stats, out_png: Path):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.6), sharey=True)
    for ax, dose in zip(axes, DOSE_SEQ):
        sub = df_paired[df_paired["dose"] == dose]
        bls = sub[sub["condition"] == "baseline"].set_index("mouse_num")
        fms = sub[sub["condition"] == "formalin"].set_index("mouse_num")
        mice = sorted(set(bls.index) & set(fms.index))
        bls_vals = [float(bls.loc[m, "ratio_HL_HR"]) for m in mice]
        fms_vals = [float(fms.loc[m, "ratio_HL_HR"]) for m in mice]

        x = [0, 1]
        means = [np.mean(bls_vals) if bls_vals else np.nan,
                 np.mean(fms_vals) if fms_vals else np.nan]
        sems = [np.std(bls_vals, ddof=1) / np.sqrt(len(bls_vals)) if len(bls_vals) > 1 else 0,
                np.std(fms_vals, ddof=1) / np.sqrt(len(fms_vals)) if len(fms_vals) > 1 else 0]
        c = DOSE_COLORS[dose]
        ax.bar(x, means, yerr=sems, capsize=4,
               color=["#7f7f7f", c], alpha=0.55,
               edgecolor="black", linewidth=0.6)

        # Per-mouse paired lines
        for m, bv, fv in zip(mice, bls_vals, fms_vals):
            ax.plot([0, 1], [bv, fv], color="black", alpha=0.45,
                    linewidth=0.9, marker="o", markersize=4)

        ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.axhline(baseline_stats["mean"], color="gray", linestyle=":",
                   linewidth=1.0, alpha=0.7)
        ax.set_xticks(x); ax.set_xticklabels(["baseline", "formalin"], fontsize=9)
        ax.set_title(f"{dose}  (n={len(mice)})", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Contour intensity ratio HL/HR")

    fig.suptitle(
        "2605 cohort: baseline vs formalin HL/HR contour intensity, per dose\n"
        f"Cohort-wide baseline (n={baseline_stats['n']}): "
        f"mean={baseline_stats['mean']:.3f} +/- SD {baseline_stats['sd']:.3f} "
        f"[median {baseline_stats['median']:.3f}, "
        f"range {baseline_stats['min']:.3f}-{baseline_stats['max']:.3f}]\n"
        "Red dashed = 1.0 (perfect L/R symmetry).  "
        "Gray dotted = cohort-wide baseline mean.",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_tc_with_baseline_band(formalin_df, baseline_stats, out_png: Path):
    """Re-render the contour ratio TC with a horizontal baseline band."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    # Group formalin sessions by dose + 5-min bin
    # Need to re-bin from the per-session CSVs to get per-bin means
    bin_rows = []
    rx_new = re.compile(r"^2605_FormOxy_S(\d+)_formalin_contour_[0-9a-f]+\.csv$")
    for p in sorted(NEW_DIR.glob("2605_FormOxy_S*_formalin_contour_*.csv")):
        m = rx_new.match(p.name)
        if not m:
            continue
        subj = int(m.group(1))
        df = pd.read_csv(p, usecols=["intensities_HL", "intensities_HR"])
        for b in range(N_BINS):
            s = b * BIN_FRAMES; e = s + BIN_FRAMES
            r, _ = session_ratio(df.iloc[s:e])
            bin_rows.append({"subject": f"2605_S{subj}", "dose": dose_2605(subj),
                             "bin_idx": b, "ratio": r})

    rx_old = re.compile(r"^2512_FormOxy_S(\d+)_contour_[0-9a-f]+\.csv$")
    for p in sorted(OLD_DIR.glob("2512_FormOxy_S*_contour_*.csv")):
        m = rx_old.match(p.name)
        if not m:
            continue
        subj = int(m.group(1))
        df = pd.read_csv(p, usecols=["intensities_HL", "intensities_HR"])
        for b in range(N_BINS):
            s = b * BIN_FRAMES; e = s + BIN_FRAMES
            r, _ = session_ratio(df.iloc[s:e])
            bin_rows.append({"subject": f"2512_S{subj}", "dose": DOSE_2512[subj],
                             "bin_idx": b, "ratio": r})
    bin_df = pd.DataFrame(bin_rows)

    fig, ax = plt.subplots(figsize=(11, 5.8))
    x_min = [i * BIN_MIN for i in range(N_BINS)]
    for dose in DOSE_SEQ:
        sub = bin_df[bin_df["dose"] == dose]
        means, sems = [], []
        for b in range(N_BINS):
            vals = sub.loc[sub["bin_idx"] == b, "ratio"].dropna().to_numpy()
            means.append(np.mean(vals) if len(vals) else np.nan)
            sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
        c = DOSE_COLORS[dose]
        ax.errorbar(x_min, means, yerr=sems, marker="o", color=c,
                    linewidth=1.6, markersize=5.5, capsize=3,
                    markerfacecolor=c, markeredgecolor=c, label=dose)

    # Symmetry line (1.0)
    ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.8,
               label="L=R symmetry (1.0)")

    # Baseline band: shade mean +/- 1 SD across the cohort baseline distribution
    bl_lo = baseline_stats["mean"] - baseline_stats["sd"]
    bl_hi = baseline_stats["mean"] + baseline_stats["sd"]
    ax.axhspan(bl_lo, bl_hi, color="#2ca02c", alpha=0.13,
               label=f"2605 baseline mean+/-SD (n={baseline_stats['n']})")
    ax.axhline(baseline_stats["mean"], color="#2ca02c", linestyle=":",
               linewidth=1.4, alpha=0.85)

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Contour intensity ratio HL/HR")
    ax.set_title("Contour Intensity Ratio TC -- with 2605 baseline reference band",
                 fontweight="bold")
    ax.set_xticks(x_min)
    ax.set_xticklabels([str(i * BIN_MIN) for i in range(N_BINS)])
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main() -> int:
    import numpy as np
    import pandas as pd
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    df_2605 = collect_2605()
    df_2512 = collect_2512_formalin()
    if df_2605.empty:
        step("! no 2605 contour CSVs found"); return 1

    bls = df_2605[df_2605["condition"] == "baseline"].dropna(subset=["ratio_HL_HR"])
    n_bl = len(bls)
    if n_bl == 0:
        step("! no 2605 baseline contour CSVs yet — orchestrator probably still running")
        return 1
    bl_vals = bls["ratio_HL_HR"].to_numpy()
    baseline_stats = {
        "n":      int(n_bl),
        "mean":   float(np.mean(bl_vals)),
        "sd":     float(np.std(bl_vals, ddof=1)) if n_bl > 1 else 0.0,
        "median": float(np.median(bl_vals)),
        "min":    float(np.min(bl_vals)),
        "max":    float(np.max(bl_vals)),
    }
    step(f"Baseline stats (n={n_bl}): {baseline_stats}")

    # Combined per-session CSV
    df_all = pd.concat([df_2605, df_2512], ignore_index=True)
    sess_csv = ANALYSIS / "baseline_vs_formalin_per_session.csv"
    df_all.to_csv(sess_csv, index=False)
    step(f"Per-session CSV: {sess_csv}")

    # Paired bars + scatter
    paired_png = ANALYSIS / "baseline_vs_formalin_per_dose.png"
    plot_baseline_vs_formalin(df_2605, baseline_stats, paired_png)
    step(f"Paired plot: {paired_png}")

    # TC with baseline reference band
    tc_png = ANALYSIS / "contour_intensity_ratio_tc_with_baseline.png"
    plot_tc_with_baseline_band(df_2512, baseline_stats, tc_png)
    step(f"TC plot: {tc_png}")

    # Per-dose post-hoc: does formalin differ from baseline within each dose?
    from scipy import stats
    posthoc_rows = []
    for dose in DOSE_SEQ:
        sub_bl = df_2605[(df_2605["dose"] == dose) & (df_2605["condition"] == "baseline")].dropna(subset=["ratio_HL_HR"])
        sub_fm = df_2605[(df_2605["dose"] == dose) & (df_2605["condition"] == "formalin")].dropna(subset=["ratio_HL_HR"])
        merged = sub_bl.set_index("mouse_num")["ratio_HL_HR"].rename("bl").to_frame().join(
                 sub_fm.set_index("mouse_num")["ratio_HL_HR"].rename("fm"))
        merged = merged.dropna()
        if len(merged) < 2:
            posthoc_rows.append({"dose": dose, "n": len(merged), "p_wilcoxon": float("nan")})
            continue
        try:
            stat_w, p_w = stats.wilcoxon(merged["bl"].to_numpy(), merged["fm"].to_numpy())
        except Exception:
            p_w = float("nan")
        posthoc_rows.append({
            "dose": dose, "n": int(len(merged)),
            "baseline_mean": round(float(merged["bl"].mean()), 4),
            "formalin_mean": round(float(merged["fm"].mean()), 4),
            "delta_mean":    round(float((merged["fm"] - merged["bl"]).mean()), 4),
            "p_wilcoxon":    round(float(p_w), 4),
        })
    posthoc_csv = ANALYSIS / "baseline_vs_formalin_posthoc.csv"
    pd.DataFrame(posthoc_rows).to_csv(posthoc_csv, index=False)
    step("Per-dose Wilcoxon (paired) baseline vs formalin:")
    step(pd.DataFrame(posthoc_rows).to_string(index=False))

    head = (
        "**FormOxy: baseline vs formalin HL/HR (2605 cohort)**\n"
        f"Cohort-wide baseline (n={baseline_stats['n']}): "
        f"mean={baseline_stats['mean']:.3f} +/- SD {baseline_stats['sd']:.3f} "
        f"(median {baseline_stats['median']:.3f}, "
        f"range {baseline_stats['min']:.3f}-{baseline_stats['max']:.3f}).\n"
        "If the baseline mean sits well below 1.0, the gap between 1.0 and "
        "the high-dose formalin ratio is mostly setup asymmetry (lighting / "
        "Otsu / camera), not residual pain.\n"
        "Per-dose Wilcoxon (paired baseline vs formalin) in the posthoc CSV; "
        "TC plot overlay shows the baseline mean+/-SD as a green band."
    )
    discord_upload(head, [paired_png, tc_png, sess_csv, posthoc_csv])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
