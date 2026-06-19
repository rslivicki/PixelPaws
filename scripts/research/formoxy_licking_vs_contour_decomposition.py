"""
Decompose the contour HL/HR intensity ratio by licking state.

Question: is the contour intensity ratio just a proxy for licking?
When the mouse licks the formalin paw the HL paw is lifted to the
snout and partially occluded; both lower the Otsu contour intensity.
Since licking follows the dose curve almost perfectly, the dose effect
on the contour ratio could simply re-state the licking effect.

For each of the 24 formalin sessions (both cohorts), join the per-frame
Left_licking prediction CSV with the per-frame contour CSV (alignment
verified: row counts match exactly), cap at the first 60 min, and
compute three per-session ratios:
  - all:      mean(HL[valid]) / mean(HR[valid])
  - lick:     same but on frames where Left_licking == 1
  - nolick:   same but on frames where Left_licking == 0

Three figures + a per-session CSV posted to #results.
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
NEW_LICK = COHORT_NEW / "results"
NEW_CTR = COHORT_NEW / "gait_limb_analysis"
ANALYSIS = COHORT_NEW / "analysis"

COHORT_OLD = Path(r"E:\RSVIDS\Blackbox\2512_Blackbox_Formalin_Oxy\Left_paws")
OLD_LICK = COHORT_OLD / "Results" / "Left_licking"
OLD_CTR = COHORT_OLD / "gait_limb_analysis"

FPS = 60
BIN_MIN = 5
BIN_FRAMES = BIN_MIN * 60 * FPS
N_BINS = 12
WINDOW_FRAMES = N_BINS * BIN_FRAMES  # 216 000

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
                 "User-Agent": "PixelPaws-FormOxy-LickContourDecomp/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            step(f"Discord upload: HTTP {r.status}")
    except Exception as e:
        step(f"Discord upload failed: {e}")


# --------------------------------------------------------------------- #
# Per-session pull + decomposition
# --------------------------------------------------------------------- #
def ratio_on(hl, hr, mask):
    """mean(HL) / mean(HR) over rows where mask is True AND both
    contour intensities are valid."""
    import numpy as np
    valid = mask & (hl > 0) & (hr > 0) & np.isfinite(hl) & np.isfinite(hr)
    n = int(valid.sum())
    if n == 0:
        return float("nan"), 0
    mh = float(np.mean(hl[valid])); mr = float(np.mean(hr[valid]))
    return (mh / mr if mr > 0 else float("nan")), n


def per_session(lick_csv: Path, ctr_csv: Path, cohort: str, subj: int, dose: str):
    import numpy as np
    import pandas as pd
    df_lick = pd.read_csv(lick_csv, usecols=["Left_licking"])
    df_ctr  = pd.read_csv(ctr_csv,  usecols=["intensities_HL", "intensities_HR"])
    n_lick = len(df_lick); n_ctr = len(df_ctr)
    n = min(n_lick, n_ctr, WINDOW_FRAMES)
    if n_lick != n_ctr:
        step(f"  ! row-count mismatch: lick={n_lick} ctr={n_ctr}; capping to "
             f"min(min,WINDOW)={n}")

    y    = df_lick["Left_licking"].astype(int).to_numpy()[:n]
    hl   = df_ctr["intensities_HL"].astype(float).to_numpy()[:n]
    hr   = df_ctr["intensities_HR"].astype(float).to_numpy()[:n]
    lick_mask = (y == 1)
    nolk_mask = (y == 0)

    r_all,  n_all  = ratio_on(hl, hr, np.ones(n, dtype=bool))
    r_lick, n_lk   = ratio_on(hl, hr, lick_mask)
    r_nolk, n_nlk  = ratio_on(hl, hr, nolk_mask)

    pct_licking = 100.0 * int(lick_mask.sum()) / n if n else float("nan")

    return {
        "cohort": cohort, "subject_id": f"{cohort}_S{subj}",
        "mouse_num": subj, "dose": dose,
        "n_frames_capped": n,
        "pct_licking": pct_licking,
        "n_lick_frames":   int(lick_mask.sum()),
        "n_nolk_frames":   int(nolk_mask.sum()),
        "ratio_all":     r_all,
        "n_valid_all":   n_all,
        "ratio_lick":    r_lick,
        "n_valid_lick":  n_lk,
        "ratio_nolk":    r_nolk,
        "n_valid_nolk":  n_nlk,
    }


def collect():
    import pandas as pd
    rows = []

    # 2605
    rx_l = re.compile(r"^2605_FormOxy_S(\d+)_formalin_Left_licking_predictions\.csv$")
    rx_c = re.compile(r"^2605_FormOxy_S(\d+)_formalin_contour_[0-9a-f]+\.csv$")
    ctr_lookup = {}
    for p in NEW_CTR.glob("2605_FormOxy_S*_formalin_contour_*.csv"):
        m = rx_c.match(p.name)
        if m: ctr_lookup[int(m.group(1))] = p
    for p in sorted(NEW_LICK.glob("2605_FormOxy_S*_formalin_Left_licking_predictions.csv")):
        m = rx_l.match(p.name)
        if not m: continue
        subj = int(m.group(1))
        ctr = ctr_lookup.get(subj)
        if not ctr:
            step(f"  ! no 2605 contour for S{subj}; skip"); continue
        step(f"  2605 S{subj} ({dose_2605(subj)})")
        rows.append(per_session(p, ctr, "2605", subj, dose_2605(subj)))

    # 2512
    rx_l = re.compile(r"^2512_FormOxy_S(\d+)_PixelPaws_Left_licking_predictions\.csv$")
    rx_c = re.compile(r"^2512_FormOxy_S(\d+)_contour_[0-9a-f]+\.csv$")
    ctr_lookup = {}
    for p in OLD_CTR.glob("2512_FormOxy_S*_contour_*.csv"):
        m = rx_c.match(p.name)
        if m: ctr_lookup[int(m.group(1))] = p
    for p in sorted(OLD_LICK.glob("2512_FormOxy_S*_PixelPaws_Left_licking_predictions.csv")):
        m = rx_l.match(p.name)
        if not m: continue
        subj = int(m.group(1))
        ctr = ctr_lookup.get(subj)
        if not ctr:
            step(f"  ! no 2512 contour for S{subj}; skip"); continue
        step(f"  2512 S{subj} ({DOSE_2512[subj]})")
        rows.append(per_session(p, ctr, "2512", subj, DOSE_2512[subj]))

    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# 5-min-bin TC, conditional on licking state
# --------------------------------------------------------------------- #
def per_session_tc(lick_csv: Path, ctr_csv: Path):
    """Per-bin ratio on (licking) and (non-licking) subsets."""
    import numpy as np
    import pandas as pd
    df_lick = pd.read_csv(lick_csv, usecols=["Left_licking"])
    df_ctr  = pd.read_csv(ctr_csv,  usecols=["intensities_HL", "intensities_HR"])
    n = min(len(df_lick), len(df_ctr), WINDOW_FRAMES)
    y  = df_lick["Left_licking"].astype(int).to_numpy()[:n]
    hl = df_ctr["intensities_HL"].astype(float).to_numpy()[:n]
    hr = df_ctr["intensities_HR"].astype(float).to_numpy()[:n]

    out = []
    for b in range(N_BINS):
        s = b * BIN_FRAMES; e = min(s + BIN_FRAMES, n)
        if e <= s:
            out.append({"bin_idx": b, "ratio_lick": float("nan"),
                        "ratio_nolk": float("nan"),
                        "n_lick": 0, "n_nolk": 0})
            continue
        y_b = y[s:e]; hl_b = hl[s:e]; hr_b = hr[s:e]
        r_lk, n_lk = ratio_on(hl_b, hr_b, (y_b == 1))
        r_nl, n_nl = ratio_on(hl_b, hr_b, (y_b == 0))
        out.append({"bin_idx": b, "ratio_lick": r_lk, "ratio_nolk": r_nl,
                    "n_lick": n_lk, "n_nolk": n_nl})
    return out


def collect_tc():
    import pandas as pd
    rows = []

    rx_l = re.compile(r"^2605_FormOxy_S(\d+)_formalin_Left_licking_predictions\.csv$")
    rx_c = re.compile(r"^2605_FormOxy_S(\d+)_formalin_contour_[0-9a-f]+\.csv$")
    ctr_lookup = {int(rx_c.match(p.name).group(1)): p
                  for p in NEW_CTR.glob("2605_FormOxy_S*_formalin_contour_*.csv")
                  if rx_c.match(p.name)}
    for p in sorted(NEW_LICK.glob("2605_FormOxy_S*_formalin_Left_licking_predictions.csv")):
        m = rx_l.match(p.name); subj = int(m.group(1))
        ctr = ctr_lookup.get(subj)
        if not ctr: continue
        for b in per_session_tc(p, ctr):
            rows.append({"cohort": "2605", "subject_id": f"2605_S{subj}",
                         "dose": dose_2605(subj), **b})

    rx_l = re.compile(r"^2512_FormOxy_S(\d+)_PixelPaws_Left_licking_predictions\.csv$")
    rx_c = re.compile(r"^2512_FormOxy_S(\d+)_contour_[0-9a-f]+\.csv$")
    ctr_lookup = {int(rx_c.match(p.name).group(1)): p
                  for p in OLD_CTR.glob("2512_FormOxy_S*_contour_*.csv")
                  if rx_c.match(p.name)}
    for p in sorted(OLD_LICK.glob("2512_FormOxy_S*_PixelPaws_Left_licking_predictions.csv")):
        m = rx_l.match(p.name); subj = int(m.group(1))
        ctr = ctr_lookup.get(subj)
        if not ctr: continue
        for b in per_session_tc(p, ctr):
            rows.append({"cohort": "2512", "subject_id": f"2512_S{subj}",
                         "dose": DOSE_2512[subj], **b})

    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------- #
def plot_scatter(df, out_png: Path):
    import numpy as np
    from scipy import stats as sp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    for dose in DOSE_SEQ:
        sub = df[df["dose"] == dose]
        for cohort, marker in [("2512", "o"), ("2605", "s")]:
            ss = sub[sub["cohort"] == cohort]
            ax.scatter(ss["pct_licking"], ss["ratio_all"],
                       color=DOSE_COLORS[dose], marker=marker, s=70,
                       edgecolors="white", linewidths=0.8, zorder=3,
                       label=f"{dose} ({cohort})")

    # Pearson r on the all-frame ratio vs licking %
    sub = df.dropna(subset=["pct_licking", "ratio_all"])
    r, p = sp.pearsonr(sub["pct_licking"].to_numpy(), sub["ratio_all"].to_numpy())
    r2 = r ** 2

    # Best-fit line
    x = sub["pct_licking"].to_numpy(); y = sub["ratio_all"].to_numpy()
    if len(x) > 1:
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(0, max(x.max(), 1.0) * 1.05, 50)
        ax.plot(xs, np.polyval(coef, xs), color="gray", linestyle="--",
                linewidth=1.0, alpha=0.7)

    ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("% time licking (Left_licking classifier)")
    ax.set_ylabel("HL/HR contour intensity ratio (all frames)")
    ax.set_title(f"Per-session: licking % vs contour ratio\n"
                 f"Pearson r = {r:.3f}  (r² = {r2:.3f})  p = {p:.3g}, "
                 f"n = {len(sub)} sessions",
                 fontweight="bold")
    # De-duplicate legend by dose only (markers per cohort already visible)
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        key = l.split()[0]
        if key not in seen:
            seen[key] = h
    from matplotlib.lines import Line2D
    extra = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="white", label="2512 (circle)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
               markeredgecolor="white", label="2605 (square)"),
    ]
    ax.legend(list(seen.values()) + extra, list(seen.keys()) + ["2512", "2605"],
              fontsize=8, loc="best", frameon=True)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140); plt.close(fig)
    return {"r": float(r), "r2": float(r2), "p": float(p), "n": int(len(sub))}


def plot_dose_bars(df, out_png: Path):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0), sharey=True)
    panels = [("ratio_all",  "All frames"),
              ("ratio_nolk", "Non-licking frames"),
              ("ratio_lick", "Licking frames")]

    for ax, (key, title) in zip(axes, panels):
        for i, dose in enumerate(DOSE_SEQ):
            vals = df.loc[df["dose"] == dose, key].dropna().astype(float).to_numpy()
            mean = np.mean(vals) if len(vals) else np.nan
            sem  = (np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            ax.bar(i, mean, yerr=sem, capsize=4,
                   color=DOSE_COLORS[dose], alpha=0.55,
                   edgecolor="black", linewidth=0.6)
            # Per-mouse scatter (cohort marker)
            sub = df[df["dose"] == dose]
            for cohort, marker in [("2512", "o"), ("2605", "s")]:
                pts = sub.loc[sub["cohort"] == cohort, key].dropna().to_numpy()
                if len(pts) == 0:
                    continue
                xs = np.random.normal(i, 0.06, len(pts))
                ax.scatter(xs, pts, color="black", marker=marker, s=22,
                           edgecolors="white", linewidths=0.5, zorder=3)
        ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_xticks(range(len(DOSE_SEQ)))
        ax.set_xticklabels(DOSE_SEQ, fontsize=9)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Contour intensity ratio HL/HR")
    fig.suptitle("Contour HL/HR ratio decomposed by licking state — combined 2512 + 2605, n=6/dose",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=140); plt.close(fig)


def plot_tc_split(df_tc, out_png: Path):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_min = [i * BIN_MIN for i in range(N_BINS)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    panels = [("ratio_nolk", "Non-licking frames only"),
              ("ratio_lick", "Licking frames only")]

    for ax, (key, title) in zip(axes, panels):
        for dose in DOSE_SEQ:
            sub = df_tc[df_tc["dose"] == dose]
            means, sems = [], []
            for b in range(N_BINS):
                vals = sub.loc[sub["bin_idx"] == b, key].dropna().astype(float).to_numpy()
                means.append(np.mean(vals) if len(vals) else np.nan)
                sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
            c = DOSE_COLORS[dose]
            ax.errorbar(x_min, means, yerr=sems, marker="o", color=c,
                        linewidth=1.6, markersize=5.5, capsize=3,
                        markerfacecolor=c, markeredgecolor=c, label=dose)
        ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_xlabel("Time (min)"); ax.set_xticks(x_min)
        ax.set_xticklabels([str(i * BIN_MIN) for i in range(N_BINS)])
        ax.set_title(title, fontweight="bold")
        ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8)
    axes[0].set_ylabel("Contour intensity ratio HL/HR")
    fig.suptitle("Contour intensity ratio TC, split by licking state (5-min bins)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=140); plt.close(fig)


# --------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------- #
def main() -> int:
    import pandas as pd
    ANALYSIS.mkdir(parents=True, exist_ok=True)

    step("Collecting per-session data...")
    df = collect()
    if df.empty:
        step("! no sessions assembled"); return 1
    step(f"Sessions: {len(df)}")

    sess_csv = ANALYSIS / "licking_vs_contour_per_session.csv"
    df.to_csv(sess_csv, index=False)
    step(f"Per-session CSV: {sess_csv}")

    # Diagnostic print
    step("\nPer-session quick view:")
    cols_show = ["cohort", "subject_id", "dose", "pct_licking",
                 "ratio_all", "ratio_nolk", "ratio_lick",
                 "n_lick_frames", "n_nolk_frames"]
    step(df[cols_show].round(3).to_string(index=False))

    # Per-dose summary
    import numpy as np
    summary_rows = []
    for d in DOSE_SEQ:
        sub = df[df["dose"] == d]
        for key, label in [("ratio_all", "all"), ("ratio_nolk", "nolk"),
                           ("ratio_lick", "lick")]:
            vals = sub[key].dropna().astype(float).to_numpy()
            summary_rows.append({
                "dose": d, "frames": label, "n": len(vals),
                "mean": round(float(np.mean(vals)), 4) if len(vals) else float("nan"),
                "sem":  round(float(np.std(vals, ddof=1) / np.sqrt(len(vals))), 4) if len(vals) > 1 else 0.0,
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = ANALYSIS / "licking_vs_contour_dose_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    step("\nPer-dose conditional summary:")
    step(summary_df.to_string(index=False))

    # Plots
    scatter_png = ANALYSIS / "licking_vs_contour_scatter.png"
    corr_stats = plot_scatter(df, scatter_png)
    step(f"Scatter: r={corr_stats['r']:.3f}  r²={corr_stats['r2']:.3f}  "
         f"p={corr_stats['p']:.3g}  n={corr_stats['n']}")

    bars_png = ANALYSIS / "licking_vs_contour_dose_bars.png"
    plot_dose_bars(df, bars_png)

    step("\nBuilding split TC...")
    df_tc = collect_tc()
    tc_png = ANALYSIS / "licking_vs_contour_tc_split.png"
    plot_tc_split(df_tc, tc_png)

    # Headline numbers for the Discord post
    def dose_mean(dose, key):
        v = df.loc[df["dose"] == dose, key].dropna().astype(float).to_numpy()
        return float(np.mean(v)) if len(v) else float("nan")
    nolk_means = {d: dose_mean(d, "ratio_nolk") for d in DOSE_SEQ}

    head = (
        "**FormOxy: contour ratio decomposed by licking state**\n"
        "Tests whether the contour HL/HR ratio dose effect is an "
        "independent paw-guarding biomarker or just a re-statement of "
        "licking-induced occlusion of the formalin paw.\n\n"
        f"**Per-session licking% vs ratio_all correlation:** "
        f"r = {corr_stats['r']:.3f},  r² = {corr_stats['r2']:.3f},  "
        f"p = {corr_stats['p']:.3g},  n = {corr_stats['n']} sessions.\n"
        "**Per-dose mean ratio on non-licking frames only:** "
        + ", ".join(f"{d} = {nolk_means[d]:.3f}" for d in DOSE_SEQ) +
        ".\n\nReading guide:\n"
        "• If the **non-licking** dose bars are flat across doses, the "
        "original signal is largely a licking proxy.\n"
        "• If they still trend Veh→Oxy10 even on non-licking frames, "
        "there's a posture/guarding component independent of licking."
    )
    discord_upload(head, [scatter_png, bars_png, tc_png, sess_csv, summary_csv])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
