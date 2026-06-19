"""
Combined Left_licking dose-response for the formalin+oxycodone study.

Cohorts:
  - 2512_Blackbox_Formalin_Oxy/Left_paws  (existing, 12 mice, predictions on disk)
  - 2605_FormOxy_LeftPaws                 (new,      12 mice, predictions generated here)

Both cohorts use a 4-dose Latin-square (Veh / Oxy1 / Oxy3 / Oxy10) with
3 mice per dose per cohort -> n=6 per dose in the pooled analysis.

For comparability each session is truncated to the first 60 min
(216,000 frames at 60 fps) before computing percent-licking and bout
metrics.

Output:
  - per-session predictions for 2605 (frame, probability, Left_licking)
  - combined summary CSV
  - dose-response figure (per-mouse scatter + group mean +/- SEM, by cohort)
  - posts the figure + summary to #results on Discord
"""
from __future__ import annotations

import json
import pickle
import re
import sys
import time
import urllib.request
import uuid
from pathlib import Path

REPO = Path(r"E:\PixelPaws")
sys.path.insert(0, str(REPO))

# --------------------------------------------------------------------- #
# Cohort paths
# --------------------------------------------------------------------- #
COHORT_NEW = Path(r"E:\RSVIDS\Blackbox\2605_FormOxy_LeftPaws")
NEW_VIDS = COHORT_NEW / "videos"
NEW_FEATS = COHORT_NEW / "features"
NEW_RESULTS = COHORT_NEW / "results"
ANALYSIS = COHORT_NEW / "analysis"

COHORT_OLD = Path(r"E:\RSVIDS\Blackbox\2512_Blackbox_Formalin_Oxy\Left_paws")
OLD_PRED_DIR = COHORT_OLD / "Results" / "Left_licking"

CLASSIFIER_PATH = COHORT_OLD / "Classifiers" / "PixelPaws_Left_licking.pkl"

FEATURE_HASH = "8aed1c22"      # 2605 features hash (with flow)
SHUFFLE = 9
FPS = 60
WINDOW_MIN = 60                # cap each session at first 60 minutes
WINDOW_FRAMES = WINDOW_MIN * 60 * FPS  # 216,000

# Dose keys (subject_number -> dose)
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

# #results channel webhook
WEBHOOK = (
    ""
)


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def step(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def find_bouts(y):
    bouts = []
    in_b = False
    start = 0
    for i, v in enumerate(y):
        if v and not in_b:
            in_b = True
            start = i
        elif not v and in_b:
            in_b = False
            bouts.append((start, i - 1))
    if in_b:
        bouts.append((start, len(y) - 1))
    return bouts


def apply_postprocess(y_pred, min_bout=1, min_after_bout=0, max_gap=0):
    import numpy as np
    y = y_pred.astype(int).copy()
    if max_gap > 0:
        bouts = find_bouts(y)
        for i in range(len(bouts) - 1):
            gap = bouts[i + 1][0] - bouts[i][1] - 1
            if 0 < gap <= max_gap:
                y[bouts[i][1] + 1: bouts[i + 1][0]] = 1
    if min_bout > 1:
        for s, e in find_bouts(y):
            if e - s + 1 < min_bout:
                y[s: e + 1] = 0
    if min_after_bout > 0:
        bouts = find_bouts(y)
        for i in range(1, len(bouts)):
            gap = bouts[i][0] - bouts[i - 1][1] - 1
            if gap < min_after_bout:
                s, e = bouts[i]
                y[s: e + 1] = 0
    return y


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
        WEBHOOK,
        data=b"\r\n".join(body),
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "PixelPaws-FormOxy-DoseResponse/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            step(f"Discord upload: HTTP {r.status}")
    except Exception as e:
        step(f"Discord upload failed: {e}")


# --------------------------------------------------------------------- #
# Per-cohort metric extraction
# --------------------------------------------------------------------- #
def metrics_from_binary(y_full, n_frames_full: int) -> dict:
    """Cap to WINDOW_FRAMES and compute pct, total time, n_bouts, mean bout."""
    import numpy as np
    y = y_full[:WINDOW_FRAMES]
    n = len(y)
    n_pos = int(y.sum())
    bouts = find_bouts(y)
    return {
        "n_frames_capped": n,
        "n_frames_total": n_frames_full,
        "pct_licking": 100.0 * n_pos / n if n else 0.0,
        "total_licking_s": n_pos / FPS,
        "n_bouts": len(bouts),
        "mean_bout_s": float(np.mean([b[1] - b[0] + 1 for b in bouts])) / FPS if bouts else 0.0,
    }


# --------------------------------------------------------------------- #
# 2605: predict + summarize
# --------------------------------------------------------------------- #
def predict_new_cohort(clf_data, model) -> list[dict]:
    import numpy as np
    import pandas as pd
    from prediction_pipeline import predict_with_xgboost, augment_features_post_cache

    NEW_RESULTS.mkdir(parents=True, exist_ok=True)

    thresh = float(clf_data["best_thresh"])
    mb = int(clf_data["min_bout"])
    ma = int(clf_data["min_after_bout"])
    mg = int(clf_data["max_gap"])

    rows = []
    pkl_pattern = f"*_formalin_features_{FEATURE_HASH}.pkl"
    pkls = sorted(NEW_FEATS.glob(pkl_pattern))
    if not pkls:
        step(f"! no 2605 feature pkls matched {pkl_pattern}")
        return rows
    step(f"2605 cohort: {len(pkls)} formalin feature pickles")

    rx = re.compile(r"2605_FormOxy_S(\d+)_formalin")
    for pkl in pkls:
        m = rx.search(pkl.name)
        if not m:
            step(f"  ? skipping (no subject match): {pkl.name}")
            continue
        subj = int(m.group(1))
        base = f"2605_FormOxy_S{subj}_formalin"
        out_csv = NEW_RESULTS / f"{base}_Left_licking_predictions.csv"

        if out_csv.is_file():
            step(f"  cached preds: {base}")
            pred = pd.read_csv(out_csv)
            y = pred["Left_licking"].astype(int).to_numpy()
            n_full = len(y)
        else:
            step(f"  predict: {base}")
            with open(pkl, "rb") as f:
                X = pickle.load(f)
            h5 = list(NEW_VIDS.glob(f"{base}*shuffle{SHUFFLE}*_filtered.h5"))
            if not h5:
                step(f"    ! no filtered .h5 for {base}; skipping")
                continue
            X_aug = augment_features_post_cache(X.copy(), clf_data, model, str(h5[0]))
            y_proba = predict_with_xgboost(
                model, X_aug,
                calibrator=clf_data.get("prob_calibrator"),
                fold_models=clf_data.get("fold_models"),
            )
            y_raw = (y_proba >= thresh).astype(int)
            y = apply_postprocess(y_raw, mb, ma, mg)
            n_full = len(y)
            pred = pd.DataFrame({
                "frame": np.arange(n_full),
                "probability": y_proba,
                "Left_licking": y.astype(int),
            })
            pred.to_csv(out_csv, index=False)
            step(f"    -> {out_csv.name} ({n_full} frames)")

        met = metrics_from_binary(y, n_full)
        rows.append({
            "cohort": "2605",
            "subject_id": f"2605_S{subj}",
            "mouse_num": subj,
            "dose": dose_2605(subj),
            **met,
        })
    return rows


# --------------------------------------------------------------------- #
# 2512: re-read existing per-frame predictions, apply same cap
# --------------------------------------------------------------------- #
def summarize_old_cohort() -> list[dict]:
    import pandas as pd
    rows = []
    rx = re.compile(r"2512_FormOxy_S(\d+)_PixelPaws_Left_licking_predictions\.csv$")
    csvs = sorted(OLD_PRED_DIR.glob("2512_FormOxy_S*_PixelPaws_Left_licking_predictions.csv"))
    if not csvs:
        step(f"! no 2512 prediction CSVs found in {OLD_PRED_DIR}")
        return rows
    step(f"2512 cohort: {len(csvs)} prediction CSVs")
    for csv_path in csvs:
        m = rx.search(csv_path.name)
        if not m:
            continue
        subj = int(m.group(1))
        df = pd.read_csv(csv_path, usecols=["Left_licking"])
        y = df["Left_licking"].astype(int).to_numpy()
        met = metrics_from_binary(y, len(y))
        rows.append({
            "cohort": "2512",
            "subject_id": f"2512_S{subj}",
            "mouse_num": subj,
            "dose": DOSE_2512[subj],
            **met,
        })
    return rows


# --------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------- #
def plot_dose_response(summary_df, out_png: Path) -> None:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("pct_licking",     "% time licking",          "%"),
        ("total_licking_s", "Total licking",           "min",  60.0),
        ("n_bouts",         "Bout count",              "n"),
        ("mean_bout_s",     "Mean bout duration",      "s"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, spec in zip(axes.flat, metrics):
        key = spec[0]; title = spec[1]; unit = spec[2]
        scale = spec[3] if len(spec) > 3 else 1.0

        # Means + SEMs per dose
        means, sems = [], []
        for d in DOSE_SEQ:
            vals = summary_df.loc[summary_df["dose"] == d, key].astype(float).to_numpy() / scale
            means.append(np.mean(vals) if len(vals) else np.nan)
            sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)

        x = np.arange(len(DOSE_SEQ))
        colors = [DOSE_COLORS[d] for d in DOSE_SEQ]
        ax.bar(x, means, yerr=sems, capsize=4, color=colors, alpha=0.55,
               edgecolor="black", linewidth=0.6)

        # Per-mouse scatter, marker by cohort
        for i, d in enumerate(DOSE_SEQ):
            sub = summary_df[summary_df["dose"] == d]
            for cohort, marker in [("2512", "o"), ("2605", "s")]:
                vals = sub.loc[sub["cohort"] == cohort, key].astype(float).to_numpy() / scale
                if len(vals) == 0:
                    continue
                xs = np.random.normal(i, 0.06, len(vals))
                ax.scatter(xs, vals, color="black", marker=marker, s=28,
                           edgecolors="white", linewidths=0.6, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(DOSE_SEQ)
        ax.set_ylabel(f"{title} ({unit})")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    # Legend (markers by cohort)
    from matplotlib.lines import Line2D
    leg = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
               markeredgecolor="white", label="2512 cohort (n=3/dose)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="black",
               markeredgecolor="white", label="2605 cohort (n=3/dose)"),
    ]
    fig.legend(handles=leg, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.005))

    fig.suptitle(
        f"FormOxy: Left_licking dose-response (formalin, first {WINDOW_MIN} min)\n"
        f"Combined cohorts: n=6 per dose (3 per cohort)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------- #
def main() -> int:
    import pandas as pd
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    NEW_RESULTS.mkdir(parents=True, exist_ok=True)

    step(f"Loading classifier: {CLASSIFIER_PATH.name}")
    with open(CLASSIFIER_PATH, "rb") as f:
        clf_data = pickle.load(f)
    model = clf_data["clf_model"]
    step(f"  best_thresh={clf_data['best_thresh']:.3f}  min_bout={clf_data['min_bout']}  "
         f"min_after_bout={clf_data['min_after_bout']}  max_gap={clf_data['max_gap']}")

    rows_new = predict_new_cohort(clf_data, model)
    rows_old = summarize_old_cohort()

    summary = pd.DataFrame(rows_new + rows_old)
    summary = summary.sort_values(["cohort", "mouse_num"]).reset_index(drop=True)
    summary_csv = ANALYSIS / "leftlicking_dose_response_combined.csv"
    summary.to_csv(summary_csv, index=False)
    step(f"\nCombined summary: {summary_csv}")

    # Per-dose stats
    stats_rows = []
    for d in DOSE_SEQ:
        sub = summary[summary["dose"] == d]
        stats_rows.append({
            "dose": d,
            "n_total": len(sub),
            "n_2512": int((sub["cohort"] == "2512").sum()),
            "n_2605": int((sub["cohort"] == "2605").sum()),
            "pct_licking_mean": round(sub["pct_licking"].mean(), 2),
            "pct_licking_sem":  round(sub["pct_licking"].std(ddof=1) / max(1, len(sub)) ** 0.5, 2),
            "total_min_mean":   round(sub["total_licking_s"].mean() / 60.0, 2),
            "n_bouts_mean":     round(sub["n_bouts"].mean(), 1),
            "mean_bout_s_mean": round(sub["mean_bout_s"].mean(), 2),
        })
    stats_df = pd.DataFrame(stats_rows)
    stats_csv = ANALYSIS / "leftlicking_dose_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    step(f"Dose stats: {stats_csv}")
    step("\n" + stats_df.to_string(index=False))

    out_png = ANALYSIS / "leftlicking_dose_response.png"
    plot_dose_response(summary, out_png)
    step(f"Figure: {out_png}")

    # ----- Discord post -----
    table_lines = ["```",
                   f"{'Dose':<6} {'n':>3} {'%lick':>7} {'SEM':>6} {'min':>6} {'bouts':>6} {'bout_s':>7}"]
    for r in stats_rows:
        table_lines.append(
            f"{r['dose']:<6} {r['n_total']:>3} "
            f"{r['pct_licking_mean']:>7.2f} {r['pct_licking_sem']:>6.2f} "
            f"{r['total_min_mean']:>6.2f} {r['n_bouts_mean']:>6.1f} "
            f"{r['mean_bout_s_mean']:>7.2f}"
        )
    table_lines.append("```")

    head = (
        "**FormOxy Left_licking dose-response (formalin, first 60 min)**\n"
        f"Combined cohorts 2512 + 2605, n=6 per dose.\n"
        f"Classifier: `PixelPaws_Left_licking.pkl` (thresh={clf_data['best_thresh']:.2f}, "
        f"min_bout={clf_data['min_bout']}, max_gap={clf_data['max_gap']}).\n\n"
        + "\n".join(table_lines)
    )
    discord_upload(head, [out_png, summary_csv, stats_csv])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
