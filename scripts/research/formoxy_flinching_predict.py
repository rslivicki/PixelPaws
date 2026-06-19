"""
Try the latest L_flinching classifier on the 2605 FormOxy formalin cohort.

Uses the SHAP-pruned 120-feature variant (same CV F1 as AllFeatures-899
but more interpretable / less overfit):
  PixelPaws_L_flinching_pruned_120_20260429_165631.pkl
  (mean_cv_f1 = 0.721, thresh 0.65, min_bout 8, max_gap 20)

CAVEAT: the flinching classifier was trained with snout brightness ROI
window 15 px / pixel_threshold 0.4, but the 2605 cached features were
extracted with window 40 / threshold 0.3 (matching the Left_licking
brightness config). Of the 120 selected features ~7 are snout-brightness-
derived (Pix_snout / Pix_*_snout) and will be quantitatively wrong; the
~100 pose / paw-brightness features match exactly. Treat this run as a
qualitative "does the dose-response look plausible" check rather than
a publication-quality estimate. For a clean run we need to re-extract
features with snout window 15, threshold 0.4 (~2-3 h).

Output per session: `2605_FormOxy_LeftPaws/results/*_L_flinching_predictions.csv`
Combined dose-response (bar + per-mouse scatter) goes to the `#results`
Discord channel.
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
# Paths / params
# --------------------------------------------------------------------- #
COHORT = Path(r"E:\RSVIDS\Blackbox\2605_FormOxy_LeftPaws")
VIDS = COHORT / "videos"
FEATS = COHORT / "features"
RESULTS = COHORT / "results"
ANALYSIS = COHORT / "analysis"

CLASSIFIER_PATH = Path(r"E:\RSVIDS\Blackbox\2511_Blackbox_Formalin\PixelPaws"
                      r"\Flinching\classifiers"
                      r"\PixelPaws_L_flinching_pruned_120_20260429_165631.pkl")

FEATURE_HASH = "8aed1c22"
SHUFFLE = 9
FPS = 60
WINDOW_MIN = 60
WINDOW_FRAMES = WINDOW_MIN * 60 * FPS  # 216 000

DOSE_SEQ = ["Veh", "Oxy1", "Oxy3", "Oxy10"]
DOSE_COLORS = {
    "Veh":   "#7f7f7f",
    "Oxy1":  "#fbbf24",
    "Oxy3":  "#f97316",
    "Oxy10": "#dc2626",
}


def dose_for(n: int) -> str:
    return DOSE_SEQ[(n - 1) % 4]


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
                 "User-Agent": "PixelPaws-FormOxy-Flinching/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            step(f"Discord upload: HTTP {r.status}")
    except Exception as e:
        step(f"Discord upload failed: {e}")


# --------------------------------------------------------------------- #
# Bout / post-process helpers (same as licking script)
# --------------------------------------------------------------------- #
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


# --------------------------------------------------------------------- #
# Predict + summarize
# --------------------------------------------------------------------- #
def metrics(y_full):
    import numpy as np
    y = y_full[:WINDOW_FRAMES]
    n = len(y)
    n_pos = int(y.sum())
    bouts = find_bouts(y)
    return {
        "n_frames_capped": n,
        "n_frames_total":  len(y_full),
        "pct_flinching":   100.0 * n_pos / n if n else 0.0,
        "total_flinch_s":  n_pos / FPS,
        "n_bouts":         len(bouts),
        "mean_bout_s":     float(np.mean([b[1] - b[0] + 1 for b in bouts])) / FPS if bouts else 0.0,
    }


def main() -> int:
    import joblib
    import numpy as np
    import pandas as pd
    from prediction_pipeline import predict_with_xgboost, augment_features_post_cache

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    step(f"Loading classifier: {CLASSIFIER_PATH.name}")
    cd = joblib.load(CLASSIFIER_PATH)
    model = cd["clf_model"]
    thresh = float(cd["best_thresh"])
    mb = int(cd["min_bout"])
    ma = int(cd["min_after_bout"])
    mg = int(cd["max_gap"])
    step(f"  best_thresh={thresh:.3f}  min_bout={mb}  min_after_bout={ma}  max_gap={mg}")
    step(f"  classifier n_features_in_={getattr(model, 'n_features_in_', '?')}  "
         f"mean_cv_f1={cd.get('mean_cv_f1'):.3f}")

    pkls = sorted(FEATS.glob(f"*_formalin_features_{FEATURE_HASH}.pkl"))
    if not pkls:
        step(f"! no feature pickles matched"); return 1
    step(f"Sessions: {len(pkls)}")

    rx = re.compile(r"2605_FormOxy_S(\d+)_formalin")
    rows = []
    for pkl in pkls:
        m = rx.search(pkl.name)
        if not m:
            continue
        subj = int(m.group(1))
        base = f"2605_FormOxy_S{subj}_formalin"
        out_csv = RESULTS / f"{base}_L_flinching_predictions.csv"

        if out_csv.is_file():
            step(f"  cached: {base}")
            pred = pd.read_csv(out_csv, usecols=["L_flinching"])
            y = pred["L_flinching"].astype(int).to_numpy()
        else:
            step(f"  predict: {base}")
            with open(pkl, "rb") as f:
                X = pickle.load(f)
            h5_candidates = list(VIDS.glob(f"{base}*shuffle{SHUFFLE}*_filtered.h5"))
            if not h5_candidates:
                step(f"    ! no filtered .h5 for {base}; skipping")
                continue
            X_aug = augment_features_post_cache(X.copy(), cd, model, str(h5_candidates[0]))

            # Fixup 1: inverse log-ratio. Model wants Log10(Pix_hlpaw/Pix_hrpaw)
            # but we only have Log10(Pix_hrpaw/Pix_hlpaw); negate to recover.
            if ("Log10(Pix_hlpaw/Pix_hrpaw)" not in X_aug.columns
                    and "Log10(Pix_hrpaw/Pix_hlpaw)" in X_aug.columns):
                X_aug["Log10(Pix_hlpaw/Pix_hrpaw)"] = -X_aug["Log10(Pix_hrpaw/Pix_hlpaw)"]

            # Fixup 2: DutyCycle is a contact-derived feature that
            # augment_features_post_cache only triggers when it sees a
            # *_ContactState column in the model's feature_names_in_.
            # The pruned flinching model selected *_DutyCycle directly
            # (no ContactState), so the trigger misses and DutyCycle never
            # gets added. Compute it inline here.
            need_dc = [f for f in model.feature_names_in_
                       if f.endswith("_DutyCycle") and f not in X_aug.columns]
            if need_dc:
                from pose_features import PoseFeatureExtractor
                _ct_thresh = float(cd.get("contact_threshold", 15.0))
                _ct = PoseFeatureExtractor(bodyparts=[], contact_threshold=_ct_thresh)
                _ct_df = _ct.calculate_contact_features(X_aug)
                add_cols = [c for c in _ct_df.columns if c not in X_aug.columns]
                if add_cols:
                    X_aug = pd.concat(
                        [X_aug.reset_index(drop=True),
                         _ct_df[add_cols].iloc[:len(X_aug)].reset_index(drop=True)],
                        axis=1,
                    )

            y_proba = predict_with_xgboost(
                model, X_aug,
                calibrator=cd.get("prob_calibrator"),
                fold_models=cd.get("fold_models"),
            )
            y_raw = (y_proba >= thresh).astype(int)
            y = apply_postprocess(y_raw, mb, ma, mg)
            pd.DataFrame({
                "frame": np.arange(len(y)),
                "probability": y_proba,
                "L_flinching": y.astype(int),
            }).to_csv(out_csv, index=False)
            step(f"    -> {out_csv.name} ({len(y)} frames)")

        rows.append({
            "subject_id": f"2605_S{subj}",
            "mouse_num":  subj,
            "dose":       dose_for(subj),
            **metrics(y),
        })

    summary = pd.DataFrame(rows).sort_values("mouse_num").reset_index(drop=True)
    summary_csv = ANALYSIS / "flinching_2605_summary.csv"
    summary.to_csv(summary_csv, index=False)
    step(f"\nSummary: {summary_csv}")

    # Per-dose stats
    stats_rows = []
    for d in DOSE_SEQ:
        sub = summary[summary["dose"] == d]
        stats_rows.append({
            "dose": d,
            "n": len(sub),
            "pct_flinch_mean": round(sub["pct_flinching"].mean(), 2),
            "pct_flinch_sem":  round(sub["pct_flinching"].std(ddof=1) / max(1, len(sub)) ** 0.5, 2),
            "total_min_mean":  round(sub["total_flinch_s"].mean() / 60.0, 2),
            "n_bouts_mean":    round(sub["n_bouts"].mean(), 1),
            "mean_bout_s":     round(sub["mean_bout_s"].mean(), 2),
        })
    stats_df = pd.DataFrame(stats_rows)
    stats_csv = ANALYSIS / "flinching_2605_dose_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    step("\n" + stats_df.to_string(index=False))

    # --- Plot: 4-panel dose-response ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_panels = [
        ("pct_flinching",  "% time flinching",      "%"),
        ("total_flinch_s", "Total flinching time",  "min", 60.0),
        ("n_bouts",        "Bout count",            "n"),
        ("mean_bout_s",    "Mean bout duration",    "s"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, spec in zip(axes.flat, metrics_panels):
        key, title, unit = spec[0], spec[1], spec[2]
        scale = spec[3] if len(spec) > 3 else 1.0
        means, sems = [], []
        for d in DOSE_SEQ:
            vals = summary.loc[summary["dose"] == d, key].astype(float).to_numpy() / scale
            means.append(np.mean(vals) if len(vals) else np.nan)
            sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
        x = np.arange(len(DOSE_SEQ))
        colors = [DOSE_COLORS[d] for d in DOSE_SEQ]
        ax.bar(x, means, yerr=sems, capsize=4, color=colors, alpha=0.55,
               edgecolor="black", linewidth=0.6)
        for i, d in enumerate(DOSE_SEQ):
            vals = summary.loc[summary["dose"] == d, key].astype(float).to_numpy() / scale
            if len(vals) == 0:
                continue
            xs = np.random.normal(i, 0.06, len(vals))
            ax.scatter(xs, vals, color="black", s=28, edgecolors="white", linewidths=0.6, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(DOSE_SEQ)
        ax.set_ylabel(f"{title} ({unit})")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(
        f"FormOxy 2605: L_flinching dose-response (formalin, first {WINDOW_MIN} min)\n"
        f"PixelPaws_L_flinching_pruned_120 (CV F1 = {cd.get('mean_cv_f1', 0):.3f}),  "
        f"n=3 per dose",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    out_png = ANALYSIS / "flinching_2605_dose_response.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    step(f"Figure: {out_png}")

    # --- Discord post (with caveat) ---
    table_lines = ["```",
                   f"{'Dose':<6} {'n':>3} {'%flinch':>9} {'SEM':>6} {'min':>6} {'bouts':>6} {'bout_s':>7}"]
    for r in stats_rows:
        table_lines.append(
            f"{r['dose']:<6} {r['n']:>3} "
            f"{r['pct_flinch_mean']:>9.2f} {r['pct_flinch_sem']:>6.2f} "
            f"{r['total_min_mean']:>6.2f} {r['n_bouts_mean']:>6.1f} "
            f"{r['mean_bout_s']:>7.2f}"
        )
    table_lines.append("```")

    head = (
        "**FormOxy 2605: L_flinching dose-response (qualitative)**\n"
        f"Classifier: `PixelPaws_L_flinching_pruned_120_20260429_165631.pkl` "
        f"(thresh {thresh:.2f}, min_bout {mb}, max_gap {mg}, CV F1 "
        f"{cd.get('mean_cv_f1', 0):.3f}). "
        f"First 60 min, n=3 per dose (2605 cohort only).\n\n"
        ":warning: **Caveat:** the 2605 feature cache was extracted with "
        "snout brightness ROI = 40 px / threshold 0.3, but this flinching "
        "classifier was trained with snout ROI = 15 px / threshold 0.4. "
        "~7 of the 120 selected features are snout-brightness-derived and "
        "will be quantitatively off; the other ~110 (pose + paw brightness) "
        "match exactly. Treat as a **qualitative** check, not a final "
        "estimate. Re-extracting features for a clean run would take ~2-3 h.\n\n"
        + "\n".join(table_lines)
    )
    discord_upload(head, [out_png, summary_csv, stats_csv])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
