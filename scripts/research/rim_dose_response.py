"""
Rimonabant dose-response analysis for the 260515 female rim cohort.

Cohort design (Rim 1-9, cycle of 4):
  - Rim1, Rim5, Rim9 -> VEH
  - Rim2, Rim6       -> 1  mg/kg
  - Rim3, Rim7       -> 3  mg/kg
  - Rim4, Rim8       -> 10 mg/kg

For each session (baseline + postrim per mouse), runs the 5 SNLT
classifiers PLUS Scratching (uses _AllFeatures since features pickle
includes optical flow). Aggregates by dose, with baseline as
within-subject reference.

Output: dose-response plot for scratching (primary), 5 secondary
behavior dose-responses, summary CSV. Posts to Discord.

Run from regular py (not DEEPLABCUT):
  PYTHONIOENCODING=utf-8 py -X utf8 scripts/research/rim_dose_response.py
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sys
import time
import urllib.request
import uuid
from pathlib import Path

REPO = Path(r"E:\PixelPaws")
PROJECT_ROOT = Path(r"E:\RSVIDS\Blackbox\260515_Rim_DoseResp")
VIDEO_DIR = PROJECT_ROOT / "videos"
LOCAL_CLF_DIR = PROJECT_ROOT / "classifiers"
JSON_PATH = Path(r"E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\transitions\for_claude.json")
SCRATCH_CLF = Path(r"E:\RSVIDS\Blackbox\260506_RS_THC_Withdrawal\classifiers\PixelPaws_Scratching_AllFeatures.pkl")
SHUFFLE = 9
FPS = 60

WEBHOOK = (
    ""
)

DOSE_SEQUENCE = ["VEH", "1 mg/kg", "3 mg/kg", "10 mg/kg"]
DOSE_COLORS = {
    "VEH": "#7f7f7f",
    "1 mg/kg": "#fbbf24",
    "3 mg/kg": "#f97316",
    "10 mg/kg": "#dc2626",
}


def step(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def dose_for_mouse(n: int) -> str:
    return DOSE_SEQUENCE[(n - 1) % 4]


def parse_session(name: str):
    """fem_<baseline|postrim>_Rim<N>[_date] -> (mouse_num, condition)."""
    m = re.match(r"^fem_(baseline|postrim)_Rim(\d+)(?:_\d+)?$", name)
    if not m:
        return None
    return int(m.group(2)), m.group(1)


def find_bouts(y):
    bouts = []
    in_b = False; start = 0
    for i, v in enumerate(y):
        if v and not in_b:
            in_b = True; start = i
        elif not v and in_b:
            in_b = False; bouts.append((start, i - 1))
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


def discord_upload(content, files):
    boundary = "----PP" + uuid.uuid4().hex
    body = [f"--{boundary}".encode(),
            b'Content-Disposition: form-data; name="payload_json"',
            b'Content-Type: application/json', b"",
            json.dumps({"content": content}).encode()]
    for i, p in enumerate(files):
        mime = "image/png" if p.suffix == ".png" else "text/csv"
        body += [f"--{boundary}".encode(),
                 f'Content-Disposition: form-data; name="file{i}"; filename="{p.name}"'.encode(),
                 f"Content-Type: {mime}".encode(), b"", p.read_bytes()]
    body.append(f"--{boundary}--".encode())
    req = urllib.request.Request(WEBHOOK, data=b"\r\n".join(body), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "User-Agent": "PixelPaws-Chain/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            print(f"  Discord HTTP {r.status}")
    except Exception as e:
        print(f"  Discord upload failed: {e}")


def main():
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(REPO))
    from prediction_pipeline import (
        PixelPaws_ExtractFeatures, predict_with_xgboost, augment_features_post_cache,
    )
    from feature_cache import FeatureCacheManager

    # Discover sessions
    pairs = []
    for v in sorted(VIDEO_DIR.glob("*.mp4")):
        h5 = list(v.parent.glob(f"{v.stem}*shuffle{SHUFFLE}*filtered.h5"))
        if not h5:
            h5 = list(v.parent.glob(f"{v.stem}*shuffle{SHUFFLE}*.h5"))
        if h5:
            parsed = parse_session(v.stem)
            if parsed:
                pairs.append((v, h5[0], parsed[0], parsed[1]))
    step(f"Sessions: {len(pairs)}")
    if not pairs:
        step("No (video, h5, parsed) triples found.")
        return 1

    # Load classifiers (5 SNLT + scratching)
    with open(JSON_PATH) as f:
        spec = json.load(f)
    cls = list(spec["classifiers"])
    cls.append({"path": str(SCRATCH_CLF), "best_thresh": 0.5,
                "min_bout": 6, "min_after_bout": 0, "max_gap": 0})

    loaded = []
    union_bp_pixbrt = []
    union_square = [40]
    union_pix_thresh = 0.3
    for c in cls:
        local = LOCAL_CLF_DIR / Path(c["path"]).name
        clf_path = local if local.is_file() else Path(c["path"])
        with open(clf_path, "rb") as f:
            cd = pickle.load(f)
        for bp in cd.get("bp_pixbrt_list", []):
            if bp not in union_bp_pixbrt:
                union_bp_pixbrt.append(bp)
        union_square = cd.get("square_size", union_square)
        union_pix_thresh = cd.get("pix_threshold", union_pix_thresh)
        loaded.append({**c, "clf_data": cd, "name": Path(c["path"]).stem,
                       "resolved_path": str(clf_path)})
    step(f"Classifiers loaded: {len(loaded)} (incl Scratching)")

    cfg_for_hash = dict(
        bp_include_list=None,
        bp_pixbrt_list=union_bp_pixbrt,
        square_size=union_square,
        pix_threshold=union_pix_thresh,
        include_optical_flow=True,
        bp_optflow_list=["hrpaw", "hlpaw", "snout"],
    )
    cfg_hash = FeatureCacheManager.compute_hash(cfg_for_hash)
    step(f"Feature hash: {cfg_hash}")

    results_dir = PROJECT_ROOT / "results"
    features_dir = PROJECT_ROOT / "features"
    analysis_dir = PROJECT_ROOT / "analysis"
    for d in (results_dir, features_dir, analysis_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Per-session prediction + summary
    summary_rows = []
    for v, h5, mouse, cond in pairs:
        base = v.stem
        out_csv = results_dir / f"{base}_predictions.csv"
        feat_cache = features_dir / f"{base}_features_{cfg_hash}.pkl"
        if out_csv.is_file():
            step(f"  skip (preds exist): {base}")
            pred = pd.read_csv(out_csv)
        else:
            step(f"\n=== {base}  (mouse {mouse} {cond}) ===")
            if feat_cache.is_file():
                with open(feat_cache, "rb") as f:
                    X = pickle.load(f)
                step(f"  features from cache ({X.shape})")
            else:
                step("  extracting features (with flow)...")
                X = PixelPaws_ExtractFeatures(
                    pose_data_file=str(h5),
                    video_file_path=str(v),
                    bp_include_list=None,
                    bp_pixbrt_list=union_bp_pixbrt,
                    square_size=union_square,
                    pix_threshold=union_pix_thresh,
                    config_yaml_path=None,
                    include_optical_flow=True,
                    bp_optflow_list=["hrpaw", "hlpaw", "snout"],
                )
                with open(feat_cache, "wb") as f:
                    pickle.dump(X, f)
                step(f"  cached -> {feat_cache.name} (shape {X.shape})")

            pred = pd.DataFrame({"frame": range(len(X))})
            for L in loaded:
                cd = L["clf_data"]
                model = cd["clf_model"]
                behavior = cd.get("Behavior_type") or L["name"]
                thresh = float(L.get("best_thresh", cd.get("best_thresh", 0.5)))
                mb = int(L.get("min_bout", cd.get("min_bout", 1)))
                ma = int(L.get("min_after_bout", 0))
                mg = int(L.get("max_gap", 0))
                X_aug = augment_features_post_cache(X.copy(), cd, model, str(h5))
                y_proba = predict_with_xgboost(
                    model, X_aug,
                    calibrator=cd.get("prob_calibrator"),
                    fold_models=cd.get("fold_models"),
                )
                y_raw = (y_proba >= thresh).astype(int)
                y_post = apply_postprocess(y_raw, mb, ma, mg)
                pred[f"{behavior}_proba"] = y_proba
                pred[behavior] = y_post
            pred.to_csv(out_csv, index=False)
            step(f"  -> {out_csv.name}")

        # Per-session metrics
        n_frames = len(pred)
        for col in pred.columns:
            if col == "frame" or col.endswith("_proba"):
                continue
            y = pred[col].astype(int).to_numpy()
            n_pos = int(y.sum())
            bouts = find_bouts(y)
            mean_bout_s = float(np.mean([b[1] - b[0] + 1 for b in bouts])) / FPS if bouts else 0.0
            summary_rows.append({
                "session": base, "mouse": mouse, "condition": cond,
                "dose": dose_for_mouse(mouse),
                "behavior": col,
                "pct_time": 100 * n_pos / n_frames,
                "total_time_s": n_pos / FPS,
                "n_bouts": len(bouts),
                "mean_bout_s": round(mean_bout_s, 2),
                "n_frames": n_frames,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = analysis_dir / "rim_dose_response_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    step(f"\nSummary: {summary_csv}")

    # =========================================================
    # Dose-response plot for Scratching (primary) + others
    # =========================================================
    behaviors_plot = ["Scratching", "Facial_grooming", "Left_licking",
                      "rearing", "still", "walking"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, beh in zip(axes.flat, behaviors_plot):
        sub = summary_df[summary_df["behavior"] == beh].copy()
        if sub.empty:
            ax.text(0.5, 0.5, f"no data for {beh}", ha="center", va="center")
            ax.set_title(beh)
            continue
        # Plot per-mouse paired pre->post lines, color by dose
        for mouse, mdf in sub.groupby("mouse"):
            dose = mdf["dose"].iloc[0]
            bl = mdf[mdf["condition"] == "baseline"]["total_time_s"]
            po = mdf[mdf["condition"] == "postrim"]["total_time_s"]
            if len(bl) and len(po):
                ax.plot([0, 1], [bl.iloc[0] / 60, po.iloc[0] / 60],
                        marker="o", color=DOSE_COLORS[dose],
                        alpha=0.7, linewidth=1.5,
                        label=f"m{mouse} ({dose})")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["baseline", "postrim"])
        ax.set_ylabel("Total time (min)")
        ax.set_title(f"{beh}")
        ax.grid(alpha=0.3)
        # De-duplicate legend
        h, l = ax.get_legend_handles_labels()
        seen = {}
        for hi, li in zip(h, l):
            key = li.split()[-1]  # e.g. (VEH)
            if key not in seen:
                seen[key] = hi
        ax.legend(seen.values(), seen.keys(), fontsize=7, loc="best")

    fig.suptitle("Rimonabant dose-response -- 6 behaviors, baseline -> postrim per mouse",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    paired_png = analysis_dir / "rim_dose_response_paired.png"
    fig.savefig(paired_png, dpi=140); plt.close(fig)

    # Dose-mean plot: post-rim minus baseline, by dose
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    dose_order = DOSE_SEQUENCE
    for ax, beh in zip(axes.flat, behaviors_plot):
        sub = summary_df[summary_df["behavior"] == beh].copy()
        if sub.empty:
            ax.text(0.5, 0.5, f"no data", ha="center", va="center"); ax.set_title(beh); continue
        # Compute delta per mouse, then group
        deltas = []
        for mouse, mdf in sub.groupby("mouse"):
            dose = mdf["dose"].iloc[0]
            bl = mdf[mdf["condition"] == "baseline"]["total_time_s"]
            po = mdf[mdf["condition"] == "postrim"]["total_time_s"]
            if len(bl) and len(po):
                deltas.append({"mouse": mouse, "dose": dose,
                               "delta_min": (po.iloc[0] - bl.iloc[0]) / 60.0})
        d_df = pd.DataFrame(deltas)
        if d_df.empty:
            ax.text(0.5, 0.5, "no paired", ha="center", va="center"); ax.set_title(beh); continue

        x = np.arange(len(dose_order))
        means = [d_df[d_df["dose"] == d]["delta_min"].mean() if (d_df["dose"] == d).any() else np.nan
                 for d in dose_order]
        sems = [d_df[d_df["dose"] == d]["delta_min"].sem() if (d_df["dose"] == d).sum() > 1 else 0
                for d in dose_order]
        colors = [DOSE_COLORS[d] for d in dose_order]
        ax.bar(x, means, yerr=sems, color=colors, alpha=0.7, capsize=4,
               edgecolor="black", linewidth=0.5)
        # Scatter per-mouse on top
        for d_idx, d in enumerate(dose_order):
            ys = d_df[d_df["dose"] == d]["delta_min"].to_numpy()
            xs = np.random.normal(d_idx, 0.05, len(ys))
            ax.scatter(xs, ys, color="black", s=18, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(dose_order, rotation=15, fontsize=8)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel("postrim - baseline (min)")
        ax.set_title(beh)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Rimonabant dose-response -- delta (postrim - baseline) by dose",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    dose_png = analysis_dir / "rim_dose_response_delta.png"
    fig.savefig(dose_png, dpi=140); plt.close(fig)

    # Discord
    n_paired = summary_df.groupby(["dose"])["mouse"].nunique()
    head = (
        "**Rimonabant dose-response analysis**\n"
        f"Cohort: 260515_Rim_DoseResp ({summary_df['mouse'].nunique()} mice, "
        f"{summary_df['session'].nunique()} sessions).\n"
        "Doses per mouse: " +
        ", ".join(f"{d} (n={int(n_paired.get(d, 0))})" for d in DOSE_SEQUENCE) +
        "\n\nPrimary endpoint: Scratching. Secondary: 5 SNLT behaviors."
    )
    discord_upload(head, [paired_png, dose_png, summary_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
