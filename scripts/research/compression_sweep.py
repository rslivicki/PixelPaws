"""
Pre-analysis video compression sensitivity test (multi-video cohort).

For each source video in COHORT, extracts a SUBSET_DURATION-second clip
starting at SUBSET_START_SEC, then transcodes that subset at a grid of
CRF settings, re-runs DLC + feature extraction + 5-classifier prediction
on each. Reference (= original quality) is the corresponding slice of
the already-computed full-video h5 / features pickle on disk -- no need
to re-process the reference.

Stages (each idempotent, skip-existing):
  1. transcode    -- ffmpeg, subset of each source at each CRF
  2. dlc          -- DLC on every transcoded clip
  3. features     -- PixelPaws_ExtractFeatures (hash 8aed1c22)
  4. predict      -- 5 SNLT classifiers per clip
  5. compare      -- delta vs reference (sliced from original h5/features)
  6. aggregate    -- per-CRF mean +- SD across sources, summary CSV + plot
  7. discord      -- post summary

Defaults: 4 THC pre-rim BL videos, subset of 3 min starting at minute 5.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

REPO = Path(r"E:\PixelPaws")
DLC_PYTHON = r"C:\Users\Gereau\anaconda3\envs\DEEPLABCUT\python.exe"
DLC_CONFIG = r"E:\RSDLC\2511_RSGNKK_Blackbox\config.yaml"
SHUFFLE = 9
BATCHSIZE = 32
FPS = 60

# Subset: minutes 5-8 of each source. Skips habituation at start, samples
# from middle of recording where mouse behavior is most varied.
SUBSET_START_SEC = 5 * 60
SUBSET_DURATION_SEC = 3 * 60
SUBSET_START_FRAME = SUBSET_START_SEC * FPS
SUBSET_END_FRAME = SUBSET_START_FRAME + SUBSET_DURATION_SEC * FPS

# Cohort: 4 mice, each with full DLC h5 + features pickle on disk.
# We assume the THC rim cohort layout: videos/, features/.
_THC = Path(r"E:\RSVIDS\Blackbox\260512_THC_Rim_Cohort")
COHORT = [
    # (source_video, reference_h5, reference_features)
    (_THC / "videos" / "pre_rimonabant_THC1_20260512.mp4",
     _THC / "videos" / "pre_rimonabant_THC1_20260512DLC_Resnet50_palmreader-500Mar25shuffle9_snapshot_best-190.h5",
     _THC / "features" / "pre_rimonabant_THC1_20260512_features_8aed1c22.pkl"),
    (_THC / "videos" / "pre_rimonabant_THC2.mp4",
     _THC / "videos" / "pre_rimonabant_THC2DLC_Resnet50_palmreader-500Mar25shuffle9_snapshot_best-190.h5",
     _THC / "features" / "pre_rimonabant_THC2_features_8aed1c22.pkl"),
    (_THC / "videos" / "pre_rimonabant_THC3.mp4",
     _THC / "videos" / "pre_rimonabant_THC3DLC_Resnet50_palmreader-500Mar25shuffle9_snapshot_best-190.h5",
     _THC / "features" / "pre_rimonabant_THC3_features_8aed1c22.pkl"),
    (_THC / "videos" / "pre_rimonabant_THC4_20260512.mp4",
     _THC / "videos" / "pre_rimonabant_THC4_20260512DLC_Resnet50_palmreader-500Mar25shuffle9_snapshot_best-190.h5",
     _THC / "features" / "pre_rimonabant_THC4_20260512_features_8aed1c22.pkl"),
]
WORK_DIR = Path(r"E:\RSVIDS\Blackbox\260512_THC_Rim_Cohort\compression_sweep")

CLASSIFIER_DIR = Path(r"E:\RSVIDS\Blackbox\260506_RS_THC_Withdrawal\classifiers")
SNLT_JSON = Path(r"E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\transitions\for_claude.json")

WEBHOOK = (
    ""
)

# (codec, crf, label)
CRF_GRID = [
    ("libx264", 23, "x264_crf23"),
    ("libx264", 26, "x264_crf26"),
    ("libx264", 28, "x264_crf28"),
    ("libx264", 31, "x264_crf31"),
    ("libx264", 35, "x264_crf35"),
    ("libx265", 28, "x265_crf28"),
]

FEATURE_CFG = dict(
    bp_pixbrt_list=["hrpaw", "hlpaw", "snout"],
    square_size=[40, 40, 40],
    pix_threshold=0.3,
    include_optical_flow=True,
    bp_optflow_list=["hrpaw", "hlpaw", "snout"],
)
FEATURE_HASH = "8aed1c22"


def step(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# --------------------------------------------------------------------------- #
# Stage 1: transcode subsets
# --------------------------------------------------------------------------- #

def transcode_subset(src: Path, dst: Path, codec: str, crf: int) -> None:
    if dst.is_file():
        step(f"  skip transcode (exists): {dst.name} ({dst.stat().st_size/1e6:.1f} MB)")
        return
    step(f"  transcode subset -> {dst.name}  ({codec} crf={crf})")
    # -ss BEFORE -i = fast seek to nearest keyframe (may be inexact);
    # -ss AFTER -i = decode-and-seek (frame accurate). We use after-i for accuracy.
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-ss", str(SUBSET_START_SEC),
        "-t", str(SUBSET_DURATION_SEC),
        "-c:v", codec,
        "-crf", str(crf),
        "-preset", "slower" if codec == "libx264" else "slow",
        "-an",
        str(dst),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {dst.name}:\n{proc.stderr[-500:]}")
    step(f"    done in {time.time()-t0:.1f}s, size {dst.stat().st_size/1e6:.1f} MB")


# --------------------------------------------------------------------------- #
# Stage 2: DLC via subprocess
# --------------------------------------------------------------------------- #

DLC_INNER_SCRIPT = """
import sys, json
videos = json.loads(sys.argv[1])
import deeplabcut
deeplabcut.analyze_videos(
    r"{cfg}", videos, shuffle={shuffle}, videotype=".mp4",
    save_as_csv=False, batchsize={batch},
)
"""


def run_dlc_on(videos: list[Path]) -> None:
    pending = [v for v in videos
               if not list(v.parent.glob(f"{v.stem}*shuffle{SHUFFLE}*.h5"))]
    if not pending:
        step("  DLC: nothing to do (all .h5 exist)")
        return
    step(f"  DLC: analyzing {len(pending)} video(s)")
    inner = DLC_INNER_SCRIPT.format(cfg=DLC_CONFIG, shuffle=SHUFFLE, batch=BATCHSIZE)
    inner_path = Path(os.environ.get("TEMP", ".")) / f"dlc_inner_{uuid.uuid4().hex}.py"
    inner_path.write_text(inner)
    try:
        proc = subprocess.run(
            [DLC_PYTHON, str(inner_path), json.dumps([str(v) for v in pending])],
            cwd=str(REPO),
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    finally:
        inner_path.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(f"DLC subprocess exited {proc.returncode}")


# --------------------------------------------------------------------------- #
# Stage 3-4: features + classifiers
# --------------------------------------------------------------------------- #

def find_h5(video: Path) -> Path | None:
    cands = list(video.parent.glob(f"{video.stem}*shuffle{SHUFFLE}*.h5"))
    cands = [c for c in cands if not c.name.endswith("_filtered.h5")]
    return cands[0] if cands else None


def compute_features(video: Path, h5: Path, cache_path: Path):
    if cache_path.is_file():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    sys.path.insert(0, str(REPO))
    from prediction_pipeline import PixelPaws_ExtractFeatures
    step(f"  features -> {cache_path.name}")
    X = PixelPaws_ExtractFeatures(
        pose_data_file=str(h5),
        video_file_path=str(video),
        bp_include_list=None,
        config_yaml_path=None,
        **FEATURE_CFG,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(X, f)
    return X


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


def run_classifiers(X, h5_path, classifiers_spec):
    import pandas as pd
    sys.path.insert(0, str(REPO))
    from prediction_pipeline import predict_with_xgboost, augment_features_post_cache
    out = pd.DataFrame({"frame": range(len(X))})
    for c in classifiers_spec:
        local = CLASSIFIER_DIR / Path(c["path"]).name
        clf_path = local if local.is_file() else Path(c["path"])
        with open(clf_path, "rb") as f:
            cd = pickle.load(f)
        model = cd["clf_model"]
        behavior = cd.get("Behavior_type") or Path(c["path"]).stem
        thresh = float(c.get("best_thresh", 0.5))
        min_bout = int(c.get("min_bout", 1))
        min_after = int(c.get("min_after_bout", 0))
        max_gap = int(c.get("max_gap", 0))
        X_aug = augment_features_post_cache(X.copy(), cd, model, str(h5_path))
        y_proba = predict_with_xgboost(
            model, X_aug,
            calibrator=cd.get("prob_calibrator"),
            fold_models=cd.get("fold_models"),
        )
        y_raw = (y_proba >= thresh).astype(int)
        y_post = apply_postprocess(y_raw, min_bout, min_after, max_gap)
        out[f"{behavior}_proba"] = y_proba
        out[behavior] = y_post
    return out


# --------------------------------------------------------------------------- #
# Stage 5: compare to reference (sliced from original)
# --------------------------------------------------------------------------- #

def slice_h5(h5_path: Path, start_frame: int, end_frame: int):
    """Return DataFrame slice [start:end] of a DLC h5 file."""
    import pandas as pd
    df = pd.read_hdf(h5_path)
    df.columns = df.columns.droplevel(0)
    return df.iloc[start_frame:end_frame].reset_index(drop=True)


def slice_features(features_path: Path, start_frame: int, end_frame: int):
    import pandas as pd
    with open(features_path, "rb") as f:
        X = pickle.load(f)
    if hasattr(X, "iloc"):
        return X.iloc[start_frame:end_frame].reset_index(drop=True)
    return X[start_frame:end_frame]


def compare_h5_to_ref(test_h5: Path, ref_df) -> dict:
    import numpy as np
    import pandas as pd
    test = pd.read_hdf(test_h5)
    test.columns = test.columns.droplevel(0)
    n = min(len(ref_df), len(test))
    ref = ref_df.iloc[:n]; test = test.iloc[:n]
    bps = sorted(set(b for b, _ in ref.columns))
    out = {}
    for bp in bps:
        lk_ref = ref[(bp, "likelihood")].to_numpy()
        lk_test = test[(bp, "likelihood")].to_numpy()
        dx = ref[(bp, "x")].to_numpy() - test[(bp, "x")].to_numpy()
        dy = ref[(bp, "y")].to_numpy() - test[(bp, "y")].to_numpy()
        pos_delta = np.sqrt(dx*dx + dy*dy)
        valid = (lk_ref >= 0.6) & (lk_test >= 0.6)
        out[bp] = {
            "lk_drop_mean": float(lk_ref.mean() - lk_test.mean()),
            "pos_delta_median_px": float(np.median(pos_delta[valid])) if valid.sum() else float("nan"),
        }
    return out


def compare_features_to_ref(test_X, ref_X) -> dict:
    import numpy as np
    cols = [c for c in ref_X.columns if c in test_X.columns]
    n = min(len(ref_X), len(test_X))
    ref = ref_X[cols].iloc[:n].to_numpy()
    test = test_X[cols].iloc[:n].to_numpy()
    rs = []
    for j in range(len(cols)):
        a = ref[:, j]; b = test[:, j]
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if not np.isnan(r):
            rs.append(r)
    rs = np.array(rs)
    return {
        "n_features": len(cols),
        "min_corr": float(rs.min()) if len(rs) else float("nan"),
        "p5_corr": float(np.percentile(rs, 5)) if len(rs) else float("nan"),
        "median_corr": float(np.median(rs)) if len(rs) else float("nan"),
        "frac_below_0_95": float((rs < 0.95).mean()) if len(rs) else float("nan"),
    }


def compare_preds_to_ref(test_pred, ref_pred) -> dict:
    import numpy as np
    behavior_cols = [c for c in ref_pred.columns
                     if c != "frame" and not c.endswith("_proba")]
    n = min(len(ref_pred), len(test_pred))
    out = {}
    agrees = []
    for b in behavior_cols:
        r = ref_pred[b].to_numpy()[:n].astype(int)
        t = test_pred[b].to_numpy()[:n].astype(int)
        a = float((r == t).mean())
        out[b] = a
        agrees.append(a)
    out["_overall"] = float(np.mean(agrees))
    return out


# --------------------------------------------------------------------------- #
# Stage 6: aggregate + plot + Discord
# --------------------------------------------------------------------------- #

def discord_upload(content: str, file_paths: list[Path]):
    boundary = "----PP" + uuid.uuid4().hex
    body = [
        f"--{boundary}".encode(),
        b'Content-Disposition: form-data; name="payload_json"',
        b'Content-Type: application/json', b"",
        json.dumps({"content": content}).encode(),
    ]
    for i, p in enumerate(file_paths):
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
                 "User-Agent": "PixelPaws-Chain/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            step(f"  Discord HTTP {r.status}")
    except Exception as e:
        step(f"  Discord upload failed: {e}")


def make_summary_plot(agg_df, per_video_df, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    crfs = agg_df["label"].tolist()
    x = np.arange(len(crfs))

    # Panel 1: size reduction
    ax = axes[0, 0]
    ax.bar(x, agg_df["size_mb_mean"], yerr=agg_df["size_mb_sd"],
           color="#3b82f6", alpha=0.8, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(crfs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("File size (MB, mean +/- SD across videos)")
    ax.set_title("Subset file size by CRF")
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: keypoint likelihood drop
    ax = axes[0, 1]
    ax.errorbar(x, agg_df["lk_drop_mean"], yerr=agg_df["lk_drop_sd"],
                marker="o", color="#d62728", capsize=4)
    # Per-video lines
    for video in per_video_df["video"].unique():
        sub = per_video_df[per_video_df["video"] == video].sort_values("label_idx")
        ax.plot(sub["label_idx"], sub["lk_drop_mean_overall"],
                color="#d62728", alpha=0.2, linewidth=1)
    ax.axhline(0.05, ls="--", color="gray", alpha=0.5, label="0.05 redline")
    ax.set_xticks(x); ax.set_xticklabels(crfs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean keypoint likelihood drop")
    ax.set_title("DLC quality drop vs original")
    ax.legend(); ax.grid(alpha=0.3)

    # Panel 3: feature correlation (min)
    ax = axes[1, 0]
    ax.errorbar(x, agg_df["feature_min_corr_mean"], yerr=agg_df["feature_min_corr_sd"],
                marker="o", color="#16a34a", capsize=4, label="min (worst feature)")
    ax.errorbar(x, agg_df["feature_median_corr_mean"], yerr=agg_df["feature_median_corr_sd"],
                marker="s", color="#84cc16", capsize=4, label="median")
    for video in per_video_df["video"].unique():
        sub = per_video_df[per_video_df["video"] == video].sort_values("label_idx")
        ax.plot(sub["label_idx"], sub["feature_min_corr"],
                color="#16a34a", alpha=0.2, linewidth=1)
    ax.axhline(0.95, ls="--", color="gray", alpha=0.5, label="0.95 redline")
    ax.set_xticks(x); ax.set_xticklabels(crfs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Pearson r vs reference features")
    ax.set_title("Feature correlation (per-CRF mean +/- SD across videos)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_ylim(0.4, 1.02)

    # Panel 4: classifier agreement
    ax = axes[1, 1]
    ax.errorbar(x, agg_df["clf_overall_mean"], yerr=agg_df["clf_overall_sd"],
                marker="o", color="#7c3aed", capsize=4)
    for video in per_video_df["video"].unique():
        sub = per_video_df[per_video_df["video"] == video].sort_values("label_idx")
        ax.plot(sub["label_idx"], sub["clf_overall"],
                color="#7c3aed", alpha=0.2, linewidth=1)
    ax.axhline(0.98, ls="--", color="gray", alpha=0.5, label="0.98 redline")
    ax.set_xticks(x); ax.set_xticklabels(crfs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Per-frame agreement (mean across 5 classifiers)")
    ax.set_title("Classifier agreement vs reference")
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim(0.8, 1.005)

    fig.suptitle(f"Compression sweep -- {len(per_video_df['video'].unique())} mice, "
                 f"3-min subset (min {SUBSET_START_SEC//60}-{(SUBSET_START_SEC+SUBSET_DURATION_SEC)//60})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main():
    import numpy as np
    import pandas as pd

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    step(f"Subset: minutes {SUBSET_START_SEC/60:.0f}-{(SUBSET_START_SEC+SUBSET_DURATION_SEC)/60:.0f} "
         f"({SUBSET_DURATION_SEC}s, frames {SUBSET_START_FRAME}-{SUBSET_END_FRAME})")
    step(f"Cohort: {len(COHORT)} video(s)")
    step(f"CRF grid: {[g[2] for g in CRF_GRID]}")
    step(f"Work dir: {WORK_DIR}")

    # ===== Stage 1: transcode all subsets =====
    step("\n=== Stage 1: transcode subsets ===")
    all_targets = []  # (src_path, codec, crf, label, out_path)
    for src_video, _, _ in COHORT:
        for codec, crf, crf_label in CRF_GRID:
            stem = src_video.stem
            out = WORK_DIR / f"{stem}_{crf_label}.mp4"
            transcode_subset(src_video, out, codec, crf)
            all_targets.append((src_video, codec, crf, crf_label, out))

    # ===== Stage 2: DLC on all subset clips =====
    step("\n=== Stage 2: DLC ===")
    run_dlc_on([t[4] for t in all_targets])

    # ===== Stage 3-5: features + classifiers + compare =====
    step("\n=== Stage 3-5: features + classifiers + compare ===")
    with open(SNLT_JSON, "r") as f:
        spec = json.load(f)
    cls_spec = spec["classifiers"]

    per_video_rows = []
    for src_video, ref_h5, ref_feat in COHORT:
        step(f"\n--- source: {src_video.stem} ---")
        # Build reference slice from already-computed full-video data
        ref_df = slice_h5(ref_h5, SUBSET_START_FRAME, SUBSET_END_FRAME)
        ref_X = slice_features(ref_feat, SUBSET_START_FRAME, SUBSET_END_FRAME)
        ref_pred = run_classifiers(ref_X, ref_h5, cls_spec)
        ref_pred_csv = WORK_DIR / f"{src_video.stem}_ref_predictions.csv"
        ref_pred.to_csv(ref_pred_csv, index=False)

        for src_v, codec, crf, crf_label, mp4 in all_targets:
            if src_v != src_video:
                continue
            step(f"  --- {crf_label} ---")
            size_mb = mp4.stat().st_size / 1e6
            h5 = find_h5(mp4)
            if not h5:
                step(f"    ! missing .h5 for {mp4.name}, skipping")
                continue
            feat_path = WORK_DIR / f"{mp4.stem}_features_{FEATURE_HASH}.pkl"
            X = compute_features(mp4, h5, feat_path)
            pred = run_classifiers(X, h5, cls_spec)

            dlc_metrics = compare_h5_to_ref(h5, ref_df)
            feat_metrics = compare_features_to_ref(X, ref_X)
            pred_metrics = compare_preds_to_ref(pred, ref_pred)

            lk_drops = [v["lk_drop_mean"] for v in dlc_metrics.values()]
            row = dict(
                video=src_video.stem, label=crf_label, codec=codec, crf=crf,
                size_mb=round(size_mb, 1),
                lk_drop_mean_overall=round(float(np.mean(lk_drops)), 4),
                feature_min_corr=round(feat_metrics["min_corr"], 4),
                feature_median_corr=round(feat_metrics["median_corr"], 4),
                feature_frac_below_0_95=round(feat_metrics["frac_below_0_95"], 4),
                clf_overall=round(pred_metrics["_overall"], 4),
            )
            for b, a in pred_metrics.items():
                if b.startswith("_"): continue
                row[f"agree_{b}"] = round(a, 4)
            per_video_rows.append(row)
            step(f"    size={size_mb:.1f}MB  lk_drop={row['lk_drop_mean_overall']:.3f}  "
                 f"feat_min={row['feature_min_corr']:.3f}  "
                 f"clf={row['clf_overall']:.3f}")

    per_video_df = pd.DataFrame(per_video_rows)
    label_order = [g[2] for g in CRF_GRID]
    per_video_df["label_idx"] = per_video_df["label"].apply(label_order.index)
    per_video_csv = WORK_DIR / "compression_sweep_per_video.csv"
    per_video_df.to_csv(per_video_csv, index=False)
    step(f"\nPer-video CSV: {per_video_csv}")

    # ===== Stage 6: aggregate =====
    agg_rows = []
    for codec, crf, label in CRF_GRID:
        sub = per_video_df[per_video_df["label"] == label]
        if sub.empty:
            continue
        agg_rows.append(dict(
            label=label, codec=codec, crf=crf,
            n=len(sub),
            size_mb_mean=round(sub["size_mb"].mean(), 1),
            size_mb_sd=round(sub["size_mb"].std(), 2) if len(sub) > 1 else 0.0,
            lk_drop_mean=round(sub["lk_drop_mean_overall"].mean(), 4),
            lk_drop_sd=round(sub["lk_drop_mean_overall"].std(), 4) if len(sub) > 1 else 0.0,
            feature_min_corr_mean=round(sub["feature_min_corr"].mean(), 4),
            feature_min_corr_sd=round(sub["feature_min_corr"].std(), 4) if len(sub) > 1 else 0.0,
            feature_median_corr_mean=round(sub["feature_median_corr"].mean(), 4),
            feature_median_corr_sd=round(sub["feature_median_corr"].std(), 4) if len(sub) > 1 else 0.0,
            clf_overall_mean=round(sub["clf_overall"].mean(), 4),
            clf_overall_sd=round(sub["clf_overall"].std(), 4) if len(sub) > 1 else 0.0,
        ))
    agg_df = pd.DataFrame(agg_rows)
    agg_csv = WORK_DIR / "compression_sweep_summary.csv"
    agg_df.to_csv(agg_csv, index=False)
    step(f"Summary: {agg_csv}")

    # Plot
    plot_png = WORK_DIR / "compression_sweep_summary.png"
    make_summary_plot(agg_df, per_video_df, plot_png)
    step(f"Plot: {plot_png}")

    # Discord
    summary_txt = "```\n" + agg_df[
        ["label", "n", "size_mb_mean", "lk_drop_mean",
         "feature_min_corr_mean", "clf_overall_mean"]
    ].to_string(index=False) + "\n```"
    head = (
        "**Compression sweep results -- multi-video cohort**\n"
        f"Cohort: {len(COHORT)} THC pre-rim BL videos.\n"
        f"Subset: min {SUBSET_START_SEC//60}-{(SUBSET_START_SEC+SUBSET_DURATION_SEC)//60} "
        f"of each ({SUBSET_DURATION_SEC}s = {SUBSET_DURATION_SEC*FPS} frames at {FPS} fps).\n"
        f"Redlines: keypoint drop <= 0.05, feature min_corr >= 0.95, "
        f"classifier agreement >= 0.98.\n\n" + summary_txt
    )
    discord_upload(head, [plot_png, agg_csv, per_video_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
