"""
Single-pass feature extraction + 5-classifier prediction for the
260512 THC rimonabant cohort.

For each video in videos/ that has a DLC .h5 alongside it:
  1. Build features at canonical hash 8aed1c22 (with optical flow,
     bp_pixbrt = [hrpaw, hlpaw, snout], square_size=[40,40,40]).
  2. Run the 5 SNLT classifiers (facial_grooming, left_licking,
     rearing, still, walking) with the per-classifier postprocess
     from for_claude.json.
  3. Write <session>_predictions.csv to results/.

Idempotent: skips videos whose features pickle AND predictions csv
both already exist.

Run from the regular PixelPaws env (not DEEPLABCUT):
  PYTHONIOENCODING=utf-8 py -X utf8 scripts/research/thc_rim_predict_all.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(r"E:\PixelPaws")
sys.path.insert(0, str(REPO))

PROJECT_ROOT = Path(r"E:\RSVIDS\Blackbox\260512_THC_Rim_Cohort")
VIDEO_DIR = PROJECT_ROOT / "videos"
LOCAL_CLF_DIR = PROJECT_ROOT / "classifiers"
JSON_PATH = Path(r"E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\transitions\for_claude.json")
SHUFFLE = 9

WEBHOOK = (
    ""
)
BAR_WIDTH = 24


def step(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def _send(url, payload, method="POST"):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method=method,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "PixelPaws-Chain/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read()
            return json.loads(body) if body else None
    except Exception as e:
        print(f"  ! Discord {method} failed: {e}", flush=True)
        return None


def discord_create(text):
    resp = _send(WEBHOOK + "?wait=true", {"content": text})
    return resp.get("id") if resp else None


def discord_edit(msg_id, text):
    if not msg_id:
        return
    _send(f"{WEBHOOK}/messages/{msg_id}", {"content": text}, method="PATCH")


def make_bar(cur, total, width=BAR_WIDTH):
    if total <= 0:
        return "░" * width
    filled = max(0, min(width, cur * width // total))
    return "█" * filled + "░" * (width - filled)


def fmt_dur(seconds):
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def progress_text(done, total, t_start, current=None, status=None):
    bar = make_bar(done, total)
    pct = 100 * done / total if total else 0
    elapsed = time.time() - t_start
    if done > 0 and done < total:
        rate = elapsed / done
        eta = rate * (total - done)
        eta_str = f"~{fmt_dur(eta)}"
    elif done >= total:
        eta_str = "done"
    else:
        eta_str = "..."
    lines = [
        "**THC-Rim predictions -- progress**",
        f"`[{bar}] {done}/{total} sessions  ({pct:5.1f}%)`",
    ]
    if current:
        lines.append(f"Current: `{current}`{(' -- ' + status) if status else ''}")
    lines.append(f"Elapsed: {fmt_dur(elapsed)} | ETA: {eta_str}")
    return "\n".join(lines)


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


def main() -> int:
    import numpy as np
    import pandas as pd

    with open(JSON_PATH, "r") as f:
        spec = json.load(f)
    classifiers = spec["classifiers"]
    step(f"JSON spec: {len(classifiers)} classifiers, fps={spec.get('fps')}, "
         f"transition_mode={spec.get('transition_mode')}")

    pairs = []
    for v in sorted(VIDEO_DIR.glob("*.mp4")):
        h5 = list(v.parent.glob(f"{v.stem}*shuffle{SHUFFLE}*filtered.h5"))
        if not h5:
            h5 = list(v.parent.glob(f"{v.stem}*shuffle{SHUFFLE}*.h5"))
        if h5:
            pairs.append((v, h5[0]))
        else:
            step(f"  ! no DLC h5 for {v.name} — skip")
    if not pairs:
        step("No (video, h5) pairs found. Run DLC first.")
        return 1
    step(f"Sessions to process: {len(pairs)}")

    results_dir = PROJECT_ROOT / "results"
    features_dir = PROJECT_ROOT / "features"
    results_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    loaded = []
    union_bp_pixbrt = []
    union_square_size = [40]
    union_pix_threshold = 0.3
    for c in classifiers:
        local = LOCAL_CLF_DIR / Path(c["path"]).name
        clf_path = local if local.is_file() else Path(c["path"])
        with open(clf_path, "rb") as f:
            cd = pickle.load(f)
        for bp in cd.get("bp_pixbrt_list", []):
            if bp not in union_bp_pixbrt:
                union_bp_pixbrt.append(bp)
        union_square_size = cd.get("square_size", union_square_size)
        union_pix_threshold = cd.get("pix_threshold", union_pix_threshold)
        loaded.append({**c, "clf_data": cd, "name": Path(c["path"]).stem,
                       "resolved_path": str(clf_path)})
    step(f"Union bp_pixbrt: {union_bp_pixbrt}")

    try:
        from feature_cache import FeatureCacheManager
        cfg_for_hash = {
            "bp_include_list": None,
            "bp_pixbrt_list": union_bp_pixbrt,
            "square_size": union_square_size,
            "pix_threshold": union_pix_threshold,
            "include_optical_flow": True,
            "bp_optflow_list": ["hrpaw", "hlpaw", "snout"],
        }
        cfg_hash = FeatureCacheManager.compute_hash(cfg_for_hash)
    except Exception as e:
        step(f"  ! could not compute hash: {e}")
        cfg_hash = "manual"
    step(f"feature-cache hash (with flow): {cfg_hash}")

    from prediction_pipeline import (
        PixelPaws_ExtractFeatures,
        predict_with_xgboost,
        augment_features_post_cache,
    )

    summary_rows = []
    total_sessions = len(pairs)
    t0_all = time.time()
    msg_id = discord_create(progress_text(0, total_sessions, t0_all))
    step(f"Discord progress msg id: {msg_id}")

    for i, (v, h5) in enumerate(pairs):
        base = v.stem
        out_csv = results_dir / f"{base}_predictions.csv"
        feat_cache = features_dir / f"{base}_features_{cfg_hash}.pkl"

        # Idempotency: skip session if its predictions CSV exists.
        if out_csv.is_file():
            step(f"  skip (predictions exist): {base}")
            continue

        step(f"\n=== {base} ===")
        step(f"  video: {v}")
        step(f"  dlc:   {h5}")
        discord_edit(msg_id, progress_text(
            i, total_sessions, t0_all,
            current=base, status="loading features...",
        ))

        if feat_cache.is_file():
            step("  loading cached features (with flow)...")
            with open(feat_cache, "rb") as f:
                X = pickle.load(f)
        else:
            step("  extracting features (with optical flow, single pass)...")
            discord_edit(msg_id, progress_text(
                i, total_sessions, t0_all,
                current=base, status="extracting features (with flow)...",
            ))
            X = PixelPaws_ExtractFeatures(
                pose_data_file=str(h5),
                video_file_path=str(v),
                bp_include_list=None,
                bp_pixbrt_list=union_bp_pixbrt,
                square_size=union_square_size,
                pix_threshold=union_pix_threshold,
                config_yaml_path=None,
                include_optical_flow=True,
                bp_optflow_list=["hrpaw", "hlpaw", "snout"],
            )
            with open(feat_cache, "wb") as f:
                pickle.dump(X, f)
            step(f"  cached -> {feat_cache.name}")
        step(f"  X shape: {X.shape}")

        discord_edit(msg_id, progress_text(
            i, total_sessions, t0_all,
            current=base, status="running 5 classifiers...",
        ))

        out = pd.DataFrame({"frame": range(len(X))})
        n = len(X)
        for L in loaded:
            cd = L["clf_data"]
            model = cd["clf_model"]
            behavior = cd.get("Behavior_type") or L["name"]
            best_thresh = float(L.get("best_thresh", cd.get("best_thresh", 0.5)))
            min_bout = int(L.get("min_bout", cd.get("min_bout", 1)))
            min_after_bout = int(L.get("min_after_bout", 0))
            max_gap = int(L.get("max_gap", 0))

            X_aug = augment_features_post_cache(X.copy(), cd, model, str(h5))
            y_proba = predict_with_xgboost(
                model, X_aug,
                calibrator=cd.get("prob_calibrator"),
                fold_models=cd.get("fold_models"),
            )
            y_raw = (y_proba >= best_thresh).astype(int)
            y_post = apply_postprocess(y_raw, min_bout, min_after_bout, max_gap)
            out[f"{behavior}_proba"] = y_proba
            out[behavior] = y_post

            n_pos = int(y_post.sum())
            bouts = len(find_bouts(y_post))
            step(f"  {behavior:18}  thresh={best_thresh:.2f}  "
                 f"min_bout={min_bout:>2}  max_gap={max_gap:>2}  "
                 f"pos={n_pos:>6} ({100*n_pos/n:>5.2f}%)  bouts={bouts}")
            summary_rows.append({
                "session": base, "behavior": behavior,
                "best_thresh": best_thresh, "min_bout": min_bout,
                "max_gap": max_gap,
                "n_pos": n_pos, "pct": round(100 * n_pos / n, 3),
                "bouts": bouts,
            })

        out.to_csv(out_csv, index=False)
        step(f"  -> {out_csv.name}")
        discord_edit(msg_id, progress_text(
            i + 1, total_sessions, t0_all,
            current=base, status="done",
        ))

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = PROJECT_ROOT / "thc_rim_predict_summary.csv"
    if summary_csv.is_file():
        # Append rather than overwrite, so re-runs add new sessions.
        existing = pd.read_csv(summary_csv)
        combined = pd.concat([existing, summary_df], ignore_index=True)
        combined.drop_duplicates(subset=["session", "behavior"], keep="last", inplace=True)
        combined.to_csv(summary_csv, index=False)
    else:
        summary_df.to_csv(summary_csv, index=False)
    step(f"\nSummary: {summary_csv}")
    discord_edit(msg_id, progress_text(total_sessions, total_sessions, t0_all))
    return 0


if __name__ == "__main__":
    sys.exit(main())
