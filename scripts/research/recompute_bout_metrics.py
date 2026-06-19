"""Recompute bout-level F1 on the existing oof.npz files using proper
post-processing (min_bout / max_gap merging). Sweeps a small grid to
find best post-processing settings per behaviour.

This is what users actually see when running predictions through the
GUI's post-processing pipeline. The raw bout F1 reported in
training_metrics.json (no post-processing) is essentially measuring
frame-prediction flicker, not bout-detection quality.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from train_global_classifier import (  # noqa: E402
    bout_f1, per_session_bout_metrics, postprocess_predictions, to_bouts,
)

PERF_DIR = Path(r"E:/PixelPaws/pixelpaws_global_classifier_encyclopedia/"
                r"performance_results")
CLF_DIR = Path(r"E:/PixelPaws/pixelpaws_global_classifier_encyclopedia/"
               r"classifiers")

# Joint optimization grid (threshold × min_bout × max_gap)
THRESH_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
               0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
MIN_BOUT_GRID = [1, 3, 5, 8, 12, 18, 30]
MAX_GAP_GRID = [0, 2, 5, 10, 20, 40]


def main() -> int:
    import joblib
    for behav_dir in sorted(PERF_DIR.glob("*/")):
        oof_path = behav_dir / "oof.npz"
        metrics_path = behav_dir / "training_metrics.json"
        if not oof_path.is_file() or not metrics_path.is_file():
            continue
        name = behav_dir.name
        npz = np.load(oof_path)
        oof = npz["oof_proba"]
        y = npz["y"]
        sids = npz["session_ids"]
        meta = json.loads(metrics_path.read_text())

        print(f"\n=== {name} ===")
        print(f"  frame F1 (CV) = {meta['mean_cv_f1']:.3f}")

        # Joint sweep of (threshold, min_bout, max_gap)
        best = None
        n_combos = 0
        for t in THRESH_GRID:
            for mb in MIN_BOUT_GRID:
                for mg in MAX_GAP_GRID:
                    n_combos += 1
                    m = per_session_bout_metrics(oof, y, sids, t,
                                                 min_bout=mb, max_gap=mg)
                    if best is None or m["bout_f1_mean"] > best["bout_f1_mean"]:
                        best = {**m, "thresh": t, "min_bout": mb, "max_gap": mg}

        # Also report raw (no post-proc) at the chosen threshold
        raw = per_session_bout_metrics(oof, y, sids, best["thresh"],
                                       min_bout=1, max_gap=0)
        print(f"  swept {n_combos} (thresh, min_bout, max_gap) combos")
        print(f"  best bout F1 = {best['bout_f1_mean']:.3f}  "
              f"(thresh={best['thresh']:.2f}, "
              f"min_bout={best['min_bout']}, max_gap={best['max_gap']})")
        print(f"    bout-count Pearson r = {best['bout_count_pearson_r']:.3f}  "
              f"dur err = {best['mean_bout_duration_error_sec']:.2f}s")
        print(f"    (raw-no-postproc at this thresh: bout F1 = {raw['bout_f1_mean']:.3f})")

        # Update training_metrics.json
        meta["bout_optimization"] = {
            "method": "joint sweep over (threshold, min_bout, max_gap)",
            "thresh_grid": THRESH_GRID,
            "min_bout_grid": MIN_BOUT_GRID,
            "max_gap_grid": MAX_GAP_GRID,
            "best": {
                "thresh": best["thresh"],
                "min_bout": best["min_bout"],
                "max_gap": best["max_gap"],
                "bout_f1": best["bout_f1_mean"],
                "bout_f1_std": best["bout_f1_std"],
                "bout_count_pearson_r": best["bout_count_pearson_r"],
                "mean_bout_duration_error_sec": best["mean_bout_duration_error_sec"],
            },
            "raw_at_best_thresh_no_postproc": {
                "bout_f1": raw["bout_f1_mean"],
                "bout_f1_std": raw["bout_f1_std"],
            },
        }
        meta["recommended_post_processing"] = {
            "best_thresh": best["thresh"],
            "min_bout": best["min_bout"],
            "max_gap": best["max_gap"],
            "min_after_bout": 1,  # GUI default; bout F1 was not sensitive
        }
        metrics_path.write_text(json.dumps(meta, indent=2, default=str))

        # Also bake the optimized params into the shipped pkl
        clf_path = CLF_DIR / f"classifier_{name}.pkl"
        if clf_path.is_file():
            obj = joblib.load(clf_path)
            obj["best_thresh"] = float(best["thresh"])
            obj["min_bout"] = int(best["min_bout"])
            obj["max_gap"] = int(best["max_gap"])
            obj["min_after_bout"] = 1
            obj["ui_min_bout"] = int(best["min_bout"])
            obj["ui_max_gap"] = int(best["max_gap"])
            obj["ui_min_after_bout"] = 1
            obj["bout_f1_optimized"] = float(best["bout_f1_mean"])
            obj["bout_count_pearson_r"] = float(best["bout_count_pearson_r"])
            joblib.dump(obj, clf_path, compress=("lz4", 1))
            print(f"  updated {clf_path.name} with optimized params")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
