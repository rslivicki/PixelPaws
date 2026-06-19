"""Swap our BORIS-dense still/walking retrains for the existing GUI-trained
CSV-based pkls (which got 0.77/0.78 frame F1, ~0.80 bout F1). The GUI
training used sparse CSV labels with 35% pos rate; our BORIS-dense
labels can't reproduce that without rebuilding the sparse-label
infrastructure.

Pulls existing pkl from 2603_SNLT_JG/Baseline/classifiers/, adds
expected_fps + bundle metadata, writes to encyclopedia/.
"""
import json
import time
from pathlib import Path

import joblib

ENCY = Path(r"E:/PixelPaws/pixelpaws_global_classifier_encyclopedia")
CLF_DIR = ENCY / "classifiers"
PERF_DIR = ENCY / "performance_results"
DOCS_DIR = ENCY / "docs"

SWAPS = [
    {
        "name": "still",
        "src": Path(r"E:/RSVIDS/Blackbox/2603_SNLT_JG/Baseline/classifiers/"
                    r"PixelPaws_still_AllFeatures.pkl"),
        "definition": "Mouse not moving (no walking/rearing/grooming).",
        "source_cohort": "2603_SNLT_JG/Baseline",
    },
    {
        "name": "walking",
        "src": Path(r"E:/RSVIDS/Blackbox/2603_SNLT_JG/Baseline/classifiers/"
                    r"PixelPaws_walking_AllFeatures.pkl"),
        "definition": "Mouse walking/locomoting through the arena.",
        "source_cohort": "2603_SNLT_JG/Baseline",
    },
]


def main() -> int:
    for spec in SWAPS:
        print(f"\n== {spec['name']} ==")
        if not spec["src"].is_file():
            print(f"  !! source pkl missing: {spec['src']}")
            continue
        obj = joblib.load(spec["src"])

        cv_f1 = float(obj.get("mean_cv_f1") or 0.0)
        cv_std = float(obj.get("std_cv_f1") or 0.0)
        oof_best = obj.get("oof_best_f1")
        oof_best_params = obj.get("oof_best_params") or {}
        # The GUI's oof_best_params is the joint-grid result: thresh, min_bout,
        # min_after_bout, max_gap, f1 (= bout F1).
        best_thresh = float(oof_best_params.get("thresh", obj.get("best_thresh", 0.5)))
        min_bout = int(oof_best_params.get("min_bout", obj.get("min_bout", 3)))
        min_after_bout = int(oof_best_params.get("min_after_bout", obj.get("min_after_bout", 1)))
        max_gap = int(oof_best_params.get("max_gap", obj.get("max_gap", 5)))
        bout_f1 = float(oof_best_params.get("f1", 0.0))
        train_sessions = obj.get("training_sessions") or []

        obj["expected_fps"] = 60
        obj["encyclopedia_version"] = "1.0.0"
        obj["encyclopedia_name"] = spec["name"]
        obj["encyclopedia_definition"] = spec["definition"]
        obj["encyclopedia_status"] = "SHIP_GUI_CSV"
        obj["encyclopedia_source_cohort"] = spec["source_cohort"]
        obj["encyclopedia_provenance"] = (
            "GUI-trained on sparse CSV labels (Walking_still CSV with 35% "
            "pos rate, NOT BORIS-dense). Frame F1 reflects per-fold CV on "
            "labeled rows only; bout F1 (in oof_best_params) is the "
            "joint-grid post-processing optimum the GUI saved."
        )
        # Make sure the post-proc params on the pkl reflect the bout-optimized choice
        obj["best_thresh"] = best_thresh
        obj["min_bout"] = min_bout
        obj["min_after_bout"] = min_after_bout
        obj["max_gap"] = max_gap

        out_pkl = CLF_DIR / f"classifier_{spec['name']}.pkl"
        joblib.dump(obj, out_pkl, compress=("lz4", 1))
        print(f"  wrote {out_pkl.name}  ({out_pkl.stat().st_size/1e6:.1f} MB)  "
              f"frame F1={cv_f1:.3f}  bout F1={bout_f1:.3f}")

        # Write training_metrics.json that the assembler reads
        metrics = {
            "cv_f1_mean": cv_f1,
            "mean_cv_f1": cv_f1,
            "cv_f1_std": cv_std,
            "cv_f1_per_fold": [float(x) for x in (obj.get("cv_f1_scores") or [])],
            "oof_best_f1": (float(oof_best) if oof_best is not None else None),
            "best_thresh": best_thresh,
            "min_bout": min_bout,
            "min_after_bout": min_after_bout,
            "max_gap": max_gap,
            "n_train_sessions": len(train_sessions),
            "training_sessions": list(train_sessions),
            "feature_schema": {
                "pose_feature_version": int(obj.get("pose_feature_version") or 5),
                "brightness_feature_version": int(obj.get("brightness_feature_version") or 1),
                "include_optical_flow": bool(obj.get("include_optical_flow")),
                "square_size": obj.get("square_size"),
                "bp_pixbrt_list": obj.get("bp_pixbrt_list"),
            },
            "expected_fps": 60,
            "status": "SHIP_GUI_CSV",
            "source_cohort": spec["source_cohort"],
            # Pre-fill bout_optimization so the assembler reads it
            "bout_optimization": {
                "method": "GUI's _sweep_postprocessing (sparse CSV labels, "
                          "37×8×4×7=8288 combos)",
                "best": {
                    "thresh": best_thresh,
                    "min_bout": min_bout,
                    "max_gap": max_gap,
                    "min_after_bout": min_after_bout,
                    # The GUI's bout F1 is NOT separated from bout-count r; we
                    # only have F1. We'll mark r as None (assembler treats as 0
                    # → places in Tier 3 unless we override).
                    "bout_f1": bout_f1,
                    "bout_count_pearson_r": None,
                    "mean_bout_duration_error_sec": None,
                },
            },
            "recommended_post_processing": {
                "best_thresh": best_thresh,
                "min_bout": min_bout,
                "max_gap": max_gap,
                "min_after_bout": min_after_bout,
            },
        }
        perf_dir = PERF_DIR / spec["name"]
        perf_dir.mkdir(parents=True, exist_ok=True)
        (perf_dir / "training_metrics.json").write_text(
            json.dumps(metrics, indent=2, default=str)
        )
        print(f"  wrote training_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
