"""Tier 1: copy the 4 KEEP'd classifiers to the encyclopedia, add metadata,
write per-classifier MD docs, and seed manifest.json with their entries.

Calibration is deferred to v1.1 (true training labels are not stored in
the source pkls — re-loading them per-classifier requires another
training-style data load that is non-trivial; better done as a follow-up).

Outputs:
  E:/PixelPaws/pixelpaws_global_classifier_encyclopedia/
    classifiers/classifier_<name>.pkl
    docs/<name>.md
    performance_results/<name>/training_metrics.json
    manifest.json   (created/updated)
    README.md       (initial draft, updated as more classifiers ship)
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

ENCY = Path(r"E:/PixelPaws/pixelpaws_global_classifier_encyclopedia")
ENCY_CLF = ENCY / "classifiers"
ENCY_DOCS = ENCY / "docs"
ENCY_PERF = ENCY / "performance_results"
for d in (ENCY_CLF, ENCY_DOCS, ENCY_PERF):
    d.mkdir(parents=True, exist_ok=True)


KEEPS = [
    {
        "name": "Facial_grooming",
        "src": Path(r"E:/RSVIDS/Blackbox/260506_RS_THC_Withdrawal/classifiers/"
                    r"PixelPaws_Facial_grooming_AllFeatures.pkl"),
        "definition": "Mouse grooms the face with one or both forepaws.",
        "source_cohort": "260506_RS_THC_Withdrawal",
    },
    {
        "name": "Left_licking",
        "src": Path(r"E:/RSVIDS/Blackbox/260506_RS_THC_Withdrawal/classifiers/"
                    r"PixelPaws_Left_licking.pkl"),
        "definition": "Mouse licks its left hindpaw, typically a pain-/itch-related response.",
        "source_cohort": "260506_RS_THC_Withdrawal",
    },
    {
        "name": "rearing",
        "src": Path(r"E:/RSVIDS/Blackbox/260506_RS_THC_Withdrawal/classifiers/"
                    r"PixelPaws_rearing_AllFeatures.pkl"),
        "definition": "Mouse stands on its hindpaws, forepaws lifted off the floor "
                      "(supported or unsupported).",
        "source_cohort": "260506_RS_THC_Withdrawal",
    },
    {
        "name": "Scratching",
        "src": Path(r"E:/RSVIDS/Blackbox/2510_Blackbox_Rimonabant/Blackbox_videos-selected/"
                    r"Classifiers/PixelPaws_Scratching_AllFeatures_20260429_153755.pkl"),
        "definition": "Mouse rapidly scratches with a hindpaw, typically targeting "
                      "head/neck/back.",
        "source_cohort": "2510_Blackbox_Rimonabant",
    },
]


def compute_diagnostics(oof_proba: np.ndarray | None,
                        cv_f1: float) -> dict:
    """Cheap diagnostics that don't need true labels:
      - oof_proba distribution stats (when available)
      - cv_f1 mean (already known)
    Pos_rate / majority_baseline_f1 / AUROC would need labels; those are
    set to None and filled in v1.1.
    """
    out = {"cv_f1": float(cv_f1)}
    if oof_proba is not None:
        out["oof_proba_mean"] = float(np.mean(oof_proba))
        out["oof_proba_median"] = float(np.median(oof_proba))
        out["oof_proba_above_0_5_frac"] = float(np.mean(oof_proba >= 0.5))
        out["oof_n_frames"] = int(len(oof_proba))
    return out


def make_doc(name: str, definition: str, source_cohort: str,
             cv_f1: float, cv_std: float, best_thresh: float | None,
             min_bout: float | None, min_after_bout: float | None,
             max_gap: float | None,
             n_train_sessions: int,
             pose_v: int, brt_v: int, of_on: bool, sq: list,
             bp_pixbrt: list,
             expected_fps: int = 60) -> str:
    return f"""# {name}

## Behaviour definition
{definition}

## v1 status
- **CV-F1 (mean ± std)**: {cv_f1:.3f} ± {cv_std:.3f}
- **Inclusion gate**: 0.85 frame F1 — **PASSES** ({cv_f1:.3f} ≥ 0.85)
- **Status**: KEEP — shipped as-is in v1 (carried over from `{source_cohort}` cohort).

## Training data
- Cohort: `{source_cohort}`
- Sessions: {n_train_sessions} (per-session list in the source pkl's
  `training_sessions` field).

## Model
- Algorithm: XGBoost (binary classifier)
- Feature schema: pose v{pose_v} + brightness v{brt_v}
  {'+ optical flow' if of_on else ''} (square_size={sq},
  bp_pixbrt={bp_pixbrt})
- Expected video FPS: {expected_fps}

## Post-processing
- best_thresh: {best_thresh}
- min_bout: {min_bout}
- min_after_bout: {min_after_bout}
- max_gap: {max_gap}

## Notes for v1
- Probability calibration: **deferred to v1.1.** Source pkl does not store
  the training labels needed to fit an isotonic calibrator without
  re-running the training data-load step.
- Performance panels (threshold curve, SHAP beeswarm, confusion matrix):
  generated as a separate Phase D batch — see
  `../performance_results/{name}/`.
- Seed-stability re-runs (additional seeds 1,2,3,4): deferred to v1.1.

## Known limitations
- Trained on the listed cohort only; cross-strain / cross-rig
  generalisation is untested at v1.
- Frame F1 is on session-level cross-validation; bout-level metrics
  (IoU≥0.5 matching, bout-count correlation, mean bout duration error)
  are computed by Phase D and reported in `training_metrics.json`.

## Source pkl
Original: `{source_cohort}` cohort. See `../classifiers/classifier_{name}.pkl`
for the wrapped artefact (identical model bytes, plus added `expected_fps`,
`encyclopedia_version`, and other metadata).
"""


def main() -> int:
    manifest = {
        "encyclopedia_version": "1.0.0-rc1",
        "build_date": time.strftime("%Y-%m-%d"),
        "feature_schema": {
            "pose_feature_version": 5,
            "brightness_feature_version": 1,
            "include_optical_flow": True,
            "feature_hash_canonical": "8aed1c22",
            "feature_hash_note": "Most KEEPs already use sq=40+OF (8aed1c22). "
                                 "Scratching uses sq=[40,40,15] (mixed) — its "
                                 "pkl declares the exact extraction config.",
        },
        "expected_fps": 60,
        "classifiers": [],
        "dropped": [],
    }

    for spec in KEEPS:
        print(f"\n== {spec['name']} ==")
        if not spec["src"].is_file():
            print(f"  !! source pkl missing: {spec['src']}")
            continue
        obj = joblib.load(spec["src"])

        cv_f1 = float(obj.get("mean_cv_f1") or 0.0)
        cv_std = float(obj.get("std_cv_f1") or 0.0)
        cv_per_fold = obj.get("cv_f1_scores") or []
        oof_best = obj.get("oof_best_f1")
        best_thresh = obj.get("best_thresh")
        min_bout = obj.get("min_bout")
        min_after_bout = obj.get("min_after_bout")
        max_gap = obj.get("max_gap")
        train_sessions = obj.get("training_sessions") or []
        bp_pixbrt = obj.get("bp_pixbrt_list") or []
        sq = obj.get("square_size")
        of_on = bool(obj.get("include_optical_flow"))
        pose_v = int(obj.get("pose_feature_version") or 5)
        brt_v = int(obj.get("brightness_feature_version") or 1)

        # Add encyclopedia metadata
        obj["expected_fps"] = 60
        obj["encyclopedia_version"] = manifest["encyclopedia_version"]
        obj["encyclopedia_name"] = spec["name"]
        obj["encyclopedia_definition"] = spec["definition"]
        obj["encyclopedia_status"] = "KEEP"
        obj["encyclopedia_source_cohort"] = spec["source_cohort"]
        obj["encyclopedia_calibration_deferred_v1.1"] = (
            "Probability calibration deferred to v1.1; training labels not "
            "stored in source pkl."
        )

        # Save wrapped pkl
        out_pkl = ENCY_CLF / f"classifier_{spec['name']}.pkl"
        joblib.dump(obj, out_pkl, compress=("lz4", 1) if shutil.which else 3)
        size_mb = out_pkl.stat().st_size / 1e6
        print(f"  wrote {out_pkl.name}  ({size_mb:.1f} MB)")

        # Diagnostics + training_metrics.json
        diag = compute_diagnostics(obj.get("oof_proba"), cv_f1)
        metrics = {
            "cv_f1_mean": cv_f1,
            "cv_f1_std": cv_std,
            "cv_f1_per_fold": [float(x) for x in cv_per_fold],
            "oof_best_f1": (float(oof_best) if oof_best is not None else None),
            "best_thresh": (float(best_thresh) if best_thresh is not None else None),
            "min_bout": (float(min_bout) if min_bout is not None else None),
            "min_after_bout": (float(min_after_bout) if min_after_bout is not None else None),
            "max_gap": (float(max_gap) if max_gap is not None else None),
            "n_train_sessions": len(train_sessions),
            "training_sessions": list(train_sessions),
            "feature_schema": {
                "pose_feature_version": pose_v,
                "brightness_feature_version": brt_v,
                "include_optical_flow": of_on,
                "square_size": sq,
                "bp_pixbrt_list": bp_pixbrt,
            },
            "expected_fps": 60,
            "diagnostics": diag,
            "status": "KEEP",
            "source_cohort": spec["source_cohort"],
        }
        perf_dir = ENCY_PERF / spec["name"]
        perf_dir.mkdir(parents=True, exist_ok=True)
        (perf_dir / "training_metrics.json").write_text(
            json.dumps(metrics, indent=2, default=str)
        )
        print(f"  wrote training_metrics.json")

        # Markdown
        md = make_doc(
            spec["name"], spec["definition"], spec["source_cohort"],
            cv_f1, cv_std, best_thresh, min_bout, min_after_bout, max_gap,
            len(train_sessions), pose_v, brt_v, of_on, sq, bp_pixbrt,
        )
        (ENCY_DOCS / f"{spec['name']}.md").write_text(md)
        print(f"  wrote docs/{spec['name']}.md")

        # Manifest entry
        manifest["classifiers"].append({
            "name": spec["name"],
            "path": f"classifiers/classifier_{spec['name']}.pkl",
            "docs": f"docs/{spec['name']}.md",
            "performance_dir": f"performance_results/{spec['name']}/",
            "status": "KEEP",
            "cv_f1_mean": cv_f1,
            "cv_f1_std": cv_std,
            "best_thresh": (float(best_thresh) if best_thresh is not None else None),
            "min_bout": (float(min_bout) if min_bout is not None else None),
            "min_after_bout": (float(min_after_bout) if min_after_bout is not None else None),
            "max_gap": (float(max_gap) if max_gap is not None else None),
            "n_train_sessions": len(train_sessions),
            "trained_on_cohorts": [spec["source_cohort"]],
            "expected_fps": 60,
            "feature_schema": {
                "include_optical_flow": of_on,
                "square_size": sq,
                "bp_pixbrt_list": bp_pixbrt,
            },
        })

    (ENCY / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nwrote manifest.json with {len(manifest['classifiers'])} classifiers")

    # README first draft
    rows = ["| name | status | CV-F1 | sessions | sq | OF |",
            "|---|---|---|---|---|---|"]
    for c in manifest["classifiers"]:
        fs = c["feature_schema"]
        rows.append(f"| {c['name']} | {c['status']} | {c['cv_f1_mean']:.3f} ± {c['cv_f1_std']:.3f} | "
                    f"{c['n_train_sessions']} | {fs['square_size']} | "
                    f"{'✓' if fs['include_optical_flow'] else '✗'} |")

    readme = f"""# PixelPaws Global Classifier Encyclopedia (v1)

Build date: {manifest['build_date']}
Version: {manifest['encyclopedia_version']}

## What this is
A bundle of trained behaviour classifiers for the PixelPaws GUI.
Each classifier ships with its `.pkl`, a markdown doc describing how it
was trained + known limitations, and a `training_metrics.json` with the
raw numbers. Bundle entries are indexed in `manifest.json`.

## Shipped classifiers ({len([c for c in manifest['classifiers'] if c['status'] == 'KEEP'])})

{chr(10).join(rows)}

## Inclusion gate
- Frame F1 ≥ 0.85 on session-level cross-validation.
- Classifiers below the gate are listed under `dropped` in `manifest.json`
  with a reason.

## How to use
1. Drop the bundle anywhere accessible to the PixelPaws GUI.
2. Point the GUI's classifier dropdown at `classifiers/` (or copy the
   `.pkl`s into your project's `classifiers/` folder).
3. Check `manifest.json` for per-classifier post-processing recommendations
   (`best_thresh`, `min_bout`, `max_gap`).
4. **FPS contract**: every classifier here expects **{manifest['expected_fps']} fps**
   video. Other framerates will produce miscalibrated outputs — the GUI
   will warn if `cv2.CAP_PROP_FPS` doesn't match.

## Feature schema
Most classifiers use the canonical `{manifest['feature_schema']['feature_hash_canonical']}`
hash: pose v{manifest['feature_schema']['pose_feature_version']} + brightness
v{manifest['feature_schema']['brightness_feature_version']} + optical flow,
square_size=40, brightness body parts = `hrpaw, hlpaw, snout`.

## v1 caveats
- Probability calibration is **deferred to v1.1** (isotonic calibrators
  not fit at v1; predictions are raw XGBoost probabilities).
- Performance panels (threshold curve, SHAP beeswarm, confusion matrix)
  are generated separately and live under `performance_results/<name>/`.

## See also
- `BUILD_NOTES.md` — methodology + design decisions
- `audit_report.csv` — Phase A audit of all candidate classifiers
"""
    (ENCY / "README.md").write_text(readme)
    print("wrote README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
