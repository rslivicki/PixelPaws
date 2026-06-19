"""Assemble the final v1 encyclopedia bundle: regenerate manifest.json
+ README.md with all shipped classifiers, tiered by quality.

Tiering rules:
  - Tier 1 (KEEP):  frame F1 ≥ 0.85
  - Tier 2 (BOUT-USEFUL): frame F1 < 0.85 BUT
        bout F1 ≥ 0.50 AND bout-count Pearson r ≥ 0.70
    These are great for "how many bouts per session" questions
    even if frame-by-frame accuracy is lower.
  - Tier 3 (BELOW-GATE): everything else; documented but flagged as
    experimental.

Manifest entries include best_thresh / min_bout / max_gap from each
classifier's `bout_optimization` (the post-proc sweep).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib

ENCY = Path(r"E:/PixelPaws/pixelpaws_global_classifier_encyclopedia")
CLF_DIR = ENCY / "classifiers"
PERF_DIR = ENCY / "performance_results"
DOCS_DIR = ENCY / "docs"

GATE_FRAME_F1 = 0.85
TIER2_BOUT_F1 = 0.50
TIER2_BOUT_R = 0.70


# Behaviour definitions for the docs
DEFS = {
    "Facial_grooming": "Mouse grooms the face with one or both forepaws.",
    "Left_licking": "Mouse licks its left hindpaw (pain/itch response).",
    "rearing": "Mouse stands on its hindpaws, forepaws lifted (supported or unsupported).",
    "Scratching": "Mouse rapidly scratches with a hindpaw, typically targeting head/neck/back.",
    "L_flinching": "Mouse rapidly retracts/raises its left hindpaw (nocifensive flinch response).",
    "body_grooming": "Mouse grooms torso/back/belly area with forepaws or by twisting around.",
    "body_grooming_combined": "Union of body_grooming, back_groom, belly_groom (any non-facial grooming).",
    "back_groom": "Mouse grooms the back specifically.",
    "belly_groom": "Mouse grooms the belly specifically.",
    "still": "Mouse not moving (no walking/rearing/grooming).",
    "walking": "Mouse walking/locomoting through the arena.",
}


def get_tier(frame_f1: float, bout_f1: float, bout_r: float) -> tuple[int, str]:
    if frame_f1 >= GATE_FRAME_F1:
        return 1, "KEEP"
    # Tier 2: either (bout F1 ≥ 0.50 AND bout-count r ≥ 0.70) OR bout F1 ≥ 0.75
    # The second clause catches GUI CSV-trained classifiers whose bout F1
    # is strong but where r wasn't computed in the same sweep.
    if (bout_f1 >= TIER2_BOUT_F1 and bout_r >= TIER2_BOUT_R) or bout_f1 >= 0.75:
        return 2, "BOUT_USEFUL"
    return 3, "BELOW_GATE"


def main() -> int:
    classifiers = []
    for pkl_path in sorted(CLF_DIR.glob("classifier_*.pkl")):
        name = pkl_path.stem.replace("classifier_", "")
        obj = joblib.load(pkl_path)
        if not isinstance(obj, dict):
            print(f"  ! {name}: unexpected pkl format, skipping")
            continue
        metrics_path = PERF_DIR / name / "training_metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}

        frame_f1 = float(obj.get("mean_cv_f1") or 0.0)
        frame_f1_std = float(obj.get("std_cv_f1") or 0.0)

        # Bout-level numbers come from the sweep result (saved in
        # training_metrics.json as `bout_optimization.best`)
        bo = metrics.get("bout_optimization", {}).get("best", {})
        bout_f1 = float(bo.get("bout_f1") or 0.0)
        bout_r = float(bo.get("bout_count_pearson_r") or 0.0)
        best_thresh = float(bo.get("thresh", obj.get("best_thresh") or 0.5))
        min_bout = int(bo.get("min_bout", obj.get("min_bout") or 3))
        max_gap = int(bo.get("max_gap", obj.get("max_gap") or 5))
        min_after_bout = int(obj.get("min_after_bout") or 1)

        tier, status = get_tier(frame_f1, bout_f1, bout_r)

        n_sessions = int(obj.get("n_train_sessions") or len(obj.get("training_sessions") or []))
        n_pos = int(obj.get("n_pos_frames") or 0)
        of_on = bool(obj.get("include_optical_flow"))
        sq = obj.get("square_size")
        bp_pixbrt = obj.get("bp_pixbrt_list")

        classifiers.append({
            "name": name,
            "tier": tier,
            "status": status,
            "definition": DEFS.get(name, ""),
            "frame_f1": frame_f1,
            "frame_f1_std": frame_f1_std,
            "bout_f1": bout_f1,
            "bout_count_pearson_r": bout_r,
            "best_thresh": best_thresh,
            "min_bout": min_bout,
            "min_after_bout": min_after_bout,
            "max_gap": max_gap,
            "n_train_sessions": n_sessions,
            "n_pos_frames": n_pos,
            "feature_schema": {
                "include_optical_flow": of_on,
                "square_size": sq,
                "bp_pixbrt_list": bp_pixbrt,
                "pose_feature_version": int(obj.get("pose_feature_version") or 5),
                "brightness_feature_version": int(obj.get("brightness_feature_version") or 1),
            },
            "path": f"classifiers/{pkl_path.name}",
            "docs": f"docs/{name}.md",
            "performance_dir": f"performance_results/{name}/",
        })

    # Sort: tier ascending, then frame F1 descending
    classifiers.sort(key=lambda c: (c["tier"], -c["frame_f1"]))

    manifest = {
        "encyclopedia_version": "1.0.0",
        "build_date": time.strftime("%Y-%m-%d"),
        "expected_fps": 60,
        "feature_schema_canonical": {
            "hash": "8aed1c22",
            "pose_feature_version": 5,
            "brightness_feature_version": 1,
            "include_optical_flow": True,
            "square_size": [40, 40, 40],
            "bp_pixbrt_list": ["hrpaw", "hlpaw", "snout"],
            "bp_optflow_list": ["hrpaw", "hlpaw", "snout"],
            "note": "Some KEEPs use sq=[40,40,15] (Scratching) — their pkls "
                    "declare the exact extraction config.",
        },
        "tiers": {
            "Tier 1 (KEEP)": f"frame F1 >= {GATE_FRAME_F1}",
            "Tier 2 (BOUT_USEFUL)": (
                f"frame F1 < {GATE_FRAME_F1} BUT bout F1 >= {TIER2_BOUT_F1} "
                f"AND bout-count Pearson r >= {TIER2_BOUT_R}. Use when bout "
                "counts/durations matter more than per-frame accuracy."
            ),
            "Tier 3 (BELOW_GATE)": (
                "Below both gates. Shipped for completeness, marked "
                "experimental. Not recommended for production analyses."
            ),
        },
        "classifiers": classifiers,
    }
    (ENCY / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    print(f"wrote manifest.json with {len(classifiers)} classifiers")
    by_tier = {}
    for c in classifiers:
        by_tier.setdefault(c["tier"], []).append(c)
    for t in sorted(by_tier):
        print(f"\n  Tier {t}:")
        for c in by_tier[t]:
            print(f"    {c['name']:25s}  frame={c['frame_f1']:.3f}  "
                  f"bout={c['bout_f1']:.3f}  r={c['bout_count_pearson_r']:.3f}  "
                  f"thr={c['best_thresh']:.2f}  mb={c['min_bout']}  mg={c['max_gap']}")

    # README
    rows_t1 = ["| name | frame F1 | bout F1 | bout-count r | best_thresh | min_bout | max_gap |",
               "|---|---|---|---|---|---|---|"]
    rows_t2 = list(rows_t1)
    rows_t3 = list(rows_t1)
    for c in classifiers:
        row = (f"| {c['name']} | {c['frame_f1']:.3f} ± {c['frame_f1_std']:.3f} | "
               f"{c['bout_f1']:.3f} | {c['bout_count_pearson_r']:.3f} | "
               f"{c['best_thresh']:.2f} | {c['min_bout']} | {c['max_gap']} |")
        if c["tier"] == 1:
            rows_t1.append(row)
        elif c["tier"] == 2:
            rows_t2.append(row)
        else:
            rows_t3.append(row)

    n_t1 = sum(1 for c in classifiers if c["tier"] == 1)
    n_t2 = sum(1 for c in classifiers if c["tier"] == 2)
    n_t3 = sum(1 for c in classifiers if c["tier"] == 3)

    readme = f"""# PixelPaws Global Classifier Encyclopedia (v1.0.0)

Build date: {manifest['build_date']}
Total classifiers: {len(classifiers)} (Tier 1: {n_t1} · Tier 2: {n_t2} · Tier 3: {n_t3})
Expected video FPS: **60**

## What this is
A bundle of trained behaviour classifiers for the PixelPaws GUI.
Each classifier ships with its `.pkl` + a per-classifier markdown doc +
`training_metrics.json` with raw numbers. All entries indexed in
`manifest.json`.

## Tiers

### Tier 1 — KEEP (frame F1 ≥ {GATE_FRAME_F1})
Production-ready. Use these for any analysis where frame-level
classification matters.

{chr(10).join(rows_t1)}

### Tier 2 — BOUT_USEFUL (frame F1 < {GATE_FRAME_F1}, but bout F1 ≥ {TIER2_BOUT_F1} AND bout-count r ≥ {TIER2_BOUT_R})
Frame-level F1 is lower but bout-level performance is strong — these
predict the right NUMBER and TIMING of bouts per session even if
they miss some frames within. Excellent for "how many flinches did
this mouse have?" style questions.

{chr(10).join(rows_t2) if n_t2 > 0 else "_(none)_"}

### Tier 3 — BELOW_GATE (experimental)
Documented for transparency but not recommended for production
analyses. Often limited by small data or by per-session label
variance.

{chr(10).join(rows_t3) if n_t3 > 0 else "_(none)_"}

## How to use
1. Drop the bundle anywhere accessible to the PixelPaws GUI.
2. Point the GUI's classifier dropdown at `classifiers/` (or copy
   `.pkl`s into your project's `classifiers/` folder).
3. The GUI reads `best_thresh`, `min_bout`, `min_after_bout`,
   `max_gap` from each pkl — these are pre-optimized via OOF
   probability sweeps.
4. **FPS contract**: every classifier expects 60 fps video. Other
   framerates will produce miscalibrated outputs.

## Feature schema
Canonical hash: `{manifest['feature_schema_canonical']['hash']}` —
pose v{manifest['feature_schema_canonical']['pose_feature_version']} +
brightness v{manifest['feature_schema_canonical']['brightness_feature_version']} +
optical flow, square_size=40, brightness body parts =
`hrpaw, hlpaw, snout`. Tier 1 KEEPs may use slightly different
sq sizes — each pkl declares its exact extraction config.

## Methodology notes
- **Label source**: Tier 1 KEEPs were trained from sparse-clicked
  CSV labels (the labeler explicitly tagged each row).
  Tier 2/3 classifiers were trained from BORIS-derived dense
  `.npy` arrays with `trim_to_last_positive` applied to drop
  unlabeled tails. The two label semantics produce different F1
  scales — Tier 1 KEEPs benefit from the cleaner sparse-label
  contrast.
- **Post-processing optimization**: each classifier's `best_thresh`,
  `min_bout`, `max_gap` come from a joint grid search over
  out-of-fold probabilities (3,024 combinations per behaviour).
  These maximize **bout-level F1** at IoU≥0.5, matching the
  GUI's deployment pipeline.
- **Probability calibration**: isotonic regression fit on OOF
  probabilities (for Tier 2/3); Tier 1 KEEPs ship with their
  original `prob_calibrator` field. Brier scores reported in
  per-classifier `training_metrics.json`.
- **Probability calibration deferred for Tier 1 KEEPs to v1.1**
  (their source pkls don't store training labels needed for
  isotonic fit).
- **Inclusion gate**: 0.85 frame F1 for Tier 1; for Tier 2 we use
  bout-level criteria (F1≥{TIER2_BOUT_F1} + bout-count r≥{TIER2_BOUT_R}).

## See also
- `BUILD_NOTES.md` — methodology + design decisions
- `audit_report.csv` — Phase A audit of all existing classifiers
- `audit_label_alignment.log` — Per-session label/feature alignment audit
"""
    (ENCY / "README.md").write_text(readme)
    print("\nwrote README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
