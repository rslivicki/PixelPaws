# PixelPaws Global Classifier Encyclopedia (v1.0.0)

Build date: 2026-05-25
Total classifiers: 11 (Tier 1: 4 · Tier 2: 5 · Tier 3: 2)
Expected video FPS: **60**

## What this is
A bundle of trained behaviour classifiers for the PixelPaws GUI.
Each classifier ships with its `.pkl` + a per-classifier markdown doc +
`training_metrics.json` with raw numbers. All entries indexed in
`manifest.json`.

## Tiers

### Tier 1 — KEEP (frame F1 ≥ 0.85)
Production-ready. Use these for any analysis where frame-level
classification matters.

| name | frame F1 | bout F1 | bout-count r | best_thresh | min_bout | max_gap |
|---|---|---|---|---|---|---|
| Scratching | 0.952 ± 0.003 | 0.000 | 0.000 | 0.86 | 1 | 5 |
| Left_licking | 0.906 ± 0.026 | 0.000 | 0.000 | 0.53 | 3 | 5 |
| Facial_grooming | 0.900 ± 0.038 | 0.000 | 0.000 | 0.85 | 15 | 5 |
| rearing | 0.855 ± 0.028 | 0.000 | 0.000 | 0.54 | 1 | 5 |

### Tier 2 — BOUT_USEFUL (frame F1 < 0.85, but bout F1 ≥ 0.5 AND bout-count r ≥ 0.7)
Frame-level F1 is lower but bout-level performance is strong — these
predict the right NUMBER and TIMING of bouts per session even if
they miss some frames within. Excellent for "how many flinches did
this mouse have?" style questions.

| name | frame F1 | bout F1 | bout-count r | best_thresh | min_bout | max_gap |
|---|---|---|---|---|---|---|
| walking | 0.783 ± 0.046 | 0.791 | 0.000 | 0.35 | 15 | 20 |
| still | 0.767 ± 0.083 | 0.800 | 0.000 | 0.25 | 12 | 20 |
| body_grooming | 0.699 ± 0.086 | 0.519 | 0.737 | 0.85 | 30 | 40 |
| body_grooming_combined | 0.699 ± 0.086 | 0.519 | 0.737 | 0.85 | 30 | 40 |
| L_flinching | 0.574 ± 0.041 | 0.684 | 0.965 | 0.30 | 12 | 0 |

### Tier 3 — BELOW_GATE (experimental)
Documented for transparency but not recommended for production
analyses. Often limited by small data or by per-session label
variance.

| name | frame F1 | bout F1 | bout-count r | best_thresh | min_bout | max_gap |
|---|---|---|---|---|---|---|
| belly_groom | 0.430 ± 0.280 | 0.362 | 0.275 | 0.35 | 30 | 40 |
| back_groom | 0.366 ± 0.148 | 0.275 | -0.024 | 0.50 | 30 | 20 |

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
Canonical hash: `8aed1c22` —
pose v5 +
brightness v1 +
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
  bout-level criteria (F1≥0.5 + bout-count r≥0.7).

## See also
- `BUILD_NOTES.md` — methodology + design decisions
- `audit_report.csv` — Phase A audit of all existing classifiers
- `audit_label_alignment.log` — Per-session label/feature alignment audit
