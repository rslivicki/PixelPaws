# Scratching

## Behaviour definition
Mouse rapidly scratches with a hindpaw, typically targeting head/neck/back.

## v1 status
- **CV-F1 (mean ± std)**: 0.952 ± 0.003
- **Inclusion gate**: 0.85 frame F1 — **PASSES** (0.952 ≥ 0.85)
- **Status**: KEEP — shipped as-is in v1 (carried over from `2510_Blackbox_Rimonabant` cohort).

## Training data
- Cohort: `2510_Blackbox_Rimonabant`
- Sessions: 5 (per-session list in the source pkl's
  `training_sessions` field).

## Model
- Algorithm: XGBoost (binary classifier)
- Feature schema: pose v5 + brightness v1
   (square_size=[40, 40, 15],
  bp_pixbrt=['hrpaw', 'hlpaw', 'snout'])
- Expected video FPS: 60

## Post-processing
- best_thresh: 0.8600000000000001
- min_bout: 1
- min_after_bout: 0
- max_gap: 0

## Notes for v1
- Probability calibration: **deferred to v1.1.** Source pkl does not store
  the training labels needed to fit an isotonic calibrator without
  re-running the training data-load step.
- Performance panels (threshold curve, SHAP beeswarm, confusion matrix):
  generated as a separate Phase D batch — see
  `../performance_results/Scratching/`.
- Seed-stability re-runs (additional seeds 1,2,3,4): deferred to v1.1.

## Known limitations
- Trained on the listed cohort only; cross-strain / cross-rig
  generalisation is untested at v1.
- Frame F1 is on session-level cross-validation; bout-level metrics
  (IoU≥0.5 matching, bout-count correlation, mean bout duration error)
  are computed by Phase D and reported in `training_metrics.json`.

## Source pkl
Original: `2510_Blackbox_Rimonabant` cohort. See `../classifiers/classifier_Scratching.pkl`
for the wrapped artefact (identical model bytes, plus added `expected_fps`,
`encyclopedia_version`, and other metadata).
