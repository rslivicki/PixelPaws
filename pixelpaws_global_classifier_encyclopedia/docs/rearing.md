# rearing

## Behaviour definition
Mouse stands on its hindpaws, forepaws lifted off the floor (supported or unsupported).

## v1 status
- **CV-F1 (mean ± std)**: 0.855 ± 0.028
- **Inclusion gate**: 0.85 frame F1 — **PASSES** (0.855 ≥ 0.85)
- **Status**: KEEP — shipped as-is in v1 (carried over from `260506_RS_THC_Withdrawal` cohort).

## Training data
- Cohort: `260506_RS_THC_Withdrawal`
- Sessions: 7 (per-session list in the source pkl's
  `training_sessions` field).

## Model
- Algorithm: XGBoost (binary classifier)
- Feature schema: pose v5 + brightness v1
  + optical flow (square_size=[40, 40, 40],
  bp_pixbrt=['hrpaw', 'hlpaw', 'snout'])
- Expected video FPS: 60

## Post-processing
- best_thresh: 0.54
- min_bout: 1
- min_after_bout: 0
- max_gap: 0

## Notes for v1
- Probability calibration: **deferred to v1.1.** Source pkl does not store
  the training labels needed to fit an isotonic calibrator without
  re-running the training data-load step.
- Performance panels (threshold curve, SHAP beeswarm, confusion matrix):
  generated as a separate Phase D batch — see
  `../performance_results/rearing/`.
- Seed-stability re-runs (additional seeds 1,2,3,4): deferred to v1.1.

## Known limitations
- Trained on the listed cohort only; cross-strain / cross-rig
  generalisation is untested at v1.
- Frame F1 is on session-level cross-validation; bout-level metrics
  (IoU≥0.5 matching, bout-count correlation, mean bout duration error)
  are computed by Phase D and reported in `training_metrics.json`.

## Source pkl
Original: `260506_RS_THC_Withdrawal` cohort. See `../classifiers/classifier_rearing.pkl`
for the wrapped artefact (identical model bytes, plus added `expected_fps`,
`encyclopedia_version`, and other metadata).
