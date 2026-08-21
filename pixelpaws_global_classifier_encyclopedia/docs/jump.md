# jump

## Behaviour definition
A push-off from both hind limbs launching all four paws off the floor.

## v1.1 status
- **CV-F1 (mean ± std)**: 0.888
- **Inclusion gate**: 0.85 frame F1 — **PASSES**
- **Status**: KEEP — the naloxone-precipitated withdrawal readout. Trained on the
  2606_AOW cohort with session S6 held out as an unseen test set; not validated on
  other cohorts. Jumping is brief and overlaps the mutually exclusive states, so it is
  analysed as discrete events rather than as a state in the seven-state battery.

## Operating point
- `best_thresh`: 0.68
- `min_bout`: 5
- `max_gap`: 0
- `min_after_bout`: 3

## Training data
- Cohort: `2606_AOW` (naloxone-precipitated opioid withdrawal, 9 female mice)
- Sessions: 7 (S6 held out)
- Annotated span: ~1.7–3.1 min per session (78,954 frames, 6.9% positive)

## Model
- Algorithm: XGBoost (binary classifier)
- Feature schema: pose v5 + brightness v1
