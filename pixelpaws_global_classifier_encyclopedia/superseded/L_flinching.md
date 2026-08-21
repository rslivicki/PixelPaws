# L_flinching

**Tier 2: BOUT_USEFUL**

## Behaviour definition
Mouse rapidly retracts/raises its left hindpaw (nocifensive flinch response).

## Performance
- **Frame F1 (CV)**: 0.574 ± 0.041
- **Bout F1 (post-optimized)**: 0.684
- **Bout-count Pearson r**: 0.965
- **Mean bout duration error**: 0.07s

## Recommended post-processing
- `best_thresh`: 0.30
- `min_bout`: 12
- `min_after_bout`: 1
- `max_gap`: 0

These were chosen by joint sweep over (threshold, min_bout, max_gap) on
out-of-fold probabilities to maximize bout-level F1 at IoU≥0.5.

## Training data
- Sessions: 21
- Positive frames: 29,393

## Feature schema
- Pose v5 + brightness v1
- Optical flow: True
- square_size: [40, 40, 40]
- brightness body parts: ['hrpaw', 'hlpaw', 'snout']

## Status notes
Frame-level F1 is below the 0.85 gate, but bout-level performance is strong (bout F1 ≥ 0.50 + bout-count r ≥ 0.70). Use for "how many bouts per session" questions where bout count/timing matters more than per-frame accuracy.

## See also
- `../classifiers/classifier_L_flinching.pkl` — model artefact (load with joblib)
- `../performance_results/L_flinching/training_metrics.json` — full metrics + sweep results
- `../performance_results/L_flinching/oof.npz` — out-of-fold probabilities (for plot regen / re-sweep)
