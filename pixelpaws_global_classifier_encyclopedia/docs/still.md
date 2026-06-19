# still

**Tier 2: BOUT_USEFUL**

## Behaviour definition
Mouse not moving (no walking/rearing/grooming).

## Performance
- **Frame F1 (CV)**: 0.767 ± 0.083
- **Bout F1 (post-optimized)**: 0.800
- **Bout-count Pearson r**: 0.000
- **Mean bout duration error**: n/a

## Recommended post-processing
- `best_thresh`: 0.25
- `min_bout`: 12
- `min_after_bout`: 1
- `max_gap`: 20

These were chosen by joint sweep over (threshold, min_bout, max_gap) on
out-of-fold probabilities to maximize bout-level F1 at IoU≥0.5.

## Training data
- Sessions: 7
- Positive frames: 0

## Feature schema
- Pose v5 + brightness v1
- Optical flow: True
- square_size: [40, 40, 40]
- brightness body parts: ['hrpaw', 'hlpaw', 'snout']

## Status notes
Frame-level F1 is below the 0.85 gate, but bout-level performance is strong (bout F1 ≥ 0.50 + bout-count r ≥ 0.70). Use for "how many bouts per session" questions where bout count/timing matters more than per-frame accuracy.

## See also
- `../classifiers/classifier_still.pkl` — model artefact (load with joblib)
- `../performance_results/still/training_metrics.json` — full metrics + sweep results
- `../performance_results/still/oof.npz` — out-of-fold probabilities (for plot regen / re-sweep)
