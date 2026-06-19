# body_grooming_combined

**Tier 2: BOUT_USEFUL**

## Behaviour definition
Union of body_grooming, back_groom, belly_groom (any non-facial grooming).

## Performance
- **Frame F1 (CV)**: 0.699 ± 0.086
- **Bout F1 (post-optimized)**: 0.519
- **Bout-count Pearson r**: 0.737
- **Mean bout duration error**: 1.11s

## Recommended post-processing
- `best_thresh`: 0.85
- `min_bout`: 30
- `min_after_bout`: 1
- `max_gap`: 40

These were chosen by joint sweep over (threshold, min_bout, max_gap) on
out-of-fold probabilities to maximize bout-level F1 at IoU≥0.5.

## Training data
- Sessions: 7
- Positive frames: 56,095

## Feature schema
- Pose v5 + brightness v1
- Optical flow: True
- square_size: [40, 40, 40]
- brightness body parts: ['hrpaw', 'hlpaw', 'snout']

## Status notes
Frame-level F1 is below the 0.85 gate, but bout-level performance is strong (bout F1 ≥ 0.50 + bout-count r ≥ 0.70). Use for "how many bouts per session" questions where bout count/timing matters more than per-frame accuracy.

## See also
- `../classifiers/classifier_body_grooming_combined.pkl` — model artefact (load with joblib)
- `../performance_results/body_grooming_combined/training_metrics.json` — full metrics + sweep results
- `../performance_results/body_grooming_combined/oof.npz` — out-of-fold probabilities (for plot regen / re-sweep)
