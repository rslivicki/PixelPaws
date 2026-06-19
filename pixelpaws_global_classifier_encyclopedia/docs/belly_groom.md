# belly_groom

**Tier 3: BELOW_GATE**

## Behaviour definition
Mouse grooms the belly specifically.

## Performance
- **Frame F1 (CV)**: 0.430 ± 0.280
- **Bout F1 (post-optimized)**: 0.362
- **Bout-count Pearson r**: 0.275
- **Mean bout duration error**: 1.10s

## Recommended post-processing
- `best_thresh`: 0.35
- `min_bout`: 30
- `min_after_bout`: 1
- `max_gap`: 40

These were chosen by joint sweep over (threshold, min_bout, max_gap) on
out-of-fold probabilities to maximize bout-level F1 at IoU≥0.5.

## Training data
- Sessions: 7
- Positive frames: 8,046

## Feature schema
- Pose v5 + brightness v1
- Optical flow: True
- square_size: [40, 40, 40]
- brightness body parts: ['hrpaw', 'hlpaw', 'snout']

## Status notes
Frame F1 < 0.85 AND bout-level criteria not met. Shipped for transparency but not recommended for production analyses. The main limitations are usually: (a) small training data, (b) high per-session label variance, or (c) the behaviour overlaps visually with adjacent behaviours.

## See also
- `../classifiers/classifier_belly_groom.pkl` — model artefact (load with joblib)
- `../performance_results/belly_groom/training_metrics.json` — full metrics + sweep results
- `../performance_results/belly_groom/oof.npz` — out-of-fold probabilities (for plot regen / re-sweep)
