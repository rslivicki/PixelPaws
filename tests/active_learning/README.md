# Active-learning test project

Exercises the redesigned AL engine (`active_learning_engine.py`) + headless runner
(`scripts/research/run_active_learning.py`) end-to-end.

## Files
- `make_fixture.py` — generates a synthetic AL project (N sessions: feature pkl +
  labels csv with a `gt` ground-truth column) + a "pretend pruned" classifier
  (`feature_names_in_` is a strict subset of the cache). No DLC/video needed.
- `test_active_learning.py` — self-contained test: engine units, an end-to-end
  oracle-labeled loop (asserts F1 improves, class-balance + temporal-gap + global
  pooling), warm-start via probability injection, and a headless-runner subprocess
  run that converges and writes the learning-curve JSON.
- `realdata_smoke.py` — warm-starts from a REAL encyclopedia classifier on a REAL
  `8aed1c22` cache (augment → predict → engine). Skips if the real files are absent.

## Run
```
"C:\Program Files\Python310\python.exe" tests\active_learning\test_active_learning.py
"C:\Program Files\Python310\python.exe" tests\active_learning\realdata_smoke.py     # optional, needs real data
```
`test_active_learning` is also pytest-discoverable.

## Try a real warm-started run (interactive labeling in the terminal)
```
py -X utf8 scripts\research\run_active_learning.py ^
  --features <S1_features_8aed1c22.pkl> --labels <S1_labels.csv> --dlc <S1_DLC...best-190.h5> ^
  --features <S2_features_8aed1c22.pkl> --labels <S2_labels.csv> --dlc <S2_DLC...best-190.h5> ^
  --behavior Scratching --batch-size 20 ^
  --init-classifier <encyclopedia>\classifiers\classifier_Scratching.pkl
```
Drop `--init-classifier`/`--dlc` to start from a hand-labeled seed; add `--oracle gt`
(when the labels CSV has a `gt` column) for an automated convergence run.
For real labeling with video context, use the GUI Active-Learning tab instead.
