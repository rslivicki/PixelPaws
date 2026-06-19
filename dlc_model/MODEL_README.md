# Current DLC network — `palmreader-500Mar25`, iteration-2 / shuffle 1 / snapshot-best-460

The most recent trained pose model used by the pipeline (adopted 2026-06-07).

## Files
| File | What |
|---|---|
| `snapshot-best-460.pt` | trained weights (ResNet50 backbone), **git-LFS** (~92 MB) |
| `pytorch_config.yaml` | model/architecture config for this shuffle |
| `config_local.yaml` | DLC project config (`iteration: 2`, `snapshotindex: best`, bodyparts, crop) |

There is no separate detector snapshot for this pose model.

## Provenance
Warm-started from iteration-1 `best-260` + 133 newly hand-labeled AOW(S1–9)+chloro_S6 frames
(1443-frame merged set), batch 24, 200 epochs (snapshots 261→460), proven recipe.

**Eval:** test RMSE 8.40 px / **3.58 px @ pcut 0.6**; paws 4.8–7 px (the AOW dark-frame
hindpaw-loss fix worked). The harder test split (now incl. dark AOW frames) is why raw RMSE
is a touch above iter-1's 3.36.

## Use
```python
import deeplabcut
deeplabcut.analyze_videos(
    r"...\dlc_model\config_local.yaml", [video], shuffle=1,
    videotype=".mp4", save_as_csv=False, batchsize=64)
```
DLC 3.0 resolves the **model tree from the config file's own directory** but the **shuffle
registry from the `project_path:` field** — so `config_local.yaml`'s `project_path` must point
at the real DLC project dir on the machine running inference. Env: DeepLabCut 3.0.0rc13,
torch 2.5.1+cu121 (CUDA). `pp_config.SHUFFLE` is already `1`.
