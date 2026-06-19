# PixelPaws — Processing Preferences

Canonical settings for the transcode → DLC pose → feature-extraction pipeline.
**Source of truth is `pipeline/pp_config.py`** (import from there; don't hardcode). This
doc is the human-readable summary.

## Transcode (ffmpeg)
| Setting | Value |
|---|---|
| Codec | `libx265` (HEVC) |
| Quality | `CRF 23` |
| Preset | `slow` |
| Audio | stripped (`-an`) |
| Scaling | none (re-encode at source resolution) |

Chosen via the 2026-05-30 compression sweep: CRF23/slow preserves paw photometry
(brightness/contour) for DLC while shrinking files ~25–30×. CPU (software) encode; GPU
is reserved for the DLC stage. (NVENC was rejected — lower per-bitrate fidelity.)

## DLC pose estimation
| Setting | Value |
|---|---|
| Model | `palmreader-500Mar25`, **iteration-2, shuffle 1, snapshot-best-460** |
| Config | `dlc_model/config_local.yaml` (`iteration: 2`, `snapshotindex: best`) |
| `analyze_videos` | `shuffle=1`, `batchsize=64`, `videotype=".mp4"` |
| Env | DeepLabCut 3.0.0rc13, torch 2.5.1+cu121 (CUDA) |

Weights + architecture live in `dlc_model/` (see `dlc_model/MODEL_README.md`).

## Feature extraction (PixelPaws co-pass)
| Setting | Value |
|---|---|
| Hash | `8aed1c22` (config-based) |
| Columns | **635** (verified union across the encyclopedia classifiers) |
| Brightness ROIs | `hrpaw, hlpaw, snout`, square size 40, pix threshold 0.3 |
| Optical flow | on, for `hrpaw, hlpaw, snout` |
| FPS | 60 |
| dtype | float32 caches (bitwise-identical predictions to float64) |

## Pipeline behavior
- Idempotent (skips existing outputs); one DLC subprocess per video; atomic feature pkl;
  verifies 635 cols + non-zero brightness/optical-flow.
- **Stops at features** — classification/analysis is cohort-specific and run deliberately.
- Optional `--delete-source-after-transcode` deletes each original only after its transcode
  output verifies (ffprobe-valid). Discord progress via `pipeline/pp_tracker.py`.
