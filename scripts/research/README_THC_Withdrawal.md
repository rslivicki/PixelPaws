# THC Withdrawal — DLC + PixelPaws pipeline

Resume notes so a fresh session can pick up if anything crashes.

## Locations

| What | Path |
|---|---|
| DLC env (Python) | `C:\Users\Gereau\anaconda3\envs\DEEPLABCUT\python.exe` |
| DLC project | `E:\RSDLC\2511_RSGNKK_Blackbox` |
| DLC config | `E:\RSDLC\2511_RSGNKK_Blackbox\config.yaml` |
| PixelPaws repo | `E:\PixelPaws` |
| PixelPaws project | `E:\RSVIDS\Blackbox\260506_RS_THC_Withdrawal` |
| Videos | `E:\RSVIDS\Blackbox\260506_RS_THC_Withdrawal\Videos` |
| Results CSVs | `<project>\results\<session>_predictions.csv` |
| Analysis outputs | `<project>\analysis\` |
| DLC log | `E:\PixelPaws\scripts\research\thc_dlc.log` |
| Orchestrator log | `E:\PixelPaws\scripts\research\thc_orchestrator.log` |

## Cohort & group mapping

- 15 mice, each with `Baseline` and `Postdrug` recordings = 30 sessions total.
- Group rule (per user): odd-numbered mouse = THC, even = Vehicle (in `thc_withdrawal_group_analyze.py:parse_session`).
- Filename pattern: `THC<n>_<Baseline|Postdrug>.mp4` where `n` = 1..15.

## DLC config

- Engine: pytorch
- Model: `palmreader-500Mar25`
- Shuffle: **9**
- Snapshot: `best` (resolved to `snapshot_best-190` on disk)
- Batch size: 32 (do not raise — see below)

## Timing telemetry

`collect_timing_telemetry.py` parses `thc_dlc.log`, `snlt_cohort2_dlc.log`, and `thc_orchestrator.log` for per-video timings, probes each video for metadata (duration, fps, frame count, resolution, file size, bitrate), and writes a snapshot CSV to `scripts/research/telemetry/timing.csv`.

```bash
# One-shot snapshot
py -X utf8 scripts/research/collect_timing_telemetry.py

# Watch mode (re-parses every 5 min, overwrites snapshot CSV)
py -X utf8 scripts/research/collect_timing_telemetry.py --watch
```

Schema:

| Field | Source |
|---|---|
| session, project, video_path | pairing |
| file_size_mb, duration_s, fps, frame_count, width, height, bitrate_kbps | `cv2.VideoCapture` + `stat` |
| dlc_analyze_s, dlc_analyze_fps | last completed tqdm bar in DLC log |
| feature_extract_s, feature_extract_with_flow, feature_X_rows, feature_X_cols, feature_hash | `cached (...)` lines from extract scripts |
| session_total_s | first→last `[HH:MM:SS]` step inside each `=== <session> ===` block in chain log |

**Caveat for THC:** the DLC log was truncated several times during early restart cycles, so `dlc_analyze_s` is empty for THC sessions. SNLT cohort2 will get a clean DLC dataset (29 videos).

**Initial findings** (18 THC sessions, feature-extract with optical flow):

- Per-frame extraction time is **very consistent: 4.4–5.1 ms/frame** (~720p, ~118k frames typical)
- Per-video extract: 470–590 s (~8–10 min)
- All THC videos similar bitrate (~8000 kbps), so bitrate's effect on extract time is undersampled — SNLT cohort2 has different resolutions and may show more variance
- Predicted extract seconds for a new video ≈ `frame_count × 0.0047`

DLC predictions will land once SNLT cohort2 data is in. Expectation: ~50 it/s average at batch 32 on 2080 Ti, so DLC seconds ≈ `frame_count / 50` ± large variance from video content (more activity = more keypoint compute).

## GPU bottleneck analysis (RTX 2080 Ti, 11 GB)

Live readings during a typical batch=32 run:

```
GPU util:      70 %      (busy)
Memory bw:     49 %
VRAM used:     5.8 / 11 GB     ← only HALF used
Temp:          72 °C     (no thermal throttle)
Power draw:    282 / 300 W     ← 94 % of cap
GPU clocks:    1890 / 2280 MHz  ← only 83 % of max boost
```

**The card is power-capped, not memory-capped.** VRAM at 6/11 GB looks like wasted capacity, but it's not — at batch 32 ResNet-50 just doesn't need more. The throughput limit is the 300 W power envelope: clocks have already throttled from 2280 down to 1890 MHz to stay within budget.

This is exactly why the previous batch=64 attempt collapsed:
- batch=64 fed roughly 2× the work per step
- Higher sustained power → harder clock throttle → eventually 207 min/video by video 5
- batch=128 thrashed VRAM at 95 % memory utilization

Levers that would actually help (none of them are batch size):

1. `nvidia-smi -pl 330` — raise power limit (2080 Ti FE max ~330 W). Maybe +5–10 % throughput.
2. Better GPU cooling (fan curves, ambient temp). Smaller win.
3. NVDEC / hardware video decode if DLC's data loader is also a piece of the bottleneck.

**Stay at batch 32.** The "half-empty VRAM bar" is a red herring on this hardware.

## Output naming

Per-video DLC files in `Videos/`:

```
<video>DLC_Resnet50_palmreader-500Mar25shuffle9_snapshot_best-190.h5
<video>DLC_Resnet50_palmreader-500Mar25shuffle9_snapshot_best-190_filtered.h5
<video>DLC_Resnet50_palmreader-500Mar25shuffle9_snapshot_best-190_filtered.csv
+ pickle metadata files
```

## Pipeline scripts (in order)

| Step | Script | Env | What it does |
|---|---|---|---|
| 1 | `dlc_batch_thc_withdrawal.py` | DEEPLABCUT | analyze_videos + filterpredictions on all `Videos/*.mp4` (skip if shuffle-9 .h5 exists) |
| 2 | `thc_withdrawal_extract_with_flow.py` | regular `py` | feature cache **with optical flow** (skip-existing, separate hash from no-flow) |
| 3 | `thc_withdrawal_predict_all.py` | regular `py` | re-extract features (no flow) using union `bp_pixbrt` from classifiers, run all 5 classifiers, write `results/<session>_predictions.csv`. Hash on disk: `370c2fb2` |
| 4 | `thc_withdrawal_group_analyze.py` | regular `py` | argmax-assign states with priority list, build per-group transition matrices, plots, Discord upload |

`thc_post_dlc_chain.py` runs steps 2-4 sequentially.

## Behavior priority (argmax tie-break)

`['Facial_grooming', 'Left_licking', 'rearing', 'still', 'walking']` — earlier wins. Frames with no behavior → `noise`.

## Analysis defaults (per user request)

- `EXCLUDE_NOISE = True` — `noise` bouts dropped from transition matrices, denominators ignore noise frames.
- Primary readout: `% time immobile` (still column) — Postdrug THC vs Vehicle is the primary contrast.
- Outputs (`<project>/analysis/`): `state_occupancy.csv`, `transitions_<group>_<condition>.csv`, `bout_stats.csv`, `immobility.png`, `state_occupancy.png`, `transitions.png`, `transitions_diffs.png`, `bout_structure.png`, `ethograms.png`.

## Discord webhook

```

```

## How to resume from a crash

### A) DLC didn't finish (some `.h5` missing)
Re-run step 1. It auto-skips videos that already have a shuffle-9 `.h5`:
```
"C:/Users/Gereau/anaconda3/envs/DEEPLABCUT/python.exe" scripts/research/dlc_batch_thc_withdrawal.py
```

### B) DLC done, post-DLC failed mid-way
Just re-run the chain. Steps 2-3 skip-existing on the feature caches; step 3 will overwrite results CSVs (cheap).
```
PYTHONIOENCODING=utf-8 py -X utf8 scripts/research/thc_post_dlc_chain.py
```

### C) Only group analysis failed
Re-run step 4 alone:
```
PYTHONIOENCODING=utf-8 py -X utf8 scripts/research/thc_withdrawal_group_analyze.py
```

### D) Need to verify state on disk before resuming
- Count finished DLC: `ls Videos/ | grep -c "shuffle9.*_filtered.h5"` — should hit 30 when DLC is fully done.
- Count predictions: `ls results/ | wc -l` — should hit 30.
- Check classifiers loaded: `ls classifiers/` — expect 5 .pkl files (Facial_grooming, Left_licking, rearing, still, walking).

## Running it now (single command)

The Python orchestrator (`thc_orchestrator.py`) owns the whole run: launches DLC in DEEPLABCUT env, parses tqdm `it/s` from the log to watch live FPS, kills + restarts on stall, then chains post-DLC + SNLT cohort2 DLC. Live Discord progress message updates every 60s (no notification spam — Discord doesn't ping on edits).

```bash
py -X utf8 scripts/research/thc_orchestrator.py >> scripts/research/thc_orchestrator.log 2>&1 &
```

### Stall detection thresholds (`thc_orchestrator.py`)

- `MIN_HEALTHY_FPS = 8.0` — DLC inference at batchsize 32 normally hits 25–90 fps. <8 sustained = bad.
- `FPS_STALL_TIMEOUT_MIN = 10` — kill if FPS stays low for this long (during analyze phase only — filterpredictions phase is exempt).
- `LOG_MTIME_STALL_MIN = 5` — kill if log file isn't being written (process hung).
- `HARD_STALL_TIMEOUT_MIN = 240` — backstop: kill if no .h5 file appeared in 4 hours.
- `MAX_RESTARTS = 5` — give up after 5 consecutive stall-restarts.
- After a kill, sleeps `KILL_GRACE_SECONDS = 60` to let the GPU release VRAM.

### Discord notification cadence

- New posts (DO ping): orchestrator start, DLC complete, chain complete, stall+restart, SNLT start, SNLT complete, any failure.
- Edits (silent, just live progress): the progress bar message — overall `.h5` count + a *second* bar tracking the current video's frame index parsed from tqdm output. Refreshes every 60s. Format:

```
**THC Withdrawal -- DLC progress**
[████░░░░░░░░░░░░░░░░░░░░] 13/30 filtered .h5  (43.3%) (raw 14/30)
Current: `THC11_Postdrug.mp4`
[██████░░░░░░░░░░░░░░░░░░] 28000/117450 (23.8%)
Elapsed: 1h12m | ETA: ~7h | FPS: 48.0
```

The per-video frame count comes from a 100KB log-file tail + 3 regexes per minute — negligible CPU, zero GPU.

DLC takes ~30–40 min per ~2 GB video at batchsize 32 on the 2080 Ti. 18 new THC videos ≈ 9–12h, then SNLT cohort2 (29 videos) ≈ 15–20h. Total run is roughly a day.
