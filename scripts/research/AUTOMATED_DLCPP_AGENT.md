# Automated DLC + PixelPaws Pipeline Agent

End-to-end pipeline that takes a folder of mouse-behavior videos and produces SNLT-style group-level behavioral analyses (transitions, state occupancy, immobility, bout structure, ethograms) with live Discord progress and figures auto-pushed at completion. Designed so you can hand me a new cohort folder + a blinding key and say "run the DLCPP agent" and I'll execute it with the same conventions.

---

## What this agent does

1. **DLC inference** (DEEPLABCUT env, GPU) on every `.mp4` in the video folder
2. **filterpredictions** on every video (produces `_filtered.h5`)
3. **Feature extraction** WITH optical flow, single pass per video (hash `8aed1c22` when union config matches; otherwise a new hash is computed)
4. **5-classifier prediction** (Facial_grooming, Left_licking, rearing, still, walking) per the SNLT `for_claude.json` spec
5. **Group-level analysis** with fixed-window caps and the standard transition-matrix convention (zero diagonal)
6. **Discord push** of figures + summary at each major step

Optionally re-runs the chain on filtered .h5 in parallel folders for comparison.

---

## Settings from the SNTX cohort 2 run (2026-05-10/11)

These are the conventions baked into this agent. Reuse for any new cohort unless explicitly overridden.

### DLC

| Setting | Value |
|---|---|
| DLC env (Python) | `C:\Users\Gereau\anaconda3\envs\DEEPLABCUT\python.exe` |
| DLC project | `E:\RSDLC\2511_RSGNKK_Blackbox` |
| DLC config | `E:\RSDLC\2511_RSGNKK_Blackbox\config.yaml` |
| Model | `palmreader-500Mar25` |
| Engine | pytorch |
| Shuffle | **9** |
| Snapshot | `best` (resolves to `snapshot_best-190` on disk) |
| Batch size | **32** (do not raise — see GPU note below) |
| Body parts | tailtip, tailbase, centroid, neck, snout, hlpaw, hrpaw, flpaw, frpaw |

GPU note: 2080 Ti is power-capped at batch 32. Larger batches throttle clocks and degrade over time. Stay at 32.

### Feature extraction

| Setting | Value |
|---|---|
| `bp_pixbrt_list` | `['hrpaw', 'hlpaw', 'snout']` |
| `square_size` | `[40, 40, 40]` |
| `pix_threshold` | `0.3` |
| `include_optical_flow` | **True** |
| `bp_optflow_list` | `['hrpaw', 'hlpaw', 'snout']` |
| Resulting cache hash | `8aed1c22` |
| Per-frame extract time | ~4.4–5.1 ms (cf. `scripts/research/telemetry/timing.csv`) |

**Single-pass policy**: feature extraction happens ONCE per video, with optical flow on. The 5 no-flow classifiers pick out only their needed feature columns via `model.feature_names_in_`, so they ignore the flow columns gracefully. Don't extract twice (THC pipeline used to do this — it's been fixed).

### Classifier set

5 classifiers from `E:\RSVIDS\Blackbox\260506_RS_THC_Withdrawal\classifiers\` (these were trained on the same DLC model + feature schema and are reused across projects):

- `PixelPaws_Facial_grooming_AllFeatures.pkl`
- `PixelPaws_Left_licking.pkl`
- `PixelPaws_rearing_AllFeatures.pkl`
- `PixelPaws_still_AllFeatures.pkl`
- `PixelPaws_walking_AllFeatures.pkl`

Per-classifier post-processing (best_thresh, min_bout, min_after_bout, max_gap) comes from `E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\transitions\for_claude.json`.

### Group analysis conventions

| Setting | Value |
|---|---|
| Priority list (argmax tie-break) | `Facial_grooming > Left_licking > rearing > still > walking` |
| `EXCLUDE_NOISE` | **True** — drop unclassified frames from transitions/occupancy |
| Self-transition handling | **Re-collapse after dropping noise** so diagonal of transition matrix is 0 (this is the standard behavioral-state convention; high diagonal otherwise = noise-gap fragmentation, not real state revisits) |
| Window caps (per condition) | **BL = 30 min, 1h = 60 min** at 60 fps |
| FPS | 60 |

### Group mapping (project-specific — supply per cohort)

For SNTX cohort 2 the key was: 8 CCIX (BLUE), 2 SilkX (GREEN), 6 Sham (BLACK). See `sntx_cohort2_blinding_key.json` for the JSON format — copy and edit per cohort.

For projects without a blinding key (like THC withdrawal), use a deterministic rule encoded in `parse_session()`. THC's rule: odd-numbered mouse = drug, even = vehicle.

### Discord

| Setting | Value |
|---|---|
| Webhook URL | `` |
| User-Agent required | YES — urllib without `User-Agent` returns HTTP 403. All helper functions set `PixelPaws-*/1.0`. |
| Progress message pattern | Single message PATCHed every 60 s (silent edits, no notification spam). Milestones (start, complete, failure) are NEW posts (these ping). |

---

## This run's outputs (SNTX cohort 2)

- **Project folder**: `E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\videos\2605_Cohort2`
- **Videos**: 31 `.mp4` (16 unique mouse IDs; 13 BL+1h pairs + 9970_BL alone + duplicate naming convention `050126_` vs `05026_`)
- **DLC**: 31 raw `.h5`, 31 filtered `.h5` (shuffle 9, snapshot best-190)
- **Features**: 31 `*_features_8aed1c22.pkl` in `features/`
- **Predictions**: 31 `*_predictions.csv` in `results/`
- **Group analysis** (post-window-cap, post-self-transition-collapse):
  - 28/31 sessions keyed (3 skipped: 9970_BL and 9519_BL/1h not on blinding key)
  - Figures: `immobility.png`, `state_occupancy.png`, `transitions.png`, `transitions_diffs.png`, `bout_structure.png`, `ethograms.png` in `analysis/`
  - CSVs: `state_occupancy.csv`, `bout_stats.csv`, `transitions_<group>_<condition>.csv` (×6)
- **Filtered chain status**: started, **failed** at session 5 (`9513_BL`) — same DLC↔video frame mismatch class as the brightness bug; needs further investigation OR same `brightness_features.py` style fix in any other extractor.

---

## Pipeline scripts (this agent's pieces)

All under `E:\PixelPaws\scripts\research\`. Generic enough to copy for a new cohort by editing the path constants at the top.

| Script | Env | Purpose |
|---|---|---|
| `dlc_batch_<cohort>.py` | DEEPLABCUT | analyze_videos + filterpredictions, idempotent (separate `need_analyze`/`need_filter` lists so an interrupted run resumes correctly) |
| `<cohort>_predict_all.py` | regular `py` | Single-pass with-flow features → 5 classifiers → CSV/session. Posts a live Discord progress bar with per-video status. Supports `--h5-source=raw\|filtered` and writes to `results/` or `results_filtered/`. |
| `<cohort>_group_analyze.py` | regular `py` | Reads predictions, applies window caps + self-transition collapse, writes 6 figures + CSVs, posts to Discord. Supports `--suffix=_filtered` for the filtered output dir. |
| `<cohort>_watcher.py` | regular `py` | Polls DLC output folder, runs the chain as soon as raw (and later filtered) hit target. Posts milestones to Discord. |
| `thc_orchestrator.py` | regular `py` | The bigger orchestrator that ALSO manages DLC subprocess with stall/restart logic (used during THC run; cohort2 just used the watcher since DLC was already done). |
| `collect_timing_telemetry.py` | regular `py` | Parses DLC + chain logs for per-video timings, snapshot-writes `telemetry/timing.csv`. Run with `--watch` for periodic refresh. |

### Stall/restart safety (`thc_orchestrator.py`)

- `MIN_HEALTHY_FPS = 8.0` — tqdm `it/s` floor; below this for 10 min = stall, kill + restart DLC
- `LOG_MTIME_STALL_MIN = 5` — log untouched this long = process dead
- `HARD_STALL_TIMEOUT_MIN = 240` — backstop: no new .h5 in 4 h
- `MAX_RESTARTS = 5`, `KILL_GRACE_SECONDS = 60`

---

## Per-cohort config template

Copy the block below into a JSON file when starting a new cohort and hand it to me along with the blinding key (if applicable). I'll generate the scripts from this.

```json
{
  "cohort_name": "sntx_cohort3",
  "project_root": "E:\\RSVIDS\\Blackbox\\<project>\\<subfolder>",
  "videos_dir": "<same as project_root or subdir>",
  "dlc_env_python": "C:\\Users\\Gereau\\anaconda3\\envs\\DEEPLABCUT\\python.exe",
  "dlc_config": "E:\\RSDLC\\2511_RSGNKK_Blackbox\\config.yaml",
  "dlc_shuffle": 9,
  "dlc_batchsize": 32,
  "classifier_dir": "E:\\RSVIDS\\Blackbox\\260506_RS_THC_Withdrawal\\classifiers",
  "for_claude_json": "E:\\RSVIDS\\Blackbox\\2603_SNLT_JG\\Baseline\\transitions\\for_claude.json",
  "feature_config": {
    "bp_pixbrt_list": ["hrpaw", "hlpaw", "snout"],
    "square_size": [40, 40, 40],
    "pix_threshold": 0.3,
    "include_optical_flow": true,
    "bp_optflow_list": ["hrpaw", "hlpaw", "snout"]
  },
  "filename_pattern": "regex with named groups <mouse>, <cond>; conditions list",
  "conditions": ["BL", "1h"],
  "window_minutes": {"BL": 30, "1h": 60},
  "fps": 60,
  "groups": ["Sham", "CCIX", "SilkX"],
  "group_palette": {"Sham": "#7f7f7f", "CCIX": "#d62728", "SilkX": "#2ca02c"},
  "blinding_key_path": "scripts/research/<cohort>_blinding_key.json",
  "discord_webhook": "",
  "run_filtered_pass": true
}
```

### Blinding key template (`<cohort>_blinding_key.json`)

```json
{
  "legend": {"BLUE": "GroupA", "GREEN": "GroupB", "BLACK": "GroupC"},
  "expected_counts": {"GroupA": 8, "GroupB": 2, "GroupC": 6},
  "mice": {
    "9501": {"color": "BLUE", "group": "GroupA"},
    "9504": {"color": "GREEN", "group": "GroupB"}
  },
  "in_videos_but_not_keyed": [],
  "keyed_but_no_videos_yet": []
}
```

---

## How to invoke

Give me one of:

1. **"Run the DLCPP agent on `<path>` with this blinding key"** + paste the key → I'll write the per-cohort scripts (`<cohort>_predict_all.py`, `<cohort>_group_analyze.py`, `<cohort>_watcher.py`, `dlc_batch_<cohort>.py`), kick off DLC if not done, launch the watcher with Discord pings, and post the final analysis to Discord.

2. **"Re-run group_analyze on `<cohort>` with windows X / Y"** → I'll edit just the analysis settings and rerun without redoing DLC/predict.

3. **"DLCPP agent: same as SNTX cohort 2"** → I'll prompt for the new project folder and blinding key (or take whatever defaults match this run).

---

## Known issues / recovery patterns

| Symptom | Cause | Fix |
|---|---|---|
| `IndexError: index N is out of bounds for axis 0 with size N` in `brightness_features.py:433` | DLC `.h5` row count differs from `cv2.CAP_PROP_FRAME_COUNT` by ±1 (decoder edge case) | Patched in `brightness_features.py:268-298` — coords now padded/truncated to `num_frames` with -1 sentinel |
| Orchestrator stuck in restart loop on filterpredictions | Original `dlc_batch_*.py` checked `shuffle9*.h5` (matched both raw and filtered), skipped videos with raw but no filter → filterpredictions never re-ran | Fixed: scripts now split into `need_analyze` / `need_filter` lists, idempotent |
| HTTP 403 from Discord on urllib POST | Discord requires `User-Agent` header | All helpers set `User-Agent: PixelPaws-*/1.0` |
| Duplicate Discord posts | Two orchestrators running in parallel (e.g., bash + Python both kicked off) | Kill ALL processes matching `*orchestrator*`, `*watcher*`, `*DEEPLABCUT*python*` before relaunching anything |
| Stale transition figures with non-zero diagonal | Script edited AFTER group_analyze had already run | Re-run `<cohort>_group_analyze.py` manually; subprocess.run reads the script file fresh each invocation |

---

## Files relevant to this run

- This README: `scripts/research/AUTOMATED_DLCPP_AGENT.md`
- Blinding key: `scripts/research/sntx_cohort2_blinding_key.json`
- Pipeline scripts: `scripts/research/sntx_cohort2_*.py`
- Logs: `scripts/research/sntx_cohort2_watcher.log`, `scripts/research/snlt_cohort2_dlc.log`
- Telemetry: `scripts/research/telemetry/timing.csv`
- THC parallel: `scripts/research/README_THC_Withdrawal.md`
