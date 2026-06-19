# Rimonabant cohort processing plan

This document captures the canonical pipeline used for the 2605 (260515)
female rimonabant dose-response cohort, including the choice of CRF 26
for ingestion-time transcoding. New rim animals dropped into the
transfer portal go through the same five stages and end up extending
the same combined-cohort scratching analysis.

---

## 1. Per-stage pipeline

| # | Stage | Script | Engine | Wall-clock |
|--:|-------|--------|--------|-----------:|
| 1 | **Ingest + transcode**: portal mp4 → cohort/videos at libx264 CRF 26, slower preset | `scripts/research/rim_addons_chain.py` (replaces the relevant phase of `chain_finish_thc_then_rim.py`) | ffmpeg / CPU | ~3 min / 60 min video |
| 2 | **DLC inference**: shuffle 9, snapshot best-190, batch=42 (OOM fallback 32), filterpredictions | `scripts/research/dlc_batch_rim_dose.py` | DEEPLABCUT env, RTX 2080 Ti | ~45-60 min / 60 min video |
| 3 | **Feature extraction**: single-pass brightness + sparse-LK optical flow (hash `8aed1c22`), 9 body parts, `bp_pixbrt = [hrpaw, hlpaw, snout]` | `prediction_pipeline.PixelPaws_ExtractFeatures` invoked by `rim_dose_response.py` | PixelPaws env, CPU | ~10-13 min / 60 min video |
| 4 | **6-classifier predict**: 5 SNLT (`facial_grooming`, `left_licking`, `rearing`, `still`, `walking`) + `Scratching_AllFeatures`, each with its own threshold + post-processing baked in the .pkl | inside `rim_dose_response.py` | CPU | ~10 s / classifier / video |
| 5 | **Dose-response analysis + combined scratching update** | `rim_dose_response.py` + `combined_rim_scratching_timecourse.py` | CPU | ~30 s total |

All stages are **idempotent** — re-running picks up only the new sessions.

### Folder layout (cohort 260515)

```
E:/RSVIDS/Blackbox/260515_Rim_DoseResp/
  videos/                fem_baseline_RimN.mp4 + fem_postrim_RimN.mp4 + DLC artefacts
  classifiers/           6 .pkl files copied from 260506_RS_THC_Withdrawal/classifiers
  features/              <session>_features_8aed1c22.pkl
  results/               <session>_predictions.csv (one row per frame, 6 binary + 6 proba cols)
  analysis/              rim_dose_response_summary.csv, rim_dose_response_*.png,
                         combined_rim_scratching_*.csv, *.png
```

### Naming convention

The cohort scripts parse `fem_(baseline|postrim)_Rim(\d+)(?:_\d+)?$`.
Portal-side filenames look like `Rim12_baseline.mp4` /
`Rim12_post-rim.mp4` and must be renamed to the cohort convention at
transcode time (handled by `rim_addons_chain.py`).

### Dose assignment

`scripts/research/rim_dose_response.py` uses a modular formula
`DOSE_SEQUENCE[(n - 1) % 4]` with `DOSE_SEQUENCE = ["VEH", "1 mg/kg", "3 mg/kg", "10 mg/kg"]`:

| Rim# | Dose | Rim# | Dose |
|-----:|:-----|-----:|:-----|
| 1, 5, 9     | VEH      | 2, 6, 10    | 1 mg/kg  |
| 3, 7, 11    | 3 mg/kg  | 4, 8, 12    | 10 mg/kg |

(Add new subjects by extending the index; nothing else needs to change.)

---

## 2. Why CRF 26 — compression sweep results

Before transcoding the cohort, a 6-CRF sweep was run on a 3-min subset
of 4 already-fully-analysed THC pre-rimonabant sessions
(`compression_sweep.py`). For each (video × CRF) point we transcoded
the subset, ran DLC + features + the 5 SNLT classifiers, and compared
to the same model run on the original 8 Mbps recording.

### Metrics (n = 4 videos per row)

| Label | CRF | Median size (MB, 3-min subset) | Mean Δ keypoint likelihood | Min feature corr | Median feature corr | **Classifier agreement** |
|-------|----:|------------------------------:|---------------------------:|-----------------:|--------------------:|-------------------------:|
| x264 23 | 23 | 7.8 | -0.0005 | -0.18 | 0.82 | **0.9628** |
| x264 26 | 26 | **4.4** | +0.0008 | -0.10 | 0.81 | **0.9620** |
| x264 28 | 28 | 3.4 | +0.0017 | -0.05 | 0.78 | 0.9578 |
| x264 31 | 31 | 2.3 | +0.0039 | -0.14 | 0.74 | 0.9499 |
| x264 35 | 35 | 1.6 | +0.0126 | -0.12 | 0.67 | 0.9398 |
| x265 28 | 28 | 3.9 | +0.0011 | -0.14 | 0.78 | 0.9613 |

Read as: at CRF 26 the per-frame **5-classifier agreement** with the
8 Mbps original is **96.2 %** (vs 96.3 % at CRF 23, a 0.08 pp drop),
**file size shrinks ~45 %**, and median feature correlation is 0.81
(vs 0.82 baseline). Past CRF 28 classifier agreement starts to drop
visibly (~0.5 pp per CRF step).

### Why not CRF 28 (which the data also supports)

The 60-min videos in this cohort use 8 Mbps QSV H.264 from PawCapture
as the source. CRF 28 still meets the redlines (likelihood drop ≤ 0.05,
classifier agreement ≥ 0.95) but the safety margin to "classifiers
start disagreeing" is thinner. CRF 26 was chosen for **sanity margin**
when re-using the pipeline on future cohorts with different lighting
or camera settings — same disk-space win (~70× smaller than 8 Mbps
source for the actual 60-min videos), no measurable analysis cost.

### How the original 60-min videos look at CRF 26

Disk numbers from the actual cohort (full 60-min recordings at CRF 26
libx264 slower):

| Subject | Size |
|---------|------:|
| fem_postrim_Rim1 | 26 MB |
| fem_baseline_Rim3 | 56 MB |
| fem_postrim_Rim4 | 54 MB |
| (… 18 total) | 26–56 MB |

For comparison the **same-duration THC cohort videos** (which weren't
transcoded — predate the sweep) are **3.3-3.7 GB each** at the 8 Mbps
source bitrate. So CRF 26 is ~**70× smaller** with no measurable
analysis loss.

---

## 3. Adding new animals (operational checklist)

1. **Wait for syncthing**: portal files arrive with `~syncthing~` prefix
   and `.tmp` suffix. Only act once the prefix is gone AND the
   `session_*.json` companion appears in the same folder.
2. **Run `scripts/research/rim_addons_chain.py`**. It does:
   1. Renames `Rim<N>_baseline.mp4` → `fem_baseline_Rim<N>.mp4` and
      `Rim<N>_post-rim.mp4` → `fem_postrim_Rim<N>.mp4` during the
      ffmpeg transcode (no second pass).
   2. Drops the new mp4s into `260515_Rim_DoseResp/videos/`.
   3. Removes the originals from the portal (matches the existing chain
      behaviour; the source recording on the PawCapture rig is the
      durable copy).
   4. Calls `dlc_batch_rim_dose.py` (idempotent — only the new videos
      have no `.h5` yet, so only they get analyzed).
   5. Calls `rim_dose_response.py` (idempotent — features pickle +
      predictions CSV both gate processing).
   6. Calls `combined_rim_scratching_timecourse.py` to refresh the
      combined dose-response with the new subjects pooled in.
   7. Posts a Discord update at each stage.
3. **Sanity-check the dose-response figure** posted to Discord; the
   only thing that can fluke is a video where DLC tracking fails badly
   (rare), which will show up as an outlier subject. Spot-check the
   `_p60_labeled.mp4` overlay if anything looks off.

---

## 4. Files modified / created by this cohort

- `scripts/research/dlc_batch_rim_dose.py`
- `scripts/research/rim_dose_response.py`
- `scripts/research/combined_rim_scratching.py`
- `scripts/research/combined_rim_scratching_timecourse.py`
- `scripts/research/rim_addons_chain.py` *(this addition)*
- `scripts/research/compression_sweep.py`
- `docs/rim_cohort_processing_plan.md` *(this file)*
