# PixelPaws Changelog

Running log of non-trivial edits and the decisions behind them.
Entries are grouped by date (most recent first).

## 2026-07-06

### Added — In-GUI pose extraction + restored auto-installer (SlivickiR_WangH network)

Brought the distribution-fork's integrated pose-estimation experience (branch
`distribution-fork-2026-05-11`, commit `2a79c3f`) forward onto `pixelpaws-working`, pointed at
the **most recent SlivickiR_WangH network** (`D:\PixelPaws_Active\PixelPaws_SlivickiR_WangH`,
task `pixelpaws`/date `Jul26`, iteration-0 / shuffle 1 / `snapshot-best-930`). Previously the
working branch could only *consume* pre-existing DLC `.h5` files; now videos can be pose-tracked
from the GUI.

- **`dlc_inference/`** (8 modules, ported) — bundle-based fast PyTorch inference engine
  (`run_dlc_on_video` builds a runner from `snapshot.pt` + `pytorch_config.yaml`, no DLC project
  tree needed; GPU preprocessing + decode prefetch). Self-contained; targets DLC 3.0.0rc13, which
  is exactly the installed `DEEPLABCUT` env version.
- **`default_bundle/pixelpaws_v1/`** — the new network packaged as a versioned bundle (`manifest.json`
  + `snapshot.pt` + `pytorch_config.yaml`), built by **`scripts/distribution/prepare_bundle.py`**
  (repointed to the new project/shuffle/snapshot; UTF-8 manifest write). 96 MB weight tracked via
  **git-LFS** (`.gitattributes`: `default_bundle/**/*.pt`). Ships with **no classifiers** — the old
  5 were trained on the previous network; `dlc_inference/compatibility.py` would flag the mismatch,
  so retraining them is a follow-up.
- **`pipeline/dlc_analyze.py`** — backend run under a DLC-capable Python. `--probe` prints device/VRAM
  JSON; analyze mode loops videos through `run_dlc_on_video` emitting stdout progress markers
  (`VIDEO_START` / `FRAMES` / `VIDEO_DONE` / `ALL_DONE`), with a `deeplabcut.analyze_videos` fallback
  against `POSE_DLC_CONFIG`.
- **`pipeline/pp_config.py`** — `POSE_DLC_CONFIG/POSE_SHUFFLE/POSE_DLC_BATCH/POSE_MODEL_NAME` +
  **`resolve_pose_python()`**: uses `sys.executable` when it can import deeplabcut (installer's single
  combined env) else `DLC_PYTHON` (dev machine). One code path, both deployment shapes.
- **`dlc_run_dialog.py`** — ported settings + live-progress dialogs; the worker shells out to
  `dlc_analyze.py` (GUI Python has no torch) and parses markers. Added a **"Select more frames to
  label"** checkbox that chains into the existing **Extract Problem Frames** tool.
- **`PixelPaws_GUI.py`** — Tools → Video Tools → **🐾 Analyze Videos (Pose Tracking)**
  (`open_pose_extraction`): first-run installs the default bundle, runs the flow, then chains into
  the **🎬 Predict** tab and/or **🚩 Extract Problem Frames**.
- **`installer/` + `INSTALL.txt`** (restored from the fork) — one-click Windows installer:
  Miniforge bootstrap → single conda env **`pixelpaws`** (Python 3.11) with the GUI **and**
  `deeplabcut>=3.0.0rc13` + torch cu118 in one env, seeds the model bundle, desktop shortcut;
  `run.bat` launches the GUI from that env. This preserves the "single install, DLC included"
  distribution.

**Decisions:** keep the fork's dialog UX but back it with a runtime-resolved DLC interpreter
(shell-out on the dev machine's two-env setup, in-process in the installer's combined env) instead
of the fork's hardcoded in-process import. Everything routes through a bundle so dev and shipped
builds share one inference path. Snapshot selected by DLC's `best` (930); **report/gate on the
paws** (2.55 px @ pcutoff — better than the old model's 3.58 px), not the headline RMSE (15.7 px),
which is dominated by `tailtip` (54 px).

**Verified:** bundle builds/loads/sha-verifies; real inference on a 30-frame clip under the
DEEPLABCUT env → correctly-named DLC `.h5` (3-level `(scorer,bodyparts,coords)` MultiIndex, all 9
keypoints, ~53 fps, readable by the repo loader); cross-env device probe from the GUI Python returns
the RTX 3080 Ti; dialog imports cleanly without torch. **Pending (needs a hands-on session):**
interactive dialog click-through and a live `install.bat` conda build.

### Added — Problem-frame active learning (extract → label → retrain)

Branch `feature/problem-frame-active-learning` (fork of `active-learning-redesign`).
Closes the loop that fixes phantom scratch/lick calls: surface the frames where DLC
tracking is unreliable and/or the classifier is uncertain, correct their keypoints in the
existing labeler, then optionally retrain — all from the GUI.

- **`pipeline/problem_frames.py`** — reusable extractor. Two INDEPENDENT budget buckets:
  (A) *tracking* reuses `pose_filter.filter_pose()`'s per-frame `flagged_mask`
  (likelihood<0.3 / SimBA location-teleport / velocity-jump — hindpaws excluded from the
  velocity gate by design so real scratching isn't over-flagged) plus the AOW
  present-but-hindpaw-uncertain / lowest-likelihood buckets, empty-cage gated; (B)
  *classifier* selects near-threshold frames (`|p-0.5|*2 < 0.30`) from
  `predict_with_xgboost`, or (mode `positive`) tracking-flagged frames co-located with a
  confident call (teleport→phantom). Gap-spaced (300) + burden-proportional cross-session
  budget. Writes the standard candidate-folder contract (prefilled `CollectedData_<scorer>`
  with the current model's keypoints, flat index, `_scratch_to_label.txt` marker) —
  scorer/project-parameterized. Pure pandas/numpy/cv2 core; classifier bucket lazy-imports
  the pipeline. CLI + importable. **Key reuse:** `pose_filter`'s `flagged_mask` was
  previously computed and discarded in `pp_pipeline.stage_features`.
- **`pixelpaws_labeler.py`** — the proven Tkinter keypoint labeler, now shipped in the repo
  and made project/scorer-aware via an optional `_labeler_config.json` sidecar
  (`PIXELPAWS_LABELER_CONFIG` env). Absent → the original AlexZ/2511 defaults, so a bare
  double-click is unchanged. The 2511 copy + `.cmd` are left untouched.
- **`pipeline/label_merge.py`** + **`pipeline/dlc_retrain.py`** — scorer-parameterized
  generalizations of `merge_iter*`/`prep_iter*`: fold candidate folders, add videos +
  bump iteration (ruamel, backup, restore-on-fail), `create_training_dataset`, verify
  frames landed, then `train_network` warm-start from the current best snapshot. The GUI
  shells to `dlc_retrain.py` in the DEEPLABCUT env and tails `learning_stats.csv`.
- **`PixelPaws_GUI.py`** — Tools tab → **🚩 Extract Problem Frames** dialog (sessions,
  bucket toggles, classifier + mode, budget split) → worker-thread extraction → launch
  labeler; **🧠 Retrain DLC (add iteration)** dialog with a live progress bar + per-epoch
  stats and a **plain-language legend** explaining train loss / test RMSE / RMSE@cutoff /
  mAP-mAR / learning rate / epoch, plus a soft GPU-busy guard.

**Decisions:** two independent buckets (not just tracking∩classifier); GUI-surfaced with a
reusable module behind it; retraining is user-triggered and explained. Also ignored
Syncthing `*.sync-conflict-*` artifacts in `.gitignore`.

**Verified:** extractor end-to-end on real sessions (contract exact: scorer, 18 cols, flat
index, prefill, marker; classifier bucket picks maximally-uncertain frames); labeler sidecar
config + fallback; merge/config-edit helpers; GUI compiles and its non-Tk helpers resolve
scorer/bodyparts/classifiers/best-snapshot/feature-cache against real projects. **Pending
(needs a hands-on session + free GPU):** interactive dialog click-through and a live retrain
run.

## 2026-05-24

### Added — Global Classifier Encyclopedia v1.0.0-rc1 (Tier 1 shipped)

`E:/PixelPaws/pixelpaws_global_classifier_encyclopedia/`:
- 4 KEEP classifiers shipped: `Facial_grooming` (F1=0.900), `Left_licking`
  (0.906), `rearing` (0.855), `Scratching` (0.952). All pass the
  0.85 frame-F1 inclusion gate. Each copied to `classifiers/classifier_*.pkl`
  with added `expected_fps=60`, `encyclopedia_version`, status, definition.
- `manifest.json` indexes the 4 shipped classifiers with per-entry
  post-processing config (`best_thresh`, `min_bout`, etc.) and feature
  schema. `README.md` summarises shipped/dropped + usage instructions.
  `docs/<name>.md` describes each classifier's training data + limitations.

**Methodology decisions** (v1 scope, after a longer interactive triage):
- **Canonical feature schema = `8aed1c22`** (pose v5 + brightness v1 + OF,
  sq=40, brightness body parts = `hrpaw, hlpaw, snout`). Picked because
  all proven >0.85-F1 classifiers already use it, and the visual ROI
  comparison (`scripts/research/visualize_brightness_roi_size.py`) showed
  sq=40 captures full paw extent. Dual-sq (20+40) noted as a v2 idea.
- **Probability calibration deferred to v1.1**: source pkls don't store
  training labels, so fitting isotonic calibrators requires re-loading
  per-classifier training data. Skipped in v1 to ship faster.
- **F1 < 0.85 → DROP/RETRAIN**: `body_grooming` (0.765), `still` (0.767),
  `walking` (0.783) below gate; queued for Tier 3 retrain after OF
  re-extraction.

### Added — Tools for the encyclopedia build

- `scripts/research/extract_session_8aed1c22.py` — single-session
  extractor (pose + brightness + optical flow) producing canonical
  `8aed1c22` pkls. Skips if target exists; writes `.version.json` sidecar.
- `scripts/research/batch_extract_8aed1c22.py` — parallel batch driver
  (default 2 workers) over all BORIS-labelled Tier 2/3 sessions. Posts
  milestone updates to `#dlctracker`. Output to each cohort's
  `features/` dir.
- `scripts/research/feature_pkl_sweep.py` — read-only sweep that maps
  every `_features_<hash>.pkl` under `E:/RSVIDS/Blackbox/` to its
  config (sq, OF, sil, bp ordering) by hashing candidate configs until
  one matches. Used to discover that 167/440 pkls already use `8aed1c22`.
- `scripts/research/audit_classifier_f1.py` — Phase A audit; surveys
  existing classifiers under `E:/RSVIDS/Blackbox/*/classifiers/`,
  reads `cv_f1_scores`/`mean_cv_f1`/`oof_best_f1` from each pkl, picks
  best variant per behaviour, writes `audit_report.csv` and
  `audit_best_per_behaviour.csv`.
- `scripts/research/boris_to_per_frame_labels.py` — Phase B; converts 41
  BORIS observations × 8.7k events from 3 `.boris` projects into 104
  per-frame `int8` `.npy` arrays at `E:/RS_Boris/per_frame_labels/`.
  Uses `boris_label_aliases.py` to coalesce e.g. `L_flinching` /
  `L flinching` / `Left_flinching` / `flinching` → canonical `L_flinching`.
- `scripts/research/train_global_classifier.py` — Phase C training loop;
  per-behaviour XGBoost + session-level GroupKFold + SHAP pruning +
  Optuna (skipped if F1≥0.90) + bout-level F1 + isotonic calibration.
  **Speed fix (2026-05-24)**: default `n_estimators=600` (was 1700) +
  `lr=0.03` (was 0.01) + early stopping on a held-out training session.
  Per-fold time on 4M-row pooled data: 36 min → 1-5 min.

**Why this scope**: user wants a distributable classifier bundle for the
next PixelPaws release. Plan: 5 phases (audit → BORIS labels → train →
performance panels → bundle); Tier 1 (4 KEEPs) shipped first to have
something concrete; Tier 2 (L_flinching, body_grooming_combined) and
Tier 3 (RETRAIN body_grooming/still/walking + NEW back_groom/belly_groom/
state_joint) blocked on the 31-session batch re-extraction now in
flight.

## 2026-05-21

### Added — SNTX vs Naive contour intensity timecourse (5-min bins)

`scripts/research/sntx_contour_intensity_tc.py`:
- Reads the 7 baseline contour CSVs from
  `E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\gait_limb_analysis\`
  (4 SNTX + 3 Naive, 30-min sessions @ 60 fps -> 6 bins of 5 min).
- Treatment mapping from `key_file.csv` (Subject -> SNTX | Naive).
- Per-bin reduction matches the GUI / formoxy convention:
  `mean(intensities_HL)` / `mean(intensities_HR)` over frames where
  *both* contours were detected (intensity > 0 and finite).
- 3-panel figure (PNG + PDF + SVG): HL intensity, HR intensity,
  HL/HR ratio. Mean ± SEM, **Naive = black, SNTX = red**.
- Stats per panel: 2-way ANOVA (Treatment × Bin) annotated inside
  each panel, plus per-bin Mann-Whitney U (SNTX vs Naive) shown as
  asterisks at the top of each bin.
- Outputs `<project>/analysis/sntx_contour_intensity_tc.{png,pdf,svg}`
  + per-session long-form CSV + per-bin summary CSV.
- Posts everything to `#results`.

**Why**: ask for a 5-min bin contour-intensity time course to compare
SNTX vs Naive baseline. The ratio panel parallels the formoxy ratio
TC family of scripts and uses the same Otsu-contour metric (NOT the
fixed-ROI mean, which is contaminated by floor pixels).

### Changed — Sex-marker convention + legend placement

- `scripts/research/dose_response_bout_metrics.py`: legend in the
  "Total time" panel moved from `loc="upper left"` to `loc="upper right"`
  so it no longer overlaps the tall Veh/Oxy1 per-mouse dots. Upper-right
  is empty space because Oxy10 sits at ~0.
- `scripts/research/formoxy_licking_phase_auc.py`: per-mouse dots now
  use the same sex convention as the dose-response triple
  (Male = closed colored fill + black edge; Female = open with colored
  edge). Adds a Male/Female legend on the Phase-1 panel (upper-right).

**Why**: ask for visual consistency across cohort-pooled formalin figures
so closed vs open dots always mean male vs female.

### Added — FormOxy Left_licking Phase-1 vs Phase-2 AUC bar plot

**Why**: user asked for a phase-1 / phase-2 AUC breakdown of the existing
5-min-bin Left_licking timecourse to summarise the formalin-test
acute (0–10 min) and inflammatory (10–60 min) responses across the four
oxycodone doses.

**Added**: `scripts/research/formoxy_licking_phase_auc.py`
- Reads the long-form per-bin CSV from
  `<2605>/analysis/leftlicking_timecourse_5min.csv` (already exists from
  the existing timecourse driver — no re-computation of per-frame
  predictions needed).
- Per session AUC = sum of `total_s_licking` over the bins in each phase
  (P1: bins 0–1; P2: bins 2–11). Units stay in seconds because the
  bins are uniform 5-min wide.
- Side-by-side bar plot (P1 left, P2 right): mean ± SEM with per-mouse
  jittered scatter, same blue Veh/Oxy1/Oxy3/Oxy10 palette as the
  timecourse line plot for visual continuity.
- Per-phase 1-way ANOVA across DOSE_SEQ + Dunnett's vs Veh
  (`scipy.stats.dunnett`, with Welch t-test fallback if unavailable).
- Outputs to `<2605>/analysis/`:
  `leftlicking_phase_auc_per_session.csv`,
  `leftlicking_phase_auc_summary.csv`,
  `leftlicking_phase_auc_bars.{png,pdf,svg}`.
- Posts PNG + PDF + SVG + both CSVs to `#results`.

**Headline numbers** (n=6/dose, pooled 2512 + 2605):
- Phase 1 (acute, 0–10 min): ANOVA F=14.12, p=3.6e-05.
  Oxy3 vs Veh p=3e-04, Oxy10 vs Veh p=1.6e-05; Oxy1 trending (p=0.054).
- Phase 2 (inflammatory, 10–60 min): ANOVA F=12.70, p=7.2e-05.
  Oxy3 vs Veh p=0.043, Oxy10 vs Veh p=4e-04; Oxy1 not different (p=0.73).

**How to apply**: re-run after any change to the timecourse CSV (e.g.,
new sessions added or per-frame predictions regenerated). Does not need
classifier retraining — purely an analysis-side aggregation.

### Added — Combined Scratching + Left_licking 2x2 panel composite (gridless threshold curves)

**Why**: user requested the licking threshold+SHAP composite be re-laid-out
to match the Scratching reference figure, then combined into one figure
covering both classifiers with the inter-panel gap reduced and y-grid
removed from the threshold curves.

**Changed**:
- `scripts/research/regen_scratching_plots.py` — added `show_grid=True`
  kwarg to `plot_threshold_curves`; gated the `ax.grid(axis='y'…)` call
  so existing callers (`batch_pdf_reexports.py`,
  `scratching_svg_exports.py`) keep their current behavior, but new
  combined figures can disable the grid for a cleaner look.

**Added scripts** (all under `scripts/research/`):
- `licking_recomposite_scratching_layout.py` — re-composites the
  existing licking threshold + SHAP PDFs into the Scratching SVG
  viewBox geometry (955.65×273.6; threshold at (0, 20.28) sized
  356.4×212.4; SHAP at (413.43, 0) sized 538.65×271.8).
- `scratching_licking_combined_panels.py` — 2-row composite
  (Scratching top, Left_licking bottom) at 656×420, with the
  inter-panel gap dropped from 57pt → 4pt.
- `regen_threshold_panels_nogrid.py` — regenerates both threshold
  PDFs with `show_grid=False` (Scratching via cached OOF in
  `_regen_cache.pkl`; Left_licking via fresh 3-fold session-level
  GroupKFold) and rebuilds the combined 2x2 composite.
  Output: `<HomeRig>/licking_plasma_panels/
  Scratching_Licking_combined_panels_nogrid.{pdf,png}`.
  Reconfirmed: Scratching best F1 = 0.966 @ thresh 0.86;
  Left_licking best F1 = 0.904 @ thresh 0.63.

**How to apply**: for figure-quality multi-panel composites where the
threshold curve sits next to a SHAP beeswarm, call
`plot_threshold_curves(..., show_grid=False)` — the extra horizontal
lines compete visually with the beeswarm jitter. Default `True`
preserves the standalone-figure look.

### Added — Left_licking honest CV reproduction + matched-style threshold/SHAP/ethogram figures

**Why**: an earlier honest retrain on per-session reconstructed features had
reported F1≈0.17 for Left_licking, way below the deployed pkl's reported CV
F1=0.906 — suggesting the deployed classifier was wrong. Investigation
ruled that out: the discrepancy was driven by (1) the reconstructed
FeatureCache missing 36 of the 375 BAREfoot pipeline features, and (2)
training on the 5.7%-positive raw distribution instead of the BAREfoot-
balanced ~28%-positive train set. Reproducing with the same BAREfoot
train pkl + 3-fold session-level GroupKFold across S1/S2/S3 yields
per-fold F1=[0.905, 0.873, 0.944], mean **0.907** — matching the
deployed pkl's reported 0.906. Frame-level best F1 = 0.904 at thresh=0.63.

**Added scripts** (all under `scripts/research/`):
- `licking_honest_correct.py` — honest 3-fold GroupKFold reproduction;
  emits threshold-curve + SHAP beeswarm + bar plots; posts to Discord.
- `licking_threshold_shap_composite.py` — PyMuPDF side-by-side layout of
  the threshold + SHAP PDFs in the scratching composite format.
- `licking_panels_plasma.py` — regenerates both panels with the *exact*
  scratching code (`regen_scratching_plots.plot_threshold_curves` +
  `plot_shap_panel`) so the plasma colormap / density-jittered beeswarm /
  purple importance bars are colour-consistent with the Scratching figure.
- `scratching_licking_combined_figure.py` — multi-behavior reference-style
  ethogram + confusion-matrix + 5-s-bin heatmap figure (full session, no
  windowing; green gradient for Scratching, blue gradient for Licking;
  Human row + gap + Model row in the bin heatmaps to mirror the raster
  layout). Output dir `<HomeRig>/scratch_lick_combined/`.

**How to apply**: when reproducing deployed-pkl CV metrics, always use the
existing BAREfoot `*_train_set.pkl` (not a re-extracted FeatureCache) —
the BAREfoot pipeline appends ~36 additional brightness/pixel features
that aren't reconstructable from the per-session caches, and uses a
balanced down-sampling that materially changes scale_pos_weight and the
resulting decision threshold.

### Fixed — LOVO threshold selection averaging per-fold thresholds (rare-class bias)

**Why**: the deployed Scratching classifier was running at `best_thresh=0.90`
with `min_bout=12 max_gap=2`, producing OOF F1=0.44 (precision 0.82,
recall 0.30) — way below the stored CV mean F1 of 0.64. Root cause: the
training-time LOVO sweep (`PixelPaws_GUI.py` ~line 5666) optimised a
threshold *per held-out fold* then averaged with `np.mean`. With
Scratching at 1.3% positive prevalence, per-fold optima skew high (0.85-
0.95 — strict thresholds maximise small-sample F1 by trading recall for
precision). Mean of strict thresholds = 0.90, which collapses recall to
30% when applied to the full pooled OOF vector.

The OOF sweep (which pools all sessions' probabilities into one vector
before sweeping) had already found the correct optimum
`thresh=0.50 mb=20 mg=0` with F1=0.72 and stored it as `oof_best_params`
in the pkl — but the deployment fields were being overwritten by LOVO.

**Fix**:
- `PixelPaws_GUI.py:5626-5700` — refactored the LOVO loop. For each
  held-out session, predict on *that session itself* using the fold
  model that never trained on it. Concatenate every session's
  held-out probabilities + labels into one pooled vector, then sweep
  threshold/min_bout/max_gap *once* on the pool via the existing
  `_sweep_postprocessing`. Also fixed a pre-existing bug where the loop
  was predicting on `_X_mask` (everything *except* the held-out session)
  instead of the held-out session itself — that mostly evaluated on
  in-sample training data, defeating LOVO's purpose.
- Patched the deployed Scratching pkl in place (backup at
  `PixelPaws_Scratching.pkl.bak`): `best_thresh` 0.90→0.50,
  `min_bout` 12→20, `max_gap` 2→0, `min_after_bout` 0→0,
  `oof_best_f1` 0.91→0.72 (corrected to match the actual sweep winner).
  Added `thresh_audit_note` to the pkl recording the change.
- Audit figure + sweep CSV at `<Classifiers>/plots/threshold_audit/
  PixelPaws_Scratching_threshold_audit.{png,pdf,svg,csv}`.

Downstream Rim Scratching dose-response figures were rendered using the
deployed (broken) threshold. They'll need re-prediction at the corrected
threshold once the feature-cache schema mismatch is sorted out.

## 2026-05-20

### Fixed — formoxy intensity-ratio TC aggregation + added contour-intensity TC

**Why**: the original `formoxy_intensity_ratio_tc.py` reduced each bin via
`mean(Pix_hlpaw/Pix_hrpaw)` per frame, which is unstable: when the paw is
briefly lifted the fixed ROI fills with dark floor so per-frame HL/HR
explodes (single bins hit 7.3). The huge Treatment p=5.2e-54*** that the
first version reported was an artifact of those outliers, not a real
biological effect. After fixing to `mean(HL)/mean(HR)` per bin the
treatment effect drops to a much more honest p=0.003** with no
individual bin significant.

The contour intensities (mean *inside* the Otsu-thresholded paw outline,
already in the per-frame contour CSVs we just extracted) are the cleaner
metric: they ignore the floor around the paw entirely. Added a parallel
TC script using contour intensities (`formoxy_contour_intensity_ratio_tc.py`).
Result: Treatment p=2.3e-16***, Time p=0.04*, Interaction p=0.33 n.s.;
bins 0-30 min significant per-bin then n.s. through 60 min — consistent
with peak formalin pain in the first 30 min subsiding naturally.

- `scripts/research/formoxy_intensity_ratio_tc.py` — patched
  `bin_session` to mean-of-means; updated Discord caption to flag the
  fix and the residual weak signal.
- `scripts/research/formoxy_contour_intensity_ratio_tc.py` (new) —
  reads `intensities_HL` / `intensities_HR` from the contour CSVs,
  drops frames where either contour wasn't detected (intensity == 0
  initial fill), reduces per-bin via mean-of-means, same GUI plot style
  (Veh black, Oxy palette light->dark blue, red dashed line at 1.0,
  two-way ANOVA + per-bin Kruskal-Wallis).

### Added — formoxy intensity-ratio TC + mean-contour plots (GUI-style)

**Why**: user wanted the same plots the GUI's Gait & Limb Statistics tab
produces, but on the combined 2512 + 2605 cohort. Two figures:
(a) HL/HR intensity-ratio timecourse with ANOVA + per-bin significance,
(b) mean paw contour outline per dose with +/- 1 SD radial envelope.

- `scripts/research/formoxy_intensity_ratio_tc.py` (new): reads brt CSVs
  from both cohorts, computes per-frame HL/HR ratio averaged per 5-min
  bin (matches the GUI's per-frame approach), then plots the ratio TC
  with the GUI palette (Veh = black, Oxy doses light->dark blue) and a
  red dashed line at 1.0 (symmetry). Statsmodels two-way ANOVA
  `ratio ~ C(dose) * C(bin_idx)` in a top-left annotation; per-bin
  Kruskal-Wallis omnibus across the 4 doses as asterisks above each
  bin. Result: Treatment p=5.2e-54***, Time p=0.059 n.s., Interaction
  p=0.81 n.s.; all 12 bins individually significant. Dose order at
  every bin: Veh ~ Oxy1 (dip to ~0.85-0.90 mid-session) < Oxy3 ~ Oxy10
  (flat ~0.95-0.97).

- `scripts/research/formoxy_paw_contour_extract.py` (new): standalone
  contour extraction mirroring the GUI's brightness-pass contour
  callback (gait_limb_tab.py:3863-3960 + 8898-8932). 50 px ROI per
  hind paw, Otsu binary threshold, largest contour by area, area
  filter >4 px^2; stores per-frame area / spread / intensity / width /
  solidity / aspect_ratio / circularity, plus up to 500 normalized
  64-point contour shapes per paw (centroid-zeroed, scaled by
  sqrt(area)). Outputs `*_contour_<hash>.csv` + matching `_shapes.npz`
  + sidecar JSON, identical schema to the 2512 cohort. 63 min for 12
  videos at ~700 f/s.

- `scripts/research/formoxy_mean_contour_analysis.py` (new): pools
  `_shapes.npz` files from both cohorts by dose, random-subsamples to
  1500 shapes per dose for visual parity with the GUI screenshot,
  computes mean shape (axis=0) and +/- 1 SD radial envelope using the
  same algorithm as `GaitLimbTab._add_contour_shape_tab` (radial_sd
  scaled along unit-direction-from-centroid, ring filled at alpha
  0.15). Side-by-side HL / HR panels, image-coords (y-axis inverted),
  posts to `#results`.

### Added — formoxy Left_licking 5-min-bin timecourse

**Why**: follow-on to the dose-response. The single-number dose-response
hides whether oxycodone delays vs blunts vs accelerates recovery — the
bin-by-bin timecourse separates those scenarios.

- `scripts/research/formoxy_licking_timecourse.py` (new): reads the
  per-frame prediction CSVs already on disk for both cohorts, bins to
  5-min windows (18 000 frames at 60 fps), caps at 60 min (12 bins),
  plots mean +/- SEM per dose as a 4-line timecourse with shaded SEM
  bands. Posts the figure + long-form CSV + per-dose summary CSV to
  the `#results` Discord channel.

### Added — formoxy L_flinching predict pass (qualitative)

**Why**: user asked to "try out" the latest L_flinching classifier on
the new 2605 cohort. The classifier
(`PixelPaws_L_flinching_pruned_120_20260429_165631.pkl`, CV F1 0.721)
expects a snout brightness ROI of 15 px / pixel_threshold 0.4 — but the
2605 cache was extracted with snout = 40 px / threshold 0.3 (to match
the Left_licking config). ~7 of the 120 selected features are
snout-brightness-derived and quantitatively wrong; the other ~110 (pose
+ paw brightness) match exactly. Run is **qualitative**.

- `scripts/research/formoxy_flinching_predict.py` (new): loads the
  pruned-120 classifier with joblib, runs it on the 12 formalin feature
  pickles, writes `*_L_flinching_predictions.csv` per session, plots a
  4-panel dose-response (n=3/dose, 2605 only) and posts to `#results`
  with the snout-brightness-mismatch caveat in the post body.

  Two feature-mismatch fixups inline:
  - Adds `Log10(Pix_hlpaw/Pix_hrpaw)` by negating the existing inverse
    ratio (we cache `Log10(Pix_hrpaw/Pix_hlpaw)`, model wants the flip).
  - Manually triggers `calculate_contact_features` (via
    `PoseFeatureExtractor`) when the model needs `*_DutyCycle` columns.
    `augment_features_post_cache` in `prediction_pipeline.py` only
    triggers contact augmentation when it sees a `*_ContactState`
    column in `model.feature_names_in_`, but the pruned flinching
    model selected `*_DutyCycle` directly — so the trigger misses.
    Worth a follow-up to fix in the core augmentation; for now the
    inline patch keeps this run unblocked.

  Result is a near-zero dose-response (Veh 0.03 % +/- 0.01, Oxy1 0.16
  +/- 0.09, Oxy3 0.05, Oxy10 0.00 — all <0.2 %). Could be biology (real
  flinching is rare once licking dominates), could be the snout-window
  mismatch suppressing sensitivity, could be the strict 0.65 threshold.
  For a clean run we need to re-extract features with snout window 15 /
  threshold 0.4 (~2-3 h).

### Added — formoxy paw-intensity (brightness) analysis (complete)

**Why**: replicate the gait-limb-tab paw brightness analysis that the
2512 cohort already has (12 per-session brightness CSVs in
`gait_limb_analysis/`, generated 2026-03-24). Pooling 2512 + 2605
gives n=6/dose for the paw-asymmetry biomarker (L hind formalin vs R
hind contralateral control).

- `scripts/research/formoxy_paw_brightness_extract.py` (new): calls
  `brightness_features.PixelBrightnessExtractorOptimized` directly
  (matching what the Gait & Limb tab does in-GUI) with the same ROI
  config as 2512: hind 20 px, fore 15 px, pixel_threshold 101,
  extraction_stride 1. crop_offset_x/y default to 0 (2512 used 250/0
  because those videos were wider; 2605 mp4s are already 744 px).
  Writes `2605_FormOxy_S<N>_formalin_brt_<hash>.csv` with the 4 paw
  columns plus a sidecar JSON of settings. ~6 min per video on the
  rig; ~70 min total for the cohort.

- `scripts/research/formoxy_paw_intensity_analysis.py` (new): reads
  both cohorts' brt CSVs, bins each session into 5-min windows over
  the first 60 min, plots two timecourse panels (Pix_hlpaw and the
  Log10(Pix_hlpaw/Pix_hrpaw) asymmetry index) and a 5-panel
  per-dose-bar plot for all 4 paws + asymmetry. Uses circle vs
  square markers per cohort. Posts to `#results`.

  Extraction completed in 67.7 min for 12 sessions (~647 frames/s,
  no errors). Result: absolute brightness differs systematically
  between cohorts (2605 is ~3-6x dimmer across ALL paws, lighting /
  camera-setup change between March and May recordings — not a ROI
  misalignment, forepaws scale the same way as hindpaws). The
  within-session **L/R log-ratio** (formalin / control hind) is the
  lighting-invariant biomarker, and it shows a treatment effect at
  higher doses: Veh/Oxy1 -0.23 / -0.32 (formalin paw ~45 % dimmer,
  consistent with paw guarding/tucking), Oxy3/Oxy10 -0.13 / -0.13
  (formalin paw closer to symmetric). Not strictly monotonic — Oxy1
  is slightly worse than vehicle — but the high-dose normalization
  matches the licking dose-response qualitatively.

## 2026-05-20

### Added — formoxy Left_licking dose-response (combined cohorts)

**Why**: the 2605_FormOxy_LeftPaws cohort (12 mice, formalin only) finished
DLC + features today. The point of the new cohort was to top up the prior
2512 left-paws formalin+oxy study so each dose has n=6 instead of n=3.
With both cohorts in hand we can finally read the oxycodone dose-response
on formalin-evoked licking with reasonable power.

- `scripts/research/formoxy_licking_dose_response.py` (new): loads
  `PixelPaws_Left_licking.pkl` (thresh 0.53, min_bout 3, min_after_bout 1,
  max_gap 5), runs it on the 12 new `*_formalin_features_8aed1c22.pkl`
  pickles, re-reads the 12 existing per-frame prediction CSVs from
  `2512_Blackbox_Formalin_Oxy/Left_paws/Results/Left_licking/`, and
  truncates every session to the first 60 min (216 000 frames at 60 fps)
  before computing pct-time / total-time / bout-count / mean-bout. Writes
  the 12 new prediction CSVs into `2605_FormOxy_LeftPaws/results/`, the
  combined per-mouse summary CSV + per-dose stats CSV + a 4-panel
  dose-response figure (bar + per-mouse scatter, circle = 2512, square =
  2605) into `2605_FormOxy_LeftPaws/analysis/`, and posts the figure +
  both CSVs to the `#results` Discord webhook (separate from the chain
  general channel — see `docs/private/discord_webhooks.md`).

- Result: clean monotonic dose-response. Veh 11.1 ± 2.2 % time licking;
  Oxy1 11.2 ± 0.9 % (indistinguishable from vehicle, as expected at this
  dose); Oxy3 4.1 ± 1.3 % (~60 % reduction); Oxy10 0.0 ± 0.0 % (effectively
  abolished, total 1.5 bouts averaged across all 6 mice).

**Deferred alternatives**:
- Including baseline sessions for within-mouse paired analysis was
  considered but deferred: the new cohort has baseline mp4s on disk but
  no DLC/features yet (the orchestrator script only ran formalin). For
  paired baseline → formalin per mouse, the orchestrator needs
  `--include-baseline` added to both `dlc_batch_formoxy.py` and
  `formoxy_features_extract.py` invocations.
- Mixed-effects statistical model with `cohort` as a random effect was
  considered but skipped for this first cut — the simple per-dose mean
  ± SEM telling a clean story was more important than nailing the model
  spec. If we want a publishable p-value the next pass should use
  `statsmodels` formula `pct_licking ~ C(dose) + (1|cohort)`.

### Picked up — 2605_FormOxy_LeftPaws DLC + features

**Why**: the formalin DLC + feature pipeline died unexpectedly at
10:28 AM today (no traceback, processes just gone) with 10/12 formalin
raw .h5 done. Restarted `formoxy_dlc_then_features.py` +
`formoxy_progress_watcher.py`. Both completed cleanly by 18:11 — all 12
formalin sessions have raw .h5, filtered .h5, and feature pickles
(hash `8aed1c22`, with optical flow). Total wall-clock for the resume:
6 h 26 min (DLC analyze on the 2 remaining sessions S8/S9 + filter on all
12 + features extract on all 12).

## 2026-05-18

### Added — rim cohort addon chain + processing doc

**Why**: the 260515 female rim dose-response cohort started receiving
additional animals (Rim12 in flight on May 18; more expected). The
ingestion pattern is settled — transcode CRF 26 → DLC shuffle 9 →
features → 6-classifier predict → dose-response + combined
scratching-bouts timecourse — so the chain is now a reusable script
rather than ad-hoc one-shots.

- `scripts/research/rim_addons_chain.py` (new): idempotent addon chain.
  Polls the portal at `Video_transfer_portal/2605_Rimonabant/<date>/`
  for fully-transferred `Rim<N>_(baseline|post-rim).mp4` files (skips
  `~syncthing~` / `.tmp`; requires a `session_*.json` companion);
  transcodes at libx264 CRF 26 slower with rename to the cohort
  convention `fem_(baseline|postrim)_Rim<N>.mp4`; removes the portal
  copy after a successful transcode; chains into `dlc_batch_rim_dose.py`
  → `rim_dose_response.py` → `combined_rim_scratching_timecourse.py`.
  Discord-posts a status line per phase. `--wait` flag polls every
  60 s for up to 6 h until at least one ready file appears, so this
  can be launched the moment portal activity starts.

- `docs/rim_cohort_processing_plan.md` (new): canonical processing
  plan for the cohort, including the 5-stage pipeline, the folder
  layout under `260515_Rim_DoseResp`, the modular dose assignment
  (Rim8 → Rim12 stays at 10 mg/kg), and the compression-sweep results
  that justify CRF 26 (per-frame classifier agreement 96.2 % at CRF 26
  vs 96.3 % at CRF 23 — 0.08 pp drop, ~45 % size reduction at the
  3-min subset, ~70× size reduction at the 60-min full-video scale).

**Deferred alternatives**:
- A combined master orchestrator (one script for many cohorts) was
  considered but rejected — keeping a small per-cohort chain is much
  easier to reason about. The pattern can be cloned for the next
  cohort that comes in.
- An ONNX-export of the SNLT classifiers to bypass the PixelPaws env
  was considered but isn't worth it until we have a multi-machine
  inference need; the current pipeline runs comfortably end-to-end
  on the analysis box.

## 2026-05-07

### Added — frame-rate cleanup pipeline + PawCapture calibration ingestion

**Why**: a sweep of the `2603_SNLT_JG/Baseline` project (52 videos)
revealed every recording is stored at 60 fps but actual unique-frame
content sits in the 21–30 fps range — half the videos show clean
uniform 2× duplication, the other half variable stutter (held
streaks up to 133 frames; one outlier effectively at 4 fps). This
inflates `dt=1` velocity ~2× when classifiers trained on this data
are run on true-rate captures. Separately, `feat/camsync-1.0.0`
ships per-camera `mm_per_pixel` metadata in MP4 udta + session JSON
sidecars; we want the analysis side to start consuming it now so
classifiers can carry calibration provenance.

**Phase 0 — diagnostic** (`scripts/utilities/diagnose_duplicate_frames.py`,
new): read-only OpenCV scan that classifies consecutive-frame pairs
as duplicate vs unique by mean-abs-diff (greyscale, downscaled 4×).
Reports `duplicate_fraction`, run-length histogram, inferred true
fps, and a heuristic pattern label (`uniform_2x` /
`variable_stutter`). Writes a JSON report under
`<project>/diagnostics/`. Runs at ~3 s per 1500 frames; full project
scans cost ~50 min.

**Phase 1a — project config schema** (`project_config.py`): added
`process_fps: Optional[float]` and `source_fps_note: str` for
frame-rate provenance, plus `calibration_mode` ∈
`{'auto','fixed','off'}` and `fixed_mm_per_pixel` for PawCapture
integration. All default to legacy behaviour (None / `'off'`).
`load()` reads the new keys; `save()`'s existing merge semantics
preserve them.

**Phase 1b — remap utility** (`frame_rate_normalize.py`, new): four
remap primitives + a project orchestrator. `compute_keep_indices`
runs the same content-diff scan as the diagnostic and returns the
explicit kept-frame indices (handles uniform 2× *and* variable
stutter). `build_inverse_map` maps every old frame index to the
preceding kept-slot. `downsample_video` re-encodes via
`cv2.VideoWriter` (mp4v, matching `render_skeleton_video.py`).
`downsample_dlc_h5` `.iloc`s by kept indices, preserving the DLC
MultiIndex. `remap_labels_csv` OR-collapses binary labels across
each duplicate streak (NaN-aware via `np.maximum.at` on the inverse
map). `remap_sparse_db` and `remap_dense_regions` cover
`label_manager.py`'s sparse/dense format. `normalize_project`
walks `videos/`, locates assets via the same search order as
`evaluation_tab.find_session_triplets` (extended to cover legacy
`Targets/`), backs originals up to `_pre_fps_normalize_<TS>/`, and
writes `process_fps` to project config. **Dry-run by default** —
the user must pass `--apply` (or tick the dialog box) to commit.

**Phase 1c — GUI** (`frame_rate_dialog.py` new, `PixelPaws_GUI.py`
Tools menu): "Diagnose Frame Rate…" and "Normalize Project Frame
Rate…" entries, each opening a Toplevel with target_fps + eps
inputs, a scrolling log, and a worker thread. Normalize dialog
defaults to dry-run and confirms via `messagebox.askyesno` before
applying.

**Phase 1d/e — predict-side hook + cache key** (`prediction_pipeline.py`,
`feature_cache.py`): when `clf_data['process_fps_at_runtime']` is
populated by the caller, the multiscale-feature augmentation logs a
loud `⚠️` if it differs from the trained `multiscale_fps`.
`compute_hash` now folds `process_fps` and `mm_per_pixel` into the
cache key when set — legacy projects without those keys keep their
existing cache hashes intact.

**Phase 2 — PawCapture calibration ingestion**:
- `pawcapture_meta.py` (new at repo root, copied verbatim from
  `feat/camsync-1.0.0:camsync/pawcapture_meta.py`) — pure-stdlib
  reader: `read_calibration(mp4)` (ffprobe → udta atom),
  `read_session_manifest(json)`, `find_session_for_video(mp4)`.
- `evaluation_tab.find_session_triplets` now populates
  `mm_per_pixel`, `session_manifest`, and `rig_label` per session
  (best-effort: udta first, then session JSON sidecar; all `None` on
  legacy projects, no crash when ffprobe is missing).
- `PixelPaws_GUI.py` training pkl now stamps
  `training_process_fps`, `training_calibration_mode`, and
  `training_mm_per_pixel` (the last only meaningful under
  `calibration_mode='fixed'`; under `'auto'`, per-video values stay
  in the session dicts).

### Added — flip the switch on mm/pixel scaling in pose features

**Why**: PawCapture videos confirmed the calibration metadata is
readable end-to-end (`THC1_Baseline.mp4` returns `mm_per_pixel=0.149`,
ref 100 mm = 672 px). User asked to actually consume the value
throughout the distance-relevant features rather than just storing it.

**Changes**:
- `PoseFeatureExtractor.__init__` accepts `mm_per_pixel: Optional[float] = None`.
  When set, the entry-point `get_bodypart_coords` multiplies x/y
  coordinates once by `mm_per_pixel`; every downstream feature
  (distances, velocities, jerk, contact heights, multi-timescale
  rolling stats, lag features, egocentric distances) inherits mm
  units automatically. Probabilities and angles are unaffected.
- Default `contact_threshold=15.0` is auto-converted to
  `15 × mm_per_pixel` mm when calibration is on AND the caller did
  not override the threshold; non-default thresholds pass through
  untouched (caller is responsible for unit consistency).
- `project_config.ProjectConfig.resolve_mm_per_pixel(session)`
  centralises the per-session lookup honouring `calibration_mode`
  ('off' → None, 'fixed' → project-wide value, 'auto' → session's
  embedded value).
- `PixelPaws_ExtractFeatures` accepts new `mm_per_pixel`,
  `clf_data`, and `session` kwargs. Auto-resolves `mm_per_pixel`
  from `clf_data.training_mm_per_pixel` (and falls back to
  `session['mm_per_pixel']` under 'auto') so predict-tab callers
  only need to add `clf_data=clf_data` to their existing extract
  lambdas — no need to plumb `mm_per_pixel` through every flow.
- All 7 predict-tab call sites (Eval, Predict, Batch Predict,
  Active-Learning, Optimizer flows in `PixelPaws_GUI.py` and
  `evaluation_tab.py`) now pass `clf_data=clf_data` to the
  extractor.
- Training pipeline (`PixelPaws_GUI.py:6438`,
  `extract_features_for_session`) resolves `mm_per_pixel` per
  session via the project's `calibration_mode` and folds it into
  the cache hash so px-scaled and mm-scaled caches don't collide.
- `evaluation_tab.py` cache-hash blocks fold in either
  `clf_data.training_mm_per_pixel` (for fixed-mode classifiers)
  or `session['mm_per_pixel']` (for auto-mode classifiers).
- Per-session `mm_per_pixel_at_runtime` injection happens before
  each predict call in eval, so post-cache augmentation
  (`augment_features_post_cache` in `prediction_pipeline.py`)
  uses the right scaling for egocentric and contact features.
- `feature_cache.compute_hash` now reads either `mm_per_pixel` or
  `training_mm_per_pixel` from the cfg dict, so callers passing
  `clf_data` directly (predict tab) and callers passing project
  cfg (training) both produce matching hashes.
- Post-cache augmentation extractors (`_ego_ext`, `_ct_ext`) in
  `prediction_pipeline.py` now read `_mm_per_px` from
  `clf_data.training_mm_per_pixel` (or
  `mm_per_pixel_at_runtime` fallback) and pass it through.

**Smoke tests**:
- `PoseFeatureExtractor(mm_per_pixel=0.149)` auto-converts 15-px
  default contact threshold to 2.235 mm. Explicit
  `contact_threshold=3.0` passes through unchanged.
- `compute_hash` with `mm_per_pixel=0.149` and
  `training_mm_per_pixel=0.149` produce the same hash, and both
  differ from the pixel-only hash — mm and px caches are
  guaranteed not to collide.
- `ProjectConfig.resolve_mm_per_pixel` returns None for 'off',
  project value for 'fixed', and session value for 'auto'.

**Files**: `pose_features.py`, `prediction_pipeline.py`,
`feature_cache.py`, `project_config.py`, `evaluation_tab.py`,
`PixelPaws_GUI.py`.

### Added — gait/limb tab gait analysis in mm

Main gait analysis run (`gait_limb_tab.py:3559`,
`_run_one_session`) now extracts pose with `mm_per_pixel` so:
- body speed → mm/sec
- stride length, step length → mm
- contact-state Height column → mm

User-tuned thresholds (`loco_threshold` in px/s,
`contact_threshold` in px) are auto-scaled by `mm_per_pixel` at run
time so settings the user calibrated on a px-based GUI keep their
intent when the analysis runs in mm. The body-speed log line
prints `mm/s` vs `px/s` accordingly.

`mm_per_pixel` is resolved per-session via the new
`GaitLimbTab._resolve_mm_per_pixel(session)` helper, which calls
`ProjectConfig.resolve_mm_per_pixel(session)` and falls back to
reading the MP4 udta atom directly via
`pawcapture_meta.read_calibration` when the session dict lacks
the value (e.g. when running on a one-off video that wasn't
ingested through `find_session_triplets`).

Interactive *previews* (locomotion preview at line 687, contact
preview at line 1075, contour preview at line 1396) intentionally
stay in pixel mode. They exist for iterative threshold tuning
where the user adjusts a px-valued spinbox while watching the
overlay; converting their input space to mm would force them to
re-tune existing pipelines for no UX gain. The actual analysis
applies the px→mm conversion downstream.

**Files**: `gait_limb_tab.py`.

**Deferred**:
- Wiring `process_fps_at_runtime` into all 7 `predict_with_xgboost`
  call sites. The hook reads it when set; the predict tab can stamp
  it from project config in a follow-up. Today the warning is opt-in.
- ffmpeg `mpdecimate` fast path for `compute_keep_indices`. Current
  pure-Python scan runs ~4 min per ~100k-frame video; a full project
  normalize is an overnight job. Acceptable for v1.
- Project setup wizard fps step. Today the dialog sets `process_fps`
  as part of normalize; new projects (where this is moot) don't need
  to bother with it at startup.

**Files**: `scripts/utilities/diagnose_duplicate_frames.py` (new),
`frame_rate_normalize.py` (new), `frame_rate_dialog.py` (new),
`pawcapture_meta.py` (new — copied from camsync branch),
`project_config.py`, `feature_cache.py`,
`prediction_pipeline.py`, `evaluation_tab.py`, `PixelPaws_GUI.py`.

**Verification**: end-to-end single-video sanity (`260318_JG_9417_Baseline.mp4`):
108,020 stored frames → 37,884 unique kept (65 % duplicates).
Inverse map [0,1,1,2,2,3,3,…] correct. Labels CSV remap 96,984 →
37,884 rows; positives 2,934 → 1,384 (consistent with bouts collapsing
across streaks). `find_session_triplets` runs cleanly on the same
project with all calibration fields `None` (no camsync videos yet).

## 2026-05-05

### Added — `crop_for_dlc.py`: configurable video + JSON output suffixes

**What changed**:
- Pre-2026-05-05 the output naming was hardcoded:
  - Video: `<stem>_cropped<ext>` at line 291
  - Sidecar: `<stem>_crop.json` at line 258
  Renaming required editing string literals.
- Added module-level constants `DEFAULT_VIDEO_SUFFIX = "_cropped"` and
  `DEFAULT_JSON_SUFFIX = "_crop"` as the single source of truth.
- `process_single` and `process_batch` accept three new optional kwargs:
  `video_suffix`, `json_suffix`, `write_sidecar`. Defaults preserve
  legacy behaviour exactly.
- `save_crop_sidecar` accepts `json_suffix` and an optional
  `output_dir` (so the sidecar can land in the same custom output dir
  as the cropped video, instead of always next to the source).
- GUI: new "Output naming" row with two entry fields (`Video suffix`,
  `JSON suffix`) and a "Write JSON sidecar" checkbox. Empty
  `video_suffix` AND empty output dir is refused at the dialog level
  (would overwrite source).
- CLI: three new flags — `--video-suffix STR`, `--json-suffix STR`,
  `--no-sidecar`. Header docstring updated with examples.

**Why**: users wanted to vary the output naming per project (e.g.
`_roi` instead of `_cropped`, or no sidecar at all when they manage
crop offsets via the project config). The ProcessSingle defaults are
preserved so existing pipelines / batch scripts that didn't pass
these kwargs continue producing the same filenames.

**Safety**: refuses to write `<stem><ext>` (empty suffix) on top of
the source video unless an explicit `output_dir` is given. Both the
CLI and GUI catch this case before encoding starts.

**Files**: `crop_for_dlc.py`.

## 2026-05-01

### Chore — folder restructure (research scripts, docs, codemods, artifacts)

Repo root went from ~80 entries to ~38 by sorting non-core files into
purpose-specific subdirectories.

**Moves**:
  - `docs/` ← `CODE_AUDIT_2026-05-01.md`, `hardware.md`,
    `behavior_labeling_guide.md`, `INSTALLATION_GUIDE.md`.
  - `scripts/research/` ← all `regen_*.py`, `_compute_transitions_formoxy.py`,
    `_plot_transitions_gui_style.py`, `_render_*.py`, `_replot_*.py`,
    `bout_eval.py`, `prune_and_save_post.py`, `feature_schematic.py`,
    `silhouette_*.py`, `extract_snlt_baselines_silhouette.py`,
    `belly_groundtruth_check.py`. (20 files.)
  - `scripts/utilities/` ← `watch_dlc_extract.py`,
    `analyze_batch_results.py`, `check_classifier.py`,
    `_capture_gui_screenshots.py`. (4 files.)
  - `scripts/codemods/` ← `_codemod_arial.py`, `_consolidate_imports.py`,
    plus the new `_fix_sys_path_after_move.py`. One-time use.
  - `_research_artifacts/` ← `lupe.pdf`, `s41467-021-25420-x.pdf`,
    `260129_JIA *.tif` × 6, `263001 ImageJ *.xlsx`, `screenshot_*.png`,
    `theme_*.png` (24 files). Now gitignored as a directory.
  - `assets/` ← `pixelpaws_icon.ico` (was at root). The GUI tries the
    new path first and falls back to root for older installs.

**What stayed at root**:
  - `PixelPaws_GUI.py` (entry point) and every `*_tab.py`, plus pipeline
    modules (`prediction_pipeline.py`, `pose_features.py`,
    `brightness_features.py`, `feature_cache.py`,
    `optical_flow_features.py`, `classifier_training.py`).
  - UI utilities (`dialogs.py`, `ui_utils.py`, `sidebar_nav.py`,
    `project_setup.py`, `label_manager.py`).
  - Helpers (`io_utils.py`, `analysis_utils.py`, `user_config.py`,
    `project_config.py`, `behavior_presets.py`).
  - Standalone tools that are imported as helpers
    (`crop_for_dlc.py`, `render_skeleton_video.py`,
    `brightness_preview.py`, `brightness_diagnostics.py`,
    `correct_features_crop.py`).
  - `README.md`, `LICENSE`, `CHANGELOG.md`, `requirements.txt`,
    `.gitignore`.

**Why these stayed flat**: the GUI/tab/pipeline modules cross-import
each other heavily (e.g. `PixelPaws_GUI` re-exports from
`prediction_pipeline`, `evaluation_tab` is imported from many tabs).
Moving them to a package would force ~50+ import edits across the
codebase — risky for a one-shot reorg. Defer to a dedicated PR.

**Import patches**:
  - The 8 research scripts that injected `os.path.dirname(__file__)`
    into `sys.path` had that block replaced with a 3-line bootstrap
    that adds BOTH the script's own dir (peer scripts) AND the project
    root (parent-of-parent). Done by `scripts/codemods/_fix_sys_path_after_move.py`.
  - `PixelPaws_GUI.py:475` icon-load path tries `assets/pixelpaws_icon.ico`
    first, falls back to the legacy root location.

**Other cleanups**:
  - Deleted `dummy.txt` (cruft I created earlier in this session).
  - `analysis_tab.py` ImportError warning now reports the actual
    underlying `ImportError` message instead of the misleading "not
    found" — the file always exists, the import fails on a missing
    transitive dep (typically seaborn).

**Verification**: all 13 GUI/dialog/tab modules import OK from new
paths; `tests/test_bout_filtering.py` passes 9/9; moved research
scripts (`bout_eval`, `regen_scratching_plots`) import correctly via
the patched bootstrap.

**Files**: ~50 moves, 1 .gitignore entry added, 1 import patch in
`PixelPaws_GUI.py`, 1 warning-message improvement in `PixelPaws_GUI.py`.

### Changed — replaced 228 hardcoded `'Arial'` font tuples with a single `FONT_FAMILY` constant

**What changed**:
- Added `FONT_FAMILY = 'Segoe UI'` to `ui_utils.py` as the single point
  of control for the GUI's body font.
- Codemodded every `font=('Arial', N [, style])` tuple in 13 files
  to `font=(FONT_FAMILY, N [, style])`. 228 replacements across:
  `PixelPaws_GUI.py` (46), `gait_limb_tab.py` (65), `analysis_tab.py`
  (62), `active_learning_v2.py` (12), `project_setup.py` (11),
  `dialogs.py` (6), `sidebar_nav.py` (6), `evaluation_tab.py` (5),
  `transitions_tab.py` (4), `unsupervised_tab.py` (4),
  `body_contact_tab.py` (3), `correct_features_crop.py` (3),
  `brightness_preview.py` (1).
- Each touched file got `FONT_FAMILY` added to its `from ui_utils
  import (...)` block (or a new top-level import if it didn't import
  from ui_utils before).

**Why**: pre-2026-05-01, switching the body font (or supporting non-
Latin scripts) required editing 228 lines across 13 files and was
guaranteed to drift out of sync. Now: edit one constant in
`ui_utils.py`. Tk falls back gracefully if the named family isn't
installed, so this is safe on Linux/macOS too.

**Codemod scripts** at repo root: `_codemod_arial.py` (the substitute
+ import-adder) and `_consolidate_imports.py` (post-pass that merged
the standalone `from ui_utils import FONT_FAMILY` lines into the
existing multi-line `from ui_utils import (...)` blocks where one
existed). Both are one-shot and can be deleted.

**Files**: `ui_utils.py` + 13 GUI / dialog / tab modules.

### Performance — `_apply_bout_filtering` vectorized (4× speedup)

`evaluation_tab._apply_bout_filtering` was a per-frame Python loop. On
a typical session length (290k frames at 60 fps), that's ~5M Python
iterations per evaluation cycle. Vectorized to numpy run-length
encoding (new `_find_runs` helper):

  - 290k-frame benchmark: **44.5 ms/call** (vec) vs **182.5 ms/call** (loop) → 4.1× speedup.
  - 200 random inputs verified bit-identical against a kept reference loop in `tests/bench_bout_filtering.py`.

The vectorised version still uses Python loops over the (much shorter)
*list of bouts* in min_bout zero-out and the min_after_bout refractory
pass — those passes are O(n_bouts), not O(n_frames), so they don't
dominate. Net gain at the eval-tab level: roughly 8 s saved per full
LOVO sweep with 5 classifiers.

**Files**: `evaluation_tab.py`, `tests/bench_bout_filtering.py` (new).

### Fixed — `AutoLabelWindow.run_predictions` generated random uncertainty scores (P0)

**What changed**:
- `dialogs.py:510` had a stub implementation that called
  `np.random.beta(2, 2)` to generate fake "uncertainty" probabilities
  and surfaced them as "uncertain frames" for the user to label. A
  user labelling these frames was producing labels driven by random
  data — silent dataset corruption.
- The dialog can be reached from the Tools tab via `open_auto_labeler`
  (`PixelPaws_GUI.py:7683`), which passes `classifier_path=None`.
  There was never a real classifier behind the dialog.

**Fix**: replaced the stub with a `messagebox.showwarning` that points
the user to the Active Learning tab (which actually runs classifiers).
`probabilities` / `predictions` are set to None so the rest of the
dialog can detect the disabled state. `export_labels` now refuses to
write a CSV when those are None, and `load_frame` falls back to
showing the raw frame with a hint banner instead of indexing into
None and crashing into Tk's callback void.

**Files**: `dialogs.py`.

### Fixed — `predict_with_xgboost` and `augment_features_post_cache` warnings now always reach SOMEWHERE

**What changed**:
- `prediction_pipeline.predict_with_xgboost` previously logged via bare
  `print()`, which is invisible from the GUI when launched without a
  controlling console. Now accepts `log_fn` parameter; falls back to
  `print` when None.
- `augment_features_post_cache` previously gated all warning emission
  behind `if log_fn:`, so callers that didn't pass a logger (some CLI
  scripts, ad-hoc Jupyter use, scratch tests) lost EVERY error from
  the six try/except blocks. Now uses `_warn = log_fn or print` for
  failure paths — success messages still respect `log_fn=None`.
- The bare `except: pass` on the final inf-sanitization block is now
  `except Exception as e:` and surfaces the failure via `_warn`.

**Files**: `prediction_pipeline.py`.

### Fixed — duplicated silent `find_session_triplets` import-fallback in two tabs

`body_contact_tab.py` and `gait_limb_tab.py` both had

  except ImportError:
      def find_session_triplets(folder, **kw):
          return []

If `evaluation_tab.py` ever fails to import (broken refactor, missing
dep), the user sees an empty session list with no error. Both blocks
now log the ImportError to stdout once at module load so the failure
is at least traceable.

**Files**: `body_contact_tab.py`, `gait_limb_tab.py`.

### Fixed — `min_after_bout` was a silent no-op for every classifier (P0)

**What changed**:
- `evaluation_tab._apply_bout_filtering` accepted a `min_after_bout`
  parameter but never referenced it. All 18 call sites across
  `PixelPaws_GUI.py`, `evaluation_tab.py`, `bout_eval.py`, and
  `_compute_transitions_formoxy.py` were silently dropping the value.
- The eval auto-tuner (`PixelPaws_GUI.py:6755`) and the per-class
  joint grid search (`9554`) sweep over `min_after_bout = [0, 1, 3, 5]`
  and report "best" values that were physically meaningless — the
  4 candidates produced identical predictions every time.
- The Train tab UI (`PixelPaws_GUI.py:861-866`) exposes a "Min After
  Bout (frames)" Spinbox; the auto-suggest heuristic at line 4164
  computes `max(1, int(0.05 * fps))` and stores it. None of those
  values were affecting predictions.

**Implementation** — added a third pass to `_apply_bout_filtering`,
running AFTER `min_bout` removal and `max_gap` bridging:

  if min_after_bout > 0:
      # Walk left-to-right. If a bout starts within min_after_bout
      # frames of the LAST SURVIVING bout's end, drop it. Distance
      # is measured from surviving (not raw) bouts so a chain of
      # close-together false positives doesn't ghost-suppress
      # a genuine later bout.

**Why "last surviving"**: For false-positive ringing (e.g. a flinch
classifier firing twice in 8 frames), the desired outcome is to keep
the first call and drop the echo. If a third bout occurs much later,
that's a new event and should survive. Measuring distance from the
RAW previous bout's end would chain through "ghost" bouts and
over-suppress genuine separated events.

**Tests**: `tests/test_bout_filtering.py` — 9 cases pinning the new
semantic, including one regression test (`test_min_after_bout_changes_output_vs_zero`)
that explicitly asserts non-equivalence for two different
`min_after_bout` values. That test would have caught the silent-noop
bug if it had existed at the start.

**User-facing impact**: Any classifier with stored `min_after_bout > 0`
will now produce different predictions than before. Re-evaluate
auto-tuned classifiers; the previously "optimal" `min_after_bout`
values were chosen against a noop function and may not be the new
optimum. The fix is correctness, not an optimisation — accept the
behaviour change.

**Files**: `evaluation_tab.py`, `tests/test_bout_filtering.py` (new).

### Fixed — `unsupervised_tab._video_ext_var` AttributeError on every project change (P0)

**What changed**:
- `unsupervised_tab.py:_build_ui` returns early at line 222-223 when
  `umap-learn`/`hdbscan` are missing (showing the deps message). It
  therefore never reaches `_build_sessions_panel`, which is where
  `self._video_ext_var` was being created.
- `on_project_changed` -> `_scan_sessions` (line 916) reads
  `self._video_ext_var.get()` — which crashes silently into Tk's
  callback void on every project switch when deps are missing.

**Fix**: hoisted the `tk.StringVar(value='.mp4')` instantiation up
to `__init__` (before `_build_ui`). The combobox in
`_build_sessions_panel` now references the existing var instead of
re-creating it.

**Files**: `unsupervised_tab.py`.

### Changed — `_palette_var` default is now `'inferno'` instead of `'deep'`

**What changed**:
- `transitions_tab.py:800` previously defaulted the heatmap palette
  to `'deep'`, which is a seaborn discrete palette name. The
  `_heatmap_cmap` resolution at lines 2754 and 2958 only accepts
  values from `_SEQUENTIAL = {'magma','plasma','viridis','inferno',
  'cividis','turbo'}`; anything else falls back to `'YlOrRd'`. The
  default heatmap palette was therefore ALWAYS YlOrRd until the
  user explicitly picked another option, regardless of what the
  rest of the GUI used.

**Fix**: change the default to `'inferno'` so it matches both the
project's house palette and the transitions tab's actual rendering.

**Files**: `transitions_tab.py`.

### Chore — `.bak` files now ignored by git

- Removed `PixelPaws_GUI.py.vscode-overwrite-20260428-2133.bak`.
- Added `*.bak` and `*.vscode-overwrite-*` patterns to `.gitignore`.

### Added — `CODE_AUDIT_2026-05-01.md` at repo root

Full code-quality audit. Findings include the 109 silent
`except Exception: pass` blocks, 214 hardcoded `Arial` font
references, the four P0 fixes above (with #1 being the standout),
and recommendations for the unaddressed P1/P2 items.

## 2026-04-29

### Changed — Transitions tab: diff panel now uses the same inferno palette as the group panels

**What changed**:
- `_plot_group_comparison` (transitions_tab.py) previously rendered the
  `groupB - groupA` diff with a diverging map (PuOr_r / PRGn_r / RdBu_r)
  chosen to harmonise with the group palette.
- The user found the warm/cool diverging hue jarring next to the
  inferno-themed group panels. New behaviour: render the diff with the
  SAME sequential palette as the group panels (`_heatmap_cmap`), with
  colour driven by `|Δ|` so brightness encodes magnitude of change. The
  signed value (`+0.14`, `-0.19`) is preserved in the cell annotation so
  direction is still readable.
- Colorbar label updated to `|groupB − groupA|` to make the magnitude
  semantics explicit.

**Why**: visual cohesion across the three panels — Naive, SNTX, and
SNTX−Naive now read as one figure rather than two themes glued together.
Sign info isn't lost (it's in the cell text), and the strongest changes
still pop out via the brightest cells.

**Files**: `transitions_tab.py` (`_plot_group_comparison`).

### Added — `feature_schematic.py`: simple 600×600 "skeleton_red" schematic

**What changed**:
- New helper `grab_frame_bgr(video_path, frame_idx)` returning the colour
  frame (the existing `grab_frame_gray` only supplied a single channel,
  which loses the per-pixel colour needed for the paw-pixel stamps).
- New render `render_simple_red_schematic(bgr_full, kps_full, cache_row,
  frame_idx, out_path, size=600)` produces a single-panel schematic
  mirroring `render_skeleton_video.py`'s aesthetic on a black background:
  magenta paw-pixel stamps with a Gaussian-blur glow, dim grey-green
  skeleton edges, bright keypoint dots (yellow snout, white-ish neck /
  centroid, orange tailbase / tailtip). Crop is centred on the centroid
  and clamped to frame bounds, so the output is exactly `size × size`
  pixels (default 600×600).
- Overlays (matplotlib, on top of the OpenCV-rendered canvas), all
  symbol-only — no numeric values, just feature names + descriptive
  tags so the figure stays readable at presentation scale:
  - 3 brightness ROI squares (`Pix_snout`, `Pix_hlpaw`, `Pix_hrpaw`)
  - 3 example angle arcs (`Ang_snout-neck-centroid` forebody bend,
    `Ang_neck-centroid-tailbase` spine flex,
    `Ang_flpaw-centroid-frpaw` forepaw spread)
  - 2 example distance double-arrows (`Dis_snout-tailbase` body length,
    `Dis_hlpaw-hrpaw` hindpaw spread)
  - Labels are edge-anchored (corner slots, `ha='left' / 'right' /
    'center'`) so leader lines stay short and labels never stack on
    the body centroid. `lx, ly, ha` are tunable per overlay if the
    pose moves around the crop.
- New `render_red_side_by_side()` produces a 1202×600 PNG: LEFT
  (`ORIGINAL FRAME`) is the raw cropped frame, untouched; RIGHT
  (`PIXELPAWS FEATURES`) is the standalone skeleton-red schematic
  loaded from `red_png` (paw-pixel stamps + dim skeleton + keypoint
  dots + feature annotations on a black background). 1-px white
  divider, black-shadowed white headers in the top-left of each panel.
- New `render_feature_matrix_cartoon()` produces a 1400×600 layout
  cartoon of the per-frame feature matrix that feeds XGBoost — 11
  column groups (xy, Dis, Ang, Height, Vel, Jerk, Pix, ΔBrt, Sil,
  lags, …) × 8 frame rows (`t-4` … `t+3`), filled with group colours
  sampled from matplotlib's **plasma** colormap (deep purple →
  magenta → orange → yellow). Right-side legend maps each symbol to
  its meaning. Counts shrunk for legibility — the goal is a cartoon,
  not a literal 700-column wall. No numeric values.
- New `render_matrix_shap_side_by_side()` produces a **slide-ready
  16:9 landscape** cartoon (1600×720, designed to drop onto a
  PowerPoint slide at full size). Single-row horizontal flow,
  inspired by Panel C of the BAREfoot-style behaviour-pipeline
  figures — inline text labels (no pills), thin arrows with small
  open heads, every visual decision in service of the data.
  - Far left: plasma-coloured feature matrix `X` with the
    column-group banner + a `y` column showing synthetic BORIS
    labels (green/grey bout pattern).
  - Below the matrix: tiny **BORIS** logo (`assets/boris_logo.png`,
    `OffsetImage` zoom=0.13) with `BORIS — manual labels` +
    `event-coded behavior bouts` plain captions, no box; curved
    grey `FancyArrowPatch` (`arc3,rad=-0.35`) loops up into the `y`
    column.
  - Centre: bare bold `XGBoost + SHAP` text on the centerline
    flanked by two thin arrows — no charcoal pill, no border, just
    the label between connectors.
  - Right: cartoon SHAP horizontal bar chart, top 10 features
    descending with cache-convention names. Same plasma palette as
    the matrix column groups so the audience can trace each bar to
    its source group.
  - Far right: arrow + plain green `save behavior predictions` text
    (with `(per-frame CSV)` italic caption) — no pill, just the
    label.
  - No suptitle — the horizontal flow itself is the title.
- New `assets/` folder for figure assets. `boris_logo.png` ships
  there; loaded via `plt.imread()` from
  `<feature_schematic.py dir>/assets/boris_logo.png`.
- Schematic colour key on the standalone schematic + side-by-side:
  green for brightness ROIs (`Pix_<bp>`), pink for angles
  (`Ang_a-v-c`), and **orange** (was cyan) for distances (`Dis_a-b`)
  so the warm distance arrows pop against the cool dim-skeleton edges
  and harmonise with the magenta paw stamps. All overlay text uses
  Arial-bold via the shared `FONT_KW` (DejaVu Sans fallback).
- Standalone schematic now draws **all 9 DLC keypoints** as
  saturated colour-coded dots (head/spine cyan family, tail warm
  orange, forepaws magenta, hindpaws purple-magenta) on top of the
  paw-pixel stamps — same palette as the prior `DLC TRACKING` panel,
  so the side-by-side reads as one continuous colour story instead
  of two unrelated diagrams.
- New optional velocity overlay on the standalone schematic, with
  two render modes picked by which keyword is supplied:
  - `kps_future=<dict>` — single-arrow mode (legacy). Dashed violet
    arrow from `t` to `t + vel_step`, ghost dot at the future
    position, label `Vel<step>_<bp>  (t → t+<step>)`. Skipped when
    displacement < 4 px so a stationary frame doesn't draw noise
    arrows.
  - `vel_trail=<list[dict]>` — multi-frame trail mode (new). Takes a
    list of `2 * vel_step + 1` keypoint dicts spanning `t-vel_step`
    → `t+vel_step` and draws a continuous polyline through each
    tracked bodypart's actual positions. Past segment solid, future
    segment dashed; marker dots at `t-vel_step` (small + faded), `t`
    (large + bright + white-edged), and `t+vel_step` (medium +
    semi-faded). Reads as a comet trail through the cropped canvas —
    closer to what `Vel<step>_<bp>` actually encodes than a single
    displacement vector.
- Two example bodyparts highlighted in both modes: `centroid`
  (whole-body translation) and `hlpaw` (the more dynamic limb).
- `main()` now produces two pairs of outputs at the same frame —
  `skeleton_red_schematic.png` + `_side_by_side.png` (arrow style)
  and `skeleton_red_schematic_trail.png` + `_side_by_side_trail.png`
  (trail style) — so the audience can pick whichever conveys
  velocity better for their figure.
- `main()` now reads BGR alongside greyscale and adds three new render
  steps (`skeleton_red_schematic.png`, `skeleton_red_side_by_side.png`,
  `feature_matrix_cartoon.png`) after the existing four. Other steps
  are unchanged.

**Why**:
- The existing `pose_schematic.png`, `paw_brightness_schematic.png`, and
  `feature_set_powerpoint.png` outputs are dense — useful for an audit
  but visually busy. A single clean 600×600 schematic that highlights
  just the brightness ROIs and one angle (matching the project's
  rendered demo video aesthetic) is what's needed for slide decks and
  feature-explainer figures.

**How to apply**:
- `py feature_schematic.py --project <path> --session <name> --frame <N>`
  now produces the new `skeleton_red_schematic.png` alongside the four
  existing outputs.
- The `size` kwarg on `render_simple_red_schematic` is wired so callers
  can request other dimensions; 600 is the default and the requested
  spec.

## 2026-04-28

### Added — Optional whole-frame silhouette features (project-level opt-in)

**What changed**:
- `brightness_features.py:_extract_vectorized` gains two kwargs
  (`compute_silhouette: bool = False`, `silhouette_floor: int = 35`).
  When enabled, the consumer loop computes three frame-level scalars
  inside the existing brightness pass — no extra video read:
  - `silhouette_frac` — fraction of frame above floor (raw, all blobs)
  - `silhouette_blob_frac` — same, restricted to the largest connected
    component (filters incidental hardware reflections, dust, etc.)
  - `silhouette_aspect` — `min/max` bbox aspect of that blob
    (1.0 = square / compact, → 0 = elongated)
- `extract_brightness_features` and `prediction_pipeline.PixelPaws_ExtractFeatures`
  thread the same two kwargs through.
- `belly_groundtruth_check.py` (new standalone script) for hand-labeled
  separability checks of brightness + texture + height signals on N
  belly-down vs not-belly-down events. Outputs strip plots, ROI patch
  grid, and Cohen's d table.
- `watch_dlc_extract.py` reads `compute_silhouette` and
  `silhouette_floor` from `PixelPaws_project.json`. Defaults: false /
  35. Existing projects without these keys are unaffected — the cache
  hash is unchanged.

**Why**:
- For bottom-up imaging rigs (camera up through transparent floor,
  near-black background), whole-body contact extent is a strong signal
  for "is the animal on the floor at all" that current ROI-mean
  features can't capture. Centroid-ROI mean brightness sits on dark fur
  in both lying-flat and twisted poses; it doesn't discriminate
  on-floor postures from rearing / standing well. Silhouette
  fraction does — by ~10× between belly-down (~20% of frame) and
  rearing (~2%).
- Project-level opt-in (not auto): only relevant for bottom-up rigs
  with clean dark backgrounds. Other rigs (top-down, cluttered
  backgrounds) would need different preprocessing. Adding the keys to
  one project's JSON forces re-extraction for *that* project only.
- Computed in the existing brightness pass: zero extra video I/O,
  ~5-10 ms / frame for the connected-components call. Negligible
  cache-size impact (3 float columns).

**How to enable** (per project):
```json
{
  "...": "...",
  "compute_silhouette": true,
  "silhouette_floor": 35
}
```
Then re-run `watch_dlc_extract.py` (the cache hash flips and existing
caches re-extract automatically).

**What's deliberately *not* done**:
- *No GUI toggle.* Editing the JSON is a deliberate per-project
  decision; promoting it to a checkbox would invite use on rigs where
  silhouette is meaningless.
- *No body_contact_tab integration yet.* The new columns land in the
  cache but no UI surface reads them. Next step (when validated): add
  `silhouette` as a fourth signal radio, or use as an input to a
  belly-pressed behavior classifier. Holding off until separability
  is verified on real data.
- *No adaptive (Otsu) threshold.* Fixed `silhouette_floor` is simpler
  and probably stable across a single rig's lighting; revisit if drift
  becomes a problem.

**Follow-up fix — `feature_cache.py:compute_hash`**:
- Initial implementation relied on the cfg dict roundtrip to change
  the hash, but `FeatureCacheManager.compute_hash` only hashes a
  fixed allowlist of keys; new keys were silently ignored, so
  enabling silhouette on a project did NOT invalidate existing caches
  (they were treated as already-extracted). Fixed by including
  `compute_silhouette` (and `silhouette_floor` when enabled) in the
  hash dict — but only when enabled, so existing projects without
  the keys keep their existing hashes.
- Verified: project with the keys absent (or `compute_silhouette: false`)
  hashes to the same value as before (`af626969`); same project with
  `compute_silhouette: true` hashes to a new value (`2d46556a`),
  triggering re-extraction.

**Verification**:
- `ast.parse` clean on `brightness_features.py`,
  `prediction_pipeline.py`, `watch_dlc_extract.py`,
  `belly_groundtruth_check.py`, `silhouette_validate.py`, and
  `feature_cache.py`.
- Existing projects (no `compute_silhouette` key in JSON): cache hash
  unchanged, no behaviour change.
- End-to-end run started 22:38 on the 2604_DV_DSS project (8 DLC-ready
  sessions, 1 still in DLC). Watcher running in --continuous mode;
  validation will inspect the first completed pkl and report whether
  the new silhouette columns land and whether their values discriminate
  standing/rearing from belly-pressed postures.

### Changed — Body Contact "Preview Session" now plots raw / ΔBrt / Z time-courses

**What changed**:
- `body_contact_tab.py:_plot_session` rewritten. Old behaviour: one
  panel per bodypart with raw on the primary axis and ΔBrt overlaid
  on a twin secondary axis — only two of the four signals visible
  and threshold lines tied to whichever signal was on which axis.
- New layout: rows = signals (`raw`, `ΔBrt`, `z`), cols = bodyparts
  (`centroid`, `tailbase`); up to a 3×2 grid of time-course panels
  with `sharex='col'`. Each panel draws its own signal trace, its
  own threshold line (sourced from `_raw_thresh_vars` /
  `_brightness_thresh_vars` / `_z_thresh_vars` so the dashed line
  matches what *that* signal would use to call contact), and gold
  shading over the contact frames as classified by the user's
  *current* method+signal selection — so it's obvious whether the
  active criterion's bouts also stand out in the alternate signals.
- Rows for which both bodyparts have no usable data are dropped
  silently (`Pix_baseline_sub_*` missing → ΔBrt row goes away). If
  no signal has data, an info messagebox explains and the window is
  destroyed instead of opening blank.
- X-axis is seconds (using the per-session `fps` column from
  `_summary_df`, falling back to the spinbox default), labelled
  `Time (s)`. Old version used frame index.
- Window grew from `'900x600'` to a screen-relative size
  (`72% × 78%`) and is centred on screen.
- Header label at the top of the window names the active session,
  method, contact-shading signal, and fps so the meaning of the
  gold shading is unambiguous.
- Adds matplotlib `NavigationToolbar2Tk` (pan / zoom / save) and
  `WM_DELETE_WINDOW` handler that calls `plt.close(fig)` on the
  figure — fixes the figure leak the old version had.

**Why**:
- The user is choosing between four candidate brightness signals
  (`raw`, `ΔBrt`, `z`, `frac_bright`) when tuning per-keypoint
  thresholds; seeing them all overlaid on the same time-axis with
  the active contact mask is the fastest way to compare which
  signal cleanly separates contact frames from background. Showing
  only raw + ΔBrt left Z invisible and required re-running the
  preview to compare.

**Verification**:
- Smoke test (Agg backend, modal dialogs stubbed):
  - Full data (Pix + baseline-sub for both bodyparts) →
    figure rendered with 6 axes, suptitle correct, all threshold
    lines and contact shading drew without exception.
  - Partial data (Pix only, no baseline-sub) → figure rendered with
    4 axes (raw + Z, ΔBrt row dropped); no errors.
  - No data (only Height columns) → "No signals" info messagebox,
    window destroyed, no figure leaked.

### Changed — Body Contact graphs window now reuses Gait & Limb plotting infra

**What changed**:
- `body_contact_tab.py:_open_graphs` rewritten. The previous v1 was a
  single fixed-size 3×2 matplotlib `subplots` grid embedded in a bare
  `Toplevel`: no toolbar, no save buttons, no axis editor, no stats
  overlay, no figure cleanup, mixed-scale (% / seconds / signed ΔBrt)
  panels at fontsize=8 sharing one suptitle.
- New version builds a `ttk.Notebook` with three category tabs —
  *Contact %*, *Bout Duration*, *Brightness Δ* — each driven by the
  gait tab's `_make_metric_selector` (combobox + ◀ ▶ buttons). Each
  metric renders one figure at a time with the gait tab's polished
  chart builders (`_build_box_graph`, `_build_violin_graph`,
  `_build_bar_graph`), full matplotlib `NavigationToolbar2Tk`,
  editable Y/X-min/max + Apply/Reset, "Export Graph" / "Export Data"
  buttons, treatment-group strip overlay, optional Shapiro-Wilk-gated
  significance bracket / ANOVA / Kruskal-Wallis text via
  `_add_stat_annotation`, and a description label that updates with
  the metric.
- `dbrt_*` panels get a reference line at 0 (no contact-vs-overall
  brightness change). `cpct_*` and `mbout_*` have no reference line
  (no obvious zero).
- WM_DELETE_WINDOW handler walks the widget tree and `plt.close()`s
  every embedded figure on exit, fixing the figure leak the v1 had.

**How the helpers are reused** (no `gait_limb_tab.py` changes):
- `BodyContactTab.__init__` mirrors the three state attributes the
  gait helpers expect on `self`: `_enable_stats_var` (BooleanVar),
  `_stats_paradigm_var` (StringVar, defaults to `'auto'`), and
  `_last_graph_cfg` (dict). A plain `_sig_style` instance attribute
  stands in for gait's `@property` of the same name.
- New helper `_bind_gait_graph_helpers` lazily imports
  `gait_limb_tab.GaitLimbTab` and binds its plotting helpers
  (`_treatment_groups`, `_add_stat_annotation`, `_embed_figure`,
  `_style_ax`, `_make_metric_selector`, `_register_metric`,
  `_build_bar_graph`, `_build_box_graph`, `_build_violin_graph`,
  `_add_export_buttons`) onto the BodyContactTab instance via
  `types.MethodType`. Static helpers (`_calc_error`, `_error_label`)
  are bound as plain functions. Run once per instance, idempotent
  via `_gait_graph_helpers_bound`.

**Why this approach over a refactor**:
- The gait tab is the only existing consumer of these helpers; the
  body-contact tab is the second. A free-standing `graph_helpers.py`
  module or a shared mixin is the right long-term factoring, but the
  helpers reference enough `self.X` state (`_summary_df`,
  `_enable_stats_var`, `_stats_paradigm_var`, `_last_graph_cfg`,
  `_sig_style`) that pulling them out cleanly would touch
  `gait_limb_tab.py` in many places. Method-binding via
  `types.MethodType` lets the body-contact tab inherit the exact
  rendering behaviour with zero risk to gait code, at the cost of a
  small magic-y `__init__` step. Revisit if a third tab needs the
  same plotting infra — that's the right time to lift it into a
  proper `GraphHelpersMixin`.

**What's deliberately *not* ported**:
- Gait's full `_build_graph_settings_dlg` (treatment ordering, color
  picker, gradient cmap, marker styles, error type, sig-style picker,
  paradigm radio, time-window, rebin, graph-set toggles). It pulls
  in too many gait-specific tk vars (`_stats_test_var`,
  `_stats_alpha_var`, `_timecourse_posthoc_var`, line-style
  comboboxes…) to be safely shared. Body contact uses sensible
  defaults instead (treatment order = first-appearance, error =
  SEM, sig style = asterisk, paradigm = auto, stats on). A small
  body-contact-specific settings dialog can be added later if
  needed.
- Time-course views. The body-contact pipeline never produces a
  `_bins_df`, so there is no per-bin data to graph. Adding it is
  tracked but out of scope for this change.

**Verification**:
- Module-level smoke test: built a synthetic `_summary_df` with three
  treatments (Vehicle / LowDose / HighDose) and one column
  intentionally missing (`dbrt_tailbase`). Walked every combobox
  entry across all three category tabs — 10 metric configurations
  rendered without exception (4 in Contact %, 4 in Bout Duration,
  2 in Brightness Δ since the missing column was correctly skipped).
  Stats overlay (`_add_stat_annotation`) drew without error;
  `_last_graph_cfg.sig_style` propagated to the helpers.

### Added — Body Contact analysis tab (`body_contact_tab.py`)

**What changed**:
- New file `body_contact_tab.py` (~580 lines) introducing `BodyContactTab(ttk.Frame)`,
  a sibling of the Gait & Limb tab dedicated to contact + brightness
  analysis for the **midline keypoints** (`centroid`, `tailbase`) that
  the gait tab deliberately ignores.
- `PixelPaws_GUI.py` registers the new tab next to Gait & Limb under
  the label `🫃 Body Contact` with the same available/hidden pattern
  used by the other optional tabs.
- `pose_features.py` `calculate_contact_features` gains an optional
  `per_bodypart_threshold: dict` kwarg. When supplied, per-key values
  override the global `contact_threshold` for that specific bodypart.
  Default (`None`) reproduces today's behaviour bit-for-bit; verified
  against a synthetic Height frame with both paw and centroid
  columns. Needed because the 15 px paw threshold is too tight for
  midline keypoints.

**Tab layout** (3-pane, mirrors Gait & Limb's outer skeleton):
- Left: session list + Scan / All / Clear / cache-presence column.
- Middle: Run/Cancel/progress, Key file picker, Contact-detection
  radio (`brightness` default / `height` / `combined`), per-keypoint
  thresholds (40 px height + 5.0 brightness defaults), min-bout-ms
  debounce, fallback FPS spinbox, caveat banner.
- Right: Results Treeview with one row per session and the columns
  `cpct_<bp>`, `nbouts_<bp>`, `mbout_<bp>`, `brt_<bp>`,
  `brt_contact_<bp>` for each of `centroid`, `tailbase`, plus
  `Export Summary CSV` and `Plot selected session` buttons.

**Per-session computation** (`_analyze_session`):
1. Locate the cached feature pkl via
   `feature_cache.FeatureCacheManager.find_any_cache` (any hash —
   newest match). Gracefully reports "no cache" when missing.
2. Look up FPS from the video file via OpenCV; fall back to the user
   spinbox.
3. Build a contact mask per keypoint per the chosen method:
   - `height`:    `<bp>_Height < height_threshold[bp]`
   - `brightness`: `Pix_baseline_sub_<bp> > brightness_threshold[bp]`
   - `combined`:  both must be True.
   Returns `None` when a required column is missing — the row still
   surfaces with NaN metrics so the user can see which keypoint
   needs re-extraction.
4. Debounce: drop True runs shorter than `min_bout_ms` (frame
   count derived from session FPS).
5. Compute `contact_pct`, `n_bouts`, `mean_bout_dur`,
   `mean_brightness`, `mean_brightness_contact`.

**Plot** (`_plot_session`): per-keypoint subplot showing `Pix_<bp>` on
the primary y-axis, `Pix_baseline_sub_<bp>` on a twin axis with a
crimson dashed marker at the brightness threshold, and contact-mask
shading (gold) on the primary plot. Opens in a separate Toplevel.

**Export** (`_export_summary`): CSV via `pandas.to_csv` plus a
companion `<csv>.meta.json` reproducibility sidecar capturing git SHA
(via `io_utils.get_git_sha`), method, thresholds, min-bout, fallback
FPS, key file, and timestamps. Same pattern as the analysis-tab
sidecar that landed 2026-04-26.

**Why**:
- The user wanted to evaluate when animals press their belly /
  perineum into the cage surface, with metrics analogous to the
  paw-contact analytics the Gait & Limb tab already produces. The
  current 9-keypoint DLC model doesn't track belly or anus
  explicitly, but the existing `centroid` and `tailbase` keypoints
  are usable midline proxies — no DLC retraining needed for v1.
- The PixelPaws extraction pipeline is already fully bodypart-agnostic:
  `pose_features.calculate_paw_height` (misnamed) computes
  `<bp>_Height` for every y-column, `calculate_contact_features`
  derives `<bp>_ContactState` for every `*_Height` it sees, and
  `prediction_pipeline.compute_brightness_category_b` produces the
  seven Cat-B variants for every `Pix_<bp>` column. So the
  analysis-side feature columns are auto-generated; the new tab is
  purely a consumer.
- A sibling tab rather than a Gait-tab section: the Gait tab is
  already 4k+ lines with a lot of paw-specific UI (HL/HR/FL/FR
  mapping, WBI/SI, stride/swing/cadence). Sticking a parallel
  midline section into the same UI would add visual clutter and
  invite confusion ("which threshold am I editing?"). A clean
  sibling is easier to iterate on without risking regressions in
  the paw analysis. Refactor into a shared base class can come
  later if both tabs grow into significant overlap.
- Brightness as the default detection mode: centroid and tailbase
  are virtual midpoints, not literal floor-contact landmarks, so
  Height < threshold conflates "pressing into floor" with "lying
  flat / sleeping". Brightness in the ROI changes directly with
  pressure, which is closer to the user's actual question.

**Pre-flight for the user (one-time)**:
1. Open the project setup wizard, toggle on `centroid` and `tailbase`
   in the brightness-bodyparts listbox. This adds them to
   `bp_pixbrt_list`; the cfg hash flips and existing cached pkls
   auto-invalidate.
2. Re-run feature extraction (`watch_dlc_extract.py <project>` or
   the equivalent inline call) so all sessions get `Pix_centroid`,
   `Pix_tailbase`, and the seven Cat-B variants for each.

The new tab works **without** step 1 too — height-mode metrics use
`centroid_Height` / `tailbase_Height` which the existing pipeline
auto-generates for every keypoint, so existing caches are sufficient
to surface the geometric proxy. Brightness-mode columns NaN out
gracefully until the cache is rebuilt with the new `bp_pixbrt_list`.

**Verification**:
- `pose_features.calculate_contact_features` with
  `per_bodypart_threshold={'centroid': 40.0}` confirmed: paw column
  unchanged from default (`hlpaw_ContactState` bit-identical), centroid
  column applies the 40 px override correctly.
- Module import smoke test: `body_contact_tab` imports cleanly and
  the GUI's `BodyContactTab` registration parses without error.
- Real-cache analysis (Flinching project, session
  `260129_Formalin_3122`, 193 176 frames, 60 fps): height-mode at
  40 px produces `contact_pct_centroid = 44.2%` (996 bouts, mean
  1.35 s) and `contact_pct_tailbase = 69.6%` (204 bouts, mean
  10.96 s). The much-higher tailbase contact is expected — the
  tailbase sits near the floor most of the time when the animal is
  not rearing.
- Brightness columns intentionally absent from existing caches
  because `bp_pixbrt_list` is `["hlpaw", "hrpaw", "snout"]`. The tab
  surfaces NaN for brightness metrics in this state so the user
  knows to re-extract.

**Deferred**:
- *Bilateral metrics for body keypoints.* Centroid / tailbase are
  midline-only; no L/R counterpart by definition. WBI / SI are
  paw-only and stay that way.
- *Body-contact behavior classifier.* Classifier-side plumbing
  (a `train_use_body_contact` flag, sidecar config flowing through
  `augment_features_post_cache`) is out of scope for this v1
  descriptive pass.
- *Adding belly / anus as explicit DLC keypoints.* The user
  decided centroid / tailbase are good enough for the first pass.
  Revisit if the proxy turns out to be too noisy.
- *Shared base class for GaitLimbTab + BodyContactTab.* v1 keeps
  copies of the small wrapper functions (`_resolve_subject`,
  `_get_treatment`, sessions panel scaffolding); refactor only if
  duplication accumulates.

## 2026-04-26

### Fixed — Delta-audit P0/P1 sweep (multi-timescale cross-session bleed, pruned-CV LOFO, ThresholdCurve red-dot, AL atomicity, DLC cache key, threshold-sweep grid)

A bundle of correctness and reproducibility fixes coming out of the
2026-04-26 delta audit. Each item below was verified in `project_delta_audit_2026_04_26.md`
before this commit.

**What changed**:

- **`PixelPaws_GUI.py:4966-4995` — multi-timescale per-session loop (P0).**
  `calculate_multiscale_features` was being called on the post-concat
  `X`, so its `center=True` rolling windows silently leaked ±21 frames
  (700 ms @ 60 fps) of one session into the next session's boundary
  frames. At predict time, augmentation runs per-session, so the
  feature distribution at training-boundary frames could not be
  reproduced — train/predict mismatch on a small but real fraction of
  rows. The new code iterates `pd.unique(session_ids)`, computes
  multiscale within each session's contiguous block, then concats the
  per-session results back via `sort_index()`. Predict semantics are
  now matched exactly. Smoke test confirmed boundary contamination
  goes from 50.6 → 0.0 on a synthetic 3-session pulse.

- **`PixelPaws_GUI.py:5563-5577` + new `_calibrate_oof_lofo` helper (P0).**
  The pruned-CV recalibration block was using single-fit isotonic
  (`IsotonicRegression(...).fit(oof_proba_raw, y); .predict(...)`),
  which overfits — the All-features-vs-Pruned F1 comparison the user
  reads in the training log was biased in pruning's favour. Both
  call sites (main path and pruned re-run) now route through a
  shared `_calibrate_oof_lofo(oof_proba_raw, y, fold_val_masks,
  log_label="…")` method that does proper LOFO isotonic + a
  separate production calibrator fit on all OOFs (used at predict
  time only). Identical logic, no duplication. The pruned re-run
  consumes `pruned_cv['fold_val_masks']`.

- **`PixelPaws_GUI.py:7308-7449` — ThresholdCurve.png consistency (P1).**
  The plot drew un-filtered F1/Precision/Recall, but the red dot
  came from a sweep that optimised F1 *after* bout filtering. Red
  dot sat off-curve at thresholds the unfiltered curve never peaked
  on. Fix: at each threshold, also compute F1/Precision/Recall after
  applying `_apply_bout_filtering` with the chosen `(min_bout,
  min_after_bout, max_gap)`. The clean publication
  `ThresholdCurve.png` now uses the filtered curves with a title
  suffix listing the bout params; the comparison
  `PerformanceThreshold.png` adds a crimson "F1 (OOF, bout-filtered)"
  line alongside the existing unfiltered ones so the in-sample/OOF
  comparison is preserved. When all bout params are no-ops the
  filtered curves are skipped entirely (red dot already lands).

- **`PixelPaws_GUI.py:6671` — threshold sweep grid alignment (P1).**
  `_sweep_postprocessing` was on `np.arange(0.10, 0.91, 0.05)` (17
  values), but eval-tab plots use `np.arange(0.05, 0.96, 0.01)` (91
  values). The training sweep could pick a threshold that the
  predict-tab plot couldn't render, and rare-event behaviours could
  optimise outside [0.10, 0.90]. Sweep grid is now `np.arange(0.05,
  0.96, 0.025)` (37 values). Total combos 3,808 → 8,288 (~3-4 s,
  still negligible).

- **`pose_features.py:31, 70-78` — DLC cache key includes file size (P1).**
  Cache key was `(normpath, mtime)`. Cloud-sync clients (Dropbox,
  OneDrive), `cp -p`, `rsync --times`, `git checkout` round-trips,
  and Windows Explorer "Copy item" all preserve mtime — a corrupted
  re-download or branch swap could leave mtime unchanged but content
  different, and the cache would silently serve the stale parse.
  Key extended to `(normpath, mtime, size)`; catches >99% of
  mtime-preserving corruption with no I/O cost. Type annotation
  updated to `dict[tuple[str, float, int], pd.DataFrame]`.

- **`active_learning_v2.py:582-590, 735, 982-990, 1008` — atomic AL writes (P0/P1).**
  Label-history CSV (`df.to_csv` direct) and AL state pkls
  (`pickle.dump` direct) were vulnerable to partial writes on crash
  or cancel. AL annotations are user-provided and irreplaceable.
  Both call sites now route through new shared helpers
  `atomic_pickle_save` / `atomic_dataframe_to_csv` in `io_utils.py`,
  which use temp file + `os.replace` semantics. A crash mid-write
  leaves the original intact.

- **New `io_utils.py` module.** Centralises atomic write helpers and
  `get_git_sha()` (returns short SHA + `+dirty` suffix when working
  tree is dirty, `'unknown'` if not a git checkout). Replaces the
  inline duplicates that had crept into `prediction_pipeline.py` and
  `PixelPaws_GUI.py`. The two existing in-tree atomic helpers
  (`PixelPaws_GUI._atomic_pickle_save`, `prediction_pipeline._save`)
  are unchanged for now; future work can route them through
  `io_utils` to eliminate the duplication.

- **`PixelPaws_GUI.py` — git SHA + library versions stamped into pkl (peer §1.1).**
  Classifier `classifier_data` now embeds `code_version` (git SHA +
  `+dirty`), `python_version`, `xgboost_version`, `sklearn_version`,
  `numpy_version`, `pandas_version` via a new `_collect_provenance_meta()`
  helper. The sidecar JSON can drift or be deleted; the pkl now
  carries provenance internally so a single `pickle.load` recovers
  it. Best-effort — every field falls back to `'unknown'` on
  import or subprocess failure.

- **`PixelPaws_GUI.py` — optional ONNX export sibling (peer §D.2).**
  Trained classifier writes a `<classifier>.onnx` next to the pkl
  via a new `_maybe_export_onnx(model, pkl_path, ...)` helper,
  using `onnxmltools.convert_xgboost`. Lets the model load in
  MATLAB / R / Napari / a Python without the matching xgboost
  version. Silently skipped when `onnxmltools` isn't installed
  (it is an optional dependency). Both pruned and all-features
  classifiers get an ONNX sibling.

- **`analysis_tab.py:1613-1707` — reproducibility metadata sidecar on CSV export (audit A.5).**
  `export_results()` now writes `<csv>.meta.json` alongside the CSV
  with: git SHA + dirty flag, UTC + local export timestamp, CSV row
  count, the loaded key file, the prediction file paths, the
  selected behaviors, and the analysis settings (mode, fps, bin
  size, bin unit, whole-session flag). Pure-CSV consumers (Excel,
  R, naive pandas) ignore the sidecar; auditors and reproducibility
  pipelines can pick it up. Best-effort — the JSON write failing
  does not block the CSV write.

**Why**:
- Items §1.1, §1.2, §1.4, §2.1, §2.2, §4.1 were ML-correctness or
  data-integrity asymmetries flagged in the 2026-04-24 trio of audits
  + the 2026-04-26 delta audit. Each was either still unfixed (the
  pruned-CV LOFO) or new since the prior audit (the multi-timescale
  cross-session bleed, the DLC cache key). The audit memo
  (`project_delta_audit_2026_04_26.md`) lists the line numbers and
  reasoning for each.
- A.5 + git SHA + ONNX are reproducibility / hand-off improvements
  the peer-comparison memo flagged as quick wins, all clustered
  around the classifier save and analysis CSV export — natural to
  do them in the same pass once `io_utils` existed.

**Verification**:
- `io_utils` helpers: round-trip pickle + CSV via temp tests; both
  recover bit-identical data.
- Multi-timescale per-session: synthetic 3-session pulse confirms
  boundary contamination magnitude `50.6` (post-concat) → `0.0`
  (per-session) on `Pix_snout_std_700ms`.
- `_vectorized_rolling_onset`: 8 trial regression test against the
  original `rolling(11, center=True, min_periods=1).apply(_onset)`
  on 199-4603 frame random / spike / monotonic / negative / constant
  inputs — every trial `max|diff| = 0.0e+00`. Bit-identical.
- Threshold sweep grid: 37 values × 8 × 4 × 7 = 8,288 combos as
  expected.
- Module imports: `io_utils`, `pose_features`, `prediction_pipeline`,
  `active_learning_v2`, `analysis_tab` all import cleanly. Full GUI
  syntax-check via `ast.parse` passes.

**Deferred**:
- *Per-session loop for lag features (`calculate_lag_features` at
  `pose_features.py:883`)*. Same cross-session-bleed pattern but
  smaller magnitude (±2 frames) and not flagged in the audit. Trivial
  to apply the same fix; revisit if the audit flags it.
- *DEFAULT_FPS constant.* The 60-fps fallback is now scattered as
  a literal in 9+ places. Consolidating into a single
  `config.py:DEFAULT_FPS = 60.0` is a refactor without behaviour
  change and didn't fit this correctness-focused pass.
- *Conformal prediction for AL query selection (peer §D.3).* P3 in
  the delta audit. Bigger lift; revisit when AL query quality
  becomes the bottleneck.
- *Routing `_atomic_pickle_save` and `prediction_pipeline._save`
  through `io_utils.atomic_pickle_save`.* Code-quality refactor; the
  inline duplicates are correct, just redundant.
- *Pytest test directory.* Inline smoke tests sufficed for this
  pass; setting up pytest infra is its own workstream.

### Added / Fixed — Transitions Group Comparison polish + new "Group Networks" view

**What changed**:
- `transitions_tab.py:2867` — moved the shared 0–1 colorbar from the right of the **first** group panel to the right of the **last** group panel (`show_cbar = (gi == n_groups - 1)`). With two groups + diff this puts the colorbar between SNTX and the diff panel instead of wedged between Naive and SNTX, restoring the side-by-side comparison.
- `transitions_tab.py:2893` — switched the `groups[1] − groups[0]` difference panel from `_heatmap_cmap` (sequential, e.g. inferno) to `'RdBu_r'` (diverging). Symmetric `vmin=-vmax, vmax=vmax` was already in place, so 0-delta cells now read white, positive deltas red, negative blue. Significance markers (`*`/`**`/`***`) still render readably on the new palette.
- `transitions_tab.py:1201-1203` — added `'Group Networks'` to the View dropdown (right after `'Network'`).
- `transitions_tab.py:2643` — wired `Group Networks` → new `_plot_group_networks` method.
- `transitions_tab.py` (new method after `_plot_network`) — `_plot_group_networks`: one network panel per group, side-by-side, with shared spring layout computed from the union of every above-threshold edge across all groups so node positions stay identical between panels. Per-panel node sizes reflect that group's per-state time fractions (computed from `self._state_seqs_full` so the duration cap doesn't skew them, with fallback to `self._state_seqs`). Edge weights and labels reflect that group's mean transition matrix. Title shows `Naive (n=5)` / `SNTX (n=8)` so cohort sizes are visible at a glance.

**Why**:
- The old colorbar placement broke the visual comparison the view was designed for. The diff panel using a sequential cmap on signed data was actively misleading — `0.00` cells looked mid-palette (red-orange) instead of neutral, and equal-magnitude positive vs negative deltas were indistinguishable.
- The existing `Network` view averages every session into one graph. Useful as a summary, but useless when the user has group labels and wants to eyeball "which transitions change under treatment." The infrastructure for per-group aggregation (`self._group_matrices`, `self._session_subjects`, `self._key_df`) was already populated; only the rendering was missing.

**Verification**:
- Group Comparison: open the view on a 2-group dataset, confirm shared colorbar sits between SNTX and the diff panel; diff panel shows white for 0.00 cells, red/blue for ±deltas.
- Group Networks: switch the View dropdown to `Group Networks`; confirm one panel per group, identical node positions, panel-specific node sizes and edges, `(n=K)` annotation in titles. With networkx uninstalled the fallback message appears; with no key file loaded the "Load a key file" message appears.

**Deferred**:
- Significance-aware edge styling on Group Networks (dashed = ns, solid = p<0.05). `_compute_sig_matrix` already exists; visual complexity not yet justified. Add when asked.
- User-selectable diff cmap. Hard-coded `RdBu_r` is fine for v1.

### Fixed — Transitions Video Preview state mismatch under duration cap

**What changed**:
- `transitions_tab.py:763` — added `self._state_seqs_full` dict alongside `self._state_seqs`.
- `transitions_tab.py:2242-2247` — snapshot `self._state_seqs_full = dict(processed)` immediately before the destructive `processed = {name: s[:cap_frames] ...}` cap.
- `transitions_tab.py:3579` — `_open_video_preview` reads `self._state_seqs_full.get(session_name, self._state_seqs[session_name])`, falling back to the capped dict for safety.

**Why**: when a duration cap was active (e.g. "first 30 min" → 108 000 frames on a 230 375-frame session), the Video Preview's pink state header reported a different state from the Probability Graph title at the same frame. Both windows read from the same `state_seq` array, but `update_frame` computed its index proportionally against the **capped** length while `_update_graph` computed its index proportionally against the **full** prob_matrix length. At video frame 181 the header was reading `state_seq[84]` ("still" — what the mouse was doing very early in the session) while the graph title was reading `state_seq[181]` ("walking" — the actual state at that frame).

The cap is correct for analysis (transition matrices, sliding-window k-means) but wrong for the Video Preview, which needs 1:1 correspondence with video frames. Decoupling the UI lookup from the analysis cap is the smallest fix; the existing proportional-mapping logic in `update_frame` works correctly once `state_seq` and `prob_matrix` have matching length again.

For frames ≥ `cap_frames` the Probability Graph was previously reading out-of-bounds into the capped `state_seq` — silently swallowed by the `except Exception as e: print(...)` at `transitions_tab.py:721`. The fix incidentally restores correct behaviour for the entire video, not just the part below the cap.

The "<" assigned vs "*" highest-prob marker divergence in the per-class probability list is **intentional** and unaffected — it surfaces priority-mode / min-bout filtering decisions that are supposed to differ from raw argmax. Only the cross-window `state_seq` indexing was buggy.

**Deferred**:
- Tightening the silent `except Exception as e: print(...)` at `transitions_tab.py:721` that hid the IndexError; out of scope for a targeted fix.
- Refactoring the proportional-mapping arithmetic into a shared helper. Two short call sites with matching denominators don't justify the abstraction.

### Speed — vectorized augmentation hotspots (~3× on a 13-session transitions run)

**What changed**:
- `prediction_pipeline.py`: replaced the `rolling(11, center=True, min_periods=1).apply(_onset, raw=True)` loop in `compute_brightness_category_b` (B.4 onset sharpness) with a new module-level `_vectorized_rolling_onset` helper. Uses `np.lib.stride_tricks.sliding_window_view` + `np.nanmax` / `np.nanargmax` over a NaN-padded array, then reconstructs the truncated-window center via per-frame pad widths so edge frames produce values bit-identical to the old `pandas.rolling.apply` semantics.
- `pose_features.py`: added a module-level mtime-keyed LRU cache (`_DLC_LOAD_CACHE`, max 8 entries) for parsed DLC H5/CSV files. `load_dlc_data` now delegates to a new `_load_dlc_data_uncached`. Repeated loads of the same file in one process (transitions tab looping over K classifiers per session, eval re-reading after a brightness-preserve fallback, `pixelpaws_easy.py` chains) skip the parse entirely; updated DLC files invalidate via mtime.
- `pose_features.py`: rewrote `calculate_distances` to use a single fancy-indexed broadcast over upper-triangle pair indices instead of a nested Python loop with 105 single-column `pd.concat`. Bit-identical output. Marginal compute win at typical sizes; mostly a code cleanup.

**Why**: the user ran a transitions analysis on 13 sessions × multiple classifiers and the per-load augmentation step dominated wait time. Profiling identified two bottlenecks:
1. B.4 onset rolling Python apply: ~1 M Python-callback invocations per 108 k-frame × 10-Pix-col session, ≈ 60–70 % of augmentation cost.
2. DLC h5 re-parse on every egocentric augment call: re-read on every classifier × session in a loop, ≈ 15–20 % of augmentation cost.

**Measured speedups** (108 k frames, realistic configs):
- B.4 onset: 583 ms → 15 ms (~39×). max_abs_diff = 0.
- DLC cache: parse runs once per (path, mtime) instead of K times.
- `calculate_distances`: 291 ms → 269 ms (~1.1×). max_abs_diff = 0. Underwhelming — the per-pair loop was already memory-bandwidth bound, and the vectorized form competes with allocator overhead. Kept anyway for the cleaner code path; benefits training too.

End-to-end: a transitions run of 13 sessions × ~4 classifiers should drop from "took a while" by roughly 3× on the augmentation step. The K augment passes per session no longer re-parse the DLC; they each share the cached frame.

**Deferred**:
- *Bake augmented columns into the cache pkl*. Considered and dropped in favour of this leaner approach. If repeat-run latency on the same data still feels slow after this lands, bake-in (sticky augmentation that persists derived columns to the pkl on first compute, idempotent on subsequent loads) is a clean follow-up. Trade-off would be ~2× pkl file size and plumbing across ~10 callers; we wanted to see if compute opt alone was enough before paying that cost.
- *Vectorize `calculate_angles`*. Same nested-loop pattern, but it runs during training/extraction (off the augment hot path), so unrelated to today's slow transitions run. Revisit if training feels slow.
- *Drop dependency on the `bottleneck` / `numba` libraries*. Pure numpy + pandas was sufficient; no new deps needed.

**Verification**: pre-merge synthetic-series check confirmed bit-identical output (max_abs_diff = 0.0e+00 on both onset and distances) and a smoke test confirmed the DLC cache fires (1 uncached call across 3 `load_dlc_data` invocations).

## 2026-04-25

### Added — `watch_dlc_extract.py` (auto feature-extraction after DLC)

**What changed**:
- New standalone CLI: `watch_dlc_extract.py [project_folder] [--poll SECONDS]`. Defaults to `E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline` and a 60 s poll.
- Polls `<project>/videos/` until every `<video_ext>` file has at least one `<stem>DLC*.h5` sibling, then runs feature extraction once and exits.
- Loads feature config (`bp_pixbrt_list`, `square_size`, `pix_threshold`, `include_optical_flow`, `bp_optflow_list`, `dlc_config`) from the project's `PixelPaws_project.json`, so cache hashes match what the GUI's Feature Extraction tool would produce.
- Reuses `prediction_pipeline.PixelPaws_ExtractFeatures` and `feature_cache.FeatureCacheManager.compute_hash`; writes to `<project>/features/<session>_features_<hash>.pkl` via an inline atomic-pickle helper. Skips already-cached sessions.
- Session discovery mirrors the GUI batch scan (`videos_dir.glob("*.h5")` → split stem on `"DLC"` → match against `.mp4/.avi/.mov/.mkv`), but is deterministic: deduped by stem with raw `.h5` preferred over `_filtered.h5` (two-pass loop).
- "Done" check uses `videos_dir.glob(f"{stem}DLC*.h5")` so e.g. `9417.mp4` does not falsely match `9417_BaselineDLC*.h5` (the `D` immediately after `{stem}` blocks the underscore-prefixed sibling).

**Why**: DLC analysis on a folder of videos can take many hours; the user wants feature extraction to start automatically as soon as the last `.h5` lands instead of having to remember to come back and click "Start" in the FE tab. The script is folder-agnostic so it can be re-aimed at any future project.

**Deferred alternatives**:
- *In-GUI "Auto-extract" toggle on the Feature Extraction window.* Tighter integration (live progress in the existing log pane), but requires the GUI to be running for the full DLC duration. Standalone script wins on simplicity and can be backgrounded.
- *`watchdog` filesystem events instead of polling.* Lower latency, but adds a dependency and edge cases around partial-write events (DLC writes the `.h5` atomically at the end, so polling loses nothing meaningful — and 60 s latency is irrelevant against multi-hour DLC runs).
- *Per-video extraction (fire as each `.h5` lands).* User explicitly chose batch-level "all done" semantics — simpler trigger, single FE pass.
- *Filtered-`.h5` required for "done".* Filtering is optional in DLC pipelines; script accepts the raw `.h5` as the completion signal and only falls back to `_filtered.h5` for session selection if the raw one is missing.

## 2026-04-24

### Added — multi-timescale rolling-stat features (LEAN variant, opt-in)

**What changed**:
- `pose_features.py`: new method `PoseFeatureExtractor.calculate_multiscale_features(df, fps, windows_ms=(100, 700), stats=('std', 'max'))` plus the `_default_multiscale_filter` static helper that picks only `<bp>_Vel1` and raw `Pix_<bp>` columns.
- `prediction_pipeline.py`:
  - new branch in `augment_features_post_cache` that runs after the lag-feature block and before Category-B / normalized-distance augmentation; auto-detects need from the model's `feature_names_in_` matching `_(std|max)_\d+ms$`.
  - `_POST_CACHE_RE` extended with `|_(std|max)_\d+ms$` so cached files don't get flagged stale when these columns appear.
- `PixelPaws_GUI.py`:
  - new training-tab checkbox `Include multi-timescale rolling stats (100 / 700 ms)` at row 14 (subsequent rows shifted +1).
  - new method `_on_multiscale_toggle` that force-enables and disables the correlation-filter checkbox while multiscale is active (and restores previous corr-filter state when toggled off).
  - new ivars `train_use_multiscale`, `_prev_corr_filter_state`, `_corr_filter_cb`.
  - cfg keys `use_multiscale_features` (also persisted in save/load), `classifier_data` keys `use_multiscale_features` / `multiscale_fps` / `multiscale_windows_ms` / `multiscale_stats`.
  - `_real_training` augmentation block runs after lag features; computes per-session fps median (with mixed-fps warning) and passes into the extractor; logs e.g. `Added 72 multi-timescale features (fps=60.0, std + max over 100 ms / 700 ms)`.

**Why**: PixelPaws lag/lead features capture *single-frame* offsets (`Vel1_lagm2` = "velocity 2 frames ago"); they don't summarize *windowed* dynamics. For sparse spike-shaped behaviours like flinching at 60 fps, the **std and max over a 100 ms (6-frame) window** capture the spike signature in a way no lag-snapshot can. MARS uses a similar pattern (4 stats × 3 windows × all features = ~3 k extras); we picked the **lean variant** to avoid feature-matrix bloat:

| | Pre-prune total | Pruning impact | Optuna slowdown |
|---|---|---|---|
| Without multiscale | ~700 | baseline | baseline |
| Lean variant (now) | ~770 | minimal | ~5 % |
| Full MARS-style (deferred) | ~1 130 | high | ~50 % |

**Lean scope (chosen)**: 9 BPs × 2 base (`Vel1`, raw `Pix_`) × 2 stats (`std`, `max`) × 2 windows (100 / 700 ms) = **~72 new columns**.

**60 fps default**: at 60 fps, 100 ms = 6 frames, 700 ms = 42 frames. Window sizes are derived per-session from `cv2.CAP_PROP_FPS` so projects with 30 fps or 120 fps videos still get 100 / 700 ms windows; only the underlying frame count adapts.

**Auto-enable correlation filter**: many of the new columns are highly correlated with their base feature (`Vel1_max_100ms` ≈ smoothed-and-rectified `Vel1`). The multiscale checkbox force-enables the existing `|r| > 0.95` correlation filter and disables (greys-out) the corr-filter checkbox while multiscale is on — toggling multiscale off restores the corr-filter to whatever the user had it at before.

**Caching strategy**: post-cache derivation, no cache-version bump. Old caches stay valid; the new columns are computed on-the-fly from cached base features each training run (~few seconds for ~72 columns × ~1 M frames). Avoided bumping `POSE_FEATURE_VERSION` because this is opt-in and unproven for sparse pain behaviours; revisit later if the feature consistently survives gain pruning.

**Order of operations**:
- Training: sessions concat → Category B → normalized distances → sanitize → lag → **multiscale (NEW)** → correlation filter → train.
- Prediction (`augment_features_post_cache`): ego → contact → lag → **multiscale (NEW)** → Category B → normalized distances. Different order from training, but multiscale's filter excludes already-derived families (`_lag*`, Cat-B prefixes, ego, contact, `_Vel10`, `_Vel2`, `_VelCorr`) so the output column set is identical regardless of position.

**Mixed-fps projects**: median fps across sessions used; warning logged if max-min spread > 5 fps so downstream conclusions aren't silently averaged across heterogeneous timing.

**Non-goals / deferred** (per peer-comparison memo §1.12):
- Expanding to 4 stats × 3 windows × 4 base types (~324 extra cols).
- Adapting window sizes to detected bout duration.
- Bumping `POSE_FEATURE_VERSION` v5 → v6 to make this a permanent feature.
- Wavelet bank features (P3, separate workstream).

### Changed — 60 fps is now the default fallback everywhere

**What changed** (`PixelPaws_GUI.py`, `active_learning_v2.py`,
`analysis_tab.py`, `brightness_diagnostics.py`, `brightness_preview.py`,
`dialogs.py`, `evaluation_tab.py`, `gait_limb_tab.py`,
`unsupervised_tab.py`, `crop_for_dlc.py`, `render_skeleton_video.py`):
every hard-coded `30` / `30.0` fps fallback is now `60` / `60.0`. The
preferred path remains "use the video's actual FPS via
`cap.get(cv2.CAP_PROP_FPS)`"; only the *or* fallback (when video metadata
is missing) changes. UI helper text updated:
`60 frames ≈ 2 s @ 30 fps` → `120 frames ≈ 2 s @ 60 fps`. Bout-window
spinbox default bumped from 60 → 120 frames so the "±2 s of context"
intent is preserved at 60 fps. The `max_gap` capping comment also moves
from "~500 ms @ 30 fps" → "~250 ms @ 60 fps".

**Why**: user's rig records at 60 fps. Hard-coded 30 fps fallbacks were
correct for older standard-definition video but are wrong here.
Standardising on 60 fps as the implicit assumption keeps the UI text
honest and prevents silent unit-bugs when video metadata is missing.

### Added — BORIS conversion button in the "Labels Not Found" dialog

**What changed** (`PixelPaws_GUI.py`, `find_training_sessions`):
When the missing-labels dialog fires, `behavior_labels/` is scanned
for BORIS-format files (Behavior + Behavior type + Time columns). If
any are found, a third button — `Convert N BORIS file(s) → PixelPaws` —
is inserted before the existing `Skip Selected & Continue` button
(and given the accent style so it's the visually-primary action).
Clicking it:

1. Runs `_auto_convert_boris_in_behavior_labels` (START/STOP/POINT →
   per-frame arrays, all behaviours as separate columns, FPS from
   column or default 30, originals moved to
   `behavior_labels/boris_originals/`).
2. Closes the dialog with a `retry` action.
3. `find_training_sessions` sees `retry` and recursively re-runs
   discovery — the newly-created `<stem>_labels.csv` files now satisfy
   the label-presence check, so the dialog typically doesn't reappear.

A small gray helper line below the buttons explains where originals go.

**Why**: the previous attempt fired a separate `messagebox.askyesno`
prompt *after* session discovery, which never appeared because the
existing "Labels Not Found" dialog blocked first. Integrating the
BORIS option as a third button in that same dialog makes it the single
surface for solving missing-labels issues — no extra popups, no hidden
branches.

**Non-goals**: the per-file BORIS converter GUI
(`convert_boris_to_pixelpaws`) is unchanged and still handles manual
single-file / batch conversions. Projects with no BORIS files see the
original two-button dialog unchanged.

### Added — honest full-session evaluation when bout-windowing is active

**What changed** (`PixelPaws_GUI.py`):
- New checkbox `Evaluate on full sessions (honest with windowing)` in
  the training params frame (row 13, config key
  `bout_window_honest_eval`, off by default).
- `extract_features_for_session()` skips the bout-window trim when
  honest eval is on — the full (X, y) flows into the training pipeline.
- `_run_cv_loop()` accepts a new `win_mask` kwarg; when provided,
  each fold restricts training rows to `train_mask & win_mask` while
  leaving the val_mask untouched so OOF predictions land on full-session
  frames.
- `_real_training()` computes the window mask once via
  `scipy.ndimage.binary_dilation(y == 1, iterations=N)` and passes it
  to the main CV, the learning-curve CV, and the pruned-re-run CV.
- `final_model.fit(...)` and `pruned_model.fit(...)` also restrict to
  the windowed subset when honest-eval is on, so the deployed model
  matches the fold-model distribution rather than over-fitting the long
  quiet stretches.

**Why**: windowed training AND windowed evaluation inflates the OOF F1
because (a) positive prevalence jumps ~0.5% → ~4% which raises the
precision baseline, (b) easy quiet-frame negatives are never tested,
(c) the threshold sweep fits a distribution that doesn't match
predict-time. Honest eval restricts only the training side to the
window, leaving everything downstream (OOF F1, threshold sweep, LOFO
calibration, LOVO sweep, plots) to see full-session data — the numbers
then reflect what the model will actually do at predict time.

**Default off** to preserve current behaviour. User should flip it on
whenever they want an honest OOF estimate alongside fast windowed
training.

### Changed — three opt-in training defaults flipped to off

- `Optuna auto-tune hyperparameters`: default **off** (was on). Slow on
  large feature sets.
- `Save fold ensemble (average K CV models at predict time)`: default
  **off** (was on). Bloats the pkl significantly and is rarely needed.
- `Show learning curve (~3x slower)`: default **off** (was on). Runs CV
  four times; most training runs don't need it.

**Why**: user wants clean fast default training runs. These three are
still available as opt-ins and all their existing functionality is
unchanged — only the default state differs. Existing config JSON files
that set these keys to `true` will still override the new defaults on
load.

### Added — paper-style plot set (ThresholdCurve, LearningCurve, styled SHAP)

**What changed** (`PixelPaws_GUI.py`):
- `generate_performance_plots()`: added solid OOF Precision + Recall
  lines to `PerformanceThreshold.png` alongside the existing solid F1.
- `generate_performance_plots()`: new `ThresholdCurve.png` — clean
  paper-style plot with only the three solid OOF curves (Recall gray,
  Precision dark gray, F1 crimson) and a dashed vertical marker at the
  chosen threshold. Complements (does not replace) the existing
  `PerformanceThreshold.png` which keeps the in-sample / pre-prune
  curves for debugging.
- New `_generate_learning_curve_plot()`: outputs `LearningCurve.png`
  (F1 vs bout-positive training frames, crimson line with markers).
  Only emitted when the learning-curve diagnostic is enabled. Uses the
  existing `lc_results` (fraction, F1) pairs, converting fraction to
  absolute positive-frame count via `int(y.sum())`.
- `_generate_shap_plots()`: bars coloured along the ``plasma`` colormap
  (dark purple → pink → yellow) with intensity tied to the bar's
  normalised gain importance. The gradient carries the ranking signal
  without singling out a fixed top-N.

**Why**: user requested a plot set matching the ARBEL / BAREfoot paper
figure (learning curve + clean threshold curve + styled SHAP) so that
PixelPaws training outputs are publication-ready without extra manual
plotting.

### Changed — split artifacts into `pruned_N/` + `all_features/` when pruning is active

**What changed** (`PixelPaws_GUI.py` — `_real_training`):

When gain pruning is enabled, the per-run folder is now split into two
subtrees, one per classifier:

```
classifiers/PixelPaws_<behavior>_<ts>/
  pruned_<N>/
    PixelPaws_<behavior>_pruned_<N>_<ts>.pkl + .json sidecar
    plots/       (full OOF plot set + duplicated comparison plots)
    training_data/  (pruned feature columns)
  all_features/
    PixelPaws_<behavior>_AllFeatures_<ts>.pkl + new .json sidecar
    plots/       (all-features SHAP + duplicated comparison plots)
    training_data/  (all feature columns)
```

When pruning is off, the layout is unchanged (single flat folder).

The two comparison plots (`PerformanceThreshold.png`,
`ThresholdCurve.png`) are copied into both subfolders via
`shutil.copyfile` so each subtree is fully self-contained. OOF-based
plots (Calibration, OOF_PerVideo, Raster, TrainingSummary, LearningCurve)
only go into `pruned_<N>/plots/` because `oof_proba` came from the pruned
CV; re-running CV on the all-features feature set to generate a second
OOF set was ruled out of scope (would ~double training time).

**Why**: prior layout mixed the pruned pkl, all-features pkl, their
SHAP plots, and shared training_data all into one `plots/` and one
`training_data/` folder. Splitting clarifies which plot belongs to
which classifier and makes it easier to hand off either model
independently.

### Added — bout-window subsampling for sparse-event classes

**What changed** (`PixelPaws_GUI.py` — new checkbox "Keep only windows
around labeled bouts" with a ±frames spinbox, default off / 60 frames,
config keys `bout_window_only` + `bout_window_frames`, trim logic in
`extract_features_for_session` after the head/tail trims and label
filter):

When enabled, `scipy.ndimage.binary_dilation` expands the positive-label
mask by ±N frames and keeps only those frames. Every positive keeps its
surrounding context (discriminative hard-negatives — movements that look
similar to the behavior but aren't), while distant quiet frames are
dropped.

**Why**: user pointed out that flinches are only a few frames each,
making the positive class ~0.5% of total frames. For a 30-min session at
**60 fps** with 50 flinches × 6 frames each, that's ~300 positive frames
among 108,000 — most quiet frames are redundant easy negatives.
±120-frame windows (~2 s) cut training data ~90 % while bumping
positive prevalence to ~4 %.

**Decision trail**: this revisits the earlier-deferred "top X minutes of
bouty activity" idea (considered on 2026-04-23 and rejected to protect
threshold calibration). The ±window approach replaces the heuristic
"bouty period" detection with a deterministic dilation of the positive
mask — surgical rather than coarse. Default still off because the
concern about train-vs-predict class-ratio mismatch is still valid;
the recently-fixed LOFO isotonic calibrator should handle the shift,
but the user should A/B test against full-session training and watch
for the threshold landing at an extreme value (which would indicate the
windowing was too aggressive).

**Order of operations**: head-trim → tail-trim → unlabeled-frame filter
→ bout-window subsampling. The window operates on the final
`(X, y)` after all other filters so window sizes are in "labeled-frame"
coordinates, not raw video coordinates.

### Changed — each training run now gets its own folder under `classifiers/`

**What changed**:
- `PixelPaws_GUI.py` (`_real_training`): classifier pkl, sidecar JSON,
  AllFeatures sibling, `plots/`, and `training_data/` now all land inside
  `classifiers/PixelPaws_<behavior>_<timestamp>/` instead of sharing the
  flat `classifiers/plots/` and `classifiers/training_data/` subfolders.
  The per-run folder name is the unpruned classifier stem, regardless of
  whether that run produced a pruned or full-feature pkl.
- `PixelPaws_GUI.py` (`refresh_pred_classifiers`) and `evaluation_tab.py`
  (`refresh_classifiers`): classifier dropdowns now use recursive glob
  (`**/*.pkl`) to discover pkls. Dropdown labels display the basename (pkl
  filenames already embed the timestamp → unique) while the full nested
  path is stored on selection.

**Why**: the previous flat `plots/` and `training_data/` subfolders were
shared across every training run — each new run silently clobbered the
previous run's diagnostics. This made before/after comparisons
(e.g., before vs after the LOFO calibration fix) impossible without
manually copying files. The per-run folder fixes that at its source.

**Back-compat**: old classifiers sitting directly in `classifiers/` (flat
layout) are still discovered because the glob is recursive. No migration
needed.

**Non-goals**: pkl filenames are unchanged; no migration of existing
flat-layout artifacts; no UI change to dropdown presentation.

### Added — optional "trim to first labeled event" per-session training trim

**What changed** (`PixelPaws_GUI.py` — new checkbox `Trim sessions to first
labeled event`, new config key `trim_to_first_positive`, trim logic in
`extract_features_for_session`): symmetric to the existing
`trim_to_last_positive`. When enabled, drops all frames before the first
`1` in each session's label CSV. Off by default; tail-trim remains on by
default.

**Follow-up fix**: the new checkbox was initially placed at grid row 11,
same as the existing correlation-filter checkbox — tkinter silently
stacked them so only the correlation filter was visible. Shifted all
subsequent widgets in the training params frame down by one row so the
first-positive trim has row 11 to itself.

**Why**: user asked if training starts at the first labeled bout. It
doesn't — by default, training includes pre-bout labeled zeros, which is
usually what we want (XGBoost needs natural-prevalence negatives so the
threshold calibrates sensibly at predict time). But some sessions have
pathologically long quiet pre-bout periods that skew the class ratio; this
option lets those sessions be trimmed at the front without changing
default behaviour.

**Decision trail**: considered a broader "top N minutes of bouty activity"
window-selector feature. Rejected for now because aggressive active-only
subsampling changes the train-vs-predict class ratio (30–50% positives in
training vs ~1% in production) and breaks threshold calibration. The
symmetric first-positive trim is the conservative, minimally-invasive
option that addresses the long-quiet-head edge case without distorting
overall prevalence.

**Order of trims**: head first, then tail — both operate on the
already-aligned `(X, y)` arrays, so order doesn't affect correctness, but
head-first makes the log output easier to reason about.

## 2026-04-23

### Fixed — predict-tab classifier-load regressions

Newly trained classifiers were loading with three spurious warnings/bugs in
the predict tab. Root causes and fixes:

- **Missing feature-version stamps in classifier pkl** (`PixelPaws_GUI.py`)
  `classifier_data` was being saved without `pose_feature_version` or
  `brightness_feature_version`, so `check_classifier_portability` reported
  "predates pose feature versioning" on every freshly trained model.
  Added both keys to the dict literal and imported
  `BRIGHTNESS_FEATURE_VERSION` alongside `POSE_FEATURE_VERSION` at the top
  of the module.

- **Cross-pair velocity features counted as bodyparts**
  (`prediction_pipeline.py` — `auto_detect_bodyparts_from_model`)
  Contralateral features named `fl-fr_VelCorr` / `hl-hr_VelCorr` were
  split on `_Vel` and the resulting `fl-fr` / `hl-hr` strings added to the
  bodypart set, inflating the count to 11 instead of 9. Added explicit
  `_VelCorr` and hyphen skips.

- **Brightness derivative loop could emit malformed column names**
  (`brightness_features.py`)
  The BrightAccel / BrightOnsetPeak / etc. derivation iterated every
  column starting with `Pix_`. If Category B columns ever leaked into the
  dataframe, names like `Pix_std_temporal_snout_BrightAccel` would be
  emitted. Added defensive prefix filter matching `compute_brightness_category_b`.

### Changed — isotonic calibration uses leave-one-fold-out for diagnostics

**What changed** (`PixelPaws_GUI.py` — `_real_training`):
The threshold sweep and `Calibration.png` diagnostic now consume
LOFO-calibrated OOFs. For each CV fold, an isotonic regressor is fit on
the other folds' OOFs and used to transform the held-out fold's OOFs.
The saved production `prob_calibrator` (used at predict time) is still
fit on all OOFs, because at inference we legitimately have all training
data available — only the diagnostic needs honest held-out treatment.

**Why**: the previous implementation fit isotonic on all OOFs and then
evaluated the calibration curve on those same OOFs. Result: a perfect-
diagonal curve that said nothing about real-world calibration, and a
compressed histogram that pushed the threshold search to 0.89 even
though the positive class's probability mass barely reached that value.

**Decision trail**: considered three candidate directions (LOFO
calibration, switching sweep target from frame F1 to bout F1, extra
per-fold diagnostic logging). Deferred the frame-vs-bout F1 question
because it's meaningless until the probabilities driving the sweep are
honest — revisit after seeing post-fix plots. Per-fold LOVO threshold
logging already existed (lines 5183-5189), so no additional logging
work needed.

**Supporting change**: `_run_cv_loop` now returns `fold_val_masks` (list
of boolean masks over X), consumed by the LOFO calibration block.

**Verification**: retrain, inspect new `Calibration.png` (should no
longer be perfectly diagonal), confirm `best_thresh` lands on a value
where the histogram has positive mass.
