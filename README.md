# PixelPaws — User Guide

PixelPaws is an open-source desktop application for automated scoring of mouse behavior from below-acrylic video. It combines DeepLabCut pose estimation with per-behavior XGBoost classifiers: pose and pixel-brightness features are extracted from each frame, classifiers assign per-frame behavior probabilities, and the application produces per-session predictions, group statistics, and figures. A pretrained pose network and a set of validated classifiers are included, so behavioral scoring requires no labeling or training to begin.

## Demo

Example of automated behavior scoring using a scratching classifier. Behavior detections are overlaid frame-by-frame on the video.








<video src="https://github.com/user-attachments/assets/e0a584de-79b6-440f-a04d-0090ccbda179" controls width="720"></video>

---

> 📷 **Building the filming enclosure?** See [hardware.md](hardware.md) for the full bill of materials, 3D-printed enclosure files, wiring notes, and camera/lens specs.

## Quick Start — score videos without training anything

PixelPaws includes a pretrained pose network and validated classifiers for hind-paw
licking, scratching, facial grooming, body grooming, rearing, jumping, moving, and
stillness. The licking classifier scores the left hind paw, the injected or injured
side in the assays it was validated on; if your manipulation is on the right paw,
mirror the videos or note the sidedness in your analysis. Labeling and training are
not required to begin scoring:

1. **Install and launch** (below). The Project Setup Wizard creates a project; the
   Behaviors step applies only to training new classifiers and can be skipped.
2. **Pose-track your videos** on the Pose Estimation tab. The bundled
   `PixelPaws — SlivickiR_WangH` network is installed automatically; select it and run
   your videos to produce the DeepLabCut `.h5` files used by all downstream analyses.
3. **Score behavior** on Run Classifiers: the **Run default classifier set** button
   scores every project video with the validated bundled set in one click (features
   are extracted and cached automatically). Checking *Run the default classifier set*
   in the pose-tracking dialog does the same automatically after tracking. Predict &
   Review (under Train & Evaluate) offers single-video scoring with a review player.
4. **Analyze groups**: place a key file (CSV with `Subject` and `Treatment` columns)
   anywhere in the project — it is discovered automatically — and the Analysis and
   Sequencing tabs arrive pre-populated once predictions exist. If no key file is
   found, PixelPaws offers to create one when scoring finishes.

Name videos with underscore-separated tokens (`mouse1_veh.mp4`, `m07_sni_day3.mp4`):
subjects are matched as whole tokens against the key file, so `mouse1` in the key file
finds `mouse1_veh_predictions.csv`, while spaces or hyphens in file names defeat the
matching.

Training a new classifier (Train Classifier tab, with BORIS labeling) is needed only
for behaviors the bundled set does not cover.

---

## Table of Contents

1. [Quick Start](#quick-start--score-videos-without-training-anything)
2. [Installation](#installation)
3. [Project Setup Wizard](#project-setup-wizard)
4. [Preparing Your Data](#preparing-your-data)
5. [Labeling with BORIS](#labeling-with-boris)
6. [Crop for DLC Tool](#crop-for-dlc-tool)
7. [Tab-by-Tab Guide](#tab-by-tab-guide)
   - [Pose Estimation](#pose-estimation-tab)
   - [Feature Extraction](#feature-extraction-tab)
   - [Predict & Review](#predict-tab)
   - [Run Classifiers](#run-classifiers-tab)
   - [Evaluate](#evaluate-tab)
   - [Train Classifier](#train-tab)
   - [Single-Classifier Analysis](#analyze-tab)
   - [Multi-Classifier Analysis](#multi-classifier-analysis-tab)
   - [Sequencing](#sequencing-tab)
   - [Locomotion](#locomotion-tab)
   - [Gait & Limb Use](#gait--limb-use-tab)
   - [Tools](#tools-tab)
8. [Project Folder Layout](#project-folder-layout)
9. [Requirements](#requirements)
10. [Attribution & License](#attribution--license)

---

## Installation

```bash
git clone https://github.com/rslivicki/PixelPaws.git
cd PixelPaws
pip install -r requirements.txt
```

**Launch:**
```bash
python PixelPaws_GUI.py
```

---

## Project Setup Wizard

The Project Setup Wizard opens automatically on first launch. It walks through three steps:

**Step 1 — Project folder.** Choose an existing folder or create a new one. PixelPaws creates the standard subfolder structure inside it (`videos/`, `behavior_labels/`, `classifiers/`, etc.) and writes a `PixelPaws_project.json` config file.

**Step 2 — Behaviors (training only).** If you plan to score with the bundled classifiers, click **Use defaults and finish** here — the remaining steps configure classifier training and are not needed for scoring. Otherwise, enter the names of the behaviors you plan to train classifiers for (e.g., `Lick`, `Groom`, `Rear`); these can be changed later from the Train tab.

**Step 3 — Body parts & features (training only).** Choose which DeepLabCut body parts to use for feature extraction and whether to include pixel-brightness features. Prediction with an existing classifier always uses the configuration stored in that classifier, so these choices affect only classifiers you train yourself.

After finishing the wizard the main window opens with every tab ready to use.

---

## Preparing Your Data

### DeepLabCut output

PixelPaws reads the `.h5` file that DeepLabCut produces after analyzing a video. Place the video file and its corresponding `.h5` file in the `videos/` subfolder of your project. The files must share the same base name (e.g., `session01.mp4` and `session01DLC_resnet50_...h5`). PixelPaws matches them automatically by scanning for H5 files in the same folder as the video.

DLC CSV output (`.csv`) is also supported as a fallback.

### Label CSVs

Behavior labels are one row per video frame, one column per behavior (0 = absent, 1 = present):

```
frame,Lick,Groom
0,0,0
1,1,0
2,1,0
3,0,1
```

Save label files in the `behavior_labels/` subfolder. The filename must contain the same session identifier as the video (e.g., `session01_labels.csv` for `session01.mp4`). PixelPaws discovers labels automatically during training and evaluation.

Label files can be created with any tool that produces this format — BORIS exported in the right mode, or a custom script.

### Feature caching

The first time PixelPaws processes a video it extracts all pose and brightness features and saves them to `features/` as a `.pkl` file keyed to the video. Subsequent runs skip extraction and load from cache, which makes retraining with different hyperparameters fast. Delete a cache file to force re-extraction (e.g., after changing body-part or ROI settings).

---

## Labeling with BORIS

[BORIS (Behavioral Observation Research Interactive Software)](https://www.boris.unito.it/) is a free, widely-used tool for scoring animal behavior from video. PixelPaws accepts BORIS exports directly through its built-in converter.

### Labeling workflow in BORIS

1. Open your video in BORIS and create an ethogram with the behavior(s) you want to train (e.g., `Lick`, `Groom`).
2. Score each session using **START/STOP** events (for continuous behaviors) or **POINT** events (for instantaneous ones). Both are supported.
3. When finished, export the observation via **File → Export events → Save as CSV** (or TSV). Make sure the export includes at minimum the following columns:
   - **Behavior** — the behavior name
   - **Behavior type** — `START`, `STOP`, or `POINT`
   - **Time** — timestamp in seconds
   - **FPS** *(optional)* — if included, the converter reads it automatically; otherwise you enter it manually

### Converting BORIS labels to PixelPaws format

PixelPaws needs labels as a per-frame CSV (one row per video frame, one column per behavior, values 0 or 1). The BORIS converter handles this translation.

**Via the Tools tab or Tools menu → BORIS to PixelPaws:**

1. Click **Browse** and select your BORIS export file (CSV or TSV).
2. Click **🔍 Auto-Detect** to scan the file and pick the behavior you want to convert from a list, or type the behavior name directly.
3. Enter the video **FPS** (frames per second). If your BORIS export has an FPS column, leave the field blank and it will be read automatically.
4. Choose an **output directory** (defaults to the same folder as the BORIS file).
5. Click **🔄 Convert**.

The converter produces a file named `<boris_filename>_labels.csv` with a single column named after the behavior:

```
Lick
0
0
1
1
1
0
```

Place this file in `behavior_labels/` inside your project folder (or in the same folder as the video — PixelPaws checks both). The session will then appear automatically in the Train and Evaluate session lists.

### Tips

- **One behavior per conversion run.** Run the converter once per behavior name. If you scored multiple behaviors in the same BORIS file, run it once for each and then merge the output CSVs column-by-column before training a multi-behavior session.
- **Frame alignment.** The converter multiplies each timestamp by FPS and rounds to the nearest frame. Using the exact FPS your camera recorded at (e.g., 60.0, not 59.94) avoids drift over long recordings.
- **Dense vs. sparse labels.** BORIS exports cover the full video duration with 0s between scored bouts — this is ideal for training. If only a subset of frames is labeled, sparse and dense label files are mixed automatically during training.

---

## Crop for DLC Tool

Before running DeepLabCut you may want to spatially crop your videos so that DLC focuses on the relevant region of the frame (e.g., the behavioral arena rather than the full camera view). The Crop for DLC tool in PixelPaws automates this.

**What it does:** Encodes a new video (or a batch of videos) containing only the pixels inside a rectangular region of interest, using FFmpeg under the hood. The crop offsets are saved to the project config so that all downstream coordinate math stays consistent.

**How to use it:**

1. Open **Tools → Crop for DLC** from the menu bar.
2. Select a video (single file) or a folder (batch mode).
3. A preview frame opens. Click and drag to draw the crop rectangle, or enter pixel values directly for X offset, Y offset, Width, and Height. Default values (X=286, Y=0, W=761, H=720) are pre-filled for a common rig setup and can be changed.
4. Click **Preview** to see the cropped region on the frame.
5. In batch mode, the tool first shows a confirmation dialog listing all videos it will process. Review the list and click **Proceed** to start encoding.
6. FFmpeg quality and codec settings are configurable in the dialog.
7. The crop offsets are written to `PixelPaws_project.json` so that brightness-feature extraction later maps pixel coordinates correctly.

Cropped videos are saved alongside the originals with a `_cropped` suffix. Run DeepLabCut on the cropped videos, then place the resulting H5 files in `videos/` as usual.

---

## Tab-by-Tab Guide

### Pose Estimation Tab

Runs the bundled DeepLabCut network on your project's videos and manages installed pose models. **Analyze Videos (Pose Tracking)** runs the active model — detected keypoints are written next to each video as DLC `.h5` files, and the flow can chain into Predict and Extract Problem Frames. The **Installed pose models** panel lists bundled models with their version / scorer / release date; use **Import model (.zip)…** to add a newer network, **Set active** to choose which model is used for tracking (and as the scorer/keypoints source for Extract Problem Frames), **Delete** to remove one, and **Details** to inspect scorer / keypoints / snapshot. On first run the shipped default model is seeded automatically.

**Optional transcode step.** Checking *Transcode with the intake pipeline first* in the
Analyze Videos dialog re-encodes each selected video to H.265 (CRF 23, audio dropped,
spatial-calibration tags preserved) before tracking — the same encode our transfer-portal
pipeline applies to incoming videos. Compression at this level shrinks files roughly
200-fold at no practical cost: keypoints shift ~0.5 px and behavioral output is unchanged
(validated in the manuscript). The transcoded file keeps the video's name, so downstream
pairing is unaffected; the original is moved to `videos/raw/`. Videos already H.265 are
skipped automatically. Encoding takes roughly the video's running time, so plan for it
with long recordings.

**One-pass chaining.** The same dialog can chain the rest of the workflow: *Run behavior
predictions* opens scoring when tracking finishes, and *Extract features* opens the
Feature Extraction runner pre-selecting the newly tracked sessions (useful before
training; prediction extracts and caches features automatically, so this is not required
for scoring).

### Feature Extraction Tab

Extracts pose + brightness (+ optional optical flow) features from DLC tracking, in **Batch** (scan a project folder) or **Single Video** mode. This is the same tool previously reached from the Tools tab; it writes the cached feature files that training, prediction, and evaluation consume.

### Train Tab

The Train tab is where you build a classifier for a single behavior.

**Session discovery.** When you set a project folder, PixelPaws scans `videos/` and `behavior_labels/` to find matching triplets: video + DLC H5 + label CSV. Matched sessions appear in the session list. Check the sessions you want to include in training.

**Behavior name.** Type the exact column name from your label CSV (case-sensitive). A dropdown suggests behavior names found in the discovered label files.

**Feature settings.**
- *Pose features* — kinematic features computed from body-part coordinates: pairwise distances, joint angles, velocities at multiple timescales, and in-frame probability (confidence) scores.
- *Brightness features* — pixel brightness in square ROIs around selected body parts, extracted directly from video frames. Requires the video file.
- Choose which body parts to include. Fewer body parts = faster extraction; more = richer features.

**Bout parameters.** Set the minimum bout duration (frames) and minimum inter-bout interval to merge adjacent detections. These are applied at prediction time, not during training. Click **Auto-Suggest Bout Parameters** to analyze your label files and get recommended values based on the observed bout-length distribution.

**Scan Sessions.** Click to open a popup listing all discovered session triplets (video + H5 + label CSV) with their file paths and match status — useful for verifying PixelPaws found the files you expect.

**XGBoost hyperparameters.** All core XGBoost parameters are exposed and tunable:
- *Number of Trees* (default 1700) — maximum boosting rounds
- *Max Depth* (default 6) — maximum tree depth
- *Learning Rate* (default 0.01) — step-size shrinkage
- *Subsample* (default 0.8) — fraction of training rows sampled per tree
- *Feature Sampling* (colsample_bytree, default 0.2) — fraction of features sampled per tree

**Cross-validation folds.** Adjustable from 2 to 10 folds (default 5).

**Early stopping.** On by default. Training halts when validation F1 stops improving for a configurable number of rounds (10–200, default 50) and selects the optimal number of trees automatically.

**Class imbalance handling.** `scale_pos_weight` is on by default — it up-weights the positive (behavior-present) class proportionally so that rare behaviors are not overwhelmed by majority-class frames, without requiring downsampling.

**SHAP prune + retrain.** Optional two-pass training workflow: the first pass trains a full model and computes SHAP importance, then prunes to the top N features (configurable, default 40) and retrains on the reduced feature set. This can improve generalization and reduce overfitting on noisy features.

**Trim to last labeled event.** On by default. Removes all frames after the last positive label in each session before training. This prevents BORIS trailing zeros (unlabeled tail of the video) from flooding the training set with false negatives.

**Optical flow features.** Toggle to include optical-flow-based features alongside pose and brightness. A configurable body-part list controls which body parts flow is computed around.

**Save / Load Configuration.** Persist all training settings (hyperparameters, feature options, body-part lists, bout parameters) to a JSON file for reuse across sessions or sharing with collaborators.

**Start Training.** Click to begin. A training visualization window opens showing:
- Cross-validation F1 scores across folds
- Precision/recall curve
- Threshold optimization plot (the threshold that maximizes F1 on the validation set)
- Feature importance (SHAP values, computed on the final model)

The trained classifier is saved as a `.pkl` file to `classifiers/`. The filename encodes the behavior name, threshold, and a timestamp.

### Predict Tab

The Predict tab runs a trained classifier on a single video and reports behavior statistics for that session.

#### Inputs

| Field | Required | Notes |
|---|---|---|
| Classifier | Yes | Select a `.pkl` file from `classifiers/`. Click **View Classifier Info** to confirm the behavior name, threshold, and feature settings stored inside. |
| Video file | Yes | The original (uncropped) video. |
| DLC pose file | Yes | The `.h5` file produced by DeepLabCut for this video. Click **🔍 Auto-Find DLC File** to locate it automatically based on the video filename. |
| Features file | No | A pre-extracted `.pkl` cache from `features/`. If provided, feature extraction is skipped entirely, saving several minutes. |
| DLC config | No | The `config.yaml` from your DeepLabCut project. If provided and cropping was enabled in DLC, the crop offsets are read and applied to brightness feature extraction so pixel coordinates stay correct. Click **🔍 Auto-Find Config** to search for it automatically. |
| Human labels | No | A per-frame label CSV in PixelPaws format. Providing it records the path alongside the prediction for your own reference; for a full quantitative comparison (precision, recall, F1, SHAP) use the **Evaluate tab** instead. |
| Output folder | No | Defaults to the video's folder if left blank. |

#### What it does

1. Loads the classifier and reads its stored behavior name, threshold, body-part lists, bout-filtering parameters, and feature settings.
2. Extracts pose + brightness features if no cached file is provided (this is the slow step — expect 2–10 minutes depending on video length).
3. Runs the XGBoost model on every frame to produce a per-frame probability score.
4. Applies the trained threshold to produce binary frame labels (0 = absent, 1 = present).
5. Applies **bout filtering**: removes detections shorter than the minimum bout duration and fills gaps shorter than the maximum inter-bout interval. These parameters are stored in the classifier file, not set manually here.
6. Displays a summary in the results panel: total frames, frames with behavior detected, total behavior time in seconds and minutes.

#### Output options

| Option | Output file | Contents |
|---|---|---|
| Save frame-by-frame predictions (CSV) | `<video>_predictions.csv` | One row per frame: `frame`, `prediction` (0/1), `probability` (raw score). This file is what the **Analyze tab** consumes for batch statistics. |
| Create labeled video | `<video>_labeled.mp4` | Video with the behavior label overlaid on each frame. Slower to generate. |
| Save behavior summary statistics | `<video>_summary.txt` | Total time, bout count, mean/max bout duration, percentage of session. |
| Generate ethogram plots | `<video>_ethogram.png` | Color raster showing behavior presence across the full session timeline. |

#### Comparing against human labels

The Predict tab includes an optional **Human Labels** field where you can point to a ground-truth label CSV for the session. This path is stored with the prediction for reference but the Predict tab itself does not compute agreement metrics. For a full evaluation — confusion matrix, precision, recall, F1 at the trained threshold, precision-recall curve, and SHAP feature importance — use the **Evaluate tab**, which is designed specifically for that purpose.

### Run Classifiers Tab

The Run Classifiers tab (batch scoring) runs one or more classifiers across an entire folder of videos in a single operation.

**Inputs:**
- **Data folder** — select the folder containing your videos (and their corresponding DLC H5 files).
- **Video extension** — filter by file extension (e.g., `.mp4`, `.avi`).
- **Classifiers** — pick from `[Project]`, `[Global]`, and `[Bundled]` entries; the bundled encyclopedia is always available. Use **Auto-Detect** to find all `.pkl` files in the project, or add/remove individually.
- **Prefer filtered DLC files** — when checked, PixelPaws uses filtered DLC output (e.g., `*_filtered.h5`) over unfiltered when both exist for the same session.

**Feature status checker.** Before running, click **Check Feature Status** to see which sessions already have cached features and which will require extraction from video. This lets you estimate how long the batch will take.

**Preview mapping.** Click **Preview** to see which classifier will be applied to which video, confirming the mapping before committing to a long batch run.

**Per-classifier settings.** Each classifier in the list can have its own `min_bout_sec` and `bin_size_sec` overrides for bout filtering and time binning in the output.

**Output.** For each video × classifier combination, a prediction CSV is written to the output folder (one file per video per classifier). These CSVs are in the same format as single-session Predict output and can be loaded directly into the Analyze tab.

### Evaluate Tab

The Evaluate tab measures how well a classifier performs against hand-labeled ground truth.

1. Select the **classifier** to evaluate.
2. PixelPaws discovers available labeled test sessions automatically (same triplet logic as training). You can also point it to a specific session manually.
3. Click **Run Evaluation**.

Evaluation can run in **per-session** mode (one report per session) or **pooled** mode (aggregate all selected sessions into a single evaluation).

Results include:
- Confusion matrix
- Precision, Recall, and F1 score at the trained threshold
- Precision-recall curve
- SHAP summary plot showing which features most influence predictions
- Bout-level statistics (bout count agreement, mean bout duration comparison between predicted and ground truth)

All outputs are saved to `evaluations/` as a text report and image files.

### Quick Start tab (recommended)

The app opens on the **Quick Start** tab - the guided path from raw
videos to populated analysis tabs. Pick your videos in the session table
(length, calibration, and pose-tracking status shown; add new videos with
the ➕ button), leave the five steps ticked, and press **Run pipeline**:

1. Transcode (intake H.265; already-encoded videos skip automatically)
2. Pose tracking (DLC)
3. Feature extraction
4. Classifiers (the Core-8 default set)
5. Gait & contour analysis (manuscript paw-contact preset)

After pose tracking, **🎬 Check tracking** (on Quick Start and the Pose
Estimation tab) plays any session with the keypoints overlaid - scrub,
adjust the likelihood gate, toggle labels/trails - to confirm tracking
quality before trusting the downstream numbers.

One progress bar tracks the whole run and changes color per step; steps
that are not needed (e.g. pose for already-tracked videos) are skipped
automatically. When it finishes, every Analyze tab is populated - jump
straight to Single-Classifier or Gait & Limb with the buttons under the
bar. The individual tabs (Pose Estimation, Run Classifiers, ...) remain
available for step-by-step control.

### Choosing which sessions to analyze

Every analysis tab (Single-Classifier, Multi-Classifier, Sequencing, Locomotion,
and Gait & Limb) shares the same Data-section layout — data folder where
applicable, then the key file, then a **Sessions** button. Clicking it opens a
session table (include tick plus Subject/Group and, where known, Video/Cache/
calibration columns); click rows to include or exclude them, with All/None
shortcuts. Everything is included by default, so you only touch it to leave
sessions out of the graphs, statistics, and exports.

### Single-Classifier Analysis Tab

Rebuilt on the app's standard layout: everything auto-populates when a project opens
(prediction folders, behaviors, key file, subject-to-group matching), and the graphs live
behind a single **Graph** dropdown — Time Course, Individual Traces, Total Time, Bout
Analysis, Phase Analysis (Formalin preset), two heatmaps, two cumulative views, Mean
Timecourse (1 Hz), and Latency — with a metric selector (time, bouts, bout duration,
frequency, % time), the shared 🎨⚙ style dialog, and a **Σ Stats** flip showing group
descriptives, omnibus tests with effect sizes, Bonferroni-corrected pairwise
comparisons, and per-bin timecourse post-hocs for the current graph. Analyses run on a
worker thread with live progress and a working Stop. Exports: full results CSV with a
`.meta.json` provenance sidecar (inputs, settings, software version) and per-figure
image export.

**Setup:**
1. The **key file** is discovered automatically (CSV/XLSX with `Subject` and
   `Treatment`); Generate… creates one from the project's videos. For paired designs an
   optional `Animal` (or `Pair`/`Block`) column enables within-animal permutation in the
   Sequencing tab.
2. Behaviors and the predictions folder auto-fill from `results/`; use Advanced to
   point elsewhere.
3. Set the **time bin size** (e.g., 5 minutes) or analyze the entire session; metrics
   (total time, bout count, mean bout duration, % time, bout frequency, latency) are
   always computed.

**Graphs.** Everything renders in the right pane behind the **Graph** dropdown:
Time Course (with optional faint per-animal traces), Individual Traces, Total
Time, Bout Analysis, Phase Analysis (Formalin preset: Acute 0–10 min and
Phase II 10–60 min by default), two heatmaps (time and bouts), two cumulative
views, Mean Timecourse (1 Hz), and Latency (animals with no bouts are excluded
and noted). A Metric dropdown switches time / bouts / bout duration /
frequency / % time where applicable. Styling (colors, error bars SEM/SD/95% CI,
lines, markers, significance style) lives in the shared 🎨⚙ dialog.

**Statistics.** Enable statistics to annotate graphs and use the **Σ Stats**
flip for the current graph's tables: group descriptives, the omnibus test
(Welch/Mann-Whitney for 2 groups, ANOVA/Kruskal-Wallis for more) with effect
sizes, Bonferroni-corrected pairwise comparisons (computed when the omnibus is
significant — otherwise the view says so), and two-way ANOVA with per-bin
Bonferroni post-hocs for timecourse views.

**Exports.** Export CSV writes the full results table with a `.meta.json`
provenance sidecar; Export figure saves the current graph (PNG/PDF/SVG).

### Locomotion Tab

Distance traveled and velocity straight from the pose skeleton — no classifiers needed.
The animal's trunk-centroid trajectory is likelihood-gated, median-smoothed, and
integrated with a jitter dead-band. Seven views behind the Graph dropdown: distance per
bin, cumulative distance, and mean velocity (mean ± error lines per group, any number of
groups, with an annotated group test), plus four normalized-arena views — per-animal
trails, group overlays, a representative animal per group, and an occupancy heatmap with
its own colormap dropdown. **Preview video…** (under Tracking settings) plays any session
with the tracked trajectory overlaid so you can sanity-check what the numbers integrate.
Units are real centimeters whenever every video carries the PawCapture spatial
calibration (the `mm_per_pixel` tag, preserved through the intake transcode and shown in
the pose-tracking dialog's Calibration column); uncalibrated projects fall back to pixels
with a note. Binned tables export as CSV.

### Gait & Limb Use Tab

The Gait & Limb Use tab analyzes paw contact patterns, gait timing, and limb symmetry from DLC pose data — no force plate or pressure mat needed.

Rebuilt on the app's standard layout: a left rail that reads top-to-bottom —
**Data** (key file first, then the Sessions picker with Rescan/Browse…) →
**Quick Setup** (a preset with ▶ Run Analysis and Cancel right beside it, and a
readiness strip that always names the next actionable step) → Setup → Detection
→ collapsed Advanced → **Results & Export** (CSV exports, Adjust Contact, saved
sessions, log). Results render directly in the right pane — pick a **Category**
(Paw Contact, Limb Use, Contact %/Brightness, Gait Timing/Spatial/Symmetry,
Movement, Coordination, Paw Contour, Statistics) and then a **Graph** from the
dropdowns; each graph offers CSV/PNG export, the shared 🎨⚙ style dialog, a
**Display…** dialog for gait-specific options (treatment order, per-treatment
markers, timecourse window/re-bin, Full-Stance contour categories), and a
**Σ Stats** flip to that metric's statistics. A collapsed **Session table**
strip under the graph holds the per-session results table. Analyses run on a
worker thread with live progress and Cancel; results auto-save as session
bundles you can reload from Results & Export. The compute engine and the
**Adjust Contact** re-analysis share one metrics implementation, so
adjusted-contact results always apply the same licking-exclusion and 4-paw
gating as the original run.

**Paw contact detection.** Four methods are available:
- *Contour area* (default) — a paw is "in contact" when its Otsu-segmented contour falls
  within a plausible paw-sized area band (1,500–5,000 px² by default). This is the
  validated gate from our manuscript and ships as the "Paw contact (manuscript gate)"
  preset, which also excludes licking frames — score licking first (Run Classifiers),
  then run the gait analysis
- *Height* — paw is "in contact" when its vertical coordinate falls below a threshold
- *Speed* — paw is "in contact" when its speed drops below a threshold (Kumar Lab, Cell Reports 2022)
- *Combined* — both height AND speed criteria must agree

**Injured / injected paw.** Set which hind paw carries the injury or injection in the
Paw Mapping panel (default HL). Every ratio graph is then shown as injured/contralateral,
so values below 1.0 always mean the injured paw bears less — no need to mentally invert
when the manipulated side is HR. The results pane opens on the **Paw Contact** category with
these headline ratios; per-paw breakdowns live under Paw Contour.

**Metrics computed per session:**
- **Contact percentages** — % frames each paw is in stance (ground contact) for hind-left, hind-right, fore-left, fore-right
- **Weight-bearing index (WBI)** — HL / (HL+HR) × 100 (50 = symmetric); also computed for fore paws
- **Symmetry index (SI)** — (HL−HR) / (HL+HR) × 100 (0 = symmetric)
- **Brightness during contact** — mean ROI brightness during stance frames (optional, requires video)
- **Gait timing** — stance duration, swing duration, stride duration, duty cycle, cadence, and stride count per paw
- **Gait spatial** — stride length per paw, step length and step width for hind/fore pairs
- **Interlimb coordination** — phase coupling between left-right hind paws and diagonal pairs
- **Gait symmetry** — stance symmetry index, stride length symmetry index
- **Paw contour analysis** — paw area, spread, contact intensity, width, solidity, aspect ratio, circularity (optional, requires video)

**Time binning.** All metrics can be computed in user-defined time bins (e.g., 5-minute windows) for tracking changes over a session.

**Batch processing.** All discovered sessions are included by default; untick any in the Sessions picker to exclude them, then Run. Extraction caches and session bundles live in `gait_limb_analysis/`; use Export Summary / Export Bins for CSV output.

### Multi-Classifier Analysis Tab

Cross-classifier views of a scored cohort, fed by the consolidated per-frame sheets that
Run Classifiers writes to `results/per_frame/`. Four views, each scaling to any number
of groups:

- **Probability traces** — one panel per behavior for a chosen session: the classifier's
  frame-by-frame probability with predicted bouts shaded (the format of the manuscript's
  supplementary probability plots)
- **Probability lines (all behaviors, per group)** — every selected behavior's
  probability overlaid on one panel per group over an adjustable rolling window
  (the supplement's all-behavior trace format)
- **State occupancy** — every frame resolved to a single state by the priority order
  (unscored = no classifier fired), shown as % of session time per state, mean and SEM
  per group with per-animal points
- **Group timecourse** — a panel grid, one panel per behavior: % time in behavior per
  time bin, mean and SEM lines, one line per group; bin width is adjustable and clamps
  automatically for short sessions

The tab auto-populates after a classifier run and on project open; the key file is
discovered from the project. Use Multi-Classifier Analysis for *how much and when*;
use Sequencing for *what follows what*.

### Sequencing Tab

Bout-level behavioral syntax — the order behaviors are strung together, independent of how
much of each occurs. It has its own pipeline: point it at a folder of prediction CSVs from
Run Classifiers, order the priority list, load a key file, and Compute. Views:

- **Group networks** — one pooled transition network per group (edge width = traffic,
  color = deviation from what that group's own behavioral rates predict)
- **Difference vs reference** — every route that strengthened or weakened past a threshold
- **Ordination (PCoA)** — each animal placed by how differently it sequences behavior,
  with optional group centroids, 95% ellipses or convex hulls, and a PERMANOVA test
  (exact p by permutation; within-animal permutation when the key file has an `Animal`
  column)

Sequencing reads the *filtered* binary calls from the prediction CSVs, so every classifier
keeps its own validated threshold and bout filter — no re-thresholding.

### Tools Tab

The Tools tab provides quick access to a set of utilities that complement the main pipeline:

| Tool | Description |
|---|---|
| **Video Preview** | Play a video alongside its prediction CSV — predictions are overlaid on each frame so you can visually verify classifier output. |
| **Auto-Label Assistant** | Steps through frames at a configurable interval and prompts you to label each one; outputs a label CSV in the PixelPaws format. |
| **Data Quality Checker** | Scans all label CSVs in the project for common issues: class imbalance, missing frames, duplicate rows, and sessions with very few positive examples. |
| **Brightness Diagnostics** | Plots mean brightness for each body-part ROI over time for a selected video; useful for detecting lighting artefacts or ROI misalignment. |
| **Feature File Inspector** | Opens a cached feature `.pkl` and shows column names, shapes, and summary statistics — helpful for debugging feature extraction. |
| **Brightness Preview** | Shows a single video frame with the brightness ROI rectangles drawn around each selected body part so you can confirm they are positioned correctly. |
| **Correct Crop Offset (Single / Batch)** | If videos were cropped before DLC and the crop offsets changed between sessions, these tools remap the stored offsets in prediction CSVs so that coordinates stay consistent. |
| **Crop Video for DLC** | Spatially crops a video (or batch of videos) to a user-defined rectangle and saves the result for DLC analysis. Crop offsets are written to the project config. See [Crop for DLC Tool](#crop-for-dlc-tool) above. |
| **Generate Ethogram** | Creates an ethogram image (color raster of behavior presence over time) from any prediction CSV; can be saved as PNG. |
| **Training Visualization** | Re-opens the training visualization window for the most recently trained classifier (cross-validation scores, precision-recall curve, SHAP summary). |
| **BORIS to PixelPaws** | Converts a BORIS event-log export (CSV) into the frame-indexed label CSV format that PixelPaws expects, using the video frame rate to map timestamps to frame numbers. |
| **Optimize Parameters** | Grid-searches bout-filtering parameters (minimum bout duration, minimum inter-bout interval) to maximize agreement with hand labels on a selected session. |
| **Feature Extraction** | Runs feature extraction manually on a selected video + H5 pair and saves the result to `features/`; useful for pre-caching before a training run. |
| **Skeleton Video Renderer** | Render skeleton overlays on video with customizable colors, glow effects, paw trails, and bout-clipping. Multiple colorway presets available. |
| **Theme Switcher** | Toggles between light and dark UI themes. |

---

## Project Folder Layout

```
my_project/
├── videos/            # Videos (.mp4, .avi, etc.) + DLC .h5 files
├── behavior_labels/   # Label CSVs (frame × behavior columns)
├── classifiers/       # Trained .pkl classifiers
├── features/          # Cached feature files (safe to delete)
├── results/           # Prediction outputs (per-video CSVs, videos, ethograms)
├── analysis/          # Batch analysis outputs
├── evaluations/       # Evaluation reports + SHAP plots
├── gait_limb_analysis/ # Gait & Limb Use outputs
└── PixelPaws_project.json
```

---

## Requirements

| Package | Version | Notes |
|---|---|---|
| numpy | ≥ 2.0 | |
| pandas | ≥ 2.0 | |
| xgboost | ≥ 3.0 | Classifier backend |
| scikit-learn | ≥ 1.3 | Cross-validation, metrics |
| shap | ≥ 0.43 | Feature importance plots |
| statsmodels | ≥ 0.14 | Two-way ANOVA in Analyze tab |
| scipy | ≥ 1.10 | Post-hoc tests |
| opencv-python | ≥ 4.8 | Video frame extraction |
| h5py / tables | ≥ 3.8 | Reading DLC HDF5 output |
| matplotlib | ≥ 3.7 | Graphs |
| seaborn | ≥ 0.12 | Heatmaps |
| PyYAML | ≥ 6.0 | Reading DLC config.yaml |
| openpyxl | ≥ 3.1 | XLSX key file support in Analyze tab |

---

## Attribution & License

Feature extraction is based on the BAREfoot algorithm:

> Barkai O, Zhang B, et al. *BAREfoot: Behavior with Automatic Recognition and Evaluation.* Cell Reports Methods, 2025. https://github.com/OmerBarkai/BAREfoot

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — free for academic and non-commercial use.

© 2026 rslivicki
