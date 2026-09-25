# Changelog

## 0.9.0 (2026-09-25)

First public release, with the bioRxiv preprint.

**PixelPaws (analysis)**

- Quick Start pipeline: transcode, DeepLabCut pose tracking with the bundled network, feature extraction, the eight bundled classifiers (the Core 8) and paw contour analysis, in one run.
- Bundled pose network: ResNet-50, nine keypoints, trained on 80 labeled sessions of box video (shuffle 4, snapshot-best-200). The same network scored every dataset in the paper.
- Every frame gets at most one behavior, resolved in priority order: licking, scratching, jumping, rearing, body grooming, facial grooming, moving, still.
- Review scoring plays a scored video with the calls, a bout timeline and optional keypoints. Check tracking plays the keypoints alone.
- Group analysis tabs: Single-Classifier, Multi-Classifier, Sequencing, Locomotion and Paw Contour. Paw Contour leaves out licking frames automatically.
- Train Classifier and Evaluate tabs for your own behaviors, with BORIS import.
- An example project (six 60-second formalin clips with a key file) installs with the app.
- Windows installer zip. Help > Check for Updates looks for newer releases here.

**PawCapture 0.8.0 (recording)**

- Records several See3CAM_CU27 cameras at once (tested up to four), with per-camera crop, camera settings and a two-click mm/pixel calibration stored in each file.
- If a camera's FFmpeg crashes mid-session, the recording continues into `<name>_pt2.mp4`.

**Tested on** Windows 11 with an NVIDIA RTX 3080 Ti and, on a second computer, an RTX 5070 Ti: a fresh environment build, an update over an existing install, and the full pipeline on the example project. **Not yet tested:** a machine with no conda at all, CPU-only, and a non-admin account. PawCapture 0.8.0 has not yet been run on cameras.
