"""DLC retraining orchestrator — RUN UNDER THE DEEPLABCUT ENV (the GUI shells out to
this because the GUI's Python has no deeplabcut). Adds a new iteration from freshly
labeled problem-frame folders and warm-starts training from the current best snapshot.

Steps: add new candidate sessions to config video_sets + bump iteration ->
create_training_dataset -> verify every labeled frame landed (restore config + abort on
miss) -> train_network(snapshot_path=<best>, warm-start). DLC writes learning_stats.csv
in the train dir (step, metrics/test.rmse, .rmse_pcutoff, .mAP, .mAR); the GUI tails that
for the live-progress panel — this script also prints TRAINDIR=<path> so the GUI can find it.

Usage:
  dlc_retrain.py --config C.yaml --scorer S --project-dir P --iteration N
                 --snapshot best.pt --epochs 200 --batch 24 --save-epochs 25
                 [--sessions sessions.json] [--net resnet_50]
where sessions.json = {"<session>": {"video": "<path-or-nominal>", "crop": "0, W, 0, H"}, ...}
(new candidate folders to fold in; their labeled-data/<session>/CollectedData_<scorer>.h5
must already exist). Omit --sessions to just (re)build + train the current labeled set.
"""
from __future__ import annotations
import os, sys, json, argparse, glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))     # label_merge beside us
import label_merge as lm


def _find_traindir(project_dir, iteration):
    cands = glob.glob(os.path.join(project_dir, "dlc-models-pytorch", f"iteration-{iteration}",
                                   "*trainset*shuffle1", "train"))
    return cands[0] if cands else None


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scorer", required=True)
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--iteration", type=int, required=True)
    ap.add_argument("--snapshot", default="", help="warm-start .pt (empty = from scratch/ImageNet)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--save-epochs", type=int, default=25)
    ap.add_argument("--net", default="resnet_50")
    ap.add_argument("--sessions", default="", help="sessions.json of new candidate folders")
    ap.add_argument("--ldir", default="", help="labeled-data dir (default <project>/labeled-data)")
    a = ap.parse_args(argv)

    ldir = a.ldir or os.path.join(a.project_dir, "labeled-data")
    sessions = json.load(open(a.sessions)) if a.sessions and os.path.isfile(a.sessions) else {}

    # 1) config: add new candidate sessions to video_sets + bump iteration
    expected = set()
    vids = {}
    for name, meta in sessions.items():
        # create_training_dataset matches a labeled-data folder ONLY by a video_sets
        # entry's basename STEM. The candidate folder is named `name`, so the video_sets
        # key stem must equal `name`. Use the real video path only when its stem already
        # matches; otherwise a nominal <project>/videos/<name>.mp4 (never decoded here).
        real = meta.get("video")
        if real and os.path.splitext(os.path.basename(real))[0] == name:
            vp = real
        else:
            vp = os.path.join(a.project_dir, "videos", name + ".mp4")
        vids[vp] = meta.get("crop", "0, 1280, 0, 720")
        expected |= lm.frames_in_folder(ldir, a.scorer, name)
    backup = None
    if vids:
        _, backup = lm.add_videos_to_config(a.config, vids, a.iteration)
    else:
        # still ensure iteration is set
        lm.add_videos_to_config(a.config, {}, a.iteration)

    # 2) build the training dataset (DLC env)
    print("=== create_training_dataset ===", flush=True)
    import deeplabcut
    deeplabcut.create_training_dataset(a.config, Shuffles=[1], net_type=a.net, userfeedback=False)

    # 3) verify the new frames landed
    if expected:
        total, missing = lm.verify_merged_dataset(a.project_dir, a.scorer, a.iteration, expected)
        print(f"merged training set: {total} frames; new frames present "
              f"{len(expected) - len(missing)}/{len(expected)}", flush=True)
        if missing:
            print(f"!! MISSING {len(missing)} new frames — restoring config, aborting", flush=True)
            for m in missing[:15]:
                print("   ", m, flush=True)
            if backup:
                lm.restore_config(a.config, backup)
            return 2

    traindir = _find_traindir(a.project_dir, a.iteration)
    if traindir:
        print(f"TRAINDIR={traindir}", flush=True)   # GUI watches <traindir>/learning_stats.csv

    # 4) warm-start train
    print("=== train_network ===", flush=True)
    kw = dict(shuffle=1, trainingsetindex=0, batch_size=a.batch, epochs=a.epochs,
              save_epochs=a.save_epochs, device="cuda:0")
    if a.snapshot and os.path.isfile(a.snapshot):
        kw["snapshot_path"] = a.snapshot
        print(f"warm-start from {a.snapshot}", flush=True)
    else:
        print("training from scratch (no warm-start snapshot)", flush=True)
    deeplabcut.train_network(a.config, **kw)
    print("TRAIN DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
