"""Scorer/project-parameterized helpers for folding problem-frame candidate folders
into a DLC project and prepping a training-dataset — generalizes the per-cohort
merge_iter*/prep_iter* scripts. Pure pandas/ruamel (Python310-safe, no DLC import);
the actual create_training_dataset/train_network run under the DEEPLABCUT env in
dlc_retrain.py, which imports these helpers.
"""
from __future__ import annotations
import os, glob, shutil
from pathlib import Path
import pandas as pd


def norm_index(i) -> str:
    """DLC CollectedData index entry -> canonical backslash path string (handles both the
    flat single-level string index and the 3-level tuple index)."""
    s = os.path.join(*i) if isinstance(i, tuple) else str(i)
    return s.replace("/", "\\")


def frames_in_folder(ldir: str, scorer: str, folder: str) -> set[str]:
    """Set of normalized frame index strings in one labeled-data folder's CollectedData."""
    h5 = os.path.join(ldir, folder, f"CollectedData_{scorer}.h5")
    if not os.path.isfile(h5):
        return set()
    return {norm_index(i) for i in pd.read_hdf(h5).index}


def add_videos_to_config(config_path: str, videos_with_crop: dict, new_iteration: int,
                         backup=True, log=print):
    """ruamel-edit a DLC config.yaml: ensure each {video_path: 'crop'} is in video_sets and
    bump `iteration`. Backs up to <config>.bak-pre-iter<N>. Returns (n_video_sets, backup_path)."""
    from ruamel.yaml import YAML
    bak = f"{config_path}.bak-pre-iter{new_iteration}"
    if backup and not os.path.isfile(bak):
        shutil.copyfile(config_path, bak); log(f"backed up -> {bak}")
    y = YAML()
    with open(config_path) as f:
        cfg = y.load(f)
    vs = cfg.setdefault("video_sets", {})
    for vp, crop in videos_with_crop.items():
        if vp not in vs:
            vs[vp] = {"crop": crop}
            log(f"added video_set: {vp}")
    cfg["iteration"] = int(new_iteration)
    with open(config_path, "w") as f:
        y.dump(cfg, f)
    log(f"iteration -> {new_iteration}; video_sets = {len(vs)}")
    return len(vs), bak


def verify_merged_dataset(project_dir: str, scorer: str, iteration: int,
                          expected: set[str]) -> tuple[int, list[str]]:
    """After create_training_dataset, confirm every `expected` frame landed in the merged
    training-dataset CollectedData. Returns (total_frames, missing_list)."""
    td = glob.glob(os.path.join(project_dir, "training-datasets", f"iteration-{iteration}",
                                "UnaugmentedDataSet*", f"CollectedData_{scorer}.h5"))
    if not td:
        return 0, sorted(expected)
    merged = pd.read_hdf(td[0])
    have = {norm_index(i) for i in merged.index}
    missing = sorted(e for e in expected if e not in have)
    return len(merged), missing


def restore_config(config_path: str, backup_path: str, log=print):
    if backup_path and os.path.isfile(backup_path):
        shutil.copyfile(backup_path, config_path)
        log(f"restored config from {backup_path}")
