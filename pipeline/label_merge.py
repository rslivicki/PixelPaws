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


def merge_suffixed_into_canonical(ldir: str, scorer: str, canonical: str,
                                  suffixes=("_extra", "_ft50"), remove_temp=True, log=print) -> int:
    """Fold <canonical><suffix> candidate folders into <canonical>: copy imgs, rewrite the
    index to the canonical path, concat + dedup by frame (keep first), back up, write.
    Returns net-new frame count. (Generalizes merge_iter4_thc30.py; scorer-parameterized.)
    Only needed when appending frames to a session that already has a canonical folder —
    brand-new sessions are written directly as <canonical> and need no merge."""
    cpath = os.path.join(ldir, canonical)
    ch5 = os.path.join(cpath, f"CollectedData_{scorer}.h5")
    parts = []
    before = 0
    if os.path.isfile(ch5):
        cd0 = pd.read_hdf(ch5); parts.append(cd0); before = len(cd0)
    added = 0
    for suf in suffixes:
        sp = os.path.join(ldir, canonical + suf)
        sh5 = os.path.join(sp, f"CollectedData_{scorer}.h5")
        if not os.path.isfile(sh5):
            continue
        os.makedirs(cpath, exist_ok=True)
        cd = pd.read_hdf(sh5)
        for img in glob.glob(os.path.join(sp, "img*.png")):
            shutil.copyfile(img, os.path.join(cpath, os.path.basename(img)))
        cd.index = [norm_index(i).replace(canonical + suf, canonical) for i in cd.index]
        parts.append(cd); added += len(cd)
    if added == 0:
        return 0
    merged = pd.concat(parts)
    merged = merged[~merged.index.duplicated(keep="first")]
    if os.path.isfile(ch5):
        bak = ch5 + ".bak-premerge"
        if not os.path.isfile(bak):
            shutil.copyfile(ch5, bak)
    merged.to_hdf(ch5, key="df_with_missing", mode="w")
    merged.to_csv(os.path.join(cpath, f"CollectedData_{scorer}.csv"))
    net = len(merged) - before
    log(f"  {canonical}: {before} -> {len(merged)} frames (+{net})")
    if remove_temp:
        for suf in suffixes:
            sp = os.path.join(ldir, canonical + suf)
            if os.path.isdir(sp):
                shutil.rmtree(sp, ignore_errors=True)
    return net


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
