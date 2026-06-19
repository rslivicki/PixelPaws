"""
extract_snlt_baselines_silhouette.py
====================================
One-off: extract silhouette features for ONLY the *_Baseline.mp4
sessions in the 2603_SNLT_JG/Baseline project. Does NOT modify the
project JSON on disk; the silhouette flags are added to the cfg dict
in memory only, so the non-baseline videos in that folder aren't
affected and will keep their existing caches.

Run from any cwd:
    python extract_snlt_baselines_silhouette.py
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

PROJECT = Path(r"E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline")
SUFFIX  = "_Baseline"           # only mp4 files ending in this stem


def atomic_pickle(obj, target_path: str) -> None:
    d = os.path.dirname(target_path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f)
        os.replace(tmp, target_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def discover_baseline_sessions(videos_dir: Path) -> list[dict]:
    """Walk videos_dir for *_Baseline.mp4 + matching DLC h5. Raw .h5
    preferred over _filtered.h5 when both exist."""
    by_stem: dict[str, tuple[str, str]] = {}
    for v in sorted(videos_dir.glob(f"*{SUFFIX}.mp4")):
        stem = v.stem
        # Prefer raw h5 over _filtered
        h5s = sorted(videos_dir.glob(f"{stem}DLC*.h5"))
        h5_raw = [h for h in h5s if "_filtered" not in h.name]
        chosen = (h5_raw or h5s)[0] if h5s else None
        if chosen is None:
            continue
        by_stem[stem] = (str(chosen), str(v))
    return [
        {"session_name": s, "pose_path": h, "video_path": v}
        for s, (h, v) in sorted(by_stem.items())
    ]


def main() -> int:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _HERE)                                # peer scripts in this folder
    sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # project root (E:/PixelPaws)
    from prediction_pipeline import PixelPaws_ExtractFeatures
    from feature_cache import FeatureCacheManager

    cfg_path = PROJECT / "PixelPaws_project.json"
    with open(cfg_path) as f:
        cfg = json.load(f)

    # In-memory opt-in: do NOT save back to disk.
    cfg["compute_silhouette"] = True
    cfg["silhouette_floor"]   = 35

    cfg_hash = FeatureCacheManager.compute_hash(cfg)
    cache_root = str(PROJECT / "features")
    os.makedirs(cache_root, exist_ok=True)

    videos_dir = PROJECT / "videos"
    sessions = discover_baseline_sessions(videos_dir)

    print(f"Project   : {PROJECT}")
    print(f"cfg_hash  : {cfg_hash}  (silhouette enabled in memory)")
    print(f"cache_root: {cache_root}")
    print(f"sessions  : {len(sessions)} baseline sessions discovered\n")

    if not sessions:
        print("No baseline sessions found.")
        return 1

    extracted = skipped = errors = 0
    t0 = time.time()
    for i, s in enumerate(sessions, 1):
        name = s["session_name"]
        cache_file = os.path.join(cache_root, f"{name}_features_{cfg_hash}.pkl")
        elapsed = (time.time() - t0) / 60.0
        print(f"[{i}/{len(sessions)}] {name}   (elapsed {elapsed:.1f} min)")
        if os.path.isfile(cache_file):
            print("  - already cached with this hash, skipping")
            skipped += 1
            continue
        try:
            X = PixelPaws_ExtractFeatures(
                pose_data_file=s["pose_path"],
                video_file_path=s["video_path"],
                bp_include_list=cfg.get("bp_include_list"),
                bp_pixbrt_list=cfg["bp_pixbrt_list"],
                square_size=cfg["square_size"],
                pix_threshold=cfg["pix_threshold"],
                config_yaml_path=cfg.get("dlc_config") or None,
                include_optical_flow=bool(cfg.get("include_optical_flow", False)),
                bp_optflow_list=cfg.get("bp_optflow_list") or None,
                compute_silhouette=True,
                silhouette_floor=35,
            ).reset_index(drop=True)
            atomic_pickle(X, cache_file)
            print(f"  ok {X.shape[0]} frames x {X.shape[1]} features")
            print(f"     -> {cache_file}")
            extracted += 1
        except Exception as e:
            import traceback
            print(f"  FAIL: {e}")
            traceback.print_exc()
            errors += 1

    print(f"\nDone in {(time.time() - t0)/60:.1f} min. "
          f"{extracted} extracted, {skipped} skipped, {errors} errors.")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
