"""
watch_dlc_extract.py — Watch a PixelPaws project's videos/ folder and run
feature extraction as DeepLabCut outputs land.

Two modes:

  default       — Wait for every video to have a DLC .h5 sibling, then run
                  extraction once and exit. Good for finite batches you've
                  already kicked off DLC on.
  --continuous  — Extract whatever video+DLC pairs are ready that don't yet
                  have a feature cache, sleep for --poll seconds, repeat.
                  Never exits. Good for long-running rigs where you'll keep
                  adding videos to the folder periodically.

Usage:
    python watch_dlc_extract.py [project_folder] [--poll SECONDS] [--continuous]

Defaults to E:\\RSVIDS\\Blackbox\\2603_SNLT_JG\\Baseline if no folder given.

A pair is "ready" when the video has at least one matching "<stem>DLC*.h5"
sibling. Filter outputs (_filtered.h5) are not required. The extractor
skips any session whose feature cache pkl is already present, so re-running
the script (or running it in --continuous mode) never duplicates work.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_PROJECT = r"E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline"
DEFAULT_POLL_SECONDS = 60
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def load_project(project_folder: Path) -> dict:
    cfg_path = project_folder / "PixelPaws_project.json"
    if not cfg_path.is_file():
        sys.exit(f"No PixelPaws_project.json found at {cfg_path}")
    with open(cfg_path) as f:
        return json.load(f)


def videos_with_status(videos_dir: Path, video_ext: str):
    """Return (all_videos, pending) where pending lacks a matching DLC .h5."""
    videos = sorted(videos_dir.glob(f"*{video_ext}"))
    pending = []
    for v in videos:
        # f"{stem}DLC*.h5" requires DLC to follow the stem directly, so e.g.
        # "9417.mp4" won't accidentally match "9417_BaselineDLC*.h5".
        if not any(videos_dir.glob(f"{v.stem}DLC*.h5")):
            pending.append(v)
    return videos, pending


def discover_sessions(videos_dir: Path) -> list[dict]:
    """Same logic as PixelPaws_GUI batch scan, deduped by stem; raw .h5 preferred."""
    by_stem: dict[str, tuple[str, str]] = {}
    h5_files = sorted(videos_dir.glob("*.h5"))
    # Pass 1: raw .h5 only. Pass 2: _filtered.h5 as fallback.
    for filtered_pass in (False, True):
        for h5 in h5_files:
            is_filtered = h5.stem.endswith("_filtered")
            if is_filtered != filtered_pass:
                continue
            base = h5.stem
            stem = base.split("DLC", 1)[0] if "DLC" in base else base
            if stem in by_stem:
                continue
            for ext in VIDEO_EXTS:
                vid = videos_dir / f"{stem}{ext}"
                if vid.is_file():
                    by_stem[stem] = (str(h5), str(vid))
                    break
    return [
        {"session_name": s, "pose_path": h, "video_path": v}
        for s, (h, v) in sorted(by_stem.items())
    ]


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


def run_extraction(sessions: list[dict], cache_root: str, cfg: dict) -> int:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _HERE)                                # peer scripts in this folder
    sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # project root (E:/PixelPaws)
    from prediction_pipeline import PixelPaws_ExtractFeatures
    from feature_cache import FeatureCacheManager

    cfg_hash = FeatureCacheManager.compute_hash(cfg)
    os.makedirs(cache_root, exist_ok=True)
    print(f"\n[FE] cfg_hash={cfg_hash}")
    print(f"[FE] cache_root={cache_root}\n")

    extracted = skipped = errors = 0
    for i, s in enumerate(sessions, 1):
        name = s["session_name"]
        cache_file = os.path.join(cache_root, f"{name}_features_{cfg_hash}.pkl")
        print(f"[{i}/{len(sessions)}] {name}")
        if os.path.isfile(cache_file):
            print("  - already cached, skipping")
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
                # Per-project opt-in: add `"compute_silhouette": true` (and
                # optionally `"silhouette_floor": <int>`) to
                # PixelPaws_project.json. Defaults off — won't change cache
                # hashes for existing projects.
                compute_silhouette=bool(cfg.get("compute_silhouette", False)),
                silhouette_floor=int(cfg.get("silhouette_floor", 35)),
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

    print(f"\nDone. {extracted} extracted, {skipped} skipped, {errors} errors.")
    return 0 if errors == 0 else 1


def _ready_uncached_sessions(videos_dir: Path, cache_root: str,
                              cfg: dict) -> list[dict]:
    """Return sessions whose video+DLC pair exists AND whose feature
    cache file does not. The hash is computed once; the caller can
    reuse it for the actual extraction.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from feature_cache import FeatureCacheManager

    cfg_hash = FeatureCacheManager.compute_hash(cfg)
    sessions = discover_sessions(videos_dir)
    ready = []
    for s in sessions:
        cache_file = os.path.join(
            cache_root, f"{s['session_name']}_features_{cfg_hash}.pkl")
        if not os.path.isfile(cache_file):
            ready.append(s)
    return ready


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("project_folder", nargs="?", default=DEFAULT_PROJECT)
    ap.add_argument("--poll", type=int, default=DEFAULT_POLL_SECONDS,
                    help=f"Poll interval in seconds (default {DEFAULT_POLL_SECONDS})")
    ap.add_argument("--continuous", action="store_true",
                    help="Don't wait for the full batch — extract each "
                         "video+DLC pair as it becomes ready, then keep "
                         "polling forever. Skips already-cached sessions.")
    args = ap.parse_args()

    project = Path(args.project_folder)
    if not project.is_dir():
        sys.exit(f"Project folder does not exist: {project}")
    cfg = load_project(project)
    videos_dir = project / "videos"
    if not videos_dir.is_dir():
        sys.exit(f"No videos/ subfolder at {videos_dir}")
    cache_root = str(project / "features")
    video_ext = cfg.get("video_ext", ".mp4")

    mode = "continuous" if args.continuous else "batch (exit when complete)"
    print(f"Watching {videos_dir}")
    print(f"  mode={mode}  video_ext={video_ext}  poll={args.poll}s")
    print(f"  cache_root={cache_root}")

    while True:
        videos, pending = videos_with_status(videos_dir, video_ext)

        if args.continuous:
            ready = _ready_uncached_sessions(videos_dir, cache_root, cfg)
            done = len(videos) - len(pending)
            if not videos:
                print(f"  no {video_ext} files yet")
            elif ready:
                print(f"\n[{time.strftime('%H:%M:%S')}] "
                      f"{len(ready)} pair(s) ready to extract; "
                      f"{done}/{len(videos)} have DLC; "
                      f"{len(pending)} still waiting on DLC")
                # run_extraction skips already-cached sessions internally,
                # but `ready` is already filtered so nothing is wasted.
                run_extraction(ready, cache_root, cfg)
                print(f"[{time.strftime('%H:%M:%S')}] "
                      f"extraction pass complete; sleeping {args.poll}s")
            else:
                if pending:
                    print(f"  {done}/{len(videos)} have DLC; "
                          f"all caught up; "
                          f"waiting on {len(pending)} for DLC: "
                          + ", ".join(v.name for v in pending[:3])
                          + (" ..." if len(pending) > 3 else ""))
                else:
                    print(f"  all {len(videos)} extracted; idle "
                          f"(continuous mode — Ctrl-C to stop)")
        else:
            # Original batch behaviour: wait for all DLC, extract once, exit.
            if not videos:
                print(f"  no {video_ext} files yet, sleeping...")
            elif pending:
                done = len(videos) - len(pending)
                print(f"  {done}/{len(videos)} have DLC .h5; "
                      f"waiting on {len(pending)}:")
                for v in pending[:5]:
                    print(f"    - {v.name}")
                if len(pending) > 5:
                    print(f"    (+ {len(pending) - 5} more)")
            else:
                print(f"\nAll {len(videos)} videos have DLC .h5 "
                      f"-> running feature extraction")
                sessions = discover_sessions(videos_dir)
                if not sessions:
                    print("No video+DLC session pairs discovered; aborting.")
                    return 1
                print(f"Discovered {len(sessions)} session(s).")
                return run_extraction(sessions, cache_root, cfg)

        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
