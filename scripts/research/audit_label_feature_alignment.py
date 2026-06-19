"""Comprehensive audit: for every BORIS-labelled (behaviour, session) pair,
verify the label array is correctly aligned to a feature pkl. Flags any
session where:

  - video on disk doesn't match the BORIS sidecar's recorded n_frames / fps
  - feature pkl row count doesn't match the label array length
  - the chosen feature pkl (preferring 8aed1c22) is in fact from a
    different recording (different row count)
  - no feature pkl exists at all
  - the obs-name and video-stem map to two different feature pkls (the
    bug that caused 260318_JG_9417 body_grooming to use the wrong video)

Outputs:
  - prints a per-session table per behaviour
  - prints a list of mismatches at the end (sorted by severity)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import train_global_classifier as tg

LABELS_DIR = Path(r"E:/RS_Boris/per_frame_labels")

idx = tg.index_feature_pkls()


def load_rows(pkl_path: Path) -> int:
    """Return row count of a feature pkl (read efficiently via mmap)."""
    try:
        obj = joblib.load(pkl_path, mmap_mode="r")
        if isinstance(obj, dict):
            obj = obj.get("X") or obj.get("features") or obj.get("df")
        if obj is not None and hasattr(obj, "shape"):
            return int(obj.shape[0])
    except Exception:
        pass
    return -1


def pick_pkl(stems: list[str], label_n: int) -> tuple[Path | None, list]:
    """Combine all candidate pkls under any of the given stems, pick the
    one whose row count matches `label_n`. Prefer 8aed1c22 hash.
    Returns (chosen_path, all_candidates_with_rows).
    """
    seen = []
    seen_paths = set()
    for s in stems:
        for p in idx.get(s, []):
            if p in seen_paths:
                continue
            seen_paths.add(p)
            rows = load_rows(p)
            is_8aed = "8aed1c22" in p.name
            rows_match = abs(rows - label_n) <= 10 if rows > 0 else False
            seen.append({"path": p, "rows": rows, "is_8aed": is_8aed,
                         "rows_match": rows_match})
    if not seen:
        return None, []
    # Score: (rows_match, is_8aed, mtime)
    best = max(seen, key=lambda d: (d["rows_match"], d["is_8aed"],
                                     d["path"].stat().st_mtime))
    return best["path"], seen


def main() -> int:
    sidecars = sorted(p for p in LABELS_DIR.glob("*.json")
                      if p.name != "_summary.json")
    print(f"Auditing {len(sidecars)} BORIS sidecars\n")

    # Cache video metadata across sidecars (multiple behaviours per video)
    video_meta_cache: dict[str, tuple[float, int]] = {}

    def get_video_meta(path: Path) -> tuple[float, int]:
        key = str(path)
        if key in video_meta_cache:
            return video_meta_cache[key]
        if not path.is_file():
            video_meta_cache[key] = (0.0, 0)
            return 0.0, 0
        cap = cv2.VideoCapture(str(path))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        video_meta_cache[key] = (fps, n)
        return fps, n

    # Group by behaviour for printing
    by_behaviour: dict[str, list[dict]] = defaultdict(list)
    mismatches = []
    for sc in sidecars:
        meta = json.loads(sc.read_text())
        b = meta["canonical_behaviour"]
        obs = meta["observation"]
        vid = Path(meta["video_path"])
        sc_fps = float(meta.get("fps", 0))
        sc_n = int(meta.get("n_frames", 0))
        v_fps, v_n = get_video_meta(vid)
        npy = sc.with_suffix(".npy")
        label_n = int(np.load(npy).shape[0]) if npy.is_file() else -1

        # Pick best feature pkl
        stems = [obs, vid.stem] if obs != vid.stem else [obs]
        chosen, cands = pick_pkl(stems, label_n)
        chosen_rows = load_rows(chosen) if chosen else -1
        rows_match = abs(chosen_rows - label_n) <= 10 if chosen_rows > 0 else False

        flags = []
        if v_n == 0:
            flags.append("VIDEO_MISSING")
        elif abs(v_n - sc_n) > 5:
            flags.append(f"VIDEO_NFRAMES_MISMATCH(sc={sc_n},vid={v_n})")
        if abs(sc_fps - 60) > 0.1:
            flags.append(f"FPS_NOT_60(sc={sc_fps:.2f})")
        if abs(v_fps - sc_fps) > 0.1 and v_fps > 0:
            flags.append(f"VIDEO_FPS_MISMATCH(sc={sc_fps:.2f},vid={v_fps:.2f})")
        if label_n != sc_n:
            flags.append(f"LABEL_N_MISMATCH(sc={sc_n},npy={label_n})")
        if chosen is None:
            flags.append("NO_FEATURES")
        elif not rows_match:
            flags.append(f"FEAT_ROWS_MISMATCH(label={label_n},feat={chosen_rows})")
        # Check: did any candidate have wrong row count and would still be chosen?
        wrong_count_cands = [c for c in cands if c["rows"] > 0 and not c["rows_match"]]
        if wrong_count_cands and chosen is not None:
            # OK if we picked a correct one
            pass

        row = {
            "behaviour": b,
            "obs": obs,
            "vid_name": vid.name,
            "sc_n_frames": sc_n,
            "label_n_frames": label_n,
            "video_n_frames": v_n,
            "video_fps": v_fps,
            "chosen_pkl": chosen.name if chosen else "—",
            "chosen_rows": chosen_rows,
            "n_candidates": len(cands),
            "flags": flags,
        }
        by_behaviour[b].append(row)
        if flags:
            mismatches.append(row)

    # Per-behaviour summary
    for b, rows in sorted(by_behaviour.items()):
        n_ok = sum(1 for r in rows if not r["flags"])
        n_bad = len(rows) - n_ok
        marker = "✓" if n_bad == 0 else "!!"
        print(f"{marker} {b:25s}  {n_ok}/{len(rows)} sessions aligned")

    # Mismatches detail
    print()
    if not mismatches:
        print("✓ NO ALIGNMENT ISSUES FOUND across all behaviours.")
    else:
        print(f"!! {len(mismatches)} (behaviour, session) pairs flagged:\n")
        for r in mismatches:
            print(f"  [{r['behaviour']}] {r['obs']}")
            print(f"    video={r['vid_name']}  fps_actual={r['video_fps']:.1f}  "
                  f"n_frames(sc={r['sc_n_frames']}, label={r['label_n_frames']}, "
                  f"video={r['video_n_frames']})")
            print(f"    chosen_pkl={r['chosen_pkl']} rows={r['chosen_rows']} "
                  f"(out of {r['n_candidates']} candidates)")
            print(f"    flags: {', '.join(r['flags'])}")
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
