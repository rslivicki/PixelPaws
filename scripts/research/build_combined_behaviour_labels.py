"""Build per-frame label arrays for the v1 combined / multi-class
behaviours that aren't directly emitted by Phase B:

- body_grooming_combined: OR of body_grooming + back_groom + belly_groom
- state_joint: multi-class {still, walking, rearing, Facial_grooming,
  body_grooming_combined, none}. Frames where multiple positive labels
  overlap are resolved by priority (rearing > grooming > walking > still
  > none — based on biological salience of the rarer behaviours).

Reads existing .npy arrays from E:/RS_Boris/per_frame_labels/, writes
new ones in the same directory using the new canonical behaviour name
(plus sidecar .json).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LABELS_DIR = Path(r"E:/RS_Boris/per_frame_labels")


def find_sessions_with(behaviour: str) -> dict[str, dict]:
    """Return {observation: {npy_path, sidecar_meta}} for sessions
    having a .npy for `behaviour`."""
    out = {}
    for sc in sorted(LABELS_DIR.glob(f"*__{behaviour}.json")):
        meta = json.loads(sc.read_text())
        obs = meta["observation"]
        npy = sc.with_suffix(".npy")
        if npy.is_file():
            out[obs] = {"npy": npy, "meta": meta}
    return out


def write_combined(observation: str, canonical: str, arr: np.ndarray,
                   src_meta: dict, src_behaviours: list[str]) -> None:
    npy_path = LABELS_DIR / f"{observation}__{canonical}.npy"
    np.save(npy_path, arr)
    sidecar = {
        "observation": observation,
        "canonical_behaviour": canonical,
        "boris_project": src_meta.get("boris_project"),
        "video_path": src_meta.get("video_path"),
        "fps": src_meta.get("fps"),
        "n_frames": src_meta.get("n_frames"),
        "n_intervals": -1,  # combined, no single bout count
        "total_pos_frames": int(arr.sum()) if arr.dtype != np.int16 else
                            int((arr >= 0).sum()),
        "is_combined_from": src_behaviours,
    }
    npy_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))


def main() -> int:
    # ---- 1) body_grooming_combined ----
    print("== body_grooming_combined ==")
    src_behaviours = ["body_grooming", "back_groom", "belly_groom"]
    by_obs: dict[str, dict] = {}
    for b in src_behaviours:
        for obs, info in find_sessions_with(b).items():
            by_obs.setdefault(obs, {})[b] = info

    written = 0
    for obs, srcs in sorted(by_obs.items()):
        # Need a meta to populate n_frames + video_path
        any_meta = next(iter(srcs.values()))["meta"]
        n_frames = int(any_meta["n_frames"])
        combined = np.zeros(n_frames, dtype=np.int8)
        contributors = []
        for b in src_behaviours:
            if b in srcs:
                arr = np.load(srcs[b]["npy"])
                if len(arr) != n_frames:
                    # truncate or extend
                    n = min(len(arr), n_frames)
                    combined[:n] |= arr[:n].astype(np.int8)
                else:
                    combined |= arr.astype(np.int8)
                contributors.append(b)
        if combined.sum() == 0:
            continue
        write_combined(obs, "body_grooming_combined", combined,
                       any_meta, contributors)
        print(f"  + {obs}  pos={int(combined.sum()):,} from {contributors}")
        written += 1
    print(f"wrote {written} body_grooming_combined .npy files")
    print()

    # ---- 2) state_joint (6-class) ----
    # Priority (high → low): rearing > body_grooming_combined > Facial_grooming
    #                         > walking > still > none
    print("== state_joint ==")
    CLASS_NAMES = ["still", "walking", "Facial_grooming",
                   "body_grooming_combined", "rearing", "none"]
    NONE_IDX = CLASS_NAMES.index("none")
    # priority: index in this list = priority (later = higher priority)
    priority_order = ["still", "walking", "Facial_grooming",
                      "body_grooming_combined", "rearing"]

    by_obs2: dict[str, dict] = {}
    for b in priority_order:
        for obs, info in find_sessions_with(b).items():
            by_obs2.setdefault(obs, {})[b] = info

    written2 = 0
    for obs, srcs in sorted(by_obs2.items()):
        any_meta = next(iter(srcs.values()))["meta"]
        n_frames = int(any_meta["n_frames"])
        # Default to 'none' (last class), then overlay higher-priority classes
        joint = np.full(n_frames, NONE_IDX, dtype=np.int8)
        contributors = []
        for b in priority_order:
            if b not in srcs:
                continue
            arr = np.load(srcs[b]["npy"])
            if len(arr) != n_frames:
                n = min(len(arr), n_frames)
                mask = arr[:n].astype(bool)
                cls_idx = CLASS_NAMES.index(b)
                joint[:n] = np.where(mask, cls_idx, joint[:n])
            else:
                mask = arr.astype(bool)
                cls_idx = CLASS_NAMES.index(b)
                joint = np.where(mask, cls_idx, joint).astype(np.int8)
            contributors.append(b)
        if (joint != NONE_IDX).sum() == 0:
            continue
        npy_path = LABELS_DIR / f"{obs}__state_joint.npy"
        np.save(npy_path, joint)
        sidecar = {
            "observation": obs,
            "canonical_behaviour": "state_joint",
            "class_names": CLASS_NAMES,
            "none_idx": NONE_IDX,
            "boris_project": any_meta.get("boris_project"),
            "video_path": any_meta.get("video_path"),
            "fps": any_meta.get("fps"),
            "n_frames": n_frames,
            "is_combined_from": contributors,
            "class_counts": {CLASS_NAMES[i]: int((joint == i).sum())
                              for i in range(len(CLASS_NAMES))},
        }
        npy_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
        counts = sidecar["class_counts"]
        # short class summary
        nonzero = {k: v for k, v in counts.items() if v > 0}
        print(f"  + {obs}  classes={nonzero}")
        written2 += 1
    print(f"wrote {written2} state_joint .npy files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
