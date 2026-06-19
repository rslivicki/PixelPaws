"""Phase B — convert every BORIS observation to per-frame binary label
arrays (.npy) under E:/RS_Boris/per_frame_labels/.

For each observation in each .boris file:
  - Look up the .mp4 path (BORIS stores `file` as a dict subject_id->list).
  - Read FPS + frame count from the video via cv2 (no DLC h5 required).
  - Normalise raw behaviour codes through boris_label_aliases.canonical().
  - Coalesce all START/STOP pairs for that canonical behaviour into a
    single (n_frames,) int8 array.
  - Save as `<observation>__<canonical>.npy` + sidecar `.json`.

Skips any observation whose video isn't on disk (logs as orphan).
Skips raw codes that map to None (per the alias map).
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from boris_label_aliases import ALIASES, canonical  # noqa: E402

BORIS_DIR = Path(r"E:\RS_Boris")
BORIS_FILES = [
    BORIS_DIR / "Blackbox_Formalin.boris",
    BORIS_DIR / "Blackbox_Rimonabant_Scratching.boris",
    BORIS_DIR / "SNLT_Rearing_Grooming_shaking.boris",
]
# v1 behaviour extraction policy (per user) — only emit these canonical
# names from each project, even if the BORIS file contains more.
PROJECT_KEEP = {
    "Blackbox_Formalin.boris":
        {"Left_licking", "L_flinching"},
    "Blackbox_Rimonabant_Scratching.boris":
        {"Scratching", "Left_licking"},
    "SNLT_Rearing_Grooming_shaking.boris":
        {"rearing", "body_grooming", "Facial_grooming", "walking", "still",
         "belly_groom", "back_groom", "unsupported_rear", "L_flinching"},
}

OUT_DIR = BORIS_DIR / "per_frame_labels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DLCTRACKER_WEBHOOK = (
    ""
)


def step(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def discord_milestone(text: str) -> None:
    try:
        req = urllib.request.Request(
            DLCTRACKER_WEBHOOK,
            data=json.dumps({"content": text}).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json",
                     "User-Agent": "PixelPaws-Encyclopedia-PhaseB/1.0"},
        )
        with urllib.request.urlopen(req, timeout=15):
            pass
    except Exception as e:
        step(f"  ! discord milestone failed: {e}")


# Roots to scan when a BORIS-listed video isn't at its stored path.
# Walks ~2 levels for the basename.
FALLBACK_ROOTS = [
    Path(r"E:\RSVIDS\Blackbox\2510_Blackbox_Rimonabant"),
    Path(r"E:\RSVIDS\Blackbox\2511_Blackbox_Formalin"),
    Path(r"E:\RSVIDS\Blackbox\2603_SNLT_JG"),
    Path(r"E:\RSVIDS\Blackbox\260506_RS_THC_Withdrawal"),
    Path(r"E:\RSVIDS\Blackbox\260512_THC_Rim_Cohort"),
    Path(r"E:\RSVIDS\Blackbox\260515_Rim_DoseResp"),
    Path(r"E:\RSVIDS\Blackbox\2601_JDR_videos"),
]


def _search_by_basename(basename: str) -> Path | None:
    """Search FALLBACK_ROOTS recursively (up to depth 3) for this filename."""
    for root in FALLBACK_ROOTS:
        if not root.is_dir():
            continue
        # rglob is slow on big trees — but each root has only a few subdirs.
        for hit in root.rglob(basename):
            if hit.is_file():
                return hit
    return None


def resolve_video(file_field) -> Path | None:
    """BORIS stores `file` as a dict {subject_str: [path1, ...]} or
    occasionally a flat list. Return the first existing mp4 path.
    Falls back to a basename search across FALLBACK_ROOTS."""
    if not file_field:
        return None
    paths: list[str] = []
    if isinstance(file_field, dict):
        for v in file_field.values():
            if isinstance(v, list):
                paths.extend(v)
            elif isinstance(v, str):
                paths.append(v)
    elif isinstance(file_field, list):
        paths.extend(file_field)
    elif isinstance(file_field, str):
        paths.append(file_field)

    for p in paths:
        candidate = Path(p)
        if candidate.is_file():
            return candidate
        candidate2 = Path(str(candidate).replace("/", "\\"))
        if candidate2.is_file():
            return candidate2

    # Fallback: search by basename across known root dirs.
    for p in paths:
        bn = Path(p).name
        if bn:
            hit = _search_by_basename(bn)
            if hit is not None:
                return hit
    return None


def video_meta(path: Path) -> tuple[float, int]:
    """Return (fps, n_frames) for the video. Raises on failure."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not (fps > 0):
        raise RuntimeError(f"non-positive fps {fps} for {path}")
    if not (n > 0):
        raise RuntimeError(f"non-positive frame count {n} for {path}")
    return fps, n


def extract_events_by_canonical(events: list, project_keep: set[str]
                                ) -> dict[str, list[tuple[float, float]]]:
    """Return {canonical_name: [(start_sec, stop_sec), ...]} after
    aliasing + intersecting with project_keep.

    BORIS state events alternate START / STOP per (subject, behaviour).
    Group events by (subject, raw_code), sort by time, pair consecutively.
    """
    # group raw events by (subject, raw_code)
    by_key: dict[tuple[str, str], list[tuple[float, str]]] = defaultdict(list)
    for ev in events:
        if len(ev) < 6:
            continue
        t_sec = float(ev[0])
        subject = str(ev[1])
        code = str(ev[2])
        by_key[(subject, code)].append((t_sec, code))

    out: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (subject, code), lst in by_key.items():
        try:
            canon = canonical(code)
        except KeyError:
            step(f"  ! unknown BORIS code (not in alias map): {code!r}; "
                 "see boris_label_aliases.py")
            continue
        if canon is None:
            continue
        if canon not in project_keep:
            continue
        # Sort, pair consecutive: even-indexed = start, odd-indexed = stop
        lst.sort(key=lambda t: t[0])
        if len(lst) % 2 != 0:
            step(f"  ! odd event count for ({subject},{code}) = {len(lst)} — "
                 "BORIS state events should be START/STOP pairs; dropping the dangler")
            lst = lst[:-1]
        for i in range(0, len(lst), 2):
            start_sec = lst[i][0]
            stop_sec = lst[i + 1][0]
            if stop_sec <= start_sec:
                continue
            out[canon].append((start_sec, stop_sec))
    return out


def build_label_array(intervals: list[tuple[float, float]], fps: float,
                      n_frames: int) -> np.ndarray:
    arr = np.zeros(n_frames, dtype=np.int8)
    for start_sec, stop_sec in intervals:
        s_idx = max(0, int(round(start_sec * fps)))
        e_idx = min(n_frames, int(round(stop_sec * fps)))
        if e_idx > s_idx:
            arr[s_idx:e_idx] = 1
    return arr


def main() -> int:
    discord_milestone(
        ":mag: **Encyclopedia build — Phase B** starting. "
        f"Converting BORIS labels to per-frame .npy arrays "
        f"under `{OUT_DIR}`."
    )

    # Header
    step("Alias map (raw BORIS code -> canonical):")
    grouped: dict[str | None, list[str]] = defaultdict(list)
    for raw, canon in ALIASES.items():
        grouped[canon].append(raw)
    for canon, raws in sorted(grouped.items(), key=lambda kv: (kv[0] or "")):
        if canon is None:
            step(f"  EXCLUDED: {raws}")
        else:
            step(f"  {canon:25s} <- {raws}")

    # Build labels
    summary: dict[str, dict] = defaultdict(lambda: {
        "n_observations": 0,
        "n_intervals": 0,
        "n_pos_frames": 0,
    })
    orphan_count = 0
    total_obs = 0
    written = 0
    for boris_path in BORIS_FILES:
        if not boris_path.is_file():
            step(f"!! missing {boris_path}")
            continue
        with open(boris_path, "r", encoding="utf-8") as fh:
            proj = json.load(fh)
        keep_set = PROJECT_KEEP.get(boris_path.name, set())
        obs_map = proj.get("observations", {})
        step(f"\n== {boris_path.name} ({len(obs_map)} obs; "
             f"keeping {sorted(keep_set)}) ==")
        for obs_name, obs_data in obs_map.items():
            total_obs += 1
            video = resolve_video(obs_data.get("file"))
            if video is None:
                step(f"  ! orphan (video not on disk): {obs_name}")
                orphan_count += 1
                continue
            try:
                fps, n_frames = video_meta(video)
            except Exception as e:
                step(f"  ! {obs_name}: {e}")
                continue

            events = obs_data.get("events", [])
            by_canon = extract_events_by_canonical(events, keep_set)
            if not by_canon:
                step(f"  - {obs_name}: no v1 behaviours after aliasing/filter")
                continue

            for canon, intervals in by_canon.items():
                arr = build_label_array(intervals, fps, n_frames)
                pos = int(arr.sum())
                if pos == 0:
                    continue
                npy_path = OUT_DIR / f"{obs_name}__{canon}.npy"
                np.save(npy_path, arr)
                sidecar = {
                    "observation": obs_name,
                    "canonical_behaviour": canon,
                    "boris_project": boris_path.name,
                    "video_path": str(video),
                    "fps": fps,
                    "n_frames": n_frames,
                    "n_intervals": len(intervals),
                    "total_pos_frames": pos,
                }
                npy_path.with_suffix(".json").write_text(
                    json.dumps(sidecar, indent=2)
                )
                summary[canon]["n_observations"] += 1
                summary[canon]["n_intervals"] += len(intervals)
                summary[canon]["n_pos_frames"] += pos
                written += 1
                step(f"  + {obs_name}__{canon}.npy  "
                     f"({len(intervals)} bouts, {pos} pos frames)")

    step(
        f"\n== Phase B summary: {written} .npy files written from "
        f"{total_obs} observations ({orphan_count} orphans) =="
    )
    step(f"{'behaviour':22s} {'n_sessions':>10s} {'n_bouts':>10s} "
         f"{'n_pos_frames':>14s}")
    for canon, s in sorted(summary.items()):
        step(f"{canon:22s} {s['n_observations']:>10d} "
             f"{s['n_intervals']:>10d} {s['n_pos_frames']:>14d}")

    # save a project-wide aggregate summary
    summary_path = OUT_DIR / "_summary.json"
    summary_path.write_text(json.dumps(
        {"per_behaviour": summary, "orphan_observations": orphan_count,
         "total_observations": total_obs, "files_written": written},
        indent=2, default=int,
    ))
    step(f"\nWritten: {summary_path}")

    discord_milestone(
        ":white_check_mark: **Phase B complete.**\n"
        f"Wrote {written} per-frame label .npy files across "
        f"{len(summary)} canonical behaviours. "
        f"Orphans (missing video): {orphan_count}.\n"
        "Top counts: " + ", ".join(
            f"{c}={s['n_pos_frames']:,}"
            for c, s in sorted(summary.items(),
                               key=lambda kv: -kv[1]['n_pos_frames'])[:6]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
