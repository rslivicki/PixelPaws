"""Batch-extract pose+brightness+OF features at the canonical 8aed1c22
schema (sq=40, OF=True, bp=hrpaw/hlpaw/snout) for all sessions referenced
by Tier 2/3 BORIS labels.

Runs MAX_PARALLEL extractor child processes simultaneously.
Skips sessions whose 8aed1c22 pkl already exists.

Output goes to <cohort_root>/features/<stem>_features_8aed1c22.pkl
where cohort_root is inferred from the BORIS sidecar's video_path.

Discord milestone posts on start, progress (every 5 done), and completion.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# 2 parallel by default; bump if CPU+disk can handle more.
MAX_PARALLEL = int(os.environ.get("PP_EXTRACT_PAR", "2"))

EXTRACTOR = Path(__file__).parent / "extract_session_8aed1c22.py"
EXPECTED_HASH = "8aed1c22"

BORIS_LABELS_DIR = Path(r"E:/RS_Boris/per_frame_labels")

# Behaviours whose sessions we need to extract for
TIER23_BEHAVIOURS = {
    "L_flinching", "body_grooming", "back_groom", "belly_groom",
    "still", "walking", "rearing", "Facial_grooming",
}

DLCTRACKER_WEBHOOK = (
    ""
)


def step(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def discord(text: str) -> None:
    try:
        req = urllib.request.Request(
            DLCTRACKER_WEBHOOK,
            data=json.dumps({"content": text}).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json",
                     "User-Agent": "PixelPaws-EncyclopediaBatch/1.0"},
        )
        with urllib.request.urlopen(req, timeout=15):
            pass
    except Exception as e:
        step(f"  ! discord failed: {e}")


_H5_SEARCH_ROOTS = [
    Path(r"E:/RSVIDS/Blackbox"),
]


def find_h5_for_video(video_path: Path) -> Path | None:
    """Look for a *filtered.h5 sibling, else broaden search to all known
    Blackbox roots (handles cohort splits where the .mp4 is in cohort A
    but the .h5 lives in cohort B)."""
    vstem = video_path.stem
    parent = video_path.parent
    # Local-first
    for p in parent.glob(f"{vstem}*filtered.h5"):
        if p.is_file():
            return p
    for p in parent.glob(f"{vstem}*.h5"):
        if p.is_file():
            return p
    for sibling in (parent / "Videos", parent.parent / "Videos"):
        if not sibling.is_dir():
            continue
        for p in sibling.glob(f"{vstem}*filtered.h5"):
            if p.is_file():
                return p
        for p in sibling.glob(f"{vstem}*.h5"):
            if p.is_file():
                return p
    # Broad fallback — search all Blackbox cohort roots
    for root in _H5_SEARCH_ROOTS:
        if not root.is_dir():
            continue
        # filtered first
        for p in root.rglob(f"{vstem}*filtered.h5"):
            if p.is_file():
                return p
        for p in root.rglob(f"{vstem}*.h5"):
            if p.is_file():
                return p
    return None


def _priority(stem: str) -> int:
    """Lower = higher priority. SNLT-Baseline sessions first so they
    unblock training on body_grooming / state / back_groom / belly_groom
    classifiers as soon as a handful are done; L_flinching sessions
    (2511 / 2601) come second."""
    if stem.startswith("260318_JG_") or stem.startswith("260425_JG_") or stem.startswith("2604_JG_"):
        return 0  # SNLT-Baseline
    if stem.startswith("260129_Formalin_"):
        return 1  # 2601 L_flinching cohort (newer videos)
    return 2     # 251114_*, 251126_* — 2511 L_flinching cohort


def collect_extraction_jobs() -> list[dict]:
    """Walk BORIS sidecars and produce list of {stem, h5, mp4, out_dir} jobs.
    Deduplicates by stem, then sorts by (priority, stem) so SNLT-Baseline
    extractions happen first.
    """
    by_stem: dict[str, dict] = {}
    for sc in sorted(BORIS_LABELS_DIR.glob("*.json")):
        if sc.name == "_summary.json":
            continue
        meta = json.loads(sc.read_text())
        if meta["canonical_behaviour"] not in TIER23_BEHAVIOURS:
            continue
        vid = Path(meta["video_path"])
        stem = vid.stem
        if stem in by_stem:
            continue
        if not vid.is_file():
            step(f"  ! video missing: {vid}")
            continue
        h5 = find_h5_for_video(vid)
        if h5 is None:
            step(f"  ! h5 missing for {stem}; skipping")
            continue
        # canonical out dir: <cohort>/features/  where cohort is inferred as
        # ancestor that contains a PixelPaws_project.json or videos/ dir.
        # Heuristic: walk up until we find features/ or PixelPaws_project.json,
        # then output to <that_dir>/features/. Fall back to <vid.parent>/features/.
        out_dir = None
        for anc in [vid.parent] + list(vid.parents):
            if (anc / "PixelPaws_project.json").is_file() or (anc / "features").is_dir():
                out_dir = anc / "features"
                break
        if out_dir is None:
            out_dir = vid.parent / "features"
        by_stem[stem] = {
            "stem": stem,
            "h5": str(h5),
            "mp4": str(vid),
            "out_dir": str(out_dir),
        }
    jobs = list(by_stem.values())
    jobs.sort(key=lambda j: (_priority(j["stem"]), j["stem"]))
    return jobs


def already_extracted(job: dict) -> bool:
    target = Path(job["out_dir"]) / f"{job['stem']}_features_{EXPECTED_HASH}.pkl"
    return target.is_file()


def run_one(job: dict) -> tuple[str, int, str]:
    """Run the extractor as a child process. Returns (stem, returncode, log_path)."""
    log_path = Path(r"E:/PixelPaws/scripts/research/batch_extract_logs") / f"{job['stem']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-X", "utf8", "-u", str(EXTRACTOR),
        job["stem"], job["h5"], job["mp4"], job["out_dir"],
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
    return job["stem"], proc.returncode, str(log_path)


def main() -> int:
    jobs = collect_extraction_jobs()
    if not jobs:
        step("No extraction jobs found")
        return 0

    total = len(jobs)
    done_count = sum(1 for j in jobs if already_extracted(j))
    pending = [j for j in jobs if not already_extracted(j)]
    step(f"Found {total} sessions; {done_count} already extracted; "
         f"{len(pending)} pending. MAX_PARALLEL={MAX_PARALLEL}")

    discord(
        f":rocket: **Encyclopedia: batch feature extraction starting**\n"
        f"Target hash `{EXPECTED_HASH}` (sq=40 + OF + bp=hrpaw/hlpaw/snout)\n"
        f"{total} unique sessions referenced by Tier 2/3 BORIS labels.\n"
        f"{done_count} already extracted, {len(pending)} pending.\n"
        f"Running {MAX_PARALLEL} parallel processes.\n"
        f"Estimated total: ~{len(pending)*10//MAX_PARALLEL} min."
    )

    if not pending:
        discord(":white_check_mark: Nothing to extract — all sessions already done.")
        return 0

    t_start = time.time()
    completed = done_count
    failures = []

    with ProcessPoolExecutor(max_workers=MAX_PARALLEL) as ex:
        futures = {ex.submit(run_one, j): j for j in pending}
        for fut in as_completed(futures):
            stem, rc, log = fut.result()
            completed += 1
            elapsed = (time.time() - t_start) / 60
            remaining = total - completed
            step(f"  [{completed}/{total}] {stem}: rc={rc}  elapsed={elapsed:.1f}m  remaining={remaining}")
            if rc != 0:
                failures.append((stem, log))
            # Milestone every 5 completions
            if completed % 5 == 0 or remaining == 0:
                discord(
                    f":hammer: **Extraction progress {completed}/{total}** "
                    f"({elapsed:.0f} min elapsed)\n"
                    f"Last: `{stem}` rc={rc}\n"
                    f"Failures so far: {len(failures)}"
                )

    duration = (time.time() - t_start) / 60
    step(f"\n== Batch done in {duration:.1f} min. Failures: {len(failures)} ==")
    for stem, log in failures:
        step(f"  FAIL: {stem}  log={log}")

    discord(
        f":checkered_flag: **Extraction batch finished** in {duration:.0f} min.\n"
        f"Success: {total-len(failures)}/{total}  Failures: {len(failures)}\n"
        + ("Failed sessions: " + ", ".join(s for s, _ in failures) if failures else
           "All sessions extracted at `8aed1c22`.")
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
