"""Sweep all *_features_*.pkl across E:/RSVIDS/Blackbox/ and report:
  - unique hashes
  - column counts per hash
  - presence of OF columns per hash
  - presence of silhouette columns per hash
  - which cohorts use each hash
  - try to reverse-engineer the config (sq, bp_pixbrt, OF on/off)

Output: prints a table to stdout.
"""
from __future__ import annotations

import itertools
import re
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from feature_cache import FeatureCacheManager as FCM  # noqa: E402


PAT = re.compile(r"_features_([a-f0-9]+)\.pkl$")


def reverse_engineer(target: str) -> dict | None:
    """Try common configs to find one whose compute_hash == target."""
    for bp in itertools.permutations(["hlpaw", "hrpaw", "snout"]):
        for sq in [[20], [40], [20, 20, 20], [40, 40, 40], [50], [50, 50, 50]]:
            for thresh in [0.3, 0.5]:
                for of in [False, True]:
                    for sil in [False, True]:
                        of_list = list(bp) if of else []
                        cfg = dict(
                            bp_include_list=None,
                            bp_pixbrt_list=list(bp),
                            square_size=sq,
                            pix_threshold=thresh,
                            include_optical_flow=of,
                            bp_optflow_list=of_list,
                            compute_silhouette=sil,
                            silhouette_floor=35,
                        )
                        if FCM.compute_hash(cfg) == target:
                            return cfg
    return None


def main() -> int:
    root = Path(r"E:/RSVIDS/Blackbox")
    print("Scanning feature pkls under", root)
    pkls = list(root.rglob("*_features_*.pkl"))
    print(f"found {len(pkls)} pkl files")

    # Group by hash
    by_hash: dict[str, list[Path]] = defaultdict(list)
    for p in pkls:
        m = PAT.search(p.name)
        if not m:
            continue
        by_hash[m.group(1)].append(p)

    print(f"\nUnique hashes: {len(by_hash)}\n")

    # For each hash, sample 1 pkl, get col count + flags
    rows = []
    for h, paths in sorted(by_hash.items(), key=lambda kv: -len(kv[1])):
        sample = paths[0]
        try:
            try:
                obj = joblib.load(sample)
            except Exception:
                with open(sample, "rb") as fh:
                    obj = pickle.load(fh)
            df = obj if hasattr(obj, "columns") else (
                obj.get("features") or obj.get("df")
            )
            ncols = df.shape[1] if df is not None else 0
            cols = list(df.columns) if df is not None else []
        except Exception as e:
            ncols = -1
            cols = []
            print(f"  ! could not inspect {sample}: {e}")

        of_n = sum(1 for c in cols if "Flow" in c)
        sil_n = sum(1 for c in cols if "silhouette" in c.lower())

        # cohort top-folder
        cohorts = defaultdict(int)
        for p in paths:
            try:
                rel = p.relative_to(root)
                top = rel.parts[0] if rel.parts else "?"
                cohorts[top] += 1
            except Exception:
                cohorts["?"] += 1

        cfg = reverse_engineer(h)
        cfg_str = (
            f"sq={cfg['square_size']} OF={cfg['include_optical_flow']} "
            f"sil={cfg.get('compute_silhouette', False)} "
            f"bp={cfg['bp_pixbrt_list']}"
        ) if cfg else "(no match)"

        rows.append({
            "hash": h,
            "n_pkls": len(paths),
            "cols": ncols,
            "OF_cols": of_n,
            "sil_cols": sil_n,
            "config": cfg_str,
            "cohorts": dict(cohorts),
        })

    # Print summary
    print(f"{'hash':10s} {'n_pkls':>7s} {'cols':>5s} {'OF':>4s} {'sil':>4s}  "
          f"config")
    for r in rows:
        print(f"{r['hash']:10s} {r['n_pkls']:>7d} {r['cols']:>5d} "
              f"{r['OF_cols']:>4d} {r['sil_cols']:>4d}  {r['config']}")

    # Per-cohort breakdown
    print()
    print("=== Per-cohort hash mix (counts) ===")
    cohort_hashes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        for cohort, count in r["cohorts"].items():
            cohort_hashes[cohort][r["hash"]] += count
    for cohort, hs in sorted(cohort_hashes.items()):
        print(f"\n  {cohort}")
        for h, n in sorted(hs.items(), key=lambda kv: -kv[1]):
            cfg_str = next((r["config"] for r in rows if r["hash"] == h), "?")
            print(f"    {h}: {n:>3d} pkls  ({cfg_str})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
