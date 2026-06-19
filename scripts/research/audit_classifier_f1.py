"""Phase A — audit every existing PixelPaws classifier pkl on disk.

Loads each candidate `.pkl`, extracts the stored training-history /
CV-F1 metadata, and writes a single CSV summary at
`E:/PixelPaws/pixelpaws_global_classifier_encyclopedia/audit_report.csv`.

For each behaviour the script picks the best-performing candidate
(highest frame F1) across cohorts and tags it KEEP / RETRAIN / DROP
relative to the 0.85 inclusion gate.
"""
from __future__ import annotations

import csv
import json
import pickle
import re
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ENCYCLOPEDIA = Path(r"E:\PixelPaws\pixelpaws_global_classifier_encyclopedia")
ENCYCLOPEDIA.mkdir(parents=True, exist_ok=True)

CANDIDATE_DIRS = [
    Path(r"E:\RSVIDS\Blackbox\260506_RS_THC_Withdrawal\classifiers"),
    Path(r"E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\classifiers"),
    Path(r"E:\RSVIDS\Blackbox\260512_THC_Rim_Cohort\classifiers"),
    Path(r"E:\RSVIDS\Blackbox\260515_Rim_DoseResp\classifiers"),
    # the 2510 Rimonabant cohort has higher-F1 Scratching variants
    # (Apr 29 2026 retrain reached F1=0.95) — include the root Classifiers
    # dir plus the HomeRig + old_classifiers subdirs.
    Path(r"E:\RSVIDS\Blackbox\2510_Blackbox_Rimonabant\Blackbox_videos-selected\Classifiers"),
    Path(r"E:\RSVIDS\Blackbox\2510_Blackbox_Rimonabant\Blackbox_videos-selected\PixelPaws_Classifiers\HomeRig"),
    # newest "honest" rerun directory (joblib-saved)
    Path(r"E:\RSVIDS\Blackbox\2510_Blackbox_Rimonabant\Blackbox_videos-selected\Classifiers\PixelPaws_Scratching_honest_20260521_160523"),
    # ChR2 stim project — same behaviour set, may have alt training runs
    Path(r"E:\RSVIDS\Blackbox\AM_ChR2_stim_SPB_analysis\classifiers"),
]

# Reuse the for_claude.json post-processing config from SNLT as a reference.
FOR_CLAUDE_JSON = Path(
    r"E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\transitions\for_claude.json"
)

GATE_F1 = 0.85

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
                     "User-Agent": "PixelPaws-Encyclopedia-Audit/1.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            pass
    except Exception as e:
        step(f"  ! discord milestone failed: {e}")


# ------------------------------------------------------------------ #
# Behaviour extraction
# ------------------------------------------------------------------ #

BEHAVIOUR_FROM_FILENAME = re.compile(
    r"^PixelPaws_(?P<beh>.+?)"
    r"(?:_AllFeatures|_pruned_\d+)?"
    r"(?:_honest)?"
    r"(?:_\d{8}_\d{6})?"   # optional timestamp suffix
    r"\.pkl$",
    re.IGNORECASE,
)


def parse_behaviour(path: Path) -> str | None:
    m = BEHAVIOUR_FROM_FILENAME.match(path.name)
    return m.group("beh") if m else None


def parse_variant(path: Path) -> str:
    n = path.name.lower()
    v = "default"
    if "_pruned_" in n:
        v = "pruned"
    elif "_allfeatures" in n:
        v = "allfeatures"
    if "_honest" in n:
        v = v + "_honest"
    return v


def parse_cohort(path: Path) -> str:
    """Walk parents to find the cohort name."""
    parts = [p.name.lower() for p in path.parents]
    joined = "/".join(parts)
    if "thc_withdrawal" in joined:
        return "THC_Withdrawal"
    if "snlt" in joined:
        return "SNLT"
    if "thc_rim_cohort" in joined:
        return "THC_Rim"
    if "rim_doseresp" in joined:
        return "Rim_DoseResp"
    if "2510_blackbox_rimonabant" in joined:
        return "Rim_2510"
    if "chr2_stim" in joined:
        return "ChR2_stim"
    return "unknown"


# ------------------------------------------------------------------ #
# pkl introspection — try multiple field shapes the GUI trainer used
# ------------------------------------------------------------------ #

def safe_float(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default


def extract_cv_metrics(d: dict) -> dict:
    """Return {cv_f1_mean, cv_f1_std, cv_f1_per_fold, oof_best_f1,
    best_thresh, min_bout, max_gap, n_features, n_train_sessions,
    has_optuna, was_pruned, raw_keys}."""
    out = {
        "cv_f1_mean": float("nan"),
        "cv_f1_std": float("nan"),
        "cv_f1_per_fold": "",
        "oof_best_f1": float("nan"),
        "best_thresh": float("nan"),
        "min_bout": float("nan"),
        "min_after_bout": float("nan"),
        "max_gap": float("nan"),
        "n_features": float("nan"),
        "n_train_sessions": float("nan"),
        "feature_hash": "",
        "has_optuna": False,
        "was_pruned": False,
        "raw_keys": "",
    }

    if not isinstance(d, dict):
        out["raw_keys"] = type(d).__name__
        return out

    out["raw_keys"] = ",".join(sorted(d.keys()))

    # CV-F1 — the trainer stores these as flat keys
    if "cv_f1_scores" in d:
        try:
            arr = np.array([safe_float(x) for x in d["cv_f1_scores"]
                            if np.isfinite(safe_float(x))], dtype=float)
            if len(arr):
                out["cv_f1_per_fold"] = ";".join(f"{x:.3f}" for x in arr)
        except Exception:
            pass
    if "mean_cv_f1" in d:
        out["cv_f1_mean"] = safe_float(d["mean_cv_f1"])
    if "std_cv_f1" in d:
        out["cv_f1_std"] = safe_float(d["std_cv_f1"])
    if "oof_best_f1" in d:
        out["oof_best_f1"] = safe_float(d["oof_best_f1"])

    # Post-process config
    for k_in, k_out in (("best_thresh", "best_thresh"),
                         ("min_bout", "min_bout"),
                         ("min_after_bout", "min_after_bout"),
                         ("max_gap", "max_gap")):
        if k_in in d:
            out[k_out] = safe_float(d[k_in])

    # Feature count
    if "selected_feature_cols" in d:
        try:
            out["n_features"] = float(len(d["selected_feature_cols"]))
        except Exception:
            pass
    else:
        model = d.get("clf_model")
        if model is not None and hasattr(model, "feature_names_in_"):
            try:
                out["n_features"] = float(len(model.feature_names_in_))
            except Exception:
                pass

    # Training sessions
    if "training_sessions" in d:
        try:
            out["n_train_sessions"] = float(len(d["training_sessions"]))
        except Exception:
            pass

    # Optuna footprint
    out["has_optuna"] = bool(d.get("optuna_best_params"))
    out["was_pruned"] = "_pruned_" in (out["raw_keys"].lower()) or (
        # pruned pkls have selected_feature_cols smaller than the model
        # they were trained from; can't tell precisely from metadata alone
        # so rely on filename signal (set in main loop).
        False
    )

    return out


def load_pkl(path: Path) -> dict | None:
    """Try joblib first (newer pkls), fall back to plain pickle (older ones)."""
    # joblib first
    try:
        import joblib
        d = joblib.load(path)
        if isinstance(d, dict):
            return d
    except Exception:
        pass
    # plain pickle fallback
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        step(f"  ! failed to load {path.name} via joblib + pickle: {e}")
        return None


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main() -> int:
    discord_milestone(
        ":mag: **Encyclopedia build — Phase A** starting.\n"
        f"Auditing classifier pkls across {len(CANDIDATE_DIRS)} cohort folders."
    )

    rows = []
    seen = 0
    for d in CANDIDATE_DIRS:
        if not d.is_dir():
            step(f"  skip (missing dir): {d}")
            continue
        for p in sorted(d.glob("*.pkl")):
            seen += 1
            beh = parse_behaviour(p)
            if beh is None:
                step(f"  skip (unparseable name): {p.name}")
                continue
            variant = parse_variant(p)
            cohort = parse_cohort(p)
            d_obj = load_pkl(p)
            if d_obj is None:
                continue
            metrics = extract_cv_metrics(d_obj)
            rows.append({
                "behaviour": beh,
                "variant":   variant,
                "cohort":    cohort,
                "path":      str(p),
                "size_mb":   round(p.stat().st_size / 1e6, 2),
                **metrics,
            })
            step(
                f"  {beh:25s} ({variant:11s}, {cohort:15s}): "
                f"F1={metrics['cv_f1_mean']:.3f} std={metrics['cv_f1_std']:.3f} "
                f"thresh={metrics['best_thresh']:.2f} "
                f"feats={metrics['n_features']:.0f}"
            )

    step(f"loaded {len(rows)} / {seen} candidate pkls")

    # Pick best variant per behaviour (highest F1).
    by_beh: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        # normalise behaviour name capitalisation — Scratching, Left_licking etc.
        # keep raw case for path traceability but use a canon for grouping.
        canon = r["behaviour"]
        canon = canon.replace("left_licking", "Left_licking")
        canon = {
            "facial_grooming": "Facial_grooming",
            "Facial_Grooming": "Facial_grooming",
            "body_grooming":   "body_grooming",
        }.get(canon, canon)
        r["behaviour_canon"] = canon
        by_beh[canon].append(r)

    best_rows: list[dict] = []
    for beh, candidates in by_beh.items():
        valid = [c for c in candidates if np.isfinite(c["cv_f1_mean"])]
        if not valid:
            # no metrics anywhere — pick the largest pruned variant as the
            # representative and flag it.
            pick = sorted(candidates, key=lambda c: -c["size_mb"])[0]
            pick["status"] = "UNKNOWN_F1"
            pick["chosen_reason"] = "no cv_f1 metadata found in any candidate"
        else:
            pick = max(valid, key=lambda c: c["cv_f1_mean"])
            f1 = pick["cv_f1_mean"]
            if f1 >= GATE_F1:
                pick["status"] = "KEEP"
            elif f1 >= GATE_F1 - 0.10:
                pick["status"] = "RETRAIN"
            else:
                pick["status"] = "DROP"
            pick["chosen_reason"] = f"best of {len(valid)} variants (F1={f1:.3f})"
        best_rows.append(pick)

    # write the full audit report (all rows) + the best-pick summary.
    out_csv = ENCYCLOPEDIA / "audit_report.csv"
    full_fields = ["behaviour", "behaviour_canon", "variant", "cohort",
                   "size_mb", "cv_f1_mean", "cv_f1_std", "cv_f1_per_fold",
                   "oof_best_f1", "best_thresh", "min_bout",
                   "min_after_bout", "max_gap", "n_features",
                   "n_train_sessions", "has_optuna", "was_pruned", "path",
                   "raw_keys"]
    with open(out_csv, "w", newline="", encoding="utf-8") as fout:
        w = csv.DictWriter(fout, fieldnames=full_fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    pick_csv = ENCYCLOPEDIA / "audit_best_per_behaviour.csv"
    pick_fields = ["behaviour_canon", "status", "cv_f1_mean", "cv_f1_std",
                   "oof_best_f1", "variant", "cohort", "best_thresh",
                   "min_bout", "max_gap", "n_features", "n_train_sessions",
                   "chosen_reason", "path"]
    with open(pick_csv, "w", newline="", encoding="utf-8") as fout:
        w = csv.DictWriter(fout, fieldnames=pick_fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(best_rows, key=lambda r: r["behaviour_canon"]):
            w.writerow(r)

    # Summary print
    step("\n== best-per-behaviour summary ==")
    keep, retrain, drop, unknown = 0, 0, 0, 0
    for r in sorted(best_rows, key=lambda r: r["behaviour_canon"]):
        flag = r["status"]
        f1 = r.get("cv_f1_mean", float("nan"))
        step(
            f"  {r['behaviour_canon']:25s}  {flag:10s}  "
            f"F1={f1:.3f}  ({r['variant']:11s} from {r['cohort']})"
        )
        if flag == "KEEP":      keep += 1
        elif flag == "RETRAIN": retrain += 1
        elif flag == "DROP":    drop += 1
        else:                   unknown += 1

    step(
        f"\nAudit complete: {keep} KEEP, {retrain} RETRAIN, "
        f"{drop} DROP, {unknown} UNKNOWN_F1"
    )
    step(f"Written: {out_csv}")
    step(f"Written: {pick_csv}")

    discord_milestone(
        ":white_check_mark: **Phase A complete.**\n"
        f"Audited {len(rows)} pkls across {len(CANDIDATE_DIRS)} cohort folders. "
        f"Best-per-behaviour summary: {keep} KEEP, {retrain} RETRAIN, "
        f"{drop} DROP, {unknown} UNKNOWN_F1 (no stored CV metadata).\n"
        f"Outputs:\n"
        f"`{out_csv.name}` — full per-pkl table\n"
        f"`{pick_csv.name}` — best variant per behaviour"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
