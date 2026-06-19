"""Phase C — train a single global classifier for one canonical behaviour.

Pipeline (with a 90-min wall-clock budget per behaviour):
  1. Load all per-frame BORIS labels for the behaviour from
     E:/RS_Boris/per_frame_labels/<obs>__<behaviour>.npy
  2. Match each to its feature pkl (under any *_features_<hash>.pkl on disk).
  3. Concatenate into (X, y, session_ids).
  4. Baseline XGBoost + session-level GroupKFold (5 folds, clamped to
     n_sessions). Threshold-optimise on OOF probabilities. CV frame F1.
  5. If CV F1 < 0.90: SHAP-prune ladder 200 -> 150 -> 100 -> 50, keep
     smallest feature set within 0.01 F1 of best.
  6. If CV F1 < 0.85: Optuna hyperparameter search (n_trials=30, capped
     by wall clock).
  7. Bout-level eval (IoU >= 0.5 -> bout F1; bout-count Pearson r).
  8. Isotonic calibration on OOF probabilities.
  9. Seed stability: retrain with seeds {1,2,3,4} (final model uses 42).
 10. Save .pkl + training_metrics.json + post Discord milestone.

Usage:
  python train_global_classifier.py <behaviour> [--budget-min 90]
  python train_global_classifier.py L_flinching
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
import re
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ----------------------------------------------------------------- #
ENCYCLOPEDIA = Path(r"E:\PixelPaws\pixelpaws_global_classifier_encyclopedia")
LABELS_DIR = Path(r"E:\RS_Boris\per_frame_labels")

OUT_CLF_DIR = ENCYCLOPEDIA / "classifiers"
OUT_PERF_DIR = ENCYCLOPEDIA / "performance_results"
OUT_DOCS_DIR = ENCYCLOPEDIA / "docs"
for d in (OUT_CLF_DIR, OUT_PERF_DIR, OUT_DOCS_DIR):
    d.mkdir(parents=True, exist_ok=True)

GATE_F1 = 0.85
SKIP_OPTUNA_F1 = 0.90
SEED_PRIMARY = 42
SEED_STABILITY = [1, 2, 3, 4]
SHAP_LADDER = [200, 150, 100, 50]

DLCTRACKER_WEBHOOK = (
    ""
)

FPS_DEFAULT = 60

# ----------------------------------------------------------------- #
def step(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def discord_milestone(text: str) -> None:
    try:
        req = urllib.request.Request(
            DLCTRACKER_WEBHOOK,
            data=json.dumps({"content": text}).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json",
                     "User-Agent": "PixelPaws-Encyclopedia-Train/1.0"},
        )
        with urllib.request.urlopen(req, timeout=15):
            pass
    except Exception as e:
        step(f"  ! discord milestone failed: {e}")


# ----------------------------------------------------------------- #
# Data loading
# ----------------------------------------------------------------- #

FEATURE_PKL_NAME = re.compile(r"^(.+?)_features_[0-9a-f]+\.pkl$")


def index_feature_pkls() -> dict[str, list[Path]]:
    pkls = []
    for g in [r"E:/RSVIDS/Blackbox/**/features/*_features_*.pkl",
              r"E:/RSVIDS/Blackbox/**/*_features_*.pkl"]:
        pkls.extend(glob.glob(g, recursive=True))
    out: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(set(pkls)):
        name = Path(p).name
        m = FEATURE_PKL_NAME.match(name)
        if m:
            out[m.group(1)].append(Path(p))
    return out


def load_features(path: Path) -> pd.DataFrame:
    try:
        d = joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            d = pickle.load(f)
    if isinstance(d, dict):
        for k in ("X", "features", "df"):
            if k in d:
                d = d[k]
                break
    if isinstance(d, pd.DataFrame):
        return d
    raise RuntimeError(f"unrecognised feature pkl shape at {path}")


def load_behaviour_dataset(behaviour: str,
                           exclude_sessions: list[str] | None = None,
                           prefer_hash: str | None = None,
                           neg_sample_ratio: int | None = None,
                           trim_to_last_positive: bool = False,
                           ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray,
                                       list[str], list[Path]]:
    """Returns X, y, session_ids (int array), session_names, feature_paths.

    exclude_sessions: list of observation names (or video stems) to skip.
    prefer_hash: when multiple feature pkls exist for a session, prefer
        the one whose filename contains this hash (e.g. "8aed1c22").
    """
    sidecars = [p for p in LABELS_DIR.glob(f"*__{behaviour}.json")
                if p.name != "_summary.json"]
    if not sidecars:
        raise SystemExit(f"No .npy labels found for behaviour {behaviour!r}")
    exclude_set = set(exclude_sessions or [])

    feat_index = index_feature_pkls()

    def _pkl_score(p: Path, label_n: int = 0) -> tuple:
        """Higher tuple wins. Prioritise:
        0. row count within ±10 of the label n_frames (when known)
        1. preferred hash (when caller pinned one)
        2. parent dir is exactly `features/` (the canonical cohort dir)
        3. NOT in `FeatureCache/` or `old_*` or `Barefoot*`
        4. recency (mtime)
        """
        parent = p.parent.name.lower()
        canon = (parent == "features")
        bad = ("featurecache" in parent or parent.startswith("old")
               or "barefoot" in str(p).lower())
        hash_match = (prefer_hash is not None and prefer_hash in p.name)
        # Row-count alignment: read the cached row count quickly.
        # Falls back to 0 if we can't open (will be deprioritised).
        rows_match = False
        if label_n > 0:
            try:
                import joblib
                obj = joblib.load(p, mmap_mode="r")
                df_or_dict = obj
                if isinstance(df_or_dict, dict):
                    df_or_dict = (df_or_dict.get("X") or df_or_dict.get("features")
                                  or df_or_dict.get("df"))
                if df_or_dict is not None and hasattr(df_or_dict, "shape"):
                    rows = df_or_dict.shape[0]
                    rows_match = abs(rows - label_n) <= 10
            except Exception:
                pass
        return (rows_match, hash_match, canon, not bad, p.stat().st_mtime)

    Xs, ys, sids, snames, fpaths = [], [], [], [], []
    sid = 0
    for js in sorted(sidecars):
        meta = json.load(open(js))
        obs = meta["observation"]
        video_stem = Path(meta["video_path"]).stem
        label_n_frames = int(meta.get("n_frames") or 0)
        if obs in exclude_set or video_stem in exclude_set:
            step(f"  - {obs}: excluded by --exclude-session")
            continue
        # Combine BOTH lookups (obs + video_stem). Some sessions have
        # multiple .pkl variants under different stems that come from
        # different recordings of the same animal (e.g. 260318_JG_9417
        # vs 260318_JG_9417_Baseline). We prefer the pkl whose row
        # count matches the BORIS-derived label array.
        feat_paths = list(dict.fromkeys(  # preserve order, dedupe
            (feat_index.get(obs) or []) + (feat_index.get(video_stem) or [])
        ))
        if not feat_paths:
            step(f"  ! no feature pkl for {obs}; skipping")
            continue
        # prefer row-count match with labels, then hash, then canonical dir
        fp = max(feat_paths, key=lambda p: _pkl_score(p, label_n_frames))
        try:
            X_s = load_features(fp)
        except Exception as e:
            step(f"  ! could not load {fp.name}: {e}; skipping")
            continue

        npy_path = js.with_suffix(".npy")
        y_s = np.load(npy_path).astype(np.int8)
        # Align lengths: features may have fewer rows than label array
        # (cv2 frame count can differ from DLC inferred frame count by ±1)
        n = min(len(X_s), len(y_s))
        if abs(len(X_s) - len(y_s)) > 5:
            step(f"  warn {obs}: |features|={len(X_s)} vs |labels|={len(y_s)} "
                 f"(diff {len(X_s) - len(y_s)}); truncating to {n}")
        X_s = X_s.iloc[:n].reset_index(drop=True)
        y_s = y_s[:n]

        # Trim trailing unlabeled tail. For sparse-clicked behaviours
        # (still/walking/etc.) the BORIS-dense .npy treats every frame
        # past the last labeled bout as "negative", which poisons
        # training because the labeler simply never reached those
        # frames. Trim to last_pos + 1 (matches the GUI's
        # `trim_to_last_positive` setting which defaults to True).
        if trim_to_last_positive and y_s.sum() > 0:
            last_pos = int(np.where(y_s == 1)[0].max())
            trim_n = last_pos + 1
            if trim_n < n:
                X_s = X_s.iloc[:trim_n].reset_index(drop=True)
                y_s = y_s[:trim_n]
                step(f"    trimmed {obs} to {trim_n:,} frames "
                     f"(last pos at {last_pos})")
                n = trim_n

        pos = int(y_s.sum())
        step(f"  + {obs:35s}  feats={X_s.shape[1]}  rows={n:,}  "
             f"pos={pos:,} ({100*pos/n:.2f}%)  src={fp.parent.name}")
        Xs.append(X_s)
        ys.append(y_s)
        sids.extend([sid] * n)
        snames.append(obs)
        fpaths.append(fp)
        sid += 1

    if not Xs:
        raise SystemExit(f"No usable sessions for {behaviour!r}")

    # Sanity: drop sessions whose feature count is far below the modal.
    # An off-schema pkl (e.g. 375 cols when everyone else has 599) would
    # crush the column intersection.
    from collections import Counter
    nfeats = Counter(X_s.shape[1] for X_s in Xs)
    modal_n, _ = nfeats.most_common(1)[0]
    keep_mask = []
    kept_Xs, kept_ys, kept_sids, kept_snames, kept_fpaths = [], [], [], [], []
    new_sid = 0
    for i, X_s in enumerate(Xs):
        if X_s.shape[1] < modal_n - 100:
            step(f"  ! dropping {snames[i]} (only {X_s.shape[1]} cols vs "
                 f"modal {modal_n} — likely wrong cache); was {fpaths[i]}")
            continue
        kept_Xs.append(X_s)
        kept_ys.append(ys[i])
        n_rows = len(ys[i])
        kept_sids.extend([new_sid] * n_rows)
        kept_snames.append(snames[i])
        kept_fpaths.append(fpaths[i])
        new_sid += 1
    Xs = kept_Xs
    ys = kept_ys
    sids = kept_sids
    snames = kept_snames
    fpaths = kept_fpaths

    # Align columns across sessions: intersect column sets
    common_cols = set(Xs[0].columns)
    for X_s in Xs[1:]:
        common_cols &= set(X_s.columns)
    common_cols = sorted(common_cols)
    step(f"  common feature columns across {len(Xs)} sessions: {len(common_cols)}")
    # Optional: per-session negative subsampling BEFORE the heavy concat.
    # Keeps all positives + N×|positives| randomly-sampled negatives per
    # session. Dramatically reduces memory pressure for rare-class
    # behaviours like L_flinching where the negative class dominates by
    # 100-1000x.
    if neg_sample_ratio and neg_sample_ratio > 0:
        rng = np.random.default_rng(SEED_PRIMARY)
        Xs_sub, ys_sub, sids_sub = [], [], []
        prev_offset = 0
        new_sid_counter = 0
        for X_s, y_s in zip(Xs, ys):
            n_s = len(y_s)
            pos_idx = np.where(y_s == 1)[0]
            neg_idx = np.where(y_s == 0)[0]
            n_pos = len(pos_idx)
            target_neg = max(n_pos * neg_sample_ratio, 1000)
            if len(neg_idx) > target_neg:
                neg_sample = rng.choice(neg_idx, size=target_neg, replace=False)
            else:
                neg_sample = neg_idx
            keep = np.concatenate([pos_idx, neg_sample])
            keep.sort()  # preserve temporal order
            Xs_sub.append(X_s.iloc[keep])
            ys_sub.append(y_s[keep])
            sids_sub.extend([new_sid_counter] * len(keep))
            new_sid_counter += 1
        # Recompute sids for the subsampled data (replaces the per-row
        # `sids` list built in the outer loop, which was sized for full
        # session lengths).
        Xs = Xs_sub
        ys = ys_sub
        sids = sids_sub
        n_total = sum(len(yy) for yy in ys)
        step(f"  neg-subsampled to {neg_sample_ratio}:1 → pooled rows = "
             f"{n_total:,} (from {sum(int(meta_n) for meta_n in [len(_) for _ in [None]*0]+[X_s.shape[0] for X_s in kept_Xs]):,})")

    Xs = [X.loc[:, common_cols] for X in Xs]

    X = pd.concat(Xs, ignore_index=True)
    y = np.concatenate(ys)
    session_ids = np.array(sids, dtype=np.int32)

    # Drop columns that are all-NaN or constant across the full pool
    nunique = X.nunique(dropna=False)
    bad = nunique[nunique <= 1].index.tolist()
    if bad:
        step(f"  dropping {len(bad)} constant/all-nan columns")
        X = X.drop(columns=bad)
    # Replace remaining NaN/inf with 0 (XGBoost handles NaN natively but
    # let's be safe; also keep inf->finite to avoid XGBoost rejecting them)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    step(f"  pooled X: {X.shape}, pos rate {100*y.mean():.2f}%")
    return X, y, session_ids, snames, fpaths


# ----------------------------------------------------------------- #
# Training core
# ----------------------------------------------------------------- #

def make_xgb(scale_pos_weight: float, params: dict | None = None, seed: int = SEED_PRIMARY):
    """Build a base XGBoost classifier — defaults aligned with the
    PixelPaws GUI's `_real_training` (n_estimators=1700, lr=0.01,
    eval_metric='aucpr'). With `trim_to_last_positive=True` the pool
    is small enough that 1700 trees + lr=0.01 fits within budget.

    Early stopping uses an in-fold validation session (see
    `cv_frame_f1`). Disable by passing
    ``params={'early_stopping_rounds': None}``.
    """
    from xgboost import XGBClassifier
    p = dict(
        n_estimators=1700,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.9,
        colsample_bytree=0.2,
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
        verbosity=0,
        eval_metric="aucpr",  # matches GUI; better than logloss for imbalanced
        early_stopping_rounds=40,
    )
    if params:
        p.update(params)
    # Allow callers to explicitly disable early stopping
    if p.get("early_stopping_rounds") in (None, 0):
        p.pop("early_stopping_rounds", None)
    return XGBClassifier(**p)


def cv_oof_predict(X: pd.DataFrame, y: np.ndarray, session_ids: np.ndarray,
                   params: dict | None = None, seed: int = SEED_PRIMARY
                   ) -> tuple[np.ndarray, list[float]]:
    from sklearn.model_selection import GroupKFold
    n_groups = len(np.unique(session_ids))
    k = min(5, n_groups)
    gkf = GroupKFold(n_splits=k)
    oof = np.zeros(len(y), dtype=np.float32)
    fold_log = []
    for fi, (tr, te) in enumerate(gkf.split(X, y, groups=session_ids), 1):
        t0 = time.time()
        spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        clf = make_xgb(spw, params, seed)
        clf.fit(X.iloc[tr], y[tr])
        oof[te] = clf.predict_proba(X.iloc[te])[:, 1]
        fold_log.append(time.time() - t0)
        step(f"    fold {fi}/{k} ({fold_log[-1]:.0f}s, "
             f"train={len(tr):,} test={len(te):,})")
    return oof, fold_log


def best_threshold_f1(y: np.ndarray, oof: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import f1_score
    grid = np.linspace(0.05, 0.95, 91)
    best_t, best_f1 = 0.5, 0.0
    for t in grid:
        pred = (oof >= t).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def cv_frame_f1(X, y, session_ids, params=None, seed=SEED_PRIMARY):
    """Returns dict with cv_f1_mean/std/per_fold + oof_proba + best_thresh.

    Uses session-level GroupKFold. For each training fold, carves a 10%
    in-fold session-stratified slice for early-stopping eval. This keeps
    the test fold completely held out from any model selection.
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import f1_score
    n_groups = len(np.unique(session_ids))
    k = min(5, n_groups)
    gkf = GroupKFold(n_splits=k)
    oof = np.zeros(len(y), dtype=np.float32)
    per_fold_thresh = []
    per_fold_f1 = []
    rng = np.random.default_rng(seed)
    for fi, (tr, te) in enumerate(gkf.split(X, y, groups=session_ids), 1):
        t0 = time.time()
        spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
        clf = make_xgb(spw, params, seed)
        # Carve 1 random training session out as eval-set for early
        # stopping (session-level so we never split a session). Falls
        # back to a 5% row sample if we only have 2 training sessions.
        train_sids = np.unique(session_ids[tr])
        eval_set = None
        if len(train_sids) >= 3 and clf.get_params().get("early_stopping_rounds"):
            es_sid = int(rng.choice(train_sids))
            es_mask = (session_ids[tr] == es_sid)
            tr_keep = tr[~es_mask]
            tr_es = tr[es_mask]
            if len(tr_es) > 1000 and (y[tr_es] == 1).sum() >= 5:
                eval_set = [(X.iloc[tr_es], y[tr_es])]
                tr = tr_keep
                spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
                clf = make_xgb(spw, params, seed)
        try:
            if eval_set is not None:
                clf.fit(X.iloc[tr], y[tr], eval_set=eval_set, verbose=False)
            else:
                # No eval set → disable early stopping in this fit
                clf.set_params(early_stopping_rounds=None)
                clf.fit(X.iloc[tr], y[tr])
        except TypeError:
            # Older XGBoost API
            clf.fit(X.iloc[tr], y[tr])
        p_te = clf.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p_te
        # per-fold threshold (used only for reporting; final uses global)
        t, f1 = best_threshold_f1(y[te], p_te)
        per_fold_thresh.append(t)
        per_fold_f1.append(f1)
        n_trees = getattr(clf, "best_iteration", None) or clf.n_estimators
        step(f"    fold {fi}/{k} ({time.time()-t0:.0f}s) "
             f"thresh={t:.2f} F1={f1:.3f}  trees={n_trees}")
    global_t, global_f1 = best_threshold_f1(y, oof)
    return {
        "cv_f1_mean": float(np.mean(per_fold_f1)),
        "cv_f1_std": float(np.std(per_fold_f1, ddof=1) if len(per_fold_f1) > 1 else 0.0),
        "cv_f1_per_fold": [float(x) for x in per_fold_f1],
        "per_fold_thresh": [float(x) for x in per_fold_thresh],
        "oof_proba": oof,
        "global_best_thresh": global_t,
        "global_best_f1": global_f1,
    }


# ----------------------------------------------------------------- #
# SHAP pruning
# ----------------------------------------------------------------- #

def shap_feature_importance(X: pd.DataFrame, y: np.ndarray,
                            session_ids: np.ndarray,
                            params: dict | None = None,
                            seed: int = SEED_PRIMARY) -> pd.Series:
    """Fit one model on the full data and return mean(|SHAP value|) per
    feature."""
    import shap
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    # Disable early stopping for the SHAP fit (no eval set available)
    p = dict(params or {})
    p["early_stopping_rounds"] = None
    clf = make_xgb(spw, p, seed)
    clf.fit(X, y)
    # use tree SHAP for speed; sample to cap runtime
    n_sample = min(20000, len(X))
    if n_sample < len(X):
        idx = np.random.default_rng(seed).choice(len(X), n_sample, replace=False)
        X_sample = X.iloc[idx]
    else:
        X_sample = X
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_sample, check_additivity=False)
    if isinstance(sv, list):
        sv = sv[1]  # positive class
    abs_mean = np.abs(sv).mean(axis=0)
    return pd.Series(abs_mean, index=X.columns).sort_values(ascending=False)


def shap_prune_ladder(X, y, session_ids, params, seed,
                      budget_remaining: float):
    """Returns (best_cols, best_f1, ladder_results) — `best_cols` is the
    smallest set whose F1 is within 0.01 of the best across the ladder."""
    importance = shap_feature_importance(X, y, session_ids, params, seed)
    results = []
    for n in SHAP_LADDER:
        if budget_remaining < 60:
            step(f"    SHAP ladder: out of budget, stopping at last completed")
            break
        if n >= X.shape[1]:
            step(f"    skip ladder step {n} (>= n_features {X.shape[1]})")
            continue
        cols = importance.head(n).index.tolist()
        step(f"    SHAP ladder: trying top-{n} features")
        t0 = time.time()
        m = cv_frame_f1(X[cols], y, session_ids, params, seed)
        results.append({"n_features": n, "cv_f1_mean": m["cv_f1_mean"],
                        "best_thresh": m["global_best_thresh"]})
        budget_remaining -= (time.time() - t0)

    if not results:
        return X.columns.tolist(), None, []

    best_overall = max(r["cv_f1_mean"] for r in results)
    # smallest n where F1 >= best - 0.01
    chosen = min((r for r in results if r["cv_f1_mean"] >= best_overall - 0.01),
                 key=lambda r: r["n_features"])
    cols = importance.head(chosen["n_features"]).index.tolist()
    return cols, chosen, results


# ----------------------------------------------------------------- #
# Bout-level metrics
# ----------------------------------------------------------------- #

def to_bouts(arr: np.ndarray) -> list[tuple[int, int]]:
    """[(start, end), ...] half-open intervals where arr == 1."""
    bouts = []
    in_b = False
    s = 0
    for i, v in enumerate(arr):
        if v and not in_b:
            in_b = True
            s = i
        elif not v and in_b:
            in_b = False
            bouts.append((s, i))
    if in_b:
        bouts.append((s, len(arr)))
    return bouts


def bout_f1(pred: np.ndarray, true: np.ndarray, iou_thresh: float = 0.5
            ) -> dict:
    p_b = to_bouts(pred)
    t_b = to_bouts(true)
    if not p_b and not t_b:
        return {"bout_f1": 1.0, "bout_precision": 1.0, "bout_recall": 1.0,
                "n_pred": 0, "n_true": 0, "n_matched": 0}
    matched_p = set()
    matched_t = set()
    for i, (ps, pe) in enumerate(p_b):
        for j, (ts, te) in enumerate(t_b):
            inter = max(0, min(pe, te) - max(ps, ts))
            union = max(pe, te) - min(ps, ts)
            iou = inter / union if union > 0 else 0
            if iou >= iou_thresh:
                matched_p.add(i)
                matched_t.add(j)
    tp = len(matched_p)  # predicted bouts that matched
    prec = tp / max(len(p_b), 1)
    rec = len(matched_t) / max(len(t_b), 1)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return {"bout_f1": float(f1), "bout_precision": float(prec),
            "bout_recall": float(rec), "n_pred": len(p_b),
            "n_true": len(t_b), "n_matched": tp}


def postprocess_predictions(pred: np.ndarray,
                            min_bout: int = 3,
                            max_gap: int = 5) -> np.ndarray:
    """Apply standard PixelPaws bout post-processing to raw frame
    predictions:
      1. Merge gaps: any run of 0s shorter than (or equal to) max_gap
         that sits between two 1-runs becomes 1s.
      2. Drop short bouts: any 1-run shorter than min_bout becomes 0s.

    This matches what the GUI applies at predict time. Frame F1 is the
    raw classifier metric; bout F1 should always be computed after this
    smoothing to reflect what users actually see.
    """
    p = pred.astype(np.int8).copy()
    n = len(p)
    if n == 0:
        return p
    # 1) Fill short gaps
    if max_gap > 0:
        i = 0
        while i < n:
            if p[i] == 0:
                gap_start = i
                while i < n and p[i] == 0:
                    i += 1
                gap_end = i  # exclusive
                gap_len = gap_end - gap_start
                # Only fill if BOTH sides have a 1-run (not boundary gaps)
                if (gap_len <= max_gap
                        and gap_start > 0 and p[gap_start - 1] == 1
                        and gap_end < n and p[gap_end] == 1):
                    p[gap_start:gap_end] = 1
            else:
                i += 1
    # 2) Drop short bouts
    if min_bout > 0:
        i = 0
        while i < n:
            if p[i] == 1:
                b_start = i
                while i < n and p[i] == 1:
                    i += 1
                if (i - b_start) < min_bout:
                    p[b_start:i] = 0
            else:
                i += 1
    return p


def per_session_bout_metrics(oof: np.ndarray, y: np.ndarray,
                             session_ids: np.ndarray, thresh: float,
                             min_bout: int = 3, max_gap: int = 5
                             ) -> dict:
    """Per-session bout metrics with min_bout + max_gap post-processing
    applied to the frame predictions (matches GUI behaviour)."""
    from scipy.stats import pearsonr
    raw_pred = (oof >= thresh).astype(np.int8)
    sids = np.unique(session_ids)
    f1s, p_counts, t_counts, dur_errs = [], [], [], []
    for sid in sids:
        m = session_ids == sid
        pred = postprocess_predictions(raw_pred[m], min_bout=min_bout,
                                       max_gap=max_gap)
        b = bout_f1(pred, y[m])
        f1s.append(b["bout_f1"])
        p_counts.append(b["n_pred"])
        t_counts.append(b["n_true"])
        # mean duration error in seconds
        p_durs = [(e - s) / FPS_DEFAULT for s, e in to_bouts(pred)]
        t_durs = [(e - s) / FPS_DEFAULT for s, e in to_bouts(y[m])]
        if p_durs and t_durs:
            dur_errs.append(abs(np.mean(p_durs) - np.mean(t_durs)))
    r = float("nan")
    if len(p_counts) > 2 and np.std(p_counts) > 0 and np.std(t_counts) > 0:
        try:
            r, _ = pearsonr(p_counts, t_counts)
        except Exception:
            pass
    return {
        "bout_f1_mean": float(np.mean(f1s)) if f1s else float("nan"),
        "bout_f1_std":  float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
        "bout_count_pearson_r": float(r),
        "bout_count_pred_per_session":  [int(x) for x in p_counts],
        "bout_count_true_per_session":  [int(x) for x in t_counts],
        "mean_bout_duration_error_sec": float(np.mean(dur_errs)) if dur_errs else float("nan"),
    }


# ----------------------------------------------------------------- #
# Calibration
# ----------------------------------------------------------------- #

def fit_isotonic_calibrator(oof: np.ndarray, y: np.ndarray):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof, y)
    return iso


def brier(oof, y):
    return float(np.mean((oof - y) ** 2))


# ----------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("behaviour", help="canonical behaviour name")
    ap.add_argument("--budget-min", type=int, default=90)
    ap.add_argument("--skip-seed-stability", action="store_true")
    ap.add_argument("--exclude-session", action="append", default=[],
                    help="Observation/video-stem to skip (repeatable)")
    ap.add_argument("--prefer-hash", default=None,
                    help="When >1 feature pkl per session, prefer the one "
                         "whose filename contains this hash (e.g. 8aed1c22)")
    ap.add_argument("--neg-sample-ratio", type=int, default=None,
                    help="Per-session, keep all positives + N×|pos| random "
                         "negatives (min 1000). Reduces memory + speeds CV "
                         "for rare-class behaviours.")
    ap.add_argument("--trim-to-last-positive", action="store_true", default=True,
                    help="Default True (matches GUI). Trim each session's data to "
                         "[0:last_pos_frame+1] to drop unlabeled tail. Disable "
                         "with --no-trim-to-last-positive.")
    ap.add_argument("--no-trim-to-last-positive", action="store_false",
                    dest="trim_to_last_positive",
                    help="Disable trim-to-last-positive (keep full session data).")
    args = ap.parse_args()

    behaviour = args.behaviour
    budget_sec = args.budget_min * 60
    t_start = time.time()

    discord_milestone(
        f":gear: **Training `{behaviour}`** (Phase C, budget {args.budget_min} min)..."
    )

    step(f"== Loading dataset for '{behaviour}' ==")
    if args.exclude_session:
        step(f"  excluding sessions: {args.exclude_session}")
    if args.prefer_hash:
        step(f"  preferring feature hash: {args.prefer_hash}")
    X, y, session_ids, snames, fpaths = load_behaviour_dataset(
        behaviour,
        exclude_sessions=args.exclude_session,
        prefer_hash=args.prefer_hash,
        neg_sample_ratio=args.neg_sample_ratio,
        trim_to_last_positive=args.trim_to_last_positive,
    )

    pos_total = int(y.sum())
    pos_rate = float(y.mean())
    n_sessions = len(np.unique(session_ids))

    step(f"\n== Baseline CV (defaults, seed={SEED_PRIMARY}) ==")
    baseline = cv_frame_f1(X, y, session_ids)
    step(f"  baseline CV F1 = {baseline['cv_f1_mean']:.4f} "
         f"± {baseline['cv_f1_std']:.4f}  "
         f"(global thresh={baseline['global_best_thresh']:.2f}, "
         f"global F1={baseline['global_best_f1']:.4f})")

    current = {
        "params": None,
        "cols": list(X.columns),
        "metrics": baseline,
        "stage": "baseline",
    }

    elapsed = lambda: time.time() - t_start
    remaining = lambda: budget_sec - elapsed()

    # SHAP pruning if below 0.90
    shap_ladder = []
    if baseline["cv_f1_mean"] < SKIP_OPTUNA_F1 and remaining() > 600:
        step(f"\n== SHAP pruning ladder (CV F1 = {baseline['cv_f1_mean']:.4f} < "
             f"{SKIP_OPTUNA_F1}) ==")
        cols, chosen, shap_ladder = shap_prune_ladder(
            X, y, session_ids, None, SEED_PRIMARY, remaining()
        )
        if chosen and chosen["cv_f1_mean"] > current["metrics"]["cv_f1_mean"]:
            step(f"  SHAP picked top-{chosen['n_features']} (F1={chosen['cv_f1_mean']:.4f}); "
                 "re-running CV with pruned set to obtain OOF probs")
            current["cols"] = cols
            current["metrics"] = cv_frame_f1(X[cols], y, session_ids)
            current["stage"] = f"shap_pruned_{chosen['n_features']}"
    else:
        step("  skipping SHAP ladder (F1 already ≥0.90 or budget exhausted)")

    # Optuna only if still below gate
    optuna_info = None
    if current["metrics"]["cv_f1_mean"] < GATE_F1 and remaining() > 600:
        step(f"\n== Optuna tuning (CV F1 = {current['metrics']['cv_f1_mean']:.4f} "
             f"< {GATE_F1}, budget left = {remaining()/60:.0f} min) ==")
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            X_cur = X[current["cols"]]

            def objective(trial):
                p = {
                    "max_depth": trial.suggest_int("max_depth", 4, 8),
                    "learning_rate": trial.suggest_float("learning_rate",
                                                        0.005, 0.05, log=True),
                    "n_estimators": trial.suggest_categorical(
                        "n_estimators", [800, 1200, 1700, 2400]),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree", 0.1, 0.4),
                }
                m = cv_frame_f1(X_cur, y, session_ids, p, SEED_PRIMARY)
                return m["cv_f1_mean"]

            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=42))
            n_trials = 30
            study.optimize(
                objective, n_trials=n_trials,
                timeout=int(min(remaining() - 120, 3000)),
            )
            optuna_info = {
                "best_params": study.best_params,
                "best_value": float(study.best_value),
                "n_trials_completed": len(study.trials),
            }
            step(f"  Optuna best F1 = {study.best_value:.4f}  "
                 f"params = {study.best_params}")
            if study.best_value > current["metrics"]["cv_f1_mean"]:
                step("  Optuna improved; re-running CV with best params")
                current["params"] = study.best_params
                current["metrics"] = cv_frame_f1(X[current["cols"]], y,
                                                  session_ids, study.best_params)
                current["stage"] = "optuna"
        except Exception as e:
            step(f"  ! Optuna failed: {e}")
    else:
        step("  skipping Optuna (F1 already ≥0.85 or budget exhausted)")

    # Final metrics
    final_metrics = current["metrics"]
    final_thresh = final_metrics["global_best_thresh"]
    final_f1 = final_metrics["global_best_f1"]
    step(f"\n== Final CV F1 = {final_metrics['cv_f1_mean']:.4f} "
         f"± {final_metrics['cv_f1_std']:.4f}  "
         f"(stage={current['stage']}) ==")

    # Bout metrics
    step("\n== Bout-level evaluation ==")
    bout = per_session_bout_metrics(
        final_metrics["oof_proba"], y, session_ids, final_thresh
    )
    step(f"  bout F1 = {bout['bout_f1_mean']:.4f} ± {bout['bout_f1_std']:.4f}  "
         f"bout-count Pearson r = {bout['bout_count_pearson_r']:.3f}  "
         f"mean dur err = {bout['mean_bout_duration_error_sec']:.2f}s")

    # Calibration
    step("\n== Calibration ==")
    iso = fit_isotonic_calibrator(final_metrics["oof_proba"], y)
    brier_before = brier(final_metrics["oof_proba"], y)
    cal_proba = iso.transform(final_metrics["oof_proba"])
    brier_after = brier(cal_proba, y)
    cal_thresh, cal_f1 = best_threshold_f1(y, cal_proba)
    step(f"  Brier {brier_before:.4f} -> {brier_after:.4f}  "
         f"(calibrated F1 at thresh {cal_thresh:.2f} = {cal_f1:.4f})")

    # Seed stability
    seed_f1s = []
    if not args.skip_seed_stability and remaining() > 600:
        step("\n== Seed stability (4 extra seeds) ==")
        for s in SEED_STABILITY:
            if remaining() < 120:
                step(f"  out of budget; skipping seed {s}")
                break
            step(f"  seed {s}")
            m = cv_frame_f1(X[current["cols"]], y, session_ids,
                            current["params"], s)
            seed_f1s.append({"seed": s, "cv_f1": m["cv_f1_mean"]})
        if seed_f1s:
            all_f1s = [final_metrics["cv_f1_mean"]] + [s["cv_f1"] for s in seed_f1s]
            step(f"  seed F1 std across {len(all_f1s)} seeds = "
                 f"{np.std(all_f1s, ddof=1):.4f}")

    # ----- Train final model on all data -----
    step("\n== Fitting final model on full pooled data ==")
    final_X = X[current["cols"]]
    spw_full = (y == 0).sum() / max((y == 1).sum(), 1)
    # No eval set available → disable early stopping for the final fit
    final_params = dict(current["params"] or {})
    final_params["early_stopping_rounds"] = None
    final_model = make_xgb(spw_full, final_params, SEED_PRIMARY)
    t0 = time.time()
    final_model.fit(final_X, y)
    step(f"  final fit done in {time.time()-t0:.0f}s")

    # ----- Decide ship / drop -----
    ship_status = "SHIP" if final_metrics["cv_f1_mean"] >= GATE_F1 else "DROP"

    # Default post-process knobs (use the GUI defaults; user can tune later)
    out = {
        "Behavior_type": behaviour,
        "clf_model": final_model,
        "calibrator": iso,
        "selected_feature_cols": current["cols"],
        "best_thresh": float(final_thresh),
        "min_bout": 3,
        "min_after_bout": 1,
        "max_gap": 5,
        "ui_min_bout": 3,
        "ui_min_after_bout": 1,
        "ui_max_gap": 5,
        "expected_fps": FPS_DEFAULT,
        # CV metrics
        "cv_f1_scores": final_metrics["cv_f1_per_fold"],
        "mean_cv_f1": final_metrics["cv_f1_mean"],
        "std_cv_f1": final_metrics["cv_f1_std"],
        "oof_best_f1": final_metrics["global_best_f1"],
        "oof_proba": final_metrics["oof_proba"],
        # Bout metrics
        "bout_metrics": bout,
        # Calibration
        "brier_uncalibrated": brier_before,
        "brier_calibrated": brier_after,
        # Optuna + SHAP
        "optuna_best_params": optuna_info["best_params"] if optuna_info else None,
        "optuna_info": optuna_info,
        "shap_ladder": shap_ladder,
        "stage": current["stage"],
        # Provenance
        "training_sessions": snames,
        "training_feature_pkls": [str(p) for p in fpaths],
        "n_features": len(current["cols"]),
        "n_train_sessions": int(n_sessions),
        "n_train_frames": int(len(y)),
        "n_pos_frames": int(pos_total),
        "pos_rate": pos_rate,
        "seed": SEED_PRIMARY,
        "seed_stability_f1s": seed_f1s,
        "training_seconds": int(time.time() - t_start),
        # Feature schema
        "feature_hash": "8aed1c22",
        "bp_pixbrt_list": ["hrpaw", "hlpaw", "snout"],
        "square_size": [40, 40, 40],
        "pix_threshold": 0.3,
        "include_optical_flow": True,
        "bp_optflow_list": ["hrpaw", "hlpaw", "snout"],
    }

    out_path = OUT_CLF_DIR / f"classifier_{behaviour}.pkl"
    joblib.dump(out, out_path)
    step(f"\n== {ship_status}: wrote {out_path.name} ==")

    # training_metrics.json (compact, no model)
    perf_dir = OUT_PERF_DIR / behaviour
    perf_dir.mkdir(exist_ok=True)
    metrics_json = {
        k: v for k, v in out.items()
        if k not in ("clf_model", "calibrator", "oof_proba",
                     "selected_feature_cols", "training_feature_pkls",
                     "training_sessions")
    }
    metrics_json["selected_feature_cols_count"] = len(current["cols"])
    metrics_json["selected_feature_cols_top20"] = current["cols"][:20]
    (perf_dir / "training_metrics.json").write_text(
        json.dumps(metrics_json, indent=2, default=str)
    )
    # also persist oof_proba separately for plot regeneration
    np.savez(perf_dir / "oof.npz",
             oof_proba=final_metrics["oof_proba"], y=y,
             session_ids=session_ids)
    step(f"  wrote {perf_dir / 'training_metrics.json'}")

    discord_milestone(
        f":tada: **`{behaviour}` trained** (stage: {current['stage']}, "
        f"{int(time.time()-t_start)//60} min)\n"
        f"  CV F1 = {final_metrics['cv_f1_mean']:.4f} ± {final_metrics['cv_f1_std']:.4f}  "
        f"(thresh {final_thresh:.2f})\n"
        f"  Bout F1 = {bout['bout_f1_mean']:.4f}, "
        f"bout-count r = {bout['bout_count_pearson_r']:.3f}\n"
        f"  Brier {brier_before:.3f} → {brier_after:.3f} (after calibration)\n"
        f"  → **{ship_status}**"
    )
    return 0 if ship_status == "SHIP" else 1


if __name__ == "__main__":
    raise SystemExit(main())
