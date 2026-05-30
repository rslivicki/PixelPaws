"""
active_learning_engine.py — PixelPaws Active-Learning engine (headless, Tk-free)
================================================================================
The "brains" of active learning, decoupled from the GUI so it is unit-testable
and scriptable (see scripts/research/run_active_learning.py). The Tk tab in
active_learning_v2.py drives this engine; the labeling UI stays in the GUI layer.

Design (research-grounded — see plan velvety-meandering-candy.md):
- Score = calibrated binary entropy (boosted-tree probs are over-confident, so a
  fixed 0.5-distance threshold is unreliable) [QUERY §10,§1].
- Query BOUTS not frames; exploit temporal coherence with a min frame-gap [QUERY §5].
- Batch selection = uncertainty x diversity (uncertainty-weighted k-means++ over
  bout-mean feature vectors, a BADGE analog for trees) x class-balance quota
  (so the rare class is never starved) [QUERY §4,§6, A-SOiD §c].
- Cross-video selection is GLOBAL: one pooled candidate set across all sessions,
  pick the globally most-informative; present grouped-by-video downstream.

Pure-Python: numpy required; sklearn optional (calibration / f1); no tkinter.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Sequence

try:
    from sklearn.metrics import f1_score as _f1_score
    from sklearn.isotonic import IsotonicRegression
    _SKLEARN = True
except Exception:
    _SKLEARN = False


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def binary_entropy(p: np.ndarray) -> np.ndarray:
    """H(p) = -p log2 p - (1-p) log2(1-p); peaks (=1) at p=0.5, 0 at p in {0,1}."""
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def fit_calibrator(proba_holdout: np.ndarray, y_holdout: np.ndarray):
    """Isotonic calibration map fit on a held-out slice. Returns a callable
    proba->proba (identity if sklearn missing or holdout too small/degenerate)."""
    proba_holdout = np.asarray(proba_holdout, float)
    y_holdout = np.asarray(y_holdout, int)
    if not _SKLEARN or len(y_holdout) < 20 or len(np.unique(y_holdout)) < 2:
        return (lambda p: np.asarray(p, float))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(proba_holdout, y_holdout)
    return (lambda p: iso.predict(np.clip(np.asarray(p, float), 0, 1)))


# --------------------------------------------------------------------------- #
# Candidate bouts
# --------------------------------------------------------------------------- #
@dataclass(eq=False)   # identity-based: feat_mean is an ndarray, so value-eq is unsafe
class EngineBout:
    session_idx: int
    start: int            # core start frame (within its session)
    end: int              # core end frame (inclusive)
    score: float          # mean entropy over the core (selection key)
    peak_score: float     # max entropy in core (tiebreak)
    pred_pos: bool        # model's majority guess over the core (>=0.5)
    feat_mean: np.ndarray = field(default=None, repr=False)  # bout-mean feature vector


def find_candidate_bouts(labels: np.ndarray,
                         scores: np.ndarray,
                         probas: np.ndarray,
                         session_idx: int = 0,
                         min_bout_frames: int = 5,
                         max_bout_frames: int = 300,
                         features: Optional[np.ndarray] = None) -> List[EngineBout]:
    """Group consecutive UNLABELED frames (labels < 0) into candidate bouts,
    drop runs < min_bout_frames, subdivide runs > max_bout_frames, and attach
    mean entropy / peak entropy / predicted class / mean feature vector."""
    labels = np.asarray(labels); scores = np.asarray(scores, float); probas = np.asarray(probas, float)
    n = len(labels)
    unl = labels < 0
    out: List[EngineBout] = []
    i = 0
    while i < n:
        if not unl[i]:
            i += 1; continue
        j = i
        while j < n and unl[j]:
            j += 1
        run_s, run_e = i, j - 1  # inclusive run of unlabeled frames
        i = j
        if run_e - run_s + 1 < min_bout_frames:
            continue
        # subdivide long runs into non-overlapping windows
        s = run_s
        while s <= run_e:
            e = min(s + max_bout_frames - 1, run_e)
            sc = scores[s:e + 1]
            out.append(EngineBout(
                session_idx=session_idx, start=s, end=e,
                score=float(sc.mean()), peak_score=float(sc.max()),
                pred_pos=bool(np.mean(probas[s:e + 1] >= 0.5) >= 0.5),
                feat_mean=(features[s:e + 1].mean(axis=0) if features is not None else None)))
            s = e + 1
    return out


# --------------------------------------------------------------------------- #
# Batch selection: uncertainty x diversity x class-balance, global, temporal-gap
# --------------------------------------------------------------------------- #
def _weighted_kpp(feats: np.ndarray, weights: np.ndarray, k: int, rng: np.random.RandomState) -> List[int]:
    """Uncertainty-weighted k-means++ seeding (BADGE analog): pick k diverse rows,
    seeding probability ∝ weight * D(x)^2 (distance to nearest already-picked)."""
    n = len(feats)
    if k >= n:
        return list(range(n))
    w = np.asarray(weights, float).clip(1e-12)
    # standardize features so distance isn't dominated by high-variance columns
    mu, sd = feats.mean(0), feats.std(0); sd[sd < 1e-9] = 1.0
    Z = (feats - mu) / sd
    first = int(rng.choice(n, p=w / w.sum()))
    picked = [first]
    d2 = ((Z - Z[first]) ** 2).sum(1)
    while len(picked) < k:
        prob = w * d2
        tot = prob.sum()
        if not np.isfinite(tot) or tot <= 0:
            remaining = [i for i in range(n) if i not in picked]
            picked.append(int(rng.choice(remaining)));
        else:
            nxt = int(rng.choice(n, p=prob / tot))
            if nxt in picked:
                remaining = [i for i in range(n) if i not in picked]
                nxt = int(remaining[int(np.argmax(d2[remaining]))])
            picked.append(nxt)
        d2 = np.minimum(d2, ((Z - Z[picked[-1]]) ** 2).sum(1))
    return picked


def select_batch(candidates: List[EngineBout],
                 batch_size: int = 20,
                 min_frame_gap: int = 30,
                 pos_quota_frac: float = 0.4,
                 diversity: bool = True,
                 seed: int = 0) -> List[EngineBout]:
    """Pick a globally-informative, internally-diverse, class-balanced, temporally
    de-duplicated batch from a pooled (multi-session) candidate list.

    1. temporal de-dup: no two selected bouts within `min_frame_gap` in the same session
    2. class quota: guarantee >= pos_quota_frac*batch likely-positive bouts (rare class)
    3. diversity: uncertainty-weighted k-means++ over bout-mean feature vectors
    """
    if not candidates:
        return []
    rng = np.random.RandomState(seed)

    def _temporal_ok(chosen: List[EngineBout], c: EngineBout) -> bool:
        for o in chosen:
            if o.session_idx == c.session_idx and abs(o.start - c.start) < min_frame_gap:
                return False
        return True

    have_feats = all(c.feat_mean is not None for c in candidates)
    pos_quota = int(round(pos_quota_frac * batch_size))
    chosen: List[EngineBout] = []

    def _fill_from(pool: List[EngineBout], limit: int):
        if not pool or limit <= 0:
            return
        if diversity and have_feats and len(pool) > limit:
            feats = np.stack([c.feat_mean for c in pool])
            w = np.array([max(c.score, 1e-6) for c in pool])
            order = _weighted_kpp(feats, w, min(limit * 3, len(pool)), rng)
            cand_order = [pool[i] for i in order]
        else:
            cand_order = sorted(pool, key=lambda c: (-c.score, -c.peak_score))
        added = 0
        for c in cand_order:
            if added >= limit or len(chosen) >= batch_size:
                break
            if any(c is x for x in chosen) or not _temporal_ok(chosen, c):
                continue
            chosen.append(c); added += 1

    # 1) class quota: positives first (ranked/diverse), then negatives, then fill
    positives = [c for c in candidates if c.pred_pos]
    negatives = [c for c in candidates if not c.pred_pos]
    _fill_from(positives, pos_quota)
    _fill_from(negatives, batch_size - len(chosen))
    _fill_from(candidates, batch_size - len(chosen))  # top-up ignoring class if short
    return chosen[:batch_size]


# --------------------------------------------------------------------------- #
# Convergence / stopping
# --------------------------------------------------------------------------- #
def f1_plateaued(history: Sequence[float], k: int = 3, eps: float = 0.01) -> bool:
    """True if the last k held-out F1 values are within eps of each other."""
    h = [x for x in history if x is not None]
    if len(h) < k:
        return False
    last = h[-k:]
    return (max(last) - min(last)) <= eps


def pool_drained(probas: np.ndarray, labels: np.ndarray, confidence_threshold: float) -> bool:
    """True if no UNLABELED frame is still within `confidence_threshold` of 0.5
    (A-SOiD 'dwindling low-confidence samples'). Only counts unlabeled frames so a
    high threshold over an all-labeled set doesn't read as false-converged."""
    probas = np.asarray(probas, float); labels = np.asarray(labels)
    unl = labels < 0
    if not unl.any():
        return True
    return bool(np.sum(np.abs(probas[unl] - 0.5) * 2 < confidence_threshold) == 0)


# --------------------------------------------------------------------------- #
# Headless engine over one or more sessions
# --------------------------------------------------------------------------- #
class ActiveLearningEngine:
    """Headless AL over a pool of sessions. Each session = dict(features=2D float
    array, labels=1D int array with -1 = unlabeled). Selection is GLOBAL across
    sessions; labels are applied per (session_idx, start, end)."""

    def __init__(self, sessions: List[Dict], feature_cols: Optional[Sequence[str]] = None,
                 min_bout_frames: int = 5, max_bout_frames: int = 300,
                 min_frame_gap: int = 30, holdout_frac: float = 0.2, seed: int = 0):
        self.sessions = [{'features': np.asarray(s['features'], np.float32),
                          'labels': np.asarray(s['labels']).astype(int)} for s in sessions]
        self.feature_cols = list(feature_cols) if feature_cols is not None else None
        self.min_bout_frames = min_bout_frames
        self.max_bout_frames = max_bout_frames
        self.min_frame_gap = min_frame_gap
        self.holdout_frac = holdout_frac
        self.seed = seed
        self.f1_history: List[float] = []
        self.iteration = 0

    # ---- labeled-data helpers ----
    def _pool_labeled(self):
        Xs, ys, sid = [], [], []
        for k, s in enumerate(self.sessions):
            m = s['labels'] >= 0
            if m.any():
                Xs.append(s['features'][m]); ys.append(s['labels'][m])
                sid.append(np.full(int(m.sum()), k, int))
        if not Xs:
            return (np.empty((0, self.sessions[0]['features'].shape[1]), np.float32),
                    np.array([], int), np.array([], int))
        return np.concatenate(Xs), np.concatenate(ys), np.concatenate(sid)

    def n_labeled(self) -> int:
        return int(sum(int((s['labels'] >= 0).sum()) for s in self.sessions))

    def n_positive(self) -> int:
        return int(sum(int((s['labels'] == 1).sum()) for s in self.sessions))

    # ---- training (pluggable; default XGBoost) ----
    def _fit(self, X, y, train_fn: Optional[Callable] = None):
        if len(np.unique(y)) < 2:
            raise ValueError("Need both classes labeled before training.")
        if train_fn is not None:
            return train_fn(X, y)
        from xgboost import XGBClassifier
        spw = float((y == 0).sum() / max((y == 1).sum(), 1))
        m = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          scale_pos_weight=spw, random_state=self.seed, verbosity=0)
        m.fit(X, y)
        return m

    def train(self, train_fn: Optional[Callable] = None):
        X, y, _ = self._pool_labeled()
        return self._fit(X, y, train_fn)

    def _align(self, model, X):
        """Align a feature matrix to an EXTERNAL model's feature_names_in_ (by the
        session feature_cols) — needed when warm-starting from a pre-trained / SHAP-
        pruned classifier. Identity for engine-trained models (numpy, no names)."""
        names = getattr(model, "feature_names_in_", None)
        if names is None or self.feature_cols is None:
            return X
        names = list(names)
        if list(self.feature_cols) == names:
            return X
        import pandas as pd
        missing = [c for c in names if c not in set(self.feature_cols)]
        if len(missing) > max(5, int(0.10 * len(names))):
            raise ValueError(f"init classifier needs {len(missing)}/{len(names)} features absent "
                             f"from the feature cache (schema mismatch); aborting.")
        return pd.DataFrame(X, columns=self.feature_cols).reindex(
            columns=names, fill_value=0.0).values.astype(np.float32)

    # ---- scoring + candidate generation across all sessions ----
    def score_and_candidates(self, model=None, calibrator: Optional[Callable] = None,
                             batch_size: int = 20, pos_quota_frac: float = 0.4,
                             diversity: bool = True,
                             probas_by_session: Optional[List[np.ndarray]] = None) -> List[EngineBout]:
        """Score every session and return a globally-selected batch.
        Provide EITHER `model` (engine reindexes the cache to its feature_names_in_)
        OR `probas_by_session` (per-session P(positive) arrays) — use the latter to
        warm-start from a pre-trained classifier whose features need
        augment_features_post_cache first (the engine stays agnostic to that)."""
        cal = calibrator or (lambda p: p)
        pooled: List[EngineBout] = []
        for k, s in enumerate(self.sessions):
            if probas_by_session is not None:
                p = cal(np.asarray(probas_by_session[k], dtype=float))
            else:
                p = cal(model.predict_proba(self._align(model, s['features']))[:, 1])
            ent = binary_entropy(p)
            pooled += find_candidate_bouts(
                s['labels'], ent, p, session_idx=k,
                min_bout_frames=self.min_bout_frames, max_bout_frames=self.max_bout_frames,
                features=s['features'])
        return select_batch(pooled, batch_size=batch_size, min_frame_gap=self.min_frame_gap,
                            pos_quota_frac=pos_quota_frac, diversity=diversity, seed=self.seed + self.iteration)

    # ---- apply labels for a batch: {(session_idx, start, end): label} ----
    # label may be a scalar (whole bout = one class) OR a per-frame 0/1 array of
    # length (end-start+1) when the labeler marks an exact sub-segment.
    def apply_labels(self, decisions: Dict):
        for (sidx, s, e), lab in decisions.items():
            arr = self.sessions[int(sidx)]['labels']
            s = int(s); e = min(int(e), len(arr) - 1)
            val = np.asarray(lab)
            if val.ndim == 0:
                arr[s:e + 1] = int(val)
            else:
                arr[s:e + 1] = val[:e + 1 - s].astype(int)

    # ---- honest held-out F1 via group-aware CV out-of-fold predictions ----
    def evaluate(self, train_fn: Optional[Callable] = None) -> Dict:
        """Group-aware CV-OOF F1 + positive-class recall. Groups = session, so a
        session's frames never straddle a fold (no leakage); this replaces the
        optimistic resubstitution estimate."""
        X, y, sid = self._pool_labeled()
        if len(np.unique(y)) < 2 or not _SKLEARN:
            return {'f1': None, 'pos_recall': None, 'n': int(len(y))}
        from sklearn.model_selection import GroupKFold, StratifiedKFold
        n_splits = min(3, int((y == 1).sum()), int((y == 0).sum()))
        if n_splits < 2:
            return {'f1': None, 'pos_recall': None, 'n': int(len(y)), 'note': 'too few for CV'}
        n_groups = int(len(np.unique(sid)))
        oof = np.full(len(y), np.nan)
        if n_groups >= n_splits:
            sp = GroupKFold(n_splits=min(n_splits, n_groups)).split(X, y, groups=sid)
        else:
            sp = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed).split(X, y)
        for tr, va in sp:
            m = self._fit(X[tr], y[tr], train_fn)
            oof[va] = m.predict_proba(X[va])[:, 1]
        pred = (oof >= 0.5).astype(int)
        f1 = float(_f1_score(y, pred, zero_division=0))
        pos_recall = float((pred[y == 1] == 1).mean()) if (y == 1).any() else None
        return {'f1': f1, 'pos_recall': pos_recall, 'n': int(len(y))}

    def should_stop(self, model, calibrator=None, confidence_threshold: float = 0.3,
                    max_iters: int = 20) -> Optional[str]:
        if self.iteration >= max_iters:
            return f"hard cap ({max_iters} iterations)"
        if f1_plateaued(self.f1_history, k=3, eps=0.01):
            return "held-out F1 plateaued (last 3 within 0.01)"
        all_probas = [(calibrator or (lambda p: p))(model.predict_proba(s['features'])[:, 1])
                      for s in self.sessions]
        if all(pool_drained(p, s['labels'], confidence_threshold)
               for p, s in zip(all_probas, self.sessions)):
            return "uncertain-bout pool drained"
        return None
