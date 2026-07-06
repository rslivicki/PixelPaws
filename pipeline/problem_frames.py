"""Problem-frame active learning — extract frames where DLC pose TRACKING is
unreliable and/or the behavior CLASSIFIER is uncertain, prefill them with the
current model's keypoints, and drop them into the standard labeled-data candidate
folders the existing Tkinter labeler consumes. The user then corrects the keypoints
and (optionally) retrains — closing the loop that fixes phantom scratch/lick calls.

Two INDEPENDENT budget buckets (see ExtractConfig.tracking_frac):
  A) tracking      — reuses pose_filter.filter_pose()'s per-frame flagged_mask
                     (likelihood<p_cut, SimBA location "teleport", velocity jump —
                     hindpaws excluded from the velocity gate by design) plus the
                     AOW "present-but-hindpaw-uncertain" + "lowest-mean-likelihood"
                     buckets. Empty-cage frames are gated out.
  B) classifier    — near-threshold uncertainty (|p-0.5|*2 < confidence_threshold)
                     from predict_with_xgboost; or, in classifier_mode="positive",
                     tracking-flagged frames that co-occur with a confident positive
                     call (teleport -> phantom behavior), targeting FP-causing frames.

Output per session = <ldir>\<session>\ : img<frame:06d>.png + prefilled
CollectedData_<scorer>.h5/.csv (flat single-level index, values = current model x/y)
+ marker _scratch_to_label.txt (+ optional overlay/ + manifest.csv). Scorer/bodyparts
are project-parameterized (e.g. SlivickiR_WangH), NOT hardcoded to AlexZ.

Core is pure pandas/numpy/cv2 (imports pose_filter as a sibling); the classifier
bucket lazy-imports prediction_pipeline only when used. Importable in-process by the
GUI, and runnable as a CLI for headless testing / per-cohort batch use.
"""
from __future__ import annotations
import os, sys, glob, json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))   # pose_filter beside us
import pose_filter

BPS_DEFAULT = ["tailtip", "tailbase", "centroid", "neck", "snout",
               "hlpaw", "hrpaw", "flpaw", "frpaw"]
MARKER = "_scratch_to_label.txt"
BP_COLOR = {"snout": (0, 0, 255), "frpaw": (0, 165, 255), "flpaw": (0, 255, 255),
            "hrpaw": (0, 255, 0), "hlpaw": (255, 255, 0), "neck": (255, 0, 0),
            "tailbase": (255, 0, 255), "centroid": (200, 200, 200), "tailtip": (128, 128, 128)}


@dataclass
class ExtractConfig:
    scorer: str                          # target project's labeling scorer, e.g. "SlivickiR_WangH"
    ldir: str                            # <project>\labeled-data (destination root)
    bps: list = field(default_factory=lambda: list(BPS_DEFAULT))
    # buckets
    use_tracking: bool = True
    use_classifier: bool = False
    classifier_mode: str = "uncertain"   # "uncertain" | "positive"
    behavior: str | None = None          # label only; which classifier this is
    # budget
    budget: int = 50
    tracking_frac: float = 0.5           # split of budget: tracking vs classifier
    max_per_video: int = 12
    gap: int = 300                       # min frame spacing within a session (~5 s @60fps)
    # tracking-bucket params
    p_present: float = 0.5
    n_present_min: int = 2
    filter_kw: dict = field(default_factory=dict)   # override pose_filter DEFAULTS
    # classifier-bucket params
    confidence_threshold: float = 0.30   # near-threshold band |p-0.5|*2 < this
    prob_lo: float = 0.50                # confident-positive floor (positive mode)
    thr: float = 0.5                     # classifier decision threshold (positive mode)
    win: int = 10                        # +/- window linking a teleport to a call
    # output
    overlays: bool = True
    fresh: bool = False                  # overwrite an existing marked folder (default: refuse)


@dataclass
class Session:
    name: str
    video: str
    dlc: str
    features: str | None = None


# ----------------------------------------------------------------------------- pose helpers
def load_pose(dlc_path: str) -> pd.DataFrame:
    return pd.read_hdf(dlc_path)


def _scorer_of(df) -> str:
    return df.columns.get_level_values(0)[0]


def _lik_stack(df, bps):
    sc = _scorer_of(df)
    return np.stack([df[(sc, bp, "likelihood")].to_numpy(dtype=float) for bp in bps], 1)  # (F, nbp)


def spread(cand, n, gap, taken):
    """Greedily keep up to n frames from ranked `cand`, each >gap from all `taken`+kept."""
    out = []
    seen = list(taken)
    for f in cand:
        f = int(f)
        if all(abs(f - c) > gap for c in seen + out):
            out.append(f)
        if len(out) >= n:
            break
    return out


# ----------------------------------------------------------------------------- tracking bucket
def rank_tracking(df, cfg: ExtractConfig):
    """Return a priority-ordered candidate frame list from pose-quality signals.
    Order: (1) pose_filter location/velocity 'confidently-wrong' flags, most-flagged first;
           (2) present + hindpaw-most-uncertain (AOW bucket A);
           (3) lowest mean likelihood (AOW bucket B).
    Only present frames (>= n_present_min confident keypoints) are eligible, except a
    frame flagged as a location/velocity teleport stays eligible (that IS the problem)."""
    bps = [bp for bp in cfg.bps if (_scorer_of(df), bp, "x") in df.columns]
    n = len(df)
    _, stats = pose_filter.filter_pose(df, **cfg.filter_kw)
    fmask = stats["flagged_mask"]                       # {bp: bool[n]}
    # severity: weight location/velocity (confidently-wrong) above pure low-likelihood.
    # filter_pose folds all three into flagged_mask; recover the 'confident' component by
    # re-deriving location/velocity via the reasons counts is not per-frame, so approximate:
    # count flagged bodyparts per frame (more simultaneously-wrong = worse).
    flag_count = np.zeros(n, dtype=int)
    for bp in bps:
        flag_count += fmask[bp].astype(int)
    LIK = _lik_stack(df, bps)
    nconf = (LIK > cfg.p_present).sum(1)
    present = nconf >= cfg.n_present_min
    sc = _scorer_of(df)
    hind_bps = [bp for bp in ("hlpaw", "hrpaw") if (sc, bp, "likelihood") in df.columns]
    if hind_bps:
        hind = np.minimum.reduce([df[(sc, bp, "likelihood")].to_numpy() for bp in hind_bps])
    else:
        hind = np.nanmin(LIK, 1)
    meanlik = np.nanmean(LIK, 1)

    # (1) flagged frames (present OR flagged>=1), most-flagged first, then lowest meanlik
    flagged_frames = [f for f in np.argsort(-flag_count) if flag_count[f] > 0
                      and (present[f] or flag_count[f] >= cfg.n_present_min)]
    # (2) present, hindpaw most uncertain
    hind_frames = [f for f in np.argsort(hind) if present[f]]
    # (3) lowest mean likelihood overall
    low_frames = list(np.argsort(meanlik))

    seen, ranked = set(), []
    for grp in (flagged_frames, hind_frames, low_frames):
        for f in grp:
            f = int(f)
            if f not in seen:
                seen.add(f); ranked.append(f)
    info = dict(flag_count=flag_count, present=present, hind=hind, meanlik=meanlik,
                reference_px=stats["reference_px"], reasons=stats["reasons"])
    return ranked, info


# ----------------------------------------------------------------------------- classifier bucket
def classifier_probs(features_pkl, dlc_path, clf_data):
    """Per-frame P(positive) via the standard prediction path. Lazy-imports the pipeline."""
    import pickle
    repo = str(Path(__file__).resolve().parent.parent)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from prediction_pipeline import predict_with_xgboost, augment_features_post_cache
    model = clf_data["clf_model"]
    with open(features_pkl, "rb") as f:
        X = pickle.load(f)
    Xa = augment_features_post_cache(X.copy(), clf_data, model, dlc_path)
    proba = np.asarray(predict_with_xgboost(model, Xa, calibrator=clf_data.get("prob_calibrator"),
                                            fold_models=clf_data.get("fold_models")))
    return proba


def rank_classifier(proba, cfg: ExtractConfig, tracking_ranked=None, flag_count=None):
    """Return a ranked candidate list from classifier signal.
    uncertain: frames in the near-threshold band, most-uncertain first.
    positive:  tracking-flagged frames whose +/-win window contains a proba>=prob_lo call."""
    n = len(proba)
    if cfg.classifier_mode == "positive":
        pos_band = proba >= cfg.prob_lo
        src = tracking_ranked if tracking_ranked is not None else range(n)
        cand = []
        for f in src:
            f = int(f)
            lo, hi = max(0, f - cfg.win), min(n, f + cfg.win + 1)
            if pos_band[lo:hi].any():
                cand.append(f)
        return cand
    # uncertain (default)
    unc = 1.0 - np.abs(proba - 0.5) * 2.0
    eligible = np.abs(proba - 0.5) * 2.0 < cfg.confidence_threshold
    cand = [int(f) for f in np.argsort(-unc) if eligible[f]]
    return cand


# ----------------------------------------------------------------------------- candidate-folder writer
def write_candidate_folder(session: Session, df, frames, cfg: ExtractConfig,
                           extra_manifest=None, log=print):
    """Reproduce the exact labeler contract for one session. Returns (folder, n_written)."""
    sc = _scorer_of(df)
    bps = [bp for bp in cfg.bps if (sc, bp, "x") in df.columns]
    cols = pd.MultiIndex.from_product([[cfg.scorer], bps, ["x", "y"]],
                                      names=["scorer", "bodyparts", "coords"])
    d_out = Path(cfg.ldir) / session.name
    if d_out.is_dir() and (d_out / MARKER).is_file() and not cfg.fresh:
        log(f"  ! {session.name}: candidate folder already queued (marker present) — "
            f"skipping to avoid clobbering in-progress labels (use fresh=True to override)")
        return str(d_out), 0
    d_out.mkdir(parents=True, exist_ok=True)
    if cfg.fresh:
        for old in glob.glob(str(d_out / "img*.png")):
            os.remove(old)
    cap = cv2.VideoCapture(session.video)
    ov_dir = d_out / "overlay"
    if cfg.overlays:
        ov_dir.mkdir(exist_ok=True)
    idx, rows, man = [], [], []
    for f in sorted(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if not ok or img is None:
            continue
        name = f"img{int(f):06d}.png"
        cv2.imwrite(str(d_out / name), img)
        idx.append(os.path.join("labeled-data", session.name, name))   # flat single-level index
        rows.append([v for bp in bps
                     for v in (float(df[(sc, bp, "x")].iloc[f]), float(df[(sc, bp, "y")].iloc[f]))])
        if cfg.overlays:
            ov = img.copy()
            for bp in bps:
                xx = df[(sc, bp, "x")].iloc[f]; yy = df[(sc, bp, "y")].iloc[f]
                ll = df[(sc, bp, "likelihood")].iloc[f]
                if np.isnan(xx) or np.isnan(yy):
                    continue
                cv2.circle(ov, (int(xx), int(yy)), 5, BP_COLOR.get(bp, (255, 255, 255)),
                           -1 if ll >= 0.5 else 1)
            cv2.imwrite(str(ov_dir / name), ov)
        rec = dict(session=session.name, frame=int(f))
        if extra_manifest and int(f) in extra_manifest:
            rec.update(extra_manifest[int(f)])
        man.append(rec)
    cap.release()
    cd = pd.DataFrame(rows, columns=cols, index=idx)
    cd.to_hdf(d_out / f"CollectedData_{cfg.scorer}.h5", key="df_with_missing", mode="w")
    cd.to_csv(d_out / f"CollectedData_{cfg.scorer}.csv")
    (d_out / MARKER).write_text("\n".join(os.path.basename(i) for i in idx), encoding="utf-8")
    if man:
        mpath = d_out / "manifest.csv"
        pd.DataFrame(man).to_csv(mpath, index=False)
    return str(d_out), len(idx)


# ----------------------------------------------------------------------------- orchestrator
def _alloc_budget(burden: dict, total_budget: int, max_per: int):
    """Burden-proportional integer allocation summing to min(total_budget, sum(burden))."""
    tot = sum(burden.values()) or 1
    alloc = {s: min(max_per, min(burden[s], round(total_budget * b / tot))) for s, b in burden.items()}
    order = sorted(burden, key=lambda s: -burden[s])
    cap = min(total_budget, sum(min(max_per, burden[s]) for s in burden))
    while sum(alloc.values()) > cap:
        for s in sorted(order, key=lambda s: alloc[s]):
            if sum(alloc.values()) <= cap:
                break
            if alloc[s] > 0:
                alloc[s] -= 1
    while sum(alloc.values()) < cap:
        grew = False
        for s in order:
            if sum(alloc.values()) >= cap:
                break
            if alloc[s] < min(max_per, burden[s]):
                alloc[s] += 1; grew = True
        if not grew:
            break
    return alloc


def extract_problem_frames(sessions, cfg: ExtractConfig, clf_data=None, progress=None):
    """Run selection + writing across sessions. `sessions` = list[Session].
    `progress(msg, frac)` optional callback. Returns a result dict."""
    def report(msg, frac=None):
        if progress:
            progress(msg, frac)

    # ---- per-session candidate ranking ----
    per = {}
    n_sess = len(sessions)
    for i, s in enumerate(sessions):
        report(f"scoring {s.name} ({i+1}/{n_sess})", i / max(1, n_sess) * 0.5)
        try:
            df = load_pose(s.dlc)
        except Exception as e:
            report(f"  ! {s.name}: cannot read pose ({e})")
            continue
        entry = dict(df=df)
        if cfg.use_tracking:
            ranked, info = rank_tracking(df, cfg)
            entry["track_cand"] = ranked
            entry["info"] = info
        else:
            entry["track_cand"] = []
        entry["clf_cand"] = []
        if cfg.use_classifier and clf_data is not None and s.features and os.path.isfile(s.features):
            try:
                proba = classifier_probs(s.features, s.dlc, clf_data)
                entry["proba"] = proba
                entry["clf_cand"] = rank_classifier(
                    proba, cfg, tracking_ranked=entry["track_cand"],
                    flag_count=entry.get("info", {}).get("flag_count"))
            except Exception as e:
                report(f"  ! {s.name}: classifier scoring failed, tracking-only ({e})")
        elif cfg.use_classifier:
            report(f"  ! {s.name}: no feature cache/classifier — tracking-only for this session")
        per[s.name] = entry

    # ---- budget split across buckets, then across sessions by burden ----
    track_budget = int(round(cfg.budget * cfg.tracking_frac)) if cfg.use_tracking else 0
    clf_budget = (cfg.budget - track_budget) if cfg.use_classifier else 0
    if not cfg.use_classifier:
        track_budget = cfg.budget
    if not cfg.use_tracking:
        clf_budget = cfg.budget

    track_burden = {n: len(per[n]["track_cand"]) for n in per if per[n]["track_cand"]}
    clf_burden = {n: len(per[n]["clf_cand"]) for n in per if per[n]["clf_cand"]}
    track_alloc = _alloc_budget(track_burden, track_budget, cfg.max_per_video) if track_burden else {}
    clf_alloc = _alloc_budget(clf_burden, clf_budget, cfg.max_per_video) if clf_burden else {}

    # ---- select per session (spread within, avoid cross-bucket collisions) ----
    chosen = {}
    for name, entry in per.items():
        taken = []
        tsel = spread(entry["track_cand"], track_alloc.get(name, 0), cfg.gap, taken)
        taken += tsel
        csel = spread(entry["clf_cand"], clf_alloc.get(name, 0), cfg.gap, taken)
        picks = sorted(set(tsel) | set(csel))
        if picks:
            chosen[name] = dict(picks=picks, track=set(tsel), clf=set(csel))

    # ---- write candidate folders ----
    sess_by_name = {s.name: s for s in sessions}
    result = dict(folders=[], total=0, per_session={}, track_budget=track_budget,
                  clf_budget=clf_budget)
    for j, (name, sel) in enumerate(chosen.items()):
        report(f"writing {name} ({len(sel['picks'])} frames)", 0.5 + j / max(1, len(chosen)) * 0.5)
        entry = per[name]
        man = {}
        proba = entry.get("proba")
        for f in sel["picks"]:
            man[f] = dict(bucket=("tracking" if f in sel["track"] else "classifier"),
                          proba=(round(float(proba[f]), 3) if proba is not None and f < len(proba) else ""))
        folder, nw = write_candidate_folder(sess_by_name[name], entry["df"], sel["picks"], cfg,
                                            extra_manifest=man, log=lambda m: report(m))
        result["folders"].append(folder)
        result["total"] += nw
        result["per_session"][name] = dict(written=nw,
                                            tracking=len(sel["track"]), classifier=len(sel["clf"]))
    report(f"done — {result['total']} frames across {len(result['folders'])} session(s)", 1.0)
    return result


# ----------------------------------------------------------------------------- CLI (headless testing)
def _cli(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Extract problem-tracking / classifier-uncertain frames "
                                             "into labeler candidate folders.")
    ap.add_argument("--ldir", required=True, help="destination <project>\\labeled-data")
    ap.add_argument("--scorer", required=True, help="target labeling scorer, e.g. SlivickiR_WangH")
    ap.add_argument("--video", action="append", default=[], help="video path (repeatable)")
    ap.add_argument("--dlc", action="append", default=[], help="matching DLC h5 (repeatable, same order)")
    ap.add_argument("--features", action="append", default=[], help="matching features pkl (optional)")
    ap.add_argument("--name", action="append", default=[], help="session name (optional; default video stem)")
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--tracking-frac", type=float, default=0.5)
    ap.add_argument("--max-per-video", type=int, default=12)
    ap.add_argument("--gap", type=int, default=300)
    ap.add_argument("--classifier", help="classifier pkl for the uncertainty/positive bucket")
    ap.add_argument("--classifier-mode", default="uncertain", choices=["uncertain", "positive"])
    ap.add_argument("--no-tracking", action="store_true")
    ap.add_argument("--no-overlays", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args(argv)

    clf_data = None
    if a.classifier:
        import joblib, pickle
        try:
            clf_data = joblib.load(a.classifier)
        except Exception:
            clf_data = pickle.load(open(a.classifier, "rb"))

    sessions = []
    for i, v in enumerate(a.video):
        dlc = a.dlc[i] if i < len(a.dlc) else None
        feat = a.features[i] if i < len(a.features) else None
        name = a.name[i] if i < len(a.name) else Path(v).stem
        sessions.append(Session(name=name, video=v, dlc=dlc, features=feat))

    cfg = ExtractConfig(scorer=a.scorer, ldir=a.ldir, budget=a.budget, tracking_frac=a.tracking_frac,
                        max_per_video=a.max_per_video, gap=a.gap,
                        use_tracking=not a.no_tracking, use_classifier=bool(a.classifier),
                        classifier_mode=a.classifier_mode, overlays=not a.no_overlays, fresh=a.fresh)
    res = extract_problem_frames(sessions, cfg, clf_data=clf_data,
                                 progress=lambda m, f=None: print(m, flush=True))
    print(json.dumps({k: res[k] for k in ("total", "track_budget", "clf_budget", "per_session")}, indent=1))
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    raise SystemExit(_cli())
