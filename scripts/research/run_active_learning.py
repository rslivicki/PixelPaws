"""
run_active_learning.py — headless driver for the PixelPaws AL engine
====================================================================
Runs the active_learning_engine loop on real session data without the GUI:
load (features pkl, labels csv) per session -> build ActiveLearningEngine ->
iterate: train -> score+select batch -> LABEL -> apply -> evaluate -> stop.

Labelers (pluggable):
  --oracle <col>   auto-label each selected bout from a ground-truth column in
                   the labels CSV (for automated tests / convergence demos)
  (default)        interactive text prompt: prints each bout (session, frames,
                   model guess) and reads y/n/s from stdin

Sessions are given as repeated --session FEATURES_PKL:LABELS_CSV pairs, e.g.:
  py -X utf8 scripts/research/run_active_learning.py \
     --session D:\PixelPaws_Active\2605_SNI_oxy\features\A_features_8aed1c22.pkl:...\A_labels.csv \
     --behavior Scratching --batch-size 20 --max-iters 15 --oracle gt

Outputs: writes updated labels back to each CSV (atomic) and a learning-curve
JSON (<labels_dir>/<behavior>_al_curve.json).
"""
from __future__ import annotations
import argparse, json, os, pickle, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (…/PixelPaws)
import numpy as np
import pandas as pd
import active_learning_engine as E


def load_session(features_pkl: str, labels_csv: str, label_col=0):
    # encyclopedia feature caches are joblib+LZ4; joblib.load also reads plain pickles
    try:
        import joblib
        feats = joblib.load(features_pkl)
    except Exception:
        with open(features_pkl, "rb") as f:
            feats = pickle.load(f)
    cols = None
    if isinstance(feats, pd.DataFrame):
        cols = list(feats.columns); feats = feats.values
    feats = np.asarray(feats, dtype=np.float32)
    if labels_csv and os.path.isfile(labels_csv):
        df = pd.read_csv(labels_csv)
        raw = df.iloc[:, label_col].to_numpy()
        labels = np.where(np.isnan(raw.astype(float)), -1, raw.astype(float)).astype(int)
        gt = df["gt"].to_numpy().astype(int) if "gt" in df.columns else None
    else:
        labels = np.full(len(feats), -1, dtype=int); df = None; gt = None
    # reconcile lengths
    n = min(len(feats), len(labels))
    if len(labels) < len(feats):
        labels = np.concatenate([labels, np.full(len(feats) - len(labels), -1, int)])
    else:
        labels = labels[:len(feats)]
    return {"features": feats, "labels": labels, "_cols": cols, "_csv": labels_csv,
            "_df": df, "_gt": (gt[:len(feats)] if gt is not None else None)}, cols


def oracle_labeler(eng, sessions, gt_arrays):
    def _lab(batch):
        dec = {}
        for b in batch:
            gt = gt_arrays[b.session_idx]
            if gt is None:
                continue
            # precise per-frame labels (simulates a human marking the exact
            # behavior sub-segment within the shown bout, not a coarse vote)
            dec[(b.session_idx, b.start, b.end)] = gt[b.start:b.end + 1].astype(int)
        return dec
    return _lab


def load_classifier(path):
    """Load an encyclopedia/AL classifier pkl (joblib+lz4 or plain pickle) and
    return the full clf_data dict (warm-start needs the model + calibrator +
    augmentation config)."""
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def warmstart_probas(clf_data, sess_dicts, dlc_paths, feature_cols):
    """Per-session P(positive) from a pre-trained classifier the established way:
    augment_features_post_cache (adds the egocentric/contact/lag/Cat-B features the
    encyclopedia models expect) -> predict_with_xgboost. Needs each session's DLC h5."""
    import pandas as pd
    from prediction_pipeline import augment_features_post_cache, predict_with_xgboost
    model = clf_data["clf_model"] if isinstance(clf_data, dict) else clf_data
    out = []
    for sd, dlc in zip(sess_dicts, dlc_paths):
        cols = sd.get("_cols") or feature_cols or [f"f{i}" for i in range(sd["features"].shape[1])]
        Xdf = pd.DataFrame(sd["features"], columns=cols)
        Xa = augment_features_post_cache(Xdf.copy(), clf_data, model, dlc or "")
        p = predict_with_xgboost(
            model, Xa,
            calibrator=(clf_data.get("prob_calibrator") if isinstance(clf_data, dict) else None),
            fold_models=(clf_data.get("fold_models") if isinstance(clf_data, dict) else None))
        out.append(np.asarray(p, dtype=float))
    return out


def interactive_labeler(batch):
    dec = {}
    print(f"\n--- label {len(batch)} bouts (y=behavior, n=not, s=skip) ---")
    for b in sorted(batch, key=lambda x: (x.session_idx, x.start)):  # grouped by video
        guess = "YES" if b.pred_pos else "no"
        ans = input(f"  sess{b.session_idx} frames {b.start}-{b.end} (entropy {b.score:.2f}, guess {guess}): ").strip().lower()
        if ans in ("y", "yes"):
            dec[(b.session_idx, b.start, b.end)] = 1
        elif ans in ("n", "no", ""):
            dec[(b.session_idx, b.start, b.end)] = 0
        # 's'/anything else -> skip (leave unlabeled)
    return dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", action="append", required=True,
                    help="features pkl (repeatable; paired with --labels by order)")
    ap.add_argument("--labels", action="append", required=True,
                    help="labels csv (repeatable; paired with --features by order)")
    ap.add_argument("--behavior", default="behavior")
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--max-iters", type=int, default=15)
    ap.add_argument("--pos-quota", type=float, default=0.4)
    ap.add_argument("--min-bout", type=int, default=5)
    ap.add_argument("--min-gap", type=int, default=30)
    ap.add_argument("--confidence-threshold", type=float, default=0.3)
    ap.add_argument("--oracle", default=None, help="if set, auto-label from the 'gt' column")
    ap.add_argument("--init-classifier", default=None,
                    help="warm-start: existing classifier pkl used to score iteration-0 "
                         "when there are no/insufficient seed labels (e.g. best encyclopedia model)")
    ap.add_argument("--dlc", action="append", default=None,
                    help="DLC h5 per session (paired with --features by order); needed by "
                         "warm-start to augment features before predicting")
    args = ap.parse_args()

    if len(args.features) != len(args.labels):
        ap.error("--features and --labels must be given the same number of times")
    dlc_paths = args.dlc if args.dlc else [""] * len(args.features)
    if len(dlc_paths) != len(args.features):
        ap.error("--dlc, if given, must be given the same number of times as --features")
    sess_dicts, gts, csv_paths, cols0 = [], [], [], None
    for fpkl, lcsv in zip(args.features, args.labels):
        sd, cols = load_session(fpkl, lcsv)
        cols0 = cols0 or cols
        sess_dicts.append(sd); gts.append(sd["_gt"]); csv_paths.append(sd["_csv"])

    eng = E.ActiveLearningEngine(sess_dicts, feature_cols=cols0,
                                 min_bout_frames=args.min_bout, min_frame_gap=args.min_gap)
    labeler = oracle_labeler(eng, sess_dicts, gts) if args.oracle else interactive_labeler

    init_clf_data = load_classifier(args.init_classifier) if args.init_classifier else None
    if init_clf_data is not None:
        print(f"[AL] warm-start classifier loaded: {os.path.basename(args.init_classifier)}")
    print(f"[AL] {len(sess_dicts)} session(s); {eng.n_labeled()} labeled ({eng.n_positive()} pos) at start")
    while True:
        have_both = eng.n_positive() >= 1 and (eng.n_labeled() - eng.n_positive()) >= 1
        _warm_probas = None
        if have_both:
            model = eng.train()
            X, y, _ = eng._pool_labeled()
            cal = E.fit_calibrator(model.predict_proba(X)[:, 1], y)
            ev = eng.evaluate(); eng.f1_history.append(ev["f1"])
            print(f"[AL] iter {eng.iteration}: trained; labeled={eng.n_labeled()} pos={eng.n_positive()} "
                  f"F1={ev['f1']} pos_recall={ev['pos_recall']}")
            stop = eng.should_stop(model, cal, args.confidence_threshold, args.max_iters)
            if stop:
                print(f"[AL] STOP: {stop}"); break
        elif init_clf_data is not None:
            print(f"[AL] iter {eng.iteration}: warm-start scoring from init classifier "
                  f"(augment + predict; no seed labels yet)")
            try:
                _warm_probas = warmstart_probas(init_clf_data, sess_dicts, dlc_paths, cols0)
            except Exception as ex:
                print(f"[AL] warm-start failed ({type(ex).__name__}: {ex}); supply seed labels "
                      f"or check --dlc paths. stopping."); break
            cal = (lambda p: p)
            if eng.iteration >= args.max_iters:
                print(f"[AL] STOP: hard cap ({args.max_iters})"); break
        else:
            print("[AL] need >=1 pos and >=1 neg seed label, or --init-classifier; stopping.")
            break
        if _warm_probas is not None:
            batch = eng.score_and_candidates(probas_by_session=_warm_probas,
                                             batch_size=args.batch_size, pos_quota_frac=args.pos_quota)
        else:
            batch = eng.score_and_candidates(model, cal, batch_size=args.batch_size,
                                             pos_quota_frac=args.pos_quota)
        if not batch:
            print("[AL] no candidate bouts; stopping."); break
        dec = labeler(batch)
        if not dec:
            print("[AL] no labels returned; stopping."); break
        eng.apply_labels(dec); eng.iteration += 1

    # persist labels + curve
    for sd, csv in zip(sess_dicts, csv_paths):
        if csv:
            out = pd.DataFrame({"label": np.where(sd["labels"] < 0, np.nan, sd["labels"])})
            if sd["_gt"] is not None:
                out["gt"] = sd["_gt"]
            tmp = csv + ".tmp"; out.to_csv(tmp, index=False); os.replace(tmp, csv)
    if csv_paths and csv_paths[0]:
        curve = os.path.join(os.path.dirname(csv_paths[0]), f"{args.behavior}_al_curve.json")
        json.dump({"behavior": args.behavior, "f1_history": eng.f1_history,
                   "iterations": eng.iteration, "n_labeled": eng.n_labeled()},
                  open(curve, "w"), indent=2)
        print(f"[AL] curve -> {curve}")
    print("[AL] done.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
