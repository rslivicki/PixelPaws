"""Headless parameter-exploration for back_groom / belly_groom.

Mirrors the Train tab's CV -> sweep -> honest-F1 path on the 8aed1c22 cache
features (no augmentation, so absolute F1 differs slightly from the GUI, but the
arm-vs-arm deltas are fair — same config across arms). Reports honest CV F1,
AUPRC, and the deployed operating point (thresh/min_bout/max_gap) per arm.
"""
import os, sys, glob, time
import numpy as np, pandas as pd

sys.path.insert(0, r"E:\Sync_from_lab\PixelPaws")
import active_learning_v2 as _al
# Shrink the post-processing sweep grid FOR THIS EXPLORATION ONLY (kept constant across
# arms, so model-param deltas stay fair) — the full deployed grid is unchanged in the app.
_al._SWEEP_MIN_BOUTS       = [1, 8, 30, 90]
_al._SWEEP_MIN_AFTER_BOUTS = [0]
_al._SWEEP_MAX_GAPS        = [0, 6, 30]
from active_learning_v2 import (_robust_unpickle, _sweep_postprocessing_fast,
                                _honest_pipeline_oof_f1)
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score
import xgboost as xgb

OUT  = r"E:\Sync_from_lab\Blackbox\2603_SNLT_JG\Baseline\behavior_labels"
FEAT = r"E:\Sync_from_lab\Blackbox\2603_SNLT_JG\Baseline\features"


def feat_path(session):
    c = glob.glob(os.path.join(FEAT, f"{session}*_features_8aed1c22.pkl"))
    e = [x for x in c if os.path.basename(x).startswith(session + "_features_")]
    return (e or c or [None])[0]


def load_behavior(behavior):
    """Return X (DataFrame), y (int8 0/1, -1 dropped), session_ids (int)."""
    Xs, ys, sids = [], [], []
    sidx = 0
    used = []
    for f in sorted(os.listdir(OUT)):
        if not f.endswith('_labels.csv'):
            continue
        sess = f[:-len('_labels.csv')]
        df = pd.read_csv(os.path.join(OUT, f))
        if behavior not in df.columns:
            continue
        raw = df[behavior].values
        lab = np.where(np.isnan(raw.astype(float)), -1, raw.astype(int))
        if (lab == 1).sum() == 0:
            continue
        fp = feat_path(sess)
        if not fp:
            continue
        feats = _robust_unpickle(fp)
        if not isinstance(feats, pd.DataFrame):
            continue
        n = min(len(feats), len(lab))
        feats = feats.iloc[:n].reset_index(drop=True)
        lab = lab[:n]
        m = lab != -1
        if m.sum() == 0 or (lab[m] == 1).sum() == 0:
            continue
        Xs.append(feats.iloc[m].reset_index(drop=True))
        ys.append(lab[m])
        sids.append(np.full(int(m.sum()), sidx, dtype=int))
        used.append((sess, int((lab[m] == 1).sum()), int(m.sum())))
        sidx += 1
    X = pd.concat(Xs, ignore_index=True).fillna(0.0)
    y = np.concatenate(ys).astype(np.int8)
    session_ids = np.concatenate(sids)
    return X, y, session_ids, used


def run_cv(X, y, session_ids, n_estimators, max_depth, lr, use_spw,
           n_folds=5, calibrate=False):
    """GroupKFold OOF probabilities + fold_of (mirrors Train CV)."""
    oof = np.full(len(y), np.nan, dtype=float)
    fold_of = np.full(len(y), -1, dtype=np.int32)
    uniq = np.unique(session_ids)
    folds = min(n_folds, len(uniq))
    gkf = GroupKFold(n_splits=folds)
    Xv = X.values.astype(np.float32)
    for fi, (tr, va) in enumerate(gkf.split(Xv, y, groups=session_ids)):
        ytr = y[tr]
        spw = ((len(ytr) - ytr.sum()) / max(1, ytr.sum())) if use_spw else 1.0
        m = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                              learning_rate=lr, subsample=0.8, colsample_bytree=0.5,
                              scale_pos_weight=spw, tree_method='hist', n_jobs=-1,
                              objective='binary:logistic', eval_metric='aucpr',
                              random_state=42, early_stopping_rounds=40)
        m.fit(Xv[tr], ytr, eval_set=[(Xv[va], y[va])], verbose=False)
        p = m.predict_proba(Xv[va])[:, 1]
        if calibrate:
            from sklearn.isotonic import IsotonicRegression
            # LOFO-ish: fit calibrator on training fold's own preds (cheap proxy)
            ptr = m.predict_proba(Xv[tr])[:, 1]
            iso = IsotonicRegression(out_of_bounds='clip')
            try:
                iso.fit(ptr, ytr); p = iso.predict(p)
            except Exception:
                pass
        oof[va] = p
        fold_of[va] = fi
    return oof, fold_of


def evaluate(oof, y, fold_of, session_ids):
    bp = _sweep_postprocessing_fast(oof, y, session_ids=session_ids, n_jobs=-1)
    honest, _det, mean, std, bout = _honest_pipeline_oof_f1(oof, y, fold_of,
                                                            session_ids=session_ids)
    ap = float(average_precision_score(y, oof)) if 0 < y.sum() < len(y) else float('nan')
    return bp, honest, mean, std, ap, bout


def main():
    behaviors = sys.argv[1:] or ['back_groom', 'belly_groom']
    for beh in behaviors:
        print(f"\n{'='*78}\n{beh}\n{'='*78}")
        X, y, sid, used = load_behavior(beh)
        print(f"sessions={len(used)}  frames={len(y)}  pos={int(y.sum())} "
              f"({100*y.mean():.2f}%)  features={X.shape[1]}")
        for s, p, n in used:
            print(f"   {s:30} pos={p:>6}  labeled={n:>7}")

        arms = [
            ("baseline (spw, d6, 250@.06)",  dict(n_estimators=250, max_depth=6, lr=0.06, use_spw=True)),
            ("no scale_pos_weight",          dict(n_estimators=250, max_depth=6, lr=0.06, use_spw=False)),
            ("depth=8",                      dict(n_estimators=250, max_depth=8, lr=0.06, use_spw=True)),
            ("more trees (500@.03)",         dict(n_estimators=500, max_depth=6, lr=0.03, use_spw=True)),
            ("isotonic calibration",         dict(n_estimators=250, max_depth=6, lr=0.06, use_spw=True, calibrate=True)),
        ]
        print(f"\n{'arm':32} {'honestF1':>9} {'boutF1':>7} {'AUPRC':>7} "
              f"{'thr':>5} {'mb':>4} {'gap':>4} {'optF1':>7} {'sec':>5}")
        for name, kw in arms:
            t0 = time.time()
            cal = kw.pop('calibrate', False)
            oof, fold_of = run_cv(X, y, sid, n_folds=5, calibrate=cal, **kw)
            bp, honest, mean, std, ap, bout = evaluate(oof, y, fold_of, sid)
            dt = time.time() - t0
            print(f"{name:32} {honest:9.3f} {bout['f1']:7.3f} {ap:7.3f} "
                  f"{bp['thresh']:5.2f} {bp['min_bout']:4d} {bp['max_gap']:4d} "
                  f"{bp['f1']:7.3f} {dt:5.0f}")
    print("\nDONE")


if __name__ == "__main__":
    main()
