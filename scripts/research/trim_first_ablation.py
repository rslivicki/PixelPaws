"""Does trim-to-first-positive help or hurt back/belly grooming?

Per-session trim (mirrors extract_features_for_session): optionally drop frames
BEFORE the first labeled positive (trim_first) and/or AFTER the last (trim_last).
Same baseline model + CV-eligibility(auto) across arms; positives are unchanged by
trimming negatives, so the eval set is identical → arms are comparable.
"""
import os, sys, time
import numpy as np

sys.path.insert(0, r"E:\Sync_from_lab\PixelPaws")
from param_explore_grooming import load_behavior  # also shrinks the sweep grid for speed
import active_learning_v2 as _al
from active_learning_v2 import (_sweep_postprocessing_fast, _honest_pipeline_oof_f1,
                                session_positive_counts, select_cv_eligible)
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score
import xgboost as xgb

BASE = dict(n_estimators=250, max_depth=6, learning_rate=0.06, subsample=0.8,
            colsample_bytree=0.5, tree_method='hist', n_jobs=-1,
            objective='binary:logistic', eval_metric='aucpr', random_state=42)


def apply_trim(y, sids, trim_first, trim_last):
    """Per-session keep-mask dropping pre-first / post-last positive frames."""
    keep = np.ones(len(y), dtype=bool)
    for si in np.unique(sids):
        idx = np.where(sids == si)[0]
        pos = idx[y[idx] == 1]
        if len(pos) == 0:
            continue
        if trim_first:
            keep[idx[idx < pos[0]]] = False
        if trim_last:
            keep[idx[idx > pos[-1]]] = False
    return keep


def cv_eval(Xv, y, sids, eligible, n_folds=5):
    oof = np.full(len(y), np.nan)
    fold_of = np.full(len(y), -1, np.int32)
    elig_mask = np.isin(sids, list(eligible))
    elig_idx = np.where(elig_mask)[0]
    folds = min(n_folds, len(np.unique(sids[elig_idx])))
    for fi, (_tr, va) in enumerate(GroupKFold(n_splits=folds).split(
            elig_idx, y[elig_idx], groups=sids[elig_idx])):
        vidx = elig_idx[va]
        held = set(np.unique(sids[vidx]).tolist())
        tr = np.where(~np.isin(sids, list(held)))[0]
        ytr = y[tr]
        spw = (len(ytr) - ytr.sum()) / max(1, ytr.sum())
        m = xgb.XGBClassifier(scale_pos_weight=spw, early_stopping_rounds=40, **BASE)
        m.fit(Xv[tr], ytr, eval_set=[(Xv[vidx], y[vidx])], verbose=False)
        oof[vidx] = m.predict_proba(Xv[vidx])[:, 1]
        fold_of[vidx] = fi
    valid = fold_of >= 0
    bp = _sweep_postprocessing_fast(oof[valid], y[valid], session_ids=sids[valid], n_jobs=-1)
    honest, _d, mean, std, bout = _honest_pipeline_oof_f1(oof, y, fold_of, session_ids=sids)
    ap = float(average_precision_score(y[valid], oof[valid]))
    return honest, mean, std, bout['f1'], ap, bp


def main():
    for beh in (sys.argv[1:] or ['back_groom', 'belly_groom']):
        X, y0, sids0, used = load_behavior(beh)
        Xv0 = X.values.astype(np.float32)
        print(f"\n{'='*88}\n{beh}  ({len(used)} sessions, {int(y0.sum())} pos, "
              f"{100*y0.mean():.2f}%)\n{'='*88}")
        # eligibility (auto) computed on the untrimmed labels (positives unchanged by trim)
        counts = {int(s): session_positive_counts(y0[sids0 == s], min_bout_len=3)
                  for s in np.unique(sids0)}
        elig, tonly, info = select_cv_eligible(counts, mode='auto')
        print(f"auto eligibility: {info}; train-only={sorted(used[i][0] for i in tonly)}")
        arms = [("A trim_first=OFF last=ON (default)", False, True),
                ("B trim_first=ON  last=ON",           True,  True),
                ("C no trim",                          False, False)]
        print(f"\n{'arm':36} {'frames':>8} {'pos%':>6} {'honestF1':>9} {'boutF1':>7} "
              f"{'AUPRC':>7} {'thr':>5} {'mb':>4} {'gap':>4} {'sec':>5}")
        for label, tf, tl in arms:
            t0 = time.time()
            keep = apply_trim(y0, sids0, tf, tl)
            Xv, y, sids = Xv0[keep], y0[keep], sids0[keep]
            honest, mean, std, bf1, ap, bp = cv_eval(Xv, y, sids, elig)
            print(f"{label:36} {int(keep.sum()):>8} {100*y.mean():>5.2f}% "
                  f"{honest:9.3f} {bf1:7.3f} {ap:7.3f} {bp['thresh']:5.2f} "
                  f"{bp['min_bout']:4d} {bp['max_gap']:4d} {time.time()-t0:5.0f}")
    print("\nDONE")


if __name__ == "__main__":
    main()
