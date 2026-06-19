"""Honest ablation: do sparse sessions HELP or HURT, and does excluding them from
CV evaluation stabilize honest F1?

For each behavior, a fixed baseline XGBoost config (spw, depth 6, 250 trees) is run
in 3 conditions, all evaluated on the SAME held-out folds (the auto-eligible/rich
sessions) except 'off':
  off        : every positive session is a CV fold (today's behavior; includes the
               sparse animals as held-out → the unstable baseline).
  train_only : fold out only eligible sessions; sparse sessions still TRAIN every fold.
  drop       : fold out only eligible sessions; sparse sessions excluded from training too.
train_only vs drop share identical eval folds → ΔF1 = the sparse sessions' contribution
to generalization (no outcome-based gaming).
"""
import os, sys, time
import numpy as np

sys.path.insert(0, r"E:\Sync_from_lab\PixelPaws")
# importing this also shrinks the sweep grid for speed (its module-level monkeypatch)
from param_explore_grooming import load_behavior
import active_learning_v2 as _al
from active_learning_v2 import (_sweep_postprocessing_fast, _honest_pipeline_oof_f1,
                                session_positive_counts, select_cv_eligible)
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score
import xgboost as xgb

BASE = dict(n_estimators=250, max_depth=6, learning_rate=0.06, subsample=0.8,
            colsample_bytree=0.5, tree_method='hist', n_jobs=-1,
            objective='binary:logistic', eval_metric='aucpr', random_state=42)


def run(Xv, y, sids, eligible, mode, n_folds=5):
    oof = np.full(len(y), np.nan)
    fold_of = np.full(len(y), -1, np.int32)
    elig_mask = np.isin(sids, list(eligible))
    elig_idx = np.where(elig_mask)[0]
    elig_sessions = np.unique(sids[elig_idx])
    folds = min(n_folds, len(elig_sessions))
    gkf = GroupKFold(n_splits=folds)
    for fi, (_tr, va) in enumerate(gkf.split(elig_idx, y[elig_idx],
                                             groups=sids[elig_idx])):
        val_idx = elig_idx[va]
        held = set(np.unique(sids[val_idx]).tolist())
        not_held = ~np.isin(sids, list(held))
        train_mask = (elig_mask & not_held) if mode == 'drop' else not_held
        tr = np.where(train_mask)[0]
        ytr = y[tr]
        spw = (len(ytr) - ytr.sum()) / max(1, ytr.sum())
        m = xgb.XGBClassifier(scale_pos_weight=spw, early_stopping_rounds=40, **BASE)
        m.fit(Xv[tr], ytr, eval_set=[(Xv[val_idx], y[val_idx])], verbose=False)
        oof[val_idx] = m.predict_proba(Xv[val_idx])[:, 1]
        fold_of[val_idx] = fi
    return oof, fold_of


def evaluate(oof, y, fold_of, sids):
    valid = fold_of >= 0
    bp = _sweep_postprocessing_fast(oof[valid], y[valid],
                                    session_ids=sids[valid], n_jobs=-1)
    honest, det, mean, std, bout = _honest_pipeline_oof_f1(
        oof, y, fold_of, session_ids=sids)
    ap = float(average_precision_score(y[valid], oof[valid])) \
        if 0 < y[valid].sum() < valid.sum() else float('nan')
    return honest, mean, std, bout['f1'], ap, det


def main():
    for beh in (sys.argv[1:] or ['back_groom', 'belly_groom']):
        X, y, sids, used = load_behavior(beh)
        Xv = X.values.astype(np.float32)
        # per-session positive counts (session id = index into `used`)
        counts = {}
        for si in np.unique(sids):
            counts[int(si)] = session_positive_counts(y[sids == si])
        elig, tonly, info = select_cv_eligible(counts, mode='auto')
        name = {i: used[i][0] for i in range(len(used))}
        print(f"\n{'='*84}\n{beh}  ({len(used)} sessions, {int(y.sum())} pos, "
              f"{100*y.mean():.2f}%)\n{'='*84}")
        print(f"auto eligibility: {info}")
        print("  eligible :", sorted(name[i] for i in elig))
        print("  train-only:", sorted(name[i] for i in tonly),
              "->", {name[i]: counts[i] for i in tonly})

        all_sess = set(counts)
        conds = [("off (all sessions eval'd)", all_sess, 'train_only'),
                 ("train-only (sparse train, eval rich)", elig, 'train_only'),
                 ("drop (sparse removed, eval rich)", elig, 'drop')]
        print(f"\n{'condition':40} {'honestF1':>9} {'perfold':>13} {'boutF1':>7} {'AUPRC':>7} {'sec':>5}")
        for label, eset, mode in conds:
            t0 = time.time()
            oof, fold_of = run(Xv, y, sids, eset, mode)
            honest, mean, std, bf1, ap, det = evaluate(oof, y, fold_of, sids)
            print(f"{label:40} {honest:9.3f} {mean:6.3f}±{std:5.3f} {bf1:7.3f} "
                  f"{ap:7.3f} {time.time()-t0:5.0f}")
    print("\nDONE")


if __name__ == "__main__":
    main()
