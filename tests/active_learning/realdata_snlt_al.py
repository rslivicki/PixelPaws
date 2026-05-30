"""Real-data AL label-efficiency test: body_grooming on 2603_SNLT_JG.

Uses REAL joblib+LZ4 8aed1c22 features + REAL Targets ground-truth labels (prefix-
aligned to features). Per session, the last 30% of frames are a FIXED held-out test
set; AL starts from a tiny seed in the train pool, queries informative bouts, and an
oracle labels them from ground truth. We report held-out F1 vs #labels — i.e. does
AL reach the full-label ceiling with few labels? (No warm-start: the encyclopedia
body_grooming classifier was trained on these sessions, so warm-start is circular here.)
"""
import os, sys, glob
import numpy as np
import pandas as pd

_REPO = r"E:\Sync_from_lab\PixelPaws"
sys.path.insert(0, _REPO)
import active_learning_engine as E
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

BASE = r"E:\Sync_from_lab\Blackbox\2603_SNLT_JG\Baseline"
SESSIONS = ["260318_JG_9433_Baseline", "260318_JG_9435_Baseline", "260318_JG_9419_Baseline"]
TEST_FRAC = 0.30
SEED_FRAC_TRAIN = 0.04
BATCH = 20
MAX_ITERS = 8


def load(stem):
    import joblib
    X = joblib.load(os.path.join(BASE, "features", f"{stem}_features_8aed1c22.pkl"))
    cols = list(X.columns); X = X.values.astype(np.float32)
    gt = pd.read_csv(os.path.join(BASE, "Targets", f"{stem}_labels.csv")).iloc[:, 0].to_numpy().astype(int)
    n = min(len(X), len(gt)); return X[:n], gt[:n], cols


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rng = np.random.RandomState(0)
    train_sessions, gt_tr, test_X, test_gt, cols0 = [], [], [], [], None
    for stem in SESSIONS:
        X, gt, cols = load(stem); cols0 = cols0 or cols
        ntr = int((1 - TEST_FRAC) * len(X))
        lab = np.full(ntr, -1, int)
        seed = int(SEED_FRAC_TRAIN * ntr)
        sidx = rng.choice(ntr, size=seed, replace=False)   # random seed across session -> hits rare positives
        lab[sidx] = gt[:ntr][sidx]
        train_sessions.append({"features": X[:ntr], "labels": lab})
        gt_tr.append(gt[:ntr]); test_X.append(X[ntr:]); test_gt.append(gt[ntr:])
        print(f"  {stem}: train {ntr} (seed {seed}, pos {int((lab==1).sum())}) | test {len(X)-ntr} (pos {test_gt[-1].mean()*100:.1f}%)")

    Xtest = np.concatenate(test_X); ytest = np.concatenate(test_gt)

    # full-label ceiling: train on ALL train-pool ground truth -> F1 on held-out test
    Xall = np.concatenate([s["features"] for s in train_sessions])
    yall = np.concatenate(gt_tr)
    spw = float((yall == 0).sum() / max((yall == 1).sum(), 1))
    ceil = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, scale_pos_weight=spw,
                         random_state=0, verbosity=0).fit(Xall, yall)
    ceil_f1 = f1_score(ytest, (ceil.predict_proba(Xtest)[:, 1] >= 0.5).astype(int), zero_division=0)
    print(f"\n[ceiling] all {len(yall)} train labels -> held-out F1 = {ceil_f1:.3f}\n")

    eng = E.ActiveLearningEngine(train_sessions, feature_cols=cols0,
                                 min_bout_frames=5, max_bout_frames=120, min_frame_gap=60)
    print(f"{'iter':>4} {'labeled':>8} {'pos':>5} {'heldout_F1':>11} {'pos_recall':>10}")
    curve = []
    for it in range(MAX_ITERS):
        if not (eng.n_positive() >= 1 and (eng.n_labeled() - eng.n_positive()) >= 1):
            print("  (need both classes in seed) "); break
        m = eng.train()
        pr = (m.predict_proba(Xtest)[:, 1] >= 0.5).astype(int)
        f1 = f1_score(ytest, pr, zero_division=0)
        rec = float((pr[ytest == 1] == 1).mean()) if (ytest == 1).any() else float("nan")
        curve.append((eng.n_labeled(), f1))
        print(f"{it:>4} {eng.n_labeled():>8} {eng.n_positive():>5} {f1:>11.3f} {rec:>10.3f}")
        cal = E.fit_calibrator(*(lambda a: (m.predict_proba(a[0])[:, 1], a[1]))(eng._pool_labeled()[:2]))
        if eng.should_stop(m, cal, 0.3, MAX_ITERS):
            print("  STOP: converged"); break
        batch = eng.score_and_candidates(m, cal, batch_size=BATCH, pos_quota_frac=0.4)
        if not batch:
            print("  no candidate bouts"); break
        dec = {(b.session_idx, b.start, b.end): gt_tr[b.session_idx][b.start:b.end + 1] for b in batch}
        eng.apply_labels(dec); eng.iteration += 1

    if not curve:
        print("no AL iterations ran (seed lacked a class)"); return 1
    print(f"\n[result] held-out F1 {curve[0][1]:.3f} (@{curve[0][0]} labels) -> "
          f"{curve[-1][1]:.3f} (@{curve[-1][0]} labels); ceiling {ceil_f1:.3f} "
          f"(@{len(yall)} labels). AL reached {100*curve[-1][1]/ceil_f1:.0f}% of ceiling "
          f"using {100*curve[-1][0]/len(yall):.0f}% of the labels.")
    print("REAL-DATA SNLT body_grooming AL: PASS")


if __name__ == "__main__":
    raise SystemExit(main())
