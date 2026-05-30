"""Generate a self-contained synthetic active-learning test project.

Writes, into a target dir, N sessions each with:
  - S<k>_features_8aed1c22.pkl  (canonical-style 8-col feature DataFrame)
  - S<k>_labels.csv             (col 0 = behavior labels (NaN=unlabeled), col 'gt' = ground truth)
and one "pretend pruned encyclopedia" classifier trained on a feature SUBSET
(classifier_TestBehavior.pkl) so feature_names_in_ is a strict subset of the cache
(exercises engine._align). No DLC/video needed.

The behavior is learnable + rare: positive iff f0>0.7 AND f1>0.5 (~10-15% of frames).
"""
from __future__ import annotations
import os, pickle
import numpy as np
import pandas as pd

FEATURES = [f"f{i}" for i in range(8)]


def make(project_dir, n_sessions=3, n_frames=800, seed=0, seed_frac=0.12):
    os.makedirs(project_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    sessions, allX, allg = [], [], []
    for k in range(n_sessions):
        X = rng.rand(n_frames, 8).astype(np.float32)
        gt = ((X[:, 0] > 0.7) & (X[:, 1] > 0.5)).astype(int)   # rare, learnable
        fp = os.path.join(project_dir, f"S{k}_features_8aed1c22.pkl")
        pickle.dump(pd.DataFrame(X, columns=FEATURES), open(fp, "wb"))
        lab = np.full(n_frames, np.nan)
        n_seed = int(seed_frac * n_frames)
        lab[:n_seed] = gt[:n_seed]                              # seed = first seed_frac labeled
        cs = os.path.join(project_dir, f"S{k}_labels.csv")
        pd.DataFrame({"behavior": lab, "gt": gt}).to_csv(cs, index=False)
        sessions.append({"features_pkl": fp, "labels_csv": cs})
        allX.append(X[:int(0.6 * n_frames)]); allg.append(gt[:int(0.6 * n_frames)])
    # pretend pruned classifier: trained on cols f0-f4 only (subset of the 8-col cache)
    from xgboost import XGBClassifier
    Xtr, ytr = np.concatenate(allX), np.concatenate(allg)
    clf = XGBClassifier(n_estimators=120, max_depth=4, verbosity=0,
                        scale_pos_weight=float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
                        ).fit(pd.DataFrame(Xtr[:, :5], columns=FEATURES[:5]), ytr)
    clf_pkl = os.path.join(project_dir, "classifier_TestBehavior.pkl")
    pickle.dump({"clf_model": clf, "best_thresh": 0.5, "Behavior_type": "TestBehavior"}, open(clf_pkl, "wb"))
    return {"sessions": sessions, "classifier_pkl": clf_pkl, "feature_cols": FEATURES}


if __name__ == "__main__":
    import sys, json
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "_fixture")
    info = make(d)
    print("fixture ->", d)
    print(json.dumps({"sessions": len(info["sessions"]), "classifier": os.path.basename(info["classifier_pkl"]),
                      "cols": info["feature_cols"]}, indent=2))
