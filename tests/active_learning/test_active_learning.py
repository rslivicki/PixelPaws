"""Self-contained active-learning test project.

Generates a synthetic fixture (make_fixture.py) and exercises the AL engine +
headless runner end-to-end. No real data / DLC required.

Run:   "C:\\Program Files\\Python310\\python.exe" test_active_learning.py
       (also pytest-discoverable as test_active_learning)
"""
import os, sys, glob, pickle, tempfile, subprocess
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))   # …/PixelPaws
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)

import active_learning_engine as E
import make_fixture


def _load_sessions(info):
    sess = []
    for s in info["sessions"]:
        X = pickle.load(open(s["features_pkl"], "rb")).values.astype(np.float32)
        df = pd.read_csv(s["labels_csv"]); raw = df["behavior"].to_numpy()
        lab = np.where(np.isnan(raw.astype(float)), -1, raw.astype(float)).astype(int)
        sess.append({"features": X, "labels": lab, "gt": df["gt"].to_numpy().astype(int)})
    return sess


def test_active_learning():
    # ---- engine unit checks ----
    assert abs(E.binary_entropy(np.array([0.5]))[0] - 1.0) < 1e-9
    assert E.binary_entropy(np.array([0.01]))[0] < 0.1
    assert E.f1_plateaued([0.80, 0.805, 0.807]) and not E.f1_plateaued([0.5, 0.6, 0.7])
    assert E.pool_drained(np.array([0.99, 0.01, 0.98]), np.array([1, 0, -1]), 0.05)
    assert not E.pool_drained(np.array([0.99, 0.01, 0.51]), np.array([1, 0, -1]), 0.10)

    d = tempfile.mkdtemp(prefix="al_testproj_")
    info = make_fixture.make(d, n_sessions=3, n_frames=800, seed=1)
    cols = info["feature_cols"]
    sess = _load_sessions(info)

    # ---- engine end-to-end loop (seed -> train -> CV-OOF F1 -> select -> oracle apply) ----
    eng = E.ActiveLearningEngine([{"features": s["features"], "labels": s["labels"].copy()} for s in sess],
                                 feature_cols=cols, min_bout_frames=5, max_bout_frames=60, min_frame_gap=25)
    f1s = []
    for _ in range(8):
        if not (eng.n_positive() >= 1 and (eng.n_labeled() - eng.n_positive()) >= 1):
            break
        m = eng.train()
        X, y, _ = eng._pool_labeled()
        cal = E.fit_calibrator(m.predict_proba(X)[:, 1], y)
        ev = eng.evaluate(); f1s.append(ev["f1"]); eng.f1_history.append(ev["f1"])
        if eng.should_stop(m, cal, 0.4, 8):
            break
        batch = eng.score_and_candidates(m, cal, batch_size=16, pos_quota_frac=0.4)
        if not batch:
            break
        # temporal min-gap within each session, and global uniqueness
        for k in set(b.session_idx for b in batch):
            st = sorted(b.start for b in batch if b.session_idx == k)
            assert all(j - i >= 25 for i, j in zip(st, st[1:])), "temporal min-gap violated"
        assert len(set(id(b) for b in batch)) == len(batch)
        # oracle: precise per-frame labels from ground truth
        dec = {(b.session_idx, b.start, b.end): sess[b.session_idx]["gt"][b.start:b.end + 1] for b in batch}
        eng.apply_labels(dec); eng.iteration += 1
    assert f1s and f1s[-1] is not None and f1s[-1] > 0.5, f"AL did not learn (f1s={f1s})"

    # ---- warm-start via probas injection (pretend pruned classifier, subset features) ----
    cd = pickle.load(open(info["classifier_pkl"], "rb")); clf = cd["clf_model"]
    assert set(clf.feature_names_in_).issubset(set(cols)) and len(clf.feature_names_in_) < len(cols)
    eng2 = E.ActiveLearningEngine([{"features": s["features"], "labels": np.full(len(s["features"]), -1, int)} for s in sess],
                                  feature_cols=cols, min_bout_frames=5, max_bout_frames=60, min_frame_gap=25)
    probas = [clf.predict_proba(pd.DataFrame(s["features"], columns=cols)[list(clf.feature_names_in_)])[:, 1]
              for s in sess]
    wb = eng2.score_and_candidates(probas_by_session=probas, batch_size=12, pos_quota_frac=0.4)
    assert 0 < len(wb) <= 12 and len(set(b.session_idx for b in wb)) >= 1, "warm-start produced no global batch"

    # ---- headless runner subprocess (seed-label path + oracle) converges + writes curve ----
    fa, la = [], []
    for s in info["sessions"]:
        fa += ["--features", s["features_pkl"]]; la += ["--labels", s["labels_csv"]]
    cmd = [sys.executable, os.path.join(_REPO, "scripts", "research", "run_active_learning.py"),
           *fa, *la, "--behavior", "TestBehavior", "--batch-size", "16", "--max-iters", "6",
           "--oracle", "gt", "--confidence-threshold", "0.5"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, f"runner failed: {r.stderr[-800:]}"
    assert glob.glob(os.path.join(d, "TestBehavior_al_curve.json")), "curve JSON not written"

    print("ALL AL TEST-PROJECT CHECKS PASS  "
          f"(engine F1 {f1s[0]:.2f}->{f1s[-1]:.2f} over {len(f1s)} iters; "
          f"warm-start batch {len(wb)} bouts; runner OK)")


if __name__ == "__main__":
    test_active_learning()
