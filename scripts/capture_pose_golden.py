# -*- coding: utf-8 -*-
"""Capture tests/golden/pose_features_synthetic.npz: every pose feature for the
synthetic DLC table of tests/pose_golden_data.py, for each extractor
setting the test runs.

Run it only when the feature set is changed on purpose (and
POSE_FEATURE_VERSION is bumped): the golden exists to prove that refactors of
pose_features.py change nothing.

Usage:  python scripts/capture_pose_golden.py
"""
import os, sys, tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))

import numpy as np
import pose_golden_data as T

out = {}
with tempfile.TemporaryDirectory() as tmp:
    p = os.path.join(tmp, "synthetic.csv")
    T.write_dlc_csv(T.synthetic_dlc(), p)
    for case in T.CASES:
        X = T.extract(p, case)
        out[f"{case}_columns"] = np.array(list(X.columns))
        out[f"{case}_values"] = X.to_numpy(dtype=float)
        print(f"{case}: {X.shape[1]} features x {X.shape[0]} frames, {int(np.isnan(out[f'{case}_values']).sum())} NaN")
np.savez_compressed(T.GOLDEN, **out)
print("saved", T.GOLDEN, f"{os.path.getsize(T.GOLDEN) / 1e6:.2f} MB")
