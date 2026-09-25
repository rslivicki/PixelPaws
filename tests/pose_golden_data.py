"""Synthetic DLC table and extractor settings shared by
tests/test_pose_features_golden.py and scripts/capture_pose_golden.py
(no pytest import, so the capture script runs in the app's environment)."""
import os

import numpy as np
import pandas as pd

from pose_features import PoseFeatureExtractor

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", "pose_features_synthetic.npz")
BODYPARTS = ["tailtip", "tailbase", "centroid", "neck", "snout", "hlpaw", "hrpaw", "flpaw", "frpaw"]
CASES = {                      # extractor settings -> key in the golden
    "all": dict(bodyparts=None, mm_per_pixel=None),
    "subset_mm": dict(bodyparts=["hlpaw", "hrpaw", "snout", "centroid"], mm_per_pixel=0.149254),
}


def synthetic_dlc(n=200, seed=7):
    """A DLC-style table (scorer / bodyparts / coords): random walks with the
    awkward cases real tracking has. NaN runs, two points on the same pixel
    (zero-length sides in the angle formula), frozen frames, low likelihoods."""
    rng = np.random.default_rng(seed)
    cols = pd.MultiIndex.from_product([["DLC_synthetic"], BODYPARTS, ["x", "y", "likelihood"]],
                                      names=["scorer", "bodyparts", "coords"])
    data = np.empty((n, len(cols)))
    for b in range(len(BODYPARTS)):
        start = rng.uniform(150, 450, size=2)
        walk = start + np.cumsum(rng.normal(0, 3.0, size=(n, 2)), axis=0)
        data[:, 3 * b] = walk[:, 0]
        data[:, 3 * b + 1] = walk[:, 1]
        data[:, 3 * b + 2] = rng.uniform(0, 1, size=n) ** 0.3
    df = pd.DataFrame(data, columns=cols)
    s = df.columns.levels[0][0]
    df.loc[20:24, [(s, "tailtip", "x"), (s, "tailtip", "y")]] = np.nan          # lost point
    df.loc[60:62, (s, "hrpaw", "x")] = df.loc[60:62, (s, "hlpaw", "x")]          # two paws on one pixel
    df.loc[60:62, (s, "hrpaw", "y")] = df.loc[60:62, (s, "hlpaw", "y")]
    df.iloc[100:104] = df.iloc[100].to_numpy()                                   # frozen frames
    df.loc[150:155, (s, "snout", "likelihood")] = 0.01
    return df


def write_dlc_csv(df, path):
    """DeepLabCut's CSV layout: three header rows and the frame number first."""
    df.to_csv(path)


def extract(path, case):
    ex = PoseFeatureExtractor(**CASES[case])
    return ex.extract_all_features(str(path))
