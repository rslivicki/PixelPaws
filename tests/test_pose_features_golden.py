"""Pose features must not move: trained classifiers and feature caches depend
on every column name, the column order and every value.

The golden (tests/golden/pose_features_synthetic.npz) was captured with
scripts/capture_pose_golden.py from the implementation as it stood before
the 2026-09-19 rewrite of the loader, body-part selection, angles, velocities
and distance velocities. Names and order are compared exactly; values to
1e-12, because the last bit of arccos can differ between NumPy builds (on the
machine that made the rewrite, old and new were bit-identical on the
synthetic set and on real recordings).
"""
import os

import numpy as np
import pandas as pd
import pytest

from pose_features import PoseFeatureExtractor

from pose_golden_data import GOLDEN, CASES, synthetic_dlc, write_dlc_csv, extract


@pytest.fixture(scope="module")
def golden():
    if not os.path.isfile(GOLDEN):
        pytest.skip("golden not captured")
    return np.load(GOLDEN, allow_pickle=False)


@pytest.mark.parametrize("case", list(CASES))
def test_csv_matches_golden(tmp_path, golden, case):
    p = tmp_path / "synthetic.csv"
    write_dlc_csv(synthetic_dlc(), p)
    X = extract(p, case)
    assert list(X.columns) == list(golden[f"{case}_columns"])
    np.testing.assert_allclose(X.to_numpy(dtype=float), golden[f"{case}_values"], rtol=1e-12, atol=1e-12,
                               equal_nan=True)


@pytest.mark.parametrize("case", list(CASES))
def test_h5_matches_golden(tmp_path, golden, case):
    pytest.importorskip("tables")
    p = tmp_path / "synthetic.h5"
    synthetic_dlc().to_hdf(p, key="df_with_missing", mode="w")
    X = extract(p, case)
    assert list(X.columns) == list(golden[f"{case}_columns"])
    np.testing.assert_allclose(X.to_numpy(dtype=float), golden[f"{case}_values"], rtol=1e-12, atol=1e-12,
                               equal_nan=True)


def test_angle_names_and_geometry():
    """Ang_a-b-c is the angle at b, in degrees, for a < c in column order."""
    ex = PoseFeatureExtractor(bodyparts=None)
    x = pd.DataFrame({"a_x": [1.0, 0.0], "b_x": [0.0, 0.0], "c_x": [0.0, 0.0]})
    y = pd.DataFrame({"a_y": [0.0, 0.0], "b_y": [0.0, 0.0], "c_y": [1.0, 1.0]})
    ang = ex.calculate_angles(x, y)
    assert list(ang.columns) == ["Ang_a-c-b", "Ang_a-b-c", "Ang_b-a-c"]
    assert ang["Ang_a-b-c"].iloc[0] == pytest.approx(90.0)
    assert ang["Ang_a-c-b"].iloc[0] == pytest.approx(45.0)
    # second frame: a sits on b. Seen from c they are the same direction (0 degrees);
    # the angle AT b has a zero-length side and is undefined
    assert ang["Ang_a-c-b"].iloc[1] == pytest.approx(0.0)
    assert np.isnan(ang["Ang_a-b-c"].iloc[1])


def test_velocity_edges():
    ex = PoseFeatureExtractor(bodyparts=None)
    x = pd.DataFrame({"p_x": [0.0, 3.0, 3.0, np.nan, 9.0]})
    y = pd.DataFrame({"p_y": [0.0, 4.0, 4.0, np.nan, 12.0]})
    v1 = ex.calculate_velocities(x, y, t=1)
    assert list(v1.columns) == ["p_Vel1"]
    assert v1["p_Vel1"].tolist() == [0.0, 5.0, 0.0, 0.0, 0.0]      # first frame and NaN steps are 0
    v2 = ex.calculate_velocities(x, y, t=2)
    assert v2["p_Vel2"].tolist() == [0.0, 0.0, 2.5, 0.0, 5.0]      # distance over two frames / 2
