# -*- coding: utf-8 -*-
"""A project without a key file still analyzes: every session in one group. A key row
with a blank group is Ungrouped, not ''."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_one_group_key_covers_every_file():
    import analysis_core as core
    files = ["mouse1_veh_classifier_Left_licking_predictions.csv",
             "mouse2_veh_classifier_Left_licking_predictions.csv",
             "mouse1_veh_classifier_still_predictions.csv"]
    key = core.one_group_key(files)
    assert list(key.columns) == ["Subject", "Treatment"]
    assert set(key.Treatment) == {"All"}
    # the run resolves subjects against this key the same way: every file matches
    for f in files:
        subj = core.resolve_subject(f, key)
        assert subj in set(key.Subject), (f, subj)


def test_blank_treatment_becomes_ungrouped(tmp_path):
    import analysis_core as core
    p = tmp_path / "key_file.csv"
    pd.DataFrame({"Subject": ["mouse1", "mouse2"], "Treatment": ["", None]}).to_csv(p, index=False)
    key = core.load_key_file(str(p))
    assert list(key.Treatment) == ["Ungrouped", "Ungrouped"]
