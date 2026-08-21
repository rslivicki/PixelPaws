"""
Headless unit tests for analysis_core (no tkinter, no display).

Run directly:  python tests/test_analysis_core.py
(also collectable by pytest)
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analysis_core as ac


# ---------------------------------------------------------------------------
# detect_bouts
# ---------------------------------------------------------------------------

def test_detect_bouts_empty():
    assert ac.detect_bouts(np.array([])) == []
    assert ac.detect_bouts(np.zeros(10, dtype=int)) == []


def test_detect_bouts_all_ones():
    assert ac.detect_bouts(np.ones(5, dtype=int)) == [(0, 4)]


def test_detect_bouts_trailing_open():
    # Final bout runs to the end of the array and must be closed.
    assert ac.detect_bouts(np.array([0, 1, 1])) == [(1, 2)]
    assert ac.detect_bouts(np.array([1, 0, 1])) == [(0, 0), (2, 2)]
    assert ac.detect_bouts(np.array([0, 1, 1, 0, 1, 1, 1])) == [(1, 2), (4, 6)]


# ---------------------------------------------------------------------------
# pick_prediction_column
# ---------------------------------------------------------------------------

def test_pick_prediction_column_filtered_wins():
    df = pd.DataFrame({
        'prediction_raw': [1, 1, 1],
        'prediction_filtered': [0, 1, 0],
        'other': [9, 9, 9],
    })
    assert list(ac.pick_prediction_column(df)) == [0, 1, 0]


def test_pick_prediction_column_raw_then_licking():
    df = pd.DataFrame({'prediction_raw': [1, 0], 'Left_licking': [0, 0]})
    assert list(ac.pick_prediction_column(df)) == [1, 0]
    df = pd.DataFrame({'frame': [0, 1], 'Left_licking': [0, 1], 'x': [0, 0]})
    assert list(ac.pick_prediction_column(df)) == [0, 1]
    df = pd.DataFrame({'frame': [0, 1], 'Right_licking': [1, 1], 'x': [0, 0]})
    assert list(ac.pick_prediction_column(df)) == [1, 1]


def test_pick_prediction_column_last_column_fallback():
    df = pd.DataFrame({'frame': [0, 1, 2], 'probability': [0.9, 0.2, 0.8],
                       'Scratching': [1, 0, 1]})
    assert list(ac.pick_prediction_column(df)) == [1, 0, 1]


def test_pick_prediction_column_binarizes():
    # Non-binary values are thresholded at > 0.5
    df = pd.DataFrame({'frame': [0, 1, 2], 'prob': [0.9, 0.2, 0.51]})
    assert list(ac.pick_prediction_column(df)) == [1, 0, 1]


# ---------------------------------------------------------------------------
# calculate_metrics
# ---------------------------------------------------------------------------

def test_calculate_metrics_hand_computed():
    preds = np.array([0, 0, 1, 1, 0, 1])
    fps = 2.0
    bin_dur = 3.0  # 6 frames / 2 fps
    m = ac.calculate_metrics(preds, fps, bin_dur)
    assert m['Total_Time_s'] == 1.5              # 3 frames / 2 fps
    assert m['N_Bouts'] == 2                     # (2,3) and (5,5)
    assert abs(m['Mean_Bout_Duration_s'] - 0.75) < 1e-12   # (1.0 + 0.5)/2
    assert abs(m['Bout_Frequency_per_min'] - 40.0) < 1e-12  # 2/3s * 60
    assert m['Latency_In_Bin_s'] == 1.0          # first pos at frame 2 / 2 fps
    assert m['Percent_Time'] == 50.0
    assert 'AUC' not in m                        # dropped by design


def test_calculate_metrics_no_behavior():
    m = ac.calculate_metrics(np.zeros(4, dtype=int), 2.0, 2.0)
    assert m['Total_Time_s'] == 0
    assert m['N_Bouts'] == 0
    assert m['Mean_Bout_Duration_s'] == 0
    assert m['Bout_Frequency_per_min'] == 0
    assert m['Latency_In_Bin_s'] is None
    assert m['Percent_Time'] == 0


# ---------------------------------------------------------------------------
# resolve_subject ladder
# ---------------------------------------------------------------------------

def _key(subjects):
    return pd.DataFrame({'Subject': [str(s) for s in subjects],
                         'Treatment': ['X'] * len(subjects)})


def test_resolve_subject_key_token_match():
    key = _key(['mouse1', 'mouse10'])
    assert ac.resolve_subject('mouse10_veh_classifier_A_predictions.csv', key) == 'mouse10'
    assert ac.resolve_subject('exp_mouse1_B_bouts.csv', key) == 'mouse1'


def test_resolve_subject_prefix_strip():
    # No key match → prefix rung: strip prefix, take first token.
    assert ac.resolve_subject('EXPsubjA_video_predictions.csv', None,
                              filename_prefix='EXP') == 'subjA'


def test_resolve_subject_legacy_4digit():
    # Legacy heuristic rung (skipped silently if the helper can't import).
    got = ac.resolve_subject('260129_Formalin_2801_PixelPaws_predictions.csv', None)
    assert got in ('2801', '260129_Formalin_2801_PixelPaws')


def test_resolve_subject_stem_fallback():
    # No key, no prefix, no 4-digit id → full cleaned stem.
    assert ac.resolve_subject('ratX_predictions.csv', None) == 'ratX'


# ---------------------------------------------------------------------------
# extract_behavior_name ladder
# ---------------------------------------------------------------------------

def test_extract_behavior_folder_wins():
    assert ac.extract_behavior_name(
        'mouse1_veh_classifier_Left_licking_predictions.csv', 'Left_licking') \
        == 'Left_licking'
    # Arbitrary behavior folder name wins even over the classifier marker
    assert ac.extract_behavior_name(
        'mouse1_veh_classifier_Left_licking_predictions.csv', 'MyBehavior') \
        == 'MyBehavior'


def test_extract_behavior_classifier_marker():
    # 'Results' folders are NOT behavior folders → falls to the marker split.
    assert ac.extract_behavior_name(
        'mouse1_veh_classifier_Facial_grooming_predictions.csv',
        'Mouse1_PixelPaws_Results') == 'Facial_grooming'
    assert ac.extract_behavior_name(
        '260129_Formalin_2801_PixelPaws_Left_licking_predictions.csv', None) \
        == 'Left_licking'


def test_extract_behavior_heuristic_fallback():
    # No marker: date (6-digit), id (4-digit), experiment words stripped.
    assert ac.extract_behavior_name(
        '260129_Formalin_2801_Left_licking_predictions.csv', None) \
        == 'Left_licking'
    assert ac.extract_behavior_name('2801_predictions.csv', None) == 'Unknown'


# ---------------------------------------------------------------------------
# perform_statistical_test
# ---------------------------------------------------------------------------

def test_stats_two_group_parametric_forced():
    data = {'A': [1.0, 2.0, 3.0, 4.0, 5.0], 'B': [6.0, 7.0, 8.0, 9.0, 10.0]}
    r = ac.perform_statistical_test(data, alpha=0.05, paradigm='parametric')
    assert r['test_type'] == "Welch's t-test"
    assert r['significant'] is True or r['significant'] == True  # noqa: E712
    assert r['effect_size_type'] == "Cohen's d"
    assert r['comparison'] == 'A vs B'


def test_stats_two_group_nonparametric_forced():
    data = {'A': [1.0, 2.0, 3.0, 4.0, 5.0], 'B': [6.0, 7.0, 8.0, 9.0, 10.0]}
    r = ac.perform_statistical_test(data, alpha=0.05, paradigm='nonparametric')
    assert r['test_type'] == 'Mann-Whitney U'
    assert r['effect_size_type'] == "Cohen's d"


def test_stats_three_group_anova_and_bonferroni():
    data = {'A': [1.0, 2.0, 3.0, 4.0],
            'B': [10.0, 11.0, 12.0, 13.0],
            'C': [20.0, 21.0, 22.0, 23.0]}
    r = ac.perform_statistical_test(data, alpha=0.05, paradigm='parametric')
    assert r['test_type'] == 'ANOVA'
    assert r['significant']
    assert r['effect_size_type'] == 'eta-squared'
    assert r['pairwise_correction'] == 'bonferroni'
    assert len(r['pairwise']) == 3
    for pair, res in r['pairwise'].items():
        assert res['p_corrected'] == min(res['p_raw'] * 3, 1.0)
        assert res['p_value'] == res['p_corrected']
        assert res['significant'] == (res['p_corrected'] < 0.05)


def test_stats_kruskal_three_group():
    data = {'A': [1.0, 2.0, 3.0, 4.0],
            'B': [10.0, 11.0, 12.0, 13.0],
            'C': [20.0, 21.0, 22.0, 23.0]}
    r = ac.perform_statistical_test(data, alpha=0.05, paradigm='nonparametric')
    assert r['test_type'] == 'Kruskal-Wallis'


def test_stats_empty_and_single_group():
    assert ac.perform_statistical_test({'A': [1, 2], 'B': []}) is None
    assert ac.perform_statistical_test({'A': [1, 2]}) is None


# ---------------------------------------------------------------------------
# timecourse_posthoc
# ---------------------------------------------------------------------------

def _timecourse_df(treatments, bins, n_per_group=4, offset_per_group=10.0):
    rng = np.random.RandomState(0)
    rows = []
    for bi, b in enumerate(bins):
        for gi, t in enumerate(treatments):
            for k in range(n_per_group):
                rows.append({'Subject': f'{t}{k}', 'Treatment': t,
                             'Bin_Start_Min': b,
                             'Total_Time_s': gi * offset_per_group
                             + rng.rand()})
    return pd.DataFrame(rows)


def test_timecourse_posthoc_shape_two_groups():
    df = _timecourse_df(['Veh', 'Drug'], [0.0, 5.0, 10.0])
    out = ac.timecourse_posthoc(df, ['Veh', 'Drug'], 'Total_Time_s',
                                alpha=0.05, paradigm='parametric')
    assert list(out.columns) == ['Bin_Start_Min', 'group_a', 'group_b',
                                 'p_raw', 'p_corrected', 'significant']
    assert len(out) == 3            # 1 pair × 3 bins
    # With one pair, Bonferroni is a no-op
    assert (out['p_raw'] == out['p_corrected']).all()
    assert out['significant'].all()


def test_timecourse_posthoc_three_groups_bonferroni():
    df = _timecourse_df(['A', 'B', 'C'], [0.0, 5.0])
    out = ac.timecourse_posthoc(df, ['A', 'B', 'C'], 'Total_Time_s',
                                alpha=0.05, paradigm='parametric')
    assert len(out) == 6            # 3 pairs × 2 bins
    exp = np.minimum(out['p_raw'] * 3, 1.0)
    assert np.allclose(out['p_corrected'], exp)
    assert set(out['Bin_Start_Min']) == {0.0, 5.0}


def test_timecourse_posthoc_empty():
    df = _timecourse_df(['A'], [0.0])
    out = ac.timecourse_posthoc(df, ['A'], 'Total_Time_s')
    assert len(out) == 0
    assert list(out.columns) == ['Bin_Start_Min', 'group_a', 'group_b',
                                 'p_raw', 'p_corrected', 'significant']


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def main():
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith('test_') and callable(f)]
    failed = []
    for name, fn in fns:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:
            failed.append(name)
            import traceback
            traceback.print_exc()
            print(f"FAIL {name}: {e}")
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    if failed:
        sys.exit(1)
    print("ALL GREEN (test_analysis_core)")


if __name__ == '__main__':
    main()
