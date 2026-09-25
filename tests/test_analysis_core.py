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


def test_stats_cohens_d_nan_for_n1_group():
    # n=1 group: pooled SD undefined -> d is NaN (was 0.0); Welch undefined
    r = ac.perform_statistical_test({'a': [3.0], 'b': [5.0, 6.0, 7.0]},
                                    paradigm='parametric')
    assert np.isnan(r['effect_size'])
    assert np.isnan(r['p_value'])
    assert r['significant'] is False
    assert 'n < 2' in r['note']
    # pooled SD 0 (both groups constant) -> d undefined too
    r = ac.perform_statistical_test({'a': [1.0, 1.0], 'b': [2.0, 2.0]},
                                    paradigm='parametric')
    assert np.isnan(r['effect_size'])
    assert np.isnan(r['p_value']) and r['significant'] is False
    # a normal case still matches the textbook pooled-SD formula
    a, b = np.array([1.0, 2.0, 4.0]), np.array([3.0, 5.0, 6.0, 8.0])
    sp = np.sqrt((2 * a.var(ddof=1) + 3 * b.var(ddof=1)) / 5)
    r = ac.perform_statistical_test({'a': a, 'b': b}, paradigm='parametric')
    assert abs(r['effect_size'] - abs(a.mean() - b.mean()) / sp) < 1e-12


def test_stats_anova_zero_within_variance_is_undefined():
    # every group constant: scipy gives F=inf, p=0 -> must NOT read significant
    data = {'A': [1.0, 1.0], 'B': [2.0, 2.0], 'C': [3.0, 3.0]}
    r = ac.perform_statistical_test(data, paradigm='parametric')
    assert r['test_type'] == 'ANOVA'
    assert np.isnan(r['p_value'])
    assert r['significant'] is False
    assert np.isnan(r['effect_size'])
    assert 'within-group variance' in r['note']
    assert 'pairwise' not in r
    # n = 1, 1, 2 with a constant pair: same (zero within-group SS)
    r = ac.perform_statistical_test({'a': [3.0], 'b': [5.0], 'c': [4.0, 4.0]},
                                    paradigm='parametric')
    assert np.isnan(r['p_value']) and r['significant'] is False


def test_stats_kruskal_effect_size_is_epsilon_squared_from_h():
    from scipy import stats
    data = {'A': [1, 2, 2, 3, 3], 'B': [2, 3, 3, 4, 4], 'C': [4, 4, 5, 5, 6]}
    r = ac.perform_statistical_test(data, paradigm='nonparametric')
    H = stats.kruskal(*data.values())[0]
    assert r['effect_size_type'].startswith('epsilon-squared')
    assert abs(r['effect_size'] - H / (15 - 1)) < 1e-12
    # rank-based: a monotone transform of the data leaves it unchanged
    r2 = ac.perform_statistical_test(
        {k: np.exp(np.array(v, float)) for k, v in data.items()},
        paradigm='nonparametric')
    assert abs(r2['effect_size'] - r['effect_size']) < 1e-12


def test_stats_pairwise_constant_pair_not_significant():
    # omnibus is defined (group A varies) but B vs C are both constant:
    # Welch would return p=0 for that pair
    data = {'A': [1.0, 2.0, 3.0, 2.5], 'B': [5.0, 5.0, 5.0, 5.0],
            'C': [9.0, 9.0, 9.0, 9.0]}
    r = ac.perform_statistical_test(data, paradigm='parametric')
    assert r['significant']
    bc = r['pairwise']['B_vs_C']
    assert np.isnan(bc['p_raw']) and not bc['significant']


def test_timecourse_anova_states_bins_are_independent():
    df = _timecourse_df(['Veh', 'Drug'], [0.0, 5.0, 10.0])
    an = ac.timecourse_anova(df, 'Total_Time_s')
    assert an is not None
    assert 'between-subjects' in an['note']
    assert 'bins treated as independent' in an['note']
    assert 'independent' in ac.timecourse_anova.__doc__


def test_stats_kruskal_no_variance_reports_note():
    # scipy.kruskal raises "All numbers are identical"; the core must
    # report it per metric instead of letting the whole panel error out.
    data = {'A': [0.0, 0.0, 0.0], 'B': [0.0, 0.0, 0.0], 'C': [0.0, 0.0, 0.0]}
    r = ac.perform_statistical_test(data, alpha=0.05, paradigm='nonparametric')
    assert r['test_type'] == 'Kruskal-Wallis'
    assert np.isnan(r['p_value'])
    assert r['significant'] is False
    assert 'no variance' in r['note']
    assert 'pairwise' not in r


# ---------------------------------------------------------------------------
# per_subject_metric
# ---------------------------------------------------------------------------

def _bins_df():
    # mouse1: one 10 s bout in a 60 min session of 5 min bins (12 bins).
    # mouse2: two bins with bouts (2 x 1 s, 1 x 4 s).  mouse3: none.
    rows = []
    for i in range(12):
        rows.append({'Subject': 'mouse1', 'Treatment': 'Veh', 'Bin_Index': i,
                     'N_Bouts': 1 if i == 0 else 0,
                     'Mean_Bout_Duration_s': 10.0 if i == 0 else 0.0,
                     'Total_Time_s': 10.0 if i == 0 else 0.0})
    rows.append({'Subject': 'mouse2', 'Treatment': 'Drug', 'Bin_Index': 0,
                 'N_Bouts': 2, 'Mean_Bout_Duration_s': 1.0,
                 'Total_Time_s': 2.0})
    rows.append({'Subject': 'mouse2', 'Treatment': 'Drug', 'Bin_Index': 1,
                 'N_Bouts': 1, 'Mean_Bout_Duration_s': 4.0,
                 'Total_Time_s': 4.0})
    rows.append({'Subject': 'mouse3', 'Treatment': 'Drug', 'Bin_Index': 0,
                 'N_Bouts': 0, 'Mean_Bout_Duration_s': 0.0,
                 'Total_Time_s': 0.0})
    return pd.DataFrame(rows)


def test_per_subject_mean_bout_duration_is_bout_weighted():
    df = _bins_df()
    per = ac.per_subject_metric(df, 'Mean_Bout_Duration_s', 'mean')
    per = per.set_index('Subject')['Mean_Bout_Duration_s']
    assert per['mouse1'] == 10.0                # not 10/12 = 0.83
    assert abs(per['mouse2'] - 2.0) < 1e-12     # (2*1 + 1*4) / 3
    assert np.isnan(per['mouse3'])              # no bouts -> NaN


def test_per_subject_other_metrics_unchanged():
    df = _bins_df()
    tot = ac.per_subject_metric(df, 'Total_Time_s', 'sum')
    tot = tot.set_index('Subject')['Total_Time_s']
    assert tot['mouse1'] == 10.0 and tot['mouse2'] == 6.0 and tot['mouse3'] == 0
    nb = ac.per_subject_metric(df, 'N_Bouts', 'sum').set_index('Subject')
    assert nb.loc['mouse2', 'N_Bouts'] == 3
    assert list(tot.reset_index().columns) == ['Subject', 'Total_Time_s']


# ---------------------------------------------------------------------------
# binning: a bout is counted once, in the bin where it starts
# ---------------------------------------------------------------------------

def _crossing_preds():
    # fps 1, 3-frame bins over 10 frames -> bins [0,3) [3,6) [6,9) [9,10)
    # bouts: (1,4) crosses bins 0|1, (6,6) in bin 2, (8,9) crosses bins 2|3
    return np.array([0, 1, 1, 1, 1, 0, 1, 0, 1, 1])


def test_bins_count_each_bout_once_where_it_starts():
    preds = _crossing_preds()
    cfg = ac.AnalysisConfig(bin_size_min=3 / 60.0, fps=1.0)
    rows = ac._bin_rows(preds, cfg, 's', 'Veh', 'B')
    assert [r['N_Bouts'] for r in rows] == [1, 0, 2, 0]
    assert sum(r['N_Bouts'] for r in rows) == len(ac.detect_bouts(preds)) == 3
    # full duration of the bouts that start in the bin
    assert [r['Mean_Bout_Duration_s'] for r in rows] == [4.0, 0, 1.5, 0]
    # frame-based metrics stay inside the bin
    assert [r['Total_Time_s'] for r in rows] == [2.0, 2.0, 2.0, 1.0]
    assert [r['Bin_Duration_s'] for r in rows] == [3.0, 3.0, 3.0, 1.0]
    assert rows[2]['Bout_Frequency_per_min'] == 2 / 3.0 * 60
    # bout-weighted per-subject mean = the session's mean bout duration
    df = pd.DataFrame(rows)
    mbd = ac.per_subject_metric(df, 'Mean_Bout_Duration_s', 'mean')
    assert abs(mbd['Mean_Bout_Duration_s'].iloc[0] - (4 + 1 + 2) / 3) < 1e-12
    nb = ac.per_subject_metric(df, 'N_Bouts', 'sum')
    assert nb['N_Bouts'].iloc[0] == 3


def test_whole_session_and_phase_rows_unchanged_by_start_rule():
    preds = _crossing_preds()
    whole = ac._bin_rows(preds, ac.AnalysisConfig(whole_session=True, fps=1.0),
                         's', 'Veh', 'B')
    assert len(whole) == 1 and whole[0]['N_Bouts'] == 3
    assert whole[0]['Bin_Duration_s'] == 10.0
    # phase windows count the bouts visible inside the window (clipped)
    ph = ac.phase_rows(preds, 1.0, 's', 'Veh', 'B',
                       (ac.PhaseWindow('Acute', 0, 3 / 60.0, -1),))
    assert ph[0]['N_Bouts'] == 1 and ph[0]['Total_Time_s'] == 2.0


def test_per_subject_rates_weight_partial_last_bin():
    preds = _crossing_preds()
    rows = ac._bin_rows(preds, ac.AnalysisConfig(bin_size_min=3 / 60.0,
                                                 fps=1.0), 's', 'Veh', 'B')
    df = pd.DataFrame(rows)
    pt = ac.per_subject_metric(df, 'Percent_Time', 'mean')['Percent_Time']
    assert abs(pt.iloc[0] - preds.mean() * 100) < 1e-12          # 70 %
    assert abs(df['Percent_Time'].mean() - 70.0) > 1               # plain mean is off
    bf = ac.per_subject_metric(df, 'Bout_Frequency_per_min', 'mean')
    assert abs(bf['Bout_Frequency_per_min'].iloc[0] - 3 / 10 * 60) < 1e-12
    # no Bin_Duration_s column (older results) -> the requested aggregation
    legacy = df.drop(columns=['Bin_Duration_s'])
    pt2 = ac.per_subject_metric(legacy, 'Percent_Time', 'mean')['Percent_Time']
    assert abs(pt2.iloc[0] - df['Percent_Time'].mean()) < 1e-12


# ---------------------------------------------------------------------------
# subjects_overview (whole-token subject matching)
# ---------------------------------------------------------------------------

def test_subjects_overview_matches_whole_tokens_only():
    import tempfile
    key = pd.DataFrame({'Subject': ['mouse1', 'mouse10', '1'],
                        'Treatment': ['Veh', 'Drug', 'Sham']})
    with tempfile.TemporaryDirectory() as d:
        vd = os.path.join(d, 'videos')
        os.makedirs(vd)
        for stem in ('mouse1_veh', 'mouse10_veh', 'S1_run', 'mouse100'):
            open(os.path.join(vd, stem + '.mp4'), 'w').close()
        rows = dict(ac.subjects_overview(d, key_df=key))
    assert rows['mouse1_veh'] == 'Veh'
    assert rows['mouse10_veh'] == 'Drug'      # substring match would say Veh
    assert rows['S1_run'] == '-'              # '1' is not a token of 'S1_run'
    assert rows['mouse100'] == '-'


# ---------------------------------------------------------------------------
# run_analysis progress callback
# ---------------------------------------------------------------------------

def test_run_analysis_progress_cb_one_message_per_file():
    key = pd.DataFrame({'Subject': ['mouse1'], 'Treatment': ['Veh']})
    files = [ac.FileInfo(path=f'/nonexistent/mouse{i}_x.csv', folder=None,
                         filename=f'mouse{i}_x.csv', behavior='Licking')
             for i in (1, 2)]
    got = []
    res = ac.run_analysis(files, key, ac.AnalysisConfig(),
                          progress_cb=got.append)
    assert len(got) == 2 and all(isinstance(m, str) for m in got)
    assert '(1/2)' in got[0] and '(2/2)' in got[1]
    assert len(res.skipped) == 2


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
