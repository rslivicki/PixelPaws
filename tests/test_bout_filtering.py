"""Regression tests for evaluation_tab._apply_bout_filtering.

Pre-2026-05-01 the function accepted ``min_after_bout`` but never used
it - silently no-op. These tests pin the corrected semantic so the
regression can't reappear.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluation_tab import _apply_bout_filtering


def _arr(s):
    """Compact pattern syntax: '00111100' -> int8 array."""
    return np.array([int(c) for c in s], dtype=np.int8)


def test_min_bout_removes_short_runs():
    y = _arr('011100111110')   # bouts of length 3 and 5
    out = _apply_bout_filtering(y, min_bout=4, min_after_bout=0, max_gap=0)
    assert (out == _arr('000000111110')).all(), out.tolist()


def test_max_gap_bridges_short_zero_run():
    y = _arr('11100111')
    out = _apply_bout_filtering(y, min_bout=1, min_after_bout=0, max_gap=2)
    assert (out == _arr('11111111')).all(), out.tolist()


def test_max_gap_does_not_bridge_long_zero_run():
    y = _arr('1110000111')
    out = _apply_bout_filtering(y, min_bout=1, min_after_bout=0, max_gap=2)
    assert (out == _arr('1110000111')).all(), out.tolist()


def test_min_after_bout_is_actually_applied():
    # Two bouts (len 3 each) separated by exactly 2 zeros.
    # max_gap=0 so they are NOT merged.
    # min_after_bout=4 means the 2nd bout starts too soon -> dropped.
    y = _arr('11100111')
    out = _apply_bout_filtering(y, min_bout=1, min_after_bout=4, max_gap=0)
    assert (out == _arr('11100000')).all(), out.tolist()


def test_min_after_bout_zero_is_noop():
    # When min_after_bout=0, the new pass is a no-op - verify other
    # post-2026-05-01 behaviour matches pre-fix outputs for that branch.
    y = _arr('11100111')
    out = _apply_bout_filtering(y, min_bout=1, min_after_bout=0, max_gap=0)
    assert (out == y).all(), out.tolist()


def test_min_after_bout_changes_output_vs_zero():
    # Locked-in regression: must produce DIFFERENT outputs for two
    # different min_after_bout values. This is the test that would have
    # caught the silent-noop bug.
    y = _arr('111001110011100111')
    out_a = _apply_bout_filtering(y, min_bout=1, min_after_bout=0, max_gap=0)
    out_b = _apply_bout_filtering(y, min_bout=1, min_after_bout=4, max_gap=0)
    assert not (out_a == out_b).all(), \
        f'min_after_bout had no effect: {out_a.tolist()} == {out_b.tolist()}'


def test_min_after_bout_chain_uses_last_surviving():
    # Three bouts at frames 0-2, 5-7, 10-12 (each len 3, 2-frame zero gaps).
    # min_after_bout=4 with last-surviving-bout semantic:
    #   bout 1 (end=3) survives.
    #   bout 2 (start=5) -> 5-3=2 < 4 -> drop.
    #   bout 3 (start=10) -> measured from LAST SURVIVING end (=3),
    #     so 10-3=7 >= 4 -> survives.
    # This is more behaviorally sensible than "raw" chaining
    # (which would also drop bout 3 due to a ghost reference to dropped bout 2).
    y = _arr('111001110011100')
    out = _apply_bout_filtering(y, min_bout=1, min_after_bout=4, max_gap=0)
    assert (out == _arr('111000000011100')).all(), out.tolist()


def test_min_after_bout_chain_all_too_close():
    # Three close bouts where every gap is < min_after_bout AND
    # gap from bout1's end to bout3's start is ALSO < min_after_bout.
    # Should keep only the first.
    y = _arr('111011101110')   # bouts 0-2, 4-6, 8-10; gaps of 1
    out = _apply_bout_filtering(y, min_bout=1, min_after_bout=8, max_gap=0)
    assert (out == _arr('111000000000')).all(), out.tolist()


def test_combined_min_bout_then_max_gap_then_refractory():
    y = _arr('1101110011')
    # min_bout=2 drops the leading single '1'? No: '11' is len-2, kept.
    # Then a single zero gap inside the first run gets bridged with max_gap=1.
    out = _apply_bout_filtering(y, min_bout=2, min_after_bout=0, max_gap=1)
    assert (out == _arr('1111110011')).all(), out.tolist()


if __name__ == '__main__':
    fails = []
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f'  PASS  {name}')
            except AssertionError as e:
                fails.append((name, e))
                print(f'  FAIL  {name}: {e}')
            except Exception as e:
                fails.append((name, e))
                print(f'  ERROR {name}: {e}')
    if fails:
        sys.exit(1)
    print('all tests passed')
