"""Benchmark + correctness diff between the new vectorized
_apply_bout_filtering and a reference Python-loop implementation.

Verifies they produce identical outputs across many random seeds, then
times the new implementation on a 290k-frame array (typical session
length at 60 fps × 80 min).
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from evaluation_tab import _apply_bout_filtering


def _ref_loop(y_pred, min_bout, min_after_bout, max_gap):
    """Pure-Python reference implementation matching the post-2026-05-01
    semantic. Used to verify the vectorized version. (Roughly the same
    code as before vectorisation.)"""
    y_filtered = np.asarray(y_pred).copy()
    n = len(y_filtered)
    if n == 0:
        return y_filtered

    # min_bout
    in_bout = False
    bout_start = 0
    for i in range(n):
        if y_filtered[i] == 1 and not in_bout:
            bout_start = i; in_bout = True
        elif y_filtered[i] == 0 and in_bout:
            if (i - bout_start) < min_bout:
                y_filtered[bout_start:i] = 0
            in_bout = False
    if in_bout and (n - bout_start) < min_bout:
        y_filtered[bout_start:] = 0

    # max_gap
    if max_gap > 0:
        i = 0
        while i < n:
            if y_filtered[i] == 1:
                gap_start = i + 1
                while gap_start < n and y_filtered[gap_start] == 0:
                    gap_start += 1
                gap_len = gap_start - i - 1
                if 0 < gap_len <= max_gap and gap_start < n:
                    if y_filtered[gap_start] == 1:
                        y_filtered[i + 1:gap_start] = 1
                i = gap_start
            else:
                i += 1

    # min_after_bout
    if min_after_bout > 0:
        prev_end = None
        i = 0
        while i < n:
            if y_filtered[i] == 1:
                j = i + 1
                while j < n and y_filtered[j] == 1:
                    j += 1
                if prev_end is not None and (i - prev_end) < min_after_bout:
                    y_filtered[i:j] = 0
                else:
                    prev_end = j
                i = j
            else:
                i += 1
    return y_filtered


def correctness():
    rng = np.random.default_rng(42)
    n_disagree = 0
    for trial in range(200):
        n = rng.integers(50, 5000)
        p_pos = rng.uniform(0.1, 0.6)
        y = (rng.random(n) < p_pos).astype(np.int8)
        mb = int(rng.integers(1, 20))
        ma = int(rng.integers(0, 15))
        mg = int(rng.integers(0, 12))
        a = _apply_bout_filtering(y.copy(), mb, ma, mg)
        b = _ref_loop(y.copy(), mb, ma, mg)
        if not (a == b).all():
            n_disagree += 1
            if n_disagree <= 3:
                # Show a small diff for debugging
                diffs = np.where(a != b)[0]
                print(f'  trial {trial}: n={n} mb={mb} ma={ma} mg={mg} '
                      f'disagrees at {len(diffs)} frames; first diffs: {diffs[:6].tolist()}')
    if n_disagree == 0:
        print('  PASS  vectorized == reference loop on 200 random inputs')
    else:
        print(f'  FAIL  {n_disagree}/200 trials disagree')


def benchmark():
    rng = np.random.default_rng(0)
    n = 290_000  # ≈ 80 min at 60 fps
    y = (rng.random(n) < 0.3).astype(np.int8)
    mb, ma, mg = 5, 3, 4

    # Warmup
    _apply_bout_filtering(y.copy(), mb, ma, mg)
    _ref_loop(y.copy(), mb, ma, mg)

    n_iter = 5
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _apply_bout_filtering(y.copy(), mb, ma, mg)
    t_vec = (time.perf_counter() - t0) / n_iter

    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ref_loop(y.copy(), mb, ma, mg)
    t_ref = (time.perf_counter() - t0) / n_iter

    print(f'  n={n}  min_bout={mb}  min_after_bout={ma}  max_gap={mg}')
    print(f'  vectorized:  {t_vec*1000:.1f} ms/call')
    print(f'  reference:   {t_ref*1000:.1f} ms/call')
    print(f'  speedup:     {t_ref/t_vec:.1f}x')


if __name__ == '__main__':
    print('--- correctness ---')
    correctness()
    print('\n--- benchmark ---')
    benchmark()
