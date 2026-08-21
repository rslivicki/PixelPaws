"""
Golden replay test: analysis_core.run_analysis must reproduce the OLD
analysis_tab results captured by scripts/capture_analysis_golden.py on
E:/PixelPaws_practice (all shared columns; AUC dropped by design).

Run directly:  python tests/test_golden_practice.py
Skips cleanly (exit 0) when the practice project or goldens are absent.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

PRACTICE = r"E:/PixelPaws_practice"
GOLDEN_DIR = os.path.join(REPO, "tests", "golden")

SORT_COLS = ['Subject', 'Behavior', 'Bin_Index', 'Bin_Start_Min']


def _skip(msg):
    print(f"SKIP: {msg}")
    sys.exit(0)


def _load_config():
    cfg_path = os.path.join(GOLDEN_DIR, "capture_config.json")
    if not os.path.isfile(cfg_path):
        _skip(f"{cfg_path} not found (run scripts/capture_analysis_golden.py)")
    with open(cfg_path, encoding="utf-8") as f:
        return json.load(f)


def _build_cfg(run_cfg, ac):
    phases = tuple(
        ac.PhaseWindow(name=p['name'], start_min=p['start_min'],
                       end_min=p['end_min'], bin_index=p['bin_index'])
        for p in run_cfg['phases'])
    return ac.AnalysisConfig(
        bin_size_min=float(run_cfg['bin_size_min']),
        whole_session=bool(run_cfg['whole_session']),
        fps=float(run_cfg['fps']),
        analyze_mode=run_cfg['analyze_mode'],
        phases=phases,
        filename_prefix=run_cfg.get('filename_prefix', ''))


def _sorted(df):
    out = df.sort_values(SORT_COLS, na_position='last', kind='mergesort')
    return out.reset_index(drop=True)


def _assert_frames_equal(tag, golden, new):
    golden = golden.drop(columns=['AUC'], errors='ignore')
    assert len(golden) == len(new), \
        f"{tag}: row count differs (golden {len(golden)} vs new {len(new)})"

    shared = [c for c in golden.columns if c in new.columns]
    missing = [c for c in golden.columns if c not in new.columns]
    assert not missing, f"{tag}: new results missing golden columns {missing}"

    g = _sorted(golden)
    n = _sorted(new)
    for col in shared:
        gv, nv = g[col], n[col]
        if pd.api.types.is_numeric_dtype(gv) or pd.api.types.is_numeric_dtype(nv):
            ga = pd.to_numeric(gv, errors='coerce').to_numpy(dtype=float)
            na = pd.to_numeric(nv, errors='coerce').to_numpy(dtype=float)
            both_nan = np.isnan(ga) & np.isnan(na)
            close = np.isclose(ga, na, rtol=1e-9, atol=0.0, equal_nan=True)
            ok = close | both_nan
        else:
            ga = gv.astype(str).where(gv.notna(), '<NA>')
            na = nv.astype(str).where(nv.notna(), '<NA>')
            ok = (ga.to_numpy() == na.to_numpy())
        if not ok.all():
            bad = int(np.argmin(ok))
            raise AssertionError(
                f"{tag}: column '{col}' differs at sorted row {bad}: "
                f"golden={g[col].iloc[bad]!r} new={n[col].iloc[bad]!r} "
                f"(row: {g[SORT_COLS].iloc[bad].to_dict()})")
    print(f"{tag}: {len(g)} rows × {len(shared)} shared columns equal "
          f"(AUC dropped from golden by design)")


def main():
    if not os.path.isdir(PRACTICE):
        _skip(f"practice project not found at {PRACTICE}")
    if not os.path.isdir(GOLDEN_DIR):
        _skip(f"golden dir not found at {GOLDEN_DIR}")

    import analysis_core as ac
    from project_config import find_key_files

    config = _load_config()

    # Inputs discovered the same way the capture recorded them.
    pred_folder = os.path.join(PRACTICE, "results")
    files, behaviors = ac.scan_predictions(pred_folder)
    assert files, f"no prediction files found under {pred_folder}"

    key_candidates = find_key_files(PRACTICE)
    assert key_candidates, "no key file found in practice project"
    key_df = ac.load_key_file(key_candidates[0])

    for tag, run_cfg in config['runs'].items():
        golden_path = os.path.join(GOLDEN_DIR, run_cfg['csv'])
        if not os.path.isfile(golden_path):
            _skip(f"golden {golden_path} missing")
        golden = pd.read_csv(golden_path)

        cfg = _build_cfg(run_cfg, ac)
        result = ac.run_analysis(files, key_df, cfg)
        assert not result.skipped, f"{tag}: files skipped: {result.skipped}"

        _assert_frames_equal(tag, golden, result.results_df)

        # Phase coverage sanity: the shortphase golden must contain phase rows
        # and they must survive in the new results too.
        if tag == 'formalin_shortphase':
            n_phase_g = int((golden['Bin_Index'] < 0).sum())
            n_phase_n = int((result.results_df['Bin_Index'] < 0).sum())
            assert n_phase_g > 0 and n_phase_g == n_phase_n, \
                f"{tag}: phase-row count mismatch ({n_phase_g} vs {n_phase_n})"

        # Perframe equality for the default capture.
        if tag == 'default':
            npz_path = os.path.join(GOLDEN_DIR, "practice_perframe_default.npz")
            assert os.path.isfile(npz_path), "perframe golden missing"
            with np.load(npz_path) as npz:
                golden_pf = {k: npz[k] for k in npz.files}
            new_pf = {f"{s}|{t}|{b}": arr
                      for (s, t, b), arr in result.perframe_data.items()}
            assert set(golden_pf) == set(new_pf), \
                (f"perframe key mismatch: only-golden="
                 f"{set(golden_pf) - set(new_pf)} only-new="
                 f"{set(new_pf) - set(golden_pf)}")
            for k in golden_pf:
                assert np.array_equal(golden_pf[k], new_pf[k]), \
                    f"perframe array differs for {k}"
            print(f"default: perframe equal ({len(golden_pf)} arrays)")

    print("ALL GREEN (test_golden_practice)")


if __name__ == '__main__':
    main()
