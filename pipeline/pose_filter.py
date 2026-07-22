"""PixelPaws pose-trajectory outlier filter (Part D prototype).

Cleans a DeepLabCut pose h5 BEFORE feature extraction so brightness ROIs + velocity features
use sane coordinates. Synthesis of what peer pipelines do (DLC outlier_frames "jump", SimBA
movement/location correction, Anipose median+interp):

  1. LIKELIHOOD gate  - NaN points below p_cut (DLC "uncertain", p_bound style).
  2. LOCATION criterion (SimBA, self-scaling) - reference = median inter-frame distance between two
     anatomical landmarks (a body-length proxy, default neck<->tailbase). A bodypart is flagged in a
     frame if its distance to >= `loc_min_violations` OTHER bodyparts exceeds reference * loc_criterion
     (an anatomically implausible "teleport"). Catches CONFIDENTLY-WRONG points that a likelihood gate
     misses - the usual driver of false scratch detections (a hindpaw keypoint snapping away from the
     body inflates its ROI brightness / optical flow).
  3. MOVEMENT/velocity criterion - OPTIONAL and OFF for the hindpaws by design: scratching IS high
     hindpaw velocity, so a raw inter-frame velocity gate would erase the signal we classify
     (cf. the locomotion-gate dead-end). Only enable for parts where fast motion is non-biological.
  4. CORRECTION - flagged -> NaN, then linear-interpolate gaps <= max_interp_gap; longer gaps are
     forward/back-filled so every frame keeps a coordinate for ROI sampling.

Returns the cleaned DataFrame (same MultiIndex columns, likelihood preserved) + a stats dict.
Pure pandas/numpy; no DLC import. Use `filter_h5(path)` to read+clean+write a `_ppfilt.h5`.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

DEFAULTS = dict(
    p_cut=0.3,                # only drop genuinely-uncertain points (feature extractor already
                              # applies min_prob=0.8 internally); the location criterion is the lever
    ref_bps=("snout", "tailbase"),   # robust body-length proxy from RELIABLY-tracked landmarks
    loc_criterion=1.5,        # SimBA default-ish; threshold = reference * this
    loc_min_violations=2,
    max_interp_gap=5,
    # JUMP/teleport gate (2026-06-29): flag any frame where the bodypart moves more than
    # vel_criterion*reference (~100px at a typical ~360px body length) between frames — a
    # non-biological inter-frame "teleport". Enabled for every dot EXCEPT the hindpaws.
    # Calibrated from ground truth: forepaw motion during confirmed tremor maxes ~60px
    # (0 frames >100px) and snout/neck/centroid/tail have no fast independent behaviour,
    # so >100px is unambiguously a teleport for them. Hindpaws are EXCLUDED because real
    # scratching genuinely reaches >100px (~1.4% of hlpaw frames, up to ~267px) — gating
    # them would clip real scratch strokes (cf. the locomotion-gate dead-end).
    velocity_bps=("snout", "frpaw", "flpaw", "neck", "centroid", "tailbase", "tailtip"),
    vel_criterion=0.28,       # multiples of reference per frame (~100px at ref≈360px)
)


def _coords(df, scorer, bp):
    x = df[(scorer, bp, "x")].to_numpy(dtype=float)
    y = df[(scorer, bp, "y")].to_numpy(dtype=float)
    p = df[(scorer, bp, "likelihood")].to_numpy(dtype=float)
    return x, y, p


def _interp_gaps(a, max_gap):
    """Linear-interpolate NaN runs of length <= max_gap; ffill/bfill the rest."""
    s = pd.Series(a)
    isna = s.isna().to_numpy()
    if isna.all():
        return s.to_numpy()
    # mark NaN runs longer than max_gap so interpolate(limit) leaves them, then fill
    out = s.interpolate(method="linear", limit=max_gap, limit_area="inside").to_numpy()
    out = pd.Series(out).ffill().bfill().to_numpy()
    return out


def filter_pose(df: pd.DataFrame, **kw):
    """Filter a DLC CollectedData/analyze DataFrame. Returns (cleaned_df, stats)."""
    o = {**DEFAULTS, **kw}
    scorer = df.columns.get_level_values(0)[0]
    bps = list(dict.fromkeys(df.columns.get_level_values(1)))
    n = len(df)
    X = {bp: _coords(df, scorer, bp) for bp in bps}
    flagged = {bp: np.zeros(n, dtype=bool) for bp in bps}
    reasons = {bp: {"likelihood": 0, "location": 0, "velocity": 0} for bp in bps}

    # 1) likelihood gate
    for bp in bps:
        m = X[bp][2] < o["p_cut"]
        flagged[bp] |= m
        reasons[bp]["likelihood"] = int(m.sum())

    # reference distance (robust scalar body-length proxy)
    a, b = o["ref_bps"]
    if a in X and b in X:
        ax, ay, _ = X[a]; bx, by, _ = X[b]
        ref = np.nanmedian(np.hypot(ax - bx, ay - by))
    else:
        # fallback: median pairwise spread of all bodyparts
        ref = np.nanmedian([np.nanmedian(np.hypot(X[i][0] - X[j][0], X[i][1] - X[j][1]))
                            for i in bps for j in bps if i != j])
    loc_thr = ref * o["loc_criterion"]

    # 2) location criterion - flag bp if far from >= loc_min_violations other bps in a frame
    XY = {bp: (X[bp][0], X[bp][1]) for bp in bps}
    for bp in bps:
        bxv, byv = XY[bp]
        far = np.zeros(n, dtype=int)
        for other in bps:
            if other == bp:
                continue
            oxv, oyv = XY[other]
            d = np.hypot(bxv - oxv, byv - oyv)
            far += (d > loc_thr).astype(int)
        m = far >= o["loc_min_violations"]
        reasons[bp]["location"] = int((m & ~flagged[bp]).sum())
        flagged[bp] |= m

    # 3) optional velocity criterion (NOT hindpaws)
    vel_thr = ref * o["vel_criterion"]
    for bp in o["velocity_bps"]:
        if bp not in X:
            continue
        bxv, byv = XY[bp]
        disp = np.hypot(np.diff(bxv, prepend=bxv[0]), np.diff(byv, prepend=byv[0]))
        m = disp > vel_thr
        reasons[bp]["velocity"] = int((m & ~flagged[bp]).sum())
        flagged[bp] |= m

    # 4) correct: NaN flagged coords, interpolate short gaps
    out = df.copy()
    for bp in bps:
        x, y, _ = X[bp]
        x = x.copy(); y = y.copy()
        x[flagged[bp]] = np.nan; y[flagged[bp]] = np.nan
        out[(scorer, bp, "x")] = _interp_gaps(x, o["max_interp_gap"])
        out[(scorer, bp, "y")] = _interp_gaps(y, o["max_interp_gap"])

    stats = dict(n_frames=n, reference_px=float(ref), loc_thr_px=float(loc_thr),
                 flagged={bp: int(flagged[bp].sum()) for bp in bps},
                 reasons=reasons, params=o, flagged_mask=flagged)
    return out, stats


def filter_h5(in_path, out_path=None, key="df_with_missing", **kw):
    """Read a DLC h5, filter, write `<stem>_ppfilt.h5` (or out_path). Returns (out_path, stats)."""
    import os
    df = pd.read_hdf(in_path)
    cleaned, stats = filter_pose(df, **kw)
    if out_path is None:
        base, ext = os.path.splitext(str(in_path))
        out_path = base + "_ppfilt" + ext
    cleaned.to_hdf(out_path, key=key, mode="w")
    return out_path, stats
