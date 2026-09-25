"""
Pose Feature Extraction Module

Extracts kinematic and spatial features from DeepLabCut pose estimation data.

Features extracted:
- Pairwise distances between body parts
- Joint angles (3-point law-of-cosines)
- Velocities and accelerations at multiple timescales
- Distance velocities (rate of change)
- Body part visibility (in-frame probability)
- Paw height, jerk, convex hull, body elongation, bilateral asymmetry
- Rolling velocity statistics (max, std over short windows)
- Angular velocity (rate of change of joint angles)
- Signed velocity components (x and y directions separately)
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import itertools

# Increment this when the feature set changes so cached files are invalidated
POSE_FEATURE_VERSION = 5


# Module-level LRU cache for parsed DLC files. Keyed by (normpath, mtime, size)
# — including size catches mtime-preserving operations like `cp -p`, Dropbox
# re-download of a corrupted file, and `git checkout` round-trips that can
# leave mtime unchanged but content different. Bounded to avoid blowing up
# memory in long-running sessions. Populated lazily by `load_dlc_data`.
_DLC_LOAD_CACHE: "dict[tuple[str, float, int], pd.DataFrame]" = {}
_DLC_LOAD_CACHE_MAX = 8


def read_dlc_table(filepath: str) -> pd.DataFrame:
    """Read a DeepLabCut pose file (.h5 or .csv) as a flat table.

    DLC labels each column (scorer, [individual,] body part, coordinate). The
    scorer is dropped, the rest is joined with "_", and DLC's "likelihood" is
    called "prob": hlpaw_x, hlpaw_y, hlpaw_prob, ... One row per frame.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".h5":
        table = pd.read_hdf(filepath)
    elif ext == ".csv":
        # Three header rows, frame number in the first column. round_trip parsing
        # turns every decimal string into the same float Python's float() would.
        table = pd.read_csv(filepath, header=[0, 1, 2], index_col=0, float_precision="round_trip")
        table = table.astype(float)
    else:
        raise ValueError(f"Unsupported file format: {filepath}. Only .h5 and .csv are supported.")
    table.columns = ["_".join(str(level) for level in col[1:]).replace("likelihood", "prob")
                     for col in table.columns]
    return table


def _bp_names(bp_xcord: pd.DataFrame) -> List[str]:
    """Body-part names from the x-coordinate columns ('hlpaw_x' -> 'hlpaw')."""
    return [str(c).replace("_x", "") for c in bp_xcord.columns]


class PoseFeatureExtractor:
    """
    Extracts kinematic and spatial features from pose estimation data.
    """
    
    def __init__(self,
                 bodyparts: List[str],
                 likelihood_threshold: float = 0.8,
                 velocity_delta: int = 2,
                 contact_threshold: float = 15.0,
                 mm_per_pixel: Optional[float] = None):
        """
        Initialize pose feature extractor.

        Args:
            bodyparts: List of body part names from DLC
            likelihood_threshold: Minimum confidence for including data points (default 0.8)
            velocity_delta: Time steps for middle velocity calculation (default 2)
                           Note: Velocities are always calculated for dt=1, dt=velocity_delta, and dt=10
            contact_threshold: Height below which a body part is considered in contact.
                Default 15.0 is in pixels (legacy). When ``mm_per_pixel`` is
                set and contact_threshold is left at the default, it is
                auto-converted to the equivalent mm value (15 px × mm_per_pixel).
                When you pass a non-default contact_threshold AND mm_per_pixel,
                you are expected to supply the threshold in mm.
            mm_per_pixel: Per-pixel calibration (mm). When set (e.g. from
                PawCapture metadata via ``pawcapture_meta.read_calibration``
                or a project-fixed value), all coordinate-derived features
                (distances, velocities, contact heights) come out in
                physical units (mm) instead of pixels. Angles and
                probabilities are unaffected. ``feature_cache`` includes
                this value in the cache key so mm and pixel caches do
                not collide. Default ``None`` keeps legacy pixel behaviour.
        """
        self.bodyparts = bodyparts
        self.likelihood_threshold = likelihood_threshold
        self.velocity_delta = velocity_delta
        self.mm_per_pixel = mm_per_pixel
        # Auto-convert the legacy 15-px default to mm when calibration is on
        # and the caller did not override the threshold. Keeps existing
        # rig-tuned thresholds intact for callers that already pass a value.
        if mm_per_pixel is not None and contact_threshold == 15.0:
            contact_threshold = 15.0 * float(mm_per_pixel)
        self.contact_threshold = contact_threshold
        
    def load_dlc_data(self, filepath: str) -> pd.DataFrame:
        """
        Load DeepLabCut H5 or CSV file, with a module-level mtime-keyed cache.

        Repeated loads of the same file in one process (e.g. transitions tab
        looping over K classifiers per session, or eval re-reading after a
        brightness-preserve fallback) skip the parse entirely. Cache entries
        are auto-invalidated when the file's mtime changes.
        """
        try:
            _stat = os.stat(filepath)
            mtime = _stat.st_mtime
            size = _stat.st_size
        except OSError:
            mtime = 0.0
            size = -1
        key = (os.path.normpath(filepath), mtime, size)
        cached = _DLC_LOAD_CACHE.get(key)
        if cached is not None:
            return cached.copy()
        if len(_DLC_LOAD_CACHE) >= _DLC_LOAD_CACHE_MAX:
            _DLC_LOAD_CACHE.pop(next(iter(_DLC_LOAD_CACHE)))
        df = self._load_dlc_data_uncached(filepath)
        _DLC_LOAD_CACHE[key] = df
        return df.copy()

    def _load_dlc_data_uncached(self, filepath: str) -> pd.DataFrame:
        """Parse a DLC H5 or CSV file from disk (no caching)."""
        return read_dlc_table(filepath)

    def get_bodypart_coords(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract x, y coordinates and probabilities for all body parts.

        Expects flattened columns: bodypart_x, bodypart_y, bodypart_prob

        When ``self.mm_per_pixel`` is set, x/y coordinates are scaled
        to millimetres so every downstream feature (distances,
        velocities, contact heights) emerges in physical units.
        Probabilities are not scaled.

        Returns:
            Tuple of (x_coords, y_coords, probabilities)
        """
        # DLC writes x, y and likelihood for each body part in turn, so every
        # third column is one kind. Requested body parts match by substring.
        def every_third(offset: int) -> pd.DataFrame:
            block = df.iloc[:, offset::3].reset_index(drop=True)
            if self.bodyparts:
                wanted = [c for c in block.columns if any(bp in c for bp in self.bodyparts)]
                block = block[wanted]
            return block

        bp_xcord, bp_ycord, bp_prob = every_third(0), every_third(1), every_third(2)

        # Validate that we got some body parts
        if bp_xcord.empty or bp_ycord.empty:
            available_bodyparts = list(set([c.split('_x')[0] for c in df.columns if '_x' in c]))
            raise ValueError(
                f"No matching body parts found in DLC file.\n"
                f"Requested: {self.bodyparts}\n"
                f"Available in file: {available_bodyparts}"
            )

        # Apply mm/pixel calibration once at the entry point so every
        # downstream feature (distances, velocities, contact heights,
        # multiscale rolling stats) emerges in mm. Cache key includes
        # mm_per_pixel (feature_cache.compute_hash) so px and mm caches
        # do not collide.
        if self.mm_per_pixel is not None:
            scale = float(self.mm_per_pixel)
            bp_xcord = bp_xcord * scale
            bp_ycord = bp_ycord * scale

        return bp_xcord, bp_ycord, bp_prob
    
    def calculate_distances(self, bp_xcord: pd.DataFrame, bp_ycord: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Euclidean distances between all pairs of body parts.

        Naming convention: Dis_bp1-bp2
        """
        n_bp = len(bp_xcord.columns)
        if n_bp < 2:
            return pd.DataFrame()

        # Single fancy-indexed broadcast over upper-triangle pairs. Roughly
        # matches the per-pair Python loop in wall time at M=15 because the
        # compute is memory-bandwidth bound, but produces one DataFrame in one
        # allocation instead of `pd.concat` of M*(M-1)/2 single-column frames.
        xc = bp_xcord.to_numpy(dtype=np.float64, copy=False)
        yc = bp_ycord.to_numpy(dtype=np.float64, copy=False)
        i_idx, j_idx = np.triu_indices(n_bp, k=1)
        dx = xc[:, i_idx] - xc[:, j_idx]
        dy = yc[:, i_idx] - yc[:, j_idx]
        pair_dists = np.sqrt(dx * dx + dy * dy)

        bp_names = [str(c).replace("_x", "") for c in bp_xcord.columns]
        columns = [f"Dis_{bp_names[i]}-{bp_names[j]}"
                   for i, j in zip(i_idx, j_idx)]
        return pd.DataFrame(pair_dists, columns=columns, index=bp_xcord.index)
    
    def calculate_angles(self, bp_xcord: pd.DataFrame, bp_ycord: pd.DataFrame) -> pd.DataFrame:
        """
        The angle at every body part between every pair of the others, in degrees.

        For ends A and B seen from a vertex V, the law of cosines gives
            cos(angle) = (|VB|^2 + |VA|^2 - |AB|^2) / (2 |VA| |VB|).
        A vertex sitting exactly on A or B has no angle (NaN).

        Columns are Ang_A-V-B (the angle at V), with A before B in column order:
        all vertices for the first pair of ends, then the next pair, and so on.
        """
        names = _bp_names(bp_xcord)
        n = len(names)
        if n < 3:
            return pd.DataFrame()

        x = bp_xcord.to_numpy()
        y = bp_ycord.to_numpy()
        ends = list(itertools.combinations(range(n), 2))
        # Laid out (angle, frame): each column is contiguous, so sums across columns
        # downstream add them one after another, in column order.
        angles = np.empty((len(ends) * (n - 2), len(x)), dtype=np.result_type(x.dtype, y.dtype, np.float32))
        columns = []
        at = 0
        for a, b in ends:
            vertices = [v for v in range(n) if v != a and v != b]
            side_va = np.sqrt((x[:, [a]] - x[:, vertices]) ** 2 + (y[:, [a]] - y[:, vertices]) ** 2)
            side_vb = np.sqrt((x[:, [b]] - x[:, vertices]) ** 2 + (y[:, [b]] - y[:, vertices]) ** 2)
            side_ab = np.sqrt((x[:, [a]] - x[:, [b]]) ** 2 + (y[:, [a]] - y[:, [b]]) ** 2)
            two_va_vb = 2 * side_va * side_vb
            two_va_vb[two_va_vb == 0] = np.nan
            cosine = np.clip((side_vb ** 2 + side_va ** 2 - side_ab ** 2) / two_va_vb, -1, 1)
            angles[at:at + len(vertices)] = np.rad2deg(np.arccos(cosine)).T
            columns.extend(f"Ang_{names[a]}-{names[v]}-{names[b]}" for v in vertices)
            at += len(vertices)
        return pd.DataFrame(angles.T, columns=columns, index=bp_xcord.index, copy=False)

    def calculate_velocities(self, bp_xcord: pd.DataFrame, bp_ycord: pd.DataFrame, t: int = 1) -> pd.DataFrame:
        """
        Speed of every body part: the distance it moved over ``t`` frames, per frame.

        Args:
            bp_xcord: X coordinates
            bp_ycord: Y coordinates
            t: Frame step. Positive looks back t frames, negative looks ahead.

        Returns:
            DataFrame with one <bodypart>_Vel<t> column per body part. Frames with
            no frame t steps away, or with a missing position at either end, are 0.
        """
        x = bp_xcord.to_numpy()
        y = bp_ycord.to_numpy()
        # Laid out (body part, frame), so the per-frame sum over body parts (sum_Vel*)
        # adds them one after another, in column order.
        speed = np.zeros(x.shape[::-1], dtype=np.result_type(x.dtype, y.dtype, np.float32))
        k = abs(t)
        if 0 < k < len(x):
            if t > 0:
                dx, dy, rows = x[k:] - x[:-k], y[k:] - y[:-k], slice(k, None)
            else:
                dx, dy, rows = x[:-k] - x[k:], y[:-k] - y[k:], slice(None, -k)
            speed[:, rows] = (np.sqrt(dx ** 2 + dy ** 2) / k).T
            speed[np.isnan(speed)] = 0
        return pd.DataFrame(speed.T, columns=[f"{name}_Vel{t}" for name in _bp_names(bp_xcord)],
                            index=bp_xcord.index, copy=False)

    def calculate_distance_velocities(self, bp_dist: pd.DataFrame, t: int = 1) -> pd.DataFrame:
        """Change of every pairwise distance over ``t`` frames (0 where there is no frame to compare with)."""
        return bp_dist.diff(periods=t).fillna(0).add_suffix(f"_Vel{t}")

    def calculate_in_frame_probability(self, bp_prob: pd.DataFrame, prob_thresh: float = 0.8) -> pd.DataFrame:
        """
        Calculate binary features indicating if body part is visible in frame.

        Args:
            bp_prob: Probability dataframe
            prob_thresh: Threshold for considering body part visible
            
        Returns:
            Binary dataframe (1 = visible, 0 = not visible)
        """
        bp_inFrame = (bp_prob >= prob_thresh).astype(int)
        bp_inFrame.columns = [col.replace('_prob', '_inFrame_p' + str(prob_thresh)) 
                             for col in bp_prob.columns]
        return bp_inFrame
    
    def calculate_paw_height(self, bp_xcord: pd.DataFrame, bp_ycord: pd.DataFrame,
                             window: int = 500) -> pd.DataFrame:
        """
        Estimate height of each body part above the floor.

        Image y-axis increases downward, so the floor corresponds to the
        rolling maximum of y.  Height > 0 when a paw is lifted.

        Naming convention: bpname_Height
        """
        result_cols = []
        for col in bp_ycord.columns:
            bp_name = col.replace('_y', '')
            floor = bp_ycord[col].rolling(window, min_periods=1).max()
            height = (floor - bp_ycord[col]).clip(lower=0)
            height.name = f'{bp_name}_Height'
            result_cols.append(height)
        if not result_cols:
            return pd.DataFrame()
        return pd.concat(result_cols, axis=1)

    def calculate_acceleration(self, bp_xcord: pd.DataFrame, bp_ycord: pd.DataFrame,
                               t: int = 1) -> pd.DataFrame:
        """
        Calculate magnitude of acceleration for each body part.

        Acceleration = second difference of position / t.
        Naming convention: bpname_Accel{t}
        """
        result_cols = []
        for i, bp_name in enumerate(_bp_names(bp_xcord)):
            vx = bp_xcord.iloc[:, i].diff(t)
            vy = bp_ycord.iloc[:, i].diff(t)
            ax = vx.diff(t)
            ay = vy.diff(t)
            accel = np.sqrt(ax ** 2 + ay ** 2) / t
            accel.name = f'{bp_name}_Accel{t}'
            result_cols.append(accel)
        if not result_cols:
            return pd.DataFrame()
        return pd.concat(result_cols, axis=1)

    def calculate_convex_hull_area(self, bp_xcord: pd.DataFrame,
                                   bp_ycord: pd.DataFrame) -> pd.DataFrame:
        """
        Per-frame area of the convex hull formed by all tracked body parts.

        Returns a single column 'hull_area'.  Frames with fewer than 3 valid
        points receive area = 0.
        """
        try:
            from scipy.spatial import ConvexHull
        except ImportError:
            return pd.DataFrame()

        n_frames = len(bp_xcord)
        areas = np.zeros(n_frames)
        for i in range(n_frames):
            xs = bp_xcord.iloc[i].values.astype(float)
            ys = bp_ycord.iloc[i].values.astype(float)
            mask = ~(np.isnan(xs) | np.isnan(ys))
            pts = np.column_stack([xs[mask], ys[mask]])
            if pts.shape[0] >= 3:
                try:
                    hull = ConvexHull(pts)
                    areas[i] = hull.volume  # In 2-D, .volume is the area
                except Exception:
                    areas[i] = 0
        return pd.DataFrame({'hull_area': areas})

    def calculate_body_elongation(self, bp_xcord: pd.DataFrame,
                                  bp_ycord: pd.DataFrame) -> pd.DataFrame:
        """
        Per-frame aspect ratio (width / height) of the bounding box of all
        tracked body parts.

        Returns a single column 'body_elongation'.  Degenerate frames (zero
        height) get elongation = 1.
        """
        w = bp_xcord.max(axis=1) - bp_xcord.min(axis=1)
        h = bp_ycord.max(axis=1) - bp_ycord.min(axis=1)
        elongation = (w / h.replace(0, np.nan)).fillna(0)
        elongation.name = 'body_elongation'
        return elongation.to_frame()

    def calculate_bilateral_asymmetry(self, bp_xcord: pd.DataFrame,
                                      bp_ycord: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Per-frame distance asymmetry between paired left/right body parts.

        Auto-detects pairs by common prefix conventions:
          hl/hr, fl/fr, left/right, l/r

        Returns two columns per pair: {left}-{right}_AsymX and _AsymY.
        Returns None if no pairs are detected.
        """
        bp_names = [col.replace('_x', '') for col in bp_xcord.columns]
        bp_name_set = set(bp_names)

        # Ordered from most-specific to least-specific to avoid false matches
        LEFT_RIGHT_PATTERNS = [
            ('left', 'right'),
            ('Left', 'Right'),
            ('hl', 'hr'),
            ('fl', 'fr'),
            ('l', 'r'),
        ]

        pairs = []
        seen: set = set()
        for bp in bp_names:
            if bp in seen:
                continue
            for left_pat, right_pat in LEFT_RIGHT_PATTERNS:
                if bp.startswith(left_pat):
                    candidate = right_pat + bp[len(left_pat):]
                    if candidate in bp_name_set and candidate not in seen:
                        pairs.append((bp, candidate))
                        seen.add(bp)
                        seen.add(candidate)
                        break
                elif bp.startswith(right_pat):
                    candidate = left_pat + bp[len(right_pat):]
                    if candidate in bp_name_set and candidate not in seen:
                        pairs.append((candidate, bp))  # store as (left, right)
                        seen.add(bp)
                        seen.add(candidate)
                        break

        if not pairs:
            return None

        result_dfs = []
        for left_bp, right_bp in pairs:
            left_x = bp_xcord[f'{left_bp}_x'].values
            left_y = bp_ycord[f'{left_bp}_y'].values
            right_x = bp_xcord[f'{right_bp}_x'].values
            right_y = bp_ycord[f'{right_bp}_y'].values
            col_prefix = f'{left_bp}-{right_bp}'
            result_dfs.append(pd.DataFrame({
                f'{col_prefix}_AsymX': np.abs(left_x - right_x),
                f'{col_prefix}_AsymY': np.abs(left_y - right_y),
            }))

        return pd.concat(result_dfs, axis=1).reset_index(drop=True)

    def calculate_jerk(self, bp_xcord: pd.DataFrame, bp_ycord: pd.DataFrame,
                       t: int = 1) -> pd.DataFrame:
        """
        Calculate jerk (third derivative of position) for each body part.

        Captures the explosive onset of brief high-speed behaviors like flinches.
        Naming convention: bpname_Jerk{t}
        """
        result_cols = []
        for i, bp_name in enumerate(_bp_names(bp_xcord)):
            vx = bp_xcord.iloc[:, i].diff(t)
            vy = bp_ycord.iloc[:, i].diff(t)
            jx = vx.diff(t).diff(t)
            jy = vy.diff(t).diff(t)
            jerk = np.sqrt(jx ** 2 + jy ** 2) / t
            jerk.name = f'{bp_name}_Jerk{t}'
            result_cols.append(jerk)
        if not result_cols:
            return pd.DataFrame()
        return pd.concat(result_cols, axis=1).fillna(0)

    def calculate_rolling_velocity_stats(self, bp_xcord: pd.DataFrame,
                                         bp_ycord: pd.DataFrame,
                                         windows: tuple = (5, 10)) -> pd.DataFrame:
        """
        Rolling max and std of velocity over short windows.

        Captures the peak velocity even when the classifier frame is slightly
        before/after the motion peak. Critical for brief bouts.
        Naming convention: bpname_Vel1_VelMaxW{w}, bpname_Vel1_VelStdW{w}
        """
        vel = self.calculate_velocities(bp_xcord, bp_ycord, t=1)
        result_cols = []
        for w in windows:
            roll_max = vel.rolling(w, center=True, min_periods=1).max()
            roll_std = vel.rolling(w, center=True, min_periods=1).std().fillna(0)
            roll_max.columns = [f'{c}_VelMaxW{w}' for c in vel.columns]
            roll_std.columns = [f'{c}_VelStdW{w}' for c in vel.columns]
            result_cols.extend([roll_max, roll_std])
        if not result_cols:
            return pd.DataFrame()
        return pd.concat(result_cols, axis=1)

    def calculate_velocity_components(self, bp_xcord: pd.DataFrame,
                                      bp_ycord: pd.DataFrame,
                                      t: int = 1) -> pd.DataFrame:
        """
        Signed x and y velocity components for each body part.

        Directional withdrawal has a specific sign pattern not captured by
        speed magnitude alone.
        Naming convention: bpname_Vx{t}, bpname_Vy{t}
        """
        result_cols = []
        for i, bp_name in enumerate(_bp_names(bp_xcord)):
            vx = bp_xcord.iloc[:, i].diff(t).fillna(0)
            vy = bp_ycord.iloc[:, i].diff(t).fillna(0)
            vx.name = f'{bp_name}_Vx{t}'
            vy.name = f'{bp_name}_Vy{t}'
            result_cols.extend([vx, vy])
        if not result_cols:
            return pd.DataFrame()
        return pd.concat(result_cols, axis=1)

    # ------------------------------------------------------------------
    # Flinch-discriminative features (v4)
    # ------------------------------------------------------------------

    def calculate_velocity_asymmetry(self, bp_xcord: pd.DataFrame,
                                     bp_ycord: pd.DataFrame,
                                     window: int = 30) -> pd.DataFrame:
        """Ratio of peak upward velocity to mean downward velocity.

        Flinches have fast-up / slow-down; stepping is symmetric.

        Returns per body part:
          {bp}_VelAsymmetry  — directional (Vy positive vs negative)
          {bp}_RiseFallRatio — direction-agnostic speed peak/mean ratio
        """
        result_cols = []
        for i, bp_name in enumerate(_bp_names(bp_xcord)):
            vy = bp_ycord.iloc[:, i].diff(1).fillna(0)
            speed = np.sqrt(bp_xcord.iloc[:, i].diff(1).fillna(0) ** 2 + vy ** 2)

            # Directional asymmetry: max(positive Vy) / mean(|negative Vy|)
            pos_vy = vy.clip(lower=0)
            neg_vy_abs = (-vy).clip(lower=0)
            roll_max_pos = pos_vy.rolling(window, min_periods=1).max()
            roll_mean_neg = neg_vy_abs.rolling(window, min_periods=1).mean()
            asym = roll_max_pos / (roll_mean_neg + 1e-6)
            asym.name = f'{bp_name}_VelAsymmetry'
            result_cols.append(asym)

            # Direction-agnostic: rolling max / rolling mean of speed
            roll_max_spd = speed.rolling(window, min_periods=1).max()
            roll_mean_spd = speed.rolling(window, min_periods=1).mean()
            rise_fall = roll_max_spd / (roll_mean_spd + 1e-6)
            rise_fall.name = f'{bp_name}_RiseFallRatio'
            result_cols.append(rise_fall)

        if not result_cols:
            return pd.DataFrame()
        return pd.concat(result_cols, axis=1).fillna(0)

    def calculate_peak_jerk(self, bp_xcord: pd.DataFrame,
                            bp_ycord: pd.DataFrame,
                            peak_window: int = 5,
                            baseline_window: int = 30) -> pd.DataFrame:
        """Peak jerk magnitude and peak-to-baseline ratio.

        Flinch onset has much higher peak-to-baseline jerk than voluntary
        movements.
        """
        jerk_df = self.calculate_jerk(bp_xcord, bp_ycord, t=1)
        result_cols = []
        for col in jerk_df.columns:
            bp_name = col.replace('_Jerk1', '')
            jerk_series = jerk_df[col]
            peak = jerk_series.rolling(peak_window, center=True, min_periods=1).max()
            peak.name = f'{bp_name}_JerkPeakW{peak_window}'
            result_cols.append(peak)

            baseline = jerk_series.rolling(baseline_window, min_periods=1).median()
            ratio = peak / (baseline + 1e-6)
            ratio.name = f'{bp_name}_JerkPeakRatio'
            result_cols.append(ratio)

        if not result_cols:
            return pd.DataFrame()
        return pd.concat(result_cols, axis=1).fillna(0)

    def calculate_height_velocity(self, bp_ycord: pd.DataFrame,
                                  window: int = 500) -> pd.DataFrame:
        """First derivative of paw height at dt=1 and dt=2.

        Captures the *rate* of height change — flinches produce sharp
        height-velocity spikes that static paw height misses.
        """
        height_df = self.calculate_paw_height(
            pd.DataFrame(),  # bp_xcord not used by paw_height
            bp_ycord, window=window)
        if height_df.empty:
            return pd.DataFrame()

        result_cols = []
        for col in height_df.columns:
            bp_name = col.replace('_Height', '')
            for dt in (1, 2):
                hv = height_df[col].diff(dt).fillna(0)
                hv.name = f'{bp_name}_HeightVel{dt}'
                result_cols.append(hv)

        if not result_cols:
            return pd.DataFrame()
        return pd.concat(result_cols, axis=1)

    def calculate_vy_dominance(self, bp_xcord: pd.DataFrame,
                               bp_ycord: pd.DataFrame,
                               t: int = 1) -> pd.DataFrame:
        """Fraction of motion that is vertical: |Vy| / (|Vx| + |Vy| + eps).

        Flinches are primarily vertical; grooming/locomotion have horizontal
        components.
        """
        result_cols = []
        for i, bp_name in enumerate(_bp_names(bp_xcord)):
            vx = bp_xcord.iloc[:, i].diff(t).fillna(0).abs()
            vy = bp_ycord.iloc[:, i].diff(t).fillna(0).abs()
            dom = vy / (vx + vy + 1e-6)
            dom.name = f'{bp_name}_VyDominance'
            result_cols.append(dom)
        if not result_cols:
            return pd.DataFrame()
        return pd.concat(result_cols, axis=1)

    def calculate_acceleration_std(self, bp_xcord: pd.DataFrame,
                                   bp_ycord: pd.DataFrame,
                                   t: int = 1,
                                   window: int = 5) -> pd.DataFrame:
        """Rolling std of acceleration over a short window.

        Flinches produce sharp acceleration spikes = high local std.
        """
        accel_df = self.calculate_acceleration(bp_xcord, bp_ycord, t=t)
        if accel_df.empty:
            return pd.DataFrame()

        result_cols = []
        for col in accel_df.columns:
            bp_name = col.replace(f'_Accel{t}', '')
            astd = accel_df[col].rolling(window, center=True, min_periods=1).std().fillna(0)
            astd.name = f'{bp_name}_AccelStdW{window}'
            result_cols.append(astd)

        if not result_cols:
            return pd.DataFrame()
        return pd.concat(result_cols, axis=1)

    def calculate_contralateral_velocity_corr(self, bp_xcord: pd.DataFrame,
                                              bp_ycord: pd.DataFrame,
                                              window: int = 15) -> pd.DataFrame:
        """Rolling Pearson correlation of left vs right paw velocity.

        Low correlation = unilateral movement (flinch); high = bilateral
        (walking).  Reuses the LEFT_RIGHT_PATTERNS from bilateral asymmetry.
        """
        bp_names = [col.replace('_x', '') for col in bp_xcord.columns]
        bp_name_set = set(bp_names)

        LEFT_RIGHT_PATTERNS = [
            ('left', 'right'), ('Left', 'Right'),
            ('hl', 'hr'), ('fl', 'fr'), ('l', 'r'),
        ]

        pairs = []
        seen: set = set()
        for bp in bp_names:
            if bp in seen:
                continue
            for left_pat, right_pat in LEFT_RIGHT_PATTERNS:
                if bp.startswith(left_pat):
                    candidate = right_pat + bp[len(left_pat):]
                    if candidate in bp_name_set and candidate not in seen:
                        pairs.append((bp, candidate))
                        seen.update({bp, candidate})
                        break
                elif bp.startswith(right_pat):
                    candidate = left_pat + bp[len(right_pat):]
                    if candidate in bp_name_set and candidate not in seen:
                        pairs.append((candidate, bp))
                        seen.update({bp, candidate})
                        break

        if not pairs:
            return pd.DataFrame()

        # Compute velocity magnitude for each body part
        vel_map = {}
        for i, col in enumerate(bp_xcord.columns):
            bp_name = col.replace('_x', '')
            vx = bp_xcord.iloc[:, i].diff(1).fillna(0)
            vy = bp_ycord.iloc[:, i].diff(1).fillna(0)
            vel_map[bp_name] = np.sqrt(vx ** 2 + vy ** 2)

        result_cols = []
        for left_bp, right_bp in pairs:
            left_vel = vel_map.get(left_bp)
            right_vel = vel_map.get(right_bp)
            if left_vel is None or right_vel is None:
                continue
            corr = left_vel.rolling(window, min_periods=3).corr(right_vel).fillna(0)
            corr.name = f'{left_bp}-{right_bp}_VelCorr'
            result_cols.append(corr)

        if not result_cols:
            return pd.DataFrame()
        return pd.concat(result_cols, axis=1)

    # ------------------------------------------------------------------
    # Temporal-context features (v5) — general-purpose
    # ------------------------------------------------------------------
    # These plug gaps in v4: "what was the paw doing just *before* the spike",
    # "is motion upward or downward", "how sharp is the onset", and
    # "is there high-frequency wiggle energy".  Despite being motivated by
    # flinch detection, they're useful for any brief/directional event.

    def calculate_pre_event_quiescence(self, bp_xcord: pd.DataFrame,
                                       bp_ycord: pd.DataFrame,
                                       lookback: int = 20,
                                       spike_window: int = 5) -> pd.DataFrame:
        """Inverse rolling variance of position in the window *preceding* each frame.

        Captures "explosive from stillness" — the defining flinch signature
        (nothing in v4 represents this).  High values = paw was still in the
        `lookback` frames leading up to `spike_window` frames before now.

        Returns one column per body part: `{bp}_PreQuiescence`.
        """
        result_cols = []
        for i, bp_name in enumerate(_bp_names(bp_xcord)):
            pos_var = (bp_xcord.iloc[:, i].rolling(lookback, min_periods=3).var()
                       + bp_ycord.iloc[:, i].rolling(lookback, min_periods=3).var())
            # Shift so variance is measured over [t - (lookback+spike_window), t - spike_window]
            pre_var = pos_var.shift(spike_window)
            # Reciprocal so stillness → HIGH value (better for tree splits targeting flinches)
            pre_q = 1.0 / (pre_var + 1e-3)
            pre_q.name = f'{bp_name}_PreQuiescence'
            result_cols.append(pre_q)
        if not result_cols:
            return pd.DataFrame()
        return pd.concat(result_cols, axis=1).fillna(0)

    def calculate_signed_jerk_y(self, bp_xcord: pd.DataFrame,
                                bp_ycord: pd.DataFrame,
                                t: int = 1) -> pd.DataFrame:
        """Signed vertical jerk — sign preserved (unlike `calculate_jerk` magnitude).

        Image y grows downward, so we negate so that POSITIVE = upward motion
        (the flinch direction).  Tree splits can discriminate directional
        events that the magnitude jerk blurs together.

        Returns one column per body part: `{bp}_Jy_signed`.
        """
        result_cols = []
        for i, bp_name in enumerate(_bp_names(bp_xcord)):
            vy = bp_ycord.iloc[:, i].diff(t)
            ay = vy.diff(t)
            jy = -ay.diff(t)   # negate: up (decreasing y) → positive
            jy.name = f'{bp_name}_Jy_signed'
            result_cols.append(jy)
        if not result_cols:
            return pd.DataFrame()
        return pd.concat(result_cols, axis=1).fillna(0)

    def calculate_onset_sharpness(self, bp_xcord: pd.DataFrame,
                                  bp_ycord: pd.DataFrame,
                                  peak_window: int = 5) -> pd.DataFrame:
        """Rolling ratio: peak speed / (distance of peak from window center + 1).

        Flinches peak sharply at the window center (short time-to-peak);
        locomotion has smeared peaks.  Captures the time-axis of onset that
        v4's JerkPeakRatio misses.

        Returns one column per body part: `{bp}_OnsetSharpness`.
        """
        width = 2 * peak_window + 1

        def _onset(arr):
            if arr.size < 2:
                return 0.0
            peak = arr.max()
            if peak <= 0:
                return 0.0
            offset = abs(int(np.argmax(arr)) - (arr.size // 2))
            return peak / (offset + 1.0)

        result_cols = []
        for i, bp_name in enumerate(_bp_names(bp_xcord)):
            vx = bp_xcord.iloc[:, i].diff(1).fillna(0)
            vy = bp_ycord.iloc[:, i].diff(1).fillna(0)
            speed = np.sqrt(vx ** 2 + vy ** 2)
            onset = speed.rolling(width, center=True, min_periods=1).apply(_onset, raw=True)
            onset.name = f'{bp_name}_OnsetSharpness'
            result_cols.append(onset)
        if not result_cols:
            return pd.DataFrame()
        return pd.concat(result_cols, axis=1).fillna(0)

    def calculate_hf_energy(self, bp_xcord: pd.DataFrame,
                            bp_ycord: pd.DataFrame,
                            window: int = 10) -> pd.DataFrame:
        """High-frequency energy proxy per body part.

        Uses a 2nd-order Butterworth high-pass at normalized frequency 0.5
        (~10 Hz at 20 fps, higher at higher fps) then rolling RMS over
        `window` frames.  Falls back to d²-position (second difference) when
        scipy.signal is not available.

        Captures the spectral "wiggle" signature that low-order velocity /
        accel / jerk miss.  Useful for flinches (broadband HF) and grooming
        (sustained HF) — let gain pruning decide per-behavior.

        Returns one column per body part: `{bp}_HFEnergy`.
        """
        try:
            from scipy.signal import butter, filtfilt
            b, a = butter(2, 0.5, btype='highpass')
            _use_butter = True
        except Exception:
            _use_butter = False

        result_cols = []
        for i, bp_name in enumerate(_bp_names(bp_xcord)):
            x = bp_xcord.iloc[:, i].ffill().bfill().fillna(0).values
            y = bp_ycord.iloc[:, i].ffill().bfill().fillna(0).values
            if _use_butter and len(x) > 20:
                try:
                    x_hf = filtfilt(b, a, x)
                    y_hf = filtfilt(b, a, y)
                except Exception:
                    x_hf = np.diff(np.diff(x, prepend=x[0]), prepend=0)
                    y_hf = np.diff(np.diff(y, prepend=y[0]), prepend=0)
            else:
                x_hf = np.diff(np.diff(x, prepend=x[0]), prepend=0)
                y_hf = np.diff(np.diff(y, prepend=y[0]), prepend=0)
            energy = pd.Series(np.sqrt(x_hf ** 2 + y_hf ** 2))
            energy = energy.rolling(window, min_periods=1).mean()
            energy.name = f'{bp_name}_HFEnergy'
            result_cols.append(energy)
        if not result_cols:
            return pd.DataFrame()
        return pd.concat(result_cols, axis=1).fillna(0)

    def calculate_contact_features(
            self, height_df: pd.DataFrame,
            contact_threshold: float = None,
            window: int = 30,
            per_bodypart_threshold: dict = None) -> pd.DataFrame:
        """Derive binary contact state from Height columns.

        Returns per body part:
          {bp}_ContactState      — 1 if Height < threshold, else 0
          {bp}_ContactTransition — diff of ContactState (+1 = foot strike, -1 = toe off)
          {bp}_DutyCycle         — rolling mean of ContactState over *window* frames
        Plus one global column:
          N_InContact            — sum of ContactState across all body parts

        ``per_bodypart_threshold`` is an optional ``{bodypart: threshold}``
        override. The 15 px global default is paw-calibrated; midline
        keypoints like ``centroid`` / ``tailbase`` typically need a much
        looser threshold because they sit higher above the floor than
        paws. Bodyparts not in the dict fall through to ``contact_threshold``.
        """
        if contact_threshold is None:
            contact_threshold = self.contact_threshold
        height_cols = [c for c in height_df.columns if c.endswith('_Height')]
        if not height_cols:
            return pd.DataFrame()
        per_bp = per_bodypart_threshold or {}

        result_cols = []
        contact_states = []
        for col in height_cols:
            bp_name = col.replace('_Height', '')
            thresh = float(per_bp.get(bp_name, contact_threshold))
            state = (height_df[col] < thresh).astype(int)
            state.name = f'{bp_name}_ContactState'
            contact_states.append(state)
            result_cols.append(state)

            transition = state.diff().fillna(0).astype(int)
            transition.name = f'{bp_name}_ContactTransition'
            result_cols.append(transition)

            duty = state.rolling(window, min_periods=1).mean()
            duty.name = f'{bp_name}_DutyCycle'
            result_cols.append(duty)

        n_in_contact = pd.concat(contact_states, axis=1).sum(axis=1)
        n_in_contact.name = 'N_InContact'
        result_cols.append(n_in_contact)

        return pd.concat(result_cols, axis=1).fillna(0)

    def calculate_lag_features(self, feature_df: pd.DataFrame,
                               lags: tuple = (-2, -1, 1, 2),
                               top_n: int = 10,
                               shap_importance: pd.Series = None) -> pd.DataFrame:
        """Add time-shifted copies of top features.

        If shap_importance is provided, selects top_n features by importance.
        Otherwise, selects the top_n highest-variance features.
        Lags are in frames: negative = past, positive = future.
        """
        if shap_importance is not None:
            cols = shap_importance.nlargest(top_n).index.tolist()
            cols = [c for c in cols if c in feature_df.columns]
        else:
            variances = feature_df.var().nlargest(top_n)
            cols = variances.index.tolist()

        lag_dfs = []
        for lag in lags:
            shifted = feature_df[cols].shift(lag).fillna(0)
            sign = f"m{abs(lag)}" if lag < 0 else f"p{lag}"
            shifted.columns = [f"{c}_lag{sign}" for c in cols]
            lag_dfs.append(shifted)
        if not lag_dfs:
            return pd.DataFrame()
        return pd.concat(lag_dfs, axis=1)

    def calculate_multiscale_features(
            self,
            feature_df: pd.DataFrame,
            fps: float,
            base_col_filter=None,
            windows_ms: tuple = (100, 700),
            stats: tuple = ('std', 'max'),
    ) -> pd.DataFrame:
        """Rolling-window statistics over multiple timescales.

        Lean MARS-style multi-timescale aggregation. Returns a DataFrame
        of new columns with names ``{base}_{stat}_{ms}ms`` (e.g.
        ``hlpaw_Vel1_std_100ms``). Caller is responsible for concat-ing
        the result back onto ``feature_df``.

        Window frame counts are derived per-call from the session's
        own fps so the column names stay time-anchored
        (`100ms`, `700ms`) while the actual sliding-window sizes adapt.
        """
        if base_col_filter is None:
            base_col_filter = self._default_multiscale_filter

        cols = [c for c in feature_df.columns if base_col_filter(c)]
        if not cols:
            return pd.DataFrame(index=feature_df.index)

        win_frames = [max(2, int(round(w_ms / 1000.0 * fps)))
                      for w_ms in windows_ms]

        out = {}
        for col in cols:
            s = feature_df[col]
            for w_ms, w_f in zip(windows_ms, win_frames):
                roll = s.rolling(w_f, center=True, min_periods=1)
                if 'std' in stats:
                    out[f'{col}_std_{w_ms}ms'] = roll.std().fillna(0.0)
                if 'max' in stats:
                    out[f'{col}_max_{w_ms}ms'] = roll.max()
                if 'mean' in stats:
                    out[f'{col}_mean_{w_ms}ms'] = roll.mean()
                if 'min' in stats:
                    out[f'{col}_min_{w_ms}ms'] = roll.min()
        return pd.DataFrame(out, index=feature_df.index)

    @staticmethod
    def _default_multiscale_filter(col: str) -> bool:
        """Lean scope: only Vel1 and raw Pix_<bp> columns get expanded."""
        # Skip already-derived feature families
        for tag in ('_lag', '_BrightAccel', '_BrightOnsetPeak',
                    '_BrightAsymmetry', '_SurfaceZ',
                    '_Vel10', '_Vel2', '_VelCorr', 'Ego_',
                    '_ContactState', '_DutyCycle', '_ContactTransition',
                    'baseline_sub', 'std_temporal', '_jerk',
                    'norm_', '_mean_', '_std_', '_min_', '_max_'):
            if tag in col:
                return False
        if col.endswith('_Vel1'):
            return True
        if col.startswith('Pix_'):
            rest = col[4:]
            # raw brightness columns are bare bodypart names: Pix_hlpaw, Pix_snout
            if '_' not in rest and '/' not in rest:
                return True
        return False

    def normalize_egocentric(self, bp_xcord: pd.DataFrame,
                             bp_ycord: pd.DataFrame,
                             reference_bp: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Subtract a reference body part's position from all others.

        If reference_bp is None, uses the centroid of all body parts.
        Returns normalized (bp_xcord, bp_ycord) DataFrames.
        """
        if reference_bp:
            ref_x_col = f'{reference_bp}_x'
            ref_y_col = f'{reference_bp}_y'
            if ref_x_col in bp_xcord.columns and ref_y_col in bp_ycord.columns:
                ref_x = bp_xcord[ref_x_col]
                ref_y = bp_ycord[ref_y_col]
            else:
                ref_x = bp_xcord.mean(axis=1)
                ref_y = bp_ycord.mean(axis=1)
        else:
            ref_x = bp_xcord.mean(axis=1)
            ref_y = bp_ycord.mean(axis=1)

        norm_x = bp_xcord.subtract(ref_x, axis=0)
        norm_y = bp_ycord.subtract(ref_y, axis=0)
        return norm_x, norm_y

    def extract_v4_features_only(self, dlc_file: str) -> pd.DataFrame:
        """Compute only v4-new flinch features (no brightness, no video needed).

        Used for incremental cache upgrades — requires only the DLC file.
        """
        dlc_df = self.load_dlc_data(dlc_file)
        bp_xcord, bp_ycord, bp_prob = self.get_bodypart_coords(dlc_df)
        feature_dfs = []
        for fn in [self.calculate_velocity_asymmetry,
                   self.calculate_peak_jerk,
                   self.calculate_vy_dominance,
                   self.calculate_acceleration_std]:
            result = fn(bp_xcord, bp_ycord)
            if result is not None and not result.empty:
                feature_dfs.append(result)
        # Height velocity
        height_vel = self.calculate_height_velocity(bp_ycord)
        if height_vel is not None and not height_vel.empty:
            feature_dfs.append(height_vel)
        # Contralateral correlation
        contra = self.calculate_contralateral_velocity_corr(bp_xcord, bp_ycord)
        if contra is not None and not contra.empty:
            feature_dfs.append(contra)
        if not feature_dfs:
            return pd.DataFrame()
        return pd.concat(feature_dfs, axis=1).fillna(0)

    def extract_v5_features_only(self, dlc_file: str) -> pd.DataFrame:
        """Compute only v5-new temporal-context features (no brightness, no video needed).

        Used for incremental cache upgrades — requires only the DLC file.
        Returns the four v5 columns per body part: PreQuiescence, Jy_signed,
        OnsetSharpness, HFEnergy.
        """
        dlc_df = self.load_dlc_data(dlc_file)
        bp_xcord, bp_ycord, bp_prob = self.get_bodypart_coords(dlc_df)
        feature_dfs = []
        for fn in [self.calculate_pre_event_quiescence,
                   self.calculate_signed_jerk_y,
                   self.calculate_onset_sharpness,
                   self.calculate_hf_energy]:
            result = fn(bp_xcord, bp_ycord)
            if result is not None and not result.empty:
                feature_dfs.append(result)
        if not feature_dfs:
            return pd.DataFrame()
        return pd.concat(feature_dfs, axis=1).fillna(0)

    def extract_new_kinematics_only(self, dlc_file: str) -> pd.DataFrame:
        """Compute only v3-new kinematic features (no brightness, no video needed).

        Used for incremental cache upgrades — requires only the DLC file.
        """
        dlc_df = self.load_dlc_data(dlc_file)
        bp_xcord, bp_ycord, bp_prob = self.get_bodypart_coords(dlc_df)
        feature_dfs = []
        for fn in [self.calculate_jerk, self.calculate_velocity_components]:
            result = fn(bp_xcord, bp_ycord)
            if result is not None and not result.empty:
                feature_dfs.append(result)
        rolling_stats = self.calculate_rolling_velocity_stats(bp_xcord, bp_ycord, windows=(5, 10))
        if not rolling_stats.empty:
            feature_dfs.append(rolling_stats)
        if not feature_dfs:
            return pd.DataFrame()
        return pd.concat(feature_dfs, axis=1).fillna(0)

    def extract_all_features(self,
                           dlc_file: str,
                           include_angles: bool = True,
                           include_velocities: bool = True,
                           include_distance_velocities: bool = True,
                           include_in_frame: bool = True,
                           include_new_pose: bool = True,
                           include_new_kinematics: bool = True,
                           include_flinch_features: bool = True,
                           include_temporal_context_features: bool = True,
                           include_lag_features: bool = False) -> pd.DataFrame:
        """
        Extract all pose features from DLC file.

        Args:
            dlc_file: Path to DLC H5 or CSV file
            include_angles: Include angle features
            include_velocities: Include velocity features
            include_distance_velocities: Include distance velocity features
            include_in_frame: Include in-frame probability features
            include_new_pose: Include the 5 new coordinate-based features
                (paw height, acceleration, convex hull area, body elongation,
                bilateral asymmetry).  Default True.
            include_new_kinematics: Include v3 kinematic features (jerk, rolling
                velocity stats, signed velocity components).  Default True.

        Returns:
            DataFrame with all pose features
        """
        # Load DLC data
        dlc_df = self.load_dlc_data(dlc_file)
        
        # Get coordinates and probabilities
        bp_xcord, bp_ycord, bp_prob = self.get_bodypart_coords(dlc_df)
        
        # Calculate distances
        bp_distances = self.calculate_distances(bp_xcord, bp_ycord)

        # Initialize feature list - only add distances if not empty
        feature_dfs = []
        if not bp_distances.empty:
            feature_dfs.append(bp_distances)
        
        # Add velocities with multiple time deltas
        if include_velocities:
            # Velocity with dt=1
            bp_vel_1 = self.calculate_velocities(bp_xcord, bp_ycord, t=1)
            if not bp_vel_1.empty:
                feature_dfs.append(bp_vel_1)
                # Sum of all velocities at dt=1
                sum_vel_1 = bp_vel_1.sum(axis=1).to_frame(name='sum_Vel1')
                feature_dfs.append(sum_vel_1)
            
            # Velocity with dt=velocity_delta (usually 2)
            if self.velocity_delta != 1:
                bp_vel_delta = self.calculate_velocities(bp_xcord, bp_ycord, t=self.velocity_delta)
                if not bp_vel_delta.empty:
                    feature_dfs.append(bp_vel_delta)
                    # Sum of all velocities at dt=velocity_delta
                    sum_vel_delta = bp_vel_delta.sum(axis=1).to_frame(name=f'sum_Vel{self.velocity_delta}')
                    feature_dfs.append(sum_vel_delta)
            
            # Velocity with dt=10
            if self.velocity_delta != 10:
                bp_vel_10 = self.calculate_velocities(bp_xcord, bp_ycord, t=10)
                if not bp_vel_10.empty:
                    feature_dfs.append(bp_vel_10)
                    # Sum of all velocities at dt=10
                    sum_vel_10 = bp_vel_10.sum(axis=1).to_frame(name='sum_Vel10')
                    feature_dfs.append(sum_vel_10)
        
        # Add angles
        if include_angles:
            bp_angles = self.calculate_angles(bp_xcord, bp_ycord)
            if not bp_angles.empty:
                feature_dfs.append(bp_angles)
        
        # Add distance velocities
        if include_distance_velocities and not bp_distances.empty:
            bp_dist_vel = self.calculate_distance_velocities(bp_distances, t=self.velocity_delta)
            if not bp_dist_vel.empty:
                feature_dfs.append(bp_dist_vel)
        
        # Add in-frame features
        if include_in_frame:
            bp_in_frame = self.calculate_in_frame_probability(bp_prob, self.likelihood_threshold)
            if not bp_in_frame.empty:
                feature_dfs.append(bp_in_frame)

        # Add new coordinate-based pose features (v2)
        if include_new_pose:
            for fn in [
                self.calculate_paw_height,
                self.calculate_acceleration,
                self.calculate_convex_hull_area,
                self.calculate_body_elongation,
                self.calculate_bilateral_asymmetry,
            ]:
                result = fn(bp_xcord, bp_ycord)
                if result is not None and not result.empty:
                    feature_dfs.append(result)

        # New kinematic features (v3): jerk, rolling velocity stats, signed velocity components
        if include_new_kinematics:
            for fn in [self.calculate_jerk, self.calculate_velocity_components]:
                result = fn(bp_xcord, bp_ycord)
                if result is not None and not result.empty:
                    feature_dfs.append(result)
            rolling_stats = self.calculate_rolling_velocity_stats(bp_xcord, bp_ycord, windows=(5, 10))
            if not rolling_stats.empty:
                feature_dfs.append(rolling_stats)

        # Flinch-discriminative features (v4)
        if include_flinch_features:
            for fn in [self.calculate_velocity_asymmetry,
                       self.calculate_peak_jerk,
                       self.calculate_vy_dominance,
                       self.calculate_acceleration_std]:
                result = fn(bp_xcord, bp_ycord)
                if result is not None and not result.empty:
                    feature_dfs.append(result)
            # Height velocity (depends on paw height)
            height_vel = self.calculate_height_velocity(bp_ycord)
            if height_vel is not None and not height_vel.empty:
                feature_dfs.append(height_vel)
            # Contralateral correlation
            contra = self.calculate_contralateral_velocity_corr(bp_xcord, bp_ycord)
            if contra is not None and not contra.empty:
                feature_dfs.append(contra)

        # v5 temporal-context features — general-purpose (not flinch-only)
        if include_temporal_context_features:
            for fn in [self.calculate_pre_event_quiescence,
                       self.calculate_signed_jerk_y,
                       self.calculate_onset_sharpness,
                       self.calculate_hf_energy]:
                result = fn(bp_xcord, bp_ycord)
                if result is not None and not result.empty:
                    feature_dfs.append(result)

        # Check if we have any features
        if not feature_dfs:
            raise ValueError("No features could be extracted. Check that your DLC file has valid body part data.")

        # Combine all features
        X = pd.concat(feature_dfs, axis=1)

        # Lag/lead features (computed post-concat so variance ranking is global)
        if include_lag_features:
            lag_df = self.calculate_lag_features(X, lags=(-2, -1, 1, 2), top_n=10)
            if not lag_df.empty:
                X = pd.concat([X, lag_df], axis=1)

        # Fill NaN values
        X = X.fillna(0)

        return X


if __name__ == "__main__":
    print("Pose Feature Extraction Module")
    print("=" * 50)
    print("Extracts kinematic and spatial features from DLC pose data")
    print("\nFeatures:")
    print("  - Distances between body part pairs")
    print("  - Angles at joints (3-point angles)")
    print("  - Velocities of body parts")
    print("  - Distance velocities")
    print("  - Body part visibility")
