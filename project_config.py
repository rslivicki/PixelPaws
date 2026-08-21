"""
project_config.py — Centralised Project Configuration
======================================================
Single data model for PixelPaws_project.json, used by:
  - PixelPaws_GUI.py (save_project_config, _load_project_config)
  - project_setup.py  (_save_step2_config, wizard finish)

Provides load/save with merge semantics (new values overwrite, missing
keys are preserved from disk) and safe defaults.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional


CONFIG_FILENAME = 'PixelPaws_project.json'


@dataclass
class ProjectConfig:
    """In-memory representation of a PixelPaws project config.

    Fields mirror the keys written to PixelPaws_project.json.  Not every
    field is always present — ``load()`` fills gaps with safe defaults.
    """
    project_folder: str = ''
    video_ext: str = '.avi'
    behaviors: List[str] = field(default_factory=list)
    behavior_name: str = ''
    bp_include_list: Optional[List[str]] = None
    bp_pixbrt_list: List[str] = field(default_factory=list)
    square_size: List[int] = field(default_factory=lambda: [40])
    pix_threshold: float = 0.3
    include_optical_flow: bool = False
    bp_optflow_list: List[str] = field(default_factory=list)
    roi_size: int = 20
    dlc_config: str = ''
    last_classifier: str = ''

    # Group-assignment key file (added 2026-08-21). Path to a CSV with
    # Subject/Treatment columns, relative to the project folder when
    # possible. Written by KeyFileGeneratorDialog and by the main GUI's
    # key-file discovery; consumed by Analysis/Sequencing/Gait fallbacks.
    key_file: str = ''

    # Frame-rate handling (added 2026-05-07).
    # process_fps is the rate at which features are computed for this
    # project. None = use the video's stored fps (legacy). When videos
    # were captured with held/duplicated frames (see
    # scripts/utilities/diagnose_duplicate_frames.py), set process_fps
    # to the true acquisition rate after running
    # frame_rate_normalize.normalize_project. source_fps_note is free
    # text for provenance (e.g. "duplicated_60_to_30").
    process_fps: Optional[float] = None
    source_fps_note: str = ''

    # PawCapture (camsync) calibration policy (added 2026-05-07).
    # 'auto'  = each video uses its own embedded mm_per_pixel
    #           (read by pawcapture_meta.read_calibration)
    # 'fixed' = project-wide override via fixed_mm_per_pixel
    # 'off'   = features stay in pixels (legacy default)
    calibration_mode: str = 'off'
    fixed_mm_per_pixel: Optional[float] = None

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, project_folder: str) -> 'ProjectConfig':
        """Load config from ``<project_folder>/PixelPaws_project.json``.

        Returns a ``ProjectConfig`` with safe defaults for any missing keys.
        Never raises — returns defaults on any I/O or parse error.
        """
        cfg = cls(project_folder=project_folder)
        config_path = os.path.join(project_folder, CONFIG_FILENAME)
        if not os.path.isfile(config_path):
            return cfg
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: could not load project config {config_path}: {e}")
            return cfg

        # Map JSON keys → dataclass fields (same names)
        for key in (
            'project_folder', 'video_ext', 'behaviors', 'behavior_name',
            'bp_include_list', 'bp_pixbrt_list', 'square_size',
            'pix_threshold', 'include_optical_flow', 'bp_optflow_list',
            'roi_size', 'dlc_config', 'last_classifier', 'key_file',
            'process_fps', 'source_fps_note',
            'calibration_mode', 'fixed_mm_per_pixel',
        ):
            if key in data:
                setattr(cfg, key, data[key])

        # Legacy: roi_size → square_size
        if 'square_size' not in data and 'roi_size' in data:
            rs = data['roi_size']
            cfg.square_size = [int(rs)] if not isinstance(rs, list) else [int(x) for x in rs]

        return cfg

    # ------------------------------------------------------------------
    # Save (merge semantics)
    # ------------------------------------------------------------------

    def save(self, project_folder: str = None) -> None:
        """Save config to ``<folder>/PixelPaws_project.json``.

        Uses merge semantics: loads existing file first, then overlays
        non-empty values from this instance.  This preserves any keys
        that this dataclass doesn't model.
        """
        folder = project_folder or self.project_folder
        if not folder or not os.path.isdir(folder):
            return

        config_path = os.path.join(folder, CONFIG_FILENAME)

        # Load existing to preserve unknown keys
        existing = {}
        if os.path.isfile(config_path):
            try:
                with open(config_path, 'r') as f:
                    existing = json.load(f)
            except Exception:
                pass

        # Overlay non-default values
        updates = asdict(self)
        updates['project_folder'] = folder
        for k, v in list(updates.items()):
            # Skip None / empty-string / empty-list when existing has a value
            if v is None or v == '' or v == []:
                if k in existing:
                    continue
            existing[k] = v

        try:
            os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            print(f"Warning: could not save project config: {e}")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a plain dict (for passing to hash functions etc.)."""
        return asdict(self)

    # ------------------------------------------------------------------
    # Calibration helpers (added 2026-05-07)
    # ------------------------------------------------------------------

    def resolve_mm_per_pixel(self, session: Optional[dict] = None) -> Optional[float]:
        """Return the effective mm_per_pixel for a session given this
        project's ``calibration_mode``.

        - 'off'   → None (legacy pixel features)
        - 'fixed' → ``self.fixed_mm_per_pixel`` (single rig, project-wide)
        - 'auto'  → ``session['mm_per_pixel']`` from camsync metadata,
                    or None when the video isn't calibrated.

        When ``session`` is None, only 'fixed' returns a value; 'auto'
        returns None (no per-video calibration to consult).
        """
        mode = (self.calibration_mode or 'off').lower()
        if mode == 'off':
            return None
        if mode == 'fixed':
            v = self.fixed_mm_per_pixel
            return float(v) if v is not None else None
        if mode == 'auto':
            if session is None:
                return None
            v = session.get('mm_per_pixel')
            return float(v) if v is not None else None
        return None


# ---------------------------------------------------------------------------
# Key-file discovery (module-level so it is importable headless).
# Port of the reference pattern in gait_limb_tab._scan_key_files: walk the
# project, skip junk dirs and prediction/bout exports, accept CSVs whose
# header row contains exact-case Subject and Treatment columns.
# ---------------------------------------------------------------------------

_KEY_SKIP_DIRS = {'__pycache__', '.git', 'features', 'FeatureCache',
                  'per_frame', 'videos'}
_KEY_SKIP_NAME_TOKENS = ('prediction', 'pred', 'bout', 'timebin', 'summary',
                         'frames')


def find_key_files(project_folder: str) -> List[str]:
    """Return candidate key-file CSVs under ``project_folder``, best first.

    A candidate is a .csv whose header row contains both ``Subject`` and
    ``Treatment`` (exact case, matching analysis_tab's requirement).
    Ranking: ``<project>/key_file.csv`` first, then "key" in the filename,
    then shallower paths. Never raises.
    """
    import csv as _csv
    out = []
    root = os.path.abspath(project_folder)
    if not os.path.isdir(root):
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in _KEY_SKIP_DIRS and not d.startswith('.')]
        for fn in filenames:
            low = fn.lower()
            if not low.endswith('.csv'):
                continue
            if any(tok in low for tok in _KEY_SKIP_NAME_TOKENS):
                continue
            path = os.path.join(dirpath, fn)
            try:
                with open(path, 'r', newline='', encoding='utf-8-sig') as f:
                    header = next(_csv.reader(f), [])
            except Exception:
                continue
            header = [h.strip() for h in header]
            if 'Subject' in header and 'Treatment' in header:
                out.append(path)

    def _rank(path):
        rel = os.path.relpath(path, root)
        return (0 if rel.lower() == 'key_file.csv' else 1,
                0 if 'key' in os.path.basename(path).lower() else 1,
                rel.count(os.sep),
                rel.lower())

    out.sort(key=_rank)
    return out
