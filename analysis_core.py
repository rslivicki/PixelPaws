"""
analysis_core.py — Pure compute core for Single-Classifier Analysis
===================================================================

Headless (NO tkinter) extraction of the validated compute code from
``analysis_tab.py`` (see plan "scalable-booping-axolotl", Phase A).  All math
is ported VERBATIM from the old tab; the only intentional behavior changes:

* ``AUC`` column dropped (it was a literal duplicate of ``Total_Time_s``;
  the golden-capture script asserts old.AUC == old.Total_Time_s to document).
* The six always-True metric enable flags are gone (they had no UI).
* ``pick_prediction_column`` is the ONE column ladder (analysis_tab.py
  :1511-1533); the combined-mode's drifted 3-rung copy is not ported.
* ``perform_statistical_test`` pairwise comparisons are Bonferroni-corrected
  (old path was uncorrected while the timecourse used Tukey/Bonferroni);
  result dict gains ``pairwise_correction: 'bonferroni'``.
* ``timecourse_posthoc`` is THE single per-bin pairwise implementation
  (modeled on the primary copy at analysis_tab.py :3238-3408), replacing the
  four drifted copies, with Bonferroni over pairs within each bin.

Source line references below are into E:\\Code\\PixelPaws\\analysis_tab.py
(5,842 lines) at capture time.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhaseWindow:
    """A named phase window (e.g. formalin Acute / Phase II).

    ``bin_index`` keeps the legacy sentinels: -1 for Acute, -2 for Phase II —
    phase rows are stored in results_df with these negative Bin_Index values.
    """
    name: str            # 'Acute', 'Phase_II', ... ('_' shown as ' ' in Bin label)
    start_min: float
    end_min: float
    bin_index: int       # -1 / -2 sentinels (kept from analysis_tab :1654/:1676)


@dataclass
class AnalysisConfig:
    bin_size_min: float = 5.0          # already converted to minutes by the caller
    whole_session: bool = False
    fps: float = 60.0
    analyze_mode: str = 'separate'     # 'separate' | 'combined' | 'both'
    phases: Tuple[PhaseWindow, ...] = ()   # empty = no phase analysis
    filename_prefix: str = ''          # fallback subject-extraction prefix


@dataclass(frozen=True)
class FileInfo:
    path: str
    folder: Optional[str]    # containing result folder / behavior subfolder name
    filename: str
    behavior: str


@dataclass
class AnalysisResult:
    results_df: pd.DataFrame
    # {(subject, treatment, behavior): 1 Hz np.ndarray of 0/1}
    perframe_data: Dict[Tuple[str, str, str], np.ndarray]
    skipped: List[str] = field(default_factory=list)


class SubjectNotFound(Exception):
    """Raised by analyze_file when a filename's subject is not in the key file."""


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def load_key_file(filepath: str) -> pd.DataFrame:
    """Load and validate a key file (analysis_tab :771). Raises ValueError on
    missing required columns instead of showing a messagebox."""
    if filepath.endswith('.xlsx'):
        key_df = pd.read_excel(filepath)
    else:
        key_df = pd.read_csv(filepath)

    required_cols = ['Subject', 'Treatment']
    missing_cols = [col for col in required_cols if col not in key_df.columns]
    if missing_cols:
        raise ValueError(
            f"Key file is missing required columns: {', '.join(missing_cols)}. "
            f"Found columns: {', '.join(map(str, key_df.columns))}. "
            f"Required columns: Subject, Treatment")

    # CRITICAL: Convert Subject column to string for matching
    # Excel stores as int (2801), but filenames extract as string ("2801")
    key_df['Subject'] = key_df['Subject'].astype(str)
    return key_df


_SUBJECT_ID_FN = None
_SUBJECT_ID_FN_TRIED = False


def _legacy_subject_id_fn():
    """Lazily resolve extract_subject_id_from_filename so analysis_core stays
    importable headless.  The function lives in prediction_pipeline and is
    re-exported by PixelPaws_GUI (identical object); we try the light module
    first, then the GUI module, then give up (that ladder rung is skipped)."""
    global _SUBJECT_ID_FN, _SUBJECT_ID_FN_TRIED
    if _SUBJECT_ID_FN_TRIED:
        return _SUBJECT_ID_FN
    _SUBJECT_ID_FN_TRIED = True
    try:
        from prediction_pipeline import extract_subject_id_from_filename
        _SUBJECT_ID_FN = extract_subject_id_from_filename
    except Exception:
        try:
            from PixelPaws_GUI import extract_subject_id_from_filename
            _SUBJECT_ID_FN = extract_subject_id_from_filename
        except Exception:
            _SUBJECT_ID_FN = None
    return _SUBJECT_ID_FN


def resolve_subject(filename: str,
                    key_df: Optional[pd.DataFrame] = None,
                    filename_prefix: str = '') -> str:
    """Return the subject string that matches this filename (analysis_tab :997).

    Strategy (in order):
    1. Scan every subject in the loaded key file and look for it as a
       whole underscore-delimited token inside the filename stem.
    2. Strip a user-configured filename prefix and take the first token.
    3. Smart 4-digit extraction (legacy heuristic; skipped if the helper
       module cannot be imported).
    4. Return the full stem as a last resort.
    """
    base = os.path.basename(filename)
    stem = os.path.splitext(base)[0]
    for suffix in ['_predictions', '_prediction', '_pred', '_Predictions', '_bouts']:
        stem = stem.replace(suffix, '')

    # 1. Key-file token matching
    if key_df is not None:
        tokens = stem.split('_')
        for subj in key_df['Subject']:
            subj_str = str(subj).strip()
            if subj_str in tokens:
                return subj_str
        # Also try multi-token subjects (e.g. "S 1" stored without underscore)
        for subj in key_df['Subject']:
            subj_str = str(subj).strip()
            if f'_{subj_str}_' in f'_{stem}_':
                return subj_str

    # 2. Prefix-strip
    prefix = (filename_prefix or '').strip()
    if prefix and stem.startswith(prefix):
        remainder = stem[len(prefix):]
        token = remainder.split('_')[0] if remainder else ''
        if token:
            return token

    # 3. Legacy smart extraction (lazy import; rung skipped if unavailable)
    fn = _legacy_subject_id_fn()
    if fn is not None:
        try:
            sid = fn(base)
            if sid:
                return sid
        except Exception:
            pass

    # 4. Full stem fallback
    return stem


def extract_behavior_name(filename: str, folder_name: Optional[str]) -> str:
    """Extract behavior name from filename or folder (analysis_tab :1195,
    verbatim: folder-wins ladder, classifier-marker split, heuristic fallback)."""
    # Remove extensions
    name = filename.replace('.csv', '')

    # Remove common suffixes first
    for suffix in ['_predictions', '_prediction', '_timebins', '_timebin', '_bouts', '_bout']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break

    # The batch runner writes results/<Behavior_type>/<video>_<clf>_predictions.csv,
    # so a behavior subfolder is the authoritative identity -- it groups every
    # video under one behavior regardless of how the classifier file was named.
    # (Old per-subject folders contain "Results" in their name and are excluded.)
    if (folder_name and 'result' not in folder_name.lower()
            and folder_name.lower() not in ('per_frame', 'features', 'videos')):
        return folder_name

    # Split into parts
    parts = name.split('_')

    # Behavior is whatever follows the source/classifier marker (classifier,
    # PixelPaws, BAREfoot, …).
    for _marker in ('classifier', 'PixelPaws', 'BAREfoot'):
        if _marker in parts:
            _idx = len(parts) - 1 - parts[::-1].index(_marker)  # last occurrence
            behavior_parts = parts[_idx + 1:]
            if behavior_parts:
                return '_'.join(behavior_parts)

    # Fallback: strip date-like parts, short IDs, experiment names
    filtered_parts = []
    for part in parts:
        # Skip if it's a date (6 digits) or subject ID (4 digits)
        if part.isdigit() and len(part) in [4, 6]:
            continue
        # Skip common experiment/metadata words + source markers
        if part.lower() in ['pixelpaws', 'barefoot', 'results', 'formalin', 'formoxy']:
            continue
        # Skip subject-like tokens (e.g. S1, S10, sub03)
        if re.fullmatch(r'(?i)(s|sub|subject)?\d{1,3}', part):
            continue
        filtered_parts.append(part)

    if filtered_parts:
        return '_'.join(filtered_parts)
    else:
        return 'Unknown'


def scan_predictions(folder: str) -> Tuple[List[FileInfo], Dict[str, List[FileInfo]]]:
    """Scan folder for prediction files (analysis_tab :1088, minus UI).

    Returns (prediction_files, behaviors) where behaviors maps
    behavior_name -> [FileInfo, ...].
    """
    prediction_files: List[FileInfo] = []
    behaviors: Dict[str, List[FileInfo]] = {}

    # Check if folder contains result subfolders (PixelPaws batch output format)
    items = os.listdir(folder)
    result_folders = [item for item in items if os.path.isdir(os.path.join(folder, item))
                      and ('Results' in item or 'PixelPaws_Results' in item)]

    # Check if this IS a Results folder (new single-folder format)
    is_results_folder = os.path.basename(folder).lower() in ['results', 'pixelpaws_results']

    if result_folders:
        # OLD FORMAT: Each subject has a results folder
        for result_folder in result_folders:
            folder_path = os.path.join(folder, result_folder)
            for file in os.listdir(folder_path):
                if file.endswith('.csv') and 'prediction' in file.lower():
                    pred_file = os.path.join(folder_path, file)
                    filename = os.path.basename(file)
                    behavior_name = extract_behavior_name(filename, result_folder)
                    file_info = FileInfo(path=pred_file, folder=result_folder,
                                         filename=filename, behavior=behavior_name)
                    prediction_files.append(file_info)
                    behaviors.setdefault(behavior_name, []).append(file_info)

    elif is_results_folder or not result_folders:
        # NEW FORMAT: All files in single Results folder OR direct CSV files
        # Recurse into subdirectories (e.g. results/<behavior>/)
        for dirpath, _dirnames, filenames in os.walk(folder):
            for file in filenames:
                if not file.endswith('.csv'):
                    continue
                if any(skip in file for skip in ['Summary', 'Treatment', 'Analysis']):
                    continue
                fl = file.lower()
                if 'prediction' in fl:
                    full_path = os.path.join(dirpath, file)
                    _subdir = (os.path.basename(dirpath)
                               if dirpath != folder else None)
                    behavior_name = extract_behavior_name(file, _subdir)
                    file_info = FileInfo(
                        path=full_path,
                        folder=os.path.basename(dirpath) if dirpath != folder else None,
                        filename=file, behavior=behavior_name)
                    prediction_files.append(file_info)
                    behaviors.setdefault(behavior_name, []).append(file_info)

    return prediction_files, behaviors


def scan_project(project_folder: str) -> dict:
    """Recursively scan a project folder for prediction folders and key files
    (analysis_tab :834, minus UI).

    Returns dict with:
      'pred_folders': [(folder_path, n_validated_files), ...] best-first,
      'key_files':    [path, ...] (files with exactly Subject+Treatment columns),
      'auto_pred_folder': the folder scan_predictions should be run on (or None),
      'warnings':     [str, ...]
    """
    warnings: List[str] = []
    if not project_folder or not os.path.isdir(project_folder):
        return {'pred_folders': [], 'key_files': [], 'auto_pred_folder': None,
                'warnings': ['project folder not found']}

    _SKIP_DIRS = {'__pycache__', '.git', '.claude', 'node_modules', '.idea'}
    _PRED_KEYWORDS = ('prediction', 'predictions', 'pred', 'bout', 'bouts')

    pred_folder_files: Dict[str, List[str]] = {}
    key_candidates_raw: List[str] = []

    for root, dirs, files in os.walk(project_folder):
        dirs[:] = sorted(d for d in dirs
                         if d not in _SKIP_DIRS and not d.startswith('.'))
        for fname in files:
            fl = fname.lower()
            if not fl.endswith(('.csv', '.xlsx')):
                continue
            full_path = os.path.join(root, fname)
            if any(kw in fl for kw in _PRED_KEYWORDS):
                pred_folder_files.setdefault(root, []).append(full_path)
            else:
                key_candidates_raw.append(full_path)

    # ── Validate prediction folders ───────────────────────────────────
    _PRED_COLS_VALID = {'probability', 'frame', 'start_frame', 'duration_sec'}
    pred_folders: Dict[str, int] = {}
    for folder_path, fpaths in pred_folder_files.items():
        validated = 0
        folder_ok = False
        for fpath in fpaths:
            try:
                cols = set(pd.read_csv(fpath, nrows=0).columns.tolist())
                if cols & _PRED_COLS_VALID:
                    validated += 1
                    folder_ok = True
            except Exception as _csv_err:
                warnings.append(f"could not read CSV {os.path.basename(fpath)}: {_csv_err}")
        if folder_ok:
            pred_folders[folder_path] = validated or len(fpaths)

    # ── Validate key files ────────────────────────────────────────────
    # Reject results-table exports (they carry Subject+Treatment too) and
    # rank candidates, sharing the hardened project_config rules so a saved
    # analysis export cannot become the auto-loaded key.
    try:
        from project_config import _KEY_RESULTS_COLS
    except Exception:                                      # pragma: no cover
        _KEY_RESULTS_COLS = set()
    key_candidates: List[str] = []
    for full_path in key_candidates_raw:
        try:
            if full_path.endswith('.xlsx'):
                cols = pd.read_excel(full_path, nrows=0).columns.tolist()
            else:
                cols = pd.read_csv(full_path, nrows=0).columns.tolist()
            if 'Subject' not in cols or 'Treatment' not in cols:
                continue
            base = os.path.basename(full_path).lower()
            if (_KEY_RESULTS_COLS & set(cols)) and 'key' not in base:
                warnings.append(f"ignored {os.path.basename(full_path)} as a "
                                f"key candidate (looks like a results export)")
                continue
            key_candidates.append(full_path)
        except Exception as _key_err:
            warnings.append(f"could not read key file {os.path.basename(full_path)}: {_key_err}")
    # Rank: key_file.csv, then 'key' in name, then shallower paths.
    key_candidates.sort(key=lambda p: (
        0 if os.path.basename(p).lower() == 'key_file.csv' else 1,
        0 if 'key' in os.path.basename(p).lower() else 1,
        p.count(os.sep), p.lower()))

    sorted_pred = sorted(pred_folders.items(), key=lambda x: x[1], reverse=True)

    # Auto-select: prefer a top-level results/ folder, else the single candidate.
    _auto_pred = None
    for _cand in ('results', 'Results', 'PixelPaws_Results'):
        _rp = os.path.join(project_folder, _cand)
        if os.path.isdir(_rp):
            _auto_pred = _rp
            break
    if _auto_pred is None and len(sorted_pred) == 1:
        _auto_pred = sorted_pred[0][0]

    return {'pred_folders': sorted_pred, 'key_files': key_candidates,
            'auto_pred_folder': _auto_pred, 'warnings': warnings}


def subjects_overview(project_folder: str,
                      key_df: Optional[pd.DataFrame] = None) -> List[Tuple[str, str]]:
    """Return [(subject, group), ...] for the Subjects overview
    (analysis_tab :192-281, videos/ basename scan).

    Subjects are video basenames under <project>/videos (DLC/overlay outputs
    skipped).  Group comes from ``key_df``; if key_df is None the SINGLE key
    discovery path (project_config.find_key_files + load_key_file) is used —
    the old tab's duplicated ad-hoc key walk is not ported.
    """
    if not project_folder or not os.path.isdir(project_folder):
        return []

    import glob as _g
    subjects = []
    vdir = os.path.join(project_folder, 'videos')
    if os.path.isdir(vdir):
        for _ext in ('*.mp4', '*.avi', '*.mov', '*.wmv', '*.mkv'):
            for vf in _g.glob(os.path.join(vdir, _ext)):
                _name = os.path.splitext(os.path.basename(vf))[0]
                _low = _name.lower()
                # skip DLC/overlay outputs (e.g. *_labeled, *DLC_Resnet*) — not real subjects
                if '_labeled' in _low or 'dlc_' in _low or 'dlc_resnet' in _low:
                    continue
                subjects.append(_name)
    subjects = sorted(set(subjects))

    if key_df is None:
        try:
            from project_config import find_key_files
            cands = find_key_files(project_folder)
            if cands:
                key_df = load_key_file(cands[0])
        except Exception:
            key_df = None

    group_map: Dict[str, str] = {}
    if key_df is not None:
        try:
            _cl = {col.lower(): col for col in key_df.columns}
            sc, tc = _cl.get('subject'), _cl.get('treatment')
            if sc and tc:
                group_map = {str(s): str(t) for s, t in zip(key_df[sc], key_df[tc])}
        except Exception:
            pass

    rows: List[Tuple[str, str]] = []
    for subj in subjects:
        grp = group_map.get(subj, '')
        if not grp and group_map:
            for s, t in group_map.items():
                if s and s in subj:
                    grp = t
                    break
        rows.append((subj, grp or '—'))

    # Default: grouped by treatment (sort by group, then subject).
    rows.sort(key=lambda r: (r[1], r[0]))
    return rows


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def detect_bouts(predictions) -> List[Tuple[int, int]]:
    """Detect behavioral bouts (continuous periods of behavior).
    Verbatim from analysis_tab :1854."""
    bouts = []
    in_bout = False
    bout_start = None

    for i, pred in enumerate(predictions):
        if pred == 1:
            if not in_bout:
                in_bout = True
                bout_start = i
        else:
            if in_bout:
                bouts.append((bout_start, i - 1))
                in_bout = False

    # Close final bout if needed
    if in_bout:
        bouts.append((bout_start, len(predictions) - 1))

    return bouts


def pick_prediction_column(pred_df: pd.DataFrame) -> np.ndarray:
    """THE single prediction-column ladder (analysis_tab :1511-1533).

    Priority: prediction_filtered > prediction_raw > Left_licking >
    Right_licking > last column; then binarize (>0.5) if not already 0/1.
    Replaces the combined-mode's drifted 3-rung copy.
    """
    if 'prediction_filtered' in pred_df.columns:
        predictions = pred_df['prediction_filtered'].values
    elif 'prediction_raw' in pred_df.columns:
        predictions = pred_df['prediction_raw'].values
    elif 'Left_licking' in pred_df.columns:
        predictions = pred_df['Left_licking'].values
    elif 'Right_licking' in pred_df.columns:
        predictions = pred_df['Right_licking'].values
    else:
        # Use last column (usually the prediction column)
        predictions = pred_df.iloc[:, -1].values

    # Verify predictions are binary (0 or 1)
    unique_vals = np.unique(predictions)
    if not all(val in [0, 1] for val in unique_vals):
        predictions = (predictions > 0.5).astype(int)

    return predictions


def calculate_metrics(predictions, fps: float, bin_duration_sec: float) -> dict:
    """Calculate behavior metrics for a time bin (analysis_tab :1809, verbatim
    minus the six always-True metric flags and minus AUC — AUC was a literal
    duplicate of Total_Time_s)."""
    metrics = {}

    # Total time in behavior
    frames_in_behavior = np.sum(predictions == 1)
    metrics['Total_Time_s'] = frames_in_behavior / fps

    # Number of bouts
    bouts = detect_bouts(predictions)
    metrics['N_Bouts'] = len(bouts)

    # Mean bout duration
    if len(bouts) > 0:
        bout_durations = [(end - start + 1) / fps for start, end in bouts]
        metrics['Mean_Bout_Duration_s'] = np.mean(bout_durations)
    else:
        metrics['Mean_Bout_Duration_s'] = 0

    # Bout frequency
    metrics['Bout_Frequency_per_min'] = (len(bouts) / bin_duration_sec) * 60

    # Latency to first positive frame within this bin
    if np.any(predictions == 1):
        first_pos = np.argmax(predictions == 1)
        metrics['Latency_In_Bin_s'] = first_pos / fps
    else:
        metrics['Latency_In_Bin_s'] = None

    # Percentage of time
    metrics['Percent_Time'] = (np.sum(predictions == 1) / len(predictions)) * 100

    return metrics


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _bin_rows(predictions, cfg: AnalysisConfig,
              subject: str, treatment: str, behavior_name: str) -> List[dict]:
    """Per-bin rows for one prediction array (analysis_tab :1545-1605 — the
    identical binning code appeared twice, in analyze_single_file and
    analyze_combined_behaviors; this is the single copy)."""
    fps = cfg.fps
    bin_size_min = cfg.bin_size_min
    total_frames = len(predictions)
    total_seconds = total_frames / fps
    total_minutes = total_seconds / 60

    if cfg.whole_session:
        bin_size_frames = total_frames
        n_bins = 1
    else:
        bin_size_sec = bin_size_min * 60
        bin_size_frames = int(bin_size_sec * fps)
        n_bins = int(np.ceil(total_frames / bin_size_frames))

    results = []
    for bin_idx in range(n_bins):
        start_frame = bin_idx * bin_size_frames
        end_frame = min(start_frame + bin_size_frames, total_frames)

        bin_preds = predictions[start_frame:end_frame]

        # Calculate actual bin duration (might be shorter for last bin)
        actual_bin_duration_sec = len(bin_preds) / fps

        metrics = calculate_metrics(bin_preds, fps, actual_bin_duration_sec)

        # Compute absolute latency from video start
        if metrics.get('Latency_In_Bin_s') is not None:
            metrics['Latency_s'] = (start_frame / fps) + metrics['Latency_In_Bin_s']
        else:
            metrics['Latency_s'] = None

        # Add metadata
        metrics['Subject'] = subject
        metrics['Treatment'] = treatment
        metrics['Behavior'] = behavior_name
        if cfg.whole_session:
            metrics['Bin'] = 'Whole'
            metrics['Bin_Index'] = 0
            metrics['Bin_Start_Min'] = 0
            metrics['Bin_End_Min'] = round(total_minutes, 1)
        else:
            _bin_start = round(bin_idx * bin_size_min, 4)
            _bin_end = round((bin_idx + 1) * bin_size_min, 4)
            metrics['Bin'] = f"{_bin_start}-{_bin_end}"
            metrics['Bin_Index'] = bin_idx
            metrics['Bin_Start_Min'] = _bin_start
            metrics['Bin_End_Min'] = _bin_end

        results.append(metrics)

    return results


def phase_rows(predictions, fps: float, subject: str, treatment: str,
               behavior_name: str, phases: Sequence[PhaseWindow]) -> List[dict]:
    """Phase-window rows (analysis_tab :1623-1685, verbatim math; windows come
    from cfg.phases instead of Tk vars).  These frame-accurate stored rows are
    the SINGLE phase source (Bin_Index < 0 sentinels)."""
    results = []
    total_frames = len(predictions)

    for pw in phases:
        start_frame = int(pw.start_min * 60 * fps)
        end_frame = int(pw.end_min * 60 * fps)

        if end_frame <= total_frames:
            preds = predictions[start_frame:end_frame]
            duration_sec = len(preds) / fps

            m = calculate_metrics(preds, fps, duration_sec)
            display = pw.name.replace('_', ' ')
            m.update({
                'Subject': subject,
                'Treatment': treatment,
                'Behavior': behavior_name,
                'Phase': pw.name,
                'Phase_Start_Min': pw.start_min,
                'Phase_End_Min': pw.end_min,
                'Bin': f'{display} ({pw.start_min}-{pw.end_min} min)',
                'Bin_Index': pw.bin_index,
            })
            results.append(m)

    return results


def combine_predictions(prediction_arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Combine per-behavior prediction arrays (analysis_tab :1738-1742):
    truncate to the shortest, sum, binarize (any behavior present = 1)."""
    min_frames = min(len(p) for p in prediction_arrays)
    truncated_preds = [p[:min_frames] for p in prediction_arrays]
    combined_predictions = np.sum(truncated_preds, axis=0)
    combined_predictions = (combined_predictions > 0).astype(int)
    return combined_predictions


def analyze_file(file_info: FileInfo, key_df: pd.DataFrame,
                 cfg: AnalysisConfig) -> Tuple[List[dict], Dict[tuple, np.ndarray]]:
    """Analyze a single prediction file (analysis_tab :1482 logic — binning,
    absolute Latency_s, Whole row, 1 Hz perframe via any()-per-second, phase
    rows when cfg.phases).

    Returns (rows, perframe) where perframe holds the 1 Hz downsampled
    predictions keyed by (subject, treatment, behavior).
    Raises SubjectNotFound if the subject isn't in the key file.
    """
    pred_file = file_info.path
    behavior_name = file_info.behavior or 'Unknown'

    filename = os.path.basename(pred_file)
    subject = resolve_subject(filename, key_df, cfg.filename_prefix)

    subject_row = key_df[key_df['Subject'] == subject]
    if subject_row.empty:
        raise SubjectNotFound(
            f"Subject '{subject}' not found in key file (file: {filename})")
    treatment = subject_row.iloc[0]['Treatment']

    pred_df = pd.read_csv(pred_file)
    predictions = pick_prediction_column(pred_df)

    # Store per-frame predictions downsampled to 1-second resolution
    # (analysis_tab :1536-1542)
    fps_int = max(1, int(cfg.fps))
    sec_predictions = np.array([
        int(np.any(predictions[i:i + fps_int]))
        for i in range(0, len(predictions), fps_int)
    ])
    perframe = {(subject, treatment, behavior_name): sec_predictions}

    rows = _bin_rows(predictions, cfg, subject, treatment, behavior_name)

    if cfg.phases:
        rows.extend(phase_rows(predictions, cfg.fps, subject, treatment,
                               behavior_name, cfg.phases))

    return rows, perframe


def _analyze_combined(files: Sequence[FileInfo], selected_behaviors: Sequence[str],
                      key_df: pd.DataFrame, cfg: AnalysisConfig,
                      skipped: List[str]) -> List[dict]:
    """Combined-behaviors analysis (analysis_tab :1691, minus prints; uses the
    ONE pick_prediction_column ladder instead of the drifted 3-rung copy).
    NOTE: matches the old tab — combined mode stores NO perframe data."""
    # Group files by subject
    subject_files: Dict[str, List[FileInfo]] = {}
    for file_info in files:
        filename = os.path.basename(file_info.path)
        subject = resolve_subject(filename, key_df, cfg.filename_prefix)
        subject_files.setdefault(subject, []).append(file_info)

    combined_label = 'Combined_' + '+'.join(sorted(selected_behaviors))
    results: List[dict] = []

    for subject, sfiles in subject_files.items():
        treatment_match = key_df[key_df['Subject'] == subject]
        if treatment_match.empty:
            skipped.append(f"combined: subject '{subject}' not found in key file")
            continue
        treatment = treatment_match.iloc[0]['Treatment']

        all_predictions = []
        for file_info in sfiles:
            pred_df = pd.read_csv(file_info.path)
            all_predictions.append(pick_prediction_column(pred_df))

        combined_predictions = combine_predictions(all_predictions)

        results.extend(_bin_rows(combined_predictions, cfg, subject, treatment,
                                 combined_label))

        if cfg.phases:
            results.extend(phase_rows(combined_predictions, cfg.fps, subject,
                                      treatment, combined_label, cfg.phases))

    return results


def run_analysis(files: Sequence[FileInfo], key_df: pd.DataFrame,
                 cfg: AnalysisConfig,
                 progress_cb: Optional[Callable[[str], None]] = None,
                 cancel=None) -> AnalysisResult:
    """Orchestrate the batch analysis (analysis_tab :1339, minus UI).

    ``files`` should already be filtered to the selected behaviors.
    ``cancel`` is an optional object with is_set() (e.g. threading.Event),
    checked between files.  Files that cannot be analyzed are recorded in
    AnalysisResult.skipped (the old code only printed warnings).
    """
    def _prog(msg: str):
        if progress_cb is not None:
            try:
                progress_cb(msg)
            except Exception:
                pass

    def _cancelled() -> bool:
        return cancel is not None and cancel.is_set()

    skipped: List[str] = []
    all_results: List[dict] = []
    perframe_data: Dict[Tuple[str, str, str], np.ndarray] = {}

    # Behavior order mirrors the old checkbox wall (sorted names).
    selected_behaviors = sorted({f.behavior for f in files})

    if cfg.analyze_mode in ('separate', 'both'):
        for behavior in selected_behaviors:
            if _cancelled():
                break
            _prog(f"Analyzing {behavior}…")
            behavior_files = [f for f in files if f.behavior == behavior]
            for file_info in behavior_files:
                if _cancelled():
                    break
                try:
                    rows, perframe = analyze_file(file_info, key_df, cfg)
                    all_results.extend(rows)
                    perframe_data.update(perframe)
                except SubjectNotFound as e:
                    skipped.append(f"{file_info.path}: {e}")
                except Exception as e:
                    skipped.append(f"{file_info.path}: error: {e}")

    if not _cancelled() and cfg.analyze_mode in ('combined', 'both'):
        _prog("Analyzing combined behaviors…")
        try:
            all_results.extend(_analyze_combined(files, selected_behaviors,
                                                 key_df, cfg, skipped))
        except Exception as e:
            skipped.append(f"combined: error: {e}")

    results_df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    return AnalysisResult(results_df=results_df, perframe_data=perframe_data,
                          skipped=skipped)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def perform_statistical_test(data_by_treatment: Dict[str, Sequence[float]],
                             alpha: float = 0.05,
                             paradigm: str = 'auto') -> Optional[dict]:
    """Group-comparison test (analysis_tab :5386, verbatim math with args
    instead of Tk vars).

    paradigm: 'auto' (Shapiro-Wilk decides), 'parametric', or 'nonparametric'.

    Returns dict with 'test_type', 'p_value', 'significant', 'effect_size',
    'effect_size_type', and (3+ groups, significant omnibus) 'pairwise' with
    Bonferroni-corrected p-values plus 'pairwise_correction': 'bonferroni'.
    (Change from the old tab: pairwise p-values were uncorrected there.)
    """
    from scipy import stats

    treatments = list(data_by_treatment.keys())
    groups = [data_by_treatment[t] for t in treatments]

    # Remove empty groups
    valid = [(t, g) for t, g in zip(treatments, groups) if len(g) > 0]
    if len(valid) < 2:
        return None
    treatments_valid = [v[0] for v in valid]
    groups = [v[1] for v in valid]

    # Decide parametric vs non-parametric
    use_parametric = True
    if paradigm == 'nonparametric':
        use_parametric = False
    elif paradigm == 'auto':
        # Shapiro-Wilk on each group; if any fails, go non-parametric
        use_parametric = True
        for g in groups:
            if len(g) >= 3:
                _, sw_p = stats.shapiro(g)
                if sw_p < 0.05:
                    use_parametric = False
                    break

    results = {'alpha': alpha}

    if len(groups) == 2:
        if use_parametric:
            t_stat, p_val = stats.ttest_ind(groups[0], groups[1], equal_var=False)
            results['test_type'] = "Welch's t-test"
        else:
            t_stat, p_val = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
            results['test_type'] = 'Mann-Whitney U'
        results['p_value'] = p_val
        results['significant'] = p_val < alpha
        results['comparison'] = f"{treatments_valid[0]} vs {treatments_valid[1]}"

        # Cohen's d
        n1, n2 = len(groups[0]), len(groups[1])
        var1, var2 = np.var(groups[0], ddof=1), np.var(groups[1], ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std > 0:
            cohens_d = abs(np.mean(groups[0]) - np.mean(groups[1])) / pooled_std
        else:
            cohens_d = 0.0
        results['effect_size'] = cohens_d
        results['effect_size_type'] = "Cohen's d"

    else:
        if use_parametric:
            f_stat, p_val = stats.f_oneway(*groups)
            results['test_type'] = 'ANOVA'
        else:
            f_stat, p_val = stats.kruskal(*groups)
            results['test_type'] = 'Kruskal-Wallis'
        results['p_value'] = p_val
        results['significant'] = p_val < alpha

        # Eta-squared
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        ss_total = np.sum((all_data - grand_mean) ** 2)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
        results['effect_size'] = eta_sq
        results['effect_size_type'] = 'eta-squared'

        # If significant, do pairwise comparisons (Bonferroni-corrected)
        if p_val < alpha:
            n_comparisons = len(treatments_valid) * (len(treatments_valid) - 1) // 2
            pairwise = {}
            for i in range(len(treatments_valid)):
                for j in range(i + 1, len(treatments_valid)):
                    if use_parametric:
                        _, p_pairwise = stats.ttest_ind(groups[i], groups[j], equal_var=False)
                    else:
                        _, p_pairwise = stats.mannwhitneyu(groups[i], groups[j],
                                                           alternative='two-sided')
                    p_corrected = min(p_pairwise * n_comparisons, 1.0)
                    pairwise[f"{treatments_valid[i]}_vs_{treatments_valid[j]}"] = {
                        'p_raw': p_pairwise,
                        'p_corrected': p_corrected,
                        'p_value': p_corrected,
                        'significant': p_corrected < alpha
                    }
            results['pairwise'] = pairwise
            results['pairwise_correction'] = 'bonferroni'

    return results


def timecourse_anova(df: pd.DataFrame, metric: str,
                     alpha: float = 0.05) -> Optional[dict]:
    """Two-way ANOVA (Time × Treatment) on a long results frame
    (analysis_tab :4946 with the metric parameterized — fixes the hardcoded
    Total_Time_s).  Returns None if statsmodels is unavailable or the fit
    fails."""
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols

        anova_df = df[['Subject', 'Treatment', 'Bin_Start_Min', metric]].copy()
        anova_df['Time'] = anova_df['Bin_Start_Min'].astype('category')
        anova_df['Treatment'] = anova_df['Treatment'].astype('category')

        model = ols(f'{metric} ~ C(Treatment) + C(Time) + C(Treatment):C(Time)',
                    data=anova_df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        return {
            'anova_table': anova_table,
            'treatment_p': anova_table.loc['C(Treatment)', 'PR(>F)'],
            'time_p': anova_table.loc['C(Time)', 'PR(>F)'],
            'interaction_p': anova_table.loc['C(Treatment):C(Time)', 'PR(>F)'],
            'alpha': alpha,
        }
    except Exception:
        return None


_POSTHOC_COLUMNS = ['Bin_Start_Min', 'group_a', 'group_b',
                    'p_raw', 'p_corrected', 'significant']


def timecourse_posthoc(df: pd.DataFrame, treatments: Sequence[str], metric: str,
                       alpha: float = 0.05,
                       paradigm: str = 'auto') -> pd.DataFrame:
    """Per-bin pairwise post-hoc comparisons — THE single implementation
    (modeled on the primary copy at analysis_tab :3238-3408) replacing the
    four drifted copies.

    Returns a long-format DataFrame with columns
    [Bin_Start_Min, group_a, group_b, p_raw, p_corrected, significant]:
    every treatment pair tested at every time bin (Welch's t-test or
    Mann-Whitney U per the paradigm; Shapiro-Wilk decides for 'auto'),
    Bonferroni-corrected over the pairs within each bin.
    """
    from scipy import stats

    rows = []
    time_bins = sorted(df['Bin_Start_Min'].unique())

    for bin_start in time_bins:
        bin_df = df[df['Bin_Start_Min'] == bin_start]

        # Get data for each treatment at this time bin (non-empty groups only)
        names, groups = [], []
        for treatment in treatments:
            treat_data = bin_df[bin_df['Treatment'] == treatment][metric].values
            if len(treat_data) > 0:
                names.append(treatment)
                groups.append(treat_data)

        if len(groups) < 2:
            continue

        # Paradigm decision (verbatim rule from analysis_tab :3295-3300)
        use_param = paradigm != 'nonparametric'
        if paradigm == 'auto':
            use_param = all(
                stats.shapiro(g)[1] >= 0.05 for g in groups if len(g) >= 3
            )

        n_comparisons = len(groups) * (len(groups) - 1) // 2
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if use_param:
                    _, p_raw = stats.ttest_ind(groups[i], groups[j], equal_var=False)
                else:
                    _, p_raw = stats.mannwhitneyu(groups[i], groups[j],
                                                  alternative='two-sided')
                p_corrected = min(p_raw * n_comparisons, 1.0)
                rows.append({
                    'Bin_Start_Min': bin_start,
                    'group_a': names[i],
                    'group_b': names[j],
                    'p_raw': p_raw,
                    'p_corrected': p_corrected,
                    'significant': bool(p_corrected < alpha),
                })

    return pd.DataFrame(rows, columns=_POSTHOC_COLUMNS)
