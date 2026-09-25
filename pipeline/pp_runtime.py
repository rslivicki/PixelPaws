"""What the PixelPaws app needs from this folder, with nothing lab-specific in it.

- The intake encode (H.265, calibration tags kept) and the checks around it. One
  definition, shared with the cohort pipeline (pp_pipeline imports these back), so
  the encode settings can never drift apart.
- How to find a Python interpreter that can run DeepLabCut.
"""
from __future__ import annotations
import os as _os
import subprocess
from pathlib import Path

# --- Transcode ------------------------------------------------------------------
CODEC = "libx265"
CRF = "23"
PRESET = "slow"
# Size floor for a "good" transcode. 100 kB: a one-minute clip of a still mouse encodes to
# about 450 kB at CRF 23, and the old 1 MB floor called such encodes failed (the pipeline
# then kept the original). The duration check below is what catches a truncated file.
MIN_TRANSCODE_BYTES = 100_000


def _duration(p: Path) -> float:
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=nk=1:nw=1", str(p)], capture_output=True, text=True)
        return round(float(r.stdout.strip()), 1)
    except Exception:
        return 0.0


def video_codec(path) -> str:
    """Codec name of the first video stream ('hevc', 'h264', ...) via ffprobe,
    or '' when ffprobe is missing or the probe fails."""
    import shutil
    if shutil.which("ffprobe") is None:
        return ""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "default=nk=1:nw=1", str(path)],
            capture_output=True, text=True, timeout=30)
        return r.stdout.strip().lower()
    except Exception:
        return ""


def is_hevc(path) -> bool:
    """True when the file is already H.265, i.e. the intake transcode would
    only re-encode it (the Quick Start pipeline and the Add-videos import
    both skip those)."""
    return video_codec(path) in ("hevc", "h265")


def transcode_output_ok(dst: Path) -> bool:
    """A transcode output is 'good' iff it exists, is non-trivially sized
    (> MIN_TRANSCODE_BYTES, 100 kB), and ffprobe reports a positive duration. Used both
    by the inline --delete-source-after-transcode path and the post-run cleanup
    so the verification is identical in both cases.

    Without ffprobe on PATH (ffmpeg installed alone) the duration check falls
    back to cv2: the file must open and report at least one frame - otherwise
    every good encode would be judged failed and discarded."""
    import shutil
    try:
        dst = Path(dst)
        if not dst.is_file() or dst.stat().st_size <= MIN_TRANSCODE_BYTES:
            return False
        if shutil.which("ffprobe") is None:
            import cv2
            cap = cv2.VideoCapture(str(dst))
            try:
                if not cap.isOpened():
                    return False
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if n > 0:
                    return True
                ok, _ = cap.read()          # count unknown: can a frame be decoded?
                return bool(ok)
            finally:
                cap.release()
        return _duration(dst) > 0.0
    except Exception:
        return False


def build_transcode_cmd(src: Path, dst: Path, prog=None) -> list:
    """The canonical intake encode: H.265 (CODEC/CRF/PRESET from pp_config),
    audio dropped, and the source's CUSTOM container tags preserved.
    -map_metadata 0 + -movflags use_metadata_tags: PawCapture videos carry
    spatial-calibration tags (mm_per_pixel, pixelpaws_calibrated,
    pixelpaws_ref_length_mm/ref_pixels) that the classifier mm-per-pixel path
    consumes; without these flags ffmpeg silently drops all arbitrary MP4 tags.
    No-op for sources that have none.

    prog: None, a file path for -progress, or "pipe:1" for stdout progress.
    Used by both the portal pipeline (stage_transcode) and the GUI's
    pre-tracking transcode step (dlc_run_dialog) so the encode settings can
    never drift apart.
    """
    cmd = ["ffmpeg", "-y", "-i", str(src),
           "-map_metadata", "0", "-movflags", "use_metadata_tags",
           "-c:v", CODEC, "-crf", CRF,
           "-preset", PRESET, "-an"]
    if prog is not None:
        cmd += ["-progress", str(prog), "-stats_period", "2"]
        if str(prog) == "pipe:1":
            cmd += ["-nostats", "-loglevel", "error"]
    cmd.append(str(dst))
    return cmd



def resolve_pose_python() -> str:
    """Return the Python interpreter to run DLC pose inference under.

    Handles both deployment shapes:
      - the installer's single combined `pixelpaws` conda env, where the GUI's own
        interpreter can already import deeplabcut  -> use sys.executable (in-process);
      - the dev machine, where the GUI runs in a Python without DLC but a separate
        DEEPLABCUT env exists                       -> use DLC_PYTHON.

    Order: PP_POSE_PYTHON env override -> current interpreter if deeplabcut importable
    -> pp_config.DLC_PYTHON when that lab module is present and the path exists
    -> current interpreter as a last resort.
    """
    import sys as _sys
    import importlib.util as _ilu

    override = _os.environ.get("PP_POSE_PYTHON")
    if override:
        return override
    try:
        if _ilu.find_spec("deeplabcut") is not None:
            return _sys.executable
    except Exception:
        pass
    try:                                   # lab machines: a separate DeepLabCut environment
        from pp_config import DLC_PYTHON
        if _os.path.isfile(DLC_PYTHON):
            return DLC_PYTHON
    except Exception:
        pass
    return _sys.executable
