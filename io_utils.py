"""io_utils.py — Shared atomic-write helpers + git-SHA stamping for PixelPaws.

`atomic_pickle_save` and `atomic_dataframe_to_csv` write a tempfile in the
same directory and `os.replace` it onto the target path so a crash mid-write
leaves the original file intact instead of producing a half-written stub.

`get_git_sha` returns a short SHA + dirty flag for embedding in classifier
pkls, sidecar JSON, and exported CSVs.
"""

import os
import pickle
import subprocess
import tempfile


def atomic_pickle_save(data, target_path):
    """Pickle `data` to `target_path` atomically (temp file -> os.replace)."""
    dir_path = os.path.dirname(target_path) or '.'
    os.makedirs(dir_path, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix='.tmp')
    try:
        with os.fdopen(tmp_fd, 'wb') as f:
            pickle.dump(data, f)
        os.replace(tmp_path, target_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_dataframe_to_csv(df, target_path, **to_csv_kwargs):
    """Write a DataFrame to CSV atomically (temp file -> os.replace)."""
    dir_path = os.path.dirname(target_path) or '.'
    os.makedirs(dir_path, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix='.tmp')
    os.close(tmp_fd)
    try:
        df.to_csv(tmp_path, **to_csv_kwargs)
        os.replace(tmp_path, target_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def get_git_sha(cwd: str = None) -> str:
    """Return short git SHA + '+dirty' suffix when working tree is dirty.

    Returns 'unknown' if not a git checkout or git is unavailable.
    """
    cwd = cwd or _REPO_ROOT
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()
        try:
            dirty = subprocess.check_output(
                ['git', 'status', '--porcelain'],
                cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()
            if dirty:
                sha += '+dirty'
        except Exception:
            pass
        return sha or 'unknown'
    except Exception:
        return 'unknown'
