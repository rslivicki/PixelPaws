"""Windows long-path helper.

Windows limits plain paths to 260 characters unless the system policy
``LongPathsEnabled`` is on. Lab data lives in deep Box/OneDrive trees, and a
DLC pose file name alone is ~90 characters, so full paths past 260 are
common. ``os.path.exists`` then returns False even though ``glob`` (which
walks per component) listed the file - pandas' ``read_hdf`` raises
"File ... does not exist" and feature extraction fails on every session
(seen on a tester's machine 2026-08-28, path length 265).

The ``\\\\?\\`` prefix tells Win32 to skip the limit; pandas/PyTables, OpenCV
and plain ``open`` all accept it. Apply it only at the point a file is
opened or stat'ed - never in names shown to the user or used to build
cache keys.
"""
from __future__ import annotations

import os
import sys

_PREFIX = "\\\\?\\"
_UNC_PREFIX = "\\\\?\\UNC\\"
_LIMIT = 248          # leave headroom under MAX_PATH (260) for suffixes


def long_path(path):
    """Return ``path`` in a form Windows will open regardless of length."""
    if not path or sys.platform != "win32" or not isinstance(path, str):
        return path
    if path.startswith(_PREFIX):
        return path
    p = os.path.abspath(path)
    if len(p) < _LIMIT:
        return path
    if p.startswith("\\\\"):                 # UNC share
        return _UNC_PREFIX + p[2:]
    return _PREFIX + p


def path_exists(path) -> bool:
    return os.path.exists(long_path(path))


def path_isfile(path) -> bool:
    return os.path.isfile(long_path(path))
