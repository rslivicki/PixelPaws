"""Pick the right DLC .h5 when a video has more than one.

After a pose-network update a video can carry two pose files: the old
network's and the new one's (Quick Start re-tracks on request and keeps the
old file). Every "<stem>DLC*.h5" glob used to take the first match, and the
old file sorts first, so features and classifiers silently kept reading the
old tracking. rank_h5() puts the active bundle's scorer first, newest file
first within a group. Callers keep their own filtered/unfiltered preference
on top of this order.
"""
from __future__ import annotations

import os
from typing import Iterable, List, Optional


def active_scorer() -> Optional[str]:
    """h5 scorer string of the active DLC bundle, or None (no bundle, or the
    inference package is unavailable)."""
    try:
        from dlc_inference import load_active_bundle
        b = load_active_bundle()
        return b.dlc.h5_scorer_string if b else None
    except Exception:
        return None


def rank_h5(paths: Iterable[str], scorer: Optional[str] = None) -> List[str]:
    """Return `paths` ordered: active-scorer files first, then the rest; each
    group newest-modified first. A single path comes back unchanged."""
    paths = [str(p) for p in paths]
    if len(paths) < 2:
        return paths
    if scorer is None:
        scorer = active_scorer()

    def _mtime(p):
        try:
            return os.path.getmtime(p)
        except OSError:
            return 0.0

    hits = [p for p in paths if scorer and scorer in os.path.basename(p)]
    rest = [p for p in paths if p not in hits]
    return (sorted(hits, key=_mtime, reverse=True)
            + sorted(rest, key=_mtime, reverse=True))
