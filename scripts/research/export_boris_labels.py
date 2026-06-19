"""CLI wrapper: export per-frame labels from a BORIS project to PixelPaws CSVs.

The real logic lives in the importable `boris_import` module (repo root) so the
GUI "Import BORIS Project" tool and this script share one implementation.

Usage:
    python export_boris_labels.py [BORIS.boris] [OUT_DIR] [FEATURE_DIR]

Defaults are baked for the SNLT Rearing/Grooming project. Labels use per-behavior
unobserved tails (a frame is -1 after that behavior's own last event), so behaviors
scored in separate passes don't carry assumed-0 negatives past where they were
reviewed.
"""
import os
import sys

# Make the repo root importable regardless of where this script is launched from.
_REPO = r"E:\Sync_from_lab\PixelPaws"
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from boris_import import export_boris_labels, DEF_BORIS, DEF_OUT, DEF_FEAT

# Behaviors scored in separate passes → mark each behavior -1 after its own last
# event (not just after the global last event). Set False for a single global tail.
PER_BEHAVIOR_TAILS = True


if __name__ == "__main__":
    a = sys.argv
    export_boris_labels(
        a[1] if len(a) > 1 else DEF_BORIS,
        a[2] if len(a) > 2 else DEF_OUT,
        a[3] if len(a) > 3 else DEF_FEAT,
        per_behavior_tails=PER_BEHAVIOR_TAILS,
    )
