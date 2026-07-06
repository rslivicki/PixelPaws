"""Sanity checks between an active bundle's DLC model and its classifiers.

If a user swaps in a new DLC model but keeps classifiers trained on the
previous model, feature distributions can shift silently and degrade
classifier accuracy without raising any errors. This module surfaces
that risk to the GUI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .bundle_loader import Bundle, ClassifierEntry


@dataclass
class CompatibilityIssue:
    severity: str         # "warning" | "error"
    classifier: str
    message: str


def check_bundle(bundle: Bundle) -> List[CompatibilityIssue]:
    issues: List[CompatibilityIssue] = []
    feat_hash = bundle.feature_config.get("config_hash")
    expected_model_id = (
        f"{bundle.dlc.model_name}-shuffle{bundle.dlc.shuffle}-{bundle.dlc.snapshot.replace('snapshot-', '')}"
    )

    for c in bundle.classifiers:
        if c.trained_on_feature_hash and feat_hash:
            if c.trained_on_feature_hash != feat_hash:
                issues.append(CompatibilityIssue(
                    severity="warning",
                    classifier=c.name,
                    message=(
                        f"Classifier was trained on feature hash "
                        f"{c.trained_on_feature_hash!r} but bundle's feature "
                        f"config produces {feat_hash!r}. Predictions may degrade."
                    ),
                ))
        if c.trained_on_dlc_model and c.trained_on_dlc_model != expected_model_id:
            issues.append(CompatibilityIssue(
                severity="warning",
                classifier=c.name,
                message=(
                    f"Classifier was trained against {c.trained_on_dlc_model!r} "
                    f"but bundle ships {expected_model_id!r}. "
                    "Keypoint distributions may have shifted."
                ),
            ))
    return issues


def short_summary(issues: List[CompatibilityIssue]) -> Optional[str]:
    if not issues:
        return None
    n = len(issues)
    return f"{n} compatibility warning{'s' if n != 1 else ''}"
