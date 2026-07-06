"""DLC PyTorch inference for PixelPaws distribution builds.

Wraps `deeplabcut.pose_estimation_pytorch` so PixelPaws can run pose tracking
on raw videos using a bundled snapshot. The user-facing API is intentionally
small:

    from dlc_inference import (
        run_dlc_on_video,
        load_active_bundle,
        list_bundles,
        probe_providers,
    )
"""

from .bundle_loader import load_bundle, Bundle, ClassifierEntry, DlcInfo
from .bundle_manager import (
    bundles_root,
    list_bundles,
    active_bundle_id,
    set_active_bundle,
    import_bundle_zip,
    install_default_bundle,
)
from .dlc_runner import run_dlc_on_video, InferenceStats
from .provider_probe import probe_providers, ProviderInfo

__all__ = [
    "load_bundle",
    "Bundle",
    "ClassifierEntry",
    "DlcInfo",
    "bundles_root",
    "list_bundles",
    "active_bundle_id",
    "set_active_bundle",
    "import_bundle_zip",
    "install_default_bundle",
    "run_dlc_on_video",
    "InferenceStats",
    "probe_providers",
    "ProviderInfo",
]


def load_active_bundle():
    aid = active_bundle_id()
    if aid is None:
        return None
    return load_bundle(aid)
