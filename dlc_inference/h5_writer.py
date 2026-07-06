"""Write keypoint predictions as a DLC-format .h5 file.

PixelPaws's existing `pose_features.load_dlc_data` reads .h5 with a 3-level
MultiIndex on columns: (scorer, bodypart, coord). The scorer level is stripped
during load, and coord {x, y, likelihood} -> {x, y, prob}.

This writer reproduces that exact layout so the rest of PixelPaws is unaware
that the .h5 came from our ONNX path rather than DLC's native one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def write_dlc_h5(
    out_path: Path,
    scorer: str,
    bodyparts: Sequence[str],
    keypoints_xyl: np.ndarray,
    hdf_key: str = "df_with_missing",
) -> None:
    """
    Args:
        out_path: full path to output .h5 (filename should follow DLC's
                  convention: `<videostem>DLC_<nettype>_<task><date>shuffle<N>_<snapshot>.h5`)
        scorer:   DLC's "scorer string" (e.g. "DLC_resnet50_palmreader-500Mar25shuffle9_190")
        bodyparts: list of keypoint names in column order
        keypoints_xyl: (N_frames, C, 3) array, last dim = (x, y, likelihood)
        hdf_key:  HDF5 key DLC uses
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if keypoints_xyl.ndim != 3 or keypoints_xyl.shape[-1] != 3:
        raise ValueError(
            f"keypoints_xyl must be (N, C, 3), got {keypoints_xyl.shape}"
        )
    n_frames, c, _ = keypoints_xyl.shape
    if c != len(bodyparts):
        raise ValueError(
            f"Keypoint count {c} doesn't match bodyparts {len(bodyparts)}"
        )

    columns = pd.MultiIndex.from_product(
        [[scorer], list(bodyparts), ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )

    flat = keypoints_xyl.reshape(n_frames, c * 3)
    df = pd.DataFrame(flat.astype(np.float64), columns=columns)
    df.to_hdf(
        str(out_path),
        key=hdf_key,
        mode="w",
        format="table",
        complib="zlib",
        complevel=4,
    )


def dlc_h5_filename(video_stem: str, scorer: str) -> str:
    """Compose the DLC-style filename for a given video."""
    return f"{video_stem}{scorer}.h5"
