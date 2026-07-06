"""Run DLC 3.x PyTorch inference on a video and write a DLC-format .h5.

Uses `deeplabcut.pose_estimation_pytorch.apis.utils.get_inference_runners`
to build the pose model from a bundled `snapshot.pt` + `pytorch_config.yaml`.
Bypasses DLC's project-level config — no DLC "project folder" is required at
runtime, only the snapshot and the pytorch_config dict that ship in the bundle.

Throughput optimizations vs a naive call to `runner.predict()`:

1. **GPU preprocessing** — BGR→RGB, pad-to-multiple-of-stride, ImageNet normalize
   all run on the GPU. CPU only does the cv2 video decode (drops from ~17 ms
   per frame of CPU preprocessing to ~5 ms of pure decode).
2. **Decode prefetch** — a background thread reads + stacks frames into a small
   queue so decode overlaps with the GPU forward pass.
3. **Bypass runner.predict()** — call `model + model.get_predictions` directly
   and do a single batched GPU→CPU transfer instead of one per batch-element.

Net effect on RTX 2080 Ti, 720×768 input: ~25 fps → ~48 fps end-to-end.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from .bundle_loader import Bundle, load_bundle
from .bundle_manager import active_bundle_id
from .h5_writer import dlc_h5_filename, write_dlc_h5
from .video_frame_iter import count_video_frames, prefetched_batches


@dataclass
class InferenceStats:
    video_path: Path
    out_h5_path: Path
    num_frames: int
    elapsed_s: float
    fps: float


ProgressCB = Optional[Callable[[int, int, float], None]]
# (frames_done, frames_total, current_fps) — UI updates every batch.


def _build_runner(bundle: Bundle, device: str, batch_size: int):
    """Construct a DLC PoseInferenceRunner from the bundle artifacts."""
    import yaml
    from deeplabcut.pose_estimation_pytorch.apis.utils import (
        get_inference_runners,
    )

    with bundle.dlc.pytorch_config_path.open() as f:
        model_cfg = yaml.safe_load(f)

    runner, _detector = get_inference_runners(
        model_config=model_cfg,
        snapshot_path=str(bundle.dlc.snapshot_path),
        max_individuals=1,
        num_bodyparts=len(bundle.dlc.keypoints),
        num_unique_bodyparts=0,
        batch_size=batch_size,
        device=device,
        with_identity=False,
    )
    return runner


def _gpu_preprocess(stacked_bgr, mean_t, std_t, stride: int = 16):
    """(B, H, W, 3) uint8 BGR numpy → (B, 3, Hp, Wp) float32 NCHW normalized."""
    import torch
    h, w = stacked_bgr.shape[1:3]
    nh = ((h + stride - 1) // stride) * stride
    nw = ((w + stride - 1) // stride) * stride
    t = torch.from_numpy(stacked_bgr).to(mean_t.device, non_blocking=True)
    t = t[..., [2, 1, 0]]                         # BGR → RGB
    t = t.permute(0, 3, 1, 2).float() / 255.0     # NHWC → NCHW float
    if nh != h or nw != w:
        t = torch.nn.functional.pad(t, (0, nw - w, 0, nh - h))
    return (t - mean_t) / std_t


def run_dlc_on_video(
    video_path: str | Path,
    out_h5_path: Optional[str | Path] = None,
    bundle: Optional[Bundle] = None,
    batch_size: int = 16,
    device: str = "cuda",
    progress_cb: ProgressCB = None,
    overwrite: bool = False,
) -> InferenceStats:
    """Run DLC pose tracking on a video and write a DLC-format .h5.

    Args:
        video_path: input video path
        out_h5_path: where to write the .h5. If None, written next to the video
                     using DLC's standard filename convention.
        bundle: a loaded Bundle, or None to use the active bundle.
        batch_size: number of frames per forward pass.
        device: "cuda", "cuda:N", or "cpu".
        progress_cb: called every batch with (frames_done, frames_total, fps).
        overwrite: replace an existing .h5 instead of raising.
    """
    import torch

    video_path = Path(video_path)
    if bundle is None:
        aid = active_bundle_id()
        if aid is None:
            raise RuntimeError(
                "No active bundle. Install one via the Models tab."
            )
        bundle = load_bundle(aid)

    if out_h5_path is None:
        out_h5_path = video_path.with_name(
            dlc_h5_filename(video_path.stem, bundle.dlc.h5_scorer_string)
        )
    else:
        out_h5_path = Path(out_h5_path)

    if out_h5_path.is_file() and not overwrite:
        raise FileExistsError(
            f"{out_h5_path} already exists. Pass overwrite=True to replace."
        )

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    runner = _build_runner(bundle, device=device, batch_size=batch_size)
    model = runner.model
    num_keypoints = len(bundle.dlc.keypoints)
    total_frames = count_video_frames(str(video_path))
    all_xyl = np.empty((total_frames, num_keypoints, 3), dtype=np.float32)

    mean_t = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_t  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    t0 = time.time()
    frames_done = 0
    with torch.inference_mode():
        for batch in prefetched_batches(
            str(video_path),
            batch_size=batch_size,
            queue_size=3,
        ):
            inputs = _gpu_preprocess(batch.images, mean_t, std_t, stride=16)
            outputs = model(inputs)
            raw = model.get_predictions(outputs)
            # raw["bodypart"]["poses"]: (B, num_individuals=1, C, 3)
            poses = raw["bodypart"]["poses"].cpu().numpy()
            for k, fidx in enumerate(batch.frame_indices):
                all_xyl[fidx] = poses[k, 0]

            frames_done += len(batch.frame_indices)
            if progress_cb is not None:
                elapsed = max(time.time() - t0, 1e-6)
                progress_cb(frames_done, total_frames, frames_done / elapsed)

    elapsed = time.time() - t0
    write_dlc_h5(
        out_h5_path,
        scorer=bundle.dlc.h5_scorer_string,
        bodyparts=bundle.dlc.keypoints,
        keypoints_xyl=all_xyl,
    )

    return InferenceStats(
        video_path=video_path,
        out_h5_path=out_h5_path,
        num_frames=total_frames,
        elapsed_s=elapsed,
        fps=total_frames / max(elapsed, 1e-6),
    )
