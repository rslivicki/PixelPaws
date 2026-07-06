"""Frame iterator: video → batches of raw BGR uint8 frames.

CPU preprocessing (BGR→RGB, padding, ImageNet normalize) used to live here
but ate 17 ms/frame. Now we yield raw BGR uint8 stacks and let the caller
do preprocessing on the GPU, which drops to ~1 ms/frame.

A simple background-thread prefetcher overlaps cv2 decode with downstream
GPU work.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Iterator, Optional

import cv2
import numpy as np


@dataclass
class FrameBatch:
    images: np.ndarray            # (B, H, W, 3) uint8 BGR — raw frames
    frame_indices: np.ndarray     # (B,) int64
    orig_h: int
    orig_w: int


def iterate_video_batches(
    video_path: str,
    batch_size: int,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> Iterator[FrameBatch]:
    """Yield FrameBatch objects of up to `batch_size` consecutive raw BGR frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if end_frame is None or end_frame > total:
            end_frame = total
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        orig_h = orig_w = None
        bb: list = []
        indices: list = []
        idx = start_frame

        while idx < end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            if orig_h is None:
                orig_h, orig_w = frame.shape[:2]
            bb.append(frame)
            indices.append(idx)
            idx += 1
            if len(bb) >= batch_size:
                yield FrameBatch(
                    images=np.stack(bb, axis=0),
                    frame_indices=np.array(indices, dtype=np.int64),
                    orig_h=orig_h, orig_w=orig_w,
                )
                bb.clear(); indices.clear()
        if bb:
            yield FrameBatch(
                images=np.stack(bb, axis=0),
                frame_indices=np.array(indices, dtype=np.int64),
                orig_h=orig_h, orig_w=orig_w,
            )
    finally:
        cap.release()


def count_video_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def prefetched_batches(
    video_path: str,
    batch_size: int,
    queue_size: int = 3,
    **kwargs,
) -> Iterator[FrameBatch]:
    """Decode video in a background thread so cv2 reads overlap with GPU work.

    On RTX 2080 Ti the model.forward takes ~315 ms per batch of 16, which is
    long enough to fully hide ~80 ms of video decode per batch. Net effect:
    end-to-end fps goes from ~45 (sequential) to ~50 (prefetched).
    """
    q: "queue.Queue[Optional[FrameBatch]]" = queue.Queue(maxsize=queue_size)
    err: list = []

    def _worker():
        try:
            for batch in iterate_video_batches(
                video_path, batch_size=batch_size, **kwargs,
            ):
                q.put(batch)
        except Exception as e:
            err.append(e)
        finally:
            q.put(None)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    try:
        while True:
            item = q.get()
            if item is None:
                break
            yield item
        if err:
            raise err[0]
    finally:
        t.join(timeout=1.0)
