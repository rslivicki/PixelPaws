"""Detect whether a CUDA-capable GPU is available for DLC PyTorch inference.

Returns ordered list (best first). For DLC PyTorch inference the choice is
binary: CUDA via `torch.cuda` or CPU fallback. We also auto-suggest a batch
size based on detected VRAM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ProviderInfo:
    name: str                     # "cuda" or "cpu"
    display_name: str
    device_label: str = ""        # e.g. "NVIDIA GeForce RTX 3060"
    vram_gb: Optional[float] = None
    is_gpu: bool = False
    suggested_batch: int = 1
    notes: str = ""

    def __str__(self) -> str:
        bits = [self.display_name]
        if self.device_label:
            bits.append(f"({self.device_label})")
        if self.vram_gb:
            bits.append(f"{self.vram_gb:.0f} GB")
        return " ".join(bits)


def _suggested_batch_for_vram(vram_gb: Optional[float]) -> int:
    if vram_gb is None:
        return 8
    if vram_gb >= 16:
        return 32
    if vram_gb >= 8:
        return 16
    if vram_gb >= 4:
        return 8
    return 4


def probe_providers() -> List[ProviderInfo]:
    """Best-first list of available providers; CPU is always last."""
    out: List[ProviderInfo] = []
    try:
        import torch
    except ImportError:
        return [ProviderInfo(
            name="cpu",
            display_name="CPU (PyTorch not installed)",
            suggested_batch=1,
            notes="Run install.bat to set up the env.",
        )]

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / 1e9
            out.append(ProviderInfo(
                name=f"cuda:{i}",
                display_name="NVIDIA CUDA",
                device_label=props.name,
                vram_gb=vram_gb,
                is_gpu=True,
                suggested_batch=_suggested_batch_for_vram(vram_gb),
                notes="Full DLC PyTorch inference speed.",
            ))

    out.append(ProviderInfo(
        name="cpu",
        display_name="CPU",
        suggested_batch=1,
        is_gpu=False,
        notes="No GPU acceleration. Inference will be 20-30x slower than GPU.",
    ))
    return out
