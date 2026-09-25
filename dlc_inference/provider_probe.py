"""Detect which compute devices DLC PyTorch inference can use.

Returns an ordered list (best first). Recognised backends:

  cuda:N  NVIDIA via torch.cuda - only when this torch build actually has
          kernels for the card's compute capability (a cu118 wheel cannot
          drive an RTX 50-series card; that used to crash mid-run).
  xpu:N   Intel Arc / Data Center GPU via torch.xpu (torch >= 2.5 with the
          Intel XPU wheel).
  mps     Apple Silicon via torch.backends.mps.
  cpu     always last.

GPUs that are present but unusable (AMD/Intel on Windows without a torch
backend, NVIDIA with a driver too old for CUDA, unsupported architecture)
are reported inside the CPU entry's label so the user learns *why* they
are on CPU instead of seeing a bare "CPU".

`resolve_device(name)` maps any requested device string to one that is
actually available (falling back to cpu), for use by the runner.
"""

from __future__ import annotations

import platform
import re
import subprocess
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ProviderInfo:
    name: str                     # "cuda:0", "xpu:0", "mps" or "cpu"
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


# --------------------------------------------------------------------------- #
# OS-level adapter listing (what is physically in the machine)
# --------------------------------------------------------------------------- #

def list_display_adapters() -> List[str]:
    """Names of the GPUs the OS knows about, regardless of PyTorch support."""
    names: List[str] = []
    try:
        if platform.system() == "Windows":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_VideoController | "
                 "Select-Object -ExpandProperty Name"],
                capture_output=True, text=True, timeout=20,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).stdout
            names = [l.strip() for l in out.splitlines() if l.strip()]
        elif platform.system() == "Linux":
            out = subprocess.run(["lspci"], capture_output=True, text=True,
                                 timeout=20).stdout
            names = [l.split(":", 2)[-1].strip() for l in out.splitlines()
                     if re.search(r"VGA|3D|Display", l)]
    except Exception:
        pass
    return [n for n in names
            if "microsoft basic" not in n.lower()
            and "remote display" not in n.lower()]


def _vendor(name: str) -> str:
    n = name.lower()
    if "nvidia" in n or "geforce" in n or "quadro" in n or "tesla" in n:
        return "nvidia"
    if "amd" in n or "radeon" in n:
        return "amd"
    if "intel" in n or "arc" in n:
        return "intel"
    if "apple" in n:
        return "apple"
    return "other"


# --------------------------------------------------------------------------- #
# CUDA architecture compatibility
# --------------------------------------------------------------------------- #

def _cuda_arch_supported(torch, index: int) -> bool:
    """True when this torch build has kernels (or forward-compatible PTX)
    for CUDA device `index`. Unknown -> assume supported."""
    try:
        major, minor = torch.cuda.get_device_capability(index)
        arch_list = list(torch.cuda.get_arch_list())
    except Exception:
        return True
    if not arch_list:
        return True
    cap = major * 10 + minor
    for a in arch_list:
        # SASS (sm_XY) is binary-compatible with any card of the SAME major
        # and an equal-or-higher minor: the cu126 wheel ships sm_86 and no
        # sm_89, yet runs on every RTX 40-series (8.9). Demanding an exact
        # match wrongly pushed all Ada cards to CPU (2026-08-28).
        m = re.match(r"sm_(\d+)$", a)          # sm_86, sm_120 (three digits)
        if m:
            n = int(m.group(1))
            if n // 10 == major and n % 10 <= minor:
                return True
            continue
        # PTX (compute_XY) JIT-compiles forward onto any capability >= XY.
        m = re.match(r"compute_(\d+)$", a)
        if m and int(m.group(1)) <= cap:
            return True
    return False


# --------------------------------------------------------------------------- #
# Probe
# --------------------------------------------------------------------------- #

def probe_providers() -> List[ProviderInfo]:
    """Best-first list of available providers; CPU is always last."""
    out: List[ProviderInfo] = []
    cpu_notes: List[str] = []
    try:
        import torch
    except ImportError:
        return [ProviderInfo(
            name="cpu",
            display_name="CPU (PyTorch not installed)",
            suggested_batch=1,
            notes="Run install.bat to set up the env.",
        )]

    adapters = list_display_adapters()
    matched_labels: List[str] = []

    # -- NVIDIA / CUDA -------------------------------------------------------
    cuda_ok = False
    try:
        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = False
    if cuda_ok:
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
            except Exception:
                continue
            vram_gb = props.total_memory / 1e9
            matched_labels.append(props.name)
            if not _cuda_arch_supported(torch, i):
                cap = ".".join(map(str, torch.cuda.get_device_capability(i)))
                cpu_notes.append(
                    f"{props.name} found, but this PyTorch build "
                    f"(CUDA {getattr(torch.version, 'cuda', '?')}) has no "
                    f"kernels for compute capability {cap} - a newer CUDA "
                    f"wheel is needed (edit installer/environment.yml "
                    f"extra-index-url, e.g. cu128, and reinstall)")
                continue
            out.append(ProviderInfo(
                name=f"cuda:{i}",
                display_name="NVIDIA CUDA",
                device_label=props.name,
                vram_gb=vram_gb,
                is_gpu=True,
                suggested_batch=_suggested_batch_for_vram(vram_gb),
                notes="Full DLC PyTorch inference speed.",
            ))

    # -- Intel XPU -----------------------------------------------------------
    try:
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            for i in range(xpu.device_count()):
                label, vram_gb = f"Intel XPU {i}", None
                try:
                    p = xpu.get_device_properties(i)
                    label = getattr(p, "name", label)
                    tm = getattr(p, "total_memory", None)
                    vram_gb = tm / 1e9 if tm else None
                except Exception:
                    pass
                matched_labels.append(label)
                out.append(ProviderInfo(
                    name=f"xpu:{i}",
                    display_name="Intel XPU",
                    device_label=label,
                    vram_gb=vram_gb,
                    is_gpu=True,
                    suggested_batch=_suggested_batch_for_vram(vram_gb),
                    notes="Intel GPU via torch.xpu.",
                ))
    except Exception:
        pass

    # -- Apple MPS -----------------------------------------------------------
    try:
        if torch.backends.mps.is_available():
            out.append(ProviderInfo(
                name="mps",
                display_name="Apple GPU (MPS)",
                is_gpu=True,
                suggested_batch=8,
                notes="Apple Silicon via Metal.",
            ))
            matched_labels.append("Apple")
    except Exception:
        pass

    # -- GPUs the OS sees but torch cannot use --------------------------------
    for name in adapters:
        if any(name.lower() in m.lower() or m.lower() in name.lower()
               for m in matched_labels):
            continue
        v = _vendor(name)
        if v == "nvidia":
            if not cuda_ok:
                if getattr(torch.version, "cuda", None) is None:
                    import sys as _sys
                    why = (f"the PyTorch in use ({torch.__version__}, python: "
                           f"{_sys.executable}) has no CUDA support - if the "
                           "installer reported the GPU as ready, this is the "
                           "wrong Python environment; otherwise re-run "
                           "installer\\install.bat and pick 'Fresh reinstall'")
                else:
                    why = ("CUDA is not available to PyTorch - update the "
                           "NVIDIA driver (GeForce Experience or nvidia.com) "
                           "and relaunch")
                # the NVIDIA explanation is the one the user needs first
                cpu_notes.insert(0, f"{name} found, but {why}")
        elif v == "amd":
            cpu_notes.append(
                f"{name} found - AMD GPUs are not supported by PyTorch on "
                f"{platform.system()} for DLC inference")
        elif v == "intel":
            cpu_notes.append(
                f"{name} found - Intel GPUs need the Intel XPU PyTorch wheel "
                f"(not installed)")
        # 'other' (virtual adapters, basic display) is ignored

    if not out and not cpu_notes and not adapters:
        cpu_notes.append("no GPU detected")

    # Keep the dropdown text short; the full reason travels in `notes` and
    # the UIs show it as a wrapped line under the device box.
    cpu_label = "CPU"
    if not out and cpu_notes:
        cpu_label = "CPU (a GPU was found but cannot be used - see note)"
    out.append(ProviderInfo(
        name="cpu",
        display_name=cpu_label,
        suggested_batch=1,
        is_gpu=False,
        notes=("; ".join(cpu_notes) if cpu_notes else
               "No GPU acceleration. Inference will be 20-30x slower than GPU."),
    ))
    return out


def resolve_device(device: str) -> str:
    """Map a requested device ("cuda", "cuda:1", "xpu", "mps", "cpu", ...)
    to one PyTorch can actually run on now; falls back to "cpu"."""
    d = (device or "cpu").strip().lower()
    if d == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    kind, _, idx = d.partition(":")
    try:
        if kind == "cuda":
            if not torch.cuda.is_available():
                return "cpu"
            i = int(idx) if idx else 0
            if i >= torch.cuda.device_count() or not _cuda_arch_supported(torch, i):
                return "cpu"
            return f"cuda:{i}"
        if kind == "xpu":
            xpu = getattr(torch, "xpu", None)
            if xpu is None or not xpu.is_available():
                return "cpu"
            i = int(idx) if idx else 0
            return f"xpu:{i}" if i < xpu.device_count() else "cpu"
        if kind == "mps":
            return "mps" if torch.backends.mps.is_available() else "cpu"
    except Exception:
        return "cpu"
    return "cpu"
