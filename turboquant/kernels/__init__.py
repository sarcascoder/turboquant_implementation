"""
Triton kernels for PlanarQuant / IsoQuant. Each module exposes a unified API
that dispatches to the Triton kernel on CUDA, or to a PyTorch fallback on
CPU/MPS/etc. Tests exercise the fallback path; production deployments on CUDA
get the speed.
"""
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None


def cuda_available_with_triton(t: torch.Tensor) -> bool:
    """True iff Triton is importable AND tensor is on a CUDA device."""
    return HAS_TRITON and t.is_cuda
