"""
Triton kernels for PlanarQuant.

Three operations:
  A. fused_round_trip(x, rot2, centroids, boundaries)  → x_hat
  B. quantize_only(x, rot2, boundaries)                → indices, norms
  C. dequantize_only(indices, norms, rot2, centroids)  → x_hat

On CUDA hosts: Triton kernels (single-launch, register-resident).
On non-CUDA hosts (Apple Silicon, CPU): PyTorch fallback that uses the same
public API. Tests exercise the fallback.
"""
import torch
from typing import Tuple

from turboquant.kernels import HAS_TRITON, cuda_available_with_triton, tl, triton
from turboquant.planarquant import givens_forward, givens_inverse
from turboquant.bit_packing import pack_bits, unpack_bits


# --------------------------------------------------------------------------- #
# Triton kernels (compiled only when triton is importable)
# --------------------------------------------------------------------------- #
if HAS_TRITON:

    @triton.jit
    def _quantize_nearest(val, centroids_ptr, n_levels: tl.constexpr):
        """Find nearest centroid by exhaustive comparison (compile-time unrolled)."""
        best_val = tl.load(centroids_ptr)
        best_dist = tl.abs(val - best_val)
        for i in tl.static_range(1, n_levels):
            c = tl.load(centroids_ptr + i)
            d = tl.abs(val - c)
            mask = d < best_dist
            best_dist = tl.where(mask, d, best_dist)
            best_val = tl.where(mask, c, best_val)
        return best_val

    @triton.jit
    def _planar_fused_kernel(
        input_ptr, output_ptr,
        rot2_ptr, centroids_ptr,
        n_groups, emb_dim,
        n_levels: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        """
        Fused round-trip: rotate → quantize → inverse rotate, single kernel.
        Each program handles BLOCK_G groups of one batch row.

        Input:  (batch, emb_dim) flattened (C-contiguous).
        Output: (batch, emb_dim) reconstructed.
        """
        pid_b = tl.program_id(0)
        pid_g = tl.program_id(1)

        g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
        g_mask = g_offs < n_groups

        cos_t = tl.load(rot2_ptr + g_offs * 2 + 0, mask=g_mask, other=1.0)
        sin_t = tl.load(rot2_ptr + g_offs * 2 + 1, mask=g_mask, other=0.0)

        d0 = g_offs * 2
        d1 = d0 + 1

        v0 = tl.load(input_ptr + pid_b * emb_dim + d0, mask=g_mask, other=0.0)
        v1 = tl.load(input_ptr + pid_b * emb_dim + d1, mask=g_mask, other=0.0)

        # Forward rotate
        r0 = cos_t * v0 - sin_t * v1
        r1 = sin_t * v0 + cos_t * v1

        # Quantize to nearest centroid
        q0 = _quantize_nearest(r0, centroids_ptr, n_levels)
        q1 = _quantize_nearest(r1, centroids_ptr, n_levels)

        # Inverse rotate (note: -sin in second row)
        f0 =  cos_t * q0 + sin_t * q1
        f1 = -sin_t * q0 + cos_t * q1

        tl.store(output_ptr + pid_b * emb_dim + d0, f0, mask=g_mask)
        tl.store(output_ptr + pid_b * emb_dim + d1, f1, mask=g_mask)


# --------------------------------------------------------------------------- #
# Public API — dispatches Triton vs PyTorch
# --------------------------------------------------------------------------- #
def planar_fused_round_trip(x: torch.Tensor, quantizer) -> torch.Tensor:
    """
    Apply rotate → quantize → inverse rotate as a single fused operation.

    Args:
        x: (B, D) tensor (any dtype). For multi-axis input, flatten leading dims.
        quantizer: a PlanarQuant instance providing rot2 and centroids buffers.
    Returns:
        x_hat: same shape as x.
    """
    assert x.shape[-1] == quantizer.dim, "x last dim must match quantizer.dim"
    assert x.dim() == 2, f"expected (B, D); got shape {tuple(x.shape)}"

    # Note: this kernel does NOT separate norms — it's the in-register fused
    # path. For full pipeline use quantizer.quantize / dequantize.
    if not cuda_available_with_triton(x):
        # Fallback: PyTorch path
        return _planar_fused_round_trip_pytorch(x, quantizer)

    # CUDA + Triton path
    out = torch.empty_like(x)
    n_groups = quantizer.n_groups
    n_levels = 2 ** quantizer.bits
    BLOCK_G = min(64, triton.next_power_of_2(n_groups))
    grid = (x.shape[0], triton.cdiv(n_groups, BLOCK_G))

    centroids = quantizer.centroids.to(x.dtype).contiguous()
    rot2 = quantizer.rot2.to(x.dtype).contiguous()

    _planar_fused_kernel[grid](
        x.contiguous(), out,
        rot2, centroids,
        n_groups, quantizer.dim,
        n_levels=n_levels,
        BLOCK_G=BLOCK_G,
    )
    return out


def _planar_fused_round_trip_pytorch(x: torch.Tensor, quantizer) -> torch.Tensor:
    """PyTorch fallback. Bit-exact (modulo float-rounding) to the Triton path."""
    pairs = x.reshape(*x.shape[:-1], quantizer.n_groups, 2)
    rotated = givens_forward(pairs, quantizer.rot2)

    flat = rotated.reshape(*x.shape[:-1], quantizer.dim)

    # Quantize via searchsorted on inner boundaries (same as PlanarQuant.quantize)
    inner_b = quantizer.boundaries[1:-1].to(flat.dtype)
    indices = torch.searchsorted(inner_b, flat.contiguous())
    values = quantizer.centroids.to(flat.dtype)[indices.long()]

    # Inverse rotate
    pairs_q = values.reshape(*values.shape[:-1], quantizer.n_groups, 2)
    unrot = givens_inverse(pairs_q, quantizer.rot2)
    return unrot.reshape(*x.shape[:-1], quantizer.dim)


def planar_quantize_only(x: torch.Tensor, quantizer) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize-only kernel: rotate + quantize + bit-pack. Returns (packed, norms).
    Equivalent to PlanarQuant.quantize but exposed here for kernel parity tests.
    """
    return quantizer.quantize(x)


def planar_dequantize_only(packed: torch.Tensor, norms: torch.Tensor,
                            quantizer) -> torch.Tensor:
    """Dequantize-only kernel. Equivalent to PlanarQuant.dequantize."""
    return quantizer.dequantize(packed, norms)
