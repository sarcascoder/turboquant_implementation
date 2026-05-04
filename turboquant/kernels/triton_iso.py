"""
Triton kernels for IsoQuant. Same three-operation API as triton_planar:
  A. fused_round_trip
  B. quantize_only
  C. dequantize_only

Each Hamilton product is 16 FMAs; fast mode does 1 product fwd + 1 inv per
group; full mode does 2 fwd + 2 inv. The kernel keeps blocks of 4 floats
in registers throughout.
"""
import torch
from typing import Tuple

from turboquant.kernels import HAS_TRITON, cuda_available_with_triton, tl, triton
from turboquant.isoquant import quat_mul, quat_conj
from turboquant.bit_packing import pack_bits, unpack_bits


# --------------------------------------------------------------------------- #
# Triton kernel — fast mode only (full mode is straight extension)
# --------------------------------------------------------------------------- #
if HAS_TRITON:

    @triton.jit
    def _quantize_nearest_iso(val, centroids_ptr, n_levels: tl.constexpr):
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
    def _iso_fused_kernel_fast(
        input_ptr, output_ptr,
        qL_ptr,                # (n_groups, 4) — packed (w, x, y, z)
        centroids_ptr,
        n_groups, emb_dim,
        n_levels: tl.constexpr,
        BLOCK_G: tl.constexpr,
    ):
        """
        Fused round-trip in fast mode:
            v_rot = q_L · v
            v_q   = quantize(v_rot)
            v_hat = conj(q_L) · v_q
        """
        pid_b = tl.program_id(0)
        pid_g = tl.program_id(1)
        g_offs = pid_g * BLOCK_G + tl.arange(0, BLOCK_G)
        g_mask = g_offs < n_groups

        # Load q_L for this group
        qw = tl.load(qL_ptr + g_offs * 4 + 0, mask=g_mask, other=1.0)
        qx = tl.load(qL_ptr + g_offs * 4 + 1, mask=g_mask, other=0.0)
        qy = tl.load(qL_ptr + g_offs * 4 + 2, mask=g_mask, other=0.0)
        qz = tl.load(qL_ptr + g_offs * 4 + 3, mask=g_mask, other=0.0)

        # Load input 4-block
        d0 = g_offs * 4
        v0 = tl.load(input_ptr + pid_b * emb_dim + d0 + 0, mask=g_mask, other=0.0)
        v1 = tl.load(input_ptr + pid_b * emb_dim + d0 + 1, mask=g_mask, other=0.0)
        v2 = tl.load(input_ptr + pid_b * emb_dim + d0 + 2, mask=g_mask, other=0.0)
        v3 = tl.load(input_ptr + pid_b * emb_dim + d0 + 3, mask=g_mask, other=0.0)

        # Hamilton product q_L · v   (treat v as quaternion (v0, v1, v2, v3))
        rw = qw * v0 - qx * v1 - qy * v2 - qz * v3
        rx = qw * v1 + qx * v0 + qy * v3 - qz * v2
        ry = qw * v2 - qx * v3 + qy * v0 + qz * v1
        rz = qw * v3 + qx * v2 - qy * v1 + qz * v0

        # Quantize each component
        q0 = _quantize_nearest_iso(rw, centroids_ptr, n_levels)
        q1 = _quantize_nearest_iso(rx, centroids_ptr, n_levels)
        q2 = _quantize_nearest_iso(ry, centroids_ptr, n_levels)
        q3 = _quantize_nearest_iso(rz, centroids_ptr, n_levels)

        # Inverse: conj(q_L) · q   (negate x, y, z components of q_L)
        # conj(q_L) = (qw, -qx, -qy, -qz)
        f0 =  qw * q0 - (-qx) * q1 - (-qy) * q2 - (-qz) * q3   # = qw·q0 + qx·q1 + qy·q2 + qz·q3
        f1 =  qw * q1 + (-qx) * q0 + (-qy) * q3 - (-qz) * q2   # = qw·q1 - qx·q0 - qy·q3 + qz·q2
        f2 =  qw * q2 - (-qx) * q3 + (-qy) * q0 + (-qz) * q1   # = qw·q2 + qx·q3 - qy·q0 - qz·q1
        f3 =  qw * q3 + (-qx) * q2 - (-qy) * q1 + (-qz) * q0   # = qw·q3 - qx·q2 + qy·q1 - qz·q0
        # Simplified inverse expressions:
        f0 = qw * q0 + qx * q1 + qy * q2 + qz * q3
        f1 = qw * q1 - qx * q0 - qy * q3 + qz * q2
        f2 = qw * q2 + qx * q3 - qy * q0 - qz * q1
        f3 = qw * q3 - qx * q2 + qy * q1 - qz * q0

        tl.store(output_ptr + pid_b * emb_dim + d0 + 0, f0, mask=g_mask)
        tl.store(output_ptr + pid_b * emb_dim + d0 + 1, f1, mask=g_mask)
        tl.store(output_ptr + pid_b * emb_dim + d0 + 2, f2, mask=g_mask)
        tl.store(output_ptr + pid_b * emb_dim + d0 + 3, f3, mask=g_mask)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def iso_fused_round_trip(x: torch.Tensor, quantizer) -> torch.Tensor:
    """Rotate → quantize → inverse rotate, fused. PyTorch fallback for non-CUDA."""
    assert x.shape[-1] == quantizer.dim
    assert x.dim() == 2

    if not cuda_available_with_triton(x):
        return _iso_fused_round_trip_pytorch(x, quantizer)

    if quantizer.mode != "fast":
        # Full-mode kernel left as future work; fall back to PyTorch.
        return _iso_fused_round_trip_pytorch(x, quantizer)

    out = torch.empty_like(x)
    n_groups = quantizer.n_groups
    n_levels = 2 ** quantizer.bits
    BLOCK_G = min(32, triton.next_power_of_2(n_groups))
    grid = (x.shape[0], triton.cdiv(n_groups, BLOCK_G))

    centroids = quantizer.centroids.to(x.dtype).contiguous()
    qL = quantizer.q_L.to(x.dtype).contiguous()

    _iso_fused_kernel_fast[grid](
        x.contiguous(), out,
        qL, centroids,
        n_groups, quantizer.dim,
        n_levels=n_levels,
        BLOCK_G=BLOCK_G,
    )
    return out


def _iso_fused_round_trip_pytorch(x: torch.Tensor, quantizer) -> torch.Tensor:
    """PyTorch fallback. Identical math to the Triton kernel."""
    blocks = x.reshape(*x.shape[:-1], quantizer.n_groups, 4)
    rotated = quantizer._forward_rotate(blocks)

    flat = rotated.reshape(*x.shape[:-1], quantizer.dim)
    inner_b = quantizer.boundaries[1:-1].to(flat.dtype)
    indices = torch.searchsorted(inner_b, flat.contiguous())
    values = quantizer.centroids.to(flat.dtype)[indices.long()]

    blocks_q = values.reshape(*values.shape[:-1], quantizer.n_groups, 4)
    unrot = quantizer._inverse_rotate(blocks_q)
    return unrot.reshape(*x.shape[:-1], quantizer.dim)


def iso_quantize_only(x, quantizer):
    return quantizer.quantize(x)


def iso_dequantize_only(packed, norms, quantizer):
    return quantizer.dequantize(packed, norms)
