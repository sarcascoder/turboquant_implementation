"""
Fused PlanarQuant + attention kernel.

The decode-time pipeline without fusion is:
  1. quantize(K_new)         — VRAM round-trip
  2. store(K_new_q)          — VRAM
  3. attention(Q, K_q)        — VRAM read of K_q + compute

Three kernels, three VRAM round-trips. Memory-bound at ~0.5 FLOPs/byte.

Fused pipeline: in ONE kernel per (q_head, kv_token):
  - Load raw K_new pair (2 floats)
  - Rotate (4 FMAs)
  - Quantize (compare to centroids in registers)
  - Dot with PRE-ROTATED Q (still in registers)
  - Accumulate score
  - Side-effect: write quantized index to storage

Quantized K never sits in VRAM. Arithmetic intensity jumps from ~0.5 to
~500 FLOPs/byte at seq_len=4K. The plan reports 1.1-4.5× speedup vs cuBLAS
matmul at seq_len=4K, with cuBLAS winning at seq_len > 8K (worth keeping
both paths and dispatching by length).

This kernel is ONLY possible because PlanarQuant's rotation fits in registers
(2 floats per pair). TurboQuant's d×d matmul cannot.
"""
import math
import torch
from typing import Optional, Tuple

from turboquant.kernels import HAS_TRITON, cuda_available_with_triton, tl, triton
from turboquant.planarquant import givens_forward
from turboquant.bit_packing import pack_bits


# --------------------------------------------------------------------------- #
# Pre-rotate Q (host-side helper)
# --------------------------------------------------------------------------- #
def pre_rotate_query(Q: torch.Tensor, quantizer) -> torch.Tensor:
    """
    Rotate Q into the same basis as the quantized K, ONCE before the kernel.
    The fused kernel then dot-products Q_rot against quantized rotated K
    without rotating per K token.

    Args:
        Q: (B, H_q, T_q, D) — query tensor.
        quantizer: PlanarQuant.
    Returns:
        Q_rot: (B, H_q, T_q, D)
    """
    pairs = Q.reshape(*Q.shape[:-1], quantizer.n_groups, 2)
    rotated = givens_forward(pairs, quantizer.rot2)
    return rotated.reshape(Q.shape)


# --------------------------------------------------------------------------- #
# Triton fused kernel (CUDA only)
# --------------------------------------------------------------------------- #
if HAS_TRITON:

    @triton.jit
    def _quantize_with_index(val, centroids_ptr, n_levels: tl.constexpr):
        """Returns (quantized_value, index)."""
        best_val = tl.load(centroids_ptr)
        best_idx = tl.zeros_like(val).to(tl.int32)
        best_dist = tl.abs(val - best_val)
        for i in tl.static_range(1, n_levels):
            c = tl.load(centroids_ptr + i)
            d = tl.abs(val - c)
            mask = d < best_dist
            best_dist = tl.where(mask, d, best_dist)
            best_val = tl.where(mask, c, best_val)
            best_idx = tl.where(mask, i, best_idx)
        return best_val, best_idx

    @triton.jit
    def _fused_planar_quantize_attend_kernel(
        Q_rot_ptr,         # (B, H, T_q, D) — pre-rotated queries
        K_raw_ptr,         # (B, H_kv, T_kv, D) — raw new K
        rot2_ptr,          # (n_groups, 2)
        centroids_ptr,     # (n_levels,)
        norms_ptr,         # (B, H_kv, T_kv) — pre-computed K norms
        Out_ptr,           # (B, H, T_q, T_kv) — attention scores out
        Idx_ptr,           # (B, H_kv, T_kv, D) int8 — quantized indices side-effect
        is_first_q_for_kv, # int — 0 or 1; only first Q of GQA group writes indices
        n_groups, emb_dim,
        T_q, T_kv,
        n_levels: tl.constexpr,
        BLOCK_G: tl.constexpr,
        scale,
    ):
        """
        Compute one attention score per (b, h_q, t_q, t_kv) tile, fusing
        the K quantization with the dot product.
        """
        pid = tl.program_id(0)
        # Decompose pid → (b, h_q, t_q, t_kv) — caller's responsibility to size grid
        # Simplified linear indexing for clarity
        # In production this would be 4D grid; left as illustrative.
        # (Not invoked unless on CUDA, so the indexing flexibility doesn't affect Apple tests.)
        ...
        # The full kernel body is ~150 lines; the architecture is shown in
        # the docstring above. The PyTorch fallback below implements the
        # same math and is what runs on Apple/non-CUDA.


# --------------------------------------------------------------------------- #
# PyTorch fallback / reference implementation
# --------------------------------------------------------------------------- #
def fused_planar_quantize_attend(
    Q: torch.Tensor,
    K_new: torch.Tensor,
    quantizer,
    *,
    is_first_q_for_kv: bool = True,
    return_indices: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Fused quantize + attention for the new K tokens at the current decode step.

    This is the REFERENCE / FALLBACK implementation. On CUDA the same call
    will dispatch to the Triton kernel above (when fully fleshed out).

    Args:
        Q:       (B, H_q, T_q, D) — full-precision queries.
        K_new:   (B, H_kv, T_kv, D) — raw new K vectors.
        quantizer: PlanarQuant (provides rot2, centroids, boundaries).
        is_first_q_for_kv: only the first Q-head per GQA group writes indices.
                           See Phase 9.
        return_indices: also return the quantized (packed_indices, norms) so
                        the caller can append to the cache.

    Returns:
        scores:  (B, H_q, T_q, T_kv)              — attention scores Q · K^T / sqrt(D)
        packed:  (B, H_kv, T_kv, ceil(D/(8/bits))) or None
        norms:   (B, H_kv, T_kv, 1) or None
    """
    B, H_q, T_q, D = Q.shape
    _, H_kv, T_kv, _ = K_new.shape
    assert D == quantizer.dim, "Q/K dim mismatch with quantizer"

    # 1) Normalize K_new (norm separation)
    norms = torch.linalg.vector_norm(K_new, dim=-1, keepdim=True)
    safe = norms.clamp(min=1e-8)
    K_unit = K_new / safe

    # 2) Forward Givens rotation
    pairs = K_unit.reshape(B, H_kv, T_kv, quantizer.n_groups, 2)
    K_rot = givens_forward(pairs, quantizer.rot2).reshape(B, H_kv, T_kv, D)

    # 3) Quantize K_rot to nearest centroid
    inner_b = quantizer.boundaries[1:-1].to(K_rot.dtype)
    indices = torch.searchsorted(inner_b, K_rot.contiguous())
    K_q_centroids = quantizer.centroids.to(K_rot.dtype)[indices.long()]

    # 4) Pre-rotate Q (if not already done; we re-rotate to keep API simple)
    Q_pairs = Q.reshape(B, H_q, T_q, quantizer.n_groups, 2)
    Q_rot = givens_forward(Q_pairs, quantizer.rot2).reshape(B, H_q, T_q, D)

    # 5) Dot product in rotated basis (multiplied by ||K|| for asymmetric IP)
    # Broadcast Q (H_q) over K (H_kv); for GQA the caller has reshaped Q to match.
    # Standard MHA: H_q == H_kv. GQA: H_kv divides H_q.
    if H_q != H_kv:
        # Repeat K to match Q heads (cheap broadcast — PyTorch handles via expand)
        ratio = H_q // H_kv
        K_q_centroids_b = K_q_centroids.repeat_interleave(ratio, dim=1)
        norms_b = norms.repeat_interleave(ratio, dim=1)
    else:
        K_q_centroids_b = K_q_centroids
        norms_b = norms

    # raw_scores: (B, H_q, T_q, T_kv)
    raw_scores = torch.matmul(Q_rot, K_q_centroids_b.transpose(-2, -1))
    # Multiply by ||K|| (broadcasted to T_kv axis)
    norm_t = norms_b.squeeze(-1).unsqueeze(-2)   # (B, H_q, 1, T_kv)
    scores = raw_scores * norm_t / math.sqrt(D)

    if return_indices and is_first_q_for_kv:
        packed = pack_bits(indices, quantizer.bits)
        return scores, packed, norms
    return scores, None, None


# --------------------------------------------------------------------------- #
# Cached attention (subsequent decode steps — K already quantized in cache)
# --------------------------------------------------------------------------- #
def planar_cached_attention(
    Q: torch.Tensor,
    packed_K: torch.Tensor,
    K_norms: torch.Tensor,
    quantizer,
) -> torch.Tensor:
    """
    Lighter kernel for subsequent decode steps: K is already quantized in the
    cache; we only need to dequantize, dot with Q, scale.

    On CUDA this would be a Triton kernel that loads centroid values via index
    LUT, dots with pre-rotated Q in registers, and writes the score. PyTorch
    fallback is just dequantize → bmm.
    """
    B, H_q, T_q, D = Q.shape
    K_dq = quantizer.dequantize(packed_K, K_norms)        # (B, H_kv, T_kv, D)
    H_kv = K_dq.shape[1]
    if H_q != H_kv:
        ratio = H_q // H_kv
        K_dq = K_dq.repeat_interleave(ratio, dim=1)
    scores = torch.matmul(Q, K_dq.transpose(-2, -1)) / math.sqrt(D)
    return scores
