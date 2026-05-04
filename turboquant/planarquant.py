"""
PlanarQuant — block-diagonal rotation quantizer using d/2 independent 2D Givens
rotations on adjacent coordinate pairs.

Why this beats TurboQuant for KV-cache use:
- TurboQuant rotation: dense (d×d) matmul → ~33,000 FMAs and 16,000 params at d=128.
- PlanarQuant rotation: per-pair → 256 forward + 256 inverse = 512 FMAs and 128 params at d=128.
- Same Lloyd-Max codebook (the post-rotation distribution is still ~Beta/Gaussian per coord).
- Fits in registers → enables Triton fused attention kernels in Phase 7/8.

INVERSE ROTATION TRAP — the inverse Givens IS NOT the same as the forward.
For a 2x2 rotation R(θ) = [[c, -s], [s, c]]:
  forward:  r0 =  c·v0 - s·v1
            r1 =  s·v0 + c·v1
  inverse (= R(θ).T):
            v0 =  c·r0 + s·r1
            v1 = -s·r0 + c·r1            ← negate sin
Applying the forward rotation in the dequant path silently doubles PPL.
Applying it in the V dequant path (Phase 6) explodes PPL to >15,000.
"""
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

from turboquant.lloyd_max import get_codebook
from turboquant.bit_packing import pack_bits, unpack_bits


def givens_forward(pairs: torch.Tensor, rot2: torch.Tensor) -> torch.Tensor:
    """
    Apply forward 2D Givens rotation per pair.
        pairs: (..., n_groups, 2)
        rot2:  (n_groups, 2)  — last dim is (cos, sin)
    Returns: (..., n_groups, 2)
    """
    cos_t = rot2[..., 0].to(pairs.dtype)
    sin_t = rot2[..., 1].to(pairs.dtype)
    v0 = pairs[..., 0]
    v1 = pairs[..., 1]
    r0 = cos_t * v0 - sin_t * v1
    r1 = sin_t * v0 + cos_t * v1
    return torch.stack([r0, r1], dim=-1)


def givens_inverse(pairs: torch.Tensor, rot2: torch.Tensor) -> torch.Tensor:
    """
    Apply inverse 2D Givens rotation per pair (transpose of the forward 2x2).
    Note the negation of sin in the second row.
    """
    cos_t = rot2[..., 0].to(pairs.dtype)
    sin_t = rot2[..., 1].to(pairs.dtype)
    q0 = pairs[..., 0]
    q1 = pairs[..., 1]
    v0 =  cos_t * q0 + sin_t * q1
    v1 = -sin_t * q0 + cos_t * q1
    return torch.stack([v0, v1], dim=-1)


class PlanarQuant(nn.Module):
    """
    Block-diagonal rotation quantizer using d/2 independent Givens rotations.
    Drop-in replacement for TurboQuantMSE with the same .quantize / .dequantize API.

    Args:
        dim: head dimension. Must be even (or set pad_to_even=True).
        bits: per-coordinate bit budget (1..8). Lloyd-Max codebook is shared
              with TurboQuantMSE for the same (dim, bits) — the post-rotation
              distribution is approximately the same Beta(d) / Gaussian(0, 1/d).
        seed: optional fixed seed for the rotation angles. Use the SAME seed
              when storing and reading from a cache; mismatched angles → garbage.
        pad_to_even: if dim is odd, pad to dim+1 internally and trim.
    """
    def __init__(self, dim: int, bits: int, seed: Optional[int] = None,
                 pad_to_even: bool = True):
        super().__init__()
        self.dim = dim
        self.bits = bits

        # Padding for odd dimensions
        if dim % 2 != 0:
            assert pad_to_even, f"dim={dim} is odd; set pad_to_even=True"
            self._padded_dim = dim + 1
        else:
            self._padded_dim = dim

        self.n_groups = self._padded_dim // 2

        # ── Codebook (shared with TurboQuantMSE for the same (dim, bits)) ──
        cb = get_codebook(dim, bits)
        self.register_buffer("centroids", cb["centroids"])
        self.register_buffer("boundaries", cb["boundaries"])

        # ── Random Givens angles, one per pair ──
        if seed is not None:
            g = torch.Generator().manual_seed(seed)
            angles = torch.empty(self.n_groups).uniform_(0, 2 * math.pi, generator=g)
        else:
            angles = torch.empty(self.n_groups).uniform_(0, 2 * math.pi)
        rot2 = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        # Shape: (n_groups, 2)
        self.register_buffer("rot2", rot2)

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _maybe_pad(self, x: torch.Tensor) -> torch.Tensor:
        if self._padded_dim == self.dim:
            return x
        pad_shape = list(x.shape)
        pad_shape[-1] = 1
        zeros = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        return torch.cat([x, zeros], dim=-1)

    def _maybe_trim(self, x: torch.Tensor) -> torch.Tensor:
        if self._padded_dim == self.dim:
            return x
        return x[..., :self.dim]

    # --------------------------------------------------------------------- #
    # API: quantize / dequantize
    # --------------------------------------------------------------------- #
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (..., dim) → (packed_indices, norms)
        norms: (..., 1)
        """
        # 1) Norm separation
        norms = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        safe_norms = norms.clamp(min=1e-8)
        x_unit = x / safe_norms

        # 2) Pad to even dim if needed, reshape to pairs
        x_padded = self._maybe_pad(x_unit)
        pairs = x_padded.reshape(*x_padded.shape[:-1], self.n_groups, 2)

        # 3) Forward Givens rotation (per pair, decoupled across groups)
        rotated = givens_forward(pairs, self.rot2)
        flat = rotated.reshape(*x_padded.shape[:-1], self._padded_dim)

        # 4) Scalar quantize via inner-boundary searchsorted
        inner_b = self.boundaries[1:-1].to(flat.dtype)
        indices = torch.searchsorted(inner_b, flat.contiguous())

        # 5) Bit-pack indices (uint8)
        packed = pack_bits(indices, self.bits)
        return packed, norms

    def dequantize(self, packed: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct x_hat ≈ x.
        Uses INVERSE Givens rotation (transpose, with negated sin).
        """
        # 1) Unpack indices → centroid values
        indices = unpack_bits(packed, self.bits, original_last_dim=self._padded_dim)
        values = self.centroids.to(norms.dtype)[indices.long()]

        # 2) Reshape to pairs, apply INVERSE Givens
        pairs = values.reshape(*values.shape[:-1], self.n_groups, 2)
        unrotated = givens_inverse(pairs, self.rot2)
        flat = unrotated.reshape(*values.shape[:-1], self._padded_dim)

        # 3) Trim padding (if any), denormalize
        flat = self._maybe_trim(flat)
        return flat * norms

    # --------------------------------------------------------------------- #
    # Diagnostics
    # --------------------------------------------------------------------- #
    @property
    def fma_count_round_trip(self) -> int:
        """4 FMAs per pair forward + 4 per pair inverse = 8 * n_groups."""
        return 8 * self.n_groups

    @property
    def parameter_count(self) -> int:
        """One (cos, sin) pair per Givens block = 2 * n_groups stored params."""
        return 2 * self.n_groups
