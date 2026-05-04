"""
IsoQuant — block-diagonal rotation quantizer using d/4 independent 4D
quaternion rotations (Hamilton products).

Why this exists when PlanarQuant already covers d/2:
- A single 2D Givens rotation has 1 degree of freedom (the angle θ).
- A single 4D quaternion rotation has 3 (fast mode) or 6 (full mode) DOF.
- At higher bit budgets (4-bit) the extra DOF lets the rotation tune the
  marginal distribution per-coordinate more carefully → small but real PPL win.
- The plan reports: 4-bit Qwen2.5-3B post-prefill PPL of 9.03 (IsoQuant) vs
  10.12 (PlanarQuant). Same MSE bound, but IsoQuant is the production default
  for 4-bit symmetric configs.

Two operating modes:
- 'fast'  : v_rot = q_L · v               (3 DOF per block; 16 FMAs/block)
- 'full'  : v_rot = q_L · v · conj(q_R)   (6 DOF per block; 32 FMAs/block)

Inverse uses conjugate quaternions (the quaternion group's inverse is the
conjugate, normalised by squared magnitude — but for unit quaternions the
conjugate IS the inverse). This is the IsoQuant analogue of "negate sin in
PlanarQuant" — the single most error-prone line.
"""
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

from turboquant.lloyd_max import get_codebook
from turboquant.bit_packing import pack_bits, unpack_bits


# --------------------------------------------------------------------------- #
# Quaternion algebra (vectorised over batch + group)
# --------------------------------------------------------------------------- #
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Hamilton product a · b for batched quaternions.
        a, b: (..., 4)   layout (w, x, y, z)
    Returns: (..., 4)
    16 FMAs per quaternion product.
    """
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    rw = aw * bw - ax * bx - ay * by - az * bz
    rx = aw * bx + ax * bw + ay * bz - az * by
    ry = aw * by - ax * bz + ay * bw + az * bx
    rz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack([rw, rx, ry, rz], dim=-1)


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of a quaternion: (w, x, y, z) → (w, -x, -y, -z)."""
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)


def random_unit_quaternions(n: int, seed: Optional[int] = None) -> torch.Tensor:
    """Generate n random unit quaternions, uniform on S³ (the 3-sphere)."""
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        q = torch.randn(n, 4, generator=g)
    else:
        q = torch.randn(n, 4)
    return q / q.norm(dim=-1, keepdim=True)


# --------------------------------------------------------------------------- #
# IsoQuant module
# --------------------------------------------------------------------------- #
class IsoQuant(nn.Module):
    """
    Block-diagonal quaternion rotation quantizer. Drop-in replacement for
    PlanarQuant / TurboQuantMSE with the same .quantize / .dequantize API.

    Args:
        dim: head dimension. Must be a multiple of 4 (or set pad_to_4=True).
        bits: per-coordinate bit budget (1..8). Lloyd-Max codebook shared
              with TurboQuantMSE / PlanarQuant for the same (dim, bits).
        mode: 'fast' (1 quaternion/block, 3 DOF) or 'full' (2 quaternions/block, 6 DOF).
        seed: fixed seed for the rotation params. ALWAYS pass when persisting.
        pad_to_4: pad dim to nearest multiple of 4 if needed.
    """
    def __init__(self, dim: int, bits: int, mode: str = "fast",
                 seed: Optional[int] = None, pad_to_4: bool = True):
        super().__init__()
        assert mode in ("fast", "full"), f"mode must be 'fast' or 'full', got {mode}"
        self.dim = dim
        self.bits = bits
        self.mode = mode

        # Padding to multiple of 4
        rem = dim % 4
        if rem != 0:
            assert pad_to_4, f"dim={dim} not divisible by 4; set pad_to_4=True"
            self._padded_dim = dim + (4 - rem)
        else:
            self._padded_dim = dim

        self.n_groups = self._padded_dim // 4

        # ── Codebook (shared with PlanarQuant / TurboQuantMSE) ──
        cb = get_codebook(dim, bits)
        self.register_buffer("centroids", cb["centroids"])
        self.register_buffer("boundaries", cb["boundaries"])

        # ── Random unit quaternions, one per group ──
        seed_l = seed
        seed_r = (seed + 1_000_003) if seed is not None else None
        q_L = random_unit_quaternions(self.n_groups, seed=seed_l)
        self.register_buffer("q_L", q_L)
        if mode == "full":
            q_R = random_unit_quaternions(self.n_groups, seed=seed_r)
            self.register_buffer("q_R", q_R)
        else:
            self.q_R = None

    # --------------------------------------------------------------------- #
    # Forward / inverse rotation
    # --------------------------------------------------------------------- #
    def _forward_rotate(self, blocks: torch.Tensor) -> torch.Tensor:
        """
        blocks: (..., n_groups, 4) raw 4-tuples.
        Returns: (..., n_groups, 4) rotated.
        """
        q_L = self.q_L.to(blocks.dtype)
        # Broadcast q_L (n_groups, 4) over leading batch dims of blocks (..., n_groups, 4)
        out = quat_mul(q_L, blocks)
        if self.mode == "full":
            q_R = self.q_R.to(blocks.dtype)
            out = quat_mul(out, quat_conj(q_R))
        return out

    def _inverse_rotate(self, blocks: torch.Tensor) -> torch.Tensor:
        """
        Inverse of _forward_rotate. For unit quaternions, q⁻¹ = conj(q).
        Fast inverse:    v = conj(q_L) · v_rot
        Full inverse:    v = conj(q_L) · v_rot · q_R
        """
        q_L = self.q_L.to(blocks.dtype)
        q_L_conj = quat_conj(q_L)
        if self.mode == "full":
            q_R = self.q_R.to(blocks.dtype)
            tmp = quat_mul(blocks, q_R)
            return quat_mul(q_L_conj, tmp)
        return quat_mul(q_L_conj, blocks)

    # --------------------------------------------------------------------- #
    # Padding helpers
    # --------------------------------------------------------------------- #
    def _maybe_pad(self, x: torch.Tensor) -> torch.Tensor:
        if self._padded_dim == self.dim:
            return x
        pad_len = self._padded_dim - self.dim
        pad_shape = list(x.shape)
        pad_shape[-1] = pad_len
        zeros = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        return torch.cat([x, zeros], dim=-1)

    def _maybe_trim(self, x: torch.Tensor) -> torch.Tensor:
        if self._padded_dim == self.dim:
            return x
        return x[..., :self.dim]

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1) Norm separation
        norms = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        safe_norms = norms.clamp(min=1e-8)
        x_unit = x / safe_norms

        # 2) Pad, reshape to 4-blocks
        x_padded = self._maybe_pad(x_unit)
        blocks = x_padded.reshape(*x_padded.shape[:-1], self.n_groups, 4)

        # 3) Rotate each block
        rotated = self._forward_rotate(blocks)
        flat = rotated.reshape(*x_padded.shape[:-1], self._padded_dim)

        # 4) Scalar quantize
        inner_b = self.boundaries[1:-1].to(flat.dtype)
        indices = torch.searchsorted(inner_b, flat.contiguous())

        # 5) Bit-pack
        packed = pack_bits(indices, self.bits)
        return packed, norms

    def dequantize(self, packed: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        indices = unpack_bits(packed, self.bits, original_last_dim=self._padded_dim)
        values = self.centroids.to(norms.dtype)[indices.long()]

        blocks = values.reshape(*values.shape[:-1], self.n_groups, 4)
        unrotated = self._inverse_rotate(blocks)
        flat = unrotated.reshape(*values.shape[:-1], self._padded_dim)

        flat = self._maybe_trim(flat)
        return flat * norms

    # --------------------------------------------------------------------- #
    # Diagnostics
    # --------------------------------------------------------------------- #
    @property
    def fma_count_round_trip(self) -> int:
        """16 FMAs per quat_mul; fast = 1 product fwd + 1 product inv = 32/group;
        full = 2 products fwd + 2 products inv = 64/group."""
        per_group = 32 if self.mode == "fast" else 64
        return per_group * self.n_groups

    @property
    def parameter_count(self) -> int:
        """4 floats per quaternion; fast = 1 quat/group; full = 2 quat/group."""
        per_group = 4 if self.mode == "fast" else 8
        return per_group * self.n_groups
