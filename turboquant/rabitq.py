"""
RaBitQ — 1-bit extreme compression for vector search and long-context retrieval.

Stores only the sign of each rotated coordinate plus a per-vector magnitude
scalar. Achieves 12.8x compression at d=128 (vs FP16). Quality is severely
degraded compared to 3+ bit quantization (PPL goes from ~7 to ~107 on
Qwen2.5-3B), so this is NOT for general LLM generation. Use cases:
- Approximate nearest-neighbor (ANN) search over 100K+ vectors
- Long-context retrieval where memory is the bottleneck
- Reranking pipelines where binary signs are "good enough" for top-k

The plan's critical finding:
  At 1-bit, ONLY the 'full' rotation backend (random d×d orthogonal matrix)
  gives acceptable quality. PlanarQuant/IsoQuant leave inter-group correlations
  that are tolerable at 3+ bits but FATAL at 1 bit. Confirmed below.

Storage per d=128 vector:
  sign bits:  d/8  =  16 bytes
  ||x||:      FP16 =   2 bytes
  x0:         FP16 =   2 bytes
  Total:           = 20 bytes  (vs 256 bytes FP16 → 12.8x)
"""
import math
import torch
import torch.nn as nn
from typing import Tuple

from turboquant.bit_packing import pack_bits, unpack_bits


def _make_full_rotation(dim: int, seed: int) -> torch.Tensor:
    """
    Random d×d orthogonal matrix via QR decomposition with sign-fix
    (so det = +1, no improper rotation).
    """
    g = torch.Generator().manual_seed(seed)
    rand = torch.randn(dim, dim, generator=g)
    Q, R = torch.linalg.qr(rand)
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    return Q * diag_sign


class RaBitQ(nn.Module):
    """
    1-bit quantizer with random rotation backend.

    Args:
        dim: vector dimension. Must be divisible by 8 for clean packing.
        rotation: 'full' (random d×d orthogonal) | 'planar' (d/2 Givens) | 'iso'
                  (d/4 quaternions). The plan's finding: ONLY 'full' gives
                  acceptable 1-bit quality. 'planar' / 'iso' are provided to
                  empirically reproduce that finding.
        seed: rotation seed.
    """
    def __init__(self, dim: int, rotation: str = "full", seed: int = 0):
        super().__init__()
        assert rotation in ("full", "planar", "iso"), \
            f"rotation must be 'full', 'planar', or 'iso', got {rotation}"
        assert dim % 8 == 0, f"dim={dim} must be divisible by 8 for sign packing"
        self.dim = dim
        self.rotation = rotation
        self.seed = seed

        if rotation == "full":
            self.register_buffer("R", _make_full_rotation(dim, seed))
        elif rotation == "planar":
            from turboquant.planarquant import PlanarQuant
            self._planar = PlanarQuant(dim=dim, bits=1, seed=seed)
        elif rotation == "iso":
            from turboquant.isoquant import IsoQuant
            self._iso = IsoQuant(dim=dim, bits=1, mode="fast", seed=seed)

    # --------------------------------------------------------------------- #
    # Rotation helpers
    # --------------------------------------------------------------------- #
    def _rotate_forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., dim) → rotated (..., dim). Norm-preserving."""
        if self.rotation == "full":
            return torch.matmul(x, self.R.to(x.dtype).T)
        elif self.rotation == "planar":
            from turboquant.planarquant import givens_forward
            pairs = x.reshape(*x.shape[:-1], self._planar.n_groups, 2)
            return givens_forward(pairs, self._planar.rot2).reshape(x.shape)
        else:  # iso
            blocks = x.reshape(*x.shape[:-1], self._iso.n_groups, 4)
            return self._iso._forward_rotate(blocks).reshape(x.shape)

    def _rotate_inverse(self, x_rot: torch.Tensor) -> torch.Tensor:
        """Inverse rotation. Used by query side in asymmetric IP."""
        if self.rotation == "full":
            return torch.matmul(x_rot, self.R.to(x_rot.dtype))
        elif self.rotation == "planar":
            from turboquant.planarquant import givens_inverse
            pairs = x_rot.reshape(*x_rot.shape[:-1], self._planar.n_groups, 2)
            return givens_inverse(pairs, self._planar.rot2).reshape(x_rot.shape)
        else:  # iso
            blocks = x_rot.reshape(*x_rot.shape[:-1], self._iso.n_groups, 4)
            return self._iso._inverse_rotate(blocks).reshape(x_rot.shape)

    # --------------------------------------------------------------------- #
    # Quantize / dequantize / IP estimator
    # --------------------------------------------------------------------- #
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            packed_signs: (..., dim/8) uint8
            norms:        (..., 1)    FP16 — the original ||x||
            x0:           (..., 1)    FP16 — alignment scalar mean(|x_rot|)
        """
        norms = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        safe = norms.clamp(min=1e-8)
        x_unit = x / safe

        x_rot = self._rotate_forward(x_unit)

        # x0 = mean of absolute values (the asymmetric-IP scaling factor)
        x0 = x_rot.abs().mean(dim=-1, keepdim=True)

        # signs ∈ {-1, +1} → binary {0, 1} for packing
        binary_signs = (x_rot >= 0).to(torch.int8)
        packed = pack_bits(binary_signs, 1)
        return packed, norms, x0

    def dequantize(self, packed: torch.Tensor, norms: torch.Tensor,
                   x0: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct x_hat = ||x|| · x0 · R⁻¹ · sign_vector. The x0 scalar makes
        the asymmetric inner product unbiased (in expectation).
        """
        binary_signs = unpack_bits(packed, 1, original_last_dim=self.dim)
        signs = binary_signs.to(norms.dtype) * 2.0 - 1.0   # {-1, +1}
        unrotated = self._rotate_inverse(signs)
        return norms * x0 * unrotated

    def estimate_inner_product_asymmetric(self,
                                            y: torch.Tensor,
                                            packed: torch.Tensor,
                                            norms: torch.Tensor,
                                            x0: torch.Tensor) -> torch.Tensor:
        """
        Asymmetric estimator: query y stays full-precision, key x is binary.

        Without correction:
            E[<R y, signs(R x)>] · x0 ≈ (2/π) · <y, x>     (BIASED)
        because E[(d · x0²)] = 2/π for unit vectors uniform on the sphere.

        Unbiased form (the (π/2) factor undoes the bias):
            <y, x>_est = ||x|| · (π/2) · x0 · <R y, signs(R x)>

        Equivalent to <y, dequantize(...)> · (π/2) — but avoids materializing
        x_hat (useful when y is one query and packed is many keys).
        """
        binary_signs = unpack_bits(packed, 1, original_last_dim=self.dim)
        signs = binary_signs.to(y.dtype) * 2.0 - 1.0
        y_rot = self._rotate_forward(y)
        raw_ip = (y_rot * signs).sum(dim=-1, keepdim=True)
        # Bias-correction factor: π/2. Don't omit.
        c = math.pi / 2.0
        return (norms * c * x0 * raw_ip).squeeze(-1)

    @property
    def bytes_per_vector(self) -> int:
        """d/8 sign bytes + 2 (norm) + 2 (x0)."""
        return self.dim // 8 + 4

    @property
    def compression_vs_fp16(self) -> float:
        return (self.dim * 2) / self.bytes_per_vector
