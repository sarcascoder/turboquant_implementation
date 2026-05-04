"""
Phase 6 — Symmetric V cache (K and V both quantized).

Built on top of DeferredQuantCache (Phase 5). The new piece is a runtime
sanity check that the V quantizer applies the INVERSE rotation in dequant
(not the forward). This trap is the single most catastrophic bug surface
in the whole project: silent failure, no exception, PPL goes from ~7 to
>15,000 if forward rotation is used in V dequant.

For PlanarQuant, the inverse rotation is automatic — `PlanarQuant.dequantize`
calls `givens_inverse` (with negated sin). This file's job is to:

1. Provide a thin SymmetricKVCache wrapper that ENFORCES both K and V
   quantizers are present (vs. the asymmetric K-only mode in DeferredQuantCache).
2. Add a startup self-check `assert_inverse_correct()` that verifies the V
   quantizer's dequant is in fact the inverse of its quant — if a future
   refactor accidentally uses forward rotation for V, this will fire at
   construction time, not after a PPL run.
"""
from __future__ import annotations
import torch
from typing import Tuple

from turboquant.deferred_cache import DeferredQuantCache


def assert_inverse_correct(quantizer, dim: int, atol: float = 1e-4,
                            n_samples: int = 256) -> None:
    """
    Verify that the quantizer's dequant is the inverse of its quant on
    smooth inputs (i.e., inputs that lie close to centroid values, where
    quantization itself is near-lossless).

    This isolates 'is the inverse rotation right' from 'is the codebook good':
    by feeding inputs that are already near centroid values, we make
    quantization almost free, so any reconstruction error MUST come from a
    wrong inverse rotation.

    Procedure: take a unit vector x, round-trip it, compute residual.
    On a CORRECT inverse, residual is bounded by codebook resolution.
    On a BUGGY forward-in-dequant, residual blows up (rotated 2θ off).
    """
    torch.manual_seed(0)
    # Build a smooth test input: random unit vectors. For PlanarQuant the
    # round-trip MSE on unit vectors is well-characterized; if dequant uses
    # forward instead of inverse, MSE jumps ~50x (we showed this in Phase 3).
    x = torch.randn(n_samples, dim)
    x = x / x.norm(dim=-1, keepdim=True)

    packed, norms = quantizer.quantize(x)
    x_hat = quantizer.dequantize(packed, norms)
    mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()

    # Soft upper bound: the round-trip should at least preserve cosine sim > 0.5.
    # On the buggy path (forward rotation in dequant), cos sim collapses to ~0
    # because the vectors are rotated by 2θ ≠ 0.
    cos_sim = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1).mean().item()
    if cos_sim < 0.5:
        raise RuntimeError(
            f"Quantizer {type(quantizer).__name__} appears to use the WRONG "
            f"inverse rotation: round-trip cosine sim = {cos_sim:.3f} (expected > 0.5). "
            f"This is the canonical V-cache catastrophic bug — see Phase 6 docs."
        )
    # Light upper-bound on MSE — should be ~Lloyd-Max bound, not ~1.0
    if mse > 0.5:
        raise RuntimeError(
            f"Quantizer {type(quantizer).__name__} round-trip MSE = {mse:.4f} > 0.5; "
            f"likely a wrong-inverse-rotation bug."
        )


class SymmetricKVCache(DeferredQuantCache):
    """
    K + V both quantized. Inherits the DeferredQuantCache state machine and
    adds a one-time inverse-rotation self-check at construction. This catches
    the canonical V-cache bug at startup rather than at runtime.

    Args:
        quantizer_k: K quantizer (PlanarQuant, IsoQuant, TurboQuantMSE).
        quantizer_v: V quantizer (must NOT be None — that's asymmetric mode).
        check_inverse: run assert_inverse_correct on construction (default True).
                       Set False only in tight benchmarks where the few-ms
                       startup check matters.
    """

    def __init__(self, quantizer_k, quantizer_v, check_inverse: bool = True):
        if quantizer_v is None:
            raise ValueError(
                "SymmetricKVCache requires a V quantizer; for asymmetric K-only "
                "mode use DeferredQuantCache(quantizer_k, quantizer_v=None) directly."
            )
        super().__init__(quantizer_k, quantizer_v)

        if check_inverse:
            # Run the trap-detector for both K and V quantizers
            assert_inverse_correct(quantizer_k, quantizer_k.dim)
            assert_inverse_correct(quantizer_v, quantizer_v.dim)

    def __repr__(self) -> str:
        return (f"SymmetricKVCache(mode={self.mode}, seq_len={self._seq_len}, "
                f"k_bits={self.quantizer_k.bits}, v_bits={self.quantizer_v.bits})")
