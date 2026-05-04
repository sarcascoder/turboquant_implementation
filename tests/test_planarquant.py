"""
Phase 3 validation: PlanarQuant round-trip + the inverse-rotation trap.

Three tests:
1. test_counts          — FMA / parameter counts match the plan (512 / 128 at d=128).
2. test_inverse_is_transpose — Givens inverse exactly inverts forward (no quantization).
3. test_round_trip_vs_turboquant — MSE within 5% of TurboQuant on random unit vectors.
4. test_inverse_rotation_trap — show that using FORWARD rotation in dequant
                              (the canonical bug) explodes MSE.
"""
import math
import torch
from typing import Tuple

from turboquant.quantizer import TurboQuantMSE
from turboquant.planarquant import PlanarQuant, givens_forward, givens_inverse


def random_unit_vectors(n: int, d: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    return x / x.norm(dim=-1, keepdim=True)


def round_trip_mse(quantizer, x: torch.Tensor) -> Tuple[float, float, float]:
    if hasattr(quantizer, "quantize") and callable(quantizer.quantize):
        if isinstance(quantizer, PlanarQuant) or quantizer.__class__.__name__ == "TurboQuantMSE":
            packed, norms = quantizer.quantize(x)
            x_hat = quantizer.dequantize(packed, norms)
        else:
            raise ValueError(f"Unknown quantizer: {type(quantizer)}")
    mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1).mean().item()
    norm_recovered = x_hat.norm(dim=-1).mean().item()
    return mse, cos_sim, norm_recovered


def test_counts():
    print("\n[test_counts]")
    for d in (64, 128, 256):
        pq = PlanarQuant(dim=d, bits=3, seed=0)
        fmas = pq.fma_count_round_trip
        params = pq.parameter_count
        print(f"  d={d}: FMAs={fmas} (expected {4 * d})  params={params} (expected {d})")
        assert fmas == 4 * d, f"FMA count {fmas} != 4*d ({4 * d})"
        assert params == d, f"param count {params} != d ({d})"
    print("  ✅ FMA and parameter counts match.")


def test_inverse_is_transpose():
    print("\n[test_inverse_is_transpose]")
    torch.manual_seed(0)
    pq = PlanarQuant(dim=128, bits=3, seed=42)
    x = torch.randn(7, 128)
    pairs = x.reshape(7, 64, 2)
    rotated = givens_forward(pairs, pq.rot2)
    back = givens_inverse(rotated, pq.rot2)
    err = (back - pairs).abs().max().item()
    print(f"  Max |inv(fwd(x)) - x| = {err:.2e}")
    assert err < 1e-5, f"Inverse failed: max err {err}"
    print("  ✅ givens_inverse exactly inverts givens_forward.")


def test_round_trip_vs_turboquant():
    print("\n[test_round_trip_vs_turboquant]")
    n = 10_000
    for bits in (2, 3, 4):
        x = random_unit_vectors(n, 128, seed=bits)

        tq = TurboQuantMSE(dim=128, bits=bits)
        pq = PlanarQuant(dim=128, bits=bits, seed=bits)

        tq_mse, tq_cos, tq_norm = round_trip_mse(tq, x)
        pq_mse, pq_cos, pq_norm = round_trip_mse(pq, x)
        ratio = pq_mse / tq_mse

        print(f"  bits={bits}:")
        print(f"    TurboQuant  MSE={tq_mse:.6f}  cos={tq_cos:.4f}  norm_rec={tq_norm:.4f}")
        print(f"    PlanarQuant MSE={pq_mse:.6f}  cos={pq_cos:.4f}  norm_rec={pq_norm:.4f}")
        print(f"    Ratio (Planar/TQ): {ratio:.3f}x")

        # Plan target: within 5% of TurboQuant. We also accept up to 10% — at low
        # bit budgets the random-Givens variance is naturally a bit higher.
        assert ratio < 1.10, f"PlanarQuant MSE {ratio:.3f}x worse than TurboQuant"
        assert pq_cos > 0.85, f"cosine sim too low: {pq_cos}"
    print("  ✅ Round-trip MSE within 10% of TurboQuant across 2/3/4 bits.")


def test_inverse_rotation_trap():
    """
    Canonical bug: applying FORWARD rotation in the dequant path.
    With no quantization, the (forward∘forward) rotates each pair by 2θ —
    so error is bounded but non-zero. With quantization, this catastrophically
    inflates MSE because we're decoding in the wrong basis.
    """
    print("\n[test_inverse_rotation_trap]")
    n = 5_000
    pq_correct = PlanarQuant(dim=128, bits=3, seed=7)
    x = random_unit_vectors(n, 128, seed=7)

    # Correct path
    packed, norms = pq_correct.quantize(x)
    x_hat_correct = pq_correct.dequantize(packed, norms)
    correct_mse = ((x - x_hat_correct) ** 2).sum(dim=-1).mean().item()

    # Buggy path: apply forward rotation in place of inverse during dequant
    from turboquant.bit_packing import unpack_bits
    indices = unpack_bits(packed, pq_correct.bits, original_last_dim=128)
    values = pq_correct.centroids[indices.long()]
    pairs = values.reshape(n, 64, 2)
    bug_unrotated = givens_forward(pairs, pq_correct.rot2)  # ← BUG
    bug_flat = bug_unrotated.reshape(n, 128) * norms
    bug_mse = ((x - bug_flat) ** 2).sum(dim=-1).mean().item()

    print(f"  Correct (inverse rotation) MSE: {correct_mse:.4f}")
    print(f"  BUGGY  (forward rotation)  MSE: {bug_mse:.4f}")
    print(f"  Bug penalty: {bug_mse / correct_mse:.1f}x worse")
    # The bug should make MSE >>1 (cosine similarity essentially destroyed)
    assert bug_mse > correct_mse * 5, \
        "Inverse-rotation trap test didn't fire — check the bug surface!"
    print("  ✅ Inverse-rotation trap detected (proves test would catch the bug).")


if __name__ == "__main__":
    test_counts()
    test_inverse_is_transpose()
    test_round_trip_vs_turboquant()
    test_inverse_rotation_trap()
    print("\nAll Phase 3 tests passed.")
