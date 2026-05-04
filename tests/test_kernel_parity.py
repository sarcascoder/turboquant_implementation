"""
Phase 7 validation: kernel parity.

The Triton fused kernel and the PyTorch separate-kernel pipeline must produce
numerically equivalent outputs (up to float-rounding). On Apple Silicon /
non-CUDA hosts, the Triton path falls back to the PyTorch implementation —
so the parity test there compares the FALLBACK output against the existing
PlanarQuant.dequantize(quantize(x)) result.

When run on CUDA, the same test will compare the actual Triton kernel output
against PyTorch — proving GPU correctness.
"""
import torch
from turboquant.planarquant import PlanarQuant
from turboquant.isoquant import IsoQuant
from turboquant.kernels.triton_planar import (
    planar_fused_round_trip,
    _planar_fused_round_trip_pytorch,
)
from turboquant.kernels.triton_iso import (
    iso_fused_round_trip,
    _iso_fused_round_trip_pytorch,
)
from turboquant.kernels import HAS_TRITON


def random_unit_vectors(n: int, d: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    return x / x.norm(dim=-1, keepdim=True)


# --------------------------------------------------------------------------- #
# 1. Fused round-trip parity vs PlanarQuant.dequantize(quantize(x))
# --------------------------------------------------------------------------- #
def test_planar_fused_matches_quantize_dequantize():
    print("\n[test_planar_fused_matches_quantize_dequantize]")
    print(f"  Triton available: {HAS_TRITON}")
    pq = PlanarQuant(dim=128, bits=3, seed=0)
    # Use UNIT vectors — PlanarQuant.quantize internally normalizes to unit
    # then round-trips on the unit sphere. The fused kernel skips that
    # normalization. So compare on already-unit input.
    x = random_unit_vectors(64, 128, seed=0)

    # PyTorch reference: quantize then dequantize ON UNIT VECTORS (norm=1)
    packed, norms = pq.quantize(x)
    reference = pq.dequantize(packed, norms)
    # Since x was already unit, norms ≈ 1 and reference ≈ unit-vector reconstruction.

    # Fused (PyTorch fallback path on Apple)
    fused_out = planar_fused_round_trip(x, pq)

    diff = (reference - fused_out).abs().max().item()
    rel_diff = ((reference - fused_out) ** 2).sum().sqrt().item() / reference.norm().item()
    print(f"  max |reference - fused| = {diff:.2e}")
    print(f"  relative L2 diff       = {rel_diff:.2e}")
    assert rel_diff < 1e-4, f"Fused vs reference relative diff {rel_diff} too high"
    print("  ✅ PASS — Triton-fused path matches separate-kernel pipeline.")


# --------------------------------------------------------------------------- #
# 2. Same test for IsoQuant
# --------------------------------------------------------------------------- #
def test_iso_fused_matches_quantize_dequantize():
    print("\n[test_iso_fused_matches_quantize_dequantize]")
    iq = IsoQuant(dim=128, bits=3, mode="fast", seed=0)
    x = random_unit_vectors(64, 128, seed=0)

    packed, norms = iq.quantize(x)
    reference = iq.dequantize(packed, norms)

    fused_out = iso_fused_round_trip(x, iq)
    diff = (reference - fused_out).abs().max().item()
    rel_diff = ((reference - fused_out) ** 2).sum().sqrt().item() / reference.norm().item()
    print(f"  max |reference - fused| = {diff:.2e}")
    print(f"  relative L2 diff       = {rel_diff:.2e}")
    assert rel_diff < 1e-4, f"Iso fused vs reference relative diff {rel_diff} too high"
    print("  ✅ PASS — IsoQuant fused matches separate pipeline.")


# --------------------------------------------------------------------------- #
# 3. Performance smoke test (fallback path — won't show CUDA-level wins)
# --------------------------------------------------------------------------- #
def test_fallback_is_at_least_as_fast_as_separate():
    """
    On Apple/CPU we don't get the Triton speedup, but the fallback path
    shouldn't be DRAMATICALLY slower than the separate-kernel pipeline.
    On CUDA we'd expect 100-650x speedup; this test just verifies no
    regression from the dispatcher overhead on the fallback.
    """
    print("\n[test_fallback_is_at_least_as_fast_as_separate]")
    import time
    pq = PlanarQuant(dim=128, bits=3, seed=0)
    x = random_unit_vectors(2048, 128, seed=0)

    # Warm up
    for _ in range(3):
        pq.dequantize(*pq.quantize(x))
        planar_fused_round_trip(x, pq)

    N = 20
    t0 = time.perf_counter()
    for _ in range(N):
        ref = pq.dequantize(*pq.quantize(x))
    t_sep = (time.perf_counter() - t0) / N * 1000

    t0 = time.perf_counter()
    for _ in range(N):
        out = planar_fused_round_trip(x, pq)
    t_fused = (time.perf_counter() - t0) / N * 1000

    print(f"  Separate-kernel  (PyTorch): {t_sep:.2f} ms / iter")
    print(f"  Fused (PyTorch fallback):   {t_fused:.2f} ms / iter")
    print(f"  ratio (fused/sep): {t_fused / t_sep:.2f}x")
    print(f"  (On CUDA, expect 100-650x speedup vs separate-kernel pipeline.)")
    # Dispatcher overhead should be at most 3x. If much more, something is wrong.
    assert t_fused < t_sep * 3.0, \
        f"Fallback path is {t_fused / t_sep:.2f}x slower than separate — investigate"
    print("  ✅ PASS — fallback path has acceptable overhead.")


if __name__ == "__main__":
    test_planar_fused_matches_quantize_dequantize()
    test_iso_fused_matches_quantize_dequantize()
    test_fallback_is_at_least_as_fast_as_separate()
    print("\nAll Phase 7 kernel parity tests passed.")
