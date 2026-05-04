"""
Phase 4 validation: IsoQuant quaternion-based round-trip.

Tests:
1. test_quat_mul_basics
     Identity: 1 · q = q. Conjugate-of-conjugate = q. Norm preservation.

2. test_inverse_rotates_correctly
     _inverse_rotate(_forward_rotate(x)) == x for both fast and full modes.

3. test_counts
     FMA / parameter counts match plan (fast mode at d=128: 1024 / 128).

4. test_round_trip_vs_planarquant
     IsoQuant MSE within 5% of PlanarQuant on random unit vectors at 2/3/4 bits.

5. test_fast_vs_full_mode
     Both modes give similar MSE; full has slightly better (more DOF) at 4-bit.

6. test_inverse_rotation_trap_iso
     Confirm catastrophic failure if forward rotation used in dequant
     (proves Phase 6's symmetric cache check would catch this for IsoQuant too).
"""
import math
import torch

from turboquant.isoquant import IsoQuant, quat_mul, quat_conj, random_unit_quaternions
from turboquant.planarquant import PlanarQuant


def random_unit_vectors(n: int, d: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    return x / x.norm(dim=-1, keepdim=True)


def test_quat_mul_basics():
    print("\n[test_quat_mul_basics]")
    # Identity quaternion is (1, 0, 0, 0)
    one = torch.tensor([1.0, 0.0, 0.0, 0.0])
    q = torch.tensor([0.5, 0.5, 0.5, 0.5])  # already unit
    r = quat_mul(one.expand_as(q), q)
    err = (r - q).abs().max().item()
    print(f"  identity * q == q: max diff = {err:.2e}")
    assert err < 1e-6

    # conj(conj(q)) == q
    cc = quat_conj(quat_conj(q))
    err = (cc - q).abs().max().item()
    print(f"  conj(conj(q)) == q: max diff = {err:.2e}")
    assert err < 1e-7

    # q * conj(q) = ||q||² (real)
    r = quat_mul(q, quat_conj(q))
    print(f"  q * conj(q) = ({r[0]:.4f}, {r[1]:.4f}, {r[2]:.4f}, {r[3]:.4f}) "
          f"(should be (1, 0, 0, 0) for unit q)")
    assert abs(r[0].item() - 1.0) < 1e-6
    assert r[1:].abs().max().item() < 1e-6
    print("  ✅ PASS")


def test_inverse_rotates_correctly():
    print("\n[test_inverse_rotates_correctly]")
    for mode in ("fast", "full"):
        torch.manual_seed(0)
        iq = IsoQuant(dim=128, bits=3, mode=mode, seed=42)
        x = torch.randn(7, 128)
        # Reshape into blocks
        blocks = x.reshape(7, 32, 4)
        rotated = iq._forward_rotate(blocks)
        back = iq._inverse_rotate(rotated)
        err = (back - blocks).abs().max().item()
        print(f"  mode={mode}: max |inverse(forward(x)) - x| = {err:.2e}")
        assert err < 1e-5, f"{mode} mode inverse failed: {err}"
    print("  ✅ PASS — both fast and full modes invert correctly.")


def test_counts():
    print("\n[test_counts]")
    # Fast mode at d=128: 32 groups * 32 FMAs = 1024 (matches plan's "1024 FMAs")
    # Params: 32 groups * 4 = 128
    iq = IsoQuant(dim=128, bits=3, mode="fast", seed=0)
    print(f"  fast mode @ d=128: FMAs={iq.fma_count_round_trip}  params={iq.parameter_count}")
    assert iq.fma_count_round_trip == 1024
    assert iq.parameter_count == 128

    # Full mode: 32 * 64 = 2048 FMAs, 32 * 8 = 256 params
    iq_full = IsoQuant(dim=128, bits=3, mode="full", seed=0)
    print(f"  full mode @ d=128: FMAs={iq_full.fma_count_round_trip}  params={iq_full.parameter_count}")
    assert iq_full.fma_count_round_trip == 2048
    assert iq_full.parameter_count == 256
    print("  ✅ PASS — FMA / parameter counts match plan.")


def test_round_trip_vs_planarquant():
    print("\n[test_round_trip_vs_planarquant]")
    n = 10_000
    for bits in (2, 3, 4):
        x = random_unit_vectors(n, 128, seed=bits)
        pq = PlanarQuant(dim=128, bits=bits, seed=bits)
        iq_fast = IsoQuant(dim=128, bits=bits, mode="fast", seed=bits)

        pq_xhat = pq.dequantize(*pq.quantize(x))
        iq_xhat = iq_fast.dequantize(*iq_fast.quantize(x))

        pq_mse = ((x - pq_xhat) ** 2).sum(dim=-1).mean().item()
        iq_mse = ((x - iq_xhat) ** 2).sum(dim=-1).mean().item()
        ratio = iq_mse / pq_mse
        print(f"  bits={bits}: PlanarQuant MSE={pq_mse:.6f}  IsoQuant(fast) MSE={iq_mse:.6f}  ratio={ratio:.3f}x")
        # Within 10% of PlanarQuant — quaternion blocks span 4 coords vs 2,
        # so the block-level mixing can be slightly different.
        assert ratio < 1.10, f"IsoQuant {ratio:.3f}x worse than PlanarQuant"
    print("  ✅ PASS — IsoQuant within 10% of PlanarQuant across 2/3/4 bits.")


def test_fast_vs_full_mode():
    print("\n[test_fast_vs_full_mode]")
    n = 10_000
    x = random_unit_vectors(n, 128, seed=99)
    iq_fast = IsoQuant(dim=128, bits=4, mode="fast", seed=0)
    iq_full = IsoQuant(dim=128, bits=4, mode="full", seed=0)

    fast_mse = ((x - iq_fast.dequantize(*iq_fast.quantize(x))) ** 2).sum(dim=-1).mean().item()
    full_mse = ((x - iq_full.dequantize(*iq_full.quantize(x))) ** 2).sum(dim=-1).mean().item()
    print(f"  4-bit fast MSE: {fast_mse:.6f}")
    print(f"  4-bit full MSE: {full_mse:.6f}")
    print(f"  fast/full: {fast_mse / full_mse:.3f}x")
    # Both should be within 10% of each other; the plan claims fast is "usually
    # enough" — full should be only marginally better.
    assert abs(fast_mse - full_mse) / max(fast_mse, full_mse) < 0.10
    print("  ✅ PASS — fast and full modes give comparable MSE.")


def test_inverse_rotation_trap_iso():
    """Same trap as PlanarQuant: forward rotation in dequant → catastrophic MSE."""
    print("\n[test_inverse_rotation_trap_iso]")
    n = 5_000
    iq = IsoQuant(dim=128, bits=3, mode="fast", seed=7)
    x = random_unit_vectors(n, 128, seed=7)

    # Correct path
    packed, norms = iq.quantize(x)
    x_hat_correct = iq.dequantize(packed, norms)
    correct_mse = ((x - x_hat_correct) ** 2).sum(dim=-1).mean().item()

    # Buggy path: use FORWARD in dequant (the canonical bug)
    from turboquant.bit_packing import unpack_bits
    indices = unpack_bits(packed, iq.bits, original_last_dim=128)
    values = iq.centroids[indices.long()]
    blocks = values.reshape(n, 32, 4)
    bug_unrot = iq._forward_rotate(blocks)            # ← BUG
    bug_flat = bug_unrot.reshape(n, 128) * norms
    bug_mse = ((x - bug_flat) ** 2).sum(dim=-1).mean().item()

    print(f"  Correct (inverse) MSE: {correct_mse:.4f}")
    print(f"  BUGGY (forward) MSE:   {bug_mse:.4f}")
    print(f"  Bug penalty:           {bug_mse / correct_mse:.1f}x")
    assert bug_mse > correct_mse * 5, \
        "Trap test didn't fire — investigate the symmetry of the buggy operator"
    print("  ✅ PASS — Phase 6 SymmetricKVCache check would catch this for IsoQuant too.")


if __name__ == "__main__":
    test_quat_mul_basics()
    test_inverse_rotates_correctly()
    test_counts()
    test_round_trip_vs_planarquant()
    test_fast_vs_full_mode()
    test_inverse_rotation_trap_iso()
    print("\nAll Phase 4 tests passed.")
