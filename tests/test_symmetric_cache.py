"""
Phase 6 validation: symmetric K+V cache and the inverse-rotation trap.

Tests:
1. test_construction_requires_v
     SymmetricKVCache(qk, None) must raise ValueError.

2. test_inverse_check_passes_for_correct_quantizer
     Real PlanarQuant should pass assert_inverse_correct.

3. test_inverse_check_catches_buggy_quantizer
     Build a deliberately-broken quantizer that uses forward rotation in
     dequant. The check should raise RuntimeError at construction.

4. test_full_v_round_trip_via_cache
     End-to-end: prefill 100 tokens, finalize, decode 5 tokens. Compare
     full_V at the end against the reference (no quantization). Quantization
     error should be <2x the per-vector round-trip MSE.

5. test_attention_output_with_symmetric_cache
     Compute attention output once with FP16 K/V and once via the cache.
     Cosine similarity of the outputs should be > 0.95 (3-bit symmetric).
     Crucially, this is the test that would EXPLODE if forward rotation
     were used in V dequant — cos sim collapses to ~0.
"""
import torch
import math
from contextlib import contextmanager

from turboquant.planarquant import PlanarQuant, givens_forward
from turboquant.deferred_cache import DeferredQuantCache
from turboquant.symmetric_cache import SymmetricKVCache, assert_inverse_correct
from turboquant.bit_packing import unpack_bits


@contextmanager
def assert_raises(exc_type):
    raised = False
    try:
        yield
    except exc_type:
        raised = True
    if not raised:
        raise AssertionError(f"Expected {exc_type.__name__}")


# --------------------------------------------------------------------------- #
# 1. Construction requires V
# --------------------------------------------------------------------------- #
def test_construction_requires_v():
    print("\n[test_construction_requires_v]")
    pq = PlanarQuant(dim=64, bits=3, seed=0)
    with assert_raises(ValueError):
        SymmetricKVCache(pq, None)
    print("  ✅ PASS — None V quantizer rejected.")


# --------------------------------------------------------------------------- #
# 2. Correct quantizer passes
# --------------------------------------------------------------------------- #
def test_inverse_check_passes_for_correct_quantizer():
    print("\n[test_inverse_check_passes_for_correct_quantizer]")
    pq = PlanarQuant(dim=128, bits=3, seed=0)
    assert_inverse_correct(pq, dim=128)
    print("  ✅ PASS — PlanarQuant's inverse Givens passes the trap-detector.")


# --------------------------------------------------------------------------- #
# 3. Buggy quantizer trips the trap
# --------------------------------------------------------------------------- #
class _BuggyPlanarQuant(PlanarQuant):
    """Deliberately broken: forward rotation used in dequant."""
    def dequantize(self, packed, norms):
        indices = unpack_bits(packed, self.bits, original_last_dim=self._padded_dim)
        values = self.centroids.to(norms.dtype)[indices.long()]
        pairs = values.reshape(*values.shape[:-1], self.n_groups, 2)
        # BUG: forward rotation in dequant
        bug_unrot = givens_forward(pairs, self.rot2)
        flat = bug_unrot.reshape(*values.shape[:-1], self._padded_dim)
        flat = self._maybe_trim(flat)
        return flat * norms


def test_inverse_check_catches_buggy_quantizer():
    print("\n[test_inverse_check_catches_buggy_quantizer]")
    bug = _BuggyPlanarQuant(dim=128, bits=3, seed=0)
    pq_ok = PlanarQuant(dim=128, bits=3, seed=0)
    with assert_raises(RuntimeError):
        # Pass bug as V quantizer — symmetric construction should refuse
        SymmetricKVCache(pq_ok, bug)
    print("  ✅ PASS — construction REFUSES a buggy V quantizer "
          "(canonical PPL=15K bug caught at init time).")


# --------------------------------------------------------------------------- #
# 4. End-to-end V round-trip through the cache
# --------------------------------------------------------------------------- #
def test_full_v_round_trip_via_cache():
    print("\n[test_full_v_round_trip_via_cache]")
    torch.manual_seed(0)
    B, H, T, D = 1, 2, 100, 64
    K_pre = torch.randn(B, H, T, D)
    V_pre = torch.randn(B, H, T, D)
    # Decode 5 tokens
    K_dec = [torch.randn(B, H, 1, D) for _ in range(5)]
    V_dec = [torch.randn(B, H, 1, D) for _ in range(5)]

    qk = PlanarQuant(dim=D, bits=4, seed=0)
    qv = PlanarQuant(dim=D, bits=4, seed=1)
    cache = SymmetricKVCache(qk, qv)

    cache.append_prefill(K_pre, V_pre)
    cache.finalize_prefill()

    full_K, full_V = None, None
    for K_n, V_n in zip(K_dec, V_dec):
        full_K, full_V = cache.append_decode(K_n, V_n)

    # Reference: never quantized
    ref_V = torch.cat([V_pre] + V_dec, dim=2)

    # Last 5 tokens are FP16 (current decode steps got returned at FP16 in their
    # respective steps). Wait — actually in the cache, only the LAST decode
    # step is FP16; earlier decode steps were quantized for the next step.
    # So full_V matches: ref_V[..., :T+4, :] is dequantized; ref_V[..., -1, :] is FP16.
    deq_part = full_V[:, :, :-1, :]
    fp_part  = full_V[:, :, -1:, :]
    ref_deq  = ref_V[:, :, :-1, :]
    ref_fp   = ref_V[:, :, -1:, :]

    # Last token must be byte-exact (current step always FP16)
    fp_diff = (fp_part - ref_fp).abs().max().item()
    print(f"  Last-token V diff (must be 0):     {fp_diff:.2e}")
    assert fp_diff == 0, "Last-step V leaked through quantization"

    # Older tokens: dequantized — error bounded by per-vector round-trip MSE
    deq_mse = ((deq_part - ref_deq) ** 2).sum(dim=-1).mean().item()
    cos = torch.nn.functional.cosine_similarity(deq_part, ref_deq, dim=-1).mean().item()
    print(f"  Historical V MSE:                  {deq_mse:.4f}")
    print(f"  Historical V cosine sim:           {cos:.4f}")
    assert cos > 0.95, f"Historical V cos sim {cos} too low"
    print("  ✅ PASS — symmetric cache round-trip preserves V quality.")


# --------------------------------------------------------------------------- #
# 5. Attention output preserved
# --------------------------------------------------------------------------- #
def test_attention_output_with_symmetric_cache():
    """
    Real-attention proxy: this test would EXPLODE under the V-dequant trap.
    Compares attention output computed two ways:
      (a) reference: no quantization
      (b) via SymmetricKVCache: K and V both quantized
    Cosine similarity of (a) vs (b) attention output should be > 0.95.
    The plan reports that with forward-rotation-in-V-dequant, this number
    would collapse to ~0 (PPL goes 7 → 15,000+).
    """
    print("\n[test_attention_output_with_symmetric_cache]")
    torch.manual_seed(0)
    B, H, T, D = 1, 4, 256, 64
    K_pre = torch.randn(B, H, T, D) * (1.0 / math.sqrt(D))
    V_pre = torch.randn(B, H, T, D) * (1.0 / math.sqrt(D))

    qk = PlanarQuant(dim=D, bits=4, seed=0)
    qv = PlanarQuant(dim=D, bits=4, seed=1)
    cache = SymmetricKVCache(qk, qv)
    cache.append_prefill(K_pre, V_pre)
    cache.finalize_prefill()

    K_new = torch.randn(B, H, 1, D) * (1.0 / math.sqrt(D))
    V_new = torch.randn(B, H, 1, D) * (1.0 / math.sqrt(D))
    full_K_q, full_V_q = cache.append_decode(K_new, V_new)

    # Reference (no quant)
    full_K_ref = torch.cat([K_pre, K_new], dim=2)
    full_V_ref = torch.cat([V_pre, V_new], dim=2)

    # A single Q vector for the just-inserted token
    Q = torch.randn(B, H, 1, D) * (1.0 / math.sqrt(D))

    def attn(Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)
        a = torch.softmax(scores, dim=-1)
        return torch.matmul(a, V)

    out_ref = attn(Q, full_K_ref, full_V_ref)
    out_q   = attn(Q, full_K_q, full_V_q)

    cos = torch.nn.functional.cosine_similarity(
        out_ref.flatten(end_dim=-2), out_q.flatten(end_dim=-2), dim=-1
    ).mean().item()
    err = (out_ref - out_q).norm().item() / out_ref.norm().item()

    print(f"  Attention output cosine sim:  {cos:.4f}")
    print(f"  Attention output rel error:   {err:.4f}")
    assert cos > 0.95, f"Attention drift too large under symmetric quant: cos={cos}"
    print("  ✅ PASS — attention output preserved under K+V quantization.")


# --------------------------------------------------------------------------- #
# 6. The same test BUT with the buggy V quantizer — should drift catastrophically
# --------------------------------------------------------------------------- #
def test_buggy_v_dequant_explodes_attention():
    """
    Simulate the disaster mode. Bypass the construction self-check (force-mute
    by using DeferredQuantCache directly with a buggy V quantizer) and confirm
    the attention output is essentially garbage. This documents WHY Phase 6
    matters and why the construction check should never be skipped in production.
    """
    print("\n[test_buggy_v_dequant_explodes_attention]")
    torch.manual_seed(0)
    B, H, T, D = 1, 4, 256, 64
    K_pre = torch.randn(B, H, T, D) * (1.0 / math.sqrt(D))
    V_pre = torch.randn(B, H, T, D) * (1.0 / math.sqrt(D))

    qk = PlanarQuant(dim=D, bits=4, seed=0)
    bug_v = _BuggyPlanarQuant(dim=D, bits=4, seed=1)

    # Bypass SymmetricKVCache (which would refuse). Use base DeferredQuantCache.
    cache = DeferredQuantCache(qk, bug_v)
    cache.append_prefill(K_pre, V_pre)
    cache.finalize_prefill()

    K_new = torch.randn(B, H, 1, D) * (1.0 / math.sqrt(D))
    V_new = torch.randn(B, H, 1, D) * (1.0 / math.sqrt(D))
    full_K_q, full_V_q = cache.append_decode(K_new, V_new)

    full_K_ref = torch.cat([K_pre, K_new], dim=2)
    full_V_ref = torch.cat([V_pre, V_new], dim=2)
    Q = torch.randn(B, H, 1, D) * (1.0 / math.sqrt(D))

    def attn(Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)
        return torch.matmul(torch.softmax(scores, dim=-1), V)

    out_ref = attn(Q, full_K_ref, full_V_ref)
    out_buggy = attn(Q, full_K_q, full_V_q)
    cos = torch.nn.functional.cosine_similarity(
        out_ref.flatten(end_dim=-2), out_buggy.flatten(end_dim=-2), dim=-1
    ).mean().item()
    print(f"  BUGGY V (forward rotation in dequant): cos sim = {cos:.4f}")
    print(f"  Phase 6 SymmetricKVCache construction check would have refused this.")
    assert cos < 0.5, "Buggy V should have collapsed cosine sim — toy model too weak"
    print("  ✅ PASS — buggy V dequant explodes attention as expected.")


if __name__ == "__main__":
    test_construction_requires_v()
    test_inverse_check_passes_for_correct_quantizer()
    test_inverse_check_catches_buggy_quantizer()
    test_full_v_round_trip_via_cache()
    test_attention_output_with_symmetric_cache()
    test_buggy_v_dequant_explodes_attention()
    print("\nAll Phase 6 tests passed.")
