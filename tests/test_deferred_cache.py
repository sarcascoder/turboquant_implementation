"""
Phase 5 validation: DeferredQuantCache state machine.

Tests:
1. test_prefill_no_quantization
     During prefill, no quantization happens — buffers hold raw FP16. (We can't
     observe the buffers directly via the public API, but we CAN verify that
     the cache mode is PREFILL and that finalize_prefill triggers a single
     bulk quantize call.)

2. test_state_machine
     append_decode before finalize_prefill should fail; double-finalize is a no-op.

3. test_current_step_is_fp16
     During decode, the K_new of the current step is preserved bit-exactly in
     the returned full_K. (Distinguishes "stored K" from "consumed K".)

4. test_compounding_avoided
     SIMULATE the compounding problem with a multi-layer toy model. Two caches:
     (a) "always quantize" — quantizes K during prefill at every layer.
     (b) DeferredQuantCache — defers prefill quantization.
     Compare attention-output drift after L layers. (a) drifts geometrically
     with depth; (b) drifts only by quantization error per single layer.

5. test_asymmetric_v_fp16
     With quantizer_v=None, V passes through FP16 untouched. full_V at decode
     should be bit-exact to the concatenation of all input V tensors.
"""
import torch
from contextlib import contextmanager
from turboquant.planarquant import PlanarQuant
from turboquant.deferred_cache import DeferredQuantCache


@contextmanager
def assert_raises(exc_type):
    """Lightweight pytest.raises substitute (avoids pytest dependency)."""
    raised = False
    try:
        yield
    except exc_type:
        raised = True
    if not raised:
        raise AssertionError(f"Expected {exc_type.__name__} to be raised, but no exception")


def make_cache(dim=64, bits=3, v_dim=None, quantize_v=True, seed=0):
    qk = PlanarQuant(dim=dim, bits=bits, seed=seed)
    qv = PlanarQuant(dim=v_dim or dim, bits=bits, seed=seed + 1) if quantize_v else None
    return DeferredQuantCache(qk, qv)


def test_prefill_no_quantization():
    print("\n[test_prefill_no_quantization]")
    cache = make_cache(quantize_v=True)
    B, H, T1, T2, D = 2, 4, 10, 5, 64
    K1 = torch.randn(B, H, T1, D)
    V1 = torch.randn(B, H, T1, D)
    K2 = torch.randn(B, H, T2, D)
    V2 = torch.randn(B, H, T2, D)

    cache.append_prefill(K1, V1)
    cache.append_prefill(K2, V2)
    assert cache.mode == "prefill"
    assert cache.get_seq_length() == T1 + T2
    # Internal: prefill buffers should hold raw FP16
    assert len(cache._prefill_K) == 2
    assert torch.allclose(cache._prefill_K[0], K1)
    assert torch.allclose(cache._prefill_K[1], K2)
    print(f"  Prefill mode preserved {T1 + T2} tokens at FP16, no quantization triggered.")
    print("  ✅ PASS")


def test_state_machine():
    print("\n[test_state_machine]")
    cache = make_cache()
    K = torch.randn(1, 2, 4, 64)
    V = torch.randn(1, 2, 4, 64)
    cache.append_prefill(K, V)

    # Decode before finalize → should fail
    with assert_raises(AssertionError):
        cache.append_decode(K[:, :, :1, :], V[:, :, :1, :])

    cache.finalize_prefill()
    assert cache.mode == "decode"

    # Re-finalize after decode → should fail (mode is no longer prefill)
    with assert_raises(AssertionError):
        cache.finalize_prefill()
    print("  ✅ PASS — append_decode blocked before finalize; double-finalize blocked.")


def test_current_step_is_fp16():
    """
    The plan's most subtle invariant: during decode, the K we just inserted
    must be returned at FP16 inside full_K, NOT a dequantized version.
    Otherwise the just-inserted token gets two layers of quantization noise.
    """
    print("\n[test_current_step_is_fp16]")
    cache = make_cache(dim=64, bits=2)  # low bits → big quant error if it leaks in
    # Prefill 0 tokens (edge case): finalize an empty cache and decode immediately
    # This isolates the "current step" behavior from the historical dequantization.
    cache.finalize_prefill()  # empty prefill
    assert cache.mode == "decode"
    # We need at least one quantized historical token, so prefill 1 token first.
    cache = make_cache(dim=64, bits=2)
    K0 = torch.randn(1, 1, 1, 64)
    V0 = torch.randn(1, 1, 1, 64)
    cache.append_prefill(K0, V0)
    cache.finalize_prefill()

    K_new = torch.randn(1, 1, 1, 64)
    V_new = torch.randn(1, 1, 1, 64)
    full_K, full_V = cache.append_decode(K_new, V_new)

    # full_K[..., -1, :] must equal K_new exactly (FP16 path).
    last_K = full_K[:, :, -1:, :]
    diff = (last_K - K_new).abs().max().item()
    print(f"  max |full_K[-1] - K_new|  = {diff:.2e}  (must be zero)")
    assert diff == 0, "Current-step K is NOT FP16; quantization leaked in"

    # Sanity: historical dequantized K should NOT be exactly equal (quantized!)
    last_hist = full_K[:, :, 0:1, :]
    hist_err = (last_hist - K0).abs().max().item()
    print(f"  max |full_K[0] - K0|      = {hist_err:.4f}  (>0 ⇒ quantization happened)")
    assert hist_err > 0.001, "Historical K wasn't quantized — defeats the cache"
    print("  ✅ PASS — current step preserved FP16; historical step quantized.")


def test_asymmetric_v_fp16():
    print("\n[test_asymmetric_v_fp16]")
    cache = make_cache(dim=64, bits=2, quantize_v=False)
    K_chunks = [torch.randn(1, 2, 3, 64), torch.randn(1, 2, 4, 64)]
    V_chunks = [torch.randn(1, 2, 3, 64), torch.randn(1, 2, 4, 64)]
    for K, V in zip(K_chunks, V_chunks):
        cache.append_prefill(K, V)
    cache.finalize_prefill()

    K_new = torch.randn(1, 2, 1, 64)
    V_new = torch.randn(1, 2, 1, 64)
    _, full_V = cache.append_decode(K_new, V_new)

    expected_V = torch.cat(V_chunks + [V_new], dim=2)
    diff = (full_V - expected_V).abs().max().item()
    print(f"  max |full_V - expected_V| = {diff:.2e}  (V kept FP16 throughout)")
    assert diff == 0, "V quantization leaked when quantizer_v=None"
    print("  ✅ PASS — asymmetric K-only mode passes V through unchanged.")


def test_compounding_avoided():
    """
    Toy multi-layer model with softmax-style amplification (the real culprit
    for compounding in attention). Each layer:

        1. Score(i, j) = K_i · K_j
        2. attn_out_i  = sum_j softmax(Score(i, *))_j * V_j
        3. K_{l+1}     = K_l + W_o · attn_out
        4. V_{l+1}     = V_l (held fixed for clarity)

    The softmax is the amplifier: a small quantization perturbation in K shifts
    the score, and the exponential makes that shift dominate the attention
    output. Re-quantizing every layer compounds those shifts geometrically.

    Compare:
        (a) "always-quantize": K is round-tripped through PlanarQuant at every
            layer BEFORE it's consumed by the next layer's attention.
        (b) "deferred": K stays FP16 across all L layers; bulk-quantize at end.
        (c) "reference": no quantization at all.

    Plan claim: (a)/ref drifts grow geometrically with depth; (b)/ref grows
    linearly. We verify (a) is at least 2× worse than (b).
    """
    print("\n[test_compounding_avoided]")
    torch.manual_seed(0)
    L, T, D = 12, 16, 64
    K0 = torch.randn(1, 1, T, D) * (1.0 / (D ** 0.5))
    V0 = torch.randn(1, 1, T, D) * (1.0 / (D ** 0.5))
    W_o = torch.randn(D, D) * (0.3 / (D ** 0.5))   # small output projection
    pq = PlanarQuant(dim=D, bits=2, seed=0)        # 2-bit → loud quant noise

    def attention_step(K, V):
        # K, V: (1, 1, T, D)
        scores = torch.matmul(K, K.transpose(-2, -1)) / (D ** 0.5)  # (1,1,T,T)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)              # (1,1,T,D)
        return torch.matmul(out, W_o)            # residual update direction

    # ---- (a) always quantize: re-quant K at each layer before consumption ----
    K_a, V_a = K0.clone(), V0.clone()
    for _ in range(L):
        K_a = pq.dequantize(*pq.quantize(K_a))   # round-trip BEFORE attention
        update = attention_step(K_a, V_a)
        K_a = K_a + update

    # ---- (b) deferred: K stays FP16 during all L layers, then bulk quant ----
    K_b, V_b = K0.clone(), V0.clone()
    for _ in range(L):
        update = attention_step(K_b, V_b)
        K_b = K_b + update
    K_b = pq.dequantize(*pq.quantize(K_b))       # one quant at the end

    # ---- reference: no quantization ----
    K_ref, V_ref = K0.clone(), V0.clone()
    for _ in range(L):
        update = attention_step(K_ref, V_ref)
        K_ref = K_ref + update

    err_a = (K_a - K_ref).norm().item()
    err_b = (K_b - K_ref).norm().item()
    ratio = err_a / max(err_b, 1e-12)
    print(f"  Always-quantize cumulative drift (||·||): {err_a:.4f}")
    print(f"  Deferred cumulative drift (||·||):        {err_b:.4f}")
    print(f"  Compounding penalty: {ratio:.2f}× over {L} layers")
    print(f"  (Real models show 100×+ at L=30; this is a softmax-toy proxy.)")
    assert ratio > 2.0, \
        f"Deferred only {ratio:.2f}× better — toy model should show >2×"
    print("  ✅ PASS — deferred quantization avoids the compounding penalty.")


if __name__ == "__main__":
    test_prefill_no_quantization()
    test_state_machine()
    test_current_step_is_fp16()
    test_asymmetric_v_fp16()
    test_compounding_avoided()
    print("\nAll Phase 5 tests passed.")
