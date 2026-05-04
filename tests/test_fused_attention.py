"""
Phase 8 + Phase 9 validation: fused quantize+attention kernel and GQA support.

Tests:
1. test_pre_rotate_query_unitary
     pre_rotate_query is invertible (norm preserving).

2. test_fused_attention_matches_reference
     fused_planar_quantize_attend gives the SAME attention scores as
     (PlanarQuant.quantize → store → manual attention with dequantized K).

3. test_indices_match_quantize_only
     The indices returned as the side-effect of fused attention are bit-exact
     to what PlanarQuant.quantize would produce.

4. test_gqa_index_storage
     For H_q=8, H_kv=2 (GQA ratio 4): the K storage size scales with H_kv,
     not H_q (i.e., we don't store 4 redundant copies). Caller is responsible
     for passing is_first_q_for_kv per Q head; this test verifies the API.

5. test_cached_attention_matches_full_dequant
     planar_cached_attention (later decode steps) gives the same answer as
     dequantize → manual attention.
"""
import math
import torch
from turboquant.planarquant import PlanarQuant
from turboquant.kernels.fused_planar_attn import (
    pre_rotate_query,
    fused_planar_quantize_attend,
    planar_cached_attention,
)


def random_unit_vectors(*shape, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(*shape, generator=g)
    return x / x.norm(dim=-1, keepdim=True)


def test_pre_rotate_query_unitary():
    print("\n[test_pre_rotate_query_unitary]")
    pq = PlanarQuant(dim=128, bits=3, seed=0)
    Q = torch.randn(2, 4, 1, 128)
    Q_rot = pre_rotate_query(Q, pq)
    norm_q = Q.norm(dim=-1)
    norm_q_rot = Q_rot.norm(dim=-1)
    diff = (norm_q - norm_q_rot).abs().max().item()
    print(f"  max ||Q|| - ||Q_rot|| = {diff:.2e}")
    assert diff < 1e-5, "Pre-rotation not norm-preserving"
    print("  ✅ PASS — Q rotation is norm-preserving (orthogonal).")


def test_fused_attention_matches_reference():
    print("\n[test_fused_attention_matches_reference]")
    B, H, T_q, T_kv, D = 2, 4, 1, 16, 128
    Q = random_unit_vectors(B, H, T_q, D, seed=0) / math.sqrt(D)
    K = random_unit_vectors(B, H, T_kv, D, seed=1)

    pq = PlanarQuant(dim=D, bits=4, seed=0)

    # Fused path
    scores_fused, packed, norms = fused_planar_quantize_attend(Q, K, pq)

    # Reference path: quantize K externally, then dequant, then attention
    packed_ref, norms_ref = pq.quantize(K)
    K_deq = pq.dequantize(packed_ref, norms_ref)
    scores_ref = torch.matmul(Q, K_deq.transpose(-2, -1)) / math.sqrt(D)

    diff = (scores_fused - scores_ref).abs().max().item()
    rel = (scores_fused - scores_ref).norm().item() / scores_ref.norm().item()
    print(f"  max |fused - reference| = {diff:.4e}")
    print(f"  relative L2 diff        = {rel:.4e}")
    # Two paths have rounding differences in the final FMA accumulation order
    assert rel < 1e-4, f"Fused vs reference rel diff {rel} too high"
    print("  ✅ PASS — fused kernel attention scores match reference within float precision.")


def test_indices_match_quantize_only():
    """The side-effect indices must be bit-exactly what PlanarQuant.quantize produces."""
    print("\n[test_indices_match_quantize_only]")
    B, H, T_q, T_kv, D = 2, 4, 1, 16, 128
    Q = torch.randn(B, H, T_q, D)
    K = random_unit_vectors(B, H, T_kv, D, seed=2)

    pq = PlanarQuant(dim=D, bits=3, seed=0)
    _, packed_fused, _ = fused_planar_quantize_attend(Q, K, pq)
    packed_quant_only, _ = pq.quantize(K)

    eq = torch.equal(packed_fused, packed_quant_only)
    print(f"  fused indices bit-exact equal to quantize-only output: {eq}")
    assert eq, "Index mismatch — fused kernel produces different bits than quantize"
    print("  ✅ PASS — side-effect indices are bit-exact.")


def test_gqa_index_storage():
    """
    Phase 9: 8 Q heads, 2 KV heads. K storage must scale with H_kv=2, not H_q=8.
    The fused kernel handles this via the H_q != H_kv broadcasting in the
    PyTorch path. The plan's GQA fix in Triton uses `is_first_q_for_kv` but
    here the dispatch is implicit because we pass K_new with H_kv heads.
    """
    print("\n[test_gqa_index_storage]")
    B, H_q, H_kv, T_q, T_kv, D = 1, 8, 2, 1, 16, 128
    assert H_q % H_kv == 0
    Q = torch.randn(B, H_q, T_q, D)
    K = random_unit_vectors(B, H_kv, T_kv, D, seed=3)

    pq = PlanarQuant(dim=D, bits=3, seed=0)
    scores, packed, norms = fused_planar_quantize_attend(Q, K, pq)

    print(f"  Q shape:      {tuple(Q.shape)}")
    print(f"  K shape:      {tuple(K.shape)}")
    print(f"  scores shape: {tuple(scores.shape)}  (expected (1, 8, 1, 16))")
    print(f"  packed shape: {tuple(packed.shape)}  (expected H_kv=2 leading)")
    print(f"  norms  shape: {tuple(norms.shape)}")
    # Critical: packed should have H_kv (not H_q) along the head axis
    assert packed.shape[1] == H_kv, \
        f"GQA broken: packed.shape[1]={packed.shape[1]} should be {H_kv}"
    assert scores.shape == (B, H_q, T_q, T_kv)
    print(f"  Stored size scales with H_kv={H_kv}, not H_q={H_q}: ✓")
    print("  ✅ PASS — GQA storage scales with H_kv as intended.")


def test_cached_attention_matches_full_dequant():
    print("\n[test_cached_attention_matches_full_dequant]")
    B, H, T_q, T_kv, D = 1, 4, 1, 64, 128
    Q = torch.randn(B, H, T_q, D)
    K = random_unit_vectors(B, H, T_kv, D, seed=4)
    pq = PlanarQuant(dim=D, bits=4, seed=0)
    packed, norms = pq.quantize(K)

    cached = planar_cached_attention(Q, packed, norms, pq)

    K_deq = pq.dequantize(packed, norms)
    ref   = torch.matmul(Q, K_deq.transpose(-2, -1)) / math.sqrt(D)

    rel = (cached - ref).norm().item() / ref.norm().item()
    print(f"  relative L2 diff = {rel:.2e}")
    assert rel < 1e-5, f"Cached attention drift {rel}"
    print("  ✅ PASS — cached attention matches full dequant + matmul.")


if __name__ == "__main__":
    test_pre_rotate_query_unitary()
    test_fused_attention_matches_reference()
    test_indices_match_quantize_only()
    test_gqa_index_storage()
    test_cached_attention_matches_full_dequant()
    print("\nAll Phase 8 + 9 fused-attention / GQA tests passed.")
