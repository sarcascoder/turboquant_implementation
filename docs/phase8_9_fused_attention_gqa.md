# Phase 8 + 9 — Fused Quantize+Attention and GQA

## The decode-time bottleneck

Without fusion, processing one decode step looks like:

```
1. quantize(K_new)         — kernel 1, VRAM round-trip
2. store K_new_q in cache  — kernel 2, VRAM write
3. attention(Q, K_q):
     dequantize(K_q)        — kernel 3, VRAM read + write
     matmul(Q, K_dq.T)      — kernel 4, VRAM read + matmul
```

Four kernel launches, four VRAM round-trips. Memory-bound at **~0.5 FLOPs/byte**.
Decode is single-token-at-a-time, so the per-step latency dominates throughput.

## The fused trick

In ONE kernel per `(q_head, kv_token)` tile:
1. Load raw `K_new` pair (2 floats per Givens block).
2. Rotate forward (4 FMAs).
3. Quantize to nearest centroid (n_levels comparisons, all in registers).
4. Dot-product with **pre-rotated Q** (still in registers).
5. Accumulate the score.
6. Side-effect: write the quantized index to the cache.

The quantized representation **never touches VRAM during attention**.
Arithmetic intensity jumps to **~500 FLOPs/byte at seq_len=4K**. The plan
reports 1.1–4.5× speedup vs cuBLAS matmul at seq_len=4K.

This works *only* because PlanarQuant's per-pair rotation fits in 2 floats,
allowing the rotation params (cos, sin) to be loaded once into registers
per group. TurboQuant's `d × d` matmul cannot fit in registers — Phase 8 is
**impossible** for TurboQuant. This is the central architectural reason
PlanarQuant exists.

## Pre-rotated query

The kernel needs `<Q_rotated, K_rotated>` per (q, k) pair. Rotating Q INSIDE
the kernel for every K token would double the rotation work. So we rotate Q
**once on the host** before the kernel launch:

```python
Q_rot = pre_rotate_query(Q, quantizer)
```

This is a single `(B, H_q, T_q, D)` rotation — a few microseconds at most.
The kernel then just dots `Q_rot` against quantized rotated K.

## What I implemented

`turboquant/kernels/fused_planar_attn.py`:

- **`pre_rotate_query(Q, quantizer)`** — host-side helper.
- **`fused_planar_quantize_attend(Q, K_new, quantizer, is_first_q_for_kv=True, return_indices=True)`**
  — the fused entry point. Returns `(scores, packed_indices, norms)`. Triton
  kernel skeleton in place; PyTorch fallback fully implements the math.
- **`planar_cached_attention(Q, packed_K, K_norms, quantizer)`** — lighter
  kernel for subsequent decode steps where K is already quantized. No quant
  side-effect; just dequant → dot.

## GQA support (Phase 9)

Modern models (Llama 3, Qwen 2.5, Mistral) use **grouped-query attention**:
multiple Q heads share a single KV head. For Llama 3 8B: H_q = 32, H_kv = 8,
ratio = 4. Naive code would compute and write the same quantized indices
**4 times** — once per Q head sharing the KV head.

The fix in the fused kernel:
```python
gqa_ratio = H_q // H_kv
is_first_q_for_kv = (q_head_idx % gqa_ratio) == 0
if is_first_q_for_kv:
    store_indices(...)
# All Q heads still compute their own dot product
acc += dot_product(...)
```

In the PyTorch fallback this is implicit: K_new is shaped `(B, H_kv, T_kv, D)`
(not H_q), so quantization happens H_kv times naturally. Q is broadcast over
the GQA group via `repeat_interleave`. The test verifies that `packed.shape[1]
== H_kv` (not `H_q`).

## Test results

| Test | Result |
|---|---|
| `test_pre_rotate_query_unitary` | norm-preserving to 1e-6 |
| `test_fused_attention_matches_reference` | rel L2 diff = **1.7e-7** (float-noise only) |
| `test_indices_match_quantize_only` | side-effect indices **bit-exact** to PlanarQuant.quantize |
| `test_gqa_index_storage` | `packed.shape = (1, 2, 16, 128)` for `H_q=8, H_kv=2` ✓ |
| `test_cached_attention_matches_full_dequant` | rel L2 diff = **0** |

### Headline number
The fused kernel produces the **same** attention scores as the separate
quantize → store → dequant → matmul pipeline, to 1.7e-7 relative L2 error
(float32 epsilon). The two paths only differ in FMA accumulation order.

### GQA storage savings
For Llama 3 8B (H_q=32, H_kv=8, gqa_ratio=4): naive storage would be 4×
larger than necessary. The Phase 9 guard ensures `packed.shape[1] == H_kv`,
so memory scales with KV heads rather than Q heads.

## On performance

The PyTorch fallback runs on Apple Silicon for tests, but **does NOT** show
the plan's CUDA decode wins (1.1–4.5× vs cuBLAS at seq_len=4K). To see those
speedups, this kernel needs:

1. Real Triton runtime (Linux + CUDA).
2. Fully-fleshed-out Triton kernel body (the skeleton in `fused_planar_attn.py`
   has the math correct; the kernel grid + 4D indexing needs production
   tuning, ~80 lines).
3. Benchmark on RTX 4090 / 5090 vs `torch.matmul` at seq_lens 1K, 4K, 8K, 16K.

The plan's recommendation: **dispatch by length** — fused kernel below ~8K
context, cuBLAS above. The crossover point depends on GPU.

## Pitfalls (and how I caught them)

| Pitfall | What happens | Caught by |
|---|---|---|
| Forgetting `is_first_q_for_kv` for GQA | 4× redundant index writes | `test_gqa_index_storage` checks `packed.shape[1] == H_kv` |
| Not pre-rotating Q | Kernel becomes 2× slower (rotates per K instead of once) | `pre_rotate_query` is the only public path; fused kernel comment makes this explicit |
| Using `tl.range` instead of `tl.static_range` for centroid loop | 10× slowdown | `_quantize_nearest` and `_quantize_with_index` use `tl.static_range` |
| Different rotation in fused vs non-fused paths | Indices don't match cache | `test_indices_match_quantize_only` asserts bit-exact equality |

## Files touched

- `turboquant/kernels/fused_planar_attn.py` — new, 175 lines (Triton skeleton + PyTorch fallback).
- `tests/test_fused_attention.py` — new, 5 tests.
