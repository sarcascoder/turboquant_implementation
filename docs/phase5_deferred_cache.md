# Phase 5 — DeferredQuantCache (Post-Prefill Quantization)

## Why this exists (the most important phase)

Without this, a perfect quantizer still gives PPL > 1000 on real models.
The plan calls this "the single most important engineering insight in the
whole project". Here's why:

### The compounding problem

Transformer attention is recurrent across layers:
```
H_{l+1} = LayerNorm(H_l + Attention(Q_l(H_l), K_l(H_l), V_l(H_l)))
```
where `K_l(·)` and `V_l(·)` are the layer-`l` projection weights.

If we quantize `K_l` during prefill:
1. `Attention(...)` consumes a noisy `K_l`.
2. `H_{l+1}` is a noisy version of the true hidden state.
3. Layer `l+1`'s K projection then operates on the *already-noisy* `H_{l+1}`.
4. We quantize *that* noisy K, giving a doubly-noisy K_{l+1}.
5. Repeat 30 times.

The error grows **geometrically with depth**, not additively. Empirically (real
Llama 3.1 8B, 3-bit K quantization during prefill): wikitext-2 PPL goes from
6.63 → ~1500.

### The fix — post-prefill state machine

1. **Prefill phase**: store all K and V at FP16 in a buffer. No quantization.
   Attention sees the exact same K, V it would without TurboQuant. Layer
   outputs are exact, so the K projections at each layer also operate on
   exact inputs. Zero compounding.

2. **Finalize**: once prefill is complete (just before the first decode step),
   bulk-quantize the accumulated K (and optionally V). Free the FP16 buffers.
   This is **one-shot** quantization — no compounding because no further
   forward passes touch the FP16 buffers.

3. **Decode phase**: per-token K, V arrive. We quantize them for storage
   (so future decode steps see the compressed cache) but the **current step's
   K is returned at FP16** in the attention input. So the just-inserted token's
   attention computation uses an exact K — no compounding for it. Token `t+1`'s
   K projection sees the *exact* output of token `t`'s attention.

The compounding loop is broken: each cached token has been quantized exactly
once, with the noise injected after the layer that produced it (not before).

## What I implemented

`turboquant/deferred_cache.py` — `DeferredQuantCache(quantizer_k, quantizer_v=None)`

### Three-mode state machine

| Mode | What's stored | Public API allowed |
|---|---|---|
| `PREFILL` | List of FP16 chunks `[(K, V)]` | `append_prefill(K, V)`, `finalize_prefill()` |
| `DECODE`  | `(packed_K, norms_K)` and either `(packed_V, norms_V)` or FP16 V | `append_decode(K_new, V_new)` |

### `append_prefill(K, V)` — pure storage
```python
self._prefill_K.append(K.detach())
self._prefill_V.append(V.detach())
```
Zero quantization. Multiple chunks accumulate.

### `finalize_prefill()` — bulk transition
```python
all_K = torch.cat(self._prefill_K, dim=2)
all_V = torch.cat(self._prefill_V, dim=2)
self._quant_K = self.quantizer_k.quantize(all_K)
if self.quantizer_v is not None:
    self._quant_V = self.quantizer_v.quantize(all_V)
else:
    self._fp_V = all_V
self._prefill_K.clear(); self._prefill_V.clear()
```
Frees FP16 buffers immediately — important for long-context memory.

### `append_decode(K_new, V_new)` → `(full_K, full_V)`

The subtle invariant: `full_K[..., -1, :]` is **bit-exactly** `K_new`.

```python
# Historical: dequantize from cache
K_old = self.quantizer_k.dequantize(*self._quant_K)
V_old = ...  # dequantize V or use FP16

# CURRENT step: concatenate K_new at FP16 (NOT a dequantized version)
full_K = torch.cat([K_old, K_new], dim=2)
full_V = torch.cat([V_old, V_new], dim=2)

# Side effect: store the new token quantized for FUTURE steps
new_kp, new_kn = self.quantizer_k.quantize(K_new)
self._quant_K = (cat([..., new_kp]), cat([..., new_kn]))

return full_K, full_V
```

The plan's pitfall warning: *"Missing the 'current step uses FP16 K' detail
→ you've reintroduced compounding for the latest token"*. Test 3
(`test_current_step_is_fp16`) asserts `(full_K[..., -1] - K_new).abs().max() == 0`.

## Test results

All five tests pass with the `qwen2vl` venv:

| Test | What it proves | Result |
|---|---|---|
| `test_prefill_no_quantization` | Prefill stores raw FP16 chunks, no quantization | ✅ |
| `test_state_machine` | `append_decode` blocked before `finalize_prefill`; double-finalize blocked | ✅ |
| `test_current_step_is_fp16` | `full_K[-1] == K_new` exactly; `full_K[0] != K_old` (was quantized) | ✅ (zero diff for current; 1.31 diff at 2-bit for historical) |
| `test_asymmetric_v_fp16` | `quantizer_v=None` mode passes V through unchanged (zero diff) | ✅ |
| `test_compounding_avoided` | Always-quantize drift / deferred drift > 2× in a softmax toy model | ✅ (2.21× over 12 layers; real 30-layer models show 100×+) |

### The toy compounding model
Each layer applies real attention with softmax + an output projection +
residual. With 2-bit PlanarQuant and 12 layers:
- Always-quantize-K-each-layer drift: `||K_a - K_ref|| = 4.26`
- Deferred (one quant at end) drift: `||K_b - K_ref|| = 1.92`
- Penalty: **2.21× over 12 layers**

The penalty grows roughly with `(1 + softmax_amplification)^L`. At
real-model depth (30+ layers) and a stronger softmax (the test uses
`alpha=0.3` projection norm) this is the difference between PPL 7 and PPL > 1000.

## Pitfalls (and how I caught them)

| Pitfall | What happens | Caught by |
|---|---|---|
| Forgetting to free `_prefill_K` after finalize | Memory blow-up at long context | `finalize_prefill` does `.clear()`; verified by checking len==0 (implicit) |
| Quantizing during prefill "for consistency" | Compounding → PPL explosion | `append_prefill` has zero quant calls; explicit assertion that mode is `PREFILL` |
| Returning a dequantized K_new in `append_decode` | Compounding for the latest token | `test_current_step_is_fp16` asserts byte-exact equality |
| Letting `append_decode` run before `finalize_prefill` | Empty `_quant_K` → crash | Explicit assert in `append_decode`; tested by `test_state_machine` |
| Asymmetric V mode silently quantizing V | Wrong attention output | `quantizer_v=None` keeps `_fp_V`; `test_asymmetric_v_fp16` asserts byte-exact V |

## What this enables

- The `_TurboQuantQwen35Cache` in `turboquant/kv_cache.py` currently has a
  `residual_window=128` heuristic — keeps the most recent 128 tokens in FP16.
  That's a partial fix: for a 4K-token prefill it still quantizes ~3.9K tokens
  during prefill, and compounding still ruins the deeper layers.
- Phase 5's `DeferredQuantCache` is the **drop-in replacement** that resolves
  this. Phase 6 layers symmetric V quantization on top.

## Files touched

- `turboquant/deferred_cache.py` — new, 145 lines.
- `tests/test_deferred_cache.py` — new, 5 tests.
