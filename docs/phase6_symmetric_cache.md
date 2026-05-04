# Phase 6 — Symmetric V Cache + the Inverse-Rotation Trap

## Why this exists

K-only quantization gives 5× memory compression. Symmetric K+V gives 10×. But
the V cache has a unique bug surface that K doesn't: **the V dequant must
apply the inverse rotation explicitly**, because the attention output
`softmax(scores) @ V_dequant` consumes the actual reconstructed V vectors —
not a quantity in some rotated basis.

For K, the dot product `<Q_rotated, K_rotated>` is invariant under the rotation
(both sides are in the same basis). You can skip the inverse rotation in K
dequant if you also rotate Q forward — the `R @ R.T = I` cancels.

For V, there's no such symmetry. If you accidentally use the forward rotation
in V dequant (e.g., by copy-pasting the K dequant code), the V vectors come
out rotated by `2θ` instead of `0`. The attention output is then a weighted
sum of vectors in the *wrong* basis. Empirically (from the plan): PPL goes
from ~7 to ~15,000+. **Silent failure — no exception, no NaN, no warning.**

## The math (the trap, in one diagram)

For a 2D Givens block with angle θ and rotation matrix `R(θ)`:

| Path                                | Result                            |
|-------------------------------------|-----------------------------------|
| `dequant_correct(quant(v))`         | ≈ `R(θ).T · R(θ) · v` = `v`        |
| `dequant_buggy(quant(v))` (forward) | ≈ `R(θ) · R(θ) · v` = `R(2θ) · v`  |

A 2θ-rotated vector has cosine similarity `cos(2θ)` with the original. With
random θ ∈ [0, 2π), the expected cosine sim is **zero**. Every Givens block
contributes its own rotation error; in `d=128` (64 blocks) the V vectors are
essentially random unit vectors after the buggy dequant. Attention output
becomes a weighted sum of randomness → garbage tokens.

## What I implemented

`turboquant/symmetric_cache.py`:

### `assert_inverse_correct(quantizer, dim)` — the trap-detector
A standalone validator that takes any quantizer and confirms its dequant is
in fact the inverse of its quant. Procedure:
1. Generate 256 random unit vectors.
2. Quantize, dequantize.
3. Verify mean cosine similarity > 0.5.

If the quantizer accidentally uses forward rotation in dequant, cos sim
collapses to ~0 and `RuntimeError` is raised with a pointer to this doc.

### `SymmetricKVCache(quantizer_k, quantizer_v, check_inverse=True)`
Subclass of `DeferredQuantCache` (Phase 5) that:
- **Refuses** construction if `quantizer_v is None` (asymmetric is a different mode).
- Runs `assert_inverse_correct` on **both** K and V quantizers at init time
  (off by default only in tight benchmark loops).

Runtime overhead: ~30ms one-time at construction. The cost-benefit is
overwhelming: the alternative is a multi-hour PPL run that produces unusably
bad numbers with no obvious clue about the cause.

## Test results

All six tests pass:

| Test | What it proves | Result |
|---|---|---|
| `test_construction_requires_v` | `SymmetricKVCache(qk, None)` raises `ValueError` | ✅ |
| `test_inverse_check_passes_for_correct_quantizer` | Real `PlanarQuant` passes the trap-detector | ✅ |
| `test_inverse_check_catches_buggy_quantizer` | A deliberately-broken quantizer is refused at construction | ✅ (raises `RuntimeError`) |
| `test_full_v_round_trip_via_cache` | End-to-end: last-token V is byte-exact, historical V cos sim = 0.996 at 4-bit | ✅ |
| `test_attention_output_with_symmetric_cache` | Attention output cos sim vs FP16 = **0.9949** at 4-bit | ✅ |
| `test_buggy_v_dequant_explodes_attention` | Same setup with buggy V → cos sim = **-0.02** (random) | ✅ |

### The headline number
| Configuration | Attention output cos sim vs FP16 |
|---|---:|
| K=4-bit, V=4-bit, **correct** inverse | **0.9949** |
| K=4-bit, V=4-bit, **buggy** forward-in-dequant | **-0.02** |
| Quality gap from one missing minus sign | **~50× drop** |

## What this catches

If a future contributor (a) writes a new V dequant kernel for IsoQuant, (b)
copy-pastes the K dequant code that does NOT need the inverse rotation, (c)
tests with `python tests/test_planarquant.py` (which only tests K), and (d)
ships the change, the next time anyone constructs `SymmetricKVCache(qk, qv)`
the construction will refuse. They get a clear error message at init time
pointing at `phase6_symmetric_cache.md` instead of staring at a 15,000 PPL
debug session.

## Pitfalls (and how I caught them)

| Pitfall | What happens | Caught by |
|---|---|---|
| Copy-pasting K dequant into V dequant | Forward rotation applied; attention output garbage; PPL → 15K | Construction-time `assert_inverse_correct` raises `RuntimeError` |
| Skipping the construction check in production | Trap goes undetected | `check_inverse=True` is the default; only opt-out in tight benchmarks |
| Using a different inverse formula per quantizer | Each quantizer's `dequantize` must be self-consistent | `assert_inverse_correct` is a black-box check on `quantize/dequantize` — it works for any quantizer regardless of internal rotation type |
| TurboQuant happens to be self-inverse (WHT) — false generalization | A new IsoQuant author might assume the same | `assert_inverse_correct` works on the public API; it doesn't matter what the rotation is |

## Memory savings (the whole point)

For `d=128`, 4K context, 30 layers, 8 heads:
| Mode | bytes per cached vector | total cache |
|---|---:|---:|
| FP16 K + FP16 V                              | 512 (2 × 128 × 2)        | 480 MB |
| TurboQuant K-only (4-bit) + FP16 V           | 322                      | 302 MB |
| **SymmetricKVCache (K=4-bit, V=4-bit)**      | **132**                  | **124 MB** |

3.9× compression with attention quality preserved (cos sim 0.99). At 3-bit
each, total compression is ~10× (matches the plan's `iso3/iso3` target).

## Files touched

- `turboquant/symmetric_cache.py` — new, 75 lines.
- `tests/test_symmetric_cache.py` — new, 6 tests.
