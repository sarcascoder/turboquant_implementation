# Phase 10 — RaBitQ (1-Bit Sign Packing for Extreme Compression)

## When to use

| Scenario | Use? |
|---|---|
| LLM generation (need readable outputs) | ❌ — PPL goes from ~7 → ~107 |
| Long-context retrieval (top-k, no decode) | ✅ — 12.8× memory savings |
| Approximate nearest-neighbor (ANN) search | ✅ |
| Vector reranking with FP16 query | ✅ — asymmetric IP is unbiased |

The plan classifies this as **optional** for general use. It's the right choice
when memory is the absolute bottleneck and bit-rank-style retrieval is the
consumer of the index.

## Storage layout (`d=128`)

| Field | Size | Notes |
|---|---:|---|
| Packed signs | 16 bytes | `d/8` — 1 bit per coordinate |
| `||x||` (norm) | 2 bytes | FP16 |
| `x0` (alignment scalar) | 2 bytes | FP16 — `mean(|R x|)` |
| **Total** | **20 bytes** | vs **256 bytes** FP16 → **12.8× compression** |

## The math

### Quantize
```
u   = x / ||x||                                        # unit vector
r   = R u                          ∈ R^d              # rotated unit vector
s   = sign(r)                       ∈ {-1, +1}^d      # binary signs
x0  = (1/d) · ||r||₁ = mean(|r|)                       # scalar alignment factor
```
Stored: `pack_bits(s)` + `||x||` + `x0`.

### Asymmetric inner product (query stays full-precision)
```
q       = R y                      ∈ R^d
raw_ip  = <q, s>                                      # signed sum, scalar
<y, x> ≈ ||x|| · (π/2) · x0 · raw_ip                  # UNBIASED
```

### Where the (π/2) factor comes from

For unit vectors `x, y` uniform on the (d-1)-sphere, with rotation `R`,
let `r = R u`, `q = R y` (both unit). Decompose `q` along `s = sign(r)`:

```
q = α · s/||s|| + q_⊥        with α = <q, s>/√d
```

Then for fixed `r`:
```
E_q [<q, r> · <q, s>] = (1/d) · sum_i r_i · sign(r_i) = ||r||₁ / d = x0
```

Across pairs `(r, q)` with `q` independent of `r`:
```
E[<q, r>] = 0,  E[<q, s>] = 0
Cov(true_ip, raw_est) = E_r[x0 · x0] = E[x0²]
Var(true_ip) = E[<q, r>²] = ||r||²/d = 1/d
```

So the regression slope of `(x0 · raw_ip)` against `<y, x>` is:
```
slope = E[x0²] / (1/d) = d · E[x0²]
```

For uniform unit vectors, `E[x0] ≈ √(2/(π·d))`, so:
```
slope ≈ d · (2/(π·d)) = 2/π ≈ 0.637
```

**Multiplying the estimator by `π/2 ≈ 1.5708` cancels exactly this bias →
slope = 1.** This is exactly the empirical correction my test showed:

| Estimator | Slope (10K pairs, d=128) |
|---|---:|
| `||x|| · x0 · <q, s>` (no correction) | **0.6415** ≈ 2/π |
| `||x|| · (π/2) · x0 · <q, s>` (corrected) | **1.0077** ≈ 1.0 |

## What I implemented

`turboquant/rabitq.py` — `RaBitQ(dim, rotation, seed)`:

- Three rotation backends:
  - `'full'` — random `d × d` orthogonal matrix (sign-fixed QR)
  - `'planar'` — reuses `PlanarQuant` rotation
  - `'iso'`    — reuses `IsoQuant` rotation (fast mode)
- `quantize(x)` → `(packed_signs, norms, x0)`
- `dequantize(packed, norms, x0)` → `x_hat` (norm biased by `√(2/π) ≈ 0.798`,
  small but consistent — fine for visualization, NOT for unbiased reconstruction)
- `estimate_inner_product_asymmetric(y, packed, norms, x0)` — returns
  **unbiased** `<y, x>` with the `π/2` correction baked in.

## Test results

| Test | Result |
|---|---|
| `test_storage_layout` | 20 bytes / vec, 12.80× compression — matches plan |
| `test_signs_packed_correctly` | `pack ∘ unpack` is identity at 1-bit |
| `test_round_trip_norm` | Norm recovered = 0.795 ≈ √(2/π) — expected bias |
| `test_asymmetric_ip` | slope = **1.0077**, intercept ≈ 0, mean bias = 3.5e-4 |
| `test_rotation_backends` | All three backends ~equivalent on synthetic data |

### Why all three backends look equivalent on synthetic data

The plan's claim is that on **real LLM activations**, only the `'full'`
rotation gives acceptable 1-bit quality (PPL 107 vs 600+ for the
block-diagonal backends). On my synthetic random-unit-vector test, all three
backends give cos sim ≈ 0.80 and slope ≈ 1.

The plan's reasoning: real K cache vectors have inter-coordinate correlations
(due to RoPE, projection structure). PlanarQuant/IsoQuant rotate independent
groups, so within-group correlations are decorrelated but cross-group
correlations leak through. At 3+ bits the codebook absorbs that residual; at
1 bit there's no headroom.

To reproduce the plan's finding requires real model activations, which I
don't have a CUDA-accessible model to extract from in this environment. The
test framework is in place; passing real LLM K vectors through it would show
the gap.

## Pitfalls (and how I caught them)

| Pitfall | What happens | Caught by |
|---|---|---|
| Storing signs as `int8` instead of packed bits | 8× memory waste — defeats the purpose | `pack_bits(..., 1)` packs 8 signs/byte; `bytes_per_vector` property verifies size |
| Forgetting the `π/2` scaling factor in IP | Estimator biased by 2/π | `test_asymmetric_ip` slope check (caught my own first version: 0.64 vs target 1.0) |
| Using PlanarQuant rotation at 1-bit on real LLM KV | Quality collapse beyond what synthetic data shows | Documented in this file — needs real LLM activation tests to reproduce |
| `dequantize` consumed as if unbiased | Vectors are 0.798× shorter than original | Docstring warns; users should call `estimate_inner_product_asymmetric` instead |

## Compression numbers

For `d=128`, 4K context, 30 layers, 8 KV heads:

| Mode | bytes/vector | total cache (single sequence) |
|---|---:|---:|
| FP16 K + FP16 V | 512 | 480 MB |
| `iso3 + iso3` (Phase 6 symmetric) | 132 | **124 MB** (3.9×) |
| **RaBitQ K + RaBitQ V** | **40** | **38 MB (12.6×)** |

12.6× total cache compression vs FP16 at the cost of generation quality.
For a 100K-context retrieval index this is the difference between fitting in
RAM or not.

## Files touched

- `turboquant/rabitq.py` — new, 130 lines.
- `tests/test_rabitq.py` — new, 5 tests.
