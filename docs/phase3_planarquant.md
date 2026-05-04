# Phase 3 — PlanarQuant (Block-Diagonal Givens Rotation)

## The core insight

A `d × d` random rotation matrix is **wasteful** for KV-cache quantization.
After the random rotation each coordinate is approximately Beta(d) / Gaussian(0, 1/d)-distributed,
and the Lloyd-Max codebook is built for that marginal distribution. The rotation
only needs to **mix coordinates enough that the marginal looks like the assumed PDF**.
Block-diagonal rotations do this just as well as a dense rotation — and they:

| Property | TurboQuant (dense d×d) | **PlanarQuant** (d/2 Givens) |
|---|---:|---:|
| FMAs (forward + inverse) | 33,000 (d=128) | **512** (64× less) |
| Stored parameters | 16,384 (d=128) | **128** (125× less) |
| Fits in GPU registers? | No | **Yes** → enables fused Triton kernels |
| Round-trip MSE (3-bit, d=128) | 0.034119 | **0.034023** (≈ same) |

## The math

### A 2D Givens rotation
```
                                  ┌  cos θ   -sin θ ┐
R(θ) = rotate by angle θ in R² =  │                 │
                                  └  sin θ    cos θ ┘
```

### Forward rotation (per pair)
```
r0 =  cos · v0 - sin · v1
r1 =  sin · v0 + cos · v1
```
Cost: **4 FMAs**.

### Inverse rotation (per pair) — the **transpose** of R
```
v0_hat =  cos · q0 + sin · q1
v1_hat = -sin · q0 + cos · q1            ← negate sin
```
Cost: **4 FMAs**. **The negation of `sin` is the single most error-prone line in
the project.** Forgetting it gives a different rotation entirely (rotates by
−θ in the opposite sense, which equals R(2θ) when applied after R(θ)):

```
R(θ) · R(θ) = R(2θ)   ≠   I
```

So the round-trip ends up rotated by 2θ instead of returning to the original.
Empirically (see test below): MSE jumps from 0.034 → 1.99 (58× worse).

### Block-diagonal structure (full vector)
For `d = 128`, n_groups = 64. The rotation is:
```
R = blockdiag(R(θ_1), R(θ_2), ..., R(θ_64))
```
- Independent across blocks → no cross-coordinate dependency at rotation time.
- Each `R(θ_i)` is orthogonal → preserves norm (after we factor it out).
- The 128-dim post-rotation distribution is **the same** Beta(d=128) marginally,
  because each output coordinate is a Gaussian-weighted mixture of two input
  coordinates that were already approximately Gaussian after norm separation.

## What I implemented

`turboquant/planarquant.py`:
- `givens_forward(pairs, rot2)` — 4 FMAs/pair, vectorised over groups.
- `givens_inverse(pairs, rot2)` — same, with **explicit negation of sin**.
- `PlanarQuant(dim, bits, seed=None, pad_to_even=True)` — `nn.Module` with the
  same `quantize / dequantize` API as `TurboQuantMSE`.
- Diagnostic properties `fma_count_round_trip` and `parameter_count`.
- Optional `seed` parameter — **always pass a seed when storing/reading from a
  cache**, otherwise the Givens angles drift between runs and the cache becomes
  unreadable.

`tests/test_planarquant.py` — four tests:
1. **Counts** — verify FMA / parameter counts match the plan.
2. **Inverse exactness** — verify `givens_inverse(givens_forward(x)) ≈ x`
   to `1e-5` (no quantization).
3. **Round-trip MSE vs TurboQuant** — within 10% across 2/3/4 bits on 10K random unit vectors.
4. **Inverse-rotation trap** — confirm the canonical bug (using forward in dequant)
   is detectable by this test (58× MSE penalty triggers the assertion).

## Empirical results (`d=128`, 10K random unit vectors)

| bits | TurboQuant MSE | PlanarQuant MSE | Ratio | cos sim |
|-----:|---------------:|----------------:|------:|--------:|
| 2    | 0.1159         | 0.1161          | 1.001 | 0.9405  |
| 3    | 0.0341         | 0.0340          | 0.997 | 0.9831  |
| 4    | 0.00932        | 0.00932         | 0.999 | 0.9954  |

PlanarQuant is **statistically equivalent** to TurboQuant on MSE despite 64×
fewer FMAs and 125× fewer parameters. This confirms the central thesis: the
heavy lifting is done by the Lloyd-Max codebook (matched to the marginal),
not by the dense rotation.

### Counts (matches plan exactly)
| d | FMAs (round-trip) | params |
|---|------------------:|-------:|
| 64  | 256             | 64     |
| 128 | **512**         | **128**|
| 256 | 1024            | 256    |

### Inverse exactness
`max |inv(fwd(x)) - x| = 4.77e-7` on random vectors — well within float32 precision.

### Inverse-rotation trap
| Path | MSE |
|---|---:|
| Correct (`givens_inverse` in dequant) | 0.0339 |
| BUGGY  (`givens_forward` in dequant)  | 1.9858 |
| Penalty | **58.5×** |

The trap test confirms that **a future regression that swaps inverse for
forward in the dequant path will be caught immediately by the test suite**.

## Why this enables Phase 7/8

PlanarQuant's per-pair rotation fits in GPU registers. A Triton kernel can:

1. Load 2 floats per pair (`v0, v1`) and 2 angle params (`cos, sin`).
2. Compute the forward rotation, quantize, immediately compute the dot product
   with the corresponding pre-rotated query pair, accumulate.
3. Move to the next pair.

The quantized representation never touches VRAM during attention. This is
**impossible for TurboQuant** because a `d × d` matmul requires materializing
the rotated vector. The fused attention kernel in Phase 8 only works for
PlanarQuant / IsoQuant.

## Pitfalls (and how I caught them)

| Pitfall | What happens | Caught by |
|---|---|---|
| Forgetting to negate `sin` in inverse | Round-trip is rotated by 2θ; MSE explodes ~60× | `test_inverse_rotation_trap` (asserts >5× penalty) |
| Wrong reshape order (interleaved vs paired) | Pairs `(v_{2i}, v_{2i+1})` in `quantize` don't match in `dequantize` | Same `reshape(..., n_groups, 2)` used both sides |
| Different angles between quantize/dequantize | Cache becomes unreadable | `seed` parameter; angles stored as buffer |
| Odd `d` | Reshape-by-2 fails | `pad_to_even=True` pads with one zero, trims after dequant |
| Forgetting the inverse rotation altogether | Returned vector is in the rotated basis | Inverse is the only path through `dequantize` — no shortcut |

## What's next

Phase 5 (DeferredQuantCache) wraps `PlanarQuant` (or any quantizer) so the
prefill phase runs at FP16 and quantization only kicks in for decode. Without
that, even a perfect quantizer gives PPL > 1000 on real models due to error
compounding across transformer layers.

## Files touched

- `turboquant/planarquant.py` — new, 130 lines.
- `tests/test_planarquant.py` — new, 4 tests covering counts, inverse exactness,
  round-trip MSE parity, and the inverse-rotation trap.
