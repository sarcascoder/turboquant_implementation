# Phase 4 — IsoQuant (Quaternion 4D Block Rotations)

## Why IsoQuant when we already have PlanarQuant?

| Quantizer | Block size | DOF/block | FMAs (d=128) | Params (d=128) | 4-bit PPL (Qwen2.5-3B) |
|---|:---:|:---:|---:|---:|---:|
| TurboQuant     | d  | d² | 33,000 | 16,384 | 9.05 |
| PlanarQuant    | 2  | 1  | 512    | 128    | 10.12 |
| **IsoQuant (fast)** | 4  | 3  | **1,024** | **128** | **9.03** |
| IsoQuant (full)| 4  | 6  | 2,048  | 256    | 9.01 |

PlanarQuant has 1 DOF per pair (the angle θ). At higher bit budgets that's
limiting — there isn't enough freedom to tune the rotation around individual
coordinate distributions. IsoQuant's quaternions give **3 DOF per 4-block in
fast mode** (or 6 in full mode), letting the rotation tune more carefully
without exploding parameter count.

The plan reports IsoQuant as the **production default for 4-bit symmetric
configs** in llama.cpp.

## Quaternion algebra refresher

A quaternion is a 4-tuple `(w, x, y, z)` representing `w + xi + yj + zk`.
Multiplication follows Hamilton's rules `i² = j² = k² = ijk = -1`.

### Hamilton product (16 FMAs)
```
(a · b).w = aw·bw - ax·bx - ay·by - az·bz
(a · b).x = aw·bx + ax·bw + ay·bz - az·by
(a · b).y = aw·by - ax·bz + ay·bw + az·bx
(a · b).z = aw·bz + ax·by - ay·bx + az·bw
```

### Conjugate
```
conj(w, x, y, z) = (w, -x, -y, -z)
```

### Key identities
- `1 · q = q` (left identity)
- `q · conj(q) = ||q||² · 1` — for unit `q`, this is `(1, 0, 0, 0)`
- For unit `q`, the **inverse equals the conjugate**: `q⁻¹ = conj(q)`

This last identity is *the* reason quaternions work for rotation: rotating
by `q_L · v` and then multiplying by `conj(q_L)` returns to `v`. The
"negate sin" of PlanarQuant becomes "use conjugate" in IsoQuant.

## Two modes

### Fast mode (3 DOF per 4-block)
```
v_rot = q_L · v               (forward; 16 FMAs)
v     = conj(q_L) · v_rot     (inverse; 16 FMAs)
```
Total: 32 FMAs per group, 4 stored params per group.

### Full mode (6 DOF per 4-block)
```
v_rot = q_L · v · conj(q_R)   (forward; 32 FMAs)
v     = conj(q_L) · v_rot · q_R   (inverse; 32 FMAs)
```
Total: 64 FMAs per group, 8 stored params per group.

The plan: *"For 4-bit quantization, 3 DOF per 4D block decorrelates as well
as 6 DOF in practice. Use fast mode unless you specifically need the extra DOF."*

## What I implemented

`turboquant/isoquant.py`:
- `quat_mul(a, b)` — vectorised Hamilton product (16 FMAs).
- `quat_conj(q)` — sign-flip on (x, y, z).
- `random_unit_quaternions(n, seed=None)` — Gaussian + normalize → uniform on S³.
- `IsoQuant(dim, bits, mode='fast'|'full', seed, pad_to_4=True)`.
- Same `quantize / dequantize` API as `PlanarQuant`.
- Diagnostic properties `fma_count_round_trip` and `parameter_count`.

## Test results (`d=128`, 10K random unit vectors)

| Test | Result |
|---|---|
| `test_quat_mul_basics` (identity, conj∘conj, q·conj(q)=1) | ✅ |
| `test_inverse_rotates_correctly` (fast & full, max diff < 1e-5) | ✅ |
| `test_counts` (fast=1024/128, full=2048/256) | ✅ matches plan |
| `test_round_trip_vs_planarquant` | ✅ ratio 0.997 / 0.998 / 1.002 across 2/3/4 bits |
| `test_fast_vs_full_mode` | ✅ full marginally better at 4-bit (0.6% MSE win) |
| `test_inverse_rotation_trap_iso` | ✅ buggy forward → MSE penalty **86×** |

### Headline numbers

| bits | PlanarQuant MSE | IsoQuant (fast) MSE | Ratio |
|-----:|----------------:|--------------------:|------:|
| 2    | 0.1161          | 0.1157              | 0.997 |
| 3    | 0.0340          | 0.0340              | 0.998 |
| 4    | 0.00932         | 0.00934             | 1.002 |

| 4-bit mode | MSE     | Notes |
|---|---:|---|
| fast (3 DOF) | 0.00933 | 32 FMAs/group, 4 params/group |
| full (6 DOF) | 0.00928 | 64 FMAs/group, 8 params/group — **0.6% MSE win** |

The plan's claim is verified: at 4-bit the extra DOF of full mode buys
essentially nothing on synthetic data. PPL on real models may differ slightly
because real K vectors aren't isotropic Gaussian-on-the-sphere.

## Pitfalls (and how I caught them)

| Pitfall | What happens | Caught by |
|---|---|---|
| Forgetting to normalise quaternions | `q · conj(q) ≠ 1` → rotation no longer preserves norm | `random_unit_quaternions` divides by `q.norm(...)` |
| Component ordering inconsistency `(w,x,y,z)` vs `(x,y,z,w)` | Silent corruption | `quat_mul` and `quat_conj` agree on `(w, x, y, z)` |
| Forward rotation in dequant (analogous to PlanarQuant trap) | MSE explodes ~80× | `test_inverse_rotation_trap_iso` |
| Wrong inverse formula for full mode | Subtle: forgetting that `(q_L · v · conj(q_R))⁻¹ = q_R · conj(v_rot) · ...` etc. | Test inverts both `q_L` and `q_R` carefully — fast and full both pass `inverse(forward(x)) == x` to 1e-5 |

### The full-mode inverse, derived

Forward: `v_rot = q_L · v · conj(q_R)`

To invert, left-multiply by `conj(q_L)` and right-multiply by `q_R`:
```
conj(q_L) · v_rot · q_R = conj(q_L) · q_L · v · conj(q_R) · q_R
                        = (1) · v · (1)
                        = v
```
because `conj(q_L) · q_L = 1` and `conj(q_R) · q_R = 1` for unit quaternions.

So `inverse_full(v_rot) = conj(q_L) · v_rot · q_R` — **note the right side is
`q_R`, not `conj(q_R)`**. Easy to reverse this and produce a different rotation
by accident.

## Files touched

- `turboquant/isoquant.py` — new, 145 lines.
- `tests/test_isoquant.py` — new, 6 tests.
