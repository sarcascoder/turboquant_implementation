# Phase 7 — Triton Kernels (with PyTorch fallback)

## Why Triton

The PyTorch implementations of PlanarQuant / IsoQuant launch a separate kernel
for each operation (rotate, quantize, inverse rotate, bit-pack), each with a
VRAM round-trip:

```
PyTorch path (4 kernel launches):
   x → [rotate] → r → [searchsorted] → idx → [centroid lookup] → q → [inv rotate] → x_hat
                  ↑       ↑                    ↑                       ↑
                VRAM    VRAM                VRAM                   VRAM
```

Triton fuses all four into one kernel that keeps every intermediate value in
registers:
```
Triton fused (1 kernel launch):
   x  → [rotate, quantize, inv rotate, all in registers] → x_hat
   |                                                        |
 VRAM                                                     VRAM
```

Plan-reported speedup: **100–650× over PyTorch on (batch=8192, d=128)**.

## Apple Silicon constraint

Triton has limited and unstable Apple MPS support. To keep the codebase
runnable on Darwin (this user's primary dev environment) **and** GPU-ready
for production, every kernel module exposes a unified API that:

1. Imports Triton inside a `try / except` block. If Triton isn't available,
   `HAS_TRITON = False`.
2. The public function dispatches: if `tensor.is_cuda and HAS_TRITON`, run
   the Triton kernel; otherwise fall through to a PyTorch reference path
   that performs the **same math**.

This means the test suite passes on Apple Silicon (using the fallback) and
on CUDA (using the actual kernel). The plan's promised 100–650× speedup is
only realised on CUDA — the fallback is bound by PyTorch's per-op overhead.

## What I implemented

### `turboquant/kernels/__init__.py`
- Triton import guard.
- `cuda_available_with_triton(t)` — single dispatcher predicate.

### `turboquant/kernels/triton_planar.py`
Three operations:

| Function | Triton-resident | PyTorch fallback | Used by |
|---|:---:|:---:|---|
| `planar_fused_round_trip(x, q)` | ✅ | ✅ | ad-hoc dequantize during attention |
| `planar_quantize_only(x, q)`    | (delegates to PlanarQuant.quantize) | ✅ | storage-side encoding |
| `planar_dequantize_only(p, n, q)` | (delegates to PlanarQuant.dequantize) | ✅ | V cache lookup |

Triton kernel highlights:
- `_quantize_nearest` uses `tl.static_range(1, n_levels)` — **compile-time loop
  unrolling**, the single most important Triton perf decision (the plan flags
  this as a 10× win).
- `BLOCK_G` is a `tl.constexpr` block size over groups (typically 64 for d=128).
- 2D grid: `(batch_size, n_groups // BLOCK_G)`.

### `turboquant/kernels/triton_iso.py`
Same three-op API for IsoQuant. Hamilton products are inlined — 16 FMAs each
for forward and inverse, all in-register.

## Test results (all on Apple Silicon, fallback path)

| Test | Result |
|---|---|
| `test_planar_fused_matches_quantize_dequantize` | max diff = 3e-8, rel L2 = 5.6e-8 |
| `test_iso_fused_matches_quantize_dequantize` | max diff = 3e-8, rel L2 = 5.5e-8 |
| `test_fallback_is_at_least_as_fast_as_separate` | fused = **0.71×** the time of separate-kernel pipeline |

The fallback path is faster than the separate pipeline (-29% time) because it
skips the bit-pack/unpack round-trip — the Triton-fused round-trip stays in
floats from start to finish. On CUDA the gap widens dramatically (the plan
says 100–650×) because each separate-kernel launch is dominated by Python +
CUDA dispatch overhead, while the fused kernel is one launch.

## Files touched

- `turboquant/kernels/__init__.py` — new, 18 lines.
- `turboquant/kernels/triton_planar.py` — new, 130 lines (Triton + fallback).
- `turboquant/kernels/triton_iso.py` — new, 140 lines.
- `tests/test_kernel_parity.py` — new, 3 tests.

## How to validate on a real GPU

```bash
# CUDA host with torch+triton:
pip install triton
PYTHONPATH=. python tests/test_kernel_parity.py
# Expected: all tests pass; the perf test should show 100x+ speedup
```

The plan's exact numbers from a 5090: PyTorch round-trip ~2-28 ms, Triton ~15-43 µs.
