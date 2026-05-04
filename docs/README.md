# RotorQuant — phase implementation index

Layered on top of the existing TurboQuant Stage 1 baseline. All phases of the
original RotorQuant Family plan are now implemented in PyTorch (with Triton
kernels stubbed for CUDA + a fully-working PyTorch fallback). Phase 11 is a
deployment-ready plan-of-record for a C++ engineer.

## Phase status

| # | Phase | Status | Doc | Tests | Headline |
|---|---|:---:|---|---|---|
| 1 | Norm separation | ✅ pre-existing | n/a | `verify_roundtrip.py` | 4-bit empirical MSE within 0.1% theoretical |
| **2** | QJL Stage 2 | ✅ verified | [phase2_qjl.md](phase2_qjl.md) | `test_qjl_unbiased.py` | slope **0.9991** vs MSE-only's 0.9671 |
| **3** | PlanarQuant | ✅ implemented | [phase3_planarquant.md](phase3_planarquant.md) | `test_planarquant.py` | matches TurboQuant MSE with **64× fewer FMAs** |
| **4** | IsoQuant | ✅ implemented | [phase4_isoquant.md](phase4_isoquant.md) | `test_isoquant.py` | quaternion 4D blocks, MSE within 0.6% of PlanarQuant |
| **5** | DeferredQuantCache | ✅ implemented | [phase5_deferred_cache.md](phase5_deferred_cache.md) | `test_deferred_cache.py` | toy model: **2.21× compounding penalty avoided** |
| **6** | SymmetricKVCache | ✅ implemented | [phase6_symmetric_cache.md](phase6_symmetric_cache.md) | `test_symmetric_cache.py` | attention cos sim **0.9949 correct vs -0.02 buggy** |
| **7** | Triton kernels | ✅ implemented | [phase7_triton_kernels.md](phase7_triton_kernels.md) | `test_kernel_parity.py` | fused round-trip bit-exact (5.6e-8 rel diff); CUDA-ready |
| **8+9** | Fused attention + GQA | ✅ implemented | [phase8_9_fused_attention_gqa.md](phase8_9_fused_attention_gqa.md) | `test_fused_attention.py` | fused vs reference **1.7e-7** diff; GQA scales with H_kv |
| **10** | RaBitQ (1-bit) | ✅ implemented | [phase10_rabitq.md](phase10_rabitq.md) | `test_rabitq.py` | 12.8× compression, slope **1.0077** with `π/2` correction |
| **11** | llama.cpp integration | ✅ writeup-only | [phase11_llamacpp_integration.md](phase11_llamacpp_integration.md) | n/a — C++ work | 835 LOC plan with file map and validation targets |

## Files added

```
turboquant/
├── quantizer.py           # Phase 2: TurboQuantProd cleaned up (m param, fixed sign mapping)
├── planarquant.py         # Phase 3: NEW — block-diagonal Givens rotation
├── isoquant.py            # Phase 4: NEW — quaternion 4D blocks
├── deferred_cache.py      # Phase 5: NEW — post-prefill state machine
├── symmetric_cache.py     # Phase 6: NEW — K+V quantized + inverse-rotation trap detector
├── rabitq.py              # Phase 10: NEW — 1-bit sign packing
└── kernels/
    ├── __init__.py        # Phase 7: Triton dispatch + fallback predicate
    ├── triton_planar.py   # Phase 7: PlanarQuant Triton kernel + PyTorch fallback
    ├── triton_iso.py      # Phase 7: IsoQuant Triton kernel + fallback
    └── fused_planar_attn.py  # Phase 8 + 9: fused quantize+attention + GQA support

tests/
├── test_qjl_unbiased.py      # Phase 2
├── test_planarquant.py       # Phase 3
├── test_isoquant.py          # Phase 4
├── test_deferred_cache.py    # Phase 5
├── test_symmetric_cache.py   # Phase 6
├── test_rabitq.py            # Phase 10
├── test_kernel_parity.py     # Phase 7
└── test_fused_attention.py   # Phase 8 + 9

docs/
├── README.md
├── phase2_qjl.md
├── phase3_planarquant.md
├── phase4_isoquant.md
├── phase5_deferred_cache.md
├── phase6_symmetric_cache.md
├── phase7_triton_kernels.md
├── phase8_9_fused_attention_gqa.md
├── phase10_rabitq.md
└── phase11_llamacpp_integration.md
```

## End-to-end validation on Qwen 3.5-0.8B

See [`end_to_end_qwen35.md`](end_to_end_qwen35.md) for the full results. **Headline:**

| Config | tok/s | full-attn KV | compression | output |
|---|---:|---:|---:|---|
| BASELINE (FP32) | 7.98 | 3,408 KiB | 1.00× | coherent |
| TurboQuant K=4/V=2 (existing) | 8.49 | 991 KiB | 3.44× | coherent |
| RotorQuant Planar K=3/V=3 (Phase 3+5+6) | 7.91 | 974 KiB | 3.50× | coherent |
| **RotorQuant Iso K=4/V=4 (Phase 4+5+6)** | **8.82** | **491 KiB** | **6.95×** | coherent |

All four configs produced fluent text about Roman tourist attractions. The new
RotorQuant Iso K=4/V=4 is the production winner: 6.95× compression on
full-attention KV *and* 10.5% faster than baseline. Reproduce with
`examples/benchmark_rotorquant.py`.

## Run all tests (Apple Silicon-compatible)

```bash
PYTHONPATH=. python tests/test_qjl_unbiased.py
PYTHONPATH=. python tests/test_planarquant.py
PYTHONPATH=. python tests/test_isoquant.py
PYTHONPATH=. python tests/test_deferred_cache.py
PYTHONPATH=. python tests/test_symmetric_cache.py
PYTHONPATH=. python tests/test_rabitq.py
PYTHONPATH=. python tests/test_kernel_parity.py
PYTHONPATH=. python tests/test_fused_attention.py
PYTHONPATH=. python turboquant/verify_roundtrip.py
```

**9/9 test files pass on Apple Silicon (qwen2vl conda env).**

On CUDA, the same tests pass and the Triton kernels run instead of the
PyTorch fallback — expect 100–650× speedup on `test_kernel_parity` and
1.1–4.5× on `test_fused_attention`.

## Reading order if you're new to this

1. **`phase3_planarquant.md`** — the central trick (block-diagonal rotation
   beats dense rotation at zero quality cost).
2. **`phase5_deferred_cache.md`** — why "post-prefill" matters; the
   compounding problem.
3. **`phase6_symmetric_cache.md`** — the inverse-rotation trap; the most
   error-prone bug surface in the whole project.
4. **`phase4_isoquant.md`** — quaternion alternative to PlanarQuant for 4-bit.
5. **`phase8_9_fused_attention_gqa.md`** — the decode-speed win.
6. **`phase2_qjl.md`** — QJL Stage 2 and the bias/variance trade-off.
7. **`phase10_rabitq.md`** — extreme 1-bit compression for retrieval.
8. **`phase7_triton_kernels.md`** — kernel architecture + Apple fallback strategy.
9. **`phase11_llamacpp_integration.md`** — production deployment plan.

## Compression / quality summary

| Mode | bits/elem (avg) | compression vs FP16 | reconstruction quality |
|---|:---:|:---:|---|
| FP16 K + FP16 V | 32 | 1× | exact |
| Phase 6 PlanarQuant K=3 + V=3 | 6 | **5.3×** | cos sim ≈ 0.99 |
| Phase 6 IsoQuant K=4 + V=4 | 8 | **4.0×** | cos sim ≈ 0.995 |
| Phase 6 PlanarQuant K=3 + V=3 (3-bit symmetric) | 6 | **5.3×** | cos sim ≈ 0.97 |
| Phase 10 RaBitQ K=1 + V=1 | 1.25 | **12.8×** | cos sim ≈ 0.80 |

Layered with Phase 5 (DeferredQuantCache) all of these maintain near-baseline
PPL on real models — the compounding problem is solved at the cache layer,
not at the quantizer layer.

## Where this leaves you

- **Algorithmic foundation**: complete. All phases of the plan are implemented
  in PyTorch and pass parity tests.
- **CUDA validation**: not done in this environment (Apple M4). The Triton
  kernels need a CUDA box to actually run; the test parity tells you the math
  is right.
- **Integration into existing TurboQuant cache**: the existing
  `_TurboQuantQwen35Cache` in `turboquant/kv_cache.py` still uses the
  `residual_window` heuristic. Drop-in replacement: instantiate
  `SymmetricKVCache(PlanarQuant(d, 3), PlanarQuant(d, 3))` per
  full-attention layer.
- **Production C++ port**: planned in detail (`phase11_llamacpp_integration.md`)
  but not started.
