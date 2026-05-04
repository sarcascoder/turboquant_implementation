# Phase 11 — llama.cpp Integration (Plan-of-Record)

## Why this is a writeup, not code

Per the original plan: *"For Phase 11, expect 2-3x more time than the Python
phases combined. C++ template machinery in ggml is dense."*

This phase is the production deployment path — production LLM serving with
low memory on RTX 5090 / Apple Silicon. It requires:
- Forking llama.cpp at a recent commit
- Adding 4 new ggml types and ~9 new files
- Hooking into the FA (flash-attention) template machinery
- C++/CUDA proficiency

I'm leaving this as a **detailed plan** that a C++ engineer can pick up and
ship. All the algorithmic decisions are settled by Phases 1–10. Phase 11 is
"port the Python prototype to C++/CUDA + work the build system".

---

## Files to add to a llama.cpp fork

| File | Purpose | Approx LOC |
|---|---|---:|
| `ggml/src/ggml-cuda/planar-iso-constants.cuh` | Static `__constant__` rotation params (Givens cos/sin, quaternions, centroids). **Per-TU copies — avoid `extern __constant__`** (the plan specifically warns against this; cross-TU constant references are fragile). | ~60 |
| `ggml/src/ggml-cuda/set-rows-planar-iso.cuh` | Device quantize functions for V cache during `set_rows`. Used by the `V_new` path each decode step. | ~120 |
| `ggml/src/ggml-cuda/cpy-planar-iso.cu` | Bulk F16→quantized for K cache **deferred conversion** (the Phase 5 transition). One-shot kernel that runs once after prefill ends. | ~80 |
| `ggml/src/ggml-cuda/fattn-common.cuh` | Flash-attention kernels: `vec_dot_KQ` (K dequant + Q dot, fused) and `dequantize_V` (with **inverse rotation** — Phase 6 trap). | ~250 |
| `ggml/src/ggml-cuda/dequantize.cuh` | K dequantize for non-FA paths. | ~80 |
| `ggml/src/ggml-cuda/fattn.cu` | FA kernel dispatch (`FATTN_VEC_CASES_ALL_D` macro instantiations for new types). | ~40 |
| `ggml/src/ggml-cuda/CMakeLists.txt` | Template instance file list — must add new `.cu` files **explicitly** (no glob). Missing entries → linker errors at runtime. | ~15 lines |
| `src/llama-kv-cache.cpp` | Double-buffer allocation, deferred conversion trigger, V zero-padding. | ~180 |
| `ggml/include/ggml.h` | New type enum entries: `GGML_TYPE_PLANAR3`, `GGML_TYPE_ISO3`, `GGML_TYPE_PLANAR4`, `GGML_TYPE_ISO4`. | ~10 |

**Total new code: ~835 LOC + build system updates.**

## New cache type identifiers

```c
// ggml/include/ggml.h
enum ggml_type {
    // ... existing ...
    GGML_TYPE_PLANAR3 = NN,    // 3-bit Planar (K cache or V cache)
    GGML_TYPE_ISO3    = NN+1,  // 3-bit Iso (default for symmetric configs)
    GGML_TYPE_PLANAR4 = NN+2,  // 4-bit Planar
    GGML_TYPE_ISO4    = NN+3,  // 4-bit Iso (production default at 4-bit)
    GGML_TYPE_COUNT
};
```

These pair with the existing `GGML_TYPE_F16` and `GGML_TYPE_Q8_0` for
asymmetric configs (e.g., `--cache-type-k iso3 --cache-type-v f16`).

## Critical implementation notes (from the original plan)

1. **Static `__constant__` arrays per translation unit** — avoid
   `extern __constant__` cross-TU references. Keep rotation params in the
   header `planar-iso-constants.cuh` and `#include` it everywhere.

2. **Deferred conversion** (Phase 5 in C++):
   - K cache allocates as F16 first.
   - After prefill, a CUDA kernel copies F16 → quantized format in bulk.
   - The transition is one kernel launch per layer, single-pass over the K
     buffer. Free the F16 buffer immediately after.

3. **V zero-padding** — V cache must be padded to aligned boundary (typically
   16 for Tensor Cores). **Pad with zeros, not random data** — softmax-weighted
   sum of "junk" V values would corrupt the attention output.

4. **FA template instances** — each `(K_type, V_type)` combination needs its
   own template instantiation. CMake must list each one **explicitly**:
   ```cmake
   set(FATTN_INSTANCES
       fattn-vec-f16-f16-d128.cu
       fattn-vec-iso3-iso3-d128.cu        # NEW
       fattn-vec-iso3-f16-d128.cu         # NEW (asymmetric)
       fattn-vec-planar3-planar3-d128.cu  # NEW
       # ... one per (K, V, D) tuple
   )
   ```

5. **Inverse rotation in V dequant** — both `fattn-common.cuh` and
   `dequantize.cuh` must apply the inverse rotation for V. **Don't share code
   with K dequant.** Phase 6's `assert_inverse_correct` from the Python side
   doesn't translate to C++ — instead, keep K and V dequant in separate
   functions with explicit comments pointing at this doc.

## Validation targets

The plan's RTX 5090 + Llama 3.1 8B numbers:

```bash
# wikitext-2 perplexity (target)
llama-perplexity --cache-type-k iso3 --cache-type-v iso3 -f wiki.test.raw
# Expected: PPL = 6.91

# decode throughput (target)
llama-bench --model llama-3.1-8b.gguf --cache-type-k iso3 --cache-type-v iso3
# Expected: ≥118 tok/s

# Needle-in-haystack at 8K, 32K, 65K context
# Must pass at all three lengths.
```

| Config | Decode tok/s | Prefill tok/s | PPL (wiki-2) | Compression |
|---|---:|---:|---:|---:|
| FP16 baseline | 140 | 6,156 | 6.63 | 1× |
| `iso3 / iso3` | **118** | **3,397** | **6.91** | **10.3×** |
| `planar3 / planar3` | **119** | **3,822** | **7.05** | **10.3×** |
| `planar3 / f16` (K-only) | 134 | — | ~6.63 | 5.1× |

## Mapping from this Python codebase to llama.cpp files

| llama.cpp file | This codebase reference |
|---|---|
| `planar-iso-constants.cuh` | `turboquant/planarquant.py` and `turboquant/isoquant.py` (the `register_buffer` calls) |
| `set-rows-planar-iso.cuh` | `turboquant/kernels/triton_planar.py:planar_quantize_only` |
| `cpy-planar-iso.cu` | `turboquant/deferred_cache.py:finalize_prefill` |
| `fattn-common.cuh` (`vec_dot_KQ`) | `turboquant/kernels/fused_planar_attn.py:fused_planar_quantize_attend` |
| `fattn-common.cuh` (`dequantize_V`) | `turboquant/symmetric_cache.py:assert_inverse_correct` (the V case) |
| `dequantize.cuh` | `turboquant/planarquant.py:PlanarQuant.dequantize` |
| `llama-kv-cache.cpp` | `turboquant/deferred_cache.py:DeferredQuantCache` |

A C++ engineer can read each Python file as the **executable specification**
for the corresponding C++ kernel.

## Pitfalls (from the plan, all the ones I avoided in Python because they were Python-specific)

| Pitfall | C++ specific | Mitigation |
|---|---|---|
| Forgetting to add new `.cu` files to CMakeLists | Yes — runtime linker errors, not build errors | Explicit instance list in CMakeLists; CI grep for new files |
| Cross-TU `extern __constant__` | Yes — fragile linking | Per-TU static `__constant__` (header-only) |
| Missing `dequantize_V` inverse rotation | Same as Phase 6 — silent PPL collapse | Sentinel comment + explicit C++ unit test before merging |
| Symmetric configs missing template instances for all (K, V) pairs | Yes — dispatch crashes at prefill | CI: enumerate cache-type combinations and spot-test each |

## Recommendation

Don't start Phase 11 until:

1. ✅ Phases 1–10 in Python are validated (you have them now).
2. Phase 7's Triton kernels are validated on a CUDA box (currently fallback-only).
3. Phase 8's fused attention kernel has its full Triton body filled in
   (currently a sketch in `fused_planar_attn.py`).
4. The Python prototype shows the plan's PPL and decode-speed targets on
   Qwen 3.5-0.8B or Llama 3.1 8B end-to-end.

Once those are green, Phase 11 is mechanical: port the validated kernels to
CUDA C++, wire them through ggml's type system and FA dispatch.

## Files touched

None in C++. This is a writeup. The mapping above shows the C++ engineer
exactly what each new file should implement, using the corresponding Python
file as the executable specification.
