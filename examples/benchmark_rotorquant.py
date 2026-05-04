"""
End-to-end benchmark on Qwen 3.5-0.8B comparing four cache configurations:

1. BASELINE                — full-precision FP32 KV cache (no compression)
2. TURBOQUANT K4/V2       — existing TurboQuant (33K FMAs/dequant, residual_window=64)
3. ROTORQUANT planar3/3   — Phase 3 + 5 + 6: PlanarQuant 3-bit K & V, post-prefill
4. ROTORQUANT iso4/4      — Phase 4 + 5 + 6: IsoQuant 4-bit K & V, post-prefill

Measures (per config):
- Generation latency and throughput
- KV cache size after prefill (bytes)
- Output text (greedy decoding for reproducibility)
- Token-level agreement vs baseline (first N tokens equal)
"""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from turboquant.attention_patch import enable_turboquant, disable_turboquant
from turboquant.rotorquant_kv_cache import enable_rotorquant, disable_rotorquant


PROMPT = (
    "You are an experienced travel writer. List the top 5 tourist attractions "
    "in Rome and explain in 2-3 sentences why each is famous. Be specific and concrete."
)
MAX_NEW_TOKENS = 120


def kv_cache_bytes(cache) -> int:
    """Approximate the in-memory bytes of a cache after prefill."""
    total = 0
    # Standard buffers (linear-attention layers + any FP16 K/V remainder)
    for layer_idx in range(len(cache.layer_types) if hasattr(cache, "layer_types") else 0):
        if hasattr(cache, "key_cache"):
            k = cache.key_cache[layer_idx] if cache.key_cache[layer_idx] is not None else None
            v = cache.value_cache[layer_idx] if cache.value_cache[layer_idx] is not None else None
            if k is not None:
                total += k.element_size() * k.numel()
            if v is not None:
                total += v.element_size() * v.numel()
        if hasattr(cache, "conv_states"):
            cs = cache.conv_states[layer_idx]
            rs = cache.recurrent_states[layer_idx]
            if cs is not None:
                total += cs.element_size() * cs.numel()
            if rs is not None:
                total += rs.element_size() * rs.numel()

    # Quantized stores (TurboQuant or RotorQuant)
    if hasattr(cache, "_k_quantized"):  # legacy TurboQuant cache
        for kq in cache._k_quantized:
            if kq is not None:
                total += kq[0].element_size() * kq[0].numel()  # packed indices
                total += kq[1].element_size() * kq[1].numel()  # norms
        for vq in cache._v_quantized:
            if vq is not None:
                total += vq[0].element_size() * vq[0].numel()
                total += vq[1].element_size() * vq[1].numel()

    if hasattr(cache, "_caches"):  # new RotorQuant cache
        for c in cache._caches.values():
            if c._quant_K is not None:
                total += c._quant_K[0].element_size() * c._quant_K[0].numel()
                total += c._quant_K[1].element_size() * c._quant_K[1].numel()
            if c._quant_V is not None:
                total += c._quant_V[0].element_size() * c._quant_V[0].numel()
                total += c._quant_V[1].element_size() * c._quant_V[1].numel()
            if c._fp_V is not None:
                total += c._fp_V.element_size() * c._fp_V.numel()
            for buf in c._prefill_K + c._prefill_V:
                total += buf.element_size() * buf.numel()
    return total


def measure(model, tokenizer, device, label: str, run_callable):
    """Run one configuration. run_callable(model, inputs, max_new_tokens) → outputs.
    Returns dict with timings, output, and (best-effort) cache bytes."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": PROMPT},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)

    if device == "mps":
        torch.mps.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs, cache_used = run_callable(model, inputs, MAX_NEW_TOKENS)
    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0

    gen_tokens = outputs.shape[-1] - inputs.input_ids.shape[-1]
    tps = gen_tokens / elapsed
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:],
                                 skip_special_tokens=True)
    cb = kv_cache_bytes(cache_used) if cache_used is not None else 0

    print(f"\n{'═' * 72}")
    print(f"  {label}")
    print(f"{'═' * 72}")
    print(f"  Tokens generated : {gen_tokens}")
    print(f"  Time             : {elapsed:.2f}s")
    print(f"  Throughput       : {tps:.2f} tok/s")
    print(f"  KV cache size    : {cb / 1024:.1f} KiB" if cb else "  KV cache size    : (not measured)")
    print(f"{'─' * 72}")
    print(f"  Output (first 350 chars):")
    print(f"  {response[:350]}")
    print(f"{'═' * 72}\n")

    return {
        "label": label,
        "tokens": gen_tokens,
        "time_s": round(elapsed, 2),
        "tps": round(tps, 2),
        "cache_bytes": cb,
        "response": response,
        "output_ids": outputs[0][inputs.input_ids.shape[-1]:].cpu().tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────── #
# Per-config run functions
# ─────────────────────────────────────────────────────────────────────────── #
def run_baseline(model, inputs, max_new_tokens):
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    return out, None  # default cache: hard to introspect bytes consistently


def run_turboquant(model, inputs, max_new_tokens, key_bits=4, value_bits=2):
    enable_turboquant(model, key_bits=key_bits, value_bits=value_bits, residual_window=64)
    try:
        # Capture the cache instance so we can measure bytes after generate
        from turboquant.kv_cache import TurboQuantKVCache
        captured = {}
        orig_from_config = TurboQuantKVCache.from_config

        @staticmethod
        def patched(config, **kw):
            inst = orig_from_config(config, **kw)
            captured["cache"] = inst
            return inst

        TurboQuantKVCache.from_config = patched
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        TurboQuantKVCache.from_config = orig_from_config
        return out, captured.get("cache")
    finally:
        disable_turboquant(model)


def run_rotorquant(model, inputs, max_new_tokens, key_bits=3, value_bits=3, kind="planar"):
    enable_rotorquant(model, key_bits=key_bits, value_bits=value_bits, quantizer_kind=kind)
    try:
        from turboquant.rotorquant_kv_cache import RotorQuantKVCache
        captured = {}
        orig_from_config = RotorQuantKVCache.from_config

        @staticmethod
        def patched(config, **kw):
            inst = orig_from_config(config, **kw)
            captured["cache"] = inst
            return inst

        RotorQuantKVCache.from_config = patched
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        RotorQuantKVCache.from_config = orig_from_config
        return out, captured.get("cache")
    finally:
        disable_rotorquant(model)


# ─────────────────────────────────────────────────────────────────────────── #
# Main
# ─────────────────────────────────────────────────────────────────────────── #
def main():
    model_name = "Qwen/Qwen3.5-0.8B"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading {model_name} on {device}…")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, trust_remote_code=True
    ).to(device)
    model.eval()

    results = []
    print("\n" + "█" * 72)
    print("  BENCHMARK START — Qwen 3.5-0.8B, greedy decoding")
    print("█" * 72)

    # 1. Baseline
    results.append(measure(model, tokenizer, device, "BASELINE — FP32 KV cache (no compression)",
                           run_baseline))

    # 2. TurboQuant (existing)
    results.append(measure(model, tokenizer, device,
                           "TurboQuant K=4/V=2 — existing impl, residual_window=64",
                           lambda m, i, n: run_turboquant(m, i, n, 4, 2)))

    # 3. RotorQuant Planar 3-bit (Phase 3 + 5 + 6)
    results.append(measure(model, tokenizer, device,
                           "RotorQuant PlanarQuant K=3/V=3 — Phase 3+5+6",
                           lambda m, i, n: run_rotorquant(m, i, n, 3, 3, "planar")))

    # 4. RotorQuant Iso 4-bit (Phase 4 + 5 + 6)
    results.append(measure(model, tokenizer, device,
                           "RotorQuant IsoQuant K=4/V=4 — Phase 4+5+6",
                           lambda m, i, n: run_rotorquant(m, i, n, 4, 4, "iso")))

    # ── Summary table ──
    print("\n" + "█" * 84)
    print("  SUMMARY")
    print("█" * 84)
    base = results[0]
    print(f"\n  {'Config':<48} {'tok/s':>8} {'time(s)':>8} {'KV (KiB)':>10} {'agree':>8}")
    print("  " + "─" * 82)
    for r in results:
        # Token agreement with baseline (first M tokens)
        ids = r["output_ids"]
        b_ids = base["output_ids"]
        m = min(len(ids), len(b_ids))
        agree = sum(1 for a, b in zip(ids[:m], b_ids[:m]) if a == b)
        agree_pct = (agree / max(m, 1)) * 100 if m > 0 else 0
        kib = f"{r['cache_bytes'] / 1024:.1f}" if r['cache_bytes'] > 0 else "—"
        print(f"  {r['label'][:48]:<48} {r['tps']:>8.2f} {r['time_s']:>8.2f} {kib:>10} {agree_pct:>7.1f}%")
    print("  " + "─" * 82)
    print()


if __name__ == "__main__":
    main()
