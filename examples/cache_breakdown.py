"""
Detailed breakdown of where KV cache memory actually lives in Qwen 3.5-0.8B
and how much each cache strategy saves on the FULL-ATTENTION portion (the
only part TurboQuant / RotorQuant compresses).

Qwen 3.5 hybrid architecture:
  - 18 'linear_attention' layers — store conv_states + recurrent_states (FIXED size)
  - 6  'full_attention'  layers — store K, V (GROWS with context)

Only the 6 full-attention layers' K/V are eligible for quantization.
Linear-attention state is per-architecture-fixed and not touched.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from turboquant.attention_patch import enable_turboquant
from turboquant.rotorquant_kv_cache import enable_rotorquant


def cache_breakdown(cache, label):
    """Print per-layer-type bytes and totals."""
    layer_types = cache.layer_types if hasattr(cache, "layer_types") else []
    full_idx = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    lin_idx  = [i for i, t in enumerate(layer_types) if t != "full_attention"]

    full_bytes = 0
    lin_bytes  = 0

    # Standard buffers (FP precision)
    for i in range(len(layer_types)):
        for buf_name in ("key_cache", "value_cache"):
            buf = getattr(cache, buf_name)[i] if hasattr(cache, buf_name) else None
            if buf is not None:
                b = buf.element_size() * buf.numel()
                if i in full_idx: full_bytes += b
                else:             lin_bytes  += b
        for buf_name in ("conv_states", "recurrent_states"):
            buf = getattr(cache, buf_name)[i] if hasattr(cache, buf_name) else None
            if buf is not None:
                b = buf.element_size() * buf.numel()
                lin_bytes += b   # linear-attn state always

    # Quantized stores (TurboQuant)
    if hasattr(cache, "_k_quantized"):
        for i, kq in enumerate(cache._k_quantized):
            if kq is not None:
                full_bytes += kq[0].element_size() * kq[0].numel()
                full_bytes += kq[1].element_size() * kq[1].numel()
        for i, vq in enumerate(cache._v_quantized):
            if vq is not None:
                full_bytes += vq[0].element_size() * vq[0].numel()
                full_bytes += vq[1].element_size() * vq[1].numel()

    # Quantized stores (RotorQuant)
    if hasattr(cache, "_caches"):
        for i, c in cache._caches.items():
            if c._quant_K is not None:
                full_bytes += c._quant_K[0].element_size() * c._quant_K[0].numel()
                full_bytes += c._quant_K[1].element_size() * c._quant_K[1].numel()
            if c._quant_V is not None:
                full_bytes += c._quant_V[0].element_size() * c._quant_V[0].numel()
                full_bytes += c._quant_V[1].element_size() * c._quant_V[1].numel()
            if c._fp_V is not None:
                full_bytes += c._fp_V.element_size() * c._fp_V.numel()
            for buf in c._prefill_K + c._prefill_V:
                full_bytes += buf.element_size() * buf.numel()

    print(f"\n  {label}")
    print(f"    Linear-attention state (NOT compressed): {lin_bytes / 1024:>8.1f} KiB  ({len(lin_idx)} layers)")
    print(f"    Full-attention KV (compressed by us):    {full_bytes / 1024:>8.1f} KiB  ({len(full_idx)} layers)")
    print(f"    Total:                                   {(lin_bytes + full_bytes) / 1024:>8.1f} KiB")
    return full_bytes, lin_bytes


def run_one(model, tokenizer, device, prompt, label, setup):
    """Setup is a no-arg fn that enables the desired quant on `model`. Returns cache instance."""
    setup()
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)

    # Capture the cache instance via from_config patching
    cache_holder = {}
    if "TurboQuant" in label:
        from turboquant.kv_cache import TurboQuantKVCache as cls
    elif "RotorQuant" in label:
        from turboquant.rotorquant_kv_cache import RotorQuantKVCache as cls
    else:
        cls = None

    if cls is not None:
        orig = cls.from_config
        @staticmethod
        def patched(config, **kw):
            inst = orig(config, **kw)
            cache_holder["c"] = inst
            return inst
        cls.from_config = patched
        try:
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=8, do_sample=False)
        finally:
            cls.from_config = orig
    else:
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=8, do_sample=False)

    return cache_breakdown(cache_holder.get("c"), label) if cache_holder else (0, 0)


def main():
    model_name = "Qwen/Qwen3.5-0.8B"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading {model_name} on {device}…")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, trust_remote_code=True
    ).to(device).eval()

    # Generous prompt to make full-attention K/V cache visible
    prompt = ("Explain in detail the history of the Roman Empire from its founding "
              "in 753 BC to the fall of the Western Empire in 476 AD. Include key emperors, "
              "wars, and cultural achievements. Be thorough.") * 3   # ≈400 tokens

    print(f"\n  Prompt length (tokens): "
          f"{tokenizer(prompt, return_tensors='pt')['input_ids'].shape[-1]}")

    print("\n" + "█" * 78)
    print("  KV CACHE BREAKDOWN — by layer type")
    print("█" * 78)

    # Note: BASELINE doesn't expose a captureable cache class easily — skip.
    full_tq, _ = run_one(model, tokenizer, device, prompt,
                         "TurboQuant K=4/V=2 (existing)",
                         lambda: enable_turboquant(model, key_bits=4, value_bits=2,
                                                    residual_window=64))
    from turboquant.attention_patch import disable_turboquant
    disable_turboquant(model)

    full_rp, _ = run_one(model, tokenizer, device, prompt,
                         "RotorQuant Planar K=3/V=3 (Phase 3+5+6)",
                         lambda: enable_rotorquant(model, key_bits=3, value_bits=3,
                                                    quantizer_kind="planar"))
    from turboquant.rotorquant_kv_cache import disable_rotorquant
    disable_rotorquant(model)

    full_ri, _ = run_one(model, tokenizer, device, prompt,
                         "RotorQuant Iso K=4/V=4 (Phase 4+5+6)",
                         lambda: enable_rotorquant(model, key_bits=4, value_bits=4,
                                                    quantizer_kind="iso"))
    disable_rotorquant(model)

    # Compute baseline FP32 KV size for the 6 full-attention layers analytically
    text_config = model.config.text_config if hasattr(model.config, "text_config") else model.config
    head_dim = getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads)
    n_kv_heads = getattr(text_config, "num_key_value_heads", text_config.num_attention_heads)
    n_full_layers = sum(1 for t in text_config.layer_types if t == "full_attention")
    seq_len = tokenizer(prompt, return_tensors="pt")['input_ids'].shape[-1]
    bytes_per_elem_fp32 = 4
    baseline_full = 2 * n_full_layers * n_kv_heads * seq_len * head_dim * bytes_per_elem_fp32

    print("\n" + "█" * 78)
    print("  COMPRESSION RATIO (full-attention KV only)")
    print("█" * 78)
    print(f"\n  Analytical FP32 baseline (6 full-attn layers, {seq_len} tokens): "
          f"{baseline_full / 1024:.1f} KiB")
    print(f"\n  {'Config':<46} {'KiB':>8} {'compression':>12}")
    print("  " + "─" * 70)
    for label, b in [("TurboQuant  K=4/V=2 (3-bit avg)", full_tq),
                      ("RotorQuant  Planar K=3/V=3", full_rp),
                      ("RotorQuant  Iso    K=4/V=4", full_ri)]:
        ratio = baseline_full / b if b else 0
        print(f"  {label:<46} {b/1024:>8.1f} {ratio:>11.2f}×")
    print("  " + "─" * 70)


if __name__ == "__main__":
    main()
