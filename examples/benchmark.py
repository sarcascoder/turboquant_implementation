"""
Benchmark script: compares baseline (no compression) vs TurboQuant KV cache.
Measures generation time, memory, and output quality side-by-side.
"""
import torch
import time
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquant.attention_patch import enable_turboquant, disable_turboquant


def measure_generation(model, tokenizer, device, prompt, max_new_tokens=200, label=""):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)

    # Warm-up (1 token)
    with torch.no_grad():
        _ = model(**{k: v for k, v in inputs.items()}, use_cache=True)

    torch.mps.synchronize() if device == "mps" else None

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for reproducibility
        )
    torch.mps.synchronize() if device == "mps" else None
    elapsed = time.time() - start

    gen_tokens = outputs.shape[-1] - inputs.input_ids.shape[-1]
    tps = gen_tokens / elapsed
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Tokens generated : {gen_tokens}")
    print(f"  Time             : {elapsed:.2f}s")
    print(f"  Throughput       : {tps:.1f} tokens/sec")
    print(f"{'='*60}")
    print(response[:500])
    print(f"{'='*60}\n")

    return {
        "label": label,
        "tokens": gen_tokens,
        "time_s": round(elapsed, 2),
        "tokens_per_sec": round(tps, 1),
        "response": response,
    }


def main():
    model_name = "Qwen/Qwen3.5-0.8B"
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, trust_remote_code=True
    ).to(device)

    prompt = "List the top 5 tourist attractions in Rome and explain why each is famous."

    # ── Baseline (no compression) ──
    print("\n>>> Running BASELINE (full-precision KV cache)...")
    baseline = measure_generation(model, tokenizer, device, prompt, label="BASELINE (FP32 KV Cache)")

    # ── TurboQuant K4/V2 ──
    enable_turboquant(model, key_bits=4, value_bits=2, residual_window=64)
    print(">>> Running TURBOQUANT K4/V2 (3-bit avg)...")
    tq_k4v2 = measure_generation(model, tokenizer, device, prompt, label="TURBOQUANT K4/V2 (3-bit avg)")
    disable_turboquant(model)

    # ── TurboQuant K4/V4 ──
    enable_turboquant(model, key_bits=4, value_bits=4, residual_window=64)
    print(">>> Running TURBOQUANT K4/V4 (4-bit avg)...")
    tq_k4v4 = measure_generation(model, tokenizer, device, prompt, label="TURBOQUANT K4/V4 (4-bit avg)")
    disable_turboquant(model)

    # ── Summary table ──
    print("\n" + "="*70)
    print(f"{'Config':<30} {'Tokens':>8} {'Time(s)':>8} {'Tok/s':>8}")
    print("-"*70)
    for r in [baseline, tq_k4v2, tq_k4v4]:
        print(f"{r['label']:<30} {r['tokens']:>8} {r['time_s']:>8} {r['tokens_per_sec']:>8}")
    print("="*70)


if __name__ == "__main__":
    main()
