import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

from turboquant.attention_patch import enable_turboquant


def main():
    model_name = "Qwen/Qwen3.5-0.8B"

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Apple Silicon → MPS
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,  # MPS works best with float32 for Qwen3.5
        trust_remote_code=True,
    ).to(device)

    # Enable TurboQuant: K4/V2 = 3-bit average
    model = enable_turboquant(model, key_bits=4, value_bits=2, residual_window=64)
    print("TurboQuant enabled (K4/V2, residual_window=64)")

    # Prompt
    prompt = "List the top 5 tourist attractions in Rome and explain why each is famous."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(device)

    print(f"\nGenerating with compressed KV cache...")
    start = time.time()

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    elapsed = time.time() - start
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

    print(f"\nTime: {elapsed:.1f}s")
    print(f"Generated {outputs.shape[-1] - inputs.input_ids.shape[-1]} tokens")
    print("\n" + "=" * 60)
    print(response)
    print("=" * 60)


if __name__ == "__main__":
    main()
