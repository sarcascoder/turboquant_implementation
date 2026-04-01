# TurboQuant Implementation

A from-scratch PyTorch implementation of [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (ICLR 2026, Google Research) for **KV cache compression** in LLM inference.

## Features

- **MSE-only quantization** (no QJL) — community-validated best approach for KV cache
- **Asymmetric K/V bit allocation** — e.g., Keys at 4-bit, Values at 2-bit
- **Residual window** — recent tokens kept in full precision for quality
- **Qwen 3.5 support** — handles hybrid linear-attention + full-attention architecture
- **Pure PyTorch** — works on CUDA, MPS (Apple Silicon), and CPU

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquant.attention_patch import enable_turboquant
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-0.8B", dtype=torch.float32, trust_remote_code=True
).to("mps")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)

# Enable TurboQuant — one line!
enable_turboquant(model, key_bits=4, value_bits=2, residual_window=64)

# Generate as normal
inputs = tokenizer("Hello!", return_tensors="pt").to("mps")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Results

### MSE Distortion (10K random unit vectors, d=128)

| Bits | Empirical MSE | Ratio to Lower Bound | Paper Guarantee | Status |
|------|--------------|---------------------|-----------------|--------|
| 1 | 0.3611 | 1.444× | ≤2.7× | ✅ |
| 2 | 0.1161 | 1.857× | ≤2.7× | ✅ |
| 3 | 0.0340 | 2.173× | ≤2.7× | ✅ |
| 4 | 0.0093 | 2.386× | ≤2.7× | ✅ |

### Qwen 3.5-0.8B on Apple M4

| Config | Throughput | KV Compression |
|--------|-----------|----------------|
| Baseline (FP32) | 10.8 tok/s | 1× |
| TurboQuant K4/V2 | 10.1 tok/s | ~5× |

~7% latency overhead for ~5× KV cache compression.

## Documentation

See [WALKTHROUGH.md](WALKTHROUGH.md) for the full implementation walkthrough, challenges, and detailed benchmarks.

## Install

```bash
pip install -r requirements.txt
```

## License

MIT
