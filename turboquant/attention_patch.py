import torch
from turboquant.kv_cache import TurboQuantKVCache


def enable_turboquant(model, key_bits: int = 4, value_bits: int = 2, residual_window: int = 128):
    """
    Configures a HuggingFace model to use TurboQuant KV Cache compression.
    Works with Qwen 3.5 and other models.
    """
    original_generate = model.generate

    def generate_with_turboquant(*args, **kwargs):
        device = next(model.parameters()).device
        cache = TurboQuantKVCache.from_config(
            model.config,
            key_bits=key_bits,
            value_bits=value_bits,
            residual_window=residual_window,
            device=str(device),
        )
        kwargs["past_key_values"] = cache
        kwargs["use_cache"] = True
        return original_generate(*args, **kwargs)

    model.generate = generate_with_turboquant
    model._turboquant_original_generate = original_generate
    return model


def disable_turboquant(model):
    if hasattr(model, "_turboquant_original_generate"):
        model.generate = model._turboquant_original_generate
        del model._turboquant_original_generate
    return model
