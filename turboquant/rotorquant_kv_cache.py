"""
RotorQuant KV cache — production wrapper that integrates Phases 3+5+6 into
the Qwen 3.5 cache interface.

Replaces the existing _TurboQuantQwen35Cache (which uses TurboQuant's d×d
rotation + a residual_window heuristic) with:
  - PlanarQuant rotation (Phase 3)   — 64x fewer FMAs, same MSE
  - DeferredQuantCache (Phase 5)     — true post-prefill state machine
  - SymmetricKVCache (Phase 6)       — K + V both quantized, inverse-rotation safe

For Qwen 3.5's hybrid architecture (linear attention + full attention):
  - Linear attention layers: untouched (use conv_states / recurrent_states)
  - Full attention layers: use the new RotorQuant cache per layer
"""
from __future__ import annotations
import torch
from typing import Any, Optional, Tuple

from turboquant.planarquant import PlanarQuant
from turboquant.deferred_cache import DeferredQuantCache


class RotorQuantKVCache:
    """
    Drop-in replacement for the existing TurboQuantKVCache. Same factory API
    (.from_config) so the attention_patch can be reused.
    """

    @staticmethod
    def from_config(config, key_bits=3, value_bits=3, quantizer_kind="planar",
                     device="cpu", seed=0):
        text_config = config
        if hasattr(config, "text_config") and config.text_config is not None:
            text_config = config.text_config
        head_dim = getattr(text_config, "head_dim", None)
        if head_dim is None:
            head_dim = text_config.hidden_size // text_config.num_attention_heads
        try:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
            base_cache_cls = Qwen3_5DynamicCache
        except ImportError:
            from transformers.cache_utils import DynamicCache
            base_cache_cls = DynamicCache

        return _RotorQuantQwen35Cache(
            config=text_config,
            base_cache_cls=base_cache_cls,
            head_dim=head_dim,
            key_bits=key_bits,
            value_bits=value_bits,
            device=device,
            seed=seed,
            quantizer_kind=quantizer_kind,
        )


def _make_quantizer(kind: str, dim: int, bits: int, seed: int):
    """Factory: pick PlanarQuant or IsoQuant for K / V."""
    if kind == "planar":
        return PlanarQuant(dim=dim, bits=bits, seed=seed)
    if kind == "iso":
        from turboquant.isoquant import IsoQuant
        return IsoQuant(dim=dim, bits=bits, mode="fast", seed=seed)
    raise ValueError(f"unknown quantizer kind: {kind}")


class _RotorQuantQwen35Cache:
    """
    Per-layer DeferredQuantCache for Qwen 3.5 full-attention layers.

    Lifecycle of a single forward pass:
      - First call to update(layer_idx) for a token → bucket appends to the
        layer's prefill buffer (FP16, no quantization).
      - When the model transitions to decode (we detect by query length == 1
        on the first full-attention layer and prefill buffer is non-empty),
        we finalize prefill across ALL layers in one pass.
      - Subsequent decode steps use append_decode for each layer.
    """

    is_compileable = False

    def __init__(self, config, base_cache_cls, head_dim, key_bits, value_bits,
                 device, seed, quantizer_kind):
        self.layer_types = config.layer_types
        self.num_layers = config.num_hidden_layers
        self.transformer_layers = [
            i for i in range(self.num_layers)
            if self.layer_types[i] == "full_attention"
        ]
        self.last_linear_layer = (
            len(self.layer_types) - 1 - self.layer_types[::-1].index("linear_attention")
            if "linear_attention" in self.layer_types else -1
        )

        self.head_dim = head_dim
        self.key_bits = key_bits
        self.value_bits = value_bits

        # Standard Qwen3.5 cache storage (linear-attn layers and prefill-time KV)
        self.conv_states = [None for _ in range(self.num_layers)]
        self.recurrent_states = [None for _ in range(self.num_layers)]
        self.key_cache = [None for _ in range(self.num_layers)]
        self.value_cache = [None for _ in range(self.num_layers)]

        # RotorQuant per-layer caches (only for full_attention layers)
        self._caches: dict[int, DeferredQuantCache] = {}
        for layer_idx in self.transformer_layers:
            qk = _make_quantizer(quantizer_kind, head_dim, key_bits, seed=seed + layer_idx).to(device)
            qv = _make_quantizer(quantizer_kind, head_dim, value_bits, seed=seed + layer_idx + 10000).to(device)
            self._caches[layer_idx] = DeferredQuantCache(qk, qv)

        # Track which mode each full-attn cache is in
        self._mode_per_layer = {i: "prefill" for i in self.transformer_layers}

    def __len__(self) -> int:
        return len(self.layer_types)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the (full_K, full_V) attention should consume for this layer.
        """
        # Linear-attention layers: pass through untouched
        if layer_idx not in self.transformer_layers:
            if self.key_cache[layer_idx] is None:
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        # Full-attention: route through DeferredQuantCache
        cache = self._caches[layer_idx]
        T_new = key_states.shape[2]

        if self._mode_per_layer[layer_idx] == "prefill":
            # Decision: stay in prefill if T_new > 1 (still feeding prompt chunks),
            # transition to decode if this is a single new token.
            if T_new > 1:
                cache.append_prefill(key_states, value_states)
                # Return the FP16 buffer we just appended; concatenate any prior chunks.
                full_K = torch.cat(cache._prefill_K, dim=2)
                full_V = torch.cat(cache._prefill_V, dim=2)
                return full_K, full_V
            else:
                # T_new == 1: finalize prefill, then process this token as decode
                cache.finalize_prefill()
                self._mode_per_layer[layer_idx] = "decode"
                # Fall through to decode path

        # Decode path
        full_K, full_V = cache.append_decode(key_states, value_states)
        return full_K, full_V

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Beam-search reordering. Re-orders both linear and full-attn buffers."""
        for layer_idx in range(self.num_layers):
            if self.key_cache[layer_idx] is not None:
                bi = beam_idx.to(self.key_cache[layer_idx].device)
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, bi)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, bi)
            if self.conv_states[layer_idx] is not None:
                bi = beam_idx.to(self.conv_states[layer_idx].device)
                self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0, bi)
                self.recurrent_states[layer_idx] = self.recurrent_states[layer_idx].index_select(0, bi)
            if layer_idx in self._caches:
                cache = self._caches[layer_idx]
                if cache._quant_K is not None:
                    bi = beam_idx.to(cache._quant_K[0].device)
                    kp, kn = cache._quant_K
                    cache._quant_K = (kp.index_select(0, bi), kn.index_select(0, bi))
                    if cache._quant_V is not None:
                        vp, vn = cache._quant_V
                        cache._quant_V = (vp.index_select(0, bi), vn.index_select(0, bi))
                    elif cache._fp_V is not None:
                        cache._fp_V = cache._fp_V.index_select(0, bi)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """
        Total sequence length for the given layer. For full-attn layers we use
        the DeferredQuantCache's tracker; for linear-attn we use the local cache.
        """
        if layer_idx not in self.transformer_layers:
            if layer_idx is None or layer_idx >= self.num_layers:
                layer_idx = self.transformer_layers[0]
            else:
                return self.key_cache[layer_idx].shape[-2] if self.key_cache[layer_idx] is not None else 0
        return self._caches[layer_idx].get_seq_length()

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> Tuple[int, int]:
        kv_offset = 0
        past = self.get_seq_length(layer_idx)
        return query_length + past, kv_offset

    @property
    def has_previous_state(self) -> bool:
        if self.last_linear_layer < 0:
            # No linear-attention layers — fall back to checking any full-attn cache
            return any(self._caches[i].get_seq_length() > 0 for i in self.transformer_layers)
        return self.conv_states[self.last_linear_layer] is not None


def enable_rotorquant(model, key_bits: int = 3, value_bits: int = 3,
                      quantizer_kind: str = "planar"):
    """
    Drop-in replacement for enable_turboquant. Wraps the model's generate()
    so each call uses a fresh RotorQuantKVCache.
    """
    original_generate = model.generate

    def generate_with_rotorquant(*args, **kwargs):
        device = next(model.parameters()).device
        cache = RotorQuantKVCache.from_config(
            model.config,
            key_bits=key_bits,
            value_bits=value_bits,
            quantizer_kind=quantizer_kind,
            device=str(device),
        )
        kwargs["past_key_values"] = cache
        kwargs["use_cache"] = True
        return original_generate(*args, **kwargs)

    model.generate = generate_with_rotorquant
    model._rotorquant_original_generate = original_generate
    return model


def disable_rotorquant(model):
    if hasattr(model, "_rotorquant_original_generate"):
        model.generate = model._rotorquant_original_generate
        del model._rotorquant_original_generate
    return model
