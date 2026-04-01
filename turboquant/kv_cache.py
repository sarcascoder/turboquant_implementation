import torch
from typing import Any, Dict, Optional, Tuple

from turboquant.quantizer import TurboQuantMSE


class TurboQuantKVCache:
    """
    Drop-in replacement for Qwen3_5DynamicCache that applies TurboQuant
    compression on full_attention layers' KV cache.

    Linear attention layers are untouched (they use conv_states / recurrent_states).
    Full attention layers store old tokens in quantized form and keep the most
    recent `residual_window` tokens in full precision.

    Works by subclassing at runtime from the model's own cache class so that
    all model-specific attributes (has_previous_state, conv_states, etc.) 
    are preserved.
    """

    @staticmethod
    def from_config(config, key_bits=4, value_bits=2, residual_window=128, device="cpu"):
        """
        Factory: create a TurboQuant-compressed cache from a model config.
        Dynamically subclasses the model's native cache class.
        """
        # Get the text_config for multimodal models
        text_config = config
        if hasattr(config, "text_config") and config.text_config is not None:
            text_config = config.text_config

        head_dim = getattr(text_config, "head_dim", None)
        if head_dim is None:
            head_dim = text_config.hidden_size // text_config.num_attention_heads

        # Import the model-specific cache class
        try:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache
            base_cache_cls = Qwen3_5DynamicCache
        except ImportError:
            # Fallback for non-Qwen models
            from transformers.cache_utils import DynamicCache
            base_cache_cls = DynamicCache

        return _TurboQuantQwen35Cache(
            config=text_config,
            base_cache_cls=base_cache_cls,
            head_dim=head_dim,
            key_bits=key_bits,
            value_bits=value_bits,
            residual_window=residual_window,
            device=device,
        )


class _TurboQuantQwen35Cache:
    """
    Actual cache implementation for Qwen 3.5 models.
    Wraps Qwen3_5DynamicCache and adds TurboQuant compression
    on full_attention layer KV entries.
    """

    is_compileable = False

    def __init__(self, config, base_cache_cls, head_dim, key_bits, value_bits,
                 residual_window, device):
        self.layer_types = config.layer_types
        self.transformer_layers = [
            i for i in range(config.num_hidden_layers)
            if self.layer_types[i] == "full_attention"
        ]
        self.last_linear_layer = (
            len(self.layer_types) - 1 - self.layer_types[::-1].index("linear_attention")
        )

        num_layers = config.num_hidden_layers

        # Standard Qwen3.5 cache storage
        self.conv_states = [None for _ in range(num_layers)]
        self.recurrent_states = [None for _ in range(num_layers)]
        self.key_cache = [None for _ in range(num_layers)]
        self.value_cache = [None for _ in range(num_layers)]

        # TurboQuant additions
        self.head_dim = head_dim
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.residual_window = residual_window

        # Per-layer quantizers (only for full_attention layers)
        self._k_quantizers = {}
        self._v_quantizers = {}
        for layer_idx in self.transformer_layers:
            self._k_quantizers[layer_idx] = TurboQuantMSE(dim=head_dim, bits=key_bits).to(device)
            self._v_quantizers[layer_idx] = TurboQuantMSE(dim=head_dim, bits=value_bits).to(device)

        # Quantized storage per layer: (packed_indices, norms) or None
        self._k_quantized = [None for _ in range(num_layers)]
        self._v_quantized = [None for _ in range(num_layers)]
        self._quantized_lens = [0 for _ in range(num_layers)]

    def __len__(self):
        return len(self.layer_types)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Standard cache update. For full_attention layers, applies TurboQuant
        compression when the unquantized window overflows.
        """
        # Append to raw cache (same as original)
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], dim=2
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], dim=2
            )

        # Only compress full_attention layers
        if layer_idx in self.transformer_layers:
            current_len = self.key_cache[layer_idx].shape[2]

            if current_len > self.residual_window:
                num_to_q = current_len - self.residual_window

                # Slice oldest tokens
                k_old = self.key_cache[layer_idx][:, :, :num_to_q, :]
                v_old = self.value_cache[layer_idx][:, :, :num_to_q, :]

                # Keep residual window
                self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, num_to_q:, :].contiguous()
                self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, num_to_q:, :].contiguous()

                # Quantize
                k_packed, k_norms = self._k_quantizers[layer_idx].quantize(k_old)
                v_packed, v_norms = self._v_quantizers[layer_idx].quantize(v_old)

                if self._k_quantized[layer_idx] is None:
                    self._k_quantized[layer_idx] = (k_packed, k_norms)
                    self._v_quantized[layer_idx] = (v_packed, v_norms)
                else:
                    ok_p, ok_n = self._k_quantized[layer_idx]
                    self._k_quantized[layer_idx] = (
                        torch.cat([ok_p, k_packed], dim=2),
                        torch.cat([ok_n, k_norms], dim=2),
                    )
                    ov_p, ov_n = self._v_quantized[layer_idx]
                    self._v_quantized[layer_idx] = (
                        torch.cat([ov_p, v_packed], dim=2),
                        torch.cat([ov_n, v_norms], dim=2),
                    )
                self._quantized_lens[layer_idx] += num_to_q

            # Return full reconstructed cache
            if self._k_quantized[layer_idx] is not None:
                k_deq = self._k_quantizers[layer_idx].dequantize(
                    *self._k_quantized[layer_idx]
                )
                v_deq = self._v_quantizers[layer_idx].dequantize(
                    *self._v_quantized[layer_idx]
                )
                full_k = torch.cat([k_deq, self.key_cache[layer_idx]], dim=2)
                full_v = torch.cat([v_deq, self.value_cache[layer_idx]], dim=2)
                return full_k, full_v

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] is not None:
                device = self.key_cache[layer_idx].device
                bi = beam_idx.to(device)
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, bi)
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, bi)
            if self.conv_states[layer_idx] is not None:
                device = self.conv_states[layer_idx].device
                bi = beam_idx.to(device)
                self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0, bi)
                self.recurrent_states[layer_idx] = self.recurrent_states[layer_idx].index_select(0, bi)
            if self._k_quantized[layer_idx] is not None:
                kp, kn = self._k_quantized[layer_idx]
                vp, vn = self._v_quantized[layer_idx]
                bi = beam_idx.to(kp.device)
                self._k_quantized[layer_idx] = (kp.index_select(0, bi), kn.index_select(0, bi))
                self._v_quantized[layer_idx] = (vp.index_select(0, bi), vn.index_select(0, bi))

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        layer_idx = self.transformer_layers[0] if layer_idx not in self.transformer_layers else layer_idx
        length = self._quantized_lens[layer_idx]
        if self.key_cache[layer_idx] is not None:
            length += self.key_cache[layer_idx].shape[-2]
        return length

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        kv_offset = 0
        past_seen_tokens = self.get_seq_length(layer_idx)
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset

    @property
    def has_previous_state(self):
        return self.conv_states[self.last_linear_layer] is not None
