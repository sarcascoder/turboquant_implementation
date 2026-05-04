"""
DeferredQuantCache — post-prefill KV cache quantization.

The single most important engineering idea in the project. Without this,
even a perfect quantizer gives catastrophic PPL on real models.

WHY: Quantizing during prefill means layer 0 sees noisy attention outputs.
     Layer 1 then quantizes its K from noisy inputs, layer 2 from noisier
     inputs, etc. Error compounds multiplicatively across 30+ transformer
     layers — PPL > 1000.

FIX: State machine with three modes — prefill / transitioning / decode.
     - prefill   : FP16 K and V buffers, ZERO quantization.
     - finalize  : bulk-convert all prefill K, V to quantized format,
                   free the FP16 buffers.
     - decode    : per-token quantize for storage, but the CURRENT step's
                   K is returned at FP16 (no compounding for the just-inserted
                   token).
"""
from __future__ import annotations
import torch
from typing import List, Optional, Tuple


class DeferredQuantCache:
    """
    Per-(batch, head) KV cache that defers quantization until prefill ends.

    Works on tensor shapes (B, H, T, D). Quantizer must implement:
        packed, norms = quantizer.quantize(x)        # x: (..., D)
        x_hat        = quantizer.dequantize(packed, norms)

    Args:
        quantizer_k: a PlanarQuant / TurboQuantMSE instance for K (must be on
                     the same device as cache tensors).
        quantizer_v: same for V. Pass None to keep V in FP16 (asymmetric mode).
    """

    PREFILL = "prefill"
    DECODE  = "decode"

    def __init__(self, quantizer_k, quantizer_v=None):
        self.quantizer_k = quantizer_k
        self.quantizer_v = quantizer_v          # None → V stays FP16
        self.mode = self.PREFILL

        # Prefill buffers: list of (B, H, t_chunk, D) FP16 tensors
        self._prefill_K: List[torch.Tensor] = []
        self._prefill_V: List[torch.Tensor] = []

        # Quantized storage (after finalize_prefill):
        # Tuple (packed_indices, norms) of shape (B, H, T, ?)
        self._quant_K: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._quant_V: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        # FP16 V cache when V is not quantized (asymmetric K-only mode)
        self._fp_V: Optional[torch.Tensor] = None

        self._seq_len = 0

    # ------------------------------------------------------------------ #
    # Prefill
    # ------------------------------------------------------------------ #
    def append_prefill(self, K: torch.Tensor, V: torch.Tensor) -> None:
        """
        Append one prefill chunk (B, H, t, D). No quantization happens here.
        Must be called only while mode == PREFILL.
        """
        assert self.mode == self.PREFILL, \
            f"append_prefill called in mode={self.mode}; call only during prefill"
        # Defensive copy reference (don't keep autograd history)
        self._prefill_K.append(K.detach())
        self._prefill_V.append(V.detach())
        self._seq_len += K.shape[2]

    def finalize_prefill(self) -> None:
        """
        Transition prefill → decode. Bulk-quantize all stored K (and V if
        V quantizer set), then free FP16 buffers.
        """
        assert self.mode == self.PREFILL, \
            f"finalize_prefill called in mode={self.mode}"

        if not self._prefill_K:
            # Nothing to finalize; just flip the switch.
            self.mode = self.DECODE
            return

        all_K = torch.cat(self._prefill_K, dim=2)
        all_V = torch.cat(self._prefill_V, dim=2)

        # K is always quantized
        kp, kn = self.quantizer_k.quantize(all_K)
        self._quant_K = (kp, kn)

        # V is quantized only if a V quantizer was provided
        if self.quantizer_v is not None:
            vp, vn = self.quantizer_v.quantize(all_V)
            self._quant_V = (vp, vn)
            self._fp_V = None
        else:
            # K-only mode: keep full V in FP16
            self._fp_V = all_V

        # Free prefill buffers
        self._prefill_K.clear()
        self._prefill_V.clear()
        self.mode = self.DECODE

    # ------------------------------------------------------------------ #
    # Decode
    # ------------------------------------------------------------------ #
    def append_decode(self, K_new: torch.Tensor, V_new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append one decode-step KV (typically t==1) and return what attention
        should consume:
            full_K — historical K dequantized + CURRENT-STEP K_new at FP16
            full_V — historical V dequantized + V_new at FP16

        Why FP16 for the current step:
            The compounding problem only happens when noisy K from token t
            feeds into the layer's output, which feeds into token t+1's K.
            By keeping the just-inserted K at full precision for THIS step's
            attention, we close that loop. The K we *store* for future steps
            is quantized — but those tokens won't be re-derived from this
            attention output.
        """
        assert self.mode == self.DECODE, \
            f"append_decode called in mode={self.mode}; call finalize_prefill first"
        assert self._quant_K is not None, "finalize_prefill must be called before append_decode"

        # 1) Dequantize historical K and V
        K_old = self.quantizer_k.dequantize(*self._quant_K)
        if self.quantizer_v is not None:
            V_old = self.quantizer_v.dequantize(*self._quant_V)
        else:
            V_old = self._fp_V

        # 2) Concatenate the FP16 K_new / V_new as the current step
        full_K = torch.cat([K_old, K_new], dim=2)
        full_V = torch.cat([V_old, V_new], dim=2)

        # 3) Quantize the new K_new and append to quantized storage so future
        #    decode steps see it (compounded error is bounded — 1 step of noise
        #    per cached token, not 30+ layers).
        new_kp, new_kn = self.quantizer_k.quantize(K_new)
        old_kp, old_kn = self._quant_K
        self._quant_K = (
            torch.cat([old_kp, new_kp], dim=2),
            torch.cat([old_kn, new_kn], dim=2),
        )

        if self.quantizer_v is not None:
            new_vp, new_vn = self.quantizer_v.quantize(V_new)
            old_vp, old_vn = self._quant_V
            self._quant_V = (
                torch.cat([old_vp, new_vp], dim=2),
                torch.cat([old_vn, new_vn], dim=2),
            )
        else:
            self._fp_V = torch.cat([self._fp_V, V_new], dim=2)

        self._seq_len += K_new.shape[2]
        return full_K, full_V

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def get_seq_length(self) -> int:
        return self._seq_len

    def __repr__(self) -> str:
        return (f"DeferredQuantCache(mode={self.mode}, seq_len={self._seq_len}, "
                f"v_quantized={self.quantizer_v is not None})")
