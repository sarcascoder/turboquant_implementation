import torch

def pack_bits(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Packs a tensor of integer indices where each element is in [0, 2^bits - 1]
    into a uint8 tensor.
    Bits 1, 2, 4: true bit-packing (multiple values per byte).
    Bits 3, 5, 6, 7, 8: one byte per element (simple storage).
    """
    if bits not in [1, 2, 4]:
        # For non-power-of-2 widths, store one index per byte
        return indices.to(torch.uint8)

    elements_per_byte = 8 // bits
    last_dim = indices.shape[-1]

    if last_dim % elements_per_byte != 0:
        pad_len = elements_per_byte - (last_dim % elements_per_byte)
        pad = torch.zeros(*indices.shape[:-1], pad_len, dtype=indices.dtype, device=indices.device)
        indices = torch.cat([indices, pad], dim=-1)

    packed_shape = list(indices.shape)
    packed_shape[-1] = packed_shape[-1] // elements_per_byte

    reshaped = indices.view(*packed_shape[:-1], packed_shape[-1], elements_per_byte)

    packed = torch.zeros(packed_shape, dtype=torch.uint8, device=indices.device)

    for i in range(elements_per_byte):
        shifted = reshaped[..., i].to(torch.uint8) << (i * bits)
        packed = packed | shifted

    return packed

def unpack_bits(packed: torch.Tensor, bits: int, original_last_dim: int = None) -> torch.Tensor:
    """
    Unpacks a uint8 tensor back into indices.
    """
    if bits not in [1, 2, 4]:
        # Byte-per-element fallback
        if original_last_dim is not None and original_last_dim != packed.shape[-1]:
            return packed[..., :original_last_dim].to(torch.long)
        return packed.to(torch.long)

    elements_per_byte = 8 // bits
    unpacked_shape = list(packed.shape)
    unpacked_shape[-1] = unpacked_shape[-1] * elements_per_byte

    mask = (1 << bits) - 1

    unpacked = torch.zeros(*packed.shape, elements_per_byte, dtype=torch.uint8, device=packed.device)

    for i in range(elements_per_byte):
        shifted = (packed >> (i * bits)) & mask
        unpacked[..., i] = shifted

    unpacked = unpacked.view(unpacked_shape)

    if original_last_dim is not None and original_last_dim != unpacked_shape[-1]:
        unpacked = unpacked[..., :original_last_dim]

    return unpacked.to(torch.long)
