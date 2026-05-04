import math
import torch
import torch.nn as nn
from typing import Tuple, Optional

from turboquant.lloyd_max import get_codebook
from turboquant.bit_packing import pack_bits, unpack_bits

class TurboQuantMSE(nn.Module):
    """
    Core MSE quantizer using random rotation and Lloyd-Max scalar quantization.
    Works independently for any input shape, applying rotation on the last dimension.
    """
    def __init__(self, dim: int, bits: int):
        super().__init__()
        self.dim = dim
        self.bits = bits
        
        # Ensure we have the codebook
        data = get_codebook(dim, bits)
        # Register as buffers so they move with the module
        self.register_buffer("centroids", data["centroids"])
        self.register_buffer("boundaries", data["boundaries"])
        
        # Generate random orthogonal matrix (Haar measure over O(d))
        # Q * R = random Gaussian matrix
        rand_mat = torch.randn(dim, dim)
        q, r = torch.linalg.qr(rand_mat)
        # Ensure uniform distribution
        d = torch.diag(r)
        q = q * torch.sign(d)
        
        self.register_buffer("rotation_matrix", q)
        
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantizes input tensor x of shape (..., dim) to packed uint8 indices and norms.
        Returns:
            packed_indices: uint8 tensor, shape (..., ceil(dim / (8/bits)))
            norms: fp16/fp32 tensor of L2 norms, shape (..., 1)
        """
        # 1. Compute L2 norm
        norms = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        # Avoid division by zero
        safe_norms = norms.clamp(min=1e-8)
        
        # 2. Normalize
        x_norm = x / safe_norms
        
        # 3. Random rotation
        # x_norm: (..., dim), rotation_matrix: (dim, dim)
        # Result: y = x_norm @ rotation_matrix.T
        r_mat = self.rotation_matrix.to(x.dtype)
        y = torch.matmul(x_norm, r_mat.T)
        
        # 4. Scalar quantization using nearest neighbor lookup via broadcasting/searchsorted
        # boundaries shape: (2^bits + 1)
        # y shape: (..., dim)
        # torch.searchsorted needs 1D boundaries
        # It returns indices in [0, len(boundaries)], we want [0, len(centroids)-1]
        
        # Make a search boundary tensor. shape: (2^bits - 1)
        # We drop the first (-1.0) and last (1.0) boundaries
        inner_boundaries = self.boundaries[1:-1].to(x.dtype)
        
        # indices will be in range [0, 2^bits - 1]
        indices = torch.searchsorted(inner_boundaries, y.contiguous())
        
        # 5. Pack indices
        packed_indices = pack_bits(indices, self.bits)
        
        return packed_indices, norms
        
    def dequantize(self, packed_indices: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs the original tensor from packed indices and norms.
        """
        # 1. Unpack bits
        indices = unpack_bits(packed_indices, self.bits, original_last_dim=self.dim)
        
        # 2. Map indices to centroids
        # centroids shape: (2^bits)
        y_hat = self.centroids.to(norms.dtype)[indices.long()]
        
        # 3. Inverse rotation
        # y_hat: (..., dim)
        # Since rotation_matrix is orthogonal, inverse = transpose
        # x_norm_hat = y_hat @ rotation_matrix
        r_mat = self.rotation_matrix.to(norms.dtype)
        x_norm_hat = torch.matmul(y_hat, r_mat)
        
        # 4. Denormalize
        x_hat = x_norm_hat * norms
        
        return x_hat


class TurboQuantProd(nn.Module):
    """
    Two-stage inner product quantizer (MSE quantizer at bits-1, QJL on residual).

    Storage per vector: (bits-1) * dim bits (MSE) + m bits (QJL signs) + 16 bits (norms)
    Inner product estimator is unbiased in expectation:
        <y, x> ≈ <y, x_mse> + sqrt(pi/2)/m * res_norm * <y @ S.T, signs(S residual)>

    The sqrt(pi/2)/m factor comes from E[|N(0,1)|] = sqrt(2/pi); without it the
    estimator is biased by a constant factor.
    """
    def __init__(self, dim: int, bits: int, m: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.bits = bits
        # m = QJL projection dimension. Defaults to dim (paper recommendation).
        self.m = m if m is not None else dim
        assert bits >= 2, "TurboQuantProd requires at least 2 bits"

        # Stage 1: MSE Quantizer with (bits - 1)
        self.mse_quantizer = TurboQuantMSE(dim, bits - 1)

        # Stage 2: QJL random projection matrix S of shape (m, dim), entries iid N(0, 1)
        qjl_mat = torch.randn(self.m, dim)
        self.register_buffer("qjl_matrix", qjl_mat)

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantizes using 2 stages.
        Returns:
            packed_mse_indices: (..., packed_dim)
            mse_norms: (..., 1)
            packed_qjl_signs: (..., ceil(m/8))     [1-bit packing of m signs]
            residual_norms: (..., 1)
        """
        # Stage 1
        packed_mse, norms = self.mse_quantizer.quantize(x)
        x_hat_mse = self.mse_quantizer.dequantize(packed_mse, norms)

        # Residual
        residual = x - x_hat_mse
        res_norms = torch.linalg.vector_norm(residual, dim=-1, keepdim=True)

        # Stage 2: QJL — project residual through S, take signs
        # residual: (..., dim), S: (m, dim) → s_res: (..., m)
        s_res = torch.matmul(residual, self.qjl_matrix.T.to(x.dtype))
        # Map sign(s_res) ∈ {-1, +1} (zeros → +1 to disambiguate)
        binary_signs = (s_res >= 0).to(torch.int8)

        # Pack to 1 bit per element along last dim
        packed_qjl = pack_bits(binary_signs, 1)

        return packed_mse, norms, packed_qjl, res_norms

    def dequantize(self, packed_mse: torch.Tensor, norms: torch.Tensor,
                   packed_qjl: torch.Tensor, res_norms: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs an estimate x_hat = x_mse + (sqrt(pi/2)/m) * res_norm * (signs @ S).
        For inner-product use: <y, x_hat> is an unbiased estimator of <y, x>.
        """
        # Stage 1
        x_hat_mse = self.mse_quantizer.dequantize(packed_mse, norms)

        # Stage 2: unpack signs ∈ {0, 1} → {-1, +1}
        binary_signs = unpack_bits(packed_qjl, 1, original_last_dim=self.m)
        signs = binary_signs.to(norms.dtype) * 2.0 - 1.0

        # x_hat_qjl = sqrt(pi/2) / m * res_norm * (signs @ S)
        # signs: (..., m), S: (m, dim) → (..., dim)
        s_t_signs = torch.matmul(signs, self.qjl_matrix.to(norms.dtype))
        c = math.sqrt(math.pi / 2.0) / self.m
        x_hat_qjl = c * res_norms * s_t_signs

        return x_hat_mse + x_hat_qjl

    def estimate_inner_product(self, y: torch.Tensor, packed_mse: torch.Tensor,
                                norms: torch.Tensor, packed_qjl: torch.Tensor,
                                res_norms: torch.Tensor) -> torch.Tensor:
        """
        Direct (algebraically equivalent) two-term unbiased estimator of <y, x>:
            term1 = <y, x_mse>
            term2 = sqrt(pi/2)/m * res_norm * <y @ S.T, signs>
        Useful in attention: avoids materializing x_hat in dim-d space.
        """
        x_hat_mse = self.mse_quantizer.dequantize(packed_mse, norms)
        term1 = (y * x_hat_mse).sum(dim=-1, keepdim=True)

        binary_signs = unpack_bits(packed_qjl, 1, original_last_dim=self.m)
        signs = binary_signs.to(y.dtype) * 2.0 - 1.0

        y_proj = torch.matmul(y, self.qjl_matrix.T.to(y.dtype))
        qjl_ip = (y_proj * signs).sum(dim=-1, keepdim=True)
        c = math.sqrt(math.pi / 2.0) / self.m
        term2 = c * res_norms * qjl_ip

        return (term1 + term2).squeeze(-1)
