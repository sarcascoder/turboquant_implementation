import torch
import math
from turboquant.quantizer import TurboQuantMSE

def test_mse_quantizer():
    # d=128, test bits=2
    dim = 128
    bits = 2
    
    quantizer = TurboQuantMSE(dim, bits)
    
    # Generate random vectors on unit sphere
    x = torch.randn(1000, dim)
    x = x / torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    
    packed_indices, norms = quantizer.quantize(x)
    
    # Check shape of packed indices
    # bits=2 -> 4 elements per byte. dim=128 -> 128/4 = 32 bytes
    assert packed_indices.dtype == torch.uint8
    assert packed_indices.shape[-1] == 32
    
    x_hat = quantizer.dequantize(packed_indices, norms)
    
    # Check distortion E[||x - x_hat||^2]
    # Theorem 1 bounds:
    # 2 bits approx 0.117
    # 3 bits approx 0.030
    
    distortion = torch.mean(torch.sum((x - x_hat) ** 2, dim=-1)).item()
    print(f"Empirical Distortion for b={bits}, d={dim}: {distortion:.4f}")
    assert distortion < 0.2
    print("Test passed!")

if __name__ == "__main__":
    test_mse_quantizer()
