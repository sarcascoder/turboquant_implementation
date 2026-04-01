"""
Validate TurboQuant distortion vs paper's theoretical bounds.
Uses 10,000 random unit vectors for statistical robustness.
"""
import torch
from turboquant.quantizer import TurboQuantMSE

def main():
    dim = 128
    n_vectors = 10000

    # Paper's theoretical bounds (Table from Theorem 1)
    paper_bounds = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}
    lower_bounds = {b: 1.0 / (4**b) for b in range(1, 5)}

    print(f"MSE Distortion Validation (d={dim}, n={n_vectors} random unit vectors)")
    print(f"{'Bits':>6} {'Empirical':>12} {'Paper UB':>12} {'Lower Bound':>12} {'Ratio':>8} {'Status':>8}")
    print("-" * 68)

    all_pass = True
    for bits in [1, 2, 3, 4]:
        q = TurboQuantMSE(dim=dim, bits=bits)

        # Generate random unit vectors
        torch.manual_seed(42)  # Reproducible
        x = torch.randn(n_vectors, dim)
        x = x / torch.linalg.vector_norm(x, dim=-1, keepdim=True)

        # Verify exact unit norm
        norms = torch.linalg.vector_norm(x, dim=-1)
        assert (norms - 1.0).abs().max() < 1e-6, f"Normalization error: max deviation {(norms-1.0).abs().max()}"

        packed, norms_out = q.quantize(x)
        x_hat = q.dequantize(packed, norms_out)

        mse = torch.mean(torch.sum((x - x_hat)**2, dim=-1)).item()
        ratio = mse / lower_bounds[bits]
        status = "✅" if ratio <= 2.7 else "❌"
        if ratio > 2.7:
            all_pass = False

        print(f"{bits:>6} {mse:>12.6f} {paper_bounds[bits]:>12.6f} {lower_bounds[bits]:>12.6f} {ratio:>7.3f}× {status:>8}")

    print()
    print("Target: Ratio ≤ 2.7× (paper's theoretical guarantee)")
    if all_pass:
        print("🎉 ALL BIT-WIDTHS PASS")
    else:
        print("⚠️  SOME BIT-WIDTHS EXCEED 2.7× BOUND")

if __name__ == "__main__":
    main()
