"""
Verify TurboQuant round-trip MSE on random unit vectors.
Compares empirical MSE against the theoretical per-coordinate Lloyd-Max distortion.
"""
import torch
import numpy as np
from turboquant.quantizer import TurboQuantMSE
from turboquant.lloyd_max import get_beta_pdf, get_codebook
from scipy.integrate import quad

def main():
    dim = 128
    n_vectors = 10_000

    for bits in [2, 4]:
        print(f"\n{'='*60}")
        print(f"  dim={dim}, bits={bits}, n_vectors={n_vectors}")
        print(f"{'='*60}")

        # 1. Build quantizer
        quantizer = TurboQuantMSE(dim=dim, bits=bits)

        # 2. Generate random unit vectors
        x = torch.randn(n_vectors, dim)
        x = x / x.norm(dim=-1, keepdim=True)

        # 3. Round-trip: quantize → dequantize
        packed, norms = quantizer.quantize(x)
        x_hat = quantizer.dequantize(packed, norms)

        # 4. Empirical MSE = E[||x - x_hat||^2]
        mse_per_vec = ((x - x_hat) ** 2).sum(dim=-1)
        empirical_mse = mse_per_vec.mean().item()
        empirical_std = mse_per_vec.std().item()

        # 5. Theoretical per-coordinate distortion from codebook
        codebook = get_codebook(dim, bits)
        centroids = codebook["centroids"].double()
        boundaries = codebook["boundaries"].double()
        pdf = get_beta_pdf(dim)
        num_levels = 2 ** bits

        per_coord_dist = 0.0
        for i in range(num_levels):
            lo = boundaries[i].item()
            hi = boundaries[i + 1].item()
            c = centroids[i].item()
            d_val, _ = quad(lambda t: (t - c) ** 2 * pdf(t), lo, hi,
                            epsabs=1e-12, epsrel=1e-12)
            per_coord_dist += d_val

        theoretical_mse = dim * per_coord_dist  # sum over d coordinates

        # 6. Information-theoretic lower bound (Shannon)
        lower_bound = 1.0 / (4 ** bits)

        print(f"\n  Empirical MSE:        {empirical_mse:.6f}  (std: {empirical_std:.6f})")
        print(f"  Theoretical MSE:      {theoretical_mse:.6f}")
        print(f"  Ratio (emp/theo):     {empirical_mse / theoretical_mse:.4f}x")
        print(f"  Shannon lower bound:  {lower_bound:.6f}")
        print(f"  Ratio (emp/lower):    {empirical_mse / lower_bound:.3f}x")

        # 7. Cosine similarity (quality check)
        cos_sim = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1)
        print(f"\n  Mean cosine sim:      {cos_sim.mean().item():.6f}")
        print(f"  Min  cosine sim:      {cos_sim.min().item():.6f}")

        # 8. Sanity: norms should be ~1 for unit vectors
        print(f"  Mean norm recovered:  {x_hat.norm(dim=-1).mean().item():.6f}")

        # Assert reasonable quality
        assert empirical_mse < theoretical_mse * 1.15, \
            f"Empirical MSE {empirical_mse} exceeds theoretical {theoretical_mse} by >15%!"
        assert cos_sim.mean().item() > 0.9, \
            f"Mean cosine similarity {cos_sim.mean().item()} too low!"
        print("\n  ✅ All checks passed.")

    print()

if __name__ == "__main__":
    main()
