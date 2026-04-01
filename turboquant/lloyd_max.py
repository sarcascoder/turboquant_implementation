import os
import math
import torch
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

CODEBOOK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "codebooks")

# High-precision integration tolerances
_EPSABS = 1e-12
_EPSREL = 1e-12


def get_beta_pdf(d):
    """
    Returns the analytical PDF of a coordinate of a uniformly distributed
    random point on the (d-1)-dimensional sphere S^{d-1} in R^d.

    f(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^((d-3)/2),  x ∈ (-1, 1)
    """
    log_c = math.lgamma(d / 2.0) - 0.5 * math.log(math.pi) - math.lgamma((d - 1.0) / 2.0)
    c = math.exp(log_c)
    power = (d - 3) / 2.0

    def pdf(x):
        if x <= -1.0 or x >= 1.0:
            return 0.0
        return c * math.pow(1.0 - x * x, power)

    return pdf


def _build_cdf(pdf, t_min=-1.0, t_max=1.0, n_points=4096):
    """
    Numerically builds a CDF table from the PDF using high-precision quadrature.
    Returns (x_grid, cdf_values) where cdf_values[i] = ∫_{t_min}^{x_grid[i]} pdf(t) dt.
    """
    x_grid = np.linspace(t_min, t_max, n_points)
    cdf_vals = np.zeros(n_points)
    for i in range(1, n_points):
        val, _ = quad(pdf, x_grid[i - 1], x_grid[i], epsabs=_EPSABS, epsrel=_EPSREL)
        cdf_vals[i] = cdf_vals[i - 1] + val
    # Normalize to exactly [0, 1]
    total = cdf_vals[-1]
    if total > 0:
        cdf_vals /= total
    return x_grid, cdf_vals


def _quantile(x_grid, cdf_vals, p):
    """
    Inverse CDF lookup: returns x such that CDF(x) ≈ p.
    Uses linear interpolation on the CDF table.
    """
    idx = np.searchsorted(cdf_vals, p, side="right")
    idx = np.clip(idx, 1, len(cdf_vals) - 1)
    # Linear interpolation
    x0, x1 = x_grid[idx - 1], x_grid[idx]
    c0, c1 = cdf_vals[idx - 1], cdf_vals[idx]
    if abs(c1 - c0) < 1e-15:
        return float(x0)
    frac = (p - c0) / (c1 - c0)
    return float(x0 + frac * (x1 - x0))


def lloyd_max(pdf, num_levels, t_min=-1.0, t_max=1.0, iterations=1000, tol=1e-12):
    """
    Lloyd-Max optimal scalar quantizer for an analytical PDF.

    Uses CDF-quantile initialization (every bin starts with equal probability mass)
    and high-precision numerical integration for centroid updates.

    Returns (centroids, boundaries) as 1D float64 tensors.
    """
    # ── Step 1: CDF-quantile initialization ──
    # Place initial centroids at midpoints of equal-probability bins.
    # This guarantees every bin has mass, even in the tails.
    x_grid, cdf_vals = _build_cdf(pdf, t_min, t_max)

    levels = []
    for i in range(num_levels):
        # Midpoint of the i-th equal-probability interval
        p_lo = i / num_levels
        p_hi = (i + 1) / num_levels
        p_mid = (p_lo + p_hi) / 2.0
        levels.append(_quantile(x_grid, cdf_vals, p_mid))

    best_levels = None
    best_distortion = float("inf")

    for iteration in range(iterations):
        # ── Step 2: Update boundaries (midpoints of adjacent centroids) ──
        boundaries = [t_min]
        for i in range(num_levels - 1):
            boundaries.append((levels[i] + levels[i + 1]) / 2.0)
        boundaries.append(t_max)

        # ── Step 3: Update centroids (conditional expectation in each bin) ──
        new_levels = []
        total_distortion = 0.0

        for i in range(num_levels):
            lo = boundaries[i]
            hi = boundaries[i + 1]

            # E[X | lo < X < hi] = ∫ x·p(x)dx / ∫ p(x)dx
            num, _ = quad(lambda x: x * pdf(x), lo, hi, epsabs=_EPSABS, epsrel=_EPSREL)
            den, _ = quad(pdf, lo, hi, epsabs=_EPSABS, epsrel=_EPSREL)

            if den > 1e-15:
                centroid = num / den
            else:
                centroid = (lo + hi) / 2.0  # Fallback for zero-mass bin

            new_levels.append(centroid)

            # Accumulate distortion: ∫(x - centroid)² · p(x) dx over this bin
            dist, _ = quad(
                lambda x, c=centroid: (x - c) ** 2 * pdf(x),
                lo, hi, epsabs=_EPSABS, epsrel=_EPSREL
            )
            total_distortion += dist

        # ── Step 4: Check convergence ──
        max_shift = max(abs(new - old) for new, old in zip(new_levels, levels))
        levels = new_levels

        if total_distortion < best_distortion:
            best_distortion = total_distortion
            best_levels = list(levels)

        if max_shift < tol:
            break

    # Use the best centroids found
    levels = best_levels

    # Final boundaries
    boundaries = [t_min]
    for i in range(num_levels - 1):
        boundaries.append((levels[i] + levels[i + 1]) / 2.0)
    boundaries.append(t_max)

    return (
        torch.tensor(levels, dtype=torch.float64),
        torch.tensor(boundaries, dtype=torch.float64),
    )


def get_codebook(dim, bits):
    """
    Retrieves or computes the Lloyd-Max codebook for a given dimension and bit-width.
    Codebooks are cached to disk after first computation.
    """
    os.makedirs(CODEBOOK_DIR, exist_ok=True)
    file_path = os.path.join(CODEBOOK_DIR, f"codebook_d{dim}_b{bits}.pt")

    if os.path.exists(file_path):
        return torch.load(file_path, weights_only=True)

    print(f"Computing Lloyd-Max codebook for dim={dim}, bits={bits}...")
    pdf = get_beta_pdf(dim)
    num_levels = 2 ** bits

    centroids, boundaries = lloyd_max(pdf, num_levels)

    # Verify: compute theoretical distortion for this codebook
    total_dist = 0.0
    for i in range(num_levels):
        lo = boundaries[i].item()
        hi = boundaries[i + 1].item()
        c = centroids[i].item()
        d, _ = quad(lambda x: (x - c) ** 2 * pdf(x), lo, hi, epsabs=_EPSABS, epsrel=_EPSREL)
        total_dist += d
    # The per-coordinate distortion is total_dist; for d coordinates, MSE = d * total_dist
    # For unit vectors, E[||x - Q(x)||²] ≈ d * total_dist (after rotation)
    print(f"  Per-coordinate distortion: {total_dist:.8f}")
    print(f"  Predicted MSE (d={dim}):    {dim * total_dist:.6f}")
    lower_bound = 1.0 / (4 ** bits)
    print(f"  Lower bound:               {lower_bound:.6f}")
    print(f"  Ratio:                      {dim * total_dist / lower_bound:.3f}×")

    # Store as float32 for GPU efficiency
    data = {
        "centroids": centroids.float(),
        "boundaries": boundaries.float(),
    }
    torch.save(data, file_path)
    return data
