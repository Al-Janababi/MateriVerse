"""Diffusion utilities (NumPy reference implementations; the torch model mirrors these).

* Lattice parameters and (relaxed one-hot) species use Gaussian DDPM with a cosine schedule.
* Fractional coordinates use a variance-exploding process on the torus with a wrapped-normal
  kernel, following the construction popularised by DiffCSP.
"""
from __future__ import annotations

import numpy as np


def cosine_alpha_bar(T: int, s: float = 0.008) -> np.ndarray:
    """alpha_bar_t for t = 1..T (Nichol & Dhariwal cosine schedule); alpha_bar_0 = 1."""
    t = np.arange(T + 1, dtype=float)
    f = np.cos(((t / T + s) / (1 + s)) * np.pi / 2) ** 2
    ab = f / f[0]
    betas = np.clip(1.0 - ab[1:] / ab[:-1], 0.0, 0.999)
    return np.cumprod(1.0 - betas)


def geometric_sigmas(T: int, sigma_min: float = 0.005, sigma_max: float = 0.5) -> np.ndarray:
    """sigma_t for t = 1..T, geometric from sigma_min to sigma_max."""
    return np.exp(np.linspace(np.log(sigma_min), np.log(sigma_max), T))


def wrapped_normal_score(d, sigma, K: int = 10) -> np.ndarray:
    """Score d/dd log sum_k N(d + k; 0, sigma^2) for displacement d (any shape), truncated to |k| <= K."""
    d = np.asarray(d, dtype=float)
    ks = np.arange(-K, K + 1, dtype=float)
    z = d[..., None] + ks
    logw = -0.5 * (z / sigma) ** 2
    logw -= logw.max(axis=-1, keepdims=True)
    w = np.exp(logw)
    num = (-(z / sigma**2) * w).sum(axis=-1)
    return num / w.sum(axis=-1)


def wrapped_score_norm_sq(sigma: float, K: int = 10, n_grid: int = 4001) -> float:
    """E_d[score^2] under the wrapped normal, by numerical integration over one period."""
    grid = np.linspace(-0.5, 0.5, n_grid)
    ks = np.arange(-K, K + 1, dtype=float)
    dens = np.exp(-0.5 * ((grid[:, None] + ks) / sigma) ** 2).sum(axis=-1)
    dens /= np.trapezoid(dens, grid)
    s = wrapped_normal_score(grid, sigma, K)
    return float(np.trapezoid(dens * s**2, grid))


def noise_coords(x0, sigma, rng: np.random.Generator):
    """x_t = (x0 + sigma * eps) mod 1; returns (x_t, eps, wrapped displacement d in [-0.5, 0.5))."""
    eps = rng.standard_normal(np.shape(x0))
    xt = (np.asarray(x0) + sigma * eps) % 1.0
    d = xt - np.asarray(x0)
    d = d - np.round(d)
    return xt, eps, d


def reverse_coord_step(xt, score, sigma_t, sigma_prev, rng: np.random.Generator):
    """Ancestral variance-exploding step from sigma_t to sigma_prev (< sigma_t), wrapped to [0, 1)."""
    gap = sigma_t**2 - sigma_prev**2
    mean = xt + gap * score
    std = np.sqrt(sigma_prev**2 * gap / sigma_t**2)
    return (mean + std * rng.standard_normal(np.shape(xt))) % 1.0
