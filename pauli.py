"""Pauli-motivated exclusion bias for attention (our concrete instantiation of Eq. 2).

The article defines P in {-inf, 0}^{NxN} as an exclusion-motivated mask between latent particle
pairs. We instantiate it geometrically: two atoms closer than `d_min` (Angstrom) violate a hard-core
exclusion, and attention between such pairs is penalised. This is an inductive bias, not an exact
fermionic antisymmetrisation.
"""
from __future__ import annotations

import numpy as np


def pauli_bias(dist, d_min: float = 1.0, mode: str = "soft") -> np.ndarray:
    """Bias matrix P (N, N) added to attention logits as alpha * P.

    mode 'hard': -inf for i != j with dist < d_min, else 0.
    mode 'soft': -relu(d_min - dist) / d_min  (in [-1, 0]), else 0.
    mode 'off' : zeros.
    The diagonal is always 0 so that every atom can attend to itself.
    """
    dist = np.asarray(dist, dtype=float)
    n = dist.shape[-1]
    eye = np.eye(n, dtype=bool)
    if mode == "off":
        return np.zeros_like(dist)
    close = (dist < d_min) & ~eye
    if mode == "hard":
        return np.where(close, -np.inf, 0.0)
    if mode == "soft":
        p = -np.maximum(d_min - dist, 0.0) / d_min
        return np.where(eye, 0.0, p)
    raise ValueError(f"unknown mode '{mode}'")


def biased_softmax_attention(q, k, v, bias=None, alpha: float = 1.0):
    """Single-head reference: softmax((q k^T)/sqrt(d) + alpha*bias) v. Shapes (N, d)."""
    d = q.shape[-1]
    logits = q @ k.T / np.sqrt(d)
    if bias is not None:
        logits = logits + alpha * bias
    logits = logits - logits.max(axis=-1, keepdims=True)
    w = np.exp(logits)
    w /= w.sum(axis=-1, keepdims=True)
    return w @ v, w
