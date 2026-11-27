"""Uncertainty modelling: GP surrogate (Matern-5/2), split-conformal intervals, Gaussian KL (Eq. 8)."""
from __future__ import annotations

import math

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


class MaternGP:
    """Gaussian process with a Matern-5/2 kernel and a white-noise term."""

    def __init__(self, length_scale=1.0, noise=1e-3, seed=0):
        kernel = ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=2.5) + WhiteKernel(noise_level=noise)
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=seed)

    def fit(self, X, y):
        self.gp.fit(np.asarray(X, float), np.asarray(y, float))
        return self

    def predict(self, X):
        mean, std = self.gp.predict(np.asarray(X, float), return_std=True)
        return mean, std


def select_uncertain(std, threshold):
    """Indices whose predictive standard deviation exceeds the threshold (e.g. 0.1 eV) for re-evaluation."""
    return np.flatnonzero(np.asarray(std) > threshold)


def split_conformal_halfwidth(cal_pred, cal_true, alpha=0.1) -> float:
    """Half-width q of the split-conformal interval [pred - q, pred + q] with coverage >= 1 - alpha
    under exchangeability of calibration and test data."""
    r = np.sort(np.abs(np.asarray(cal_true, float) - np.asarray(cal_pred, float)))
    n = len(r)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float("inf")
    return float(r[k - 1])


def kl_gaussian(mu_q, var_q, mu_p, var_p) -> float:
    """KL( N(mu_q, var_q) || N(mu_p, var_p) ), elementwise then summed."""
    mu_q, var_q, mu_p, var_p = (np.asarray(x, float) for x in (mu_q, var_q, mu_p, var_p))
    return float(np.sum(0.5 * (np.log(var_p / var_q) + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0)))


def active_loss(pred, target, mu_q, var_q, mu_p, var_p, beta=1.0) -> float:
    """L_active = MSE + beta * KL(q || p) (Eq. 8) for Gaussian q and p."""
    mse = float(np.mean((np.asarray(pred, float) - np.asarray(target, float)) ** 2))
    return mse + beta * kl_gaussian(mu_q, var_q, mu_p, var_p)
