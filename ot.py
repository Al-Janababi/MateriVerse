"""Entropic optimal transport (Sinkhorn) with a (T_c, E_hull) cost, our instantiation of Eq. 7."""
from __future__ import annotations

import numpy as np


def tc_hull_cost(tc, ehull, tc_target, ehull_target, tc_scale=100.0, ehull_scale=50.0) -> np.ndarray:
    """c[i, j] = ((Tc_i - Tc*_j)/s_T)^2 + ((E_i - E*_j)/s_E)^2 between generated i and target j."""
    tc, ehull = np.asarray(tc, float), np.asarray(ehull, float)
    tt, et = np.asarray(tc_target, float), np.asarray(ehull_target, float)
    return ((tc[:, None] - tt[None, :]) / tc_scale) ** 2 + ((ehull[:, None] - et[None, :]) / ehull_scale) ** 2


def sinkhorn(a, b, C, eps=0.05, n_iter=500, tol=1e-9) -> np.ndarray:
    """Transport plan with marginals a and b minimising <P, C> - eps * H(P)."""
    a, b, C = np.asarray(a, float), np.asarray(b, float), np.asarray(C, float)
    if abs(a.sum() - b.sum()) > 1e-9:
        raise ValueError("marginals must have equal mass")
    K = np.exp(-(C - C.min()) / eps)
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(n_iter):
        u_prev = u
        u = a / (K @ v)
        v = b / (K.T @ u)
        if np.max(np.abs(u - u_prev)) < tol:
            break
    return u[:, None] * K * v[None, :]
