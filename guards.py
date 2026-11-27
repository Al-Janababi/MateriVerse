"""Extrapolation-aware rewards: a trust-region gate for surrogate-based T_c credit.

Problem. A reward that trusts a learned T_c surrogate everywhere can be exploited: a generator that
maximises the surrogate is pushed towards regions far from the training data, where the surrogate
extrapolates without control and over-predicts (reward over-optimisation).

Remedy proposed here (two parts, both model-agnostic):
  1. Support gate. Credit is multiplied by g(x) = exp(-max(0, d(x) - tau) / (rho * tau)), where d(x) is the
     mean distance to the k nearest training points in descriptor space and tau is the q-quantile of the
     leave-one-out distances of the training set itself. Inside the training support g = 1.
  2. Credit cap. Credited T_c is min(T_hat, T_cap), with T_cap the largest T_c observed in the training data.
The gated T_c reward is therefore  R_Tc^guard(x) = g(x) * tanh(min(T_hat(x), T_cap) / s).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class SupportGate:
    k: int = 5
    quantile: float = 0.95
    rho: float = 0.25
    tau_: float = float("nan")

    def fit(self, X_train) -> "SupportGate":
        X = np.asarray(X_train, dtype=float)
        if len(X) <= self.k + 1:
            raise ValueError("need more training points than k + 1")
        self._nn = NearestNeighbors(n_neighbors=self.k + 1).fit(X)
        dist, _ = self._nn.kneighbors(X)
        loo = dist[:, 1:].mean(axis=1)                  # drop the self-distance
        self.tau_ = float(np.quantile(loo, self.quantile))
        return self

    def distance(self, X) -> np.ndarray:
        dist, _ = self._nn.kneighbors(np.asarray(X, dtype=float), n_neighbors=self.k)
        return dist.mean(axis=1)

    def weight(self, X) -> np.ndarray:
        d = self.distance(X)
        return np.exp(-np.maximum(d - self.tau_, 0.0) / (self.rho * self.tau_))


def guarded_tc_reward(tc_pred, gate_weight, tc_cap, tc_scale=100.0) -> np.ndarray:
    """R_Tc^guard = g * tanh(min(T_hat, T_cap) / s); T_hat is clipped below at 0."""
    t = np.minimum(np.maximum(np.asarray(tc_pred, dtype=float), 0.0), tc_cap)
    return np.asarray(gate_weight, dtype=float) * np.tanh(t / tc_scale)
