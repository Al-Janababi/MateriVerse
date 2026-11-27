"""Cox proportional-hazards model for synthesis outcomes (Eq. 9), Breslow partial likelihood."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def neg_log_partial_likelihood(beta, X, time, event, l2=0.0) -> float:
    X, time, event = np.asarray(X, float), np.asarray(time, float), np.asarray(event, int)
    eta = X @ beta
    order = np.argsort(-time)                      # descending time: risk set = cumulative sum
    eta_o, ev_o, t_o = eta[order], event[order], time[order]
    max_eta = eta_o.max()
    exp_o = np.exp(eta_o - max_eta)
    cum = np.cumsum(exp_o)
    # handle tied times: the risk set of a tied group includes every member with that time
    last_idx = {}
    for i, t in enumerate(t_o):
        last_idx[t] = i
    cum_tied = np.array([cum[last_idx[t]] for t in t_o])
    ll = np.sum(ev_o * (eta_o - max_eta - np.log(cum_tied)))
    return float(-ll + 0.5 * l2 * np.sum(beta**2))


def fit_cox(X, time, event, l2=1e-6):
    X = np.asarray(X, float)
    res = minimize(neg_log_partial_likelihood, np.zeros(X.shape[1]), args=(X, time, event, l2), method="BFGS")
    return res.x
