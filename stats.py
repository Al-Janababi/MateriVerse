"""Statistics for comparing variants across seeds (Section 4.1 and 5.3)."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def summarize(values) -> dict:
    v = np.asarray([x for x in values if not (isinstance(x, float) and math.isnan(x))], dtype=float)
    n = len(v)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else float("nan")
    if n > 1:
        half = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n))
    else:
        half = float("nan")
    return {"n": n, "mean": mean, "sd": sd, "ci95_low": mean - half, "ci95_high": mean + half}


def min_wilcoxon_p(n: int) -> float:
    """Smallest attainable two-sided p of the exact Wilcoxon signed-rank test with n pairs."""
    return min(1.0, 2.0 / (2 ** n)) if n >= 1 else 1.0


def paired_comparison(reference, other) -> dict:
    """Paired comparison of two equally ordered samples (same seeds).

    Returns the mean difference (reference - other), paired t-test p, Cohen's d_z,
    and the exact Wilcoxon p (None when it cannot be computed).
    """
    a = np.asarray(reference, dtype=float)
    b = np.asarray(other, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired samples must have equal length")
    n = len(a)
    d = a - b
    out = {"n_pairs": n, "mean_diff": float(d.mean()) if n else float("nan")}
    if n < 2:
        out.update({"t_p": float("nan"), "cohen_dz": float("nan"), "wilcoxon_p": float("nan"), "min_wilcoxon_p": min_wilcoxon_p(n)})
        return out
    sd = d.std(ddof=1)
    out["cohen_dz"] = float(d.mean() / sd) if sd > 0 else (float("inf") if d.mean() != 0 else 0.0)
    if sd == 0:
        out["t_p"] = 0.0 if d.mean() != 0 else 1.0
    else:
        out["t_p"] = float(stats.ttest_rel(a, b).pvalue)
    try:
        out["wilcoxon_p"] = float(stats.wilcoxon(a, b, method="exact").pvalue)
    except ValueError:
        out["wilcoxon_p"] = float("nan")
    out["min_wilcoxon_p"] = min_wilcoxon_p(n)
    return out


def holm(pvalues) -> list[float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj.tolist()
