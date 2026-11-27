"""Multi-objective utilities: non-dominated sorting, crowding distance, NSGA-II selection."""
from __future__ import annotations

import numpy as np


def _to_min(F, maximize):
    F = np.asarray(F, dtype=float).copy()
    for j, mx in enumerate(maximize):
        if mx:
            F[:, j] = -F[:, j]
    return F


def dominates(a, b) -> bool:
    """a dominates b (minimisation): no worse in all objectives and strictly better in one."""
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_sort(F, maximize=None) -> list[list[int]]:
    F = _to_min(F, maximize if maximize is not None else [False] * np.shape(F)[1])
    n = len(F)
    dom_count = np.zeros(n, dtype=int)
    dominated = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                if dominates(F[i], F[j]):
                    dominated[i].append(j)
                elif dominates(F[j], F[i]):
                    dom_count[i] += 1
    fronts, current = [], [i for i in range(n) if dom_count[i] == 0]
    while current:
        fronts.append(current)
        nxt = []
        for i in current:
            for j in dominated[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    nxt.append(j)
        current = nxt
    return fronts


def crowding_distance(F) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    n, m = F.shape
    dist = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for j in range(m):
        order = np.argsort(F[:, j])
        dist[order[0]] = dist[order[-1]] = np.inf
        span = F[order[-1], j] - F[order[0], j]
        if span == 0:
            continue
        dist[order[1:-1]] += (F[order[2:], j] - F[order[:-2], j]) / span
    return dist


def nsga2_select(F, k, maximize=None) -> list[int]:
    """Indices of k individuals chosen by front rank, then crowding distance (larger preferred)."""
    F = np.asarray(F, dtype=float)
    chosen: list[int] = []
    for front in non_dominated_sort(F, maximize):
        if len(chosen) + len(front) <= k:
            chosen.extend(front)
        else:
            cd = crowding_distance(_to_min(F, maximize if maximize is not None else [False] * F.shape[1])[front])
            order = np.argsort(-cd, kind="stable")
            chosen.extend([front[i] for i in order[: k - len(chosen)]])
            break
    return chosen


def hypervolume_2d(F, ref, maximize=None) -> float:
    """Hypervolume dominated by a 2-objective set w.r.t. reference point (in the original objective space)."""
    maximize = maximize if maximize is not None else [False, False]
    P = _to_min(F, maximize)
    r = _to_min(np.asarray([ref], dtype=float), maximize)[0]
    P = P[np.all(P < r, axis=1)]
    if len(P) == 0:
        return 0.0
    P = P[np.argsort(P[:, 0])]
    hv, best_y = 0.0, r[1]
    for x, y in P:
        if y < best_y:
            hv += (r[0] - x) * (best_y - y)
            best_y = y
    return float(hv)
