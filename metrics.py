"""Metric definitions used in Section 4.1 of the article.

All metrics are computed from a *run log* (see docs/RUN_LOG_FORMAT.md). Nothing here
trains a model or runs DFT; it only evaluates logs produced by the study code.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def load_log(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        log = json.load(f)
    for key in ("variant", "seed", "wall_clock_hours", "candidates"):
        if key not in log:
            raise ValueError(f"{path}: missing required field '{key}'")
    if not log["wall_clock_hours"] > 0:
        raise ValueError(f"{path}: wall_clock_hours must be > 0")
    return log


def roc_auc(scores, labels) -> float:
    """Rank-based ROC-AUC (Mann-Whitney U), with average ranks for ties."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    u = ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def run_metrics(log: dict, stability_threshold: float = 50.0, tc_within_k: float = 10.0) -> dict:
    """Compute the per-run metrics of Table 5/6 of the article.

    throughput_per_h        valid candidates per hour of wall-clock time
    stability_rate          fraction of valid candidates with E_hull < threshold (meV/atom)
    space_group_retention   fraction of valid candidates whose generated space group equals the target
    dft_cost_per_candidate  mean DFT CPU-hours per valid candidate
    effective_cost_per_stable  dft_cost_per_candidate / stability_rate
    """
    cands = [c for c in log["candidates"] if c.get("valid", False)]
    n_valid = len(cands)
    out = {
        "variant": log["variant"],
        "seed": int(log["seed"]),
        "n_candidates": len(log["candidates"]),
        "n_valid": n_valid,
        "throughput_per_h": n_valid / float(log["wall_clock_hours"]),
    }
    if n_valid:
        e = np.array([c["e_hull_mev_per_atom"] for c in cands], dtype=float)
        out["stability_rate"] = float((e < stability_threshold).mean())
        out["dft_cost_per_candidate"] = float(np.mean([c["dft_cpu_hours"] for c in cands]))
        sg = [(c.get("target_space_group"), c.get("generated_space_group")) for c in cands
              if c.get("target_space_group") is not None and c.get("generated_space_group") is not None]
        out["space_group_retention"] = float(np.mean([t == g for t, g in sg])) if sg else float("nan")
        rate = out["stability_rate"]
        out["effective_cost_per_stable"] = out["dft_cost_per_candidate"] / rate if rate > 0 else float("nan")
        out["n_predicted_tc_above_77k"] = int(sum(1 for c in cands if c.get("predicted_tc_k", -1) > 77.0))
    else:
        for k in ("stability_rate", "dft_cost_per_candidate", "space_group_retention", "effective_cost_per_stable"):
            out[k] = float("nan")
        out["n_predicted_tc_above_77k"] = 0

    tp = log.get("test_predictions") or []
    tc = [(p["tc_pred_k"], p["tc_ref_k"]) for p in tp if "tc_pred_k" in p and "tc_ref_k" in p]
    if tc:
        err = np.abs(np.array([a - b for a, b in tc], dtype=float))
        out["tc_mae_k"] = float(err.mean())
        out["tc_within_tol"] = float((err < tc_within_k).mean())
    else:
        out["tc_mae_k"] = float("nan")
        out["tc_within_tol"] = float("nan")
    st = [(p["stable_score"], p["stable_label"]) for p in tp if "stable_score" in p and "stable_label" in p]
    out["stability_roc_auc"] = roc_auc([a for a, _ in st], [b for _, b in st]) if st else float("nan")
    out["synthetic"] = bool(log.get("_synthetic", False))
    return out


METRIC_COLUMNS = [
    "throughput_per_h", "stability_rate", "space_group_retention", "dft_cost_per_candidate",
    "effective_cost_per_stable", "tc_mae_k", "tc_within_tol", "stability_roc_auc",
]
