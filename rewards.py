"""Composite reward R = R_stab + R_Tc + R_synth (Eq. 4) with pluggable evaluators.

The evaluators (force field, DFT, T_c surrogate, barrier model) are supplied by the user; this module
only turns their outputs into a bounded scalar reward and documents the shaping.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RewardConfig:
    hull_threshold_mev: float = 50.0       # stability criterion of Section 4.1
    tc_scale_k: float = 100.0              # T_c at which the T_c reward saturates (tanh scale)
    tc_cap_k: Optional[float] = None       # optional cap on credited T_c; see docs (guards surrogate extrapolation)
    barrier_sigma_ev: float = 0.1          # barrier-uncertainty criterion sigma_dG < 0.1 eV (Table 3)
    tc_uncertainty_penalty: float = 1.0    # weight of the relative-uncertainty penalty on the T_c term


def stability_reward(e_hull_mev, cfg: RewardConfig) -> np.ndarray:
    """+1 at E_hull = 0, linearly to 0 at the threshold, then down to -1 at twice the threshold (clipped)."""
    e = np.asarray(e_hull_mev, dtype=float)
    thr = cfg.hull_threshold_mev
    r = 1.0 - e / thr
    return np.clip(r, -1.0, 1.0)


def tc_reward(tc_pred_k, tc_std_k, cfg: RewardConfig) -> np.ndarray:
    tc = np.maximum(np.asarray(tc_pred_k, dtype=float), 0.0)
    if cfg.tc_cap_k is not None:
        tc = np.minimum(tc, cfg.tc_cap_k)
    base = np.tanh(tc / cfg.tc_scale_k)
    rel_unc = np.asarray(tc_std_k, dtype=float) / np.maximum(tc, 1.0)
    penalty = cfg.tc_uncertainty_penalty * np.clip(rel_unc, 0.0, 1.0)
    return base * (1.0 - penalty)


def synthesizability_reward(barrier_sigma_ev, cfg: RewardConfig) -> np.ndarray:
    """1 when sigma_dG <= criterion, Gaussian decay beyond it."""
    s = np.asarray(barrier_sigma_ev, dtype=float)
    excess = np.maximum(s - cfg.barrier_sigma_ev, 0.0) / cfg.barrier_sigma_ev
    return np.exp(-(excess**2))


def composite_reward(e_hull_mev, tc_pred_k, tc_std_k, barrier_sigma_ev, cfg: RewardConfig | None = None) -> dict:
    cfg = cfg or RewardConfig()
    rs = stability_reward(e_hull_mev, cfg)
    rt = tc_reward(tc_pred_k, tc_std_k, cfg)
    ry = synthesizability_reward(barrier_sigma_ev, cfg)
    return {"R_stab": rs, "R_Tc": rt, "R_synth": ry, "R": rs + rt + ry}
