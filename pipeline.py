"""Generate structures with a trained CrystalDiT, score them with evaluators, and compute rewards."""
from __future__ import annotations

import numpy as np
import torch

from .data import batch_to_crystals
from .diffusion_torch import DiffusionSchedule, sample
from .rewards import RewardConfig, composite_reward


def generate_and_score(model, n_samples, n_atoms_choices, ehull_evaluator, tc_evaluator, barrier_evaluator,
                       reward_cfg: RewardConfig | None = None, T: int = 1000, device="cpu", seed=0):
    """ehull_evaluator -> meV/atom; tc_evaluator -> (mean, std) in K; barrier_evaluator -> sigma_dG in eV."""
    g = torch.Generator(device=device).manual_seed(seed)
    schedule = DiffusionSchedule(T=T, device=device)
    n_atoms = torch.tensor(np.random.default_rng(seed).choice(n_atoms_choices, n_samples), device=device)
    model.eval()
    batch = sample(model, schedule, n_atoms, generator=g)
    crystals = batch_to_crystals(batch)
    ehull = np.asarray(ehull_evaluator(crystals), dtype=float)
    tc_mean, tc_std = tc_evaluator(crystals)
    sigma = np.asarray(barrier_evaluator(crystals), dtype=float)
    rewards = composite_reward(ehull, tc_mean, tc_std, sigma, reward_cfg)
    return crystals, {"e_hull_mev": ehull, "tc_pred_k": np.asarray(tc_mean), "tc_std_k": np.asarray(tc_std),
                      "barrier_sigma_ev": sigma, **rewards}
