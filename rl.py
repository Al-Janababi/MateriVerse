"""Reward-weighted fine-tuning of the diffusion model with physical rewards.

Simplification relative to the article: the article describes policy-gradient (PPO-style) optimisation.
Here we implement reward-weighted regression, a stable, likelihood-based alternative: sampled structures
are re-weighted by exp(reward / temperature) in the denoising loss. It uses the same reward interface
(`materiverse.rewards`) and can be replaced by a policy-gradient method without touching the rest.
"""
from __future__ import annotations

import numpy as np
import torch

from .diffusion_torch import training_loss


def rwr_weights(rewards, temperature: float = 1.0) -> np.ndarray:
    """Positive weights proportional to exp(r / T), normalised to mean 1."""
    r = np.asarray(rewards, dtype=float)
    w = np.exp((r - r.max()) / temperature)
    return w / w.sum() * len(r)


def rwr_step(model, optimizer, batch, schedule, rewards, temperature=1.0, grad_clip=1.0) -> float:
    per, _ = training_loss(model, batch, schedule, per_sample=True)
    w = torch.as_tensor(rwr_weights(rewards, temperature), dtype=per.dtype, device=per.device)
    loss = (w * per).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return float(loss.item())
