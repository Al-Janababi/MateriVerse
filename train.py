"""Minimal training loop for CrystalDiT."""
from __future__ import annotations

import torch

from .diffusion_torch import DiffusionSchedule, training_loss


def fit(model, batches, steps: int, lr: float = 1e-4, device="cpu", T: int = 1000, log_every: int = 100, grad_clip=1.0):
    """batches: iterable (re-iterated as needed) of batch dicts from data.crystals_to_batch."""
    model.to(device).train()
    schedule = DiffusionSchedule(T=T, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    it, history = iter(batches), []
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(batches)
            batch = next(it)
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, parts = training_loss(model, batch, schedule)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        history.append(float(loss.item()))
        if step % log_every == 0:
            print(f"step {step}: loss={history[-1]:.4f} {parts}")
    return history
