"""Training loss and sampler for CrystalDiT (PyTorch); mirrors the NumPy references in diffusion.py."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .diffusion import cosine_alpha_bar, geometric_sigmas
from .models.dit import decode_lattice, encode_lattice


class DiffusionSchedule:
    def __init__(self, T=1000, sigma_min=0.005, sigma_max=0.5, device="cpu"):
        self.T = T
        self.alpha_bar = torch.tensor(cosine_alpha_bar(T), dtype=torch.float32, device=device)
        self.sigmas = torch.tensor(geometric_sigmas(T, sigma_min, sigma_max), dtype=torch.float32, device=device)


def wrapped_normal_score_torch(d: torch.Tensor, sigma: torch.Tensor, K: int = 10) -> torch.Tensor:
    """sigma must broadcast against d (e.g. shape (B,1,1)). Mirrors diffusion.wrapped_normal_score."""
    ks = torch.arange(-K, K + 1, device=d.device, dtype=d.dtype)
    z = d[..., None] + ks
    s = sigma[..., None]
    logw = -0.5 * (z / s) ** 2
    logw = logw - logw.max(dim=-1, keepdim=True).values
    w = torch.exp(logw)
    return (-(z / s ** 2) * w).sum(-1) / w.sum(-1)


def species_to_onehot(species: torch.Tensor, mask: torch.Tensor, num_species: int) -> torch.Tensor:
    """species holds atomic numbers Z (1..num_species); padding positions are ignored via mask."""
    idx = (species - 1).clamp(min=0, max=num_species - 1)
    return F.one_hot(idx, num_species).float() * mask[..., None].float()


def training_loss(model, batch, schedule, per_sample=False, weights=(1.0, 1.0, 1.0)):
    """Denoising loss: eps-MSE for species and lattice, sigma-scaled wrapped-score MSE for coordinates.

    batch: dict with lengths (B,3), angles (B,3), frac (B,N,3), species (B,N) atomic numbers, mask (B,N) bool.
    """
    device = batch["frac"].device
    mask = batch["mask"]
    B, N = mask.shape
    K = model.num_species
    t_idx = torch.randint(0, schedule.T, (B,), device=device)
    ab = schedule.alpha_bar[t_idx]
    sigma = schedule.sigmas[t_idx]
    sa, sb = ab.sqrt()[:, None, None], (1 - ab).sqrt()[:, None, None]

    x0 = species_to_onehot(batch["species"], mask, K)
    eps_s = torch.randn_like(x0)
    sp_t = (sa * x0 + sb * eps_s) * mask[..., None].float()

    lat0 = (encode_lattice(batch["lengths"], batch["angles"]) - model.lat_mean) / model.lat_std
    eps_l = torch.randn_like(lat0)
    lat_t = ab.sqrt()[:, None] * lat0 + (1 - ab).sqrt()[:, None] * eps_l

    sig = sigma[:, None, None]
    eps_f = torch.randn_like(batch["frac"])
    frac_t = (batch["frac"] + sig * eps_f) % 1.0
    d = frac_t - batch["frac"]
    d = d - torch.round(d)
    target_c = sig * wrapped_normal_score_torch(d, sig)

    out_s, out_l, out_c = model(sp_t, lat_t, frac_t, (t_idx + 1).float() / schedule.T, mask, batch.get("cond"))
    m = mask.float()
    n_at = m.sum(1).clamp(min=1.0)
    loss_s = (((out_s - eps_s) ** 2 * m[..., None]).sum((1, 2))) / (n_at * K)   # padded atoms excluded
    loss_l = ((out_l - eps_l) ** 2).mean(1)
    loss_c = (((out_c - target_c) ** 2 * m[..., None]).sum((1, 2))) / (n_at * 3)
    per = weights[0] * loss_s + weights[1] * loss_l + weights[2] * loss_c
    parts = {"species": loss_s.mean().item(), "lattice": loss_l.mean().item(), "coords": loss_c.mean().item()}
    return (per if per_sample else per.mean()), parts


@torch.no_grad()
def sample(model, schedule, n_atoms: torch.Tensor, generator: torch.Generator | None = None):
    """Ancestral sampling. Returns dict(lengths, angles, frac, species (atomic numbers), mask)."""
    device = n_atoms.device
    B, N, K, T = len(n_atoms), int(n_atoms.max()), model.num_species, schedule.T
    mask = torch.arange(N, device=device)[None, :] < n_atoms[:, None]
    mf = mask[..., None].float()
    rnd = lambda *s: torch.randn(*s, device=device, generator=generator)
    sp, lat = rnd(B, N, K) * mf, rnd(B, 6)
    frac = torch.rand(B, N, 3, device=device, generator=generator)
    for t in reversed(range(T)):
        tt = torch.full((B,), (t + 1) / T, device=device)
        eps_s, eps_l, out_c = model(sp, lat, frac, tt, mask)
        ab_t = schedule.alpha_bar[t]
        ab_prev = schedule.alpha_bar[t - 1] if t > 0 else torch.tensor(1.0, device=device)
        alpha_t = ab_t / ab_prev
        beta_t = 1 - alpha_t

        def ddpm(x, eps):
            mean = (x - beta_t / (1 - ab_t).sqrt() * eps) / alpha_t.sqrt()
            if t > 0:
                var = beta_t * (1 - ab_prev) / (1 - ab_t)
                mean = mean + var.sqrt() * torch.randn(x.shape, device=device, generator=generator)
            return mean

        sp, lat = ddpm(sp, eps_s) * mf, ddpm(lat, eps_l)
        s_t = schedule.sigmas[t]
        s_prev = schedule.sigmas[t - 1] if t > 0 else torch.tensor(0.0, device=device)
        gap = s_t ** 2 - s_prev ** 2
        step_std = torch.sqrt(torch.clamp(s_prev ** 2 * gap / s_t ** 2, min=0.0))
        frac = (frac + gap * out_c / s_t + step_std * torch.randn(frac.shape, device=device, generator=generator)) % 1.0
    lat_raw = lat * model.lat_std + model.lat_mean
    lengths, angles = decode_lattice(lat_raw)
    species = (sp.argmax(-1) + 1) * mask.long()
    return {"lengths": lengths, "angles": angles, "frac": frac, "species": species, "mask": mask}
