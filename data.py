"""Conversion between Crystal objects and padded torch batches."""
from __future__ import annotations

import numpy as np
import torch

from .crystal import Crystal


def crystals_to_batch(crystals: list[Crystal], device="cpu") -> dict:
    B, N = len(crystals), max(len(c.species) for c in crystals)
    frac = np.zeros((B, N, 3), dtype=np.float32)
    species = np.zeros((B, N), dtype=np.int64)
    mask = np.zeros((B, N), dtype=bool)
    for i, c in enumerate(crystals):
        n = len(c.species)
        frac[i, :n], species[i, :n], mask[i, :n] = c.frac % 1.0, c.species, True
    return {
        "lengths": torch.tensor(np.stack([c.lengths for c in crystals]), dtype=torch.float32, device=device),
        "angles": torch.tensor(np.stack([c.angles for c in crystals]), dtype=torch.float32, device=device),
        "frac": torch.tensor(frac, device=device), "species": torch.tensor(species, device=device),
        "mask": torch.tensor(mask, device=device),
    }


def batch_to_crystals(batch: dict) -> list[Crystal]:
    out = []
    for i in range(batch["mask"].shape[0]):
        m = batch["mask"][i].cpu().numpy()
        out.append(Crystal(batch["lengths"][i].cpu().numpy(), batch["angles"][i].cpu().numpy(),
                           batch["frac"][i].cpu().numpy()[m], batch["species"][i].cpu().numpy()[m]))
    return out


def lattice_stats(crystals: list[Crystal]):
    """Mean/std of the encoded lattice parameters for model.set_lattice_stats."""
    enc = np.stack([np.concatenate([np.log(c.lengths), c.angles / 90.0 - 1.0]) for c in crystals])
    return torch.tensor(enc.mean(0), dtype=torch.float32), torch.tensor(enc.std(0) + 1e-6, dtype=torch.float32)
