"""Symmetry-violation loss (Eq. 10): sum over group elements of the deviation between g.x and x."""
from __future__ import annotations

import numpy as np

from .crystal import lattice_matrix


def apply_op(frac, rotation, translation) -> np.ndarray:
    """Fractional-coordinate action x -> R x + t (row-vector convention), wrapped to [0, 1)."""
    return (np.asarray(frac) @ np.asarray(rotation).T + np.asarray(translation)) % 1.0


def symmetry_loss(frac, species, ops, lattice) -> float:
    """Mean squared distance (Angstrom^2) from each transformed atom to its nearest same-species atom.

    ops: iterable of (R (3x3 in fractional basis), t (3,)). Zero for a structure invariant under all ops.
    The nearest-neighbour matching makes the loss invariant to atom ordering.
    """
    frac = np.asarray(frac, dtype=float)
    species = np.asarray(species)
    lat = np.asarray(lattice, dtype=float)
    same = species[:, None] == species[None, :]
    total, n_ops = 0.0, 0
    for R, t in ops:
        gx = apply_op(frac, R, t)
        d = gx[:, None, :] - frac[None, :, :]
        d -= np.round(d)
        dist2 = ((d @ lat) ** 2).sum(-1)
        dist2 = np.where(same, dist2, np.inf)
        total += float(dist2.min(axis=1).mean())
        n_ops += 1
    return total / max(n_ops, 1)
