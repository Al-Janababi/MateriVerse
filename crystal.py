"""Crystal data structure and periodic geometry (NumPy reference)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def lattice_matrix(lengths, angles_deg) -> np.ndarray:
    """Rows are the lattice vectors a1, a2, a3 (Angstrom), standard orientation."""
    a, b, c = (float(x) for x in lengths)
    al, be, ga = (np.deg2rad(float(x)) for x in angles_deg)
    cos_al, cos_be, cos_ga, sin_ga = np.cos(al), np.cos(be), np.cos(ga), np.sin(ga)
    v1 = [a, 0.0, 0.0]
    v2 = [b * cos_ga, b * sin_ga, 0.0]
    cx = c * cos_be
    cy = c * (cos_al - cos_be * cos_ga) / sin_ga
    cz2 = c * c - cx * cx - cy * cy
    if cz2 <= 0:
        raise ValueError("invalid lattice angles (non-positive volume)")
    return np.array([v1, v2, [cx, cy, np.sqrt(cz2)]], dtype=float)


def lengths_angles(matrix) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(matrix, dtype=float)
    lengths = np.linalg.norm(m, axis=1)
    def ang(i, j):
        return np.degrees(np.arccos(np.clip(m[i] @ m[j] / (lengths[i] * lengths[j]), -1.0, 1.0)))
    return lengths, np.array([ang(1, 2), ang(0, 2), ang(0, 1)])


@dataclass
class Crystal:
    lengths: np.ndarray      # (3,) Angstrom
    angles: np.ndarray       # (3,) degrees (alpha, beta, gamma)
    frac: np.ndarray         # (N, 3) fractional coordinates
    species: np.ndarray      # (N,) atomic numbers

    def __post_init__(self):
        self.lengths = np.asarray(self.lengths, dtype=float)
        self.angles = np.asarray(self.angles, dtype=float)
        self.frac = np.asarray(self.frac, dtype=float)
        self.species = np.asarray(self.species, dtype=int)
        if self.frac.shape != (len(self.species), 3):
            raise ValueError("frac must have shape (N, 3) matching species")

    @property
    def lattice(self) -> np.ndarray:
        return lattice_matrix(self.lengths, self.angles)

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.lattice)))

    def wrapped(self) -> "Crystal":
        return Crystal(self.lengths, self.angles, self.frac % 1.0, self.species)


def min_image_frac_displacement(frac) -> np.ndarray:
    """d[i, j] = wrapped(frac[j] - frac[i]) in [-0.5, 0.5)^3, shape (N, N, 3)."""
    f = np.asarray(frac, dtype=float)
    d = f[None, :, :] - f[:, None, :]
    return d - np.round(d)


def min_image_distance_matrix(frac, lattice) -> np.ndarray:
    """Pairwise distances (Angstrom) using the minimum-image convention on fractional displacements.

    Exact for reasonably reduced cells; for strongly skewed cells use a lattice-reduced cell.
    """
    d = min_image_frac_displacement(frac)
    cart = d @ np.asarray(lattice, dtype=float)
    return np.linalg.norm(cart, axis=-1)
