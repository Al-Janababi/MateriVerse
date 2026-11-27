"""Property evaluators (force field, DFT, T_c surrogate, barrier model) behind a common interface.

These adapters are NOT exercised by the unit tests (they need ASE, MACE or pymatgen and DFT codes that are
not part of this repository). They define the contract used by `pipeline.generate_and_score`.
"""
from __future__ import annotations

from typing import Callable, Protocol, Sequence

import numpy as np

from .crystal import Crystal, lattice_matrix


class Evaluator(Protocol):
    def __call__(self, crystals: Sequence[Crystal]) -> np.ndarray: ...


class CallableEvaluator:
    """Wrap any function crystals -> array (e.g. your T_c surrogate or barrier model)."""

    def __init__(self, fn: Callable[[Sequence[Crystal]], np.ndarray]):
        self.fn = fn

    def __call__(self, crystals):
        return np.asarray(self.fn(crystals), dtype=float)


def to_ase_atoms(c: Crystal):
    from ase import Atoms  # optional dependency
    return Atoms(numbers=c.species, cell=lattice_matrix(c.lengths, c.angles), scaled_positions=c.frac, pbc=True)


class MaceEnergyEvaluator:
    """Energy per atom (eV/atom) from a MACE-MP-0 foundation model. Requires `mace-torch` and `ase`."""

    def __init__(self, model: str = "medium", device: str = "cpu"):
        from mace.calculators import mace_mp  # optional dependency
        self.calc = mace_mp(model=model, device=device, default_dtype="float32")

    def __call__(self, crystals):
        out = []
        for c in crystals:
            atoms = to_ase_atoms(c)
            atoms.calc = self.calc
            out.append(atoms.get_potential_energy() / len(atoms))
        return np.asarray(out)


def ehull_mev_from_entries(crystal: Crystal, energy_per_atom_ev: float, reference_entries) -> float:
    """Energy above hull in meV/atom against reference entries (pymatgen ComputedEntry list)."""
    from pymatgen.analysis.phase_diagram import PhaseDiagram  # optional dependency
    from pymatgen.core import Composition
    from pymatgen.entries.computed_entries import ComputedEntry
    comp = Composition({int(z): int((crystal.species == z).sum()) for z in np.unique(crystal.species)})
    entry = ComputedEntry(comp, energy_per_atom_ev * comp.num_atoms)
    pd = PhaseDiagram(list(reference_entries) + [entry])
    return 1000.0 * float(pd.get_e_above_hull(entry))
