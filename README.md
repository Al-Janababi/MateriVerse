# MateriVerse — reference implementation and evaluation toolkit

Companion repository for the article *MateriVerse: A Physics-Grounded Diffusion Transformer Framework for the
Computational Discovery of Candidate High-Temperature Superconductors*.

> **Read this first.** This repository is a **reference re-implementation** written to document the method. It is
> **not** the code that produced the numbers reported in the article, and running it does **not** reproduce them:
> doing so requires training on the 824,516-structure corpus, DFT calculations and a T_c surrogate that are not
> part of this repository. The tables of the article must be regenerated from your own run logs with the
> evaluation toolkit (`docs/RUN_LOG_FORMAT.md`). This repository contains no ablation results.

## Status of each component

| Component of the article | Status in this repository |
|---|---|
| Metrics, statistics, ablation tables, group-aware split (`materiverse_eval`) | Implemented, unit-tested |
| Periodic crystal geometry, lattice conversion (`crystal.py`) | Implemented, unit-tested |
| Cosine / wrapped-normal diffusion utilities (`diffusion.py`) | Implemented, unit-tested (NumPy reference) |
| Pauli-motivated attention bias (`pauli.py`) | Geometric instantiation (hard-core exclusion); unit-tested |
| Symmetry loss, Eq. 10 (`symmetry.py`) | Implemented, unit-tested |
| Composite reward, Eq. 4 (`rewards.py`) | Implemented, unit-tested; evaluators are user-supplied |
| Pareto / NSGA-II selection, Eqs. 6 (`pareto.py`) | Implemented, unit-tested |
| Optimal-transport coupling, Eq. 7 (`ot.py`) | Sinkhorn with a (T_c, E_hull) cost; unit-tested |
| GP (Matérn-5/2), conformal intervals, Gaussian KL, Eq. 8 (`active_learning.py`) | Implemented, unit-tested |
| Cox survival model, Eq. 9 (`survival.py`) | Implemented, unit-tested (recovers simulated coefficients) |
| Extrapolation-aware T_c reward, Eq. 11 (`guards.py`) | Implemented, unit-tested; evaluated in a controlled **synthetic** study (`experiments/`, `docs/EXPERIMENT_SURROGATE_EXPLOITATION.md`); not used in the article's main benchmark |
| Crystal diffusion transformer, Eqs. 1–2 (`models/dit.py`, `diffusion_torch.py`, `train.py`) | Implemented in PyTorch; **not executed in the authoring environment** (no PyTorch available); parity/invariance tests provided in `tests/test_torch_parity.py` |
| Physical-reward fine-tuning (`rl.py`) | **Reward-weighted regression** instead of the PPO-style policy gradient described in the article |
| Force-field / pymatgen adapters (`evaluators.py`) | Interfaces only; not tested; require `ase`, `mace-torch`, `pymatgen` |
| Projective symmetry layers | **Not implemented** (symmetry is encouraged through the symmetry loss only) |
| Hamiltonian-fidelity term, neural wavefunction, learned exchange–correlation functional (Eqs. 1, 5) | **Not implemented** |
| Quantum-annealing-assisted Nash optimisation | **Not implemented** |
| Neural Eliashberg T_c layer, DFT (QUANTUM ESPRESSO) workflows | **Not included**; supply your own T_c surrogate and DFT results |
| Experimental validation (Stage 5) | Protocol only (article, Section 3.7) |

Architecture note: the network uses rotation-invariant lattice parameters, translation-invariant relative
displacements and permutation-equivariant attention. This is an invariant-features design, not a steerable SE(3)
network; describe it accordingly in the article.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[torch]"
python -m unittest discover -s tests -v      # NumPy tests run anywhere; torch tests need PyTorch
```

**Run the torch tests first on your machine** (`python -m unittest tests.test_torch_parity -v`). They were written
but could not be executed where the repository was prepared. One of them (overfitting a single batch) is stochastic;
increase its step count if it fails marginally.

Evaluation of your runs:

```bash
materiverse-eval summarize --logs runs --out results --config configs/ablation.yaml
```

## Layout

```
src/materiverse/        model, diffusion, rewards, Pareto, OT, active learning, survival, symmetry
src/materiverse_eval/   metrics, statistics, ablation tables, split, CLI
configs/                ablation variants and seeds; data sources
docs/                   log format, ablation protocol, reproducibility, publishing
examples/               template that wraps your own code and writes run logs
experiments/            controlled synthetic study of surrogate exploitation (Section 4.8)
results/                outputs of that study (raw JSON, CSV summaries, figure)
scripts/                code that draws Figs. 4 and 5 of the article from its reported numbers
tests/                  unit tests
```

## Statistical note

With five paired seeds the smallest attainable two-sided exact Wilcoxon p-value is 0.0625, so significance at 5% is
impossible with that test; use at least six seeds (`configs/ablation.yaml` lists eight) and report the paired t-test
with effect sizes and confidence intervals.

## Licence and citation

MIT licence (`LICENSE`); citation metadata in `CITATION.cff`. Datasets are subject to the terms of the Materials
Project, OQMD and NOMAD.
