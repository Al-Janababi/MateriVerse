# Reproducibility checklist

- [ ] Data snapshot: database versions and download dates for Materials Project, OQMD and NOMAD
- [ ] Split file produced with `materiverse-eval split` (seed recorded)
- [ ] Python, PyTorch, QUANTUM ESPRESSO, MACE-MP-0 versions; pseudopotentials; k-point and cutoff settings
- [ ] Hyperparameters of every stage (config files committed)
- [ ] Random seeds for all runs (at least six per variant)
- [ ] Hardware (CPU/GPU model, number of cores) and wall-clock time per run
- [ ] Run logs (`runs/*.json`) archived with the results
- [ ] Generated candidate structures (CIF files) archived
- [ ] Trained model weights archived (or reason for withholding stated)

## Scope of the released code

State precisely in the article which code produced which result. This repository is a reference
re-implementation (see README, "Status of each component"). If the results of the article were produced by other
code, release that code as well, or state that it is not available.
