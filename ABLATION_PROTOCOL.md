# Ablation protocol

Run every variant in `configs/ablation.yaml` with identical seeds, data split, hardware class and compute budget.

| Variant | Change relative to the full framework |
|---|---|
| `full` | none |
| `no_pauli_attention` | remove the Pauli-aware attention bias |
| `no_projective_layers` | remove the projective symmetry layers |
| `no_physical_rewards` | no in-process rewards; post hoc DFT screening only |
| `scalarised_objective` | weighted-sum objective instead of Pareto sampling |
| `no_active_learning` | remove the Bayesian active-learning loop |

Rules for reporting:

1. Report mean ± SD and the 95% CI over seeds (at least six); never report a single run.
2. Compare each variant with `full` using paired tests on matched seeds; report Holm-adjusted p-values and Cohen's d_z.
3. Report every variant, including those that do not hurt performance.
4. Record seeds, software versions, hardware and wall-clock time for each run (`docs/REPRODUCIBILITY.md`).
5. External baselines (`baseline_*`) must use the same split and the stability definition of Section 4.1;
   state their source (re-run by the authors, or literature values with the original protocol).
