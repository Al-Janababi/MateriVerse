# Run log format

One JSON file per (variant, seed), validated against `schema/run_log.schema.json`.

```json
{
  "variant": "full",
  "seed": 0,
  "wall_clock_hours": 72.0,
  "candidates": [
    {"id": "c000001", "valid": true, "e_hull_mev_per_atom": 12.3, "dft_cpu_hours": 0.3,
     "target_space_group": 123, "generated_space_group": 123, "predicted_tc_k": 64.0}
  ],
  "test_predictions": [
    {"id": "t001", "tc_pred_k": 52.0, "tc_ref_k": 50.0, "stable_score": 0.9, "stable_label": 1}
  ]
}
```

Definitions (identical to Section 4.1 of the article):

* **throughput** = number of valid candidates / `wall_clock_hours`
* **stability rate** = fraction of valid candidates with `e_hull_mev_per_atom` < threshold (default 50; strict inequality)
* **space-group retention** = fraction of valid candidates with `generated_space_group == target_space_group`
* **DFT cost per candidate** = mean `dft_cpu_hours` over valid candidates
* **effective cost per stable candidate** = DFT cost per candidate / stability rate
* **T_c MAE / within tolerance** are computed on `test_predictions` from the held-out test split only

Use `materiverse_eval.adapter.RunWriter` from your own code to write these files.
