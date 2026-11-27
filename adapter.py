"""Helper that the study code uses to write run logs in the format expected by the toolkit."""
from __future__ import annotations

import json
from pathlib import Path


class RunWriter:
    """Collect candidates of one (variant, seed) run and write a schema-conformant JSON log.

    Example
    -------
    >>> w = RunWriter("full", seed=0)
    >>> w.add_candidate("c0001", valid=True, e_hull_mev_per_atom=12.0, dft_cpu_hours=0.3,
    ...                 target_space_group=123, generated_space_group=123, predicted_tc_k=64.0)
    >>> w.write("runs/full_seed0.json", wall_clock_hours=72.0)   # doctest: +SKIP
    """

    def __init__(self, variant: str, seed: int, synthetic: bool = False):
        self.variant, self.seed, self.synthetic = variant, int(seed), bool(synthetic)
        self.candidates: list[dict] = []
        self.test_predictions: list[dict] = []

    def add_candidate(self, id, valid, e_hull_mev_per_atom, dft_cpu_hours,
                      target_space_group=None, generated_space_group=None, predicted_tc_k=None):
        if dft_cpu_hours < 0:
            raise ValueError("dft_cpu_hours must be >= 0")
        c = {"id": str(id), "valid": bool(valid), "e_hull_mev_per_atom": float(e_hull_mev_per_atom),
             "dft_cpu_hours": float(dft_cpu_hours)}
        if target_space_group is not None:
            c["target_space_group"] = int(target_space_group)
        if generated_space_group is not None:
            c["generated_space_group"] = int(generated_space_group)
        if predicted_tc_k is not None:
            c["predicted_tc_k"] = float(predicted_tc_k)
        self.candidates.append(c)

    def add_test_prediction(self, id, tc_pred_k=None, tc_ref_k=None, stable_score=None, stable_label=None):
        p = {"id": str(id)}
        if tc_pred_k is not None and tc_ref_k is not None:
            p["tc_pred_k"], p["tc_ref_k"] = float(tc_pred_k), float(tc_ref_k)
        if stable_score is not None and stable_label is not None:
            p["stable_score"], p["stable_label"] = float(stable_score), int(stable_label)
        self.test_predictions.append(p)

    def write(self, path, wall_clock_hours: float):
        if wall_clock_hours <= 0:
            raise ValueError("wall_clock_hours must be > 0")
        log = {"variant": self.variant, "seed": self.seed, "wall_clock_hours": float(wall_clock_hours),
               "candidates": self.candidates}
        if self.test_predictions:
            log["test_predictions"] = self.test_predictions
        if self.synthetic:
            log["_synthetic"] = True
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(log, f)
        return path
