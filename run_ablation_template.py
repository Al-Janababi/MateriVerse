"""TEMPLATE: wrap YOUR trained MateriVerse code and write one run log per (variant, seed).

This file contains no model. Replace `run_variant` with calls into your own implementation.
Every variant must use the same seeds and the same data split (configs/ablation.yaml).
"""
import time
import yaml
from materiverse_eval.adapter import RunWriter


def run_variant(variant: str, seed: int):
    """Return (candidates, test_predictions) produced by YOUR code for this variant and seed.

    candidates: iterable of dicts with keys id, valid, e_hull_mev_per_atom, dft_cpu_hours,
                and optionally target_space_group, generated_space_group, predicted_tc_k
    test_predictions: iterable of dicts with keys id and (tc_pred_k, tc_ref_k) and/or (stable_score, stable_label)
    """
    raise NotImplementedError("connect this function to the study code")


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/ablation.yaml", encoding="utf-8"))
    for variant in list(cfg["variants"]) + list(cfg.get("baselines", {})):
        for seed in cfg["seeds"]:
            t0 = time.time()
            cands, preds = run_variant(variant, seed)
            hours = (time.time() - t0) / 3600.0      # or the wall-clock time recorded by your scheduler
            w = RunWriter(variant, seed)
            for c in cands:
                w.add_candidate(**c)
            for p in preds:
                w.add_test_prediction(**p)
            w.write(f"runs/{variant}_seed{seed}.json", wall_clock_hours=hours)
