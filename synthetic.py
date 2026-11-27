"""Synthetic run logs used ONLY to test the pipeline. They are flagged and watermarked everywhere."""
from __future__ import annotations

import numpy as np

from .adapter import RunWriter

# Arbitrary generator settings for testing; they carry no scientific meaning.
_SETTINGS = {
    "full": dict(p_stable=0.80, cost=0.4, p_sg=0.90, wall=10.0, n=300),
    "no_pauli_attention": dict(p_stable=0.74, cost=0.4, p_sg=0.88, wall=10.0, n=290),
    "no_physical_rewards": dict(p_stable=0.60, cost=2.0, p_sg=0.85, wall=10.0, n=150),
}


def make_synthetic_logs(out_dir, variants=None, seeds=(0, 1, 2, 3, 4, 5), rng_seed=123):
    rng = np.random.default_rng(rng_seed)
    variants = variants or list(_SETTINGS)
    paths = []
    for v in variants:
        s = _SETTINGS[v]
        for seed in seeds:
            w = RunWriter(v, seed, synthetic=True)
            for i in range(s["n"]):
                stable = rng.random() < s["p_stable"]
                e = rng.uniform(0, 49) if stable else rng.uniform(51, 200)
                tsg = int(rng.choice([123, 139, 221]))
                gsg = tsg if rng.random() < s["p_sg"] else tsg + 1
                w.add_candidate(f"{v}-{seed}-{i}", True, e, max(0.0, rng.normal(s["cost"], 0.05)),
                                tsg, gsg, rng.uniform(0, 100))
            for j in range(100):
                ref = rng.uniform(1, 130)
                label = int(rng.random() < 0.5)
                score = float(np.clip(rng.normal(0.7 if label else 0.3, 0.2), 0, 1))
                w.add_test_prediction(f"t{j}", ref + rng.normal(0, 4), ref, score, label)
            paths.append(w.write(f"{out_dir}/{v}_seed{seed}.json", s["wall"]))
    return paths
