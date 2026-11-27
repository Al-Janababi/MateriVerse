import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from materiverse_eval import ablation
from materiverse_eval.adapter import RunWriter
from materiverse_eval.cli import main
from materiverse_eval.split import group_split
from materiverse_eval.synthetic import make_synthetic_logs

CFG = str(Path(__file__).resolve().parents[1] / "configs" / "ablation.yaml")


class TestSplit(unittest.TestCase):
    def test_groups_never_cross_splits_and_is_deterministic(self):
        ids = [f"s{i}" for i in range(20000)]
        groups = [f"g{i % 5000}" for i in range(20000)]
        a = group_split(ids, groups, seed=0)
        b = group_split(ids, groups, seed=0)
        self.assertEqual(a, b)
        by_group = {}
        for i, g in zip(ids, groups):
            by_group.setdefault(g, set()).add(a[i])
        self.assertTrue(all(len(v) == 1 for v in by_group.values()))
        frac = pd.Series(list(a.values())).value_counts(normalize=True)
        self.assertAlmostEqual(frac["train"], 0.8, delta=0.03)
        self.assertAlmostEqual(frac["validation"], 0.1, delta=0.03)
        self.assertAlmostEqual(frac["test"], 0.1, delta=0.03)

    def test_bad_fractions(self):
        with self.assertRaises(ValueError):
            group_split(["a"], ["g"], fractions=(0.5, 0.2, 0.2))


class TestAdapter(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            w = RunWriter("full", 3)
            w.add_candidate("c1", True, 12.0, 0.3, 123, 123, 64.0)
            p = w.write(os.path.join(d, "r.json"), 1.5)
            log = json.load(open(p))
            self.assertEqual(log["variant"], "full"); self.assertEqual(log["seed"], 3)
            self.assertNotIn("_synthetic", log)

    def test_rejects_bad_values(self):
        w = RunWriter("full", 0)
        with self.assertRaises(ValueError):
            w.add_candidate("c", True, 1.0, -1.0)
        with self.assertRaises(ValueError):
            w.write("x.json", 0)


class TestPipeline(unittest.TestCase):
    def test_refuses_synthetic_without_flag(self):
        with tempfile.TemporaryDirectory() as d:
            make_synthetic_logs(os.path.join(d, "runs"))
            rc = main(["summarize", "--logs", os.path.join(d, "runs"), "--out", os.path.join(d, "out"), "--config", CFG])
            self.assertEqual(rc, 2)
            self.assertFalse(os.path.exists(os.path.join(d, "out", "summary.csv")))

    def test_end_to_end_with_synthetic_is_watermarked(self):
        with tempfile.TemporaryDirectory() as d:
            make_synthetic_logs(os.path.join(d, "runs"))
            out = os.path.join(d, "out")
            rc = main(["summarize", "--logs", os.path.join(d, "runs"), "--out", out, "--config", CFG, "--allow-synthetic"])
            self.assertEqual(rc, 0)
            names = set(os.listdir(out))
            for n in ("SYNTHETIC_summary.csv", "SYNTHETIC_comparisons.csv", "SYNTHETIC_ablation_table.tex",
                      "SYNTHETIC_ablation_figure.png", "SYNTHETIC_report.md"):
                self.assertIn(n, names)
            self.assertIn("SYNTHETIC TEST DATA", open(os.path.join(out, "SYNTHETIC_ablation_table.tex")).read())
            s = pd.read_csv(os.path.join(out, "SYNTHETIC_summary.csv")).set_index("variant")
            self.assertGreater(s.loc["full", "stability_rate_mean"], s.loc["no_physical_rewards", "stability_rate_mean"])
            c = pd.read_csv(os.path.join(out, "SYNTHETIC_comparisons.csv"))
            self.assertIn("t_p_holm", c.columns)
            self.assertTrue((c["t_p_holm"] >= c["t_p"] - 1e-12).all())

    def test_duplicate_variant_seed_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("a.json", "b.json"):
                w = RunWriter("full", 0); w.add_candidate("c", True, 1.0, 0.1, 1, 1)
                w.write(os.path.join(d, name), 1.0)
            logs = ablation.discover_logs(d)
            with self.assertRaises(ValueError):
                ablation.per_run_table(logs)

    def test_few_seeds_warning(self):
        with tempfile.TemporaryDirectory() as d:
            make_synthetic_logs(os.path.join(d, "runs"), seeds=(0, 1, 2))
            logs = ablation.discover_logs(os.path.join(d, "runs"))
            per_run = ablation.per_run_table(logs)
            _, warns = ablation.comparison_table(per_run, "full")
            self.assertTrue(any("paired seeds" in w for w in warns))


if __name__ == "__main__":
    unittest.main()
