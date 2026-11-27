import math
import unittest

import numpy as np
from scipy import stats as sps

from materiverse_eval.metrics import roc_auc, run_metrics
from materiverse_eval.stats import holm, min_wilcoxon_p, paired_comparison, summarize


def fixture():
    cands = [
        {"id": "a", "valid": True, "e_hull_mev_per_atom": 10.0, "dft_cpu_hours": 0.2, "target_space_group": 1, "generated_space_group": 1, "predicted_tc_k": 90.0},
        {"id": "b", "valid": True, "e_hull_mev_per_atom": 49.9, "dft_cpu_hours": 0.4, "target_space_group": 2, "generated_space_group": 2, "predicted_tc_k": 10.0},
        {"id": "c", "valid": True, "e_hull_mev_per_atom": 50.0, "dft_cpu_hours": 0.2, "target_space_group": 3, "generated_space_group": 3},
        {"id": "d", "valid": True, "e_hull_mev_per_atom": 80.0, "dft_cpu_hours": 0.2, "target_space_group": 4, "generated_space_group": 5},
        {"id": "x", "valid": False, "e_hull_mev_per_atom": 0.0, "dft_cpu_hours": 9.0},
    ]
    tp = [{"id": "t1", "tc_pred_k": 52.0, "tc_ref_k": 50.0, "stable_score": 0.9, "stable_label": 1},
          {"id": "t2", "tc_pred_k": 38.0, "tc_ref_k": 50.0, "stable_score": 0.2, "stable_label": 0}]
    return {"variant": "v", "seed": 0, "wall_clock_hours": 2.0, "candidates": cands, "test_predictions": tp}


class TestMetrics(unittest.TestCase):
    def test_roc_auc_classic(self):
        self.assertAlmostEqual(roc_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]), 0.75)

    def test_roc_auc_ties(self):
        self.assertAlmostEqual(roc_auc([0.5, 0.5], [0, 1]), 0.5)

    def test_roc_auc_single_class_is_nan(self):
        self.assertTrue(math.isnan(roc_auc([0.1, 0.2], [1, 1])))

    def test_run_metrics_hand_computed(self):
        m = run_metrics(fixture(), stability_threshold=50.0, tc_within_k=10.0)
        self.assertEqual(m["n_valid"], 4)
        self.assertAlmostEqual(m["throughput_per_h"], 2.0)          # 4 valid / 2 h
        self.assertAlmostEqual(m["stability_rate"], 0.5)            # strict < 50: only 10 and 49.9
        self.assertAlmostEqual(m["space_group_retention"], 0.75)    # 3 of 4
        self.assertAlmostEqual(m["dft_cost_per_candidate"], 0.25)   # invalid candidate excluded
        self.assertAlmostEqual(m["effective_cost_per_stable"], 0.5)
        self.assertEqual(m["n_predicted_tc_above_77k"], 1)
        self.assertAlmostEqual(m["tc_mae_k"], 7.0)                  # errors 2 and 12
        self.assertAlmostEqual(m["tc_within_tol"], 0.5)
        self.assertAlmostEqual(m["stability_roc_auc"], 1.0)

    def test_no_valid_candidates(self):
        log = fixture(); log["candidates"] = [{"id": "x", "valid": False, "e_hull_mev_per_atom": 1.0, "dft_cpu_hours": 1.0}]
        m = run_metrics(log)
        self.assertEqual(m["throughput_per_h"], 0.0)
        self.assertTrue(math.isnan(m["stability_rate"]))


class TestStats(unittest.TestCase):
    def test_summarize(self):
        s = summarize([1.0, 2.0, 3.0])
        self.assertAlmostEqual(s["mean"], 2.0)
        self.assertAlmostEqual(s["sd"], 1.0)
        half = sps.t.ppf(0.975, 2) / math.sqrt(3)
        self.assertAlmostEqual(s["ci95_high"], 2.0 + half)

    def test_holm(self):
        self.assertEqual([round(x, 10) for x in holm([0.01, 0.04, 0.03])], [0.03, 0.06, 0.06])

    def test_min_wilcoxon(self):
        self.assertAlmostEqual(min_wilcoxon_p(5), 0.0625)
        self.assertAlmostEqual(min_wilcoxon_p(6), 0.03125)

    def test_paired_comparison_matches_scipy(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]); b = np.array([0.8, 1.7, 2.9, 3.5, 4.6, 5.1])
        r = paired_comparison(a, b)
        self.assertAlmostEqual(r["t_p"], float(sps.ttest_rel(a, b).pvalue))
        d = a - b
        self.assertAlmostEqual(r["cohen_dz"], d.mean() / d.std(ddof=1))
        self.assertAlmostEqual(r["wilcoxon_p"], 0.03125)  # all differences positive, n=6

    def test_unequal_lengths_rejected(self):
        with self.assertRaises(ValueError):
            paired_comparison([1, 2], [1])


if __name__ == "__main__":
    unittest.main()
