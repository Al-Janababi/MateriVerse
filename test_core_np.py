import math
import unittest

import numpy as np

from materiverse import active_learning as al
from materiverse import diffusion as df
from materiverse import ot, pareto, pauli, rewards, survival, symmetry
from materiverse.crystal import Crystal, lattice_matrix, lengths_angles, min_image_distance_matrix, min_image_frac_displacement


class TestCrystal(unittest.TestCase):
    def test_cubic_lattice_and_volume(self):
        m = lattice_matrix([4, 4, 4], [90, 90, 90])
        self.assertTrue(np.allclose(m, np.diag([4, 4, 4]), atol=1e-12))
        c = Crystal([4, 4, 4], [90, 90, 90], [[0, 0, 0]], [29])
        self.assertAlmostEqual(c.volume, 64.0)

    def test_roundtrip_lengths_angles(self):
        L, A = [3.9, 4.1, 12.4], [90.0, 88.0, 92.0]
        l2, a2 = lengths_angles(lattice_matrix(L, A))
        self.assertTrue(np.allclose(l2, L)); self.assertTrue(np.allclose(a2, A))

    def test_triclinic_volume_formula(self):
        a, b, c, al, be, ga = 3.0, 4.0, 5.0, 80.0, 85.0, 95.0
        ca, cb, cg = (math.cos(math.radians(x)) for x in (al, be, ga))
        vol = a * b * c * math.sqrt(1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg)
        self.assertAlmostEqual(abs(np.linalg.det(lattice_matrix([a, b, c], [al, be, ga]))), vol, places=9)

    def test_min_image(self):
        frac = np.array([[0.1, 0, 0], [0.9, 0, 0]])
        d = min_image_frac_displacement(frac)
        self.assertAlmostEqual(d[0, 1, 0], -0.2); self.assertAlmostEqual(d[1, 0, 0], 0.2)
        dist = min_image_distance_matrix(frac, lattice_matrix([4, 4, 4], [90, 90, 90]))
        self.assertAlmostEqual(dist[0, 1], 0.8)

    def test_shape_validation(self):
        with self.assertRaises(ValueError):
            Crystal([4, 4, 4], [90, 90, 90], [[0, 0, 0]], [1, 2])


class TestDiffusion(unittest.TestCase):
    def test_cosine_schedule(self):
        ab = df.cosine_alpha_bar(1000)
        self.assertTrue(np.all(np.diff(ab) <= 0)); self.assertTrue(0 < ab[0] <= 1); self.assertLess(ab[-1], 1e-3)

    def test_sigmas(self):
        s = df.geometric_sigmas(10, 0.01, 1.0)
        self.assertAlmostEqual(s[0], 0.01); self.assertAlmostEqual(s[-1], 1.0); self.assertTrue(np.all(np.diff(s) > 0))

    def test_wrapped_score_small_sigma_matches_gaussian(self):
        self.assertAlmostEqual(float(df.wrapped_normal_score(0.005, 0.01)), -0.005 / 0.01**2, places=6)

    def test_wrapped_score_properties(self):
        d = np.array([0.05, 0.2, -0.3])
        s = df.wrapped_normal_score(d, 0.2)
        self.assertTrue(np.allclose(s, -df.wrapped_normal_score(-d, 0.2)))        # odd
        self.assertTrue(np.allclose(s, df.wrapped_normal_score(d + 1.0, 0.2)))    # periodic
        self.assertAlmostEqual(float(df.wrapped_normal_score(0.0, 0.2)), 0.0)

    def test_score_norm(self):
        self.assertAlmostEqual(df.wrapped_score_norm_sq(0.02) * 0.02**2, 1.0, delta=0.02)

    def test_noise_and_reverse_step(self):
        rng = np.random.default_rng(0)
        x0 = rng.random((100, 3))
        xt, eps, d = df.noise_coords(x0, 0.05, rng)
        self.assertTrue(np.all((xt >= 0) & (xt < 1)))
        self.assertTrue(np.allclose(d, 0.05 * eps - np.round(0.05 * eps), atol=1e-9))
        x_prev = df.reverse_coord_step(xt, np.zeros_like(xt), 0.05, 0.04, rng)
        self.assertTrue(np.all((x_prev >= 0) & (x_prev < 1)))


class TestPauli(unittest.TestCase):
    def setUp(self):
        self.dist = np.array([[0, 0.5, 3.0], [0.5, 0, 2.0], [3.0, 2.0, 0]])

    def test_hard_and_soft(self):
        h = pauli.pauli_bias(self.dist, 1.0, "hard")
        self.assertTrue(np.isneginf(h[0, 1])); self.assertEqual(h[0, 0], 0); self.assertEqual(h[0, 2], 0)
        s = pauli.pauli_bias(self.dist, 1.0, "soft")
        self.assertAlmostEqual(s[0, 1], -0.5); self.assertEqual(s[1, 1], 0.0)
        self.assertTrue(np.all(pauli.pauli_bias(self.dist, 1.0, "off") == 0))

    def test_hard_mask_zeroes_attention_weight(self):
        rng = np.random.default_rng(1)
        q = k = v = rng.standard_normal((3, 4))
        _, w = pauli.biased_softmax_attention(q, k, v, pauli.pauli_bias(self.dist, 1.0, "hard"))
        self.assertEqual(w[0, 1], 0.0); self.assertTrue(np.allclose(w.sum(axis=1), 1.0))


class TestSymmetry(unittest.TestCase):
    lat = lattice_matrix([4, 4, 4], [90, 90, 90])
    def test_cscl_invariants(self):
        frac = np.array([[0, 0, 0], [0.5, 0.5, 0.5]]); sp = np.array([55, 55])
        inv = (-np.eye(3), np.zeros(3)); tr = (np.eye(3), np.array([0.5, 0.5, 0.5]))
        self.assertAlmostEqual(symmetry.symmetry_loss(frac, sp, [inv, tr], self.lat), 0.0, places=12)

    def test_breaks_with_perturbation_and_species(self):
        tr = (np.eye(3), np.array([0.5, 0.5, 0.5]))
        frac = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
        self.assertGreater(symmetry.symmetry_loss(np.array([[0, 0, 0], [0.55, 0.5, 0.5]]), np.array([1, 1]), [tr], self.lat), 0.0)
        self.assertGreater(symmetry.symmetry_loss(frac, np.array([1, 2]), [tr], self.lat), 1.0)

    def test_permutation_invariance(self):
        rng = np.random.default_rng(3)
        frac = rng.random((6, 3)); sp = np.array([1, 1, 2, 2, 3, 3]); op = [(np.eye(3), np.array([0.1, 0.2, 0.3]))]
        a = symmetry.symmetry_loss(frac, sp, op, self.lat)
        p = rng.permutation(6)
        self.assertAlmostEqual(a, symmetry.symmetry_loss(frac[p], sp[p], op, self.lat), places=12)


class TestRewards(unittest.TestCase):
    cfg = rewards.RewardConfig()
    def test_stability(self):
        r = rewards.stability_reward([0, 50, 100, 500], self.cfg)
        self.assertTrue(np.allclose(r, [1, 0, -1, -1]))

    def test_tc_monotone_cap_and_uncertainty(self):
        r = rewards.tc_reward([20, 60, 120], [0, 0, 0], self.cfg)
        self.assertTrue(np.all(np.diff(r) > 0))
        capped = rewards.RewardConfig(tc_cap_k=100.0)
        self.assertAlmostEqual(float(rewards.tc_reward(200, 0, capped)), float(rewards.tc_reward(100, 0, capped)))
        self.assertLess(float(rewards.tc_reward(80, 40, self.cfg)), float(rewards.tc_reward(80, 0, self.cfg)))

    def test_synth_and_composite(self):
        self.assertAlmostEqual(float(rewards.synthesizability_reward(0.05, self.cfg)), 1.0)
        self.assertAlmostEqual(float(rewards.synthesizability_reward(0.2, self.cfg)), math.exp(-1))
        out = rewards.composite_reward(0.0, 100.0, 0.0, 0.05, self.cfg)
        self.assertAlmostEqual(float(out["R"]), 1.0 + math.tanh(1.0) + 1.0)


class TestPareto(unittest.TestCase):
    F = np.array([[1, 5], [2, 3], [3, 1], [3, 4], [4, 4]], dtype=float)
    def test_fronts(self):
        self.assertEqual([sorted(f) for f in pareto.non_dominated_sort(self.F)], [[0, 1, 2], [3], [4]])

    def test_maximize_flag(self):
        fronts = pareto.non_dominated_sort(self.F, maximize=[True, True])
        # maximising both: (1, 5) has the largest second objective and (4, 4) the largest first; neither dominates the other
        self.assertEqual(sorted(fronts[0]), [0, 4])

    def test_nsga2_prefers_extremes(self):
        self.assertEqual(sorted(pareto.nsga2_select(self.F, 2)), [0, 2])

    def test_hypervolume(self):
        self.assertAlmostEqual(pareto.hypervolume_2d(self.F[:3], (6, 6)), 19.0)


class TestOT(unittest.TestCase):
    def test_marginals_and_identity_plan(self):
        n = 5; a = b = np.full(n, 1 / n); x = np.arange(n) / 4.0
        C = (x[:, None] - (x[None, :] + 0.01)) ** 2
        P = ot.sinkhorn(a, b, C, eps=1e-3, n_iter=5000)
        self.assertTrue(np.allclose(P.sum(1), a, atol=1e-7)); self.assertTrue(np.allclose(P.sum(0), b, atol=1e-7))
        self.assertTrue(np.allclose(np.diag(P), 0.2, atol=1e-3))

    def test_cost_shape(self):
        C = ot.tc_hull_cost([10, 50], [5, 40], [30, 80, 100], [0, 10, 20])
        self.assertEqual(C.shape, (2, 3))


class TestActiveLearning(unittest.TestCase):
    def test_gp_uncertainty_grows_off_data(self):
        X = np.linspace(0, 5, 25)[:, None]; y = np.sin(X[:, 0])
        gp = al.MaternGP().fit(X, y)
        m, s = gp.predict(np.array([[2.5], [12.0]]))
        self.assertAlmostEqual(m[0], math.sin(2.5), delta=0.05)
        self.assertGreater(s[1], 5 * s[0])
        self.assertEqual(list(al.select_uncertain([0.05, 0.2, 0.11], 0.1)), [1, 2])

    def test_conformal_coverage(self):
        rng = np.random.default_rng(0)
        cal_pred = np.zeros(500); cal_true = rng.standard_t(5, 500)
        q = al.split_conformal_halfwidth(cal_pred, cal_true, 0.1)
        test = rng.standard_t(5, 20000)
        self.assertGreaterEqual(float(np.mean(np.abs(test) <= q)), 0.88)
        self.assertEqual(al.split_conformal_halfwidth([0] * 5, [1] * 5, 0.01), float("inf"))

    def test_kl_and_active_loss(self):
        self.assertAlmostEqual(al.kl_gaussian(0, 1, 0, 1), 0.0)
        self.assertAlmostEqual(al.kl_gaussian(1, 1, 0, 1), 0.5)
        self.assertAlmostEqual(al.active_loss([1, 2], [1, 4], 1, 1, 0, 1, beta=2.0), 2.0 + 1.0)


class TestSurvival(unittest.TestCase):
    def test_null_model_value(self):
        nll = survival.neg_log_partial_likelihood(np.zeros(1), np.zeros((3, 1)), [1, 2, 3], [1, 1, 1])
        self.assertAlmostEqual(nll, math.log(6.0), places=9)

    def test_recovers_coefficients(self):
        rng = np.random.default_rng(0)
        n = 4000; X = rng.standard_normal((n, 2)); beta = np.array([0.8, -0.5])
        T = rng.exponential(1.0 / np.exp(X @ beta)); C = rng.exponential(1.0 / 0.3, n)
        time, event = np.minimum(T, C), (T <= C).astype(int)
        est = survival.fit_cox(X, time, event)
        self.assertTrue(np.allclose(est, beta, atol=0.1), est)


if __name__ == "__main__":
    unittest.main()


class TestGuards(unittest.TestCase):
    def setUp(self):
        from materiverse.guards import SupportGate
        rng = np.random.default_rng(0)
        self.X = rng.standard_normal((800, 4))
        self.gate = SupportGate().fit(self.X)

    def test_weight_is_one_inside_and_decays_outside(self):
        inside = self.gate.weight(np.zeros((1, 4)))[0]
        far = self.gate.weight(np.full((1, 4), 8.0))[0]
        mid = self.gate.weight(np.full((1, 4), 3.5))[0]
        self.assertAlmostEqual(inside, 1.0)
        self.assertLess(far, 1e-6)
        self.assertTrue(far < mid < 1.0)

    def test_most_training_points_have_unit_weight(self):
        # tau is the 95% quantile of leave-one-out distances, so ~95% of training points are fully inside
        # (distances to training points include the point itself, which makes them slightly smaller)
        w = self.gate.weight(self.X)
        self.assertGreater(float(np.mean(w == 1.0)), 0.94)

    def test_guarded_reward_cap_and_gate(self):
        from materiverse.guards import guarded_tc_reward
        r = guarded_tc_reward([50, 300, 300, -10], [1, 1, 0.5, 1], tc_cap=130.0)
        self.assertAlmostEqual(r[0], np.tanh(0.5))
        self.assertAlmostEqual(r[1], np.tanh(1.3))                 # capped at 130 K
        self.assertAlmostEqual(r[2], 0.5 * np.tanh(1.3))           # gated
        self.assertAlmostEqual(r[3], 0.0)                          # negative predictions earn nothing

    def test_too_few_points(self):
        from materiverse.guards import SupportGate
        with self.assertRaises(ValueError):
            SupportGate(k=5).fit(np.zeros((5, 2)))


class TestExperimentSmoke(unittest.TestCase):
    def test_one_small_replicate_runs_and_has_expected_keys(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
        import surrogate_exploitation as se
        out = se.run_replicate(d=4, width=1.5, seed=0, n_train=200)
        self.assertEqual(set(out), set(se.VARIANTS))
        for v in out.values():
            self.assertTrue(np.isfinite(v["gap"]) and 0.0 <= v["frac_outside"] <= 1.0)
