"""Tests for the PyTorch modules. They are skipped automatically when PyTorch is not installed.

These tests could NOT be executed in the environment where the repository was prepared (no PyTorch).
Run them first on your machine:  python -m unittest tests.test_torch_parity -v
"""
import unittest

import numpy as np

try:
    import torch
    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "PyTorch not installed")
class TestTorchModules(unittest.TestCase):
    def setUp(self):
        from materiverse.models.dit import CrystalDiT
        torch.manual_seed(0)
        self.K = 10
        self.model = CrystalDiT(num_species=self.K, dim=32, depth=2, heads=4, rbf_dim=8, edge_dim=16, pauli_mode="hard")

    def batch(self, B=3, N=5, full=False):
        g = torch.Generator().manual_seed(1)
        n_at = torch.full((B,), N) if full else torch.tensor([N, N - 1, N - 2][:B])
        mask = torch.arange(N)[None, :] < n_at[:, None]
        return {
            "lengths": 3.0 + 2.0 * torch.rand(B, 3, generator=g), "angles": 85.0 + 10.0 * torch.rand(B, 3, generator=g),
            "frac": torch.rand(B, N, 3, generator=g) * mask[..., None], "species": (torch.randint(1, self.K + 1, (B, N), generator=g)) * mask,
            "mask": mask,
        }

    def test_lattice_parity(self):
        from materiverse.crystal import lattice_matrix
        from materiverse.models.dit import lattice_matrix_torch
        L, A = [[3.9, 4.1, 12.4]], [[90.0, 88.0, 92.0]]
        got = lattice_matrix_torch(torch.tensor(L), torch.tensor(A))[0].numpy()
        self.assertTrue(np.allclose(got, lattice_matrix(L[0], A[0]), atol=1e-5))

    def test_wrapped_score_parity(self):
        from materiverse.diffusion import wrapped_normal_score
        from materiverse.diffusion_torch import wrapped_normal_score_torch
        d = np.linspace(-0.6, 0.6, 13)
        for sigma in (0.05, 0.2):
            ref = wrapped_normal_score(d, sigma)
            got = wrapped_normal_score_torch(torch.tensor(d, dtype=torch.float64), torch.tensor(sigma, dtype=torch.float64)).numpy()
            self.assertTrue(np.allclose(ref, got, atol=1e-8))

    def test_pauli_parity(self):
        from materiverse.models.dit import pauli_bias_torch
        from materiverse.pauli import pauli_bias
        dist = np.array([[0, 0.5, 3.0], [0.5, 0, 2.0], [3.0, 2.0, 0]])
        for mode in ("hard", "soft", "off"):
            ref = pauli_bias(dist, 1.0, mode)
            got = pauli_bias_torch(torch.tensor(dist, dtype=torch.float64)[None], 1.0, mode)[0].numpy()
            self.assertTrue(np.array_equal(np.isneginf(ref), np.isneginf(got)))
            fin = np.isfinite(ref)
            self.assertTrue(np.allclose(ref[fin], got[fin]))

    def test_forward_shapes_and_finite_with_padding_and_overlaps(self):
        b = self.batch()
        b["frac"][:, 1] = b["frac"][:, 0]                      # force overlapping atoms (triggers the hard Pauli mask)
        sp = torch.nn.functional.one_hot((b["species"] - 1).clamp(min=0), self.K).float() * b["mask"][..., None]
        lat = torch.randn(3, 6)
        o_s, o_l, o_c = self.model(sp, lat, b["frac"], torch.rand(3), b["mask"])
        self.assertEqual(o_s.shape, (3, 5, self.K)); self.assertEqual(o_l.shape, (3, 6)); self.assertEqual(o_c.shape, (3, 5, 3))
        for o in (o_s, o_l, o_c):
            self.assertTrue(torch.isfinite(o).all())

    def test_translation_invariance_and_permutation_equivariance(self):
        b = self.batch(full=True)
        sp = torch.nn.functional.one_hot(b["species"] - 1, self.K).float()
        lat, t = torch.randn(3, 6), torch.rand(3)
        self.model.eval()
        with torch.no_grad():
            base = self.model(sp, lat, b["frac"], t, b["mask"])
            shifted = self.model(sp, lat, (b["frac"] + torch.tensor([0.31, 0.12, 0.77])) % 1.0, t, b["mask"])
            perm = torch.randperm(5)
            permuted = self.model(sp[:, perm], lat, b["frac"][:, perm], t, b["mask"][:, perm])
        for x, y in zip(base, shifted):
            self.assertTrue(torch.allclose(x, y, atol=1e-4))
        self.assertTrue(torch.allclose(base[0][:, perm], permuted[0], atol=1e-4))
        self.assertTrue(torch.allclose(base[2][:, perm], permuted[2], atol=1e-4))
        self.assertTrue(torch.allclose(base[1], permuted[1], atol=1e-4))

    def test_loss_backward_and_overfit_single_batch(self):
        from materiverse.diffusion_torch import DiffusionSchedule, training_loss
        b = self.batch(full=True)
        sched = DiffusionSchedule(T=50)
        opt = torch.optim.Adam(self.model.parameters(), lr=3e-3)
        losses = []
        for _ in range(300):
            loss, _ = training_loss(self.model, b, sched)
            self.assertTrue(torch.isfinite(loss))
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        self.assertLess(np.mean(losses[-30:]), 0.9 * np.mean(losses[:30]))   # stochastic; raise steps if it flakes

    def test_sampling_shapes_and_ranges(self):
        from materiverse.diffusion_torch import DiffusionSchedule, sample
        out = sample(self.model, DiffusionSchedule(T=5), torch.tensor([3, 5]))
        self.assertEqual(out["frac"].shape, (2, 5, 3))
        self.assertTrue(((out["frac"] >= 0) & (out["frac"] < 1)).all())
        self.assertTrue(((out["species"][out["mask"]] >= 1) & (out["species"][out["mask"]] <= self.K)).all())
        self.assertTrue((out["species"][~out["mask"]] == 0).all())
        self.assertTrue(torch.isfinite(out["lengths"]).all() and torch.isfinite(out["angles"]).all())

    def test_rwr_weights(self):
        from materiverse.rl import rwr_weights
        w = rwr_weights([0.0, 1.0, 2.0], temperature=1.0)
        self.assertAlmostEqual(w.mean(), 1.0); self.assertTrue(np.all(np.diff(w) > 0))


if __name__ == "__main__":
    unittest.main()
