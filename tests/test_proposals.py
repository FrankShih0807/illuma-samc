"""Tests for proposal distributions."""

import torch

from illuma_samc.proposals import GaussianProposal, LangevinProposal


class TestGaussianProposal:
    def test_propose_shape(self):
        p = GaussianProposal(step_size=0.5)
        x = torch.zeros(10)
        x_new = p.propose(x)
        assert x_new.shape == x.shape

    def test_symmetric_log_ratio(self):
        p = GaussianProposal()
        x = torch.randn(5)
        x_new = p.propose(x)
        assert p.log_ratio(x, x_new) == 0.0

    def test_step_size_controls_spread(self):
        torch.manual_seed(0)
        small = GaussianProposal(step_size=0.01)
        large = GaussianProposal(step_size=10.0)
        x = torch.zeros(100)

        small_proposals = torch.stack([small.propose(x) for _ in range(100)])
        torch.manual_seed(0)
        large_proposals = torch.stack([large.propose(x) for _ in range(100)])

        assert small_proposals.std() < large_proposals.std()

    def test_proposals_centered_on_x(self):
        torch.manual_seed(42)
        p = GaussianProposal(step_size=1.0)
        x = torch.tensor([3.0, -2.0])
        proposals = torch.stack([p.propose(x) for _ in range(10000)])
        mean = proposals.mean(dim=0)
        assert torch.allclose(mean, x, atol=0.1)

    def test_adapt_reduces_step_when_all_rejected(self):
        """If every proposal is rejected, step size should decrease."""
        p = GaussianProposal(step_size=1.0, adapt=True, adapt_warmup=200)
        initial = p.step_size
        for _ in range(200):
            p.report_accept(False)
        assert p.step_size < initial * 0.5

    def test_adapt_increases_step_when_all_accepted(self):
        """If every proposal is accepted, step size should increase."""
        p = GaussianProposal(step_size=0.01, adapt=True, adapt_warmup=200)
        initial = p.step_size
        for _ in range(200):
            p.report_accept(True)
        assert p.step_size > initial * 2

    def test_adapt_freezes_after_warmup(self):
        """Step size should not change after warmup."""
        p = GaussianProposal(step_size=0.5, adapt=True, adapt_warmup=100)
        for _ in range(100):
            p.report_accept(True)
        frozen = p.step_size
        assert p.adapted
        # More calls should be no-ops
        for _ in range(100):
            p.report_accept(False)
        assert p.step_size == frozen

    def test_adapt_converges_near_target(self):
        """With 35% acceptance, step size should stay roughly stable."""
        import random

        random.seed(42)
        p = GaussianProposal(step_size=0.5, adapt=True, target_rate=0.35, adapt_warmup=2000)
        for _ in range(2000):
            p.report_accept(random.random() < 0.35)
        # Step size shouldn't have drifted wildly from initial
        assert 0.05 < p.step_size < 5.0

    def test_no_adapt_by_default(self):
        """report_accept is a no-op when adapt=False."""
        p = GaussianProposal(step_size=0.5)
        initial = p.step_size
        for _ in range(100):
            p.report_accept(False)
        assert p.step_size == initial


def _quadratic_energy(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sum(x**2)


class TestLangevinProposal:
    def test_propose_shape(self):
        p = LangevinProposal(_quadratic_energy, step_size=0.1)
        x = torch.randn(5)
        x_new = p.propose(x)
        assert x_new.shape == x.shape

    def test_drift_toward_mode(self):
        """Langevin should drift toward the mode (x=0 for quadratic energy)."""
        p = LangevinProposal(_quadratic_energy, step_size=0.1)

        torch.manual_seed(0)
        x = torch.tensor([5.0, 5.0])
        proposals = torch.stack([p.propose(x) for _ in range(1000)])
        mean = proposals.mean(dim=0)
        # Mean should be closer to 0 than x
        assert torch.norm(mean) < torch.norm(x)

    def test_log_ratio_nonzero(self):
        """Langevin is asymmetric — log_ratio should be nonzero."""
        p = LangevinProposal(_quadratic_energy, step_size=0.1)

        x = torch.tensor([1.0, 1.0])
        x_new = torch.tensor([0.5, 0.5])
        lr = p.log_ratio(x, x_new)
        # Should be nonzero for asymmetric proposal
        assert lr != 0.0

    def test_log_ratio_symmetric_at_mode(self):
        """At the mode, gradients are zero, so proposal is symmetric."""
        p = LangevinProposal(_quadratic_energy, step_size=0.1)

        x = torch.zeros(2)
        x_new = torch.tensor([0.01, -0.01])
        lr = p.log_ratio(x, x_new)
        # Near-zero because at the mode the drift is zero
        assert abs(lr) < 0.1
