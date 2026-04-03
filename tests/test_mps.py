"""MPS (Apple Silicon GPU) compatibility smoke tests.

Skipped when MPS is not available. These verify that SAMC runs correctly
on MPS with float64 theta/counts, LangevinProposal autograd, and multi-chain.
"""

import pytest
import torch

MPS_AVAILABLE = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
pytestmark = pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS not available")


def _quadratic(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return 0.5 * torch.sum(x**2)
    return 0.5 * torch.sum(x**2, dim=-1)


class TestMPSSingleChain:
    def test_basic_samc_on_mps(self):
        from illuma_samc.sampler import SAMC

        sampler = SAMC(
            energy_fn=_quadratic,
            dim=2,
            n_partitions=10,
            e_min=0.0,
            e_max=5.0,
            proposal_std=0.5,
            device="mps",
            gain="1/t",
            gain_kwargs={"t0": 100},
        )
        result = sampler.run(n_steps=500, progress=False, seed=42)
        assert result.samples.device.type == "mps"
        assert result.acceptance_rate > 0

    def test_float64_theta_counts_on_cpu(self):
        """Theta and counts stay on CPU as float64 (MPS lacks float64)."""
        from illuma_samc.sampler import SAMC

        sampler = SAMC(
            energy_fn=_quadratic,
            dim=2,
            n_partitions=10,
            e_min=0.0,
            e_max=5.0,
            device="mps",
            gain="1/t",
            gain_kwargs={"t0": 100},
        )
        result = sampler.run(n_steps=200, progress=False, seed=42)
        assert result.log_weights.dtype == torch.float64
        assert result.log_weights.device.type == "cpu"
        assert result.bin_counts.dtype == torch.float64
        assert result.bin_counts.device.type == "cpu"
        # Samples should be on MPS
        assert result.samples.device.type == "mps"


class TestMPSMultiChain:
    def test_multi_chain_on_mps(self):
        from illuma_samc.sampler import SAMC

        sampler = SAMC(
            energy_fn=_quadratic,
            dim=2,
            n_chains=3,
            n_partitions=10,
            e_min=0.0,
            e_max=5.0,
            proposal_std=0.5,
            device="mps",
            gain="1/t",
            gain_kwargs={"t0": 100},
        )
        result = sampler.run(n_steps=500, progress=False, seed=42)
        assert result.samples.shape[0] == 3
        assert result.acceptance_rate > 0


class TestMPSLangevin:
    def test_langevin_autograd_on_mps(self):
        """LangevinProposal uses torch.autograd — verify it works on MPS."""
        from illuma_samc.proposals import LangevinProposal
        from illuma_samc.sampler import SAMC

        proposal = LangevinProposal(_quadratic, step_size=0.05)
        sampler = SAMC(
            energy_fn=_quadratic,
            dim=2,
            n_partitions=10,
            e_min=0.0,
            e_max=5.0,
            proposal_fn=proposal,
            device="mps",
            gain="1/t",
            gain_kwargs={"t0": 100},
        )
        result = sampler.run(n_steps=200, progress=False, seed=42)
        assert result.samples.device.type == "mps"


class TestMPSAdaptive:
    def test_adaptive_proposal_on_mps(self):
        from illuma_samc.sampler import SAMC

        sampler = SAMC(
            energy_fn=_quadratic,
            dim=2,
            n_partitions=10,
            e_min=0.0,
            e_max=5.0,
            proposal_std=2.0,
            adapt_proposal=True,
            adapt_warmup=200,
            device="mps",
            gain="1/t",
            gain_kwargs={"t0": 100},
        )
        result = sampler.run(n_steps=500, progress=False, seed=42)
        assert result.acceptance_rate > 0
        assert sampler._proposal.step_size < 2.0


class TestMPSSAMCWeights:
    def test_samcweights_on_mps(self):
        """SAMCWeights with device='mps'."""
        import math

        from illuma_samc import SAMCWeights

        wm = SAMCWeights(device="mps")
        x = torch.zeros(2, device="mps")
        fx = _quadratic(x).item()

        for t in range(1, 201):
            x_new = x + 0.5 * torch.randn(2, device="mps")
            fy = _quadratic(x_new).item()
            log_r = (-fy + fx) + wm.correction(fx, fy)
            if log_r > 0 or math.log(torch.rand(1).item() + 1e-300) < log_r:
                x, fx = x_new, fy
            wm.step(t, fx)

        assert wm.theta.device.type == "cpu"  # theta always on CPU (MPS lacks float64)
        assert wm.counts.device.type == "cpu"
        assert wm.n_bins > 0
