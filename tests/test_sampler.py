"""Tests for the core SAMC sampler."""

import torch

from illuma_samc.gain import GainSequence
from illuma_samc.partitions import UniformPartition
from illuma_samc.proposals import GaussianProposal
from illuma_samc.sampler import SAMC, SAMCResult


def _quadratic_energy(x: torch.Tensor) -> torch.Tensor:
    """Simple quadratic energy: E(x) = 0.5 * ||x||^2."""
    return 0.5 * torch.sum(x**2)


def _multimodal_energy(x: torch.Tensor) -> torch.Tensor:
    """Two-well energy: minima at x=[-2,0] and x=[2,0]."""
    return torch.min(
        0.5 * torch.sum((x - torch.tensor([2.0, 0.0])) ** 2),
        0.5 * torch.sum((x + torch.tensor([2.0, 0.0])) ** 2),
    )


class TestSimpleMode:
    def test_basic_run(self):
        torch.manual_seed(42)
        sampler = SAMC(
            energy_fn=_quadratic_energy,
            dim=2,
            n_partitions=10,
            e_min=-5.0,
            e_max=5.0,
            gain="1/t",
            gain_kwargs={"t0": 100},
        )
        result = sampler.run(n_steps=1000, save_every=10, progress=False)

        assert isinstance(result, SAMCResult)
        assert result.samples.shape == (100, 2)
        assert result.log_weights.shape == (10,)
        assert result.energy_history.shape == (1000,)
        assert result.bin_counts.shape == (10,)
        assert 0.0 <= result.acceptance_rate <= 1.0
        assert result.best_x.shape == (2,)

    def test_state_stored_on_sampler(self):
        torch.manual_seed(0)
        sampler = SAMC(
            energy_fn=_quadratic_energy,
            dim=2,
            n_partitions=5,
            e_min=-5.0,
            e_max=5.0,
            gain="1/t",
            gain_kwargs={"t0": 50},
        )
        sampler.run(n_steps=500, progress=False)

        assert sampler.log_weights is not None
        assert sampler.bin_counts is not None
        assert sampler.best_x is not None
        assert sampler.best_energy < float("inf")
        assert sampler.acceptance_rate > 0

    def test_finds_minimum_quadratic(self):
        """Quadratic has minimum at origin — best_energy should be near 0."""
        torch.manual_seed(42)
        sampler = SAMC(
            energy_fn=_quadratic_energy,
            dim=2,
            n_partitions=10,
            e_min=-5.0,
            e_max=5.0,
            proposal_std=0.5,
            gain="1/t",
            gain_kwargs={"t0": 100},
        )
        result = sampler.run(n_steps=10000, progress=False)
        assert result.best_energy < 0.5

    def test_default_x0_zeros(self):
        torch.manual_seed(0)
        sampler = SAMC(
            energy_fn=_quadratic_energy,
            dim=3,
            n_partitions=5,
            e_min=-5.0,
            e_max=5.0,
            gain="1/t",
        )
        result = sampler.run(n_steps=100, progress=False)
        assert result.samples.shape[1] == 3

    def test_custom_x0(self):
        torch.manual_seed(0)
        sampler = SAMC(
            energy_fn=_quadratic_energy,
            dim=2,
            n_partitions=5,
            e_min=-5.0,
            e_max=5.0,
            gain="1/t",
        )
        x0 = torch.tensor([3.0, -2.0])
        result = sampler.run(n_steps=100, x0=x0, progress=False)
        assert result.samples.shape == (1, 2)


class TestFlexibleMode:
    def test_custom_components(self):
        torch.manual_seed(42)
        sampler = SAMC(
            energy_fn=_quadratic_energy,
            dim=2,
            proposal_fn=GaussianProposal(step_size=0.3),
            partition_fn=UniformPartition(e_min=-5.0, e_max=5.0, n_bins=8),
            gain=GainSequence("1/t", t0=100),
        )
        result = sampler.run(n_steps=1000, progress=False)
        assert result.samples.shape[0] == 10
        assert result.log_weights.shape == (8,)

    def test_custom_log_accept(self):
        """Use a custom acceptance function (Boltzmann with temperature)."""
        torch.manual_seed(42)
        temperature = 0.5

        def log_accept(x, x_new, e_x, e_new):
            return (-e_new.item() + e_x.item()) / temperature

        sampler = SAMC(
            energy_fn=_quadratic_energy,
            dim=2,
            n_partitions=5,
            e_min=-5.0,
            e_max=5.0,
            log_accept_fn=log_accept,
            gain="1/t",
            gain_kwargs={"t0": 50},
        )
        result = sampler.run(n_steps=500, progress=False)
        assert result.acceptance_rate > 0


class TestEquivalence:
    def test_simple_vs_flexible_same_result(self):
        """Simple and flexible modes with identical config should produce identical results."""
        torch.manual_seed(123)
        simple = SAMC(
            energy_fn=_quadratic_energy,
            dim=2,
            n_partitions=10,
            e_min=-5.0,
            e_max=5.0,
            proposal_std=0.3,
            gain="1/t",
            gain_kwargs={"t0": 100},
        )
        result_simple = simple.run(n_steps=500, progress=False)

        torch.manual_seed(123)
        flexible = SAMC(
            energy_fn=_quadratic_energy,
            dim=2,
            proposal_fn=GaussianProposal(step_size=0.3),
            partition_fn=UniformPartition(e_min=-5.0, e_max=5.0, n_bins=10),
            gain=GainSequence("1/t", t0=100),
        )
        result_flexible = flexible.run(n_steps=500, progress=False)

        assert torch.allclose(result_simple.samples, result_flexible.samples)
        assert torch.allclose(result_simple.log_weights, result_flexible.log_weights)
        assert result_simple.acceptance_rate == result_flexible.acceptance_rate


class TestRampGain:
    def test_ramp_schedule(self):
        """Test with ramp gain matching sample_code.py defaults."""
        torch.manual_seed(0)
        sampler = SAMC(
            energy_fn=_quadratic_energy,
            dim=2,
            n_partitions=10,
            e_min=-5.0,
            e_max=5.0,
            gain="ramp",
            gain_kwargs={"rho": 1.0, "tau": 1.0, "warmup": 1, "step_scale": 100},
        )
        result = sampler.run(n_steps=1000, progress=False)
        assert result.acceptance_rate > 0


def _quadratic_energy_batch(x: torch.Tensor) -> torch.Tensor:
    """Batch-compatible quadratic energy: E(x) = 0.5 * ||x||^2.

    Accepts (dim,) or (N, dim) and returns scalar or (N,).
    """
    if x.dim() == 1:
        return 0.5 * torch.sum(x**2)
    return 0.5 * torch.sum(x**2, dim=-1)


class TestMultiChain:
    """Tests for parallel chains with shared weights."""

    def test_multi_chain_basic_run(self):
        """Multi-chain run produces correct output shapes."""
        torch.manual_seed(42)
        n_chains = 4
        sampler = SAMC(
            energy_fn=_quadratic_energy_batch,
            dim=2,
            n_partitions=10,
            e_min=-5.0,
            e_max=5.0,
            gain="1/t",
            gain_kwargs={"t0": 100},
        )
        x0 = torch.randn(n_chains, 2)
        result = sampler.run(n_steps=1000, x0=x0, save_every=10, progress=False)

        assert isinstance(result, SAMCResult)
        # (N, n_saved, dim)
        assert result.samples.shape == (n_chains, 100, 2)
        assert result.log_weights.shape == (10,)
        # energy_history: (n_steps, N)
        assert result.energy_history.shape == (1000, n_chains)
        assert result.bin_counts.shape == (10,)
        assert 0.0 <= result.acceptance_rate <= 1.0
        assert result.best_x.shape == (2,)

    def test_multi_chain_finds_minimum(self):
        """Multi-chain should find the minimum at least as well as single chain."""
        torch.manual_seed(42)
        sampler = SAMC(
            energy_fn=_quadratic_energy_batch,
            dim=2,
            n_partitions=10,
            e_min=-5.0,
            e_max=5.0,
            proposal_std=0.5,
            gain="1/t",
            gain_kwargs={"t0": 100},
        )
        x0 = torch.randn(4, 2)
        result = sampler.run(n_steps=5000, x0=x0, progress=False)
        assert result.best_energy < 0.5

    def test_multi_chain_state_stored(self):
        """Sampler state is updated after multi-chain run."""
        torch.manual_seed(0)
        sampler = SAMC(
            energy_fn=_quadratic_energy_batch,
            dim=2,
            n_partitions=5,
            e_min=-5.0,
            e_max=5.0,
            gain="1/t",
            gain_kwargs={"t0": 50},
        )
        x0 = torch.randn(3, 2)
        sampler.run(n_steps=500, x0=x0, progress=False)

        assert sampler.log_weights is not None
        assert sampler.bin_counts is not None
        assert sampler.best_x is not None
        assert sampler.best_energy < float("inf")
        assert sampler.acceptance_rate > 0

    def test_single_chain_unchanged(self):
        """Passing 1-D x0 still uses single-chain path."""
        torch.manual_seed(42)
        sampler = SAMC(
            energy_fn=_quadratic_energy_batch,
            dim=2,
            n_partitions=10,
            e_min=-5.0,
            e_max=5.0,
            gain="1/t",
            gain_kwargs={"t0": 100},
        )
        x0 = torch.tensor([1.0, -1.0])
        result = sampler.run(n_steps=200, x0=x0, save_every=10, progress=False)
        # Single chain: (n_saved, dim)
        assert result.samples.shape == (20, 2)
        assert result.energy_history.shape == (200,)

    def test_multi_chain_shared_weights_converge(self):
        """Multi-chain with shared weights should converge similarly to single chain.

        Both should find roughly the same best energy on the same problem.
        """
        # Single chain
        torch.manual_seed(99)
        single = SAMC(
            energy_fn=_quadratic_energy_batch,
            dim=2,
            n_partitions=10,
            e_min=-5.0,
            e_max=5.0,
            proposal_std=0.3,
            gain="1/t",
            gain_kwargs={"t0": 200},
        )
        single_result = single.run(n_steps=5000, progress=False)

        # Multi-chain (same total work: fewer steps but more chains)
        torch.manual_seed(99)
        multi = SAMC(
            energy_fn=_quadratic_energy_batch,
            dim=2,
            n_partitions=10,
            e_min=-5.0,
            e_max=5.0,
            proposal_std=0.3,
            gain="1/t",
            gain_kwargs={"t0": 200},
        )
        x0 = torch.randn(4, 2)
        multi_result = multi.run(n_steps=5000, x0=x0, progress=False)

        # Both should find energy near 0 (quadratic minimum)
        assert single_result.best_energy < 1.0
        assert multi_result.best_energy < 1.0


class TestGPU:
    """Tests that GPU results match CPU (skipped if CUDA unavailable)."""

    @staticmethod
    def _skip_if_no_cuda():
        if not torch.cuda.is_available():
            import pytest

            pytest.skip("CUDA not available")

    def test_gpu_single_chain(self):
        self._skip_if_no_cuda()
        torch.manual_seed(42)
        sampler = SAMC(
            energy_fn=_quadratic_energy_batch,
            dim=2,
            n_partitions=10,
            e_min=-5.0,
            e_max=5.0,
            gain="1/t",
            gain_kwargs={"t0": 100},
            device="cuda",
        )
        result = sampler.run(n_steps=500, progress=False)
        assert result.samples.device.type == "cuda"
        assert result.best_x.device.type == "cuda"
        assert result.best_energy < float("inf")

    def test_gpu_multi_chain(self):
        self._skip_if_no_cuda()
        torch.manual_seed(42)
        sampler = SAMC(
            energy_fn=_quadratic_energy_batch,
            dim=2,
            n_partitions=10,
            e_min=-5.0,
            e_max=5.0,
            gain="1/t",
            gain_kwargs={"t0": 100},
            device="cuda",
        )
        x0 = torch.randn(3, 2)
        result = sampler.run(n_steps=500, x0=x0, progress=False)
        assert result.samples.device.type == "cuda"
        assert result.samples.shape[0] == 3
        assert result.best_energy < float("inf")

    def test_gpu_matches_cpu_single_chain(self):
        """GPU and CPU with same seed should produce identical results."""
        self._skip_if_no_cuda()

        torch.manual_seed(42)
        cpu_sampler = SAMC(
            energy_fn=_quadratic_energy_batch,
            dim=2,
            n_partitions=10,
            e_min=-5.0,
            e_max=5.0,
            gain="1/t",
            gain_kwargs={"t0": 100},
            device="cpu",
        )
        cpu_result = cpu_sampler.run(n_steps=500, progress=False)

        torch.manual_seed(42)
        gpu_sampler = SAMC(
            energy_fn=_quadratic_energy_batch,
            dim=2,
            n_partitions=10,
            e_min=-5.0,
            e_max=5.0,
            gain="1/t",
            gain_kwargs={"t0": 100},
            device="cuda",
        )
        gpu_result = gpu_sampler.run(n_steps=500, progress=False)

        assert torch.allclose(cpu_result.samples, gpu_result.samples.cpu(), atol=1e-5)
        assert abs(cpu_result.best_energy - gpu_result.best_energy) < 1e-5
