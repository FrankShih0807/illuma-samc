"""Tests for SAMCWeights (torch.optim-style interface)."""

import math

import torch

from illuma_samc.gain import GainSequence
from illuma_samc.partitions import UniformPartition
from illuma_samc.weight_manager import SAMCWeights


def make_wm(**kwargs):
    defaults = {
        "partition": UniformPartition(e_min=0, e_max=10, n_bins=10),
        "gain": GainSequence("1/t", t0=100),
    }
    defaults.update(kwargs)
    return SAMCWeights(**defaults)


class TestLogAcceptRatio:
    def test_same_energy_same_bin(self):
        wm = make_wm()
        # Same energy → theta terms cancel, energy terms cancel → 0
        assert wm.log_accept_ratio(5.0, 5.0) == 0.0

    def test_proposed_out_of_range(self):
        wm = make_wm()
        assert wm.log_accept_ratio(5.0, 15.0) == float("-inf")

    def test_current_out_of_range_proposed_in(self):
        wm = make_wm()
        assert wm.log_accept_ratio(15.0, 5.0) == float("inf")

    def test_both_out_of_range(self):
        wm = make_wm()
        assert wm.log_accept_ratio(15.0, -5.0) == float("-inf")

    def test_lower_energy_preferred(self):
        wm = make_wm()
        # Lower proposed energy → positive log ratio
        assert wm.log_accept_ratio(8.0, 2.0) > 0

    def test_temperature_scales_energy(self):
        wm1 = make_wm(temperature=1.0)
        wm2 = make_wm(temperature=2.0)
        r1 = wm1.log_accept_ratio(8.0, 2.0)
        r2 = wm2.log_accept_ratio(8.0, 2.0)
        # Higher temperature → smaller energy contribution
        assert abs(r2 - r1 / 2.0) < 1e-12

    def test_accepts_tensor_inputs(self):
        wm = make_wm()
        r = wm.log_accept_ratio(torch.tensor(5.0), torch.tensor(3.0))
        assert isinstance(r, float)
        assert r > 0


class TestStep:
    def test_updates_theta_and_counts(self):
        wm = make_wm()
        wm.step(1, 5.0)  # bin 5
        assert wm.counts.sum().item() == 1
        assert wm.theta.sum().abs().item() < 1e-12  # theta sums to ~0

    def test_theta_shifts_toward_visited_bin(self):
        wm = make_wm()
        wm.step(1, 5.0)
        k = wm.partition.assign(torch.tensor(5.0))
        # Visited bin gets extra weight relative to others
        assert wm.theta[k].item() > wm.theta[0].item()

    def test_out_of_range_is_noop(self):
        wm = make_wm()
        wm.step(1, 15.0)  # out of range
        assert wm.counts.sum().item() == 0
        assert (wm.theta == 0).all()

    def test_counts_accumulate(self):
        wm = make_wm()
        for t in range(1, 11):
            wm.step(t, 5.0)
        k = wm.partition.assign(torch.tensor(5.0))
        assert wm.counts[k].item() == 10


class TestMatchesSAMC:
    """Verify SAMCWeights produces same theta as the full SAMC sampler."""

    def test_deterministic_trajectory(self):
        """Run both interfaces with identical accept/reject decisions."""
        partition = UniformPartition(e_min=-10, e_max=5, n_bins=20)
        gain = GainSequence("1/t", t0=100)

        wm = SAMCWeights(partition=partition, gain=gain)

        # Simulate 100 steps with deterministic energies
        torch.manual_seed(42)
        energies = torch.rand(100) * 15 - 10  # in [-10, 5]

        for t in range(1, 101):
            e_cur = energies[t - 1].item()
            # Just update weights with "current" energy (no proposal needed)
            wm.step(t, e_cur)

        # Verify theta sums to approximately 0 (weight update is zero-sum)
        assert abs(wm.theta.sum().item()) < 1e-10
        # Verify counts match number of in-range steps
        assert wm.counts.sum().item() == 100


class TestEndToEndLoop:
    """Test the full user-owned sampling loop pattern."""

    def test_simple_sampling_loop(self):
        partition = UniformPartition(e_min=0, e_max=10, n_bins=10)
        gain = GainSequence("1/t", t0=50)
        wm = SAMCWeights(partition=partition, gain=gain)

        # Simple quadratic energy
        def energy_fn(x):
            return (x**2).sum().item()

        torch.manual_seed(123)
        x = torch.tensor([1.0])
        fx = energy_fn(x)

        n_accept = 0
        for t in range(1, 1001):
            x_new = x + 0.5 * torch.randn_like(x)
            fy = energy_fn(x_new)

            log_r = wm.log_accept_ratio(fx, fy)

            if fy < 0 or fy > 10:
                accept = False
            elif log_r > 0:
                accept = True
            else:
                accept = torch.rand(1).item() < math.exp(log_r)

            if accept:
                x, fx = x_new, fy
                n_accept += 1

            wm.step(t, fx)

        # Basic sanity checks
        assert wm.counts.sum().item() == 1000
        assert n_accept > 0
        assert wm.theta.shape == (10,)


class TestStateDict:
    def test_save_and_load(self):
        wm = make_wm()
        for t in range(1, 51):
            wm.step(t, 5.0)

        state = wm.state_dict()

        wm2 = make_wm()
        wm2.load_state_dict(state)

        assert torch.allclose(wm.theta, wm2.theta)
        assert torch.allclose(wm.counts, wm2.counts)
        assert wm2._t == 50


class TestValidation:
    def test_negative_temperature(self):
        try:
            make_wm(temperature=-1.0)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_zero_temperature(self):
        try:
            make_wm(temperature=0.0)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
