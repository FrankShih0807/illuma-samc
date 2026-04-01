"""Tests for SAMCWeights (drop-in SAMC for MH loops)."""

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


class TestCorrection:
    def test_zero_when_same_bin(self):
        wm = make_wm()
        # Same energy, same bin → theta terms cancel → 0
        assert wm.correction(5.0, 5.0) == 0.0

    def test_proposed_out_of_range(self):
        wm = make_wm()
        assert wm.correction(5.0, 15.0) == float("-inf")

    def test_current_out_of_range_proposed_in(self):
        wm = make_wm()
        assert wm.correction(15.0, 5.0) == float("inf")

    def test_both_out_of_range(self):
        wm = make_wm()
        assert wm.correction(15.0, -5.0) == float("-inf")

    def test_correction_is_pure_theta(self):
        """Correction should be theta[kx] - theta[ky], no energy term."""
        wm = make_wm()
        # With fresh theta (all zeros), correction is always 0 for in-range
        assert wm.correction(2.0, 8.0) == 0.0
        assert wm.correction(8.0, 2.0) == 0.0

    def test_correction_reflects_weight_updates(self):
        wm = make_wm()
        # Visit bin 5 many times → theta[5] increases
        for t in range(1, 51):
            wm.step(t, 5.0)
        # Correction should discourage moving TO the visited bin
        # (theta[kx] - theta[ky] is negative when ky is the heavy bin)
        c = wm.correction(1.0, 5.0)  # proposing to move to visited bin
        assert c < 0
        # And encourage moving AWAY from it
        c2 = wm.correction(5.0, 1.0)
        assert c2 > 0

    def test_accepts_tensor_inputs(self):
        wm = make_wm()
        c = wm.correction(torch.tensor(5.0), torch.tensor(3.0))
        assert isinstance(c, float)


class TestStep:
    def test_updates_theta_and_counts(self):
        wm = make_wm()
        wm.step(1, 5.0)
        assert wm.counts.sum().item() == 1
        assert wm.theta.sum().abs().item() < 1e-12  # theta sums to ~0

    def test_theta_shifts_toward_visited_bin(self):
        wm = make_wm()
        wm.step(1, 5.0)
        k = wm.partition.assign(torch.tensor(5.0))
        assert wm.theta[k].item() > wm.theta[0].item()

    def test_out_of_range_is_noop(self):
        wm = make_wm()
        wm.step(1, 15.0)
        assert wm.counts.sum().item() == 0
        assert (wm.theta == 0).all()

    def test_counts_accumulate(self):
        wm = make_wm()
        for t in range(1, 11):
            wm.step(t, 5.0)
        k = wm.partition.assign(torch.tensor(5.0))
        assert wm.counts[k].item() == 10


class TestMHPlusSAMC:
    """Test the actual MH + correction pattern."""

    def test_mh_loop_with_correction(self):
        partition = UniformPartition(e_min=0, e_max=20, n_bins=20)
        gain = GainSequence("1/t", t0=50)
        wm = SAMCWeights(partition=partition, gain=gain)

        # Simple quadratic energy
        def energy_fn(x):
            return (x**2).sum().item()

        torch.manual_seed(123)
        T = 1.0
        x = torch.tensor([2.0])
        fx = energy_fn(x)

        n_accept = 0
        for t in range(1, 1001):
            x_new = x + 0.5 * torch.randn_like(x)
            fy = energy_fn(x_new)

            # Standard MH ratio + SAMC correction
            log_r = (-fy + fx) / T + wm.correction(fx, fy)

            if fy < 0 or fy > 20:
                accept = False
            elif log_r > 0:
                accept = True
            else:
                accept = torch.rand(1).item() < math.exp(log_r)

            if accept:
                x, fx = x_new, fy
                n_accept += 1

            wm.step(t, fx)

        assert wm.counts.sum().item() == 1000
        assert n_accept > 0
        assert 0.1 < n_accept / 1000 < 0.95  # reasonable acceptance rate

    def test_low_temperature_still_explores(self):
        """At low T, standard MH gets stuck. SAMC correction helps."""
        partition = UniformPartition(e_min=0, e_max=20, n_bins=20)
        gain = GainSequence("1/t", t0=200)
        wm = SAMCWeights(partition=partition, gain=gain)

        def energy_fn(x):
            return (x**2).sum().item()

        torch.manual_seed(42)
        T = 0.1
        x = torch.tensor([1.0])
        fx = energy_fn(x)

        for t in range(1, 5001):
            x_new = x + 0.3 * torch.randn_like(x)
            fy = energy_fn(x_new)

            log_r = (-fy + fx) / T + wm.correction(fx, fy)

            if fy < 0 or fy > 20:
                accept = False
            elif log_r > 0:
                accept = True
            else:
                accept = math.exp(log_r) > torch.rand(1).item()

            if accept:
                x, fx = x_new, fy

            wm.step(t, fx)

        # SAMC should visit multiple bins, not just stay near 0
        visited = (wm.counts > 0).sum().item()
        assert visited >= 3  # at least 3 bins visited


class TestFlatness:
    def test_no_visits(self):
        wm = make_wm()
        assert wm.flatness() == 0.0

    def test_perfect_flatness(self):
        wm = make_wm()
        # Visit every bin equally
        wm.counts = torch.ones(10, dtype=torch.float64) * 100
        assert abs(wm.flatness() - 1.0) < 1e-12

    def test_skewed_visits(self):
        wm = make_wm()
        wm.counts = torch.zeros(10, dtype=torch.float64)
        wm.counts[0] = 1000  # all visits in one bin
        assert wm.flatness() < 0.5


class TestImportanceWeights:
    def test_returns_correct_shape(self):
        wm = make_wm()
        for t in range(1, 101):
            wm.step(t, 5.0)
        energies = torch.tensor([5.0, 5.0, 5.0])
        w = wm.importance_weights(energies)
        assert w.shape == (3,)

    def test_sums_to_one(self):
        wm = make_wm()
        # Visit multiple bins
        for t in range(1, 101):
            wm.step(t, float(t % 10))
        energies = torch.tensor([1.0, 3.0, 5.0, 7.0])
        w = wm.importance_weights(energies)
        assert abs(w.sum().item() - 1.0) < 1e-10

    def test_out_of_range_gets_zero(self):
        wm = make_wm()
        for t in range(1, 51):
            wm.step(t, 5.0)
        energies = torch.tensor([5.0, 15.0])  # second is out of range
        w = wm.importance_weights(energies)
        assert w[1].item() == 0.0

    def test_unvisited_bin_gets_zero(self):
        wm = make_wm()
        # Only visit bin 5
        for t in range(1, 51):
            wm.step(t, 5.0)
        energies = torch.tensor([5.0, 1.0])  # bin 1 unvisited
        w = wm.importance_weights(energies)
        assert w[1].item() == 0.0
        assert w[0].item() > 0.0

    def test_all_out_of_range_returns_zeros(self):
        wm = make_wm()
        energies = torch.tensor([15.0, 20.0])
        w = wm.importance_weights(energies)
        assert (w == 0).all()

    def test_log_weights_shape(self):
        wm = make_wm()
        for t in range(1, 51):
            wm.step(t, 5.0)
        energies = torch.tensor([5.0, 3.0])
        lw = wm.importance_log_weights(energies)
        assert lw.shape == (2,)


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
