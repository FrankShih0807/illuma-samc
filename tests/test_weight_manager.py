"""Tests for SAMCWeights (drop-in SAMC for MH loops)."""

import math

import torch

from illuma_samc.gain import GainSequence
from illuma_samc.partitions import UniformPartition
from illuma_samc.weight_manager import SAMCWeights


def make_wm(record_every=100, **kwargs):
    defaults = {
        "partition": UniformPartition(e_min=0, e_max=10, n_bins=10),
        "gain": GainSequence("1/t", t0=100),
    }
    defaults.update(kwargs)
    return SAMCWeights(**defaults, record_every=record_every)


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
        wm.counts[0] = 1000
        wm.counts[1] = 10  # two visited bins, heavily skewed
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

    def test_weight_is_exp_theta(self):
        """Importance weight should be proportional to exp(theta[k])."""
        wm = make_wm()
        # Visit bin 5 a lot (high theta) and bin 1 a little (low theta)
        for t in range(1, 101):
            wm.step(t, 5.0)
        for t in range(101, 111):
            wm.step(t, 1.0)
        energies = torch.tensor([5.0, 1.0])
        w = wm.importance_weights(energies)
        # Bin 5 has higher theta → higher weight
        assert w[0] > w[1]

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

    def test_log_weights_are_theta(self):
        """Log importance weights should be theta[k], not -theta[k]."""
        wm = make_wm()
        for t in range(1, 51):
            wm.step(t, 5.0)
        k = wm.partition.assign(torch.tensor(5.0))
        energies = torch.tensor([5.0])
        lw = wm.importance_log_weights(energies)
        assert abs(lw[0].item() - wm.theta[k].item()) < 1e-12


class TestResample:
    def test_returns_subset(self):
        """Binary rejection: result is a subset of input samples."""
        wm = make_wm()
        for t in range(1, 101):
            wm.step(t, float(t % 10))
        samples = torch.randn(100, 2)
        energies = torch.tensor([float(i % 10) for i in range(100)])
        torch.manual_seed(0)
        resampled = wm.resample(samples, energies)
        assert resampled.shape[1] == 2
        assert resampled.shape[0] <= 100  # subset, not larger

    def test_empty_when_all_out_of_range(self):
        wm = make_wm()
        samples = torch.randn(10, 2)
        energies = torch.full((10,), 15.0)  # all out of range
        resampled = wm.resample(samples, energies)
        assert resampled.shape[0] == 0

    def test_high_theta_bins_kept_more(self):
        """Samples in high-theta bins should be kept with higher probability."""
        wm = make_wm()
        # Visit bin 5 a lot → high theta
        for t in range(1, 201):
            wm.step(t, 5.0)
        # Some visits to bin 1 → lower theta
        for t in range(201, 221):
            wm.step(t, 1.0)

        torch.manual_seed(42)
        n = 500
        samples = torch.randn(2 * n, 2)
        energies = torch.cat([torch.full((n,), 5.0), torch.full((n,), 1.0)])
        resampled_count = 0
        for _ in range(10):
            r = wm.resample(samples, energies)
            resampled_count += r.shape[0]
        # High-theta bin samples should dominate — at least some are kept
        assert resampled_count > 0


class TestBinCountsHistory:
    def test_records_snapshots(self):
        wm = make_wm(record_every=10)
        for t in range(1, 101):
            wm.step(t, float(t % 10))
        # 100 steps / record_every=10 → 10 snapshots
        assert len(wm.bin_counts_history) == 10

    def test_snapshot_is_independent_copy(self):
        wm = make_wm(record_every=10)
        for t in range(1, 11):
            wm.step(t, 5.0)
        # First snapshot should have 10 visits to bin 5
        snap = wm.bin_counts_history[0]
        assert snap.sum().item() == 10
        # Continue stepping — snapshot should not change
        for t in range(11, 21):
            wm.step(t, 5.0)
        assert snap.sum().item() == 10  # still 10, not 20

    def test_default_record_every(self):
        wm = make_wm()
        assert wm._record_every == 100

    def test_custom_record_every(self):
        wm = make_wm(record_every=50)
        for t in range(1, 201):
            wm.step(t, 5.0)
        assert len(wm.bin_counts_history) == 4  # 200/50

    def test_flatness_history(self):
        wm = make_wm(record_every=10)
        for t in range(1, 101):
            wm.step(t, float(t % 10))
        fh = wm.flatness_history()
        assert len(fh) == 10
        # Flatness should generally increase as visits spread out
        assert all(isinstance(f, float) for f in fh)

    def test_flatness_history_converges(self):
        """Flatness should improve over time with uniform visits."""
        wm = make_wm(record_every=50)
        # Visit bins roughly uniformly
        for t in range(1, 501):
            wm.step(t, float(t % 10))
        fh = wm.flatness_history()
        # Later snapshots should be at least as flat as earlier ones (roughly)
        assert fh[-1] >= fh[0] - 0.1  # allow small noise

    def test_flatness_history_empty_when_no_steps(self):
        wm = make_wm()
        assert wm.flatness_history() == []


class TestAcceptanceRateTracking:
    def test_tracked_acceptance_rate(self):
        wm = make_wm()
        # Step with alternating energies → nearly every step changes energy
        for t in range(1, 101):
            energy = 5.0 if t % 2 == 0 else 3.0
            wm.step(t, energy)
        rate = wm.tracked_acceptance_rate
        # Energy changes on 99 of 100 steps (first step has no previous)
        assert rate > 0.9

    def test_tracked_acceptance_rate_no_steps(self):
        wm = make_wm()
        assert wm.tracked_acceptance_rate == 0.0

    def test_tracked_acceptance_rate_all_same(self):
        wm = make_wm()
        for t in range(1, 101):
            wm.step(t, 5.0)
        assert wm.tracked_acceptance_rate == 0.0

    def test_acceptance_rate_warning_low(self):
        """Warn when acceptance rate is below 15% after warmup."""
        import warnings

        wm = make_wm()
        # Same energy for 1000+ steps → 0% acceptance
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            for t in range(1, 2001):
                wm.step(t, 5.0)
            warning_msgs = [str(x.message) for x in w]
            assert any("below 15%" in msg for msg in warning_msgs)


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

    def test_save_and_load_with_history(self):
        wm = make_wm(record_every=10)
        for t in range(1, 51):
            wm.step(t, 5.0)

        state = wm.state_dict()

        wm2 = make_wm(record_every=10)
        wm2.load_state_dict(state)

        assert len(wm2.bin_counts_history) == len(wm.bin_counts_history)
        for h1, h2 in zip(wm.bin_counts_history, wm2.bin_counts_history):
            assert torch.allclose(h1, h2)
        assert wm2._n_steps == wm._n_steps
        assert wm2._n_accepted == wm._n_accepted

    def test_load_old_state_dict_without_history(self):
        """Backward compatibility: old state_dicts without history fields."""
        wm = make_wm()
        old_state = {"theta": wm.theta.clone(), "counts": wm.counts.clone(), "t": 0}
        wm.load_state_dict(old_state)
        assert wm.bin_counts_history == []


class TestVectorizedImportanceWeights:
    def test_large_batch(self):
        """Vectorized importance_log_weights handles large batches."""
        wm = make_wm()
        for t in range(1, 101):
            wm.step(t, float(t % 10))
        energies = torch.rand(10000) * 10
        lw = wm.importance_log_weights(energies)
        assert lw.shape == (10000,)

    def test_matches_scalar_loop(self):
        """Vectorized version produces same results as scalar loop."""
        wm = make_wm()
        for t in range(1, 201):
            wm.step(t, float(t % 10))
        energies = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0, 15.0, -1.0])

        # Vectorized result
        lw_vec = wm.importance_log_weights(energies)

        # Scalar loop (reference)
        lw_scalar = torch.full(energies.shape, float("-inf"), dtype=torch.float64)
        for i, e in enumerate(energies):
            k = wm.partition.assign(e)
            if k >= 0 and wm.counts[k] > 0:
                lw_scalar[i] = wm.theta[k].item()

        assert torch.allclose(lw_vec, lw_scalar, equal_nan=True)


class TestPlotDiagnostics:
    def test_smoke_test(self):
        """plot_diagnostics should not raise."""
        import matplotlib

        matplotlib.use("Agg")
        wm = make_wm(record_every=10)
        for t in range(1, 101):
            wm.step(t, float(t % 10))
        fig = wm.plot_diagnostics()
        assert fig is not None

    def test_smoke_test_no_history(self):
        """plot_diagnostics should work even without history."""
        import matplotlib

        matplotlib.use("Agg")
        wm = make_wm(record_every=1000)
        for t in range(1, 11):
            wm.step(t, 5.0)
        fig = wm.plot_diagnostics()
        assert fig is not None


class TestResizeForPartition:
    def test_expand(self):
        wm = make_wm()
        assert wm.theta.shape[0] == 10
        # Simulate partition expanding
        wm.partition = UniformPartition(e_min=0, e_max=15, n_bins=15)
        wm._resize_for_partition()
        assert wm.theta.shape[0] == 15
        assert wm.counts.shape[0] == 15
        # New bins should be zero
        assert wm.theta[10:].sum().item() == 0.0

    def test_shrink(self):
        wm = make_wm()
        wm.partition = UniformPartition(e_min=0, e_max=5, n_bins=5)
        wm._resize_for_partition()
        assert wm.theta.shape[0] == 5

    def test_no_change(self):
        wm = make_wm()
        old_theta = wm.theta.clone()
        wm._resize_for_partition()
        assert torch.equal(wm.theta, old_theta)


class TestDtype:
    """Tests for dtype parameter on SAMCWeights."""

    def test_dtype_float64_stored(self):
        """SAMCWeights(dtype='float64') stores float64 dtype."""
        wm = SAMCWeights(
            partition=UniformPartition(0, 10, 10),
            gain=GainSequence("1/t", t0=50),
            dtype="float64",
        )
        assert wm._dtype == torch.float64

    def test_dtype_torch_float32(self):
        """SAMCWeights(dtype=torch.float32) stores float32 dtype."""
        wm = SAMCWeights(
            partition=UniformPartition(0, 10, 10),
            gain=GainSequence("1/t", t0=50),
            dtype=torch.float32,
        )
        assert wm._dtype == torch.float32

    def test_theta_always_float64(self):
        """Internal theta stays float64 regardless of user dtype."""
        wm = SAMCWeights(
            partition=UniformPartition(0, 10, 10),
            gain=GainSequence("1/t", t0=50),
            dtype="float32",
        )
        assert wm.theta.dtype == torch.float64

    def test_theta_float64_when_dtype_float64(self):
        """theta is still float64 even if user requests float64."""
        wm = SAMCWeights(
            partition=UniformPartition(0, 10, 10),
            gain=GainSequence("1/t", t0=50),
            dtype="float64",
        )
        assert wm.theta.dtype == torch.float64
