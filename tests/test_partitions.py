"""Tests for energy-space partitions."""

import torch

from illuma_samc.partitions import (
    AdaptivePartition,
    ExpandablePartition,
    GrowingPartition,
    QuantilePartition,
    UniformPartition,
)


class TestUniformPartition:
    def test_basic_assignment(self):
        p = UniformPartition(e_min=0.0, e_max=10.0, n_bins=10)
        assert p.n_partitions == 10
        assert p.assign(torch.tensor(0.5)) == 0
        assert p.assign(torch.tensor(5.0)) == 5
        assert p.assign(torch.tensor(9.9)) == 9

    def test_below_range(self):
        p = UniformPartition(e_min=0.0, e_max=10.0, n_bins=10)
        assert p.assign(torch.tensor(-5.0)) == -1

    def test_above_range(self):
        p = UniformPartition(e_min=0.0, e_max=10.0, n_bins=10)
        assert p.assign(torch.tensor(100.0)) == -1

    def test_edges_shape(self):
        p = UniformPartition(e_min=-8.2, e_max=0.0, n_bins=42)
        assert p.edges.shape == (43,)

    def test_uniform_partition_edges_cached(self):
        """Bug E: edges should return the same object each time (cached)."""
        p = UniformPartition(e_min=-8.2, e_max=0.0, n_bins=42)
        assert p.edges is p.edges, "edges should return the same cached tensor"

    def test_matches_sample_code_binning(self):
        """Verify binning matches sample_code.py get_bin logic."""
        p = UniformPartition(e_min=-8.2, e_max=0.0, n_bins=42)
        # sample_code.py uses scale=5, which is n_bins/(e_max - e_min) ≈ 42/8.2 ≈ 5.12
        # Close but not exact — our partition is more general
        assert p.assign(torch.tensor(-8.2)) == 0
        assert p.assign(torch.tensor(0.0)) == 42 - 1
        assert p.assign(torch.tensor(1.0)) == -1  # above range


class TestUniformPartitionOverflow:
    def test_overflow_bins_n_partitions(self):
        p = UniformPartition(e_min=0.0, e_max=10.0, n_bins=10, overflow_bins=True)
        assert p.n_partitions == 12  # 10 core + 2 overflow

    def test_overflow_low_bin(self):
        p = UniformPartition(e_min=0.0, e_max=10.0, n_bins=10, overflow_bins=True)
        assert p.assign(torch.tensor(-5.0)) == 0  # low overflow

    def test_overflow_high_bin(self):
        p = UniformPartition(e_min=0.0, e_max=10.0, n_bins=10, overflow_bins=True)
        assert p.assign(torch.tensor(100.0)) == 11  # high overflow

    def test_overflow_in_range_shifted(self):
        p = UniformPartition(e_min=0.0, e_max=10.0, n_bins=10, overflow_bins=True)
        # In-range energies map to bins 1..10 (shifted by 1)
        assert p.assign(torch.tensor(0.5)) == 1
        assert p.assign(torch.tensor(5.0)) == 6
        assert p.assign(torch.tensor(9.9)) == 10

    def test_overflow_edges_shape(self):
        p = UniformPartition(e_min=0.0, e_max=10.0, n_bins=10, overflow_bins=True)
        assert p.edges.shape == (13,)  # 12 bins + 1 = 13 edges
        assert p.edges[0] == float("-inf")
        assert p.edges[-1] == float("inf")

    def test_overflow_batch_assignment(self):
        p = UniformPartition(e_min=0.0, e_max=10.0, n_bins=10, overflow_bins=True)
        energies = torch.tensor([-5.0, 0.5, 5.0, 9.9, 100.0])
        bins = p.assign_batch(energies)
        assert bins[0].item() == 0  # low overflow
        assert bins[1].item() == 1  # first core bin
        assert bins[2].item() == 6  # middle
        assert bins[3].item() == 10  # last core bin
        assert bins[4].item() == 11  # high overflow

    def test_backward_compatible_default(self):
        """Default overflow_bins=False should behave identically to old behavior."""
        p = UniformPartition(e_min=0.0, e_max=10.0, n_bins=10)
        assert p.n_partitions == 10
        assert p.assign(torch.tensor(-5.0)) == -1
        assert p.assign(torch.tensor(100.0)) == -1

    def test_overflow_samc_weights_integration(self):
        """SAMCWeights should work with overflow bins — no out-of-range rejections."""
        from illuma_samc.gain import GainSequence
        from illuma_samc.weight_manager import SAMCWeights

        p = UniformPartition(e_min=0.0, e_max=10.0, n_bins=10, overflow_bins=True)
        gain = GainSequence("1/t", t0=50)
        wm = SAMCWeights(partition=p, gain=gain)
        assert wm.n_bins == 12

        # Step with out-of-range energy — should not be skipped
        wm.step(1, -5.0)
        assert wm.counts[0].item() == 1  # low overflow bin visited

        wm.step(2, 100.0)
        assert wm.counts[11].item() == 1  # high overflow bin visited

        # Correction should not return -inf for overflow bins
        c = wm.correction(5.0, -5.0)
        assert c != float("-inf")


class TestAdaptivePartition:
    def test_initial_uniform(self):
        p = AdaptivePartition(e_min=0.0, e_max=10.0, n_bins=5)
        assert p.n_partitions == 5
        # Initially uniform
        assert p.assign(torch.tensor(1.0)) == 0

    def test_adapts_after_enough_samples(self):
        p = AdaptivePartition(e_min=0.0, e_max=10.0, n_bins=5, adapt_interval=100, min_samples=50)
        # Feed 100 samples concentrated in [2, 4]
        for _ in range(100):
            p.assign(torch.tensor(3.0))

        # After adaptation, edges should be tighter around 3.0
        # (all samples are 3.0, so edges collapse)
        assert p.edges[0].item() <= 3.0
        assert p.edges[-1].item() >= 3.0

    def test_no_adapt_before_min_samples(self):
        p = AdaptivePartition(e_min=0.0, e_max=10.0, n_bins=5, adapt_interval=10, min_samples=100)
        original_edges = p.edges.clone()
        for _ in range(50):
            p.assign(torch.tensor(3.0))
        # Not enough samples yet — edges unchanged
        assert torch.allclose(p.edges, original_edges)

    def test_adaptive_partition_ignores_outliers(self):
        """Bug D: Out-of-range energies should not influence adapted bin edges."""
        p = AdaptivePartition(e_min=0.0, e_max=10.0, n_bins=5, adapt_interval=200, min_samples=100)
        # Feed 199 in-range values
        for _ in range(199):
            p.assign(torch.tensor(5.0))
        # Feed 1 extreme outlier (triggers adaptation at call 200)
        p.assign(torch.tensor(1000.0))

        # Edges should NOT expand to cover the outlier
        assert p.edges[-1].item() <= 10.1, (
            f"Edges expanded to {p.edges[-1].item()} due to outlier — should stay near 10.0"
        )

    def test_adaptive_partition_memory_bounded(self):
        """Bug C: History should be bounded regardless of iteration count."""
        p = AdaptivePartition(e_min=0.0, e_max=10.0, n_bins=5, adapt_interval=1000, min_samples=100)
        max_history = 50_000  # expected default bound
        for i in range(100_000):
            p.assign(torch.tensor(float(i % 10)))
        assert len(p._history) <= max_history, (
            f"History grew to {len(p._history)}, expected max {max_history}"
        )


class TestQuantilePartition:
    def test_uniform_samples(self):
        energies = torch.linspace(0, 10, 1000)
        p = QuantilePartition(energies, n_bins=10)
        assert p.n_partitions == 10
        # Quantile edges should be roughly uniform for uniform data
        assert abs(p.edges[5].item() - 5.0) < 0.1

    def test_skewed_samples(self):
        # Exponential-like distribution — more mass at low energies
        energies = torch.exp(torch.linspace(0, 3, 1000))
        p = QuantilePartition(energies, n_bins=4)
        # Bins should be narrower where data is dense (low energy)
        widths = (p.edges[1:] - p.edges[:-1]).tolist()
        assert widths[0] < widths[-1]  # first bin narrower than last

    def test_all_bins_reachable(self):
        energies = torch.randn(10000)
        p = QuantilePartition(energies, n_bins=5)
        bins_hit = set()
        for e in energies[:500]:
            bins_hit.add(p.assign(e))
        assert len(bins_hit) >= 4  # most bins should be hit


class TestExpandablePartition:
    def test_in_range_no_expansion(self):
        p = ExpandablePartition(e_min=0.0, e_max=10.0, n_bins=10)
        assert p.n_partitions == 10
        idx = p.assign(torch.tensor(5.0))
        assert idx == 5
        assert not p.expanded

    def test_expand_high(self):
        p = ExpandablePartition(e_min=0.0, e_max=10.0, n_bins=10, expand_step=5)
        idx = p.assign(torch.tensor(15.0))
        assert p.expanded
        assert p.n_partitions == 15
        assert idx >= 0  # should now be assigned

    def test_expand_low(self):
        p = ExpandablePartition(e_min=0.0, e_max=10.0, n_bins=10, expand_step=5)
        idx = p.assign(torch.tensor(-3.0))
        assert p.expanded
        assert p.n_partitions == 15
        assert idx >= 0

    def test_max_bins_cap(self):
        p = ExpandablePartition(e_min=0.0, e_max=10.0, n_bins=10, expand_step=5, max_bins=12)
        p.assign(torch.tensor(15.0))
        assert p.n_partitions == 12  # capped at max_bins

    def test_max_bins_stops_expansion(self):
        p = ExpandablePartition(e_min=0.0, e_max=10.0, n_bins=10, expand_step=5, max_bins=10)
        idx = p.assign(torch.tensor(100.0))
        assert not p.expanded  # can't expand
        assert idx == -1  # still out of range

    def test_batch_expansion(self):
        p = ExpandablePartition(e_min=0.0, e_max=10.0, n_bins=10, expand_step=5)
        energies = torch.tensor([5.0, 15.0, -3.0])
        bins = p.assign_batch(energies)
        assert p.expanded
        assert p.n_partitions == 20  # expanded both directions
        assert (bins >= 0).all()  # all assigned after expansion

    def test_samc_weights_auto_resize(self):
        """SAMCWeights should auto-resize when ExpandablePartition expands."""
        from illuma_samc.gain import GainSequence
        from illuma_samc.weight_manager import SAMCWeights

        p = ExpandablePartition(e_min=0.0, e_max=10.0, n_bins=10, expand_step=5)
        gain = GainSequence("1/t", t0=50)
        wm = SAMCWeights(partition=p, gain=gain)
        assert wm.theta.shape[0] == 10

        # Step with out-of-range energy triggers expansion + resize
        wm.step(1, 15.0)
        assert wm.theta.shape[0] == 15
        assert wm.counts.shape[0] == 15

    def test_correction_with_expansion(self):
        """correction() should handle expansion gracefully."""
        from illuma_samc.gain import GainSequence
        from illuma_samc.weight_manager import SAMCWeights

        p = ExpandablePartition(e_min=0.0, e_max=10.0, n_bins=10, expand_step=5)
        gain = GainSequence("1/t", t0=50)
        wm = SAMCWeights(partition=p, gain=gain)

        # Correction to out-of-range proposed energy should trigger expansion
        c = wm.correction(5.0, 15.0)
        assert p.n_partitions == 15
        assert c != float("-inf")  # should be valid after expansion

    def test_edges_update_after_expansion(self):
        p = ExpandablePartition(e_min=0.0, e_max=10.0, n_bins=10, expand_step=5)
        old_edges = p.edges.clone()
        p.assign(torch.tensor(15.0))
        assert p.edges.shape[0] == 16  # 15 bins + 1
        assert p.edges[-1].item() > old_edges[-1].item()


class TestGrowingPartition:
    def test_starts_with_3_bins(self):
        p = GrowingPartition(bin_width=1.0, center=0.0)
        assert p.n_partitions == 3
        # Edges: [-inf, -0.5, 0.5, inf]
        assert p.edges[0].item() == float("-inf")
        assert p.edges[-1].item() == float("inf")

    def test_first_energy_sets_center(self):
        p = GrowingPartition(bin_width=1.0)
        assert p.n_partitions == 2  # placeholder: 0 core + 2 overflow
        idx = p.assign(torch.tensor(5.0))
        assert p.n_partitions == 3  # now initialized
        assert idx == 1  # center bin
        assert p._e_min == 4.5
        assert p._e_max == 5.5

    def test_eager_growth_high(self):
        p = GrowingPartition(bin_width=1.0, center=0.0, growth="eager")
        assert p.n_partitions == 3
        # Energy at 3.0 is above e_max=0.5, triggers eager expansion
        idx = p.assign(torch.tensor(3.0))
        assert p.n_partitions > 3
        assert idx >= 1  # should be assigned to a core bin, not overflow

    def test_eager_growth_low(self):
        p = GrowingPartition(bin_width=1.0, center=0.0, growth="eager")
        idx = p.assign(torch.tensor(-3.0))
        assert p.n_partitions > 3
        assert idx >= 1  # core bin

    def test_lazy_growth_waits(self):
        p = GrowingPartition(bin_width=1.0, center=0.0, growth="lazy", expand_threshold=5)
        # First 4 out-of-range visits should NOT trigger expansion
        for _ in range(4):
            idx = p.assign(torch.tensor(3.0))
        assert p.n_partitions == 3  # no growth yet
        assert idx == 2  # high overflow bin

    def test_lazy_growth_triggers_after_threshold(self):
        p = GrowingPartition(bin_width=1.0, center=0.0, growth="lazy", expand_threshold=5)
        for _ in range(5):
            p.assign(torch.tensor(3.0))
        assert p.n_partitions > 3  # growth triggered

    def test_max_bins_respected(self):
        p = GrowingPartition(bin_width=0.1, center=0.0, max_bins=10, growth="eager")
        # Try to force massive expansion
        p.assign(torch.tensor(100.0))
        assert p.n_partitions <= 10

    def test_assign_batch(self):
        p = GrowingPartition(bin_width=1.0, center=0.0, growth="eager")
        energies = torch.tensor([-3.0, 0.0, 3.0])
        bins = p.assign_batch(energies)
        assert bins.shape == (3,)
        assert p.n_partitions > 3  # should have grown
        # All should be assigned to valid bins
        assert (bins >= 0).all()

    def test_assign_batch_lazy(self):
        p = GrowingPartition(bin_width=1.0, center=0.0, growth="lazy", expand_threshold=5)
        # Batch with 6 values above range should trigger growth
        energies = torch.tensor([3.0] * 6)
        p.assign_batch(energies)
        assert p.n_partitions > 3

    def test_edges_shape(self):
        p = GrowingPartition(bin_width=1.0, center=0.0, growth="eager")
        p.assign(torch.tensor(5.0))
        assert p.edges.shape[0] == p.n_partitions + 1

    def test_samc_weights_auto_integration(self):
        """SAMCWeights.auto() runs end-to-end on 2D problem."""
        import math

        from illuma_samc.problems.multimodal_2d import energy_fn
        from illuma_samc.weight_manager import SAMCWeights

        wm = SAMCWeights.auto(bin_width=0.2, growth="eager")
        x = torch.zeros(2)
        raw = energy_fn(x)
        fx = float(raw[0]) if isinstance(raw, tuple) else float(raw)

        for t in range(1, 1001):
            x_new = x + 0.25 * torch.randn(2)
            raw = energy_fn(x_new)
            if isinstance(raw, tuple):
                fy, in_r = float(raw[0]), bool(raw[1])
            else:
                fy, in_r = float(raw), True

            log_r = (-fy + fx) / 1.0 + wm.correction(fx, fy)
            if in_r and (log_r > 0 or math.log(torch.rand(1).item() + 1e-300) < log_r):
                x, fx = x_new.clone(), fy
            wm.step(t, fx)

        # Partition should have grown beyond initial 3 bins
        assert wm.n_bins > 3
        # Should have some counts
        assert wm.counts.sum().item() > 0

    def test_resize_preserves_weights_on_high_growth(self):
        """Weights should be preserved when growing high."""
        from illuma_samc.gain import GainSequence
        from illuma_samc.weight_manager import SAMCWeights

        p = GrowingPartition(bin_width=1.0, center=0.0, growth="eager")
        gain = GainSequence("1/t", t0=50)
        wm = SAMCWeights(partition=p, gain=gain)

        # Step a few times in range to build up some counts
        for t in range(1, 5):
            wm.step(t, 0.0)
        center_counts = wm.counts[1].item()
        assert center_counts > 0

        # Now trigger high growth
        wm.step(5, 5.0)
        # Center bin counts should be preserved (still at index 1)
        assert wm.counts[1].item() == center_counts

    def test_resize_preserves_weights_on_low_growth(self):
        """Weights should be preserved when growing low (bins shift right)."""
        from illuma_samc.gain import GainSequence
        from illuma_samc.weight_manager import SAMCWeights

        p = GrowingPartition(bin_width=1.0, center=5.0, growth="eager")
        gain = GainSequence("1/t", t0=50)
        wm = SAMCWeights(partition=p, gain=gain)

        # Step in the center bin
        for t in range(1, 5):
            wm.step(t, 5.0)
        old_n = wm.n_bins
        center_counts = wm.counts[1].item()

        # Trigger low growth
        wm.step(5, -5.0)
        new_n = wm.n_bins
        added = new_n - old_n

        # The old center bin should now be at index 1 + added
        assert wm.counts[1 + added].item() == center_counts
