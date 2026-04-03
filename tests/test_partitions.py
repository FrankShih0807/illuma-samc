"""Tests for energy-space partitions."""

import torch

from illuma_samc.partitions import (
    ExpandablePartition,
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


class TestDevice:
    """Tests for device parameter on partitions."""

    def test_uniform_partition_device_cpu(self):
        """UniformPartition edges are on the specified device."""
        from illuma_samc.partitions import UniformPartition

        p = UniformPartition(0, 10, 20, device="cpu")
        assert p.edges.device.type == "cpu"

    def test_expandable_partition_device_cpu(self):
        """ExpandablePartition edges are on the specified device."""
        from illuma_samc.partitions import ExpandablePartition

        p = ExpandablePartition(0, 10, 20, device="cpu")
        assert p.edges.device.type == "cpu"

    def test_assign_batch_result_on_input_device(self):
        """assign_batch result is on the same device as the input tensor."""
        from illuma_samc.partitions import UniformPartition

        p = UniformPartition(0, 10, 10, device="cpu")
        energies = torch.tensor([1.0, 5.0, 9.0])
        bins = p.assign_batch(energies)
        assert bins.device.type == energies.device.type
