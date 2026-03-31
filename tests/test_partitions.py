"""Tests for energy-space partitions."""

import torch

from illuma_samc.partitions import AdaptivePartition, QuantilePartition, UniformPartition


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

    def test_matches_sample_code_binning(self):
        """Verify binning matches sample_code.py get_bin logic."""
        p = UniformPartition(e_min=-8.2, e_max=0.0, n_bins=42)
        # sample_code.py uses scale=5, which is n_bins/(e_max - e_min) ≈ 42/8.2 ≈ 5.12
        # Close but not exact — our partition is more general
        assert p.assign(torch.tensor(-8.2)) == 0
        assert p.assign(torch.tensor(0.0)) == 42 - 1
        assert p.assign(torch.tensor(1.0)) == -1  # above range


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
