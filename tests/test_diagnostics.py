"""Tests for diagnostics (verify no errors, not plot correctness)."""

import torch

from illuma_samc.sampler import SAMC


def _quadratic_energy(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sum(x**2)


class TestPlotDiagnostics:
    def test_no_error_after_run(self):
        """plot_diagnostics should run without error after a sampler run."""
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend

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
        sampler.run(n_steps=2000, progress=False)
        fig = sampler.plot_diagnostics()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_error_before_run(self):
        """plot_diagnostics should raise if sampler hasn't been run."""
        sampler = SAMC(
            energy_fn=_quadratic_energy,
            dim=2,
            n_partitions=5,
            e_min=-5.0,
            e_max=5.0,
            gain="1/t",
        )
        try:
            sampler.plot_diagnostics()
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass

    def test_short_run_no_rolling(self):
        """Short runs should still produce a plot (no rolling rate)."""
        import matplotlib

        matplotlib.use("Agg")

        torch.manual_seed(0)
        sampler = SAMC(
            energy_fn=_quadratic_energy,
            dim=2,
            n_partitions=5,
            e_min=-5.0,
            e_max=5.0,
            gain="1/t",
        )
        sampler.run(n_steps=100, progress=False)
        fig = sampler.plot_diagnostics(rolling_window=500)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)
