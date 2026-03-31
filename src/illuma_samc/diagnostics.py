"""Diagnostic plots for SAMC runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from illuma_samc.sampler import SAMC

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]


def _require_matplotlib() -> None:
    if plt is None:
        raise ImportError(
            "matplotlib is required for diagnostics. Install with: pip install illuma-samc[viz]"
        )


def plot_diagnostics(
    sampler: SAMC,
    *,
    rolling_window: int = 1000,
    figsize: tuple[float, float] = (14, 10),
) -> None:
    """Plot diagnostic panels for a completed SAMC run.

    Parameters
    ----------
    sampler : SAMC
        A sampler that has already called ``.run()``.
    rolling_window : int
        Window size for the rolling acceptance rate.
    figsize : tuple
        Figure size in inches.
    """
    _require_matplotlib()

    if sampler.log_weights is None:
        raise RuntimeError("Sampler has not been run yet — call .run() first.")

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # (0,0) Weight trajectory (theta), centered by subtracting mean
    theta_raw = sampler.log_weights.detach().cpu()
    theta = (theta_raw - theta_raw.mean()).numpy()
    axes[0, 0].bar(range(len(theta)), theta, color="steelblue", alpha=0.8)
    axes[0, 0].axhline(0, color="black", linewidth=0.5, alpha=0.5)
    axes[0, 0].set_xlabel("Bin index")
    axes[0, 0].set_ylabel("Log weight (θ − mean)")
    axes[0, 0].set_title("SAMC Log Weights (centered)")
    axes[0, 0].grid(alpha=0.2)

    # (0,1) Energy trace
    energies = sampler.energy_history
    if isinstance(energies, torch.Tensor):
        energies = energies.detach().cpu().numpy()
    axes[0, 1].plot(energies, linewidth=0.3, alpha=0.7, color="darkorange")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Energy")
    axes[0, 1].set_title("Energy Trace")
    axes[0, 1].grid(alpha=0.2)

    # (1,0) Bin visit histogram
    counts = sampler.bin_counts.detach().cpu().numpy()
    axes[1, 0].bar(range(len(counts)), counts, color="green", alpha=0.8)
    axes[1, 0].set_xlabel("Bin index")
    axes[1, 0].set_ylabel("Visit count")
    axes[1, 0].set_title("Bin Visit Histogram")
    axes[1, 0].grid(alpha=0.2)

    # (1,1) Rolling acceptance rate
    n = len(energies)
    if n > rolling_window:
        # Compute acceptance from energy changes (energy changed => accepted)
        e_tensor = torch.tensor(energies)
        changed = (e_tensor[1:] != e_tensor[:-1]).float()
        # Pad with zeros at start
        padded = torch.cat([torch.zeros(rolling_window - 1), changed])
        kernel = torch.ones(rolling_window) / rolling_window
        rolling_rate = torch.nn.functional.conv1d(
            padded.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
        ).squeeze()
        axes[1, 1].plot(rolling_rate.numpy(), linewidth=0.5, alpha=0.8, color="crimson")
    else:
        axes[1, 1].axhline(sampler.acceptance_rate, color="crimson", linewidth=1.5)
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Acceptance rate")
    axes[1, 1].set_title(f"Rolling Acceptance Rate (window={rolling_window})")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(alpha=0.2)

    plt.tight_layout()
    return fig
