"""Diagnostic plots for SAMC runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from illuma_samc.sampler import SAMC
    from illuma_samc.weight_manager import SAMCWeights

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
    save_path: str | None = None,
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
        e_np = energies.detach().cpu().numpy()
        # Multi-chain: shape (n_steps, N) — plot mean across chains
        if e_np.ndim == 2:
            axes[0, 1].plot(e_np.mean(axis=1), linewidth=0.3, alpha=0.7, color="darkorange")
        else:
            axes[0, 1].plot(e_np, linewidth=0.3, alpha=0.7, color="darkorange")
    elif isinstance(energies, list) and len(energies) > 0 and isinstance(energies[0], torch.Tensor):
        # Multi-chain: list of (N,) tensors → stack and average
        e_np = torch.stack(energies).cpu().mean(dim=1).numpy()
        axes[0, 1].plot(e_np, linewidth=0.3, alpha=0.7, color="darkorange")
    else:
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
    # Flatten energy history to 1-D for rolling rate computation
    e_hist = sampler.energy_history
    if isinstance(e_hist, torch.Tensor):
        e_1d = e_hist.detach().cpu()
        if e_1d.ndim == 2:
            e_1d = e_1d.mean(dim=1)  # average across chains
    elif isinstance(e_hist, list) and len(e_hist) > 0 and isinstance(e_hist[0], torch.Tensor):
        # Multi-chain: list of (N,) tensors → stack and average
        e_1d = torch.stack(e_hist).cpu().mean(dim=1)
    else:
        e_1d = torch.tensor(e_hist)
    n = len(e_1d)
    if n > rolling_window:
        # Compute acceptance from energy changes (energy changed => accepted)
        changed = (e_1d[1:] != e_1d[:-1]).float()
        # Pad with zeros at start
        padded = torch.cat([torch.zeros(rolling_window - 1).to(changed.device), changed])
        kernel = torch.ones(rolling_window).to(changed.device) / rolling_window
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
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_weight_diagnostics(
    wm: SAMCWeights,
    *,
    figsize: tuple[float, float] = (14, 10),
    save_path: str | None = None,
) -> object:
    """Plot diagnostic panels for a SAMCWeights instance.

    Panels:
    - (0,0) Bin visits vs energy (visited bins only)
    - (0,1) Flatness over time (from recorded history)
    - (1,0) Theta bar chart (centered)
    - (1,1) Bin visit trajectories (from history snapshots)

    Parameters
    ----------
    wm : SAMCWeights
        A weight manager that has been used in a sampling loop.
    figsize : tuple
        Figure size in inches.
    save_path : str, optional
        If provided, save the figure to this path (e.g. ``"samc_diagnostic.png"``).
        If ``None``, display the plot interactively.

    Returns
    -------
    matplotlib.figure.Figure
        The diagnostic figure.
    """
    _require_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # (0,0) Bin visit histogram — visited bins only, x-axis = energy
    counts = wm.counts.detach().cpu()
    visited = counts > 0
    if visited.any() and hasattr(wm, "partition") and wm.partition is not None:
        edges = wm.partition.edges.detach().cpu().float()
        centers = (edges[:-1] + edges[1:]) / 2
        # Clamp inf edges for display
        finite_edges = edges[torch.isfinite(edges)]
        if len(finite_edges) >= 2:
            lo, hi = finite_edges[0].item(), finite_edges[-1].item()
            centers = centers.clamp(lo, hi)
        vis_centers = centers[visited].numpy()
        vis_counts = counts[visited].numpy()
        bin_width = (vis_centers[-1] - vis_centers[0]) / max(len(vis_centers) - 1, 1) * 0.8
        axes[0, 0].bar(vis_centers, vis_counts, width=bin_width, color="green", alpha=0.8)
        axes[0, 0].set_xlabel("Energy")
    else:
        axes[0, 0].bar(range(len(counts)), counts.numpy(), color="green", alpha=0.8)
        axes[0, 0].set_xlabel("Bin index")
    axes[0, 0].set_ylabel("Visit count")
    axes[0, 0].set_title("Bin Visits (visited only)")
    axes[0, 0].grid(alpha=0.2)

    # (0,1) Flatness over time
    flatness_hist = wm.flatness_history()
    if flatness_hist:
        steps = [(i + 1) * wm._record_every for i in range(len(flatness_hist))]
        axes[0, 1].plot(steps, flatness_hist, linewidth=1.0, color="teal")
        axes[0, 1].set_xlabel("Iteration")
        axes[0, 1].set_ylabel("Flatness")
        axes[0, 1].set_title("Flatness Over Time")
        axes[0, 1].set_ylim(-0.1, 1.1)
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            "No history recorded",
            ha="center",
            va="center",
            transform=axes[0, 1].transAxes,
        )
        axes[0, 1].set_title("Flatness Over Time")
    axes[0, 1].grid(alpha=0.2)

    # (1,0) Theta bar chart (centered) — visited bins only, x-axis = energy
    theta_raw = wm.theta.detach().cpu()
    if visited.any() and hasattr(wm, "partition") and wm.partition is not None:
        vis_theta = theta_raw[visited]
        vis_theta = (vis_theta - vis_theta.mean()).numpy()
        axes[1, 0].bar(vis_centers, vis_theta, width=bin_width, color="steelblue", alpha=0.8)
        axes[1, 0].set_xlabel("Energy")
    else:
        theta = (theta_raw - theta_raw.mean()).numpy()
        axes[1, 0].bar(range(len(theta)), theta, color="steelblue", alpha=0.8)
        axes[1, 0].set_xlabel("Bin index")
    axes[1, 0].axhline(0, color="black", linewidth=0.5, alpha=0.5)
    axes[1, 0].set_ylabel("Log weight (theta - mean)")
    axes[1, 0].set_title("SAMC Log Weights (visited only)")
    axes[1, 0].grid(alpha=0.2)

    # (1,1) Theta trajectory from history
    if wm.bin_counts_history:
        # Pad snapshots to max length (partition may have expanded over time)
        max_bins = max(h.shape[0] for h in wm.bin_counts_history)
        padded = []
        for h in wm.bin_counts_history:
            if h.shape[0] < max_bins:
                h = torch.cat([h, torch.zeros(max_bins - h.shape[0], dtype=h.dtype)])
            padded.append(h)
        history = torch.stack(padded).cpu().numpy()
        steps = [(i + 1) * wm._record_every for i in range(len(wm.bin_counts_history))]
        n_bins = history.shape[1]
        for b in range(n_bins):
            axes[1, 1].plot(steps, history[:, b], linewidth=0.5, alpha=0.6)
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("Cumulative visits")
        axes[1, 1].set_title("Bin Visit Trajectories")
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "No history recorded",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].set_title("Bin Visit Trajectories")
    axes[1, 1].grid(alpha=0.2)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig
