"""Polished all-in-one demo for illuma-samc README.

Generates a publication-quality figure showing:
1. Target distribution contour (5-mode Gaussian mixture)
2. SAMC samples overlaid on contours
3. Weight convergence over time
4. Bin visit histogram (flat = good)
5. SAMC vs MH comparison (MH stuck, SAMC explores all modes)
"""

import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from illuma_samc.partitions import UniformPartition
from illuma_samc.proposals import GaussianProposal
from illuma_samc.sampler import SAMC

matplotlib.use("Agg")

# ── 5-mode Gaussian mixture ──

MEANS = torch.tensor(
    [
        [0.0, 0.0],
        [4.0, 4.0],
        [-4.0, 4.0],
        [4.0, -4.0],
        [-4.0, -4.0],
    ]
)
SIGMA = 0.7


def gaussian_mixture_energy(x: torch.Tensor) -> torch.Tensor:
    """Negative log of a 5-component Gaussian mixture."""
    if x.dim() == 2:
        # Batch mode: x is (N, 2)
        diffs = x.unsqueeze(1) - MEANS.unsqueeze(0)  # (N, 5, 2)
        log_components = -0.5 * torch.sum(diffs**2, dim=2) / SIGMA**2
        return -torch.logsumexp(log_components, dim=1)
    # Single mode: x is (2,)
    diffs = x.unsqueeze(0) - MEANS  # (5, 2)
    log_components = -0.5 * torch.sum(diffs**2, dim=1) / SIGMA**2
    return -torch.logsumexp(log_components, dim=0)


def energy_grid(resolution: int = 300):
    """Evaluate energy on a grid for contour plotting."""
    t = torch.linspace(-7, 7, resolution)
    X, Y = torch.meshgrid(t, t, indexing="xy")
    pts = torch.stack([X.flatten(), Y.flatten()], dim=1)
    E = gaussian_mixture_energy(pts)
    return X.numpy(), Y.numpy(), E.reshape(resolution, resolution).numpy()


# ── MH baseline ──


def run_mh(energy_fn, n_iters: int = 200_000, proposal_std: float = 0.5):
    """Standard Metropolis-Hastings (no SAMC weights)."""
    x = torch.zeros(2)
    fx = energy_fn(x).item()
    best_x, best_e = x.clone(), fx
    path = [x.clone()]
    accept_count = 0

    for _ in range(1, n_iters + 1):
        y = x + proposal_std * torch.randn(2)
        fy = energy_fn(y).item()
        log_r = -fy + fx

        if log_r > 0 or torch.rand(1).item() < math.exp(log_r):
            x = y.clone()
            fx = fy
            accept_count += 1
        if fx < best_e:
            best_e = fx
            best_x = x.clone()
        path.append(x.clone())

    return {
        "path": torch.stack(path).numpy(),
        "best_x": best_x.numpy(),
        "best_e": best_e,
        "accept_rate": accept_count / n_iters,
    }


def main():
    torch.manual_seed(42)

    # ── Run SAMC (Simple API) ──
    print("Running SAMC (200K iterations, 5-mode Gaussian mixture)...")
    sampler = SAMC(
        energy_fn=gaussian_mixture_energy,
        dim=2,
        n_partitions=25,
        e_min=-2.0,
        e_max=15.0,
        proposal_std=0.8,
        gain="1/t",
        gain_kwargs={"t0": 2000},
    )
    result = sampler.run(n_steps=200_000, save_every=10, progress=True)
    print(
        f"  SAMC — Accept rate: {result.acceptance_rate:.3f}, Best energy: {result.best_energy:.4f}"
    )

    # ── Run MH for comparison ──
    print("Running MH (200K iterations)...")
    torch.manual_seed(42)
    mh = run_mh(gaussian_mixture_energy, n_iters=200_000, proposal_std=0.8)
    print(f"  MH   — Accept rate: {mh['accept_rate']:.3f}, Best energy: {mh['best_e']:.4f}")

    # ── Run Flexible API demo (custom proposal + acceptance) ──
    print("Running SAMC (Flexible API demo)...")
    custom_proposal = GaussianProposal(step_size=0.8)
    custom_partition = UniformPartition(e_min=-2.0, e_max=15.0, n_bins=25)

    def custom_log_accept(x, x_new, energy_x, energy_new):
        """Standard Boltzmann acceptance (to demonstrate flexible API)."""
        return -energy_new.item() + energy_x.item()

    torch.manual_seed(42)
    flex_sampler = SAMC(
        energy_fn=gaussian_mixture_energy,
        dim=2,
        n_partitions=25,
        proposal_fn=custom_proposal,
        partition_fn=custom_partition,
        gain="1/t",
        gain_kwargs={"t0": 2000},
    )
    flex_result = flex_sampler.run(n_steps=200_000, save_every=10, progress=True)
    print(f"  Flexible API — Accept rate: {flex_result.acceptance_rate:.3f}")

    # ── Generate figure ──
    print("Generating showcase figure...")
    X, Y, E = energy_grid()
    samc_samples = result.samples.numpy()
    mh_samples = mh["path"][::10]  # subsample MH for plotting

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Style
    contour_levels = np.linspace(-2.0, 15.0, 40)
    mode_color = "white"
    cmap = "viridis"

    # (1) Target distribution contour
    ax = axes[0, 0]
    ax.contourf(X, Y, E, levels=contour_levels, cmap=cmap)
    for m in MEANS.numpy():
        ax.plot(
            m[0],
            m[1],
            "o",
            color=mode_color,
            markersize=8,
            markeredgecolor="k",
            markeredgewidth=1.5,
        )
    ax.set_title("Target Energy Landscape", fontsize=13, fontweight="bold")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)

    # (2) SAMC samples overlaid
    ax = axes[0, 1]
    ax.contourf(X, Y, E, levels=contour_levels, cmap=cmap, alpha=0.3)
    ax.scatter(samc_samples[:, 0], samc_samples[:, 1], s=0.5, alpha=0.4, c="red", rasterized=True)
    for m in MEANS.numpy():
        ax.plot(
            m[0],
            m[1],
            "o",
            color=mode_color,
            markersize=8,
            markeredgecolor="k",
            markeredgewidth=1.5,
        )
    ax.set_title("SAMC Samples — All 5 Modes Found", fontsize=13, fontweight="bold")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)

    # (3) SAMC vs MH comparison
    ax = axes[0, 2]
    ax.contourf(X, Y, E, levels=contour_levels, cmap=cmap, alpha=0.3)
    ax.scatter(
        mh_samples[:, 0],
        mh_samples[:, 1],
        s=0.5,
        alpha=0.4,
        c="orange",
        label="MH",
        rasterized=True,
    )
    for m in MEANS.numpy():
        ax.plot(
            m[0],
            m[1],
            "o",
            color=mode_color,
            markersize=8,
            markeredgecolor="k",
            markeredgewidth=1.5,
        )
    ax.set_title("MH Samples — Stuck in One Mode", fontsize=13, fontweight="bold")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)

    # (4) Weight convergence — centered theta (subtract mean, like reference code)
    ax = axes[1, 0]
    theta = result.log_weights.numpy()
    theta_centered = theta - theta.mean()
    ax.bar(
        range(len(theta_centered)), theta_centered, color="steelblue", alpha=0.8, edgecolor="none"
    )
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_title("SAMC Log Weights ($\\theta$ centered)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Energy bin")
    ax.set_ylabel("$\\theta_k - \\bar{\\theta}$")
    ax.grid(alpha=0.2)

    # (5) Bin visit histogram
    ax = axes[1, 1]
    counts = result.bin_counts.numpy()
    ax.bar(range(len(counts)), counts, color="darkorange", alpha=0.8, edgecolor="none")
    ax.axhline(
        counts.mean(),
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {counts.mean():.0f}",
    )
    ax.set_title("Bin Visits — Flat Exploration", fontsize=13, fontweight="bold")
    ax.set_xlabel("Energy bin")
    ax.set_ylabel("Visit count")
    ax.legend()
    ax.grid(alpha=0.2)

    # (6) Energy trace comparison
    ax = axes[1, 2]
    # SAMC energy trace (subsample for plotting)
    samc_energies = result.energy_history.numpy()
    step = max(1, len(samc_energies) // 2000)
    samc_sub = samc_energies[::step]
    ax.plot(
        np.arange(len(samc_sub)) * step,
        samc_sub,
        alpha=0.5,
        linewidth=0.5,
        color="steelblue",
        label="SAMC",
    )
    # MH energy trace
    mh_path = mh["path"]
    mh_energies = np.array(
        [
            gaussian_mixture_energy(torch.tensor(p, dtype=torch.float32)).item()
            for p in mh_path[::step]
        ]
    )
    ax.plot(
        np.arange(len(mh_energies)) * step,
        mh_energies,
        alpha=0.5,
        linewidth=0.5,
        color="orange",
        label="MH",
    )
    ax.set_title("Energy Trace: SAMC vs MH", fontsize=13, fontweight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Energy")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.suptitle(
        "illuma-samc: Stochastic Approximation Monte Carlo",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("assets/demo_showcase.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved assets/demo_showcase.png (300 DPI)")


if __name__ == "__main__":
    main()
