"""Classic multimodal Gaussian mixture demo.

Shows SAMC exploring all modes while standard MH can get stuck.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from illuma_samc.sampler import SAMC

# ── Gaussian mixture energy ──

# 4 well-separated modes in 2D
MEANS = torch.tensor([[-3.0, -3.0], [3.0, -3.0], [-3.0, 3.0], [3.0, 3.0]])
SIGMA = 0.5


def gaussian_mixture_energy(x: torch.Tensor) -> torch.Tensor:
    """Negative log of a 4-component Gaussian mixture."""
    diffs = x.unsqueeze(0) - MEANS  # [4, 2]
    log_components = -0.5 * torch.sum(diffs**2, dim=1) / SIGMA**2
    return -torch.logsumexp(log_components, dim=0)


def main():
    matplotlib.use("Agg")
    torch.manual_seed(42)

    # GPU usage: pass device="cuda" or device="mps" (Apple Silicon).
    # dtype="float32" is the default; use dtype="float64" for extra precision on CPU.
    # Example: SAMC(..., device="mps", dtype="float32")
    print("Running SAMC on 4-mode Gaussian mixture...")
    sampler = SAMC(
        energy_fn=gaussian_mixture_energy,
        dim=2,
        n_partitions=20,
        e_min=0.0,
        e_max=40.0,
        proposal_std=0.5,
        gain="1/t",
        gain_kwargs={"t0": 1000},
    )
    result = sampler.run(n_steps=200_000, save_every=20, progress=True)

    print(f"  Accept rate: {result.acceptance_rate:.3f}")
    print(f"  Best energy: {result.best_energy:.4f}")

    # ── Check that all 4 modes were visited ──
    samples = result.samples.numpy()
    quadrant_counts = {
        "(-,-)": np.sum((samples[:, 0] < 0) & (samples[:, 1] < 0)),
        "(+,-)": np.sum((samples[:, 0] > 0) & (samples[:, 1] < 0)),
        "(-,+)": np.sum((samples[:, 0] < 0) & (samples[:, 1] > 0)),
        "(+,+)": np.sum((samples[:, 0] > 0) & (samples[:, 1] > 0)),
    }
    print("  Quadrant visits:")
    for q, c in quadrant_counts.items():
        print(f"    {q}: {c}")

    # Check weight convergence
    weights = result.log_weights.numpy()
    weight_range = weights.max() - weights.min()
    print(f"  Weight range: {weight_range:.2f} (smaller = more uniform)")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Samples
    axes[0].scatter(samples[:, 0], samples[:, 1], s=0.5, alpha=0.3, c="steelblue")
    for m in MEANS.numpy():
        axes[0].plot(m[0], m[1], "r+", markersize=15, markeredgewidth=2)
    axes[0].set_title("SAMC Samples")
    axes[0].set_aspect("equal")
    axes[0].set_xlim(-6, 6)
    axes[0].set_ylim(-6, 6)

    # Weight histogram
    axes[1].bar(range(len(weights)), weights, color="steelblue", alpha=0.8)
    axes[1].set_title("Log Weights (θ)")
    axes[1].set_xlabel("Bin index")
    axes[1].grid(alpha=0.2)

    # Bin counts
    counts = result.bin_counts.numpy()
    axes[2].bar(range(len(counts)), counts, color="green", alpha=0.8)
    axes[2].set_title("Bin Visit Counts")
    axes[2].set_xlabel("Bin index")
    axes[2].grid(alpha=0.2)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "gaussian_mixture.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
