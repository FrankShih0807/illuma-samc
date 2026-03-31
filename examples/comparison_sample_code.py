"""Side-by-side comparison of sample_code.py vs illuma-samc API.

Generates a comparison plot showing both implementations produce
equivalent results on the 2D multimodal cost function.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "reference"))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")

# ── Run sample_code.py reference ──
# ── Run illuma-samc API ──
from sample_code import cost
from sample_code import run_samc as run_samc_ref

from illuma_samc.sampler import SAMC


def cost_api(z: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Wrapper for illuma-samc API (returns scalar bool)."""
    if z.dim() == 1:
        z = z.unsqueeze(0)
    energy, in_region = cost(z)
    return energy.squeeze(), in_region.squeeze().item()


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("Running reference sample_code.py SAMC...")
    ref = run_samc_ref()
    print(f"  Reference — best energy: {ref['best_e']:.5f}, accept rate: {ref['accept_rate']:.3f}")

    torch.manual_seed(42)
    print("Running illuma-samc API SAMC...")
    sampler = SAMC(
        energy_fn=cost_api,
        dim=2,
        n_partitions=42,
        e_min=-8.2,
        e_max=0.0,
        proposal_std=0.25,
        gain="ramp",
        gain_kwargs={"rho": 1.0, "tau": 1.0, "warmup": 1, "step_scale": 1000},
    )
    result = sampler.run(n_steps=1_000_000, save_every=100, progress=True)
    print(f"  API      — best energy: {result.best_energy:.5f}")
    print(f"  API      — accept rate: {result.acceptance_rate:.3f}")

    # ── Side-by-side comparison plot ──
    edges = np.linspace(-8.2, 0.0, 43)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w = (centers[1] - centers[0]) * 0.8

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Verification: sample_code.py vs illuma-samc API", fontsize=14, fontweight="bold")

    # Bin visits comparison
    axes[0, 0].bar(centers, ref["counts"], width=w, alpha=0.7, color="darkorange")
    axes[0, 0].set_title(f"Reference — Bin Visits (acc={ref['accept_rate']:.3f})")
    axes[0, 0].set_xlabel("Energy")
    axes[0, 0].set_ylabel("Visit count")

    api_counts = result.bin_counts.numpy()
    axes[0, 1].bar(centers, api_counts, width=w, alpha=0.7, color="darkorange")
    axes[0, 1].set_title(f"illuma-samc — Bin Visits (acc={result.acceptance_rate:.3f})")
    axes[0, 1].set_xlabel("Energy")
    axes[0, 1].set_ylabel("Visit count")

    # Learned weights comparison
    ref_weights = np.exp(ref["theta"] - ref["theta"].max())
    axes[1, 0].bar(centers, ref_weights, width=w, alpha=0.7, color="green")
    axes[1, 0].set_title(f"Reference — Weights (best E={ref['best_e']:.3f})")
    axes[1, 0].set_xlabel("Energy")
    axes[1, 0].set_ylabel("Weight")

    api_theta = result.log_weights.numpy()
    api_weights = np.exp(api_theta - api_theta.max())
    axes[1, 1].bar(centers, api_weights, width=w, alpha=0.7, color="green")
    axes[1, 1].set_title(f"illuma-samc — Weights (best E={result.best_energy:.3f})")
    axes[1, 1].set_xlabel("Energy")
    axes[1, 1].set_ylabel("Weight")

    for ax in axes.flat:
        ax.grid(alpha=0.2)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "comparison_verification.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nComparison plot saved to {out}")


if __name__ == "__main__":
    main()
