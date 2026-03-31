"""Diagnostic: compare SAMC partition strategies on 10D Gaussian mixture."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from illuma_samc import SAMC
from illuma_samc.gain import GainSequence
from illuma_samc.partitions import AdaptivePartition
from illuma_samc.proposals import GaussianProposal

matplotlib.use("Agg")

# 4 modes at distance 10 from origin in different directions
_MODES_10D = torch.zeros(4, 10)
_MODES_10D[0, 0] = 10.0
_MODES_10D[1, 0] = -10.0
_MODES_10D[2, 1] = 10.0
_MODES_10D[3, 1] = -10.0


def gaussian_mixture_10d(z: torch.Tensor) -> torch.Tensor:
    if z.dim() == 1:
        z = z.unsqueeze(0)
    diffs = z.unsqueeze(1) - _MODES_10D.unsqueeze(0)
    log_components = -0.5 * torch.sum(diffs**2, dim=-1)
    energy = -torch.logsumexp(log_components, dim=-1)
    return energy.squeeze()


def run_config(name, sampler, n_steps=200_000, save_every=100):
    """Run SAMC and return diagnostics dict."""
    print(f"\n--- {name} ---")
    result = sampler.run(n_steps=n_steps, save_every=save_every, progress=True)

    theta = result.log_weights.numpy()
    counts = result.bin_counts.numpy()

    # Compute sample energies
    sample_energies = []
    for i in range(result.samples.shape[0]):
        e = gaussian_mixture_10d(result.samples[i]).item()
        sample_energies.append(e)
    sample_energies = np.array(sample_energies)

    # Compute ESS from importance weights
    iw = result.importance_weights
    ess = (iw.sum() ** 2 / (iw**2).sum()).item()

    print(f"  Best energy: {result.best_energy:.4f}")
    print(f"  Accept rate: {result.acceptance_rate:.3f}")
    print(f"  Importance ESS: {ess:.0f} / {len(iw)}")
    print(f"  Theta range: [{theta.min():.1f}, {theta.max():.1f}]")
    print(f"  Empty bins: {(counts == 0).sum()} / {len(counts)}")
    print(f"  Sample E range: [{sample_energies.min():.2f}, {sample_energies.max():.2f}]")

    return {
        "theta": theta,
        "counts": counts,
        "sample_energies": sample_energies,
        "best_energy": result.best_energy,
        "acceptance_rate": result.acceptance_rate,
        "ess": ess,
        "n_samples": len(iw),
        "result": result,
    }


configs = {}

# 1. Original: bad range
torch.manual_seed(42)
sampler1 = SAMC(
    energy_fn=gaussian_mixture_10d,
    dim=10,
    n_partitions=30,
    e_min=-5.0,
    e_max=30.0,
    proposal_std=1.0,
    gain="ramp",
    gain_kwargs={"rho": 1.0, "tau": 1.0, "warmup": 1, "step_scale": 1000},
)
configs["Original (e∈[-5,30])"] = run_config("Original (e∈[-5,30])", sampler1)

# 2. Tightened range
torch.manual_seed(42)
sampler2 = SAMC(
    energy_fn=gaussian_mixture_10d,
    dim=10,
    n_partitions=30,
    e_min=0.0,
    e_max=15.0,
    proposal_std=1.0,
    gain="ramp",
    gain_kwargs={"rho": 1.0, "tau": 1.0, "warmup": 1, "step_scale": 1000},
)
configs["Tight (e∈[0,15])"] = run_config("Tight (e∈[0,15])", sampler2)

# 3. Adaptive partition
torch.manual_seed(42)
sampler3 = SAMC(
    energy_fn=gaussian_mixture_10d,
    dim=10,
    proposal_fn=GaussianProposal(step_size=1.0),
    partition_fn=AdaptivePartition(e_min=-5.0, e_max=30.0, n_bins=30, adapt_interval=5000),
    gain=GainSequence("ramp", rho=1.0, tau=1.0, warmup=1, step_scale=1000),
)
configs["Adaptive (starts [-5,30])"] = run_config("Adaptive (starts [-5,30])", sampler3)

# --- Plot ---
fig, axes = plt.subplots(3, 4, figsize=(18, 13))
fig.suptitle(
    "SAMC Partition Strategy Comparison: 10D Gaussian Mixture", fontsize=14, fontweight="bold"
)

for row, (label, d) in enumerate(configs.items()):
    theta = d["theta"]
    counts = d["counts"]
    n_bins = len(theta)

    # Infer bin centers from counts length
    # For uniform, we can just use range indices
    centers = np.arange(n_bins)
    w = 0.8

    # Log weights (centered)
    theta_c = theta - theta.mean()
    axes[row, 0].bar(centers, theta_c, width=w, color="steelblue", alpha=0.8)
    axes[row, 0].axhline(0, color="k", linewidth=0.5)
    axes[row, 0].set_title(f"{label}\nLog Weights (θ − mean)")
    axes[row, 0].set_ylabel("θ − mean(θ)")

    # Bin visits
    axes[row, 1].bar(centers, counts, width=w, color="darkorange", alpha=0.8)
    axes[row, 1].set_title("Bin Visit Counts")

    # Importance weights
    weights = np.exp(theta - theta.max())
    axes[row, 2].bar(centers, weights, width=w, color="green", alpha=0.8)
    axes[row, 2].set_title("Learned Weights")

    # Sample energy histogram
    axes[row, 3].hist(
        d["sample_energies"], bins=50, color="purple", alpha=0.7, edgecolor="black", linewidth=0.3
    )
    axes[row, 3].set_title(f"Sample Energies (ESS={d['ess']:.0f}/{d['n_samples']})")
    axes[row, 3].axvline(
        d["best_energy"], color="red", linestyle="--", label=f"best={d['best_energy']:.3f}"
    )
    axes[row, 3].legend(fontsize=8)

for ax in axes[-1]:
    ax.set_xlabel("Bin index / Energy")

plt.tight_layout()
plt.savefig("benchmarks/debug_10d_diagnostics.png", dpi=200, bbox_inches="tight")
plt.close()
print("\nPlot saved to benchmarks/debug_10d_diagnostics.png")
