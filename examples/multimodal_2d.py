"""Reproduce sample_code.py results using the illuma-samc API.

Runs both SAMC and a standard MH comparison on the 2D multimodal cost function.
"""

import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from illuma_samc.sampler import SAMC

# ── Cost function (from sample_code.py / cost.c) ──


def cost(z: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """2D multimodal cost. Returns (energy, in_region)."""
    if z.dim() == 1:
        z = z.unsqueeze(0)
    x1, x2 = z[:, 0], z[:, 1]

    out_of_bounds = (x1 < -1.1) | (x1 > 1.1) | (x2 < -1.1) | (x2 > 1.1)

    d1 = x1 * torch.sin(20 * x2) + x2 * torch.sin(20 * x1)
    sum1 = d1**2 * torch.cosh(torch.sin(10 * x1) * x1)

    d2 = x1 * torch.cos(10 * x2) - x2 * torch.sin(10 * x1)
    sum2 = d2**2 * torch.cosh(torch.cos(20 * x2) * x2)

    energy = -sum1 - sum2
    energy = torch.where(out_of_bounds, torch.tensor(1e100), energy)
    in_region = ~out_of_bounds

    return energy.squeeze(), in_region.squeeze().item()


def cost_grid(resolution: int = 500):
    """Evaluate cost on a grid for contour plotting."""
    t = torch.linspace(-1.1, 1.1, resolution)
    X, Y = torch.meshgrid(t, t, indexing="xy")
    Z = torch.stack([X.flatten(), Y.flatten()], dim=1)
    if Z.dim() == 1:
        Z = Z.unsqueeze(0)
    x1, x2 = Z[:, 0], Z[:, 1]
    d1 = x1 * torch.sin(20 * x2) + x2 * torch.sin(20 * x1)
    sum1 = d1**2 * torch.cosh(torch.sin(10 * x1) * x1)
    d2 = x1 * torch.cos(10 * x2) - x2 * torch.sin(10 * x1)
    sum2 = d2**2 * torch.cosh(torch.cos(20 * x2) * x2)
    E = -sum1 - sum2
    return X.numpy(), Y.numpy(), E.reshape(resolution, resolution).numpy()


# ── MH baseline (simple loop, no SAMC) ──


def run_mh(n_iters: int = 1_000_000, temperature: float = 1.0, proposal_std: float = 0.25):
    """Run standard Metropolis-Hastings for comparison."""
    x = torch.zeros(2)
    fx, _ = cost(x)
    fx_val = fx.item()
    best_x, best_e = x.clone(), fx_val
    path = [x.clone()]
    accept_count = 0

    for _ in range(1, n_iters + 1):
        y = x + proposal_std * torch.randn(2)
        fy, in_reg = cost(y)
        fy_val = fy.item()
        log_r = (-fy_val + fx_val) / temperature

        if not in_reg:
            accept = False
        elif log_r > 0:
            accept = True
        else:
            accept = torch.rand(1).item() < math.exp(log_r)

        if accept:
            x = y.clone()
            fx_val = fy_val
            accept_count += 1
        if fx_val < best_e:
            best_e = fx_val
            best_x = x.clone()
        path.append(x.clone())

    return {
        "path": torch.stack(path).numpy(),
        "best_x": best_x.numpy(),
        "best_e": best_e,
        "accept_rate": accept_count / n_iters,
    }


def main():
    matplotlib.use("Agg")
    torch.manual_seed(42)

    # ── Run SAMC via illuma-samc API ──
    print("=" * 50)
    print("Running SAMC (1M iterations) via illuma-samc API...")
    print("=" * 50)

    sampler = SAMC(
        energy_fn=cost,
        dim=2,
        n_partitions=42,
        e_min=-8.2,
        e_max=0.0,
        proposal_std=0.25,
        gain="ramp",
        gain_kwargs={"rho": 1.0, "tau": 1.0, "warmup": 1, "step_scale": 1000},
    )
    result = sampler.run(n_steps=1_000_000, save_every=100, progress=True)

    print(f"  Best energy: {result.best_energy:.5f}")
    print(f"  Best x:      {result.best_x.numpy()}")
    print(f"  Accept rate: {result.acceptance_rate:.3f}")

    # ── Run MH baseline ──
    print()
    print("=" * 50)
    print("Running Metropolis-Hastings (1M iterations)...")
    print("=" * 50)
    mh = run_mh()
    print(f"  Best energy: {mh['best_e']:.5f}")
    print(f"  Best x:      {mh['best_x']}")
    print(f"  Accept rate: {mh['accept_rate']:.3f}")

    # ── Plot ──
    X, Y, E = cost_grid()
    samples = result.samples.numpy()
    levels = np.linspace(-8.2, 0, 50)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    axes[0, 0].contourf(X, Y, np.clip(E, -8.2, 0), levels=levels, cmap="viridis")
    axes[0, 0].set_title("Cost Function")
    axes[0, 0].set_aspect("equal")

    axes[0, 1].contourf(X, Y, np.clip(E, -8.2, 0), levels=levels, cmap="viridis", alpha=0.4)
    axes[0, 1].scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.3, c="red")
    bx = result.best_x.numpy()
    axes[0, 1].plot(bx[0], bx[1], "w*", markersize=15, markeredgecolor="k")
    axes[0, 1].set_title(f"SAMC Path (best E={result.best_energy:.3f})")
    axes[0, 1].set_xlim(-1.1, 1.1)
    axes[0, 1].set_ylim(-1.1, 1.1)
    axes[0, 1].set_aspect("equal")

    axes[0, 2].contourf(X, Y, np.clip(E, -8.2, 0), levels=levels, cmap="viridis", alpha=0.4)
    axes[0, 2].scatter(mh["path"][:, 0], mh["path"][:, 1], s=1, alpha=0.5, c="orange")
    axes[0, 2].plot(*mh["best_x"], "w*", markersize=15, markeredgecolor="k")
    axes[0, 2].set_title(f"MH Path (best E={mh['best_e']:.3f})")
    axes[0, 2].set_xlim(-1.1, 1.1)
    axes[0, 2].set_ylim(-1.1, 1.1)
    axes[0, 2].set_aspect("equal")

    theta = result.log_weights.numpy()
    counts = result.bin_counts.numpy()
    edges = np.linspace(-8.2, 0.0, 43)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w = (centers[1] - centers[0]) * 0.8

    # Density estimate
    log_density = theta.copy()
    log_density -= log_density.max()
    log_density -= np.log(np.sum(np.exp(log_density)))
    log_density += np.log(100.0)
    axes[1, 0].bar(centers, np.exp(log_density), width=w, color="steelblue", alpha=0.7)
    axes[1, 0].set_title("SAMC Density Estimate")

    axes[1, 1].bar(centers, counts, width=w, color="darkorange", alpha=0.7)
    axes[1, 1].set_title("SAMC Bin Visits")

    weights = np.exp(theta - theta.max())
    axes[1, 2].bar(centers, weights, width=w, color="green", alpha=0.7)
    axes[1, 2].set_title("SAMC Learned Weights")

    plt.tight_layout()
    plt.savefig("multimodal_2d.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("\nPlot saved to multimodal_2d.png")


if __name__ == "__main__":
    main()
