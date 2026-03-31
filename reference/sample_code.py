"""
PyTorch translation of the SAMC C experiment (samc/ folder).

Two samplers on a 2D multimodal cost function:
  1. SAMC (Stochastic Approximation Monte Carlo) — adaptive flat-histogram
  2. Standard Metropolis-Hastings — for comparison

Cost function (from cost.c):
  f(x1, x2) = -(d1^2 * cosh(sin(10*x1)*x1) + d2^2 * cosh(cos(20*x2)*x2))
  where d1 = x1*sin(20*x2) + x2*sin(20*x1)
        d2 = x1*cos(10*x2) - x2*sin(10*x1)
  Domain: [-1.1, 1.1]^2, returns 1e100 outside.
"""

import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# ──────────────────────────────────────────────
# Cost function (cost.c)
# ──────────────────────────────────────────────

def cost(z):
    """
    Vectorized cost function.
    z: [batch, 2] or [2]
    Returns: energy (scalar or [batch]), in_region (bool)
    """
    if z.dim() == 1:
        z = z.unsqueeze(0)
    x1, x2 = z[:, 0], z[:, 1]

    out_of_bounds = (x1 < -1.1) | (x1 > 1.1) | (x2 < -1.1) | (x2 > 1.1)

    d1 = x1 * torch.sin(20 * x2) + x2 * torch.sin(20 * x1)
    sum1 = d1 ** 2 * torch.cosh(torch.sin(10 * x1) * x1)

    d2 = x1 * torch.cos(10 * x2) - x2 * torch.sin(10 * x1)
    sum2 = d2 ** 2 * torch.cosh(torch.cos(20 * x2) * x2)

    energy = -sum1 - sum2
    energy = torch.where(out_of_bounds, torch.tensor(1e100), energy)
    in_region = ~out_of_bounds

    return energy, in_region


def cost_grid(resolution=500):
    """Evaluate cost on a grid for contour plotting."""
    t = torch.linspace(-1.1, 1.1, resolution)
    X, Y = torch.meshgrid(t, t, indexing="xy")
    Z = torch.stack([X.flatten(), Y.flatten()], dim=1)
    E, _ = cost(Z)
    return X.numpy(), Y.numpy(), E.reshape(resolution, resolution).numpy()


# ──────────────────────────────────────────────
# SAMC sampler (histest0.c)
# ──────────────────────────────────────────────

def run_samc(
    n_iters=1_000_000,
    n_bins=42,       # grid+1 = 42 bins (indices 0..41)
    scale=5,
    low_e=-8.2,
    max_e=0.0,
    proposal_std=0.25,
    tau=1.0,
    rho=1.0,
    warm=1,
    step_scale=1000,
    save_every=100,
):
    """Run SAMC and return path, best solution, theta weights, bin counts."""
    # Uniform reference density
    refden = torch.full((n_bins,), 1.0 / n_bins)

    # theta (log weights) and visit counts per bin
    theta = torch.zeros(n_bins)
    counts = torch.zeros(n_bins)

    # Initialize at origin
    x = torch.zeros(2)
    fx, in_region = cost(x)
    fx = fx.item()

    best_x = x.clone()
    best_e = fx

    path = []
    accept_count = 0

    def get_bin(e):
        if e > max_e:
            return n_bins - 1
        elif e < low_e:
            return 0
        else:
            return min(int((e - low_e) * scale), n_bins - 1)

    for it in tqdm(range(1, n_iters + 1), desc="SAMC"):
        # Adaptive step size (from histest0.c)
        if it <= warm * step_scale:
            delta = rho
        else:
            delta = rho * np.exp(-tau * np.log((it - (warm - 1) * step_scale) / step_scale))

        k1 = get_bin(fx)

        # Gaussian random-walk proposal
        y = x + proposal_std * torch.randn(2)
        fy, in_reg = cost(y)
        fy = fy.item()
        in_reg = in_reg.item()

        k2 = get_bin(fy)

        # SAMC acceptance ratio (log scale)
        log_r = theta[k1].item() - theta[k2].item() - fy + fx

        if not in_reg:
            accept = False
        elif log_r > 0:
            accept = True
        else:
            accept = (np.random.rand() < np.exp(log_r))

        if accept:
            x = y.clone()
            fx = fy
            # Weight update on accepted bin
            theta -= delta * refden
            theta[k2] += delta
            counts[k2] += 1
            accept_count += 1
        else:
            # Weight update on current bin
            theta -= delta * refden
            theta[k1] += delta
            counts[k1] += 1

        if fx < best_e:
            best_e = fx
            best_x = x.clone()

        if it % save_every == 0:
            path.append(x.clone())

    accept_rate = accept_count / n_iters
    path = torch.stack(path).numpy()

    # Post-process theta into normalized log-density (from histest0.c)
    log_density = theta.clone()
    for i in range(n_bins):
        if counts[i] <= 0:
            log_density[i] += torch.log(refden[i] + refden[counts <= 0].mean())
        else:
            log_density[i] += torch.log(refden[i])
    log_density -= log_density.max()
    log_density -= torch.logsumexp(log_density, dim=0)
    log_density += np.log(100.0)

    bin_edges = torch.linspace(low_e, max_e, n_bins + 1)

    return {
        "path": path,
        "best_x": best_x.numpy(),
        "best_e": best_e,
        "theta": theta.numpy(),
        "log_density": log_density.numpy(),
        "counts": counts.numpy(),
        "accept_rate": accept_rate,
        "bin_edges": bin_edges.numpy(),
    }


# ──────────────────────────────────────────────
# Metropolis-Hastings sampler (met.c)
# ──────────────────────────────────────────────

def run_mh(
    n_iters=2000,
    temperature=0.1,
    proposal_std=0.02,
):
    """Run standard MH and return path, best solution."""
    x = torch.zeros(2)
    fx, _ = cost(x)
    fx = fx.item()

    best_x = x.clone()
    best_e = fx
    accept_count = 0
    path = [x.clone()]

    for k in tqdm(range(1, n_iters + 1), desc="MH"):
        y = x + proposal_std * torch.randn(2)
        fy, in_reg = cost(y)
        fy = fy.item()
        in_reg = in_reg.item()

        log_r = (-fy + fx) / temperature

        if not in_reg:
            accept = False
        elif log_r > 0:
            accept = True
        else:
            accept = (np.random.rand() < np.exp(log_r))

        if accept:
            x = y.clone()
            fx = fy
            accept_count += 1

        if fx < best_e:
            best_e = fx
            best_x = x.clone()

        path.append(x.clone())

    path = torch.stack(path).numpy()
    accept_rate = accept_count / n_iters
    return {
        "path": path,
        "best_x": best_x.numpy(),
        "best_e": best_e,
        "accept_rate": accept_rate,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    # Run both samplers
    print("=" * 50)
    print("Running SAMC (1M iterations)...")
    print("=" * 50)
    samc = run_samc()
    print(f"  Best energy: {samc['best_e']:.5f}")
    print(f"  Best x:      {samc['best_x']}")
    print(f"  Accept rate: {samc['accept_rate']:.3f}")

    print()
    print("=" * 50)
    print("Running Metropolis-Hastings (2K iterations)...")
    print("=" * 50)
    mh = run_mh()
    print(f"  Best energy: {mh['best_e']:.5f}")
    print(f"  Best x:      {mh['best_x']}")
    print(f"  Accept rate: {mh['accept_rate']:.3f}")

    # ── Plotting ──
    X, Y, E = cost_grid(resolution=500)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # (0,0) Cost function contour
    levels = np.linspace(-8.2, 0, 50)
    axes[0, 0].contourf(X, Y, np.clip(E, -8.2, 0), levels=levels, cmap="viridis")
    axes[0, 0].set_title("Cost Function")
    axes[0, 0].set_xlabel("x1")
    axes[0, 0].set_ylabel("x2")
    axes[0, 0].set_aspect("equal")

    # (0,1) SAMC sample path
    axes[0, 1].contourf(X, Y, np.clip(E, -8.2, 0), levels=levels, cmap="viridis", alpha=0.4)
    axes[0, 1].scatter(samc["path"][:, 0], samc["path"][:, 1], s=1, alpha=0.3, c="red")
    axes[0, 1].plot(*samc["best_x"], "w*", markersize=15, markeredgecolor="k")
    axes[0, 1].set_title(f"SAMC Path (best E={samc['best_e']:.3f})")
    axes[0, 1].set_xlabel("x1")
    axes[0, 1].set_ylabel("x2")
    axes[0, 1].set_xlim(-1.1, 1.1)
    axes[0, 1].set_ylim(-1.1, 1.1)
    axes[0, 1].set_aspect("equal")

    # (0,2) MH sample path
    axes[0, 2].contourf(X, Y, np.clip(E, -8.2, 0), levels=levels, cmap="viridis", alpha=0.4)
    axes[0, 2].scatter(mh["path"][:, 0], mh["path"][:, 1], s=1, alpha=0.5, c="orange")
    axes[0, 2].plot(*mh["best_x"], "w*", markersize=15, markeredgecolor="k")
    axes[0, 2].set_title(f"MH Path (best E={mh['best_e']:.3f})")
    axes[0, 2].set_xlabel("x1")
    axes[0, 2].set_ylabel("x2")
    axes[0, 2].set_xlim(-1.1, 1.1)
    axes[0, 2].set_ylim(-1.1, 1.1)
    axes[0, 2].set_aspect("equal")

    # (1,0) SAMC density estimate
    bin_centers = 0.5 * (samc["bin_edges"][:-1] + samc["bin_edges"][1:])
    density = np.exp(samc["log_density"])
    axes[1, 0].bar(bin_centers, density, width=(bin_centers[1] - bin_centers[0]) * 0.8, color="steelblue", alpha=0.7)
    axes[1, 0].set_xlabel("Energy")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title("SAMC Density Estimate")
    axes[1, 0].grid(alpha=0.2)

    # (1,1) SAMC bin visit counts
    axes[1, 1].bar(bin_centers, samc["counts"], width=(bin_centers[1] - bin_centers[0]) * 0.8, color="darkorange", alpha=0.7)
    axes[1, 1].set_xlabel("Energy")
    axes[1, 1].set_ylabel("Visit count")
    axes[1, 1].set_title("SAMC Bin Visits")
    axes[1, 1].grid(alpha=0.2)

    # (1,2) SAMC theta weights (exponentiated)
    weights = np.exp(samc["theta"] - samc["theta"].max())
    axes[1, 2].bar(bin_centers, weights, width=(bin_centers[1] - bin_centers[0]) * 0.8, color="green", alpha=0.7)
    axes[1, 2].set_xlabel("Energy")
    axes[1, 2].set_ylabel("Weight")
    axes[1, 2].set_title("SAMC Learned Weights")
    axes[1, 2].grid(alpha=0.2)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "samc_experiment.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {out}")