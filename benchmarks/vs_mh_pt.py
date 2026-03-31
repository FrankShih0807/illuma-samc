"""Benchmark: SAMC vs Metropolis-Hastings vs Parallel Tempering.

Two test problems:
  1. 2D multimodal cost function from sample_code.py
  2. 10D Gaussian mixture with well-separated modes

Metrics: best energy found, ESS, acceptance rate, wall-clock time.
"""

import math
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from illuma_samc import SAMC

matplotlib.use("Agg")


# ────────────────────────────────────────────────────────
# Test Problem 1: 2D multimodal cost (from sample_code.py)
# ────────────────────────────────────────────────────────


def cost_2d(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch-compatible 2D multimodal cost. Returns (energy, in_region)."""
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
    return energy.squeeze(), in_region.squeeze()


# ────────────────────────────────────────────────────────
# Test Problem 2: 10D Gaussian mixture (well-separated)
# ────────────────────────────────────────────────────────

# 4 modes at distance 10 from origin in different directions
_MODES_10D = torch.zeros(4, 10)
_MODES_10D[0, 0] = 10.0
_MODES_10D[1, 0] = -10.0
_MODES_10D[2, 1] = 10.0
_MODES_10D[3, 1] = -10.0


def gaussian_mixture_10d(z: torch.Tensor) -> torch.Tensor:
    """Batch-compatible 10D Gaussian mixture energy.

    E(x) = -log sum_k exp(-0.5 * ||x - mu_k||^2)
    """
    if z.dim() == 1:
        z = z.unsqueeze(0)
    # z: (N, 10), modes: (4, 10) → diffs: (N, 4, 10)
    diffs = z.unsqueeze(1) - _MODES_10D.unsqueeze(0)
    log_components = -0.5 * torch.sum(diffs**2, dim=-1)  # (N, 4)
    energy = -torch.logsumexp(log_components, dim=-1)  # (N,)
    return energy.squeeze()


# ────────────────────────────────────────────────────────
# Standard Metropolis-Hastings
# ────────────────────────────────────────────────────────


def run_mh(
    energy_fn,
    dim: int,
    n_iters: int,
    proposal_std: float = 0.25,
    temperature: float = 1.0,
    x0: torch.Tensor | None = None,
) -> dict:
    """Run standard MH. Returns dict of metrics + samples."""
    x = x0.clone() if x0 is not None else torch.zeros(dim)
    result = energy_fn(x)
    if isinstance(result, tuple):
        fx, _ = result
        fx = fx.item()
    else:
        fx = result.item()

    best_x, best_e = x.clone(), fx
    accept_count = 0
    energies = []

    for _ in range(1, n_iters + 1):
        y = x + proposal_std * torch.randn(dim)
        result = energy_fn(y)
        if isinstance(result, tuple):
            fy, in_r = result
            fy_val = fy.item()
            if isinstance(in_r, torch.Tensor):
                in_r = in_r.item()
        else:
            fy_val = result.item()
            in_r = True

        log_r = (-fy_val + fx) / temperature

        if not in_r:
            accept = False
        elif log_r > 0:
            accept = True
        else:
            accept = torch.rand(1).item() < math.exp(log_r)

        if accept:
            x = y.clone()
            fx = fy_val
            accept_count += 1
        if fx < best_e:
            best_e = fx
            best_x = x.clone()
        energies.append(fx)

    return {
        "best_energy": best_e,
        "best_x": best_x,
        "acceptance_rate": accept_count / n_iters,
        "energies": torch.tensor(energies),
    }


# ────────────────────────────────────────────────────────
# Parallel Tempering
# ────────────────────────────────────────────────────────


def run_parallel_tempering(
    energy_fn,
    dim: int,
    n_iters: int,
    n_replicas: int = 8,
    proposal_std: float = 0.25,
    t_min: float = 0.1,
    t_max: float = 10.0,
    swap_interval: int = 10,
) -> dict:
    """Parallel tempering with geometric temperature ladder."""
    temps = torch.logspace(math.log10(t_min), math.log10(t_max), n_replicas)

    # Initialize replicas
    states = [torch.zeros(dim) for _ in range(n_replicas)]
    energies_list: list[list[float]] = [[] for _ in range(n_replicas)]

    # Compute initial energies
    fxs = []
    for i in range(n_replicas):
        result = energy_fn(states[i])
        if isinstance(result, tuple):
            e, _ = result
            fxs.append(e.item())
        else:
            fxs.append(result.item())

    best_e = min(fxs)
    best_x = states[fxs.index(best_e)].clone()
    accept_counts = [0] * n_replicas
    swap_count = 0
    swap_attempts = 0

    for it in range(1, n_iters + 1):
        # MH step for each replica
        for i in range(n_replicas):
            y = states[i] + proposal_std * torch.randn(dim)
            result = energy_fn(y)
            if isinstance(result, tuple):
                fy, in_r = result
                fy_val = fy.item()
                if isinstance(in_r, torch.Tensor):
                    in_r = in_r.item()
            else:
                fy_val = result.item()
                in_r = True

            log_r = (-fy_val + fxs[i]) / temps[i].item()

            if not in_r:
                accept = False
            elif log_r > 0:
                accept = True
            else:
                accept = torch.rand(1).item() < math.exp(log_r)

            if accept:
                states[i] = y.clone()
                fxs[i] = fy_val
                accept_counts[i] += 1

            if fxs[i] < best_e:
                best_e = fxs[i]
                best_x = states[i].clone()

            energies_list[i].append(fxs[i])

        # Replica swaps
        if it % swap_interval == 0:
            for i in range(n_replicas - 1):
                swap_attempts += 1
                delta = (1.0 / temps[i].item() - 1.0 / temps[i + 1].item()) * (fxs[i + 1] - fxs[i])
                if delta > 0 or torch.rand(1).item() < math.exp(delta):
                    states[i], states[i + 1] = states[i + 1], states[i]
                    fxs[i], fxs[i + 1] = fxs[i + 1], fxs[i]
                    swap_count += 1

    # Return coldest replica stats
    return {
        "best_energy": best_e,
        "best_x": best_x,
        "acceptance_rate": accept_counts[0] / n_iters,
        "swap_rate": swap_count / max(swap_attempts, 1),
        "energies": torch.tensor(energies_list[0]),
    }


# ────────────────────────────────────────────────────────
# Effective Sample Size (ESS)
# ────────────────────────────────────────────────────────


def compute_ess(energies: torch.Tensor) -> float:
    """Estimate ESS from energy trace using autocorrelation."""
    n = len(energies)
    if n < 10:
        return float(n)
    x = energies.float()
    x = x - x.mean()
    var = x.var().item()
    if var < 1e-12:
        return float(n)

    # Compute autocorrelation up to lag n//2
    max_lag = min(n // 2, 5000)
    acf_sum = 0.0
    for lag in range(1, max_lag + 1):
        c = torch.mean(x[:-lag] * x[lag:]).item() / var
        if c < 0.05:
            break
        acf_sum += c

    tau = 1.0 + 2.0 * acf_sum
    return n / tau


# ────────────────────────────────────────────────────────
# Main benchmark
# ────────────────────────────────────────────────────────


def benchmark_problem(
    name: str,
    energy_fn,
    dim: int,
    n_iters: int,
    samc_kwargs: dict,
    mh_kwargs: dict,
    pt_kwargs: dict,
) -> dict:
    """Run all three methods and collect metrics."""
    results = {}
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    # --- SAMC ---
    torch.manual_seed(42)
    print(f"  Running SAMC ({n_iters:,} iters)...")
    t0 = time.perf_counter()
    sampler = SAMC(energy_fn=energy_fn, dim=dim, **samc_kwargs)
    samc_result = sampler.run(n_steps=n_iters, save_every=100, progress=True)
    t_samc = time.perf_counter() - t0
    samc_ess = compute_ess(samc_result.energy_history.flatten())
    results["samc"] = {
        "best_energy": samc_result.best_energy,
        "acceptance_rate": samc_result.acceptance_rate,
        "ess": samc_ess,
        "wall_time": t_samc,
        "energies": samc_result.energy_history.flatten(),
    }
    print(
        f"    best_E={samc_result.best_energy:.4f}  "
        f"acc={samc_result.acceptance_rate:.3f}  "
        f"ESS={samc_ess:.0f}  time={t_samc:.1f}s"
    )

    # --- MH ---
    torch.manual_seed(42)
    print(f"  Running MH ({n_iters:,} iters)...")
    t0 = time.perf_counter()
    mh_result = run_mh(energy_fn, dim, n_iters, **mh_kwargs)
    t_mh = time.perf_counter() - t0
    mh_ess = compute_ess(mh_result["energies"])
    results["mh"] = {
        "best_energy": mh_result["best_energy"],
        "acceptance_rate": mh_result["acceptance_rate"],
        "ess": mh_ess,
        "wall_time": t_mh,
        "energies": mh_result["energies"],
    }
    print(
        f"    best_E={mh_result['best_energy']:.4f}  "
        f"acc={mh_result['acceptance_rate']:.3f}  "
        f"ESS={mh_ess:.0f}  time={t_mh:.1f}s"
    )

    # --- Parallel Tempering ---
    torch.manual_seed(42)
    print(f"  Running PT ({n_iters:,} iters, {pt_kwargs.get('n_replicas', 8)} replicas)...")
    t0 = time.perf_counter()
    pt_result = run_parallel_tempering(energy_fn, dim, n_iters, **pt_kwargs)
    t_pt = time.perf_counter() - t0
    pt_ess = compute_ess(pt_result["energies"])
    results["pt"] = {
        "best_energy": pt_result["best_energy"],
        "acceptance_rate": pt_result["acceptance_rate"],
        "swap_rate": pt_result["swap_rate"],
        "ess": pt_ess,
        "wall_time": t_pt,
        "energies": pt_result["energies"],
    }
    print(
        f"    best_E={pt_result['best_energy']:.4f}  "
        f"acc={pt_result['acceptance_rate']:.3f}  "
        f"swap={pt_result['swap_rate']:.3f}  "
        f"ESS={pt_ess:.0f}  time={t_pt:.1f}s"
    )

    return results


def plot_comparison(results_2d: dict, results_10d: dict):
    """Generate comparison plots for both problems."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(
        "SAMC vs MH vs Parallel Tempering",
        fontsize=14,
        fontweight="bold",
    )

    methods = ["samc", "mh", "pt"]
    labels = ["SAMC", "MH", "PT"]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for row, (title, results) in enumerate(
        [("2D Multimodal", results_2d), ("10D Gaussian Mixture", results_10d)]
    ):
        # Best energy bar chart
        ax = axes[row, 0]
        best_es = [results[m]["best_energy"] for m in methods]
        ax.bar(labels, best_es, color=colors, alpha=0.8)
        ax.set_title(f"{title}\nBest Energy")
        ax.set_ylabel("Energy")

        # ESS bar chart
        ax = axes[row, 1]
        ess_vals = [results[m]["ess"] for m in methods]
        ax.bar(labels, ess_vals, color=colors, alpha=0.8)
        ax.set_title(f"{title}\nEffective Sample Size")
        ax.set_ylabel("ESS")

        # Acceptance rate bar chart
        ax = axes[row, 2]
        acc_vals = [results[m]["acceptance_rate"] for m in methods]
        ax.bar(labels, acc_vals, color=colors, alpha=0.8)
        ax.set_title(f"{title}\nAcceptance Rate")
        ax.set_ylabel("Rate")
        ax.set_ylim(0, 1)

        # Wall-clock time bar chart
        ax = axes[row, 3]
        time_vals = [results[m]["wall_time"] for m in methods]
        ax.bar(labels, time_vals, color=colors, alpha=0.8)
        ax.set_title(f"{title}\nWall-Clock Time")
        ax.set_ylabel("Seconds")

    for ax in axes.flat:
        ax.grid(alpha=0.2, axis="y")

    plt.tight_layout()
    plt.savefig("benchmarks/benchmark_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("\nPlot saved to benchmarks/benchmark_comparison.png")

    # Energy trace plot
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("Energy Trace Comparison", fontsize=14, fontweight="bold")

    for col, (title, results) in enumerate(
        [("2D Multimodal", results_2d), ("10D Gaussian Mixture", results_10d)]
    ):
        ax = axes2[col]
        for m, label, color in zip(methods, labels, colors):
            e = results[m]["energies"].numpy()
            # Subsample for plotting
            step = max(1, len(e) // 5000)
            ax.plot(
                np.arange(0, len(e), step),
                e[::step],
                label=label,
                color=color,
                alpha=0.7,
                linewidth=0.5,
            )
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Energy")
        ax.legend()
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig("benchmarks/energy_traces.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Plot saved to benchmarks/energy_traces.png")


def print_summary_table(results_2d: dict, results_10d: dict):
    """Print markdown-formatted summary table."""
    print("\n## Benchmark Results\n")
    print("| Problem | Method | Best Energy | ESS | Acc. Rate | Time (s) |")
    print("|---------|--------|-------------|-----|-----------|----------|")
    for problem_name, results in [
        ("2D Multimodal", results_2d),
        ("10D Gaussian", results_10d),
    ]:
        for m, label in [("samc", "SAMC"), ("mh", "MH"), ("pt", "PT")]:
            r = results[m]
            print(
                f"| {problem_name} | {label} | "
                f"{r['best_energy']:.4f} | "
                f"{r['ess']:.0f} | "
                f"{r['acceptance_rate']:.3f} | "
                f"{r['wall_time']:.1f} |"
            )


def main():
    n_iters_2d = 500_000
    n_iters_10d = 200_000

    results_2d = benchmark_problem(
        name="2D Multimodal Cost Function",
        energy_fn=cost_2d,
        dim=2,
        n_iters=n_iters_2d,
        samc_kwargs={
            "n_partitions": 42,
            "e_min": -8.2,
            "e_max": 0.0,
            "proposal_std": 0.25,
            "gain": "ramp",
            "gain_kwargs": {
                "rho": 1.0,
                "tau": 1.0,
                "warmup": 1,
                "step_scale": 1000,
            },
        },
        mh_kwargs={"proposal_std": 0.25, "temperature": 1.0},
        pt_kwargs={
            "n_replicas": 8,
            "proposal_std": 0.25,
            "t_min": 0.1,
            "t_max": 10.0,
            "swap_interval": 10,
        },
    )

    results_10d = benchmark_problem(
        name="10D Gaussian Mixture (4 modes, separation=10)",
        energy_fn=gaussian_mixture_10d,
        dim=10,
        n_iters=n_iters_10d,
        samc_kwargs={
            "n_partitions": 30,
            "e_min": -5.0,
            "e_max": 30.0,
            "proposal_std": 1.0,
            "gain": "ramp",
            "gain_kwargs": {
                "rho": 1.0,
                "tau": 1.0,
                "warmup": 1,
                "step_scale": 1000,
            },
        },
        mh_kwargs={"proposal_std": 1.0, "temperature": 1.0},
        pt_kwargs={
            "n_replicas": 8,
            "proposal_std": 1.0,
            "t_min": 0.5,
            "t_max": 20.0,
            "swap_interval": 10,
        },
    )

    plot_comparison(results_2d, results_10d)
    print_summary_table(results_2d, results_10d)


if __name__ == "__main__":
    main()
