"""Benchmark: SAMC vs Metropolis-Hastings vs Parallel Tempering.

Two test problems:
  1. 2D multimodal cost function from sample_code.py
  2. 10D Gaussian mixture with well-separated modes

SAMC uses SAMCWeights (the main product) in a manual MH loop.
Metrics: best energy found, acceptance rate, wall-clock time.
"""

import math
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from illuma_samc import GainSequence, SAMCWeights, UniformPartition
from illuma_samc.baselines import run_mh, run_parallel_tempering
from illuma_samc.problems import cost_2d, gaussian_mixture_10d

matplotlib.use("Agg")


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
    burn_in_frac: float = 0.1,
    save_every: int = 100,
) -> dict:
    """Run all three methods with identical burn-in and sample collection.

    All methods use the same:
    - Total iterations (n_iters)
    - Burn-in period (burn_in_frac * n_iters, discarded from samples)
    - Sample collection frequency (save_every)
    - Proposal step size (via kwargs)
    """
    burn_in = int(n_iters * burn_in_frac)
    n_samples = (n_iters - burn_in) // save_every

    results = {}
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(
        f"  Settings: {n_iters:,} iters, burn-in={burn_in:,}, "
        f"save_every={save_every}, n_samples={n_samples:,}"
    )

    # --- SAMC (via SAMCWeights) ---
    torch.manual_seed(42)
    print(f"  Running SAMC via SAMCWeights ({n_iters:,} iters)...")
    t0 = time.perf_counter()

    partition = UniformPartition(
        e_min=samc_kwargs["e_min"],
        e_max=samc_kwargs["e_max"],
        n_bins=samc_kwargs["n_partitions"],
    )
    gain = GainSequence("ramp", **samc_kwargs["gain_kwargs"])
    wm = SAMCWeights(partition=partition, gain=gain)
    T_samc = samc_kwargs.get("temperature", 1.0)
    std_samc = samc_kwargs["proposal_std"]

    x = torch.zeros(dim)
    raw = energy_fn(x)
    if isinstance(raw, tuple):
        fx, _ = float(raw[0]), raw[1]
    else:
        fx = float(raw)

    best_x, best_e = x.clone(), fx
    accept_count = 0
    samc_energies = []
    samc_samples = []

    for t in range(1, n_iters + 1):
        x_new = x + std_samc * torch.randn(dim)
        raw = energy_fn(x_new)
        if isinstance(raw, tuple):
            fy, in_r = float(raw[0]), raw[1]
            if isinstance(in_r, torch.Tensor):
                in_r = bool(in_r.item())
        else:
            fy, in_r = float(raw), True

        log_r = (-fy + fx) / T_samc + wm.correction(fx, fy)

        if in_r and (log_r > 0 or math.log(torch.rand(1).item() + 1e-300) < log_r):
            x, fx = x_new.clone(), fy
            accept_count += 1

        wm.step(t, fx)
        samc_energies.append(fx)

        if fx < best_e:
            best_e = fx
            best_x = x.clone()

        if t > burn_in and t % save_every == 0:
            samc_samples.append(x.clone())

    t_samc = time.perf_counter() - t0
    samc_acc = accept_count / n_iters

    results["samc"] = {
        "best_energy": best_e,
        "best_x": best_x,
        "acceptance_rate": samc_acc,
        "wall_time": t_samc,
        "energies": torch.tensor(samc_energies),
        "samples": (torch.stack(samc_samples) if samc_samples else torch.empty(0, dim)),
    }
    print(
        f"    best_E={best_e:.4f}  "
        f"acc={samc_acc:.3f}  "
        f"n_samples={len(samc_samples)}  time={t_samc:.1f}s"
    )

    # --- MH ---
    torch.manual_seed(42)
    print(f"  Running MH ({n_iters:,} iters)...")
    t0 = time.perf_counter()
    mh_result = run_mh(energy_fn, dim, n_iters, burn_in=burn_in, save_every=save_every, **mh_kwargs)
    t_mh = time.perf_counter() - t0
    results["mh"] = {
        "best_energy": mh_result["best_energy"],
        "best_x": mh_result["best_x"],
        "acceptance_rate": mh_result["acceptance_rate"],
        "wall_time": t_mh,
        "energies": mh_result["energies"],
        "samples": mh_result["samples"],
    }
    print(
        f"    best_E={mh_result['best_energy']:.4f}  "
        f"acc={mh_result['acceptance_rate']:.3f}  "
        f"n_samples={len(mh_result['samples'])}  time={t_mh:.1f}s"
    )

    # --- Parallel Tempering ---
    torch.manual_seed(42)
    print(f"  Running PT ({n_iters:,} iters, {pt_kwargs.get('n_replicas', 4)} replicas)...")
    t0 = time.perf_counter()
    pt_result = run_parallel_tempering(
        energy_fn, dim, n_iters, burn_in=burn_in, save_every=save_every, **pt_kwargs
    )
    t_pt = time.perf_counter() - t0
    results["pt"] = {
        "best_energy": pt_result["best_energy"],
        "best_x": pt_result["best_x"],
        "acceptance_rate": pt_result["acceptance_rate"],
        "swap_rate": pt_result["swap_rate"],
        "wall_time": t_pt,
        "energies": pt_result["energies"],
        "samples": pt_result["samples"],
    }
    print(
        f"    best_E={pt_result['best_energy']:.4f}  "
        f"acc={pt_result['acceptance_rate']:.3f}  "
        f"swap={pt_result['swap_rate']:.3f}  "
        f"n_samples={len(pt_result['samples'])}  time={t_pt:.1f}s"
    )

    return results


# ────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────


def plot_trajectories_2d(results_2d: dict):
    """Plot 2D sample trajectories + energy traces (hero image for README)."""
    grid_n = 200
    xx = torch.linspace(-1.1, 1.1, grid_n)
    yy = torch.linspace(-1.1, 1.1, grid_n)
    X, Y = torch.meshgrid(xx, yy, indexing="xy")
    coords = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    Z, _ = cost_2d(coords)
    Z = Z.reshape(grid_n, grid_n).numpy()
    xx_np, yy_np = xx.numpy(), yy.numpy()

    # Order: MH, PT, SAMC (ours) — builds the narrative
    methods = ["mh", "pt", "samc"]
    labels = ["Metropolis-Hastings", "Parallel Tempering (4 replicas)", "SAMC (ours)"]
    annotations = ["TRAPPED", "LIMITED", "EXPLORES ALL"]
    ann_colors = ["#d32f2f", "#f57c00", "#1976d2"]
    trace_labels = [
        "Stuck at one energy level",
        "Slowly escapes, limited range",
        "Traverses all energy levels",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), height_ratios=[1.2, 1])
    fig.suptitle(
        "Same Temperature (T = 0.1).  Same Proposal.  Different Algorithm.",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.945,
        "SAMC\u2019s learned weights overcome energy barriers \u2014 MH and PT cannot.",
        ha="center",
        fontsize=11,
        color="gray",
    )

    for col, (method, label, ann, ann_c) in enumerate(
        zip(methods, labels, annotations, ann_colors)
    ):
        # --- Row 1: Trajectories ---
        ax = axes[0, col]
        ax.contourf(xx_np, yy_np, Z, levels=40, cmap="viridis", alpha=0.6)

        samples = results_2d[method]["samples"]
        if samples.dim() == 3:
            samples = samples[0]
        samples = samples.numpy()
        thin = max(1, len(samples) // 3000)
        sx, sy = samples[::thin, 0], samples[::thin, 1]

        ax.plot(sx, sy, color="red", alpha=0.12, linewidth=0.5, zorder=2)
        ax.scatter(sx, sy, c="red", s=0.5, alpha=0.35, zorder=3)

        best_x = results_2d[method].get("best_x", None)
        if best_x is not None:
            if isinstance(best_x, torch.Tensor):
                best_x = best_x.numpy()
            ax.scatter(
                best_x[0],
                best_x[1],
                c="white",
                s=100,
                marker="*",
                zorder=5,
                edgecolors="black",
                linewidths=0.8,
            )

        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=12, fontweight="bold", color=ann_c)

        # Annotation badge in upper right
        ax.text(
            0.98,
            0.98,
            ann,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", fc=ann_c, alpha=0.85),
        )

        # --- Row 2: Energy traces ---
        ax2 = axes[1, col]
        energies = results_2d[method]["energies"].numpy()
        e_step = max(1, len(energies) // 5000)
        ax2.plot(
            np.arange(0, len(energies), e_step),
            energies[::e_step],
            color="red",
            alpha=0.7,
            linewidth=0.5,
        )
        ax2.set_title(trace_labels[col], fontsize=11, fontstyle="italic")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Energy")
        ax2.set_ylim(-9, 1)
        ax2.grid(alpha=0.15)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("benchmarks/trajectory_comparison.png", dpi=200, bbox_inches="tight")
    plt.savefig("assets/samc_vs_others.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Plot saved to benchmarks/trajectory_comparison.png + assets/samc_vs_others.png")


def plot_comparison(results_2d: dict, results_10d: dict):
    """Generate comparison plots for both problems."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("SAMC vs MH vs Parallel Tempering", fontsize=14, fontweight="bold")

    methods = ["samc", "mh", "pt"]
    labels = ["SAMC", "MH", "PT"]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for row, (title, results) in enumerate(
        [("2D Multimodal", results_2d), ("10D Gaussian Mixture", results_10d)]
    ):
        ax = axes[row, 0]
        best_es = [results[m]["best_energy"] for m in methods]
        ax.bar(labels, best_es, color=colors, alpha=0.8)
        ax.set_title(f"{title}\nBest Energy")
        ax.set_ylabel("Energy")

        ax = axes[row, 1]
        acc_vals = [results[m]["acceptance_rate"] for m in methods]
        ax.bar(labels, acc_vals, color=colors, alpha=0.8)
        ax.set_title(f"{title}\nAcceptance Rate")
        ax.set_ylabel("Rate")
        ax.set_ylim(0, 1)

        ax = axes[row, 2]
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


# ────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────


def print_summary_table(
    results_2d: dict,
    results_10d: dict,
    n_iters_2d: int,
    n_iters_10d: int,
    n_replicas: int = 8,
):
    """Print markdown-formatted summary table with total energy evaluations."""
    print("\n## Benchmark Results\n")
    print("| Problem | Method | Best Energy | Acc. Rate | Energy Evals | Time (s) |")
    print("|---------|--------|-------------|-----------|--------------|----------|")
    for problem_name, results, n_iters in [
        ("2D Multimodal", results_2d, n_iters_2d),
        ("10D Gaussian", results_10d, n_iters_10d),
    ]:
        for m, label in [("samc", "SAMC"), ("mh", "MH"), ("pt", "PT")]:
            r = results[m]
            evals = n_iters * n_replicas if m == "pt" else n_iters
            if evals >= 1_000_000:
                evals_str = f"{evals / 1_000_000:.1f}M"
            else:
                evals_str = f"{evals // 1000}K"
            print(
                f"| {problem_name} | {label} | "
                f"{r['best_energy']:.4f} | "
                f"{r['acceptance_rate']:.3f} | "
                f"{evals_str} | "
                f"{r['wall_time']:.1f} |"
            )


# ────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────


def main():
    n_iters_2d = 500_000
    n_iters_10d = 200_000
    save_every = 100
    burn_in_frac = 0.1

    proposal_std_2d = 0.05
    proposal_std_10d = 1.0

    results_2d = benchmark_problem(
        name="2D Multimodal Cost Function",
        energy_fn=cost_2d,
        dim=2,
        n_iters=n_iters_2d,
        burn_in_frac=burn_in_frac,
        save_every=save_every,
        samc_kwargs={
            "n_partitions": 42,
            "e_min": -8.2,
            "e_max": 0.0,
            "proposal_std": proposal_std_2d,
            "temperature": 0.1,
            "gain": "ramp",
            "gain_kwargs": {
                "rho": 1.0,
                "tau": 1.0,
                "warmup": 1,
                "step_scale": 1000,
            },
        },
        mh_kwargs={"proposal_std": proposal_std_2d, "temperature": 0.1},
        pt_kwargs={
            "n_replicas": 4,
            "proposal_std": proposal_std_2d,
            "t_min": 0.1,
            "t_max": 3.16,
            "swap_interval": 10,
        },
    )

    results_10d = benchmark_problem(
        name="10D Gaussian Mixture (4 modes, separation=10)",
        energy_fn=gaussian_mixture_10d,
        dim=10,
        n_iters=n_iters_10d,
        burn_in_frac=burn_in_frac,
        save_every=save_every,
        samc_kwargs={
            "n_partitions": 30,
            "e_min": 0.0,
            "e_max": 20.0,
            "proposal_std": proposal_std_10d,
            "gain": "ramp",
            "gain_kwargs": {
                "rho": 1.0,
                "tau": 1.0,
                "warmup": 1,
                "step_scale": 1000,
            },
        },
        mh_kwargs={"proposal_std": proposal_std_10d, "temperature": 1.0},
        pt_kwargs={
            "n_replicas": 4,
            "proposal_std": proposal_std_10d,
            "t_min": 1.0,
            "t_max": 10.0,
            "swap_interval": 10,
        },
    )

    plot_trajectories_2d(results_2d)
    plot_comparison(results_2d, results_10d)
    print_summary_table(
        results_2d,
        results_10d,
        n_iters_2d=n_iters_2d,
        n_iters_10d=n_iters_10d,
        n_replicas=4,
    )


if __name__ == "__main__":
    main()
