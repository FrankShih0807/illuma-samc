"""Generate plots from saved benchmark results.

Reads benchmarks/results/{samc,mh,pt_*rep}.pt (produced by run_samc/mh/pt.py).
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from illuma_samc.problems import cost_2d

matplotlib.use("Agg")

RESULTS_DIR = Path("benchmarks/results")


def load_results(pt_file="pt_4rep.pt"):
    """Load per-method result files (samc.pt, mh.pt, pt_*rep.pt)."""
    samc_path = RESULTS_DIR / "samc.pt"
    mh_path = RESULTS_DIR / "mh.pt"
    pt_path = RESULTS_DIR / pt_file

    missing = [p for p in [samc_path, mh_path, pt_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing result files: {missing}. Run run_samc.py, run_mh.py, run_pt.py first."
        )

    samc = torch.load(samc_path, weights_only=False)
    mh = torch.load(mh_path, weights_only=False)
    pt = torch.load(pt_path, weights_only=False)

    n_replicas = pt["2d"].get("n_replicas", 4)

    return {
        "results_2d": {"samc": samc["2d"], "mh": mh["2d"], "pt": pt["2d"]},
        "results_10d": {"samc": samc["10d"], "mh": mh["10d"], "pt": pt["10d"]},
        "config": {
            "n_iters_2d": 500_000,
            "n_iters_10d": 200_000,
            "save_every": 100,
            "burn_in_frac": 0.1,
            "n_replicas": n_replicas,
        },
    }


def plot_hero(results_2d: dict):
    """2-row hero image: trajectories + energy traces (README)."""
    grid_n = 200
    xx = torch.linspace(-1.1, 1.1, grid_n)
    yy = torch.linspace(-1.1, 1.1, grid_n)
    X, Y = torch.meshgrid(xx, yy, indexing="xy")
    coords = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    Z, _ = cost_2d(coords)
    Z = Z.reshape(grid_n, grid_n).numpy()
    xx_np, yy_np = xx.numpy(), yy.numpy()

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
        # Row 1: Trajectories
        ax = axes[0, col]
        ax.contourf(xx_np, yy_np, Z, levels=40, cmap="viridis", alpha=0.6)

        samples = results_2d[method]["samples"]
        if isinstance(samples, torch.Tensor) and samples.dim() == 3:
            samples = samples[0]
        if isinstance(samples, torch.Tensor):
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
        # Annotation badge in upper right (replacing Best E legend)
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

        # Row 2: Energy traces
        ax2 = axes[1, col]
        energies = results_2d[method]["energies"]
        if isinstance(energies, torch.Tensor):
            energies = energies.numpy()
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
    print("Saved: benchmarks/trajectory_comparison.png, assets/samc_vs_others.png")


def plot_comparison(results_2d: dict, results_10d: dict):
    """Bar charts: best energy, acceptance rate, wall time."""
    methods = ["samc", "mh", "pt"]
    labels = ["SAMC", "MH", "PT"]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("SAMC vs MH vs Parallel Tempering", fontsize=14, fontweight="bold")

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
    print("Saved: benchmarks/benchmark_comparison.png")


def plot_energy_traces(results_2d: dict, results_10d: dict):
    """Overlaid energy traces for all methods."""
    methods = ["samc", "mh", "pt"]
    labels = ["SAMC", "MH", "PT"]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Energy Trace Comparison", fontsize=14, fontweight="bold")

    for col, (title, results) in enumerate(
        [("2D Multimodal", results_2d), ("10D Gaussian Mixture", results_10d)]
    ):
        ax = axes[col]
        for m, label, color in zip(methods, labels, colors):
            e = results[m]["energies"]
            if isinstance(e, torch.Tensor):
                e = e.numpy()
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
    print("Saved: benchmarks/energy_traces.png")


def print_summary_table(data: dict):
    """Print markdown-formatted summary table."""
    config = data["config"]
    n_replicas = config["n_replicas"]

    print("\n## Benchmark Results\n")
    print("| Problem | Method | Best Energy | Acc. Rate | Energy Evals | Time (s) |")
    print("|---------|--------|-------------|-----------|--------------|----------|")
    for problem_name, results_key, n_iters_key in [
        ("2D Multimodal", "results_2d", "n_iters_2d"),
        ("10D Gaussian", "results_10d", "n_iters_10d"),
    ]:
        results = data[results_key]
        n_iters = config[n_iters_key]
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


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", default="pt_4rep.pt", help="PT results file in benchmarks/results/")
    args = parser.parse_args()

    data = load_results(pt_file=args.pt)
    results_2d = data["results_2d"]
    results_10d = data["results_10d"]

    plot_hero(results_2d)
    plot_comparison(results_2d, results_10d)
    plot_energy_traces(results_2d, results_10d)
    print_summary_table(data)


if __name__ == "__main__":
    main()
