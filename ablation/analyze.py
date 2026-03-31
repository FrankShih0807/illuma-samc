"""Analyze ablation results from a group directory.

Usage:
    python ablation/analyze.py outputs/ablation/samc_gain_schedule
    python ablation/analyze.py outputs/ablation/samc_gain_schedule --csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_group_results(group_dir: str | Path) -> list[dict]:
    """Load all results.json from a group directory tree."""
    group_path = Path(group_dir)
    results = []
    for rfile in sorted(group_path.rglob("results.json")):
        with open(rfile) as f:
            data = json.load(f)
        # Add path info
        rel = rfile.relative_to(group_path)
        data["_path"] = str(rfile)
        data["_rel_path"] = str(rel)
        # Parse model and params from directory structure
        parts = rel.parts
        if len(parts) >= 3:
            data["_model"] = parts[0]
            data["_params"] = parts[1]
            data["_seed_dir"] = parts[2] if len(parts) > 2 else ""

        # Load config if available
        config_path = rfile.parent / "config.yaml"
        if config_path.exists():
            import yaml

            with open(config_path) as f:
                data["_config"] = yaml.safe_load(f)

        results.append(data)
    return results


def compute_summary(results: list[dict]) -> dict[str, list[dict]]:
    """Group results by param setting and compute stats across seeds."""
    groups: dict[str, list[dict]] = {}
    for r in results:
        key = r.get("_params", "unknown")
        model = r.get("_model", "unknown")
        group_key = f"{model}/{key}"
        groups.setdefault(group_key, []).append(r)
    return groups


def print_summary(groups: dict[str, list[dict]]):
    """Print a summary table of results grouped by parameter setting."""
    print(
        f"{'Setting':<50} {'N':>3} {'Best E (mean)':>14} {'Best E (std)':>13} "
        f"{'Acc Rate':>10} {'Flatness':>10}"
    )
    print("-" * 104)

    for key, runs in sorted(groups.items()):
        n = len(runs)
        best_energies = [r["best_energy"] for r in runs if r["best_energy"] != float("inf")]
        acc_rates = [r["acceptance_rate"] for r in runs]
        flatness = [r.get("bin_flatness", float("nan")) for r in runs]

        import statistics

        if not best_energies:
            mean_e = float("inf")
            std_e = 0.0
        else:
            mean_e = statistics.mean(best_energies)
            std_e = statistics.stdev(best_energies) if len(best_energies) > 1 else 0.0
        mean_acc = statistics.mean(acc_rates)
        valid_flat = [f for f in flatness if f == f]  # skip NaN
        mean_flat = statistics.mean(valid_flat) if valid_flat else float("nan")

        print(
            f"{key:<50} {n:>3} {mean_e:>14.4f} {std_e:>13.4f} {mean_acc:>10.3f} {mean_flat:>10.3f}"
        )


def export_csv(results: list[dict], output_path: str):
    """Export results to CSV."""
    if not results:
        print("No results to export.")
        return

    # Collect all keys
    keys = set()
    for r in results:
        keys.update(k for k in r if not k.startswith("_config"))
    keys = sorted(keys)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in keys})

    print(f"Exported {len(results)} results to {output_path}")


def plot_group(groups: dict[str, list[dict]], group_dir: str | Path):
    """Generate per-group comparison plots."""
    if not HAS_MPL:
        print("matplotlib not available, skipping plots.")
        return

    group_path = Path(group_dir)
    fig_dir = group_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Bar chart: best energy by param setting
    import statistics

    settings = sorted(groups.keys())
    means = []
    stds = []
    for key in settings:
        runs = groups[key]
        energies = [r["best_energy"] for r in runs if r["best_energy"] != float("inf")]
        if not energies:
            means.append(0.0)
            stds.append(0.0)
        else:
            means.append(statistics.mean(energies))
            stds.append(statistics.stdev(energies) if len(energies) > 1 else 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Best energy
    ax = axes[0]
    x = range(len(settings))
    ax.barh(x, means, xerr=stds, capsize=3, color="steelblue", alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels([s.split("/")[-1] for s in settings], fontsize=8)
    ax.set_xlabel("Best Energy (mean +/- std)")
    ax.set_title("Best Energy by Setting")
    ax.invert_yaxis()

    # Acceptance rate
    ax = axes[1]
    acc_means = []
    for key in settings:
        runs = groups[key]
        acc_means.append(statistics.mean([r["acceptance_rate"] for r in runs]))
    ax.barh(x, acc_means, color="coral", alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels([s.split("/")[-1] for s in settings], fontsize=8)
    ax.set_xlabel("Acceptance Rate")
    ax.set_title("Acceptance Rate by Setting")
    ax.invert_yaxis()

    plt.tight_layout()
    fig_path = fig_dir / "comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze ablation results.")
    parser.add_argument("group_dir", type=str, help="Path to group results directory")
    parser.add_argument("--csv", type=str, default=None, help="Export results to CSV")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    results = load_group_results(args.group_dir)
    if not results:
        print(f"No results.json found in {args.group_dir}")
        return

    print(f"Loaded {len(results)} results from {args.group_dir}\n")

    groups = compute_summary(results)
    print_summary(groups)

    if args.csv:
        export_csv(results, args.csv)

    if not args.no_plot:
        plot_group(groups, args.group_dir)


if __name__ == "__main__":
    main()
