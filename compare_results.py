"""Compare experiment results across runs for a given model.

Usage:
    python compare_results.py --model 2d
    python compare_results.py --model 10d --sort wall_time
    python compare_results.py --model 2d --algo samc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_all_results(model: str, algo_filter: str | None = None) -> list[dict]:
    """Load all results.json files for a given model."""
    base = Path("outputs") / model
    if not base.exists():
        return []

    results = []
    for algo_dir in sorted(base.iterdir()):
        if not algo_dir.is_dir():
            continue
        algo = algo_dir.name
        if algo_filter and algo != algo_filter:
            continue
        for run_dir in sorted(algo_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            results_file = run_dir / "results.json"
            if not results_file.exists():
                continue
            with open(results_file) as f:
                data = json.load(f)
            data["algo"] = algo
            data["run_id"] = run_dir.name
            data["path"] = str(run_dir)
            results.append(data)

    return results


def print_comparison_table(results: list[dict], sort_by: str = "best_energy"):
    """Print a ranked comparison table."""
    if not results:
        print("No results found.")
        return

    # Sort (best_energy ascending for minimization)
    reverse = sort_by not in ("best_energy", "wall_time", "total_energy_evals")
    results.sort(key=lambda r: r.get(sort_by, float("inf")), reverse=reverse)

    # Header
    print(
        f"\n{'Rank':<5} {'Algo':<6} {'Best Energy':>12} {'Acc. Rate':>10} "
        f"{'Energy Evals':>13} {'Time (s)':>9} {'Run ID'}"
    )
    print("-" * 85)

    for i, r in enumerate(results, 1):
        algo = r.get("algo", "?")
        best_e = r.get("best_energy", float("nan"))
        acc = r.get("acceptance_rate", float("nan"))
        evals = r.get("total_energy_evals", 0)
        t = r.get("wall_time", float("nan"))
        run_id = r.get("run_id", "?")

        # Format energy evals
        if evals >= 1_000_000:
            evals_str = f"{evals / 1_000_000:.1f}M"
        elif evals >= 1_000:
            evals_str = f"{evals // 1000}K"
        else:
            evals_str = str(evals)

        extra = ""
        if "swap_rate" in r:
            extra = f"  (swap={r['swap_rate']:.3f})"

        print(
            f"{i:<5} {algo.upper():<6} {best_e:>12.4f} {acc:>10.3f} "
            f"{evals_str:>13} {t:>9.1f} {run_id}{extra}"
        )

    print(f"\nTotal runs: {len(results)}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare experiment results across runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model to compare (e.g., 2d, 10d)",
    )
    parser.add_argument(
        "--algo",
        default=None,
        help="Filter by algorithm (samc, mh, pt)",
    )
    parser.add_argument(
        "--sort",
        default="best_energy",
        choices=["best_energy", "acceptance_rate", "wall_time", "total_energy_evals"],
        help="Column to sort by",
    )
    args = parser.parse_args()

    print(f"Results for model: {args.model}")
    if args.algo:
        print(f"Filtered by algo: {args.algo}")

    results = load_all_results(args.model, algo_filter=args.algo)
    print_comparison_table(results, sort_by=args.sort)


if __name__ == "__main__":
    main()
