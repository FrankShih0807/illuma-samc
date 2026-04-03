"""Step 43: Robust Defaults vs Hand-Tuned Ablation Winner benchmark.

Runs two configs per problem:
  Config A — "Robust Defaults": adapt_proposal=True, no e_min/e_max/proposal_std
  Config B — "Hand-Tuned": best config from ablation studies (Steps 24-28)

Usage:
    # Smoke test (2d, 1 seed):
    conda run -n illuma-samc python ablation/robust_defaults.py --smoke

    # Full run (all problems, all seeds):
    conda run -n illuma-samc python ablation/robust_defaults.py

    # Specific problems:
    conda run -n illuma-samc python ablation/robust_defaults.py --problems 2d rosenbrock
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from illuma_samc import SAMC
from illuma_samc.problems import PROBLEMS

# ── Output ───────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("ablation/outputs/robust_defaults")

# ── Run parameters ───────────────────────────────────────────────────────────

SEEDS = [42, 123, 456, 789, 999]

# Iterations per problem
ITERS = {
    "2d": 500_000,
    "rosenbrock": 500_000,
    "10d": 200_000,
    "50d": 200_000,
    "100d": 200_000,
    "rastrigin": 200_000,
}

# ── Hand-tuned configs (ablation winners from Steps 24-28) ───────────────────

HAND_TUNED = {
    "2d": dict(
        proposal_std=0.1,
        gain="ramp",
        gain_kwargs={"rho": 1.0, "tau": 1.0, "warmup": 1, "step_scale": 1000},
        n_partitions=42,
        e_min=-8.2,
        e_max=0.0,
        n_chains=4,
    ),
    "rosenbrock": dict(
        proposal_std=0.1,
        gain="ramp",
        gain_kwargs={"rho": 1.0, "tau": 1.0, "warmup": 1, "step_scale": 1000},
        n_partitions=30,
        e_min=0.0,
        e_max=500.0,
        n_chains=8,
    ),
    "10d": dict(
        proposal_std=1.0,
        gain="ramp",
        gain_kwargs={"rho": 1.0, "tau": 1.0, "warmup": 1, "step_scale": 1000},
        n_partitions=30,
        e_min=0.0,
        e_max=20.0,
        n_chains=4,
    ),
    "rastrigin": dict(
        proposal_std=0.5,
        gain="ramp",
        gain_kwargs={"rho": 1.0, "tau": 1.0, "warmup": 1, "step_scale": 1000},
        n_partitions=40,
        e_min=0.0,
        e_max=500.0,
        n_chains=16,
    ),
    "50d": dict(
        proposal_std=0.5,
        gain="ramp",
        gain_kwargs={"rho": 1.0, "tau": 1.0, "warmup": 1, "step_scale": 1000},
        n_partitions=40,
        e_min=0.0,
        e_max=60.0,
        n_chains=4,
    ),
    "100d": dict(
        proposal_std=0.3,
        gain="ramp",
        gain_kwargs={"rho": 1.0, "tau": 1.0, "warmup": 1, "step_scale": 1000},
        n_partitions=50,
        e_min=0.0,
        e_max=60.0,
        n_chains=4,
    ),
}


# ── Flatness helper ───────────────────────────────────────────────────────────


def compute_flatness(bin_counts: torch.Tensor) -> float:
    """Flatness = 1 - std(counts) / mean(counts) for visited bins."""
    visited = bin_counts[bin_counts > 0]
    if len(visited) < 2:
        return 0.0
    mean = visited.float().mean().item()
    std = visited.float().std().item()
    if mean == 0:
        return 0.0
    return float(1.0 - std / mean)


# ── Single run ────────────────────────────────────────────────────────────────


def run_one(problem: str, config_name: str, seed: int) -> dict:
    """Run one (problem, config, seed) and return metrics dict."""
    prob = PROBLEMS[problem]
    energy_fn = prob["energy_fn"]
    dim = prob["dim"]
    n_iters = ITERS[problem]

    t0 = time.perf_counter()

    if config_name == "robust":
        sampler = SAMC(
            energy_fn=energy_fn,
            dim=dim,
            n_chains=4,
            adapt_proposal=True,
            adapt_warmup=2000,
            # no e_min, e_max, proposal_std, gain — all defaults
        )
    else:
        ht = HAND_TUNED[problem]
        sampler = SAMC(
            energy_fn=energy_fn,
            dim=dim,
            n_chains=ht["n_chains"],
            proposal_std=ht["proposal_std"],
            gain=ht["gain"],
            gain_kwargs=ht.get("gain_kwargs"),
            n_partitions=ht["n_partitions"],
            e_min=ht["e_min"],
            e_max=ht["e_max"],
        )

    result = sampler.run(n_steps=n_iters, seed=seed, progress=False)
    wall_time = time.perf_counter() - t0

    flatness = compute_flatness(result.bin_counts)

    # Final proposal std (for adaptive config)
    final_proposal_std = None
    if config_name == "robust" and hasattr(sampler._proposal, "step_size"):
        final_proposal_std = sampler._proposal.step_size

    metrics = {
        "problem": problem,
        "config": config_name,
        "seed": seed,
        "best_energy": result.best_energy,
        "acceptance_rate": result.acceptance_rate,
        "flatness": flatness,
        "final_proposal_std": final_proposal_std,
        "wall_time": wall_time,
        "n_iters": n_iters,
    }
    return metrics


# ── Save / load ───────────────────────────────────────────────────────────────


def result_path(problem: str, config_name: str, seed: int) -> Path:
    return OUTPUT_DIR / f"{problem}_{config_name}_seed{seed}.json"


def save_result(metrics: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    p = result_path(metrics["problem"], metrics["config"], metrics["seed"])
    with open(p, "w") as f:
        json.dump(metrics, f, indent=2)


# ── Print summary table ───────────────────────────────────────────────────────


def print_summary(results: list[dict]) -> None:
    from collections import defaultdict

    # Group by (problem, config)
    groups: dict[tuple, list] = defaultdict(list)
    for r in results:
        groups[(r["problem"], r["config"])].append(r)

    cols = f"{'Problem':<12} {'Config':<8} {'Seeds':>5}"
    cols += f" {'BestE mean':>12} {'BestE std':>10} {'AccRate':>8} {'Flatness':>9} {'WallTime':>9}"
    header = cols
    print(header, flush=True)
    print("-" * len(header), flush=True)

    problem_order = ["2d", "rosenbrock", "10d", "rastrigin", "50d", "100d"]
    config_order = ["robust", "tuned"]

    for prob in problem_order:
        for cfg in config_order:
            key = (prob, cfg)
            if key not in groups:
                continue
            grp = groups[key]
            energies = [r["best_energy"] for r in grp]
            acc_rates = [r["acceptance_rate"] for r in grp]
            flatnesses = [r["flatness"] for r in grp]
            wall_times = [r["wall_time"] for r in grp]

            import statistics

            e_mean = statistics.mean(energies)
            e_std = statistics.stdev(energies) if len(energies) > 1 else 0.0
            ar_mean = statistics.mean(acc_rates)
            fl_mean = statistics.mean(flatnesses)
            wt_mean = statistics.mean(wall_times)

            print(
                f"{prob:<12} {cfg:<8} {len(grp):>5} {e_mean:>12.4f} {e_std:>10.4f} "
                f"{ar_mean:>8.3f} {fl_mean:>9.3f} {wt_mean:>9.1f}s",
                flush=True,
            )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust defaults vs hand-tuned benchmark")
    parser.add_argument("--smoke", action="store_true", help="Smoke test: 2d, 1 seed only")
    parser.add_argument(
        "--problems",
        nargs="+",
        choices=list(ITERS.keys()),
        default=list(ITERS.keys()),
        help="Which problems to run",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="Seeds to use",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=["robust", "tuned"],
        default=["robust", "tuned"],
        help="Which configs to run",
    )
    args = parser.parse_args()

    if args.smoke:
        problems = ["2d"]
        seeds = [42]
        configs = ["robust", "tuned"]
    else:
        problems = args.problems
        seeds = args.seeds
        configs = args.configs

    total = len(problems) * len(configs) * len(seeds)
    print(
        f"Running {total} experiments: {len(problems)} problems"
        f" x {len(configs)} configs x {len(seeds)} seeds",
        flush=True,
    )

    all_results = []
    done = 0

    for problem in problems:
        for config_name in configs:
            for seed in seeds:
                out_path = result_path(problem, config_name, seed)
                if out_path.exists():
                    print(f"  [cached] {problem}/{config_name}/seed={seed}", flush=True)
                    with open(out_path) as f:
                        metrics = json.load(f)
                else:
                    print(
                        f"  [running] {problem}/{config_name}/seed={seed} ...", end=" ", flush=True
                    )
                    metrics = run_one(problem, config_name, seed)
                    save_result(metrics)
                    print(
                        f"best_energy={metrics['best_energy']:.4f} "
                        f"acc={metrics['acceptance_rate']:.3f} "
                        f"flat={metrics['flatness']:.3f} "
                        f"t={metrics['wall_time']:.1f}s",
                        flush=True,
                    )

                all_results.append(metrics)
                done += 1

    print(f"\nCompleted {done}/{total} runs.", flush=True)
    print("\n=== Summary Table ===", flush=True)
    print_summary(all_results)


if __name__ == "__main__":
    main()
