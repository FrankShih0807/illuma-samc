"""3-way fair comparison: MH vs PT vs SAMC (default zero-config).

All algorithms share identical compute budgets, starting points, and
adaptive proposals (GaussianProposal with adapt=True).

Fairness rules:
  - 3 algorithms: MH, PT, SAMC
  - 4 chains/replicas each = 800K total energy evals per algo (200K iters x 4)
  - Same random starting points via torch.manual_seed before each algo
  - All use GaussianProposal(step_size=1.0, adapt=True)
  - SAMC zero-config only (no e_min/e_max), defaults to independent weights

Usage:
    conda run -n illuma-samc python benchmarks/three_way.py
    conda run -n illuma-samc python benchmarks/three_way.py --smoke  # quick test
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from illuma_samc.problems import PROBLEMS
from illuma_samc.proposals import GaussianProposal
from illuma_samc.sampler import SAMC

OUTPUT_DIR = Path("benchmarks/outputs/three_way")

N_CHAINS = 4
T_MIN = 1.0
T_MAX = 10.0
SWAP_INTERVAL = 10


# ---------------------------------------------------------------------------
# MH with 4 independent adaptive chains
# ---------------------------------------------------------------------------


def run_mh_adaptive(
    problem_key: str,
    seed: int,
    n_iters: int,
) -> dict:
    """Run 4 independent MH chains with adaptive proposals."""
    prob = PROBLEMS[problem_key]
    energy_fn = prob["energy_fn"]
    dim = prob["dim"]

    torch.manual_seed(seed)
    x0 = torch.randn(N_CHAINS, dim)

    proposals = [
        GaussianProposal(step_size=1.0, adapt=True, adapt_warmup=2000) for _ in range(N_CHAINS)
    ]

    # Initial states and energies
    states = [x0[i].clone() for i in range(N_CHAINS)]
    energies = [energy_fn(states[i]).item() for i in range(N_CHAINS)]

    best_energy = min(energies)
    accept_counts = [0] * N_CHAINS

    t0 = time.time()
    for step in range(n_iters):
        for c in range(N_CHAINS):
            x = states[c]
            fx = energies[c]
            proposal = proposals[c]

            x_new = proposal.propose(x)
            fy = energy_fn(x_new).item()

            # Boltzmann acceptance at T=1.0
            log_r = fx - fy  # log exp(-(fy - fx)/T) with T=1
            if log_r > 0:
                accepted = True
            else:
                accepted = torch.rand(1).item() < math.exp(log_r)

            proposal.report_accept(accepted)

            if accepted:
                states[c] = x_new
                energies[c] = fy
                accept_counts[c] += 1

            if energies[c] < best_energy:
                best_energy = energies[c]

    wall = time.time() - t0
    avg_accept = sum(accept_counts) / (N_CHAINS * n_iters)

    return {
        "best_energy": best_energy,
        "acceptance_rate": avg_accept,
        "wall_time": wall,
        "energy_evals": n_iters * N_CHAINS,
    }


# ---------------------------------------------------------------------------
# PT with 4 replicas and adaptive proposals
# ---------------------------------------------------------------------------


def run_pt_adaptive(
    problem_key: str,
    seed: int,
    n_iters: int,
) -> dict:
    """Run parallel tempering with 4 replicas and adaptive proposals."""
    prob = PROBLEMS[problem_key]
    energy_fn = prob["energy_fn"]
    dim = prob["dim"]

    torch.manual_seed(seed)
    x0 = torch.randn(N_CHAINS, dim)

    # Geometric temperature ladder from t_min to t_max
    temps = [T_MIN * (T_MAX / T_MIN) ** (i / (N_CHAINS - 1)) for i in range(N_CHAINS)]

    proposals = [
        GaussianProposal(step_size=1.0, adapt=True, adapt_warmup=2000) for _ in range(N_CHAINS)
    ]

    states = [x0[i].clone() for i in range(N_CHAINS)]
    energies = [energy_fn(states[i]).item() for i in range(N_CHAINS)]

    best_energy = min(energies)
    accept_counts = [0] * N_CHAINS  # only track cold chain (index 0)
    total_cold_steps = 0

    t0 = time.time()
    for step in range(n_iters):
        # MH step per replica
        for r in range(N_CHAINS):
            x = states[r]
            fx = energies[r]
            T = temps[r]
            proposal = proposals[r]

            x_new = proposal.propose(x)
            fy = energy_fn(x_new).item()

            log_r = (fx - fy) / T
            if log_r > 0:
                accepted = True
            else:
                accepted = torch.rand(1).item() < math.exp(log_r)

            proposal.report_accept(accepted)

            if accepted:
                states[r] = x_new
                energies[r] = fy
                if r == 0:
                    accept_counts[0] += 1

            if energies[r] < best_energy:
                best_energy = energies[r]

            if r == 0:
                total_cold_steps += 1

        # Swap adjacent replicas every SWAP_INTERVAL steps
        if (step + 1) % SWAP_INTERVAL == 0:
            for r in range(N_CHAINS - 1):
                # Swap r and r+1
                dE = energies[r] - energies[r + 1]
                dBeta = 1.0 / temps[r] - 1.0 / temps[r + 1]
                log_swap = dE * dBeta
                if log_swap > 0 or torch.rand(1).item() < math.exp(log_swap):
                    states[r], states[r + 1] = states[r + 1], states[r]
                    energies[r], energies[r + 1] = energies[r + 1], energies[r]

    wall = time.time() - t0
    cold_accept = accept_counts[0] / total_cold_steps if total_cold_steps > 0 else 0.0

    return {
        "best_energy": best_energy,
        "acceptance_rate": cold_accept,
        "wall_time": wall,
        "energy_evals": n_iters * N_CHAINS,
    }


# ---------------------------------------------------------------------------
# SAMC zero-config
# ---------------------------------------------------------------------------


def run_samc_default(
    problem_key: str,
    seed: int,
    n_iters: int,
) -> dict:
    """Run SAMC with zero-config defaults (independent weights, bin_width=0.5)."""
    prob = PROBLEMS[problem_key]

    torch.manual_seed(seed)
    x0 = torch.randn(N_CHAINS, prob["dim"])

    sampler = SAMC(
        energy_fn=prob["energy_fn"],
        dim=prob["dim"],
        n_chains=N_CHAINS,
        adapt_proposal=True,
        adapt_warmup=2000,
    )

    t0 = time.time()
    result = sampler.run(n_steps=n_iters, x0=x0, progress=False)
    wall = time.time() - t0

    return {
        "best_energy": result.best_energy,
        "acceptance_rate": result.acceptance_rate,
        "wall_time": wall,
        "energy_evals": n_iters * N_CHAINS,
    }


# ---------------------------------------------------------------------------
# Algorithm registry
# ---------------------------------------------------------------------------

ALGOS = {
    "MH": run_mh_adaptive,
    "PT": run_pt_adaptive,
    "SAMC": run_samc_default,
}


def mean_std(vals: list[float]) -> tuple[float, float]:
    n = len(vals)
    m = sum(vals) / n
    s = (sum((v - m) ** 2 for v in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return m, s


def result_path(problem: str, algo: str, seed: int) -> Path:
    return OUTPUT_DIR / f"{problem}_{algo}_seed{seed}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="3-way fair comparison benchmark")
    parser.add_argument(
        "--smoke", action="store_true", help="Quick test: 10d only, 1 seed, 10K iters"
    )
    parser.add_argument("--problems", nargs="+", default=["10d", "50d", "100d", "rastrigin"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 999])
    parser.add_argument("--iters", type=int, default=200_000)
    parser.add_argument("--no-cache", action="store_true", help="Re-run all, ignore cached results")
    args = parser.parse_args()

    if args.smoke:
        problems = ["10d"]
        seeds = [42]
        n_iters = 10_000
    else:
        problems = args.problems
        seeds = args.seeds
        n_iters = args.iters

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = len(problems) * len(ALGOS) * len(seeds)
    n_algos = len(ALGOS)
    n_probs = len(problems)
    n_seeds = len(seeds)
    print(f"Running {total} experiments: {n_probs} problems x {n_algos} algos x {n_seeds} seeds")
    total_evals = n_iters * N_CHAINS
    print(f"Iters per run: {n_iters:,} | Chains: {N_CHAINS} | Total evals: {total_evals:,}\n")

    all_results: dict[tuple[str, str], list[dict]] = {}

    for problem in problems:
        for algo_name, algo_fn in ALGOS.items():
            key = (problem, algo_name)
            all_results[key] = []

            for seed in seeds:
                out = result_path(problem, algo_name, seed)
                if out.exists() and not args.no_cache:
                    print(f"  [cached] {problem}/{algo_name}/seed={seed}", flush=True)
                    with open(out) as f:
                        metrics = json.load(f)
                else:
                    print(f"  [running] {problem}/{algo_name}/seed={seed} ...", end=" ", flush=True)
                    metrics = algo_fn(problem, seed, n_iters)
                    metrics["problem"] = problem
                    metrics["algo"] = algo_name
                    metrics["seed"] = seed
                    metrics["n_iters"] = n_iters
                    with open(out, "w") as f:
                        json.dump(metrics, f, indent=2)
                    print(
                        f"best_e={metrics['best_energy']:.4f} "
                        f"acc={metrics['acceptance_rate']:.3f} "
                        f"t={metrics['wall_time']:.1f}s",
                        flush=True,
                    )
                all_results[key].append(metrics)
        print()

    # Summary table
    print("=" * 90)
    print("3-WAY FAIR COMPARISON TABLE")
    print(f"  {N_CHAINS} chains/replicas | {n_iters:,} iters | same starts | adaptive proposals")
    print("=" * 90)
    header = (
        f"{'Problem':>12} {'Algorithm':>10} {'Best Energy':>20} {'Acc Rate':>10} {'Wall (s)':>10}"
    )
    print(header)
    print("-" * len(header))

    for problem in problems:
        for algo_name in ALGOS:
            key = (problem, algo_name)
            runs = all_results[key]
            be_m, be_s = mean_std([r["best_energy"] for r in runs])
            ar_m, _ = mean_std([r["acceptance_rate"] for r in runs])
            wt_m, _ = mean_std([r["wall_time"] for r in runs])
            energy_str = f"{be_m:>9.2f} +/- {be_s:<7.2f}"
            print(f"{problem:>12} {algo_name:>10} {energy_str} {ar_m:>10.3f} {wt_m:>10.1f}")
        print()

    # Markdown table for README
    print("\n## README Markdown Table\n")
    algo_keys = list(ALGOS.keys())
    print("| Problem | Dim | " + " | ".join(algo_keys) + " |")
    print("|---------|-----|" + "|".join("---" for _ in algo_keys) + "|")
    for problem in problems:
        dim = PROBLEMS[problem]["dim"]
        name = PROBLEMS[problem]["name"]

        stats = {}
        for algo in algo_keys:
            runs = all_results[(problem, algo)]
            be_m, be_s = mean_std([r["best_energy"] for r in runs])
            stats[algo] = (be_m, be_s)

        best_algo = min(stats, key=lambda a: stats[a][0])

        row_parts = [f"| {name}", f"{dim}"]
        for algo in algo_keys:
            be_m, be_s = stats[algo]
            cell = f"{be_m:.2f} +/- {be_s:.2f}"
            if algo == best_algo:
                cell = f"**{cell}**"
            row_parts.append(cell)
        print(" | ".join(row_parts) + " |")


if __name__ == "__main__":
    main()
