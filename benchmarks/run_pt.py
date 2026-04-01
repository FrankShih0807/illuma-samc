"""Run Parallel Tempering benchmark. Saves to benchmarks/results/pt.pt

Usage:
    python benchmarks/run_pt.py                # default: 8 replicas
    python benchmarks/run_pt.py --replicas 2   # custom replica count
"""

import argparse
import time
from pathlib import Path

import torch

from illuma_samc.baselines import run_parallel_tempering
from illuma_samc.problems import cost_2d, gaussian_mixture_10d

RESULTS_DIR = Path("benchmarks/results")


def run_pt_benchmark(name, energy_fn, dim, n_iters, pt_kwargs, burn_in_frac=0.1, save_every=100):
    burn_in = int(n_iters * burn_in_frac)
    n_replicas = pt_kwargs.get("n_replicas", 8)
    n_samples = (n_iters - burn_in) // save_every

    print(f"\n{'=' * 60}")
    print(f"  PT ({n_replicas} replicas) — {name}")
    print(f"{'=' * 60}")
    print(
        f"  {n_iters:,} iters, burn-in={burn_in:,}, save_every={save_every}, n_samples={n_samples:,}"
    )

    torch.manual_seed(42)
    t0 = time.perf_counter()
    result = run_parallel_tempering(
        energy_fn, dim, n_iters, burn_in=burn_in, save_every=save_every, **pt_kwargs
    )
    wall_time = time.perf_counter() - t0

    out = {
        "best_energy": result["best_energy"],
        "best_x": result["best_x"],
        "acceptance_rate": result["acceptance_rate"],
        "swap_rate": result["swap_rate"],
        "wall_time": wall_time,
        "energies": result["energies"],
        "samples": result["samples"],
        "n_replicas": n_replicas,
    }
    print(
        f"  best_E={result['best_energy']:.4f}  acc={result['acceptance_rate']:.3f}  "
        f"swap={result['swap_rate']:.3f}  n_samples={len(result['samples'])}  time={wall_time:.1f}s"
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicas", type=int, default=4)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    n_replicas = args.replicas

    results = {}

    results["2d"] = run_pt_benchmark(
        name="2D Multimodal",
        energy_fn=cost_2d,
        dim=2,
        n_iters=500_000,
        pt_kwargs={
            "n_replicas": n_replicas,
            "proposal_std": 0.05,
            "t_min": 0.1,
            "t_max": 3.16,
            "swap_interval": 10,
        },
    )

    results["10d"] = run_pt_benchmark(
        name="10D Gaussian Mixture",
        energy_fn=gaussian_mixture_10d,
        dim=10,
        n_iters=200_000,
        pt_kwargs={
            "n_replicas": n_replicas,
            "proposal_std": 1.0,
            "t_min": 1.0,
            "t_max": 10.0,
            "swap_interval": 10,
        },
    )

    out_path = RESULTS_DIR / f"pt_{n_replicas}rep.pt"
    torch.save(results, out_path)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
