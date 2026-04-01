"""Run MH benchmark. Saves to benchmarks/results/mh.pt"""

import time
from pathlib import Path

import torch

from illuma_samc.baselines import run_mh
from illuma_samc.problems import cost_2d, gaussian_mixture_10d

RESULTS_DIR = Path("benchmarks/results")


def run_mh_benchmark(name, energy_fn, dim, n_iters, mh_kwargs, burn_in_frac=0.1, save_every=100):
    burn_in = int(n_iters * burn_in_frac)
    n_samples = (n_iters - burn_in) // save_every

    print(f"\n{'=' * 60}")
    print(f"  MH — {name}")
    print(f"{'=' * 60}")
    print(
        f"  {n_iters:,} iters, burn-in={burn_in:,}, save_every={save_every}, n_samples={n_samples:,}"
    )

    torch.manual_seed(42)
    t0 = time.perf_counter()
    result = run_mh(energy_fn, dim, n_iters, burn_in=burn_in, save_every=save_every, **mh_kwargs)
    wall_time = time.perf_counter() - t0

    out = {
        "best_energy": result["best_energy"],
        "best_x": result["best_x"],
        "acceptance_rate": result["acceptance_rate"],
        "wall_time": wall_time,
        "energies": result["energies"],
        "samples": result["samples"],
    }
    print(
        f"  best_E={result['best_energy']:.4f}  acc={result['acceptance_rate']:.3f}  "
        f"n_samples={len(result['samples'])}  time={wall_time:.1f}s"
    )
    return out


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    results["2d"] = run_mh_benchmark(
        name="2D Multimodal",
        energy_fn=cost_2d,
        dim=2,
        n_iters=500_000,
        mh_kwargs={"proposal_std": 0.05, "temperature": 0.1},
    )

    results["10d"] = run_mh_benchmark(
        name="10D Gaussian Mixture",
        energy_fn=gaussian_mixture_10d,
        dim=10,
        n_iters=200_000,
        mh_kwargs={"proposal_std": 1.0, "temperature": 1.0},
    )

    torch.save(results, RESULTS_DIR / "mh.pt")
    print(f"\nSaved to {RESULTS_DIR / 'mh.pt'}")


if __name__ == "__main__":
    main()
