"""Run SAMC benchmark. Saves to benchmarks/results/samc_*.pt"""

import time
from pathlib import Path

import torch

from illuma_samc import SAMC
from illuma_samc.problems import cost_2d, gaussian_mixture_10d

RESULTS_DIR = Path("benchmarks/results")


def run_samc_benchmark(
    name, energy_fn, dim, n_iters, samc_kwargs, burn_in_frac=0.1, save_every=100
):
    burn_in = int(n_iters * burn_in_frac)
    n_samples = (n_iters - burn_in) // save_every

    print(f"\n{'=' * 60}")
    print(f"  SAMC — {name}")
    print(f"{'=' * 60}")
    print(
        f"  {n_iters:,} iters, burn-in={burn_in:,}, save_every={save_every}, n_samples={n_samples:,}"
    )

    torch.manual_seed(42)
    t0 = time.perf_counter()
    sampler = SAMC(energy_fn=energy_fn, dim=dim, **samc_kwargs)
    result = sampler.run(n_steps=n_iters, save_every=save_every, progress=True)
    wall_time = time.perf_counter() - t0

    burn_in_snapshots = burn_in // save_every
    samples = result.samples[burn_in_snapshots:]

    out = {
        "best_energy": result.best_energy,
        "best_x": result.best_x,
        "acceptance_rate": result.acceptance_rate,
        "wall_time": wall_time,
        "energies": result.energy_history.flatten(),
        "samples": samples,
    }
    print(
        f"  best_E={result.best_energy:.4f}  acc={result.acceptance_rate:.3f}  "
        f"n_samples={len(samples)}  time={wall_time:.1f}s"
    )
    return out


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    results["2d"] = run_samc_benchmark(
        name="2D Multimodal",
        energy_fn=cost_2d,
        dim=2,
        n_iters=500_000,
        samc_kwargs={
            "n_partitions": 42,
            "e_min": -8.2,
            "e_max": 0.0,
            "proposal_std": 0.05,
            "temperature": 0.1,
            "gain": "ramp",
            "gain_kwargs": {"rho": 1.0, "tau": 1.0, "warmup": 1, "step_scale": 1000},
        },
    )

    results["10d"] = run_samc_benchmark(
        name="10D Gaussian Mixture",
        energy_fn=gaussian_mixture_10d,
        dim=10,
        n_iters=200_000,
        samc_kwargs={
            "n_partitions": 30,
            "e_min": 0.0,
            "e_max": 20.0,
            "proposal_std": 1.0,
            "gain": "ramp",
            "gain_kwargs": {"rho": 1.0, "tau": 1.0, "warmup": 1, "step_scale": 1000},
        },
    )

    torch.save(results, RESULTS_DIR / "samc.pt")
    print(f"\nSaved to {RESULTS_DIR / 'samc.pt'}")


if __name__ == "__main__":
    main()
