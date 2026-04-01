"""Run benchmarks: SAMC vs MH vs Parallel Tempering.

SAMC uses SAMCWeights (the main product) in a manual MH loop.
Saves results to benchmarks/results/ as .pt files.
Use plot_benchmark.py to generate plots from saved results.
"""

import math
import time
from pathlib import Path

import torch

from illuma_samc import GainSequence, SAMCWeights, UniformPartition
from illuma_samc.baselines import run_mh, run_parallel_tempering
from illuma_samc.problems import cost_2d, gaussian_mixture_10d

RESULTS_DIR = Path("benchmarks/results")


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
    """Run all three methods and return results dict."""
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
    fx = float(raw[0]) if isinstance(raw, tuple) else float(raw)

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


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

    # Save results
    torch.save(
        {
            "results_2d": results_2d,
            "results_10d": results_10d,
            "config": {
                "n_iters_2d": n_iters_2d,
                "n_iters_10d": n_iters_10d,
                "save_every": save_every,
                "burn_in_frac": burn_in_frac,
                "n_replicas": 4,
            },
        },
        RESULTS_DIR / "benchmark_results.pt",
    )
    print(f"\nResults saved to {RESULTS_DIR / 'benchmark_results.pt'}")


if __name__ == "__main__":
    main()
