"""Run SAMC benchmark using SAMCWeights (the main product).

Saves to benchmarks/results/samc.pt

Usage:
    python benchmarks/run_samc.py              # default: 1 chain
    python benchmarks/run_samc.py --n_chains 4 # multi-chain
"""

import argparse
import math
import time
from pathlib import Path

import torch

from illuma_samc import GainSequence, SAMCWeights, UniformPartition
from illuma_samc.problems import cost_2d, gaussian_mixture_10d

RESULTS_DIR = Path("benchmarks/results")


def _eval_energy(energy_fn, x):
    """Evaluate energy, handling (energy, in_region) tuple returns."""
    result = energy_fn(x)
    if isinstance(result, tuple):
        e, in_r = result
        e = float(e)
        if isinstance(in_r, torch.Tensor):
            in_r = bool(in_r.item())
        return e, in_r
    return float(result), True


def _run_single_samc_chain(
    energy_fn,
    dim,
    n_iters,
    wm,
    proposal_std,
    temperature,
    burn_in,
    save_every,
):
    """Run one MH+SAMCWeights chain. Returns dict of results."""
    x = torch.zeros(dim)
    fx, _ = _eval_energy(energy_fn, x)

    best_x, best_e = x.clone(), fx
    accept_count = 0
    energies = []
    samples = []

    for t in range(1, n_iters + 1):
        x_new = x + proposal_std * torch.randn(dim)
        fy, in_r = _eval_energy(energy_fn, x_new)

        log_r = (-fy + fx) / temperature + wm.correction(fx, fy)

        if in_r and (log_r > 0 or math.log(torch.rand(1).item() + 1e-300) < log_r):
            x, fx = x_new.clone(), fy
            accept_count += 1

        wm.step(t, fx)
        energies.append(fx)

        if fx < best_e:
            best_e = fx
            best_x = x.clone()

        if t > burn_in and t % save_every == 0:
            samples.append(x.clone())

    return {
        "best_energy": best_e,
        "best_x": best_x,
        "acceptance_rate": accept_count / n_iters,
        "energies": torch.tensor(energies),
        "samples": torch.stack(samples) if samples else torch.empty(0, dim),
        "flatness": wm.flatness(),
        "bins_visited": int((wm.counts > 0).sum().item()),
    }


def run_samc_benchmark(
    name,
    energy_fn,
    dim,
    n_iters,
    partition_kwargs,
    gain_kwargs,
    proposal_std,
    temperature,
    n_chains=1,
    burn_in_frac=0.1,
    save_every=100,
):
    burn_in = int(n_iters * burn_in_frac)
    n_samples = (n_iters - burn_in) // save_every

    chain_str = f", {n_chains} chains" if n_chains > 1 else ""
    print(f"\n{'=' * 60}")
    print(f"  SAMC (SAMCWeights) — {name}{chain_str}")
    print(f"{'=' * 60}")
    print(
        f"  {n_iters:,} iters, burn-in={burn_in:,},"
        f" save_every={save_every}, n_samples={n_samples:,}"
    )

    torch.manual_seed(42)
    t0 = time.perf_counter()

    if n_chains <= 1:
        partition = UniformPartition(**partition_kwargs)
        gain = GainSequence("ramp", **gain_kwargs)
        wm = SAMCWeights(partition=partition, gain=gain)
        result = _run_single_samc_chain(
            energy_fn,
            dim,
            n_iters,
            wm,
            proposal_std,
            temperature,
            burn_in,
            save_every,
        )
        wall_time = time.perf_counter() - t0
        out = {
            "best_energy": result["best_energy"],
            "best_x": result["best_x"],
            "acceptance_rate": result["acceptance_rate"],
            "wall_time": wall_time,
            "energies": result["energies"],
            "samples": result["samples"],
            "flatness": result["flatness"],
            "bins_visited": result["bins_visited"],
        }
    else:
        # Multi-chain: shared partition + gain, independent SAMCWeights
        chains = []
        for _c in range(n_chains):
            partition = UniformPartition(**partition_kwargs)
            gain = GainSequence("ramp", **gain_kwargs)
            wm = SAMCWeights(partition=partition, gain=gain)
            chain = _run_single_samc_chain(
                energy_fn,
                dim,
                n_iters,
                wm,
                proposal_std,
                temperature,
                burn_in,
                save_every,
            )
            chains.append(chain)
        wall_time = time.perf_counter() - t0

        best_idx = min(range(n_chains), key=lambda i: chains[i]["best_energy"])
        best = chains[best_idx]
        avg_acc = sum(c["acceptance_rate"] for c in chains) / n_chains
        out = {
            "best_energy": best["best_energy"],
            "best_x": best["best_x"],
            "acceptance_rate": avg_acc,
            "wall_time": wall_time,
            "energies": best["energies"],
            "samples": best["samples"],
            "flatness": best["flatness"],
            "bins_visited": best["bins_visited"],
            "n_chains": n_chains,
        }

    print(
        f"  best_E={out['best_energy']:.4f}  acc={out['acceptance_rate']:.3f}  "
        f"flatness={out['flatness']:.3f}  time={wall_time:.1f}s"
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_chains", type=int, default=1)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    results["2d"] = run_samc_benchmark(
        name="2D Multimodal",
        energy_fn=cost_2d,
        dim=2,
        n_iters=500_000,
        n_chains=args.n_chains,
        partition_kwargs={"e_min": -8.2, "e_max": 0.0, "n_bins": 42},
        gain_kwargs={"rho": 1.0, "tau": 1.0, "warmup": 1, "step_scale": 1000},
        proposal_std=0.05,
        temperature=0.1,
    )

    results["10d"] = run_samc_benchmark(
        name="10D Gaussian Mixture",
        energy_fn=gaussian_mixture_10d,
        dim=10,
        n_iters=200_000,
        n_chains=args.n_chains,
        partition_kwargs={"e_min": 0.0, "e_max": 20.0, "n_bins": 30},
        gain_kwargs={"rho": 1.0, "tau": 1.0, "warmup": 1, "step_scale": 1000},
        proposal_std=1.0,
        temperature=1.0,
    )

    torch.save(results, RESULTS_DIR / "samc.pt")
    print(f"\nSaved to {RESULTS_DIR / 'samc.pt'}")


if __name__ == "__main__":
    main()
