"""Ablation: eager vs lazy GrowingPartition strategies.

Compares growth strategies across bin_width values on 2D and 10D problems.
Also includes a UniformPartition baseline with known-correct range.

Usage:
    conda run -n illuma-samc python ablation/run_growing_partition_ablation.py
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from illuma_samc import GainSequence, SAMCWeights, UniformPartition
from illuma_samc.problems import PROBLEMS

# ────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────

MODELS = ["2d", "10d"]
SEEDS = [42, 123, 456]
N_ITERS = 100_000
BIN_WIDTHS = [0.1, 0.2, 0.5, 1.0]
PROPOSAL_STD = 0.25
TEMPERATURE = 1.0

# Known-good ranges for baseline
BASELINE_RANGES = {
    "2d": {"e_min": -8.2, "e_max": 2.0, "n_bins": 42},
    "10d": {"e_min": -15.0, "e_max": 5.0, "n_bins": 42},
}

OUTPUT_DIR = Path("ablation/results/growing_partition")


def _eval_energy(energy_fn, x):
    result = energy_fn(x)
    if isinstance(result, tuple):
        e, in_r = result
        e = float(e)
        if isinstance(in_r, torch.Tensor):
            in_r = bool(in_r.item())
        return e, in_r
    return float(result), True


def run_chain(energy_fn, dim, wm, n_iters, proposal_std, temperature):
    """Run one MH+SAMCWeights chain."""
    x = torch.zeros(dim)
    fx, _ = _eval_energy(energy_fn, x)
    best_e = fx
    accept_count = 0

    for t in range(1, n_iters + 1):
        x_new = x + proposal_std * torch.randn(dim)
        fy, in_r = _eval_energy(energy_fn, x_new)
        log_r = (-fy + fx) / temperature + wm.correction(fx, fy)

        if in_r and (log_r > 0 or math.log(torch.rand(1).item() + 1e-300) < log_r):
            x, fx = x_new.clone(), fy
            accept_count += 1

        wm.step(t, fx)
        if fx < best_e:
            best_e = fx

    return {
        "best_energy": best_e,
        "acceptance_rate": accept_count / n_iters,
        "bin_flatness": wm.flatness(),
        "final_n_bins": wm.n_bins,
    }


def run_experiment(model_key, method, bin_width, seed):
    """Run a single experiment configuration."""
    problem = PROBLEMS[model_key]
    energy_fn = problem["energy_fn"]
    dim = problem["dim"]
    torch.manual_seed(seed)

    gain_kwargs = {"rho": 1.0, "tau": 1.0, "warmup": 1, "step_scale": 1000}

    if method == "baseline":
        rng = BASELINE_RANGES[model_key]
        partition = UniformPartition(e_min=rng["e_min"], e_max=rng["e_max"], n_bins=rng["n_bins"])
        gain = GainSequence("ramp", **gain_kwargs)
        wm = SAMCWeights(partition=partition, gain=gain)
    elif method == "eager":
        wm = SAMCWeights.auto(
            bin_width=bin_width, growth="eager", gain="ramp", gain_kwargs=gain_kwargs
        )
    elif method == "lazy_5":
        wm = SAMCWeights.auto(
            bin_width=bin_width,
            growth="lazy",
            expand_threshold=5,
            gain="ramp",
            gain_kwargs=gain_kwargs,
        )
    elif method == "lazy_20":
        wm = SAMCWeights.auto(
            bin_width=bin_width,
            growth="lazy",
            expand_threshold=20,
            gain="ramp",
            gain_kwargs=gain_kwargs,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    t0 = time.perf_counter()
    result = run_chain(energy_fn, dim, wm, N_ITERS, PROPOSAL_STD, TEMPERATURE)
    result["wall_time"] = time.perf_counter() - t0
    result["method"] = method
    result["model"] = model_key
    result["bin_width"] = bin_width
    result["seed"] = seed
    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    methods_with_bw = ["eager", "lazy_5", "lazy_20"]
    total = (
        len(MODELS) * len(SEEDS) * len(BIN_WIDTHS) * len(methods_with_bw)
        + len(MODELS) * len(SEEDS)  # baseline
    )
    count = 0

    for model_key in MODELS:
        # Baseline (no bin_width sweep)
        for seed in SEEDS:
            count += 1
            print(f"[{count}/{total}] baseline / {model_key} / seed={seed}")
            result = run_experiment(model_key, "baseline", 0.0, seed)
            all_results.append(result)

        # Growing methods
        for bw in BIN_WIDTHS:
            for method in methods_with_bw:
                for seed in SEEDS:
                    count += 1
                    print(f"[{count}/{total}] {method} / {model_key} / bw={bw} / seed={seed}")
                    result = run_experiment(model_key, method, bw, seed)
                    all_results.append(result)

    # Save results
    output_file = OUTPUT_DIR / "results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Print summary table
    print_summary(all_results)

    return all_results


def print_summary(results):
    """Print a summary table of results."""
    from collections import defaultdict

    grouped = defaultdict(list)
    for r in results:
        key = (r["model"], r["method"], r.get("bin_width", 0.0))
        grouped[key].append(r)

    print(
        f"\n{'Model':<8} {'Method':<10} {'BW':<5} {'BestE':>8} {'Flat':>6} {'AccR':>6} {'Bins':>5}"
    )
    print("-" * 55)

    for (model, method, bw), runs in sorted(grouped.items()):
        avg_best = sum(r["best_energy"] for r in runs) / len(runs)
        avg_flat = sum(r["bin_flatness"] for r in runs) / len(runs)
        avg_acc = sum(r["acceptance_rate"] for r in runs) / len(runs)
        avg_bins = sum(r["final_n_bins"] for r in runs) / len(runs)
        print(
            f"{model:<8} {method:<10} {bw:<5.1f} "
            f"{avg_best:>8.3f} {avg_flat:>6.3f} {avg_acc:>6.3f} {avg_bins:>5.0f}"
        )


if __name__ == "__main__":
    main()
