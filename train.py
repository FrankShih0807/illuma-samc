"""Experiment CLI for running SAMC / MH / PT on benchmark problems.

Usage:
    python train.py --algo samc --model 2d
    python train.py --algo mh --model 10d --proposal_std 0.5
    python train.py --algo pt --model 2d --n_replicas 12
    python train.py --algo samc --model 2d --config configs/samc.yaml

Each invocation runs a single experiment and saves results to
``outputs/<model>/<algo>/<timestamp>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from illuma_samc import GainSequence, SAMCWeights, UniformPartition
from illuma_samc.analysis import compute_energy_mixing
from illuma_samc.baselines import run_mh, run_parallel_tempering
from illuma_samc.partitions import AdaptivePartition, ExpandablePartition, QuantilePartition
from illuma_samc.problems import PROBLEMS

# ────────────────────────────────────────────────────────
# Model registry
# ────────────────────────────────────────────────────────

MODELS = {key: {"energy_fn": val["energy_fn"], "dim": val["dim"]} for key, val in PROBLEMS.items()}


# ────────────────────────────────────────────────────────
# Config loading
# ────────────────────────────────────────────────────────


def load_yaml_defaults(config_path: str, model: str) -> dict:
    """Load YAML config and return the section for the given model."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if model not in cfg:
        raise KeyError(f"Model '{model}' not found in config {config_path}. Available: {list(cfg)}")
    return cfg[model]


def merge_config(defaults: dict, cli_overrides: dict) -> dict:
    """Merge YAML defaults with CLI overrides. CLI wins on conflicts."""
    merged = dict(defaults)
    for k, v in cli_overrides.items():
        if v is not None:
            merged[k] = v
    return merged


# ────────────────────────────────────────────────────────
# Runners
# ────────────────────────────────────────────────────────


def _build_partition(energy_fn, dim: int, cfg: dict):
    """Build partition object based on config, or None for default."""
    ptype = cfg.get("partition_type", "uniform")
    n_partitions = cfg.get("n_partitions", 42)
    e_min = cfg.get("e_min", -8.2)
    e_max = cfg.get("e_max", 0.0)
    overflow_bins = cfg.get("overflow_bins", False)

    if cfg.get("expandable", False):
        return ExpandablePartition(
            e_min=e_min, e_max=e_max, n_bins=n_partitions, expand_step=5, max_bins=200
        )
    elif ptype == "adaptive":
        return AdaptivePartition(n_bins=n_partitions, e_min=e_min, e_max=e_max)
    elif ptype == "quantile":
        # QuantilePartition needs warmup energies — evaluate random samples
        n_warmup = max(1000, n_partitions * 50)
        warmup_x = torch.randn(n_warmup, dim)
        raw = energy_fn(warmup_x)
        energies = raw[0] if isinstance(raw, tuple) else raw
        return QuantilePartition(energies=energies, n_bins=n_partitions)
    elif overflow_bins:
        return UniformPartition(e_min=e_min, e_max=e_max, n_bins=n_partitions, overflow_bins=True)
    # uniform — return None so SAMC uses its default UniformPartition
    return None


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


def _run_samc_chain(energy_fn, dim, n_iters, wm, proposal_std, temperature):
    """Run one MH+SAMCWeights chain. Returns dict."""
    import math

    x = torch.zeros(dim)
    fx, _ = _eval_energy(energy_fn, x)
    best_x, best_e = x.clone(), fx
    accept_count = 0
    energies = []

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

    return {
        "best_energy": best_e,
        "best_x": best_x,
        "acceptance_rate": accept_count / n_iters,
        "energies": torch.tensor(energies),
    }


def run_samc_experiment(energy_fn, dim: int, cfg: dict) -> dict:
    """Run SAMC via SAMCWeights and return results dict."""
    n_iters = cfg.get("n_iters", 500_000)

    gain_kwargs = cfg.get("gain_kwargs", {"rho": 1.0, "tau": 1.0, "warmup": 1, "step_scale": 1000})
    if "gain_t0" in cfg:
        gain_kwargs = dict(gain_kwargs)
        gain_kwargs["t0"] = cfg["gain_t0"]

    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)

    proposal_std = cfg.get("proposal_std", 0.25)
    temperature = cfg.get("temperature", 1.0)
    n_chains = cfg.get("n_chains", 1)
    auto_range = cfg.get("auto_range", False)

    gain_schedule = cfg.get("gain", "ramp")

    t0 = time.perf_counter()

    if auto_range:
        # Use from_warmup to auto-detect energy range
        overflow_bins = cfg.get("overflow_bins", False)
        wm = SAMCWeights.from_warmup(
            energy_fn,
            dim=dim,
            n_bins=cfg.get("n_partitions", 42),
            warmup_steps=cfg.get("warmup_steps", 5000),
            proposal_std=proposal_std,
            margin=cfg.get("margin", 0.1),
            gain=gain_schedule,
            gain_kwargs=gain_kwargs,
            overflow_bins=overflow_bins,
        )
    else:
        partition = _build_partition(energy_fn, dim, cfg)
        if partition is None:
            partition = UniformPartition(
                e_min=cfg.get("e_min", -8.2),
                e_max=cfg.get("e_max", 0.0),
                n_bins=cfg.get("n_partitions", 42),
            )
        gain = GainSequence(gain_schedule, **gain_kwargs)
        wm = SAMCWeights(partition=partition, gain=gain)

    if n_chains <= 1:
        result = _run_samc_chain(energy_fn, dim, n_iters, wm, proposal_std, temperature)
        wall_time = time.perf_counter() - t0
        mixing = compute_energy_mixing(result["energies"])
        return {
            "best_energy": float(result["best_energy"]),
            "acceptance_rate": float(result["acceptance_rate"]),
            "bin_flatness": float(wm.flatness()),
            "round_trip_time": mixing["round_trip_time"],
            "energy_autocorr_50": mixing["energy_autocorr_50"],
            "energy_autocorr_200": mixing["energy_autocorr_200"],
            "n_round_trips": mixing["n_round_trips"],
            "wall_time": wall_time,
            "total_energy_evals": n_iters,
            "n_iters": n_iters,
            "n_chains": 1,
            "wm": wm,
        }
    else:
        chains = []
        wms = []
        for _c in range(n_chains):
            if auto_range:
                chain_wm = SAMCWeights.from_warmup(
                    energy_fn,
                    dim=dim,
                    n_bins=cfg.get("n_partitions", 42),
                    warmup_steps=cfg.get("warmup_steps", 5000),
                    proposal_std=proposal_std,
                    margin=cfg.get("margin", 0.1),
                    gain=gain_schedule,
                    gain_kwargs=gain_kwargs,
                    overflow_bins=cfg.get("overflow_bins", False),
                )
            else:
                p = _build_partition(energy_fn, dim, cfg)
                if p is None:
                    p = UniformPartition(
                        e_min=cfg.get("e_min", -8.2),
                        e_max=cfg.get("e_max", 0.0),
                        n_bins=cfg.get("n_partitions", 42),
                    )
                g = GainSequence(gain_schedule, **gain_kwargs)
                chain_wm = SAMCWeights(partition=p, gain=g)
            chain = _run_samc_chain(energy_fn, dim, n_iters, chain_wm, proposal_std, temperature)
            chains.append(chain)
            wms.append(chain_wm)
        wall_time = time.perf_counter() - t0

        best_idx = min(range(n_chains), key=lambda i: chains[i]["best_energy"])
        best = chains[best_idx]
        avg_acc = sum(c["acceptance_rate"] for c in chains) / n_chains
        mixing = compute_energy_mixing(best["energies"])
        return {
            "best_energy": float(best["best_energy"]),
            "acceptance_rate": float(avg_acc),
            "bin_flatness": float(wms[best_idx].flatness()),
            "round_trip_time": mixing["round_trip_time"],
            "energy_autocorr_50": mixing["energy_autocorr_50"],
            "energy_autocorr_200": mixing["energy_autocorr_200"],
            "n_round_trips": mixing["n_round_trips"],
            "wall_time": wall_time,
            "total_energy_evals": n_iters * n_chains,
            "n_iters": n_iters,
            "n_chains": n_chains,
            "wm": wms[best_idx],
        }


def run_mh_experiment(energy_fn, dim: int, cfg: dict) -> dict:
    """Run MH and return results dict."""
    n_iters = cfg.get("n_iters", 500_000)
    save_every = cfg.get("save_every", 100)
    burn_in_frac = cfg.get("burn_in_frac", 0.1)
    burn_in = int(n_iters * burn_in_frac)
    n_chains = cfg.get("n_chains", 1)

    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)

    t0 = time.perf_counter()
    mh_result = run_mh(
        energy_fn,
        dim,
        n_iters,
        proposal_std=cfg.get("proposal_std", 0.25),
        temperature=cfg.get("temperature", 1.0),
        burn_in=burn_in,
        save_every=save_every,
        n_chains=n_chains,
    )
    wall_time = time.perf_counter() - t0

    return {
        "best_energy": float(mh_result["best_energy"]),
        "acceptance_rate": float(mh_result["acceptance_rate"]),
        "wall_time": wall_time,
        "total_energy_evals": n_iters * n_chains,
        "n_iters": n_iters,
        "n_chains": n_chains,
    }


def run_pt_experiment(energy_fn, dim: int, cfg: dict) -> dict:
    """Run PT and return results dict."""
    n_iters = cfg.get("n_iters", 500_000)
    n_replicas = cfg.get("n_replicas", 4)
    save_every = cfg.get("save_every", 100)
    burn_in_frac = cfg.get("burn_in_frac", 0.1)
    burn_in = int(n_iters * burn_in_frac)

    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)

    t0 = time.perf_counter()
    pt_result = run_parallel_tempering(
        energy_fn,
        dim,
        n_iters,
        n_replicas=n_replicas,
        proposal_std=cfg.get("proposal_std", 0.25),
        t_min=cfg.get("t_min", 0.1),
        t_max=cfg.get("t_max", 3.16),
        swap_interval=cfg.get("swap_interval", 10),
        burn_in=burn_in,
        save_every=save_every,
    )
    wall_time = time.perf_counter() - t0

    return {
        "best_energy": float(pt_result["best_energy"]),
        "acceptance_rate": float(pt_result["acceptance_rate"]),
        "swap_rate": float(pt_result["swap_rate"]),
        "wall_time": wall_time,
        "total_energy_evals": n_iters * n_replicas,
        "n_iters": n_iters,
        "n_replicas": n_replicas,
    }


# ────────────────────────────────────────────────────────
# Output management
# ────────────────────────────────────────────────────────


def save_results(output_dir: Path, cfg: dict, metrics: dict):
    """Save config snapshot and results JSON to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    # Save metrics (exclude non-serializable objects)
    serializable = {k: v for k, v in metrics.items() if k not in ("wm",)}
    with open(output_dir / "results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    # Save diagnostic plots for SAMC (SAMCWeights)
    if "wm" in metrics:
        try:
            import matplotlib

            matplotlib.use("Agg")
            metrics["wm"].plot_diagnostics()
            import matplotlib.pyplot as plt

            plt.savefig(output_dir / "diagnostics.png", dpi=150, bbox_inches="tight")
            plt.close()
        except Exception:
            pass  # matplotlib optional


# ────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SAMC/MH/PT experiment on a benchmark problem.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algo",
        choices=["samc", "mh", "pt"],
        required=True,
        help="Algorithm to run",
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        required=True,
        help="Benchmark problem",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--n_iters", type=int, default=None, help="Number of iterations")
    parser.add_argument("--proposal_std", type=float, default=None, help="Proposal step size")
    parser.add_argument(
        "--n_partitions", type=int, default=None, help="Number of energy bins (SAMC)"
    )
    parser.add_argument("--e_min", type=float, default=None, help="Min energy bound (SAMC)")
    parser.add_argument("--e_max", type=float, default=None, help="Max energy bound (SAMC)")
    parser.add_argument(
        "--gain", type=str, default=None, help="Gain schedule (SAMC): ramp, 1/t, log"
    )
    parser.add_argument("--temperature", type=float, default=None, help="Temperature (SAMC/MH)")
    parser.add_argument("--n_replicas", type=int, default=None, help="Number of replicas (PT)")
    parser.add_argument("--t_min", type=float, default=None, help="Min temperature (PT)")
    parser.add_argument("--t_max", type=float, default=None, help="Max temperature (PT)")
    parser.add_argument("--swap_interval", type=int, default=None, help="Swap interval (PT)")
    parser.add_argument("--save_every", type=int, default=None, help="Sample collection frequency")
    parser.add_argument("--burn_in_frac", type=float, default=None, help="Burn-in fraction")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory (default: outputs/<model>/<algo>/<name or timestamp>)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name for output folder (default: timestamp)",
    )
    parser.add_argument(
        "--partition_type",
        choices=["uniform", "adaptive", "quantile"],
        default=None,
        help="Partition type (SAMC)",
    )
    parser.add_argument("--n_chains", type=int, default=None, help="Number of parallel chains")
    parser.add_argument(
        "--gain_t0", type=float, default=None, help="Override gain_kwargs.t0 (SAMC)"
    )
    parser.add_argument(
        "--overflow_bins",
        action="store_true",
        default=False,
        help="Enable overflow bins on UniformPartition (SAMC)",
    )
    parser.add_argument(
        "--auto_range",
        action="store_true",
        default=False,
        help="Auto-detect energy range via warmup (SAMC, uses SAMCWeights.from_warmup)",
    )
    parser.add_argument(
        "--expandable",
        action="store_true",
        default=False,
        help="Use ExpandablePartition for dynamic bin expansion (SAMC)",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Load config defaults
    if args.config:
        config_path = args.config
    else:
        config_path = f"configs/{args.algo}.yaml"

    defaults = {}
    if os.path.exists(config_path):
        defaults = load_yaml_defaults(config_path, args.model)
        print(f"Loaded config from {config_path} [{args.model}]")

    # CLI overrides
    cli_overrides = {
        k: v
        for k, v in vars(args).items()
        if k not in ("algo", "model", "config", "output_dir") and v is not None
    }
    cfg = merge_config(defaults, cli_overrides)
    cfg["algo"] = args.algo
    cfg["model"] = args.model

    # Get model
    model_info = MODELS[args.model]
    energy_fn = model_info["energy_fn"]
    dim = model_info["dim"]

    print(f"\nRunning {args.algo.upper()} on {args.model} (dim={dim})")
    print(f"Config: {cfg}")

    # Run experiment
    runners = {
        "samc": run_samc_experiment,
        "mh": run_mh_experiment,
        "pt": run_pt_experiment,
    }
    metrics = runners[args.algo](energy_fn, dim, cfg)

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        folder = args.name or datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / args.model / args.algo / folder

    save_results(output_dir, cfg, metrics)

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"  Results: {args.algo.upper()} on {args.model}")
    print(f"{'=' * 50}")
    print(f"  Best energy:       {metrics['best_energy']:.4f}")
    print(f"  Acceptance rate:   {metrics['acceptance_rate']:.3f}")
    print(f"  Wall time:         {metrics['wall_time']:.1f}s")
    print(f"  Energy evals:      {metrics['total_energy_evals']:,}")
    if "bin_flatness" in metrics:
        print(f"  Bin flatness:      {metrics['bin_flatness']:.3f}")
    if "swap_rate" in metrics:
        print(f"  Swap rate:         {metrics['swap_rate']:.3f}")
    print(f"  Output saved to:   {output_dir}")


if __name__ == "__main__":
    main()
