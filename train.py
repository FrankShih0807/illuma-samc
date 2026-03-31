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

from illuma_samc import SAMC
from illuma_samc.analysis import compute_bin_flatness
from illuma_samc.baselines import run_mh, run_parallel_tempering
from illuma_samc.partitions import AdaptivePartition, QuantilePartition
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
    if ptype == "adaptive":
        return AdaptivePartition(n_bins=n_partitions, e_min=e_min, e_max=e_max)
    elif ptype == "quantile":
        # QuantilePartition needs warmup energies — evaluate random samples
        n_warmup = max(1000, n_partitions * 50)
        warmup_x = torch.randn(n_warmup, dim)
        raw = energy_fn(warmup_x)
        energies = raw[0] if isinstance(raw, tuple) else raw
        return QuantilePartition(energies=energies, n_bins=n_partitions)
    # uniform — return None so SAMC uses its default UniformPartition
    return None


def run_samc_experiment(energy_fn, dim: int, cfg: dict) -> dict:
    """Run SAMC and return results dict."""
    n_iters = cfg.get("n_iters", 500_000)
    save_every = cfg.get("save_every", 100)

    gain_kwargs = cfg.get("gain_kwargs", {"rho": 1.0, "tau": 1.0, "warmup": 1, "step_scale": 1000})
    # Allow gain_t0 CLI override
    if "gain_t0" in cfg:
        gain_kwargs = dict(gain_kwargs)
        gain_kwargs["t0"] = cfg["gain_t0"]

    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)

    partition_fn = _build_partition(energy_fn, dim, cfg)

    t0 = time.perf_counter()
    sampler_kwargs = {
        "energy_fn": energy_fn,
        "dim": dim,
        "n_partitions": cfg.get("n_partitions", 42),
        "e_min": cfg.get("e_min", -8.2),
        "e_max": cfg.get("e_max", 0.0),
        "proposal_std": cfg.get("proposal_std", 0.25),
        "gain": cfg.get("gain", "ramp"),
        "gain_kwargs": gain_kwargs,
    }
    if partition_fn is not None:
        sampler_kwargs["partition_fn"] = partition_fn
    sampler = SAMC(**sampler_kwargs)

    # Multi-chain support
    n_chains = cfg.get("n_chains", 1)
    if n_chains > 1:
        x0 = torch.randn(n_chains, dim)
    else:
        x0 = None
    result = sampler.run(n_steps=n_iters, x0=x0, save_every=save_every, progress=True)
    wall_time = time.perf_counter() - t0

    return {
        "best_energy": float(result.best_energy),
        "acceptance_rate": float(result.acceptance_rate),
        "bin_flatness": compute_bin_flatness(result.bin_counts),
        "wall_time": wall_time,
        "total_energy_evals": n_iters,
        "n_iters": n_iters,
        "sampler": sampler,
        "result": result,
    }


def run_mh_experiment(energy_fn, dim: int, cfg: dict) -> dict:
    """Run MH and return results dict."""
    n_iters = cfg.get("n_iters", 500_000)
    save_every = cfg.get("save_every", 100)
    burn_in_frac = cfg.get("burn_in_frac", 0.1)
    burn_in = int(n_iters * burn_in_frac)

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
    )
    wall_time = time.perf_counter() - t0

    return {
        "best_energy": float(mh_result["best_energy"]),
        "acceptance_rate": float(mh_result["acceptance_rate"]),
        "wall_time": wall_time,
        "total_energy_evals": n_iters,
        "n_iters": n_iters,
    }


def run_pt_experiment(energy_fn, dim: int, cfg: dict) -> dict:
    """Run PT and return results dict."""
    n_iters = cfg.get("n_iters", 500_000)
    n_replicas = cfg.get("n_replicas", 8)
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
        t_min=cfg.get("t_min", 1.0),
        t_max=cfg.get("t_max", 10.0),
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
    serializable = {k: v for k, v in metrics.items() if k not in ("sampler", "result")}
    with open(output_dir / "results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    # Save diagnostic plots for SAMC
    if "sampler" in metrics and "result" in metrics:
        try:
            import matplotlib

            matplotlib.use("Agg")
            metrics["sampler"].plot_diagnostics()
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
    parser.add_argument("--temperature", type=float, default=None, help="Temperature (MH)")
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
        help="Override output directory (default: outputs/<model>/<algo>/<timestamp>)",
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
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("outputs") / args.model / args.algo / timestamp

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
