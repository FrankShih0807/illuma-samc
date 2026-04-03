"""Sweep runner: reads YAML sweep specs and generates train.py commands.

Usage:
    python ablation/sweep.py ablation/specs/samc_gain.yaml --dry-run
    python ablation/sweep.py ablation/specs/samc_gain.yaml --parallel 4
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml


def load_sweep_spec(path: str) -> dict:
    """Load a YAML sweep specification."""
    with open(path) as f:
        return yaml.safe_load(f)


def generate_commands(spec: dict) -> list[str]:
    """Generate list of train.py commands from a sweep spec.

    Spec format::

        group: samc_gain_schedule
        algo: samc
        models: [2d]
        n_seeds: 3
        seeds: [42, 123, 456]
        base_config: configs/samc.yaml
        sweep:
          gain: ["1/t", "log", "ramp"]
          proposal_std: [0.01, 0.05, 0.1]
    """
    algo = spec["algo"]
    models = spec["models"]
    seeds = spec.get("seeds", [42])
    if "n_seeds" in spec and len(seeds) < spec["n_seeds"]:
        # Extend seeds if n_seeds is specified but not enough seeds given
        import random

        rng = random.Random(0)
        while len(seeds) < spec["n_seeds"]:
            seeds.append(rng.randint(1, 99999))
    base_config = spec.get("base_config")
    group = spec.get("group", "default")
    sweep_params = spec.get("sweep", {})

    # Generate all combinations of sweep parameters
    param_names = list(sweep_params.keys())
    param_values = [sweep_params[k] for k in param_names]
    combinations = list(itertools.product(*param_values)) if param_values else [()]

    commands = []
    for model in models:
        for combo in combinations:
            for seed in seeds:
                parts = [
                    sys.executable,
                    "train.py",
                    f"--algo={algo}",
                    f"--model={model}",
                    f"--seed={seed}",
                ]
                if base_config:
                    parts.append(f"--config={base_config}")

                # Output dir encodes the sweep group and params
                param_str = "_".join(f"{k}={v}" for k, v in zip(param_names, combo))
                output_dir = f"outputs/ablation/{group}/{model}/{param_str}/seed_{seed}"
                parts.append(f"--output_dir={output_dir}")

                # Add swept params as CLI overrides
                for k, v in zip(param_names, combo):
                    parts.append(f"--{k}={v}")

                commands.append(" ".join(str(p) for p in parts))

    return commands


def run_command(cmd: str) -> tuple[str, int]:
    """Run a single command and return (cmd, returncode)."""
    print(f"[RUN] {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[FAIL] {cmd}\n{result.stderr[:500]}")
    else:
        print(f"[DONE] {cmd}")
    return cmd, result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run ablation sweep from YAML spec.")
    parser.add_argument("spec", type=str, help="Path to YAML sweep spec file")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--parallel", type=int, default=1, help="Number of concurrent processes")
    args = parser.parse_args()

    spec = load_sweep_spec(args.spec)
    commands = generate_commands(spec)

    print(f"Generated {len(commands)} commands from {args.spec}")
    print(f"Group: {spec.get('group', 'default')}")
    print()

    if args.dry_run:
        for cmd in commands:
            print(cmd)
        return

    if args.parallel <= 1:
        results = [run_command(cmd) for cmd in commands]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(run_command, cmd): cmd for cmd in commands}
            for future in as_completed(futures):
                results.append(future.result())

    # Summary
    n_success = sum(1 for _, rc in results if rc == 0)
    n_fail = len(results) - n_success
    print(f"\nSweep complete: {n_success}/{len(results)} succeeded, {n_fail} failed")


if __name__ == "__main__":
    main()
