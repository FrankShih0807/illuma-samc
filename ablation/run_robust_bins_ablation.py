"""Run robust energy bin ablation (Phase 5, Step 42).

Tests 4 methods (uniform, overflow_bins, auto_range, expandable) across
5 wrong-range scenarios on all 4 problems with 3 seeds each.

Usage:
    conda run -n illuma-samc python ablation/run_robust_bins_ablation.py --parallel 4
    conda run -n illuma-samc python ablation/run_robust_bins_ablation.py --dry-run
    conda run -n illuma-samc python ablation/run_robust_bins_ablation.py --group 2d
    conda run -n illuma-samc python ablation/run_robust_bins_ablation.py --scenario wrong_range
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SEEDS = [42, 123, 456]
BASE_OUTPUT = Path("outputs/ablation/robust_bins")

# ────────────────────────────────────────────────────────
# Problem-specific "correct" energy ranges (from configs/samc.yaml)
# ────────────────────────────────────────────────────────

CORRECT_RANGES = {
    "2d": {"e_min": -8.2, "e_max": 0.0},
    "10d": {"e_min": 0.0, "e_max": 20.0},
    "rosenbrock": {"e_min": 0.0, "e_max": 500.0},
    "rastrigin": {"e_min": 0.0, "e_max": 200.0},
}

# ────────────────────────────────────────────────────────
# Scenario definitions: deliberate range misspecifications
# ────────────────────────────────────────────────────────


def _scenarios(model: str) -> dict[str, dict]:
    """Return {scenario_name: {e_min, e_max}} for a given model."""
    cr = CORRECT_RANGES[model]
    e_min_c, e_max_c = cr["e_min"], cr["e_max"]
    span = abs(e_max_c - e_min_c) if abs(e_max_c - e_min_c) > 1e-6 else 10.0
    # emin_high: raise e_min so the low-energy region is missed
    if e_min_c < 0:
        emin_high_min = e_min_c * 0.3  # e.g., -8.2 -> -2.46
        emin_high_max = max(e_max_c, e_min_c * 0.3 + span * 0.5)
    else:
        emin_high_min = e_min_c + span * 0.3
        emin_high_max = e_max_c
    # Ensure emin_high_max > emin_high_min
    if emin_high_max <= emin_high_min:
        emin_high_max = emin_high_min + span * 0.5

    return {
        "correct": {"e_min": e_min_c, "e_max": e_max_c},
        "emax_tight": {"e_min": e_min_c, "e_max": e_max_c * 0.5 if e_max_c != 0 else -4.0},
        "emax_wide": {"e_min": e_min_c, "e_max": e_max_c * 5 if e_max_c != 0 else 50.0},
        "emin_high": {"e_min": emin_high_min, "e_max": emin_high_max},
        "wrong_range": {"e_min": 10.0, "e_max": 20.0},
    }


# ────────────────────────────────────────────────────────
# Method definitions
# ────────────────────────────────────────────────────────

METHODS = {
    "uniform": {},  # baseline, no extra flags
    "overflow_bins": {"overflow_bins": True},
    "auto_range": {"auto_range": True},
    "expandable": {"expandable": True},
}

# ────────────────────────────────────────────────────────
# Generate commands
# ────────────────────────────────────────────────────────


def generate_commands(
    models: list[str] | None = None,
    scenario_filter: str | None = None,
) -> dict[str, list[str]]:
    """Generate all train.py commands organized by group (model)."""
    if models is None:
        models = list(CORRECT_RANGES.keys())

    all_commands = {}
    for model in models:
        cmds = []
        scenarios = _scenarios(model)
        if scenario_filter:
            scenarios = {k: v for k, v in scenarios.items() if k == scenario_filter}

        for scenario_name, range_cfg in scenarios.items():
            for method_name, method_flags in METHODS.items():
                for seed in SEEDS:
                    output_dir = BASE_OUTPUT / model / scenario_name / method_name / f"seed_{seed}"
                    parts = [
                        sys.executable,
                        "train.py",
                        "--algo=samc",
                        f"--model={model}",
                        f"--seed={seed}",
                        "--config=configs/samc.yaml",
                        f"--output_dir={output_dir}",
                        "--n_iters=100000",  # shorter for ablation speed
                    ]

                    # auto_range ignores e_min/e_max (discovers its own)
                    if not method_flags.get("auto_range", False):
                        parts.append(f"--e_min={range_cfg['e_min']}")
                        parts.append(f"--e_max={range_cfg['e_max']}")

                    # Add method-specific flags
                    for flag_key, flag_val in method_flags.items():
                        if isinstance(flag_val, bool) and flag_val:
                            parts.append(f"--{flag_key}")
                        else:
                            parts.append(f"--{flag_key}={flag_val}")

                    cmds.append(" ".join(str(p) for p in parts))
        all_commands[model] = cmds
    return all_commands


def run_command(cmd: str) -> tuple[str, int, str]:
    """Run a single command and return (cmd, returncode, stderr)."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)
    return cmd, result.returncode, result.stderr[:500] if result.returncode != 0 else ""


def main():
    parser = argparse.ArgumentParser(description="Run robust energy bin ablation (Phase 5).")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--parallel", type=int, default=4, help="Concurrent processes")
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Run only this model (2d, 10d, rosenbrock, rastrigin)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Run only this scenario (correct, emax_tight, emax_wide, emin_high, wrong_range)",
    )
    args = parser.parse_args()

    models = [args.group] if args.group else None
    all_commands = generate_commands(models=models, scenario_filter=args.scenario)

    total = sum(len(v) for v in all_commands.values())
    n_per_model = {k: len(v) for k, v in all_commands.items()}
    print(f"Total: {total} runs across {len(all_commands)} models")
    for model, n in n_per_model.items():
        print(f"  {model}: {n} runs")
    print()

    if args.dry_run:
        for model, cmds in all_commands.items():
            print(f"--- {model} ({len(cmds)} runs) ---")
            for cmd in cmds:
                print(cmd)
            print()
        return

    overall_success = 0
    overall_fail = 0

    for model, cmds in all_commands.items():
        print(f"\n{'=' * 60}")
        print(f"  Model: {model} ({len(cmds)} runs)")
        print(f"{'=' * 60}")

        if args.parallel <= 1:
            results = []
            for i, cmd in enumerate(cmds):
                print(f"  [{i + 1}/{len(cmds)}] Running...")
                results.append(run_command(cmd))
        else:
            results = []
            with ProcessPoolExecutor(max_workers=args.parallel) as pool:
                futures = {pool.submit(run_command, cmd): cmd for cmd in cmds}
                done_count = 0
                for future in as_completed(futures):
                    done_count += 1
                    cmd, rc, err = future.result()
                    results.append((cmd, rc, err))
                    status = "OK" if rc == 0 else "FAIL"
                    print(f"  [{done_count}/{len(cmds)}] {status}", flush=True)
                    if rc != 0:
                        print(f"    {err[:200]}")

        n_ok = sum(1 for _, rc, _ in results if rc == 0)
        n_fail = len(results) - n_ok
        overall_success += n_ok
        overall_fail += n_fail
        print(f"  Result: {n_ok}/{len(results)} succeeded")

    print(f"\n{'=' * 60}")
    print(
        f"  TOTAL: {overall_success}/{overall_success + overall_fail} succeeded, "
        f"{overall_fail} failed"
    )
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
