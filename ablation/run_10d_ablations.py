"""Run 10D Gaussian Mixture ablation groups for Step 26.

Usage:
    conda run -n illuma-samc python ablation/run_10d_ablations.py --parallel 4
    conda run -n illuma-samc python ablation/run_10d_ablations.py --dry-run
    conda run -n illuma-samc python ablation/run_10d_ablations.py --group samc_proposal_std
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

MODEL = "10d"
SEEDS = [42, 123, 456]
BASE_OUTPUT = Path("outputs/ablation")

# ────────────────────────────────────────────────────────
# Ablation group definitions
# ────────────────────────────────────────────────────────

GROUPS: dict[str, list[dict]] = {}


def _samc_base(**overrides) -> dict:
    cfg = {"algo": "samc", "config": "configs/samc.yaml"}
    cfg.update(overrides)
    return cfg


def _mh_base(**overrides) -> dict:
    cfg = {"algo": "mh", "config": "configs/mh.yaml"}
    cfg.update(overrides)
    return cfg


def _pt_base(**overrides) -> dict:
    cfg = {"algo": "pt", "config": "configs/pt.yaml"}
    cfg.update(overrides)
    return cfg


# SAMC proposal_std: 0.5, 1.0, 2.0 (9 runs)
GROUPS["samc_proposal_std"] = [_samc_base(proposal_std=s) for s in [0.5, 1.0, 2.0]]

# SAMC gain: ramp, 1/t (6 runs)
GROUPS["samc_gain_schedule"] = [_samc_base(gain=g) for g in ["ramp", "1/t"]]

# SAMC n_bins: 20, 42, 80 (9 runs)
GROUPS["samc_n_bins"] = [_samc_base(n_partitions=n) for n in [20, 42, 80]]

# SAMC n_chains: 1, 4, 8, 16 with shared_weights=true (12 runs)
GROUPS["samc_multi_chain_shared"] = [
    _samc_base(n_chains=nc, shared_weights=True) for nc in [1, 4, 8, 16]
]

# SAMC n_chains: 4, 8 with shared_weights=false (6 runs)
GROUPS["samc_multi_chain_independent"] = [
    _samc_base(n_chains=nc, shared_weights=False) for nc in [4, 8]
]

# MH proposal_std: 0.5, 1.0, 2.0 (9 runs)
GROUPS["mh_proposal_std"] = [_mh_base(proposal_std=s) for s in [0.5, 1.0, 2.0]]

# MH temperature: 1.0, 2.0 (6 runs)
GROUPS["mh_temperature"] = [_mh_base(temperature=t) for t in [1.0, 2.0]]

# PT n_replicas: 4, 8, 16 (9 runs)
GROUPS["pt_n_replicas"] = [_pt_base(n_replicas=n) for n in [4, 8, 16]]

# PT t_max: 10, 20 (6 runs)
GROUPS["pt_t_max"] = [_pt_base(t_max=t) for t in [10, 20]]


def _param_str(cfg: dict) -> str:
    skip = {"algo", "config"}
    parts = []
    for k, v in sorted(cfg.items()):
        if k not in skip:
            v_str = str(v).replace("/", "_over_")
            parts.append(f"{k}={v_str}")
    return "_".join(parts) if parts else "default"


def generate_commands() -> dict[str, list[str]]:
    all_commands = {}
    for group_name, configs in GROUPS.items():
        cmds = []
        for cfg in configs:
            for seed in SEEDS:
                param_str = _param_str(cfg)
                output_dir = BASE_OUTPUT / group_name / MODEL / param_str / f"seed_{seed}"
                parts = [
                    sys.executable,
                    "train.py",
                    f"--algo={cfg['algo']}",
                    f"--model={MODEL}",
                    f"--seed={seed}",
                    f"--config={cfg['config']}",
                    f"--output_dir={output_dir}",
                ]
                skip = {"algo", "config"}
                for k, v in cfg.items():
                    if k not in skip:
                        parts.append(f"--{k}={v}")
                cmds.append(" ".join(str(p) for p in parts))
        all_commands[group_name] = cmds
    return all_commands


def run_command(cmd: str) -> tuple[str, int, str]:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600)
    return cmd, result.returncode, result.stderr[:500] if result.returncode != 0 else ""


def main():
    parser = argparse.ArgumentParser(description="Run 10D Gaussian Mixture ablation groups.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--parallel", type=int, default=4, help="Concurrent processes")
    parser.add_argument("--group", type=str, default=None, help="Run only this group")
    args = parser.parse_args()

    all_commands = generate_commands()

    if args.group:
        if args.group not in all_commands:
            print(f"Unknown group: {args.group}. Available: {list(all_commands.keys())}")
            return
        all_commands = {args.group: all_commands[args.group]}

    total = sum(len(v) for v in all_commands.values())
    print(f"Total: {total} runs across {len(all_commands)} groups")
    print()

    if args.dry_run:
        for group_name, cmds in all_commands.items():
            print(f"--- {group_name} ({len(cmds)} runs) ---")
            for cmd in cmds:
                print(cmd)
            print()
        return

    overall_success = 0
    overall_fail = 0

    for group_name, cmds in all_commands.items():
        print(f"\n{'=' * 60}")
        print(f"  Group: {group_name} ({len(cmds)} runs)")
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
        print(f"  Group result: {n_ok}/{len(results)} succeeded")

    print(f"\n{'=' * 60}")
    print(
        f"  TOTAL: {overall_success}/{overall_success + overall_fail} succeeded, "
        f"{overall_fail} failed"
    )
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
