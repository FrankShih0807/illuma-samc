"""Step 28: Cross-algorithm comparison with best configs and 10 seeds.

After ablation results are available, this script:
1. Picks best config per algo per problem from ablation results
2. Runs each with 10 seeds for tight error bars
3. Generates comparison figures and summary tables

Usage:
    conda run -n illuma-samc python ablation/run_cross_comparison.py --parallel 4
    conda run -n illuma-samc python ablation/run_cross_comparison.py --dry-run
    conda run -n illuma-samc python ablation/run_cross_comparison.py --run-only
    conda run -n illuma-samc python ablation/run_cross_comparison.py --analyze-only
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

SEEDS = [42, 123, 456, 789, 1337, 2024, 3141, 4269, 5555, 9876]
MODELS = ["rosenbrock", "10d", "rastrigin"]
ALGOS = ["samc", "mh", "pt"]
BASE_OUTPUT = Path("outputs/ablation/cross_comparison")
FIG_DIR = Path("ablation/figures")
REPORT_DIR = Path("ablation/reports")

# ────────────────────────────────────────────────────────
# Best configs (populated from ablation results)
# ────────────────────────────────────────────────────────

# These are the best configs found from Steps 25-27 ablations.
# Each entry: {cli_args_dict} to pass to train.py
BEST_CONFIGS: dict[str, dict[str, dict]] = {
    "rosenbrock": {
        "samc": {
            "algo": "samc",
            "config": "configs/samc.yaml",
            "proposal_std": 0.5,
            "gain": "ramp",
            "gain_t0": 5000,
            "n_chains": 8,
            "shared_weights": False,
        },
        "mh": {
            "algo": "mh",
            "config": "configs/mh.yaml",
            "proposal_std": 0.05,
            "temperature": 1.0,
        },
        "pt": {
            "algo": "pt",
            "config": "configs/pt.yaml",
            "n_replicas": 4,
            "t_max": 10,
        },
    },
    "10d": {
        "samc": {
            "algo": "samc",
            "config": "configs/samc.yaml",
            "proposal_std": 1.0,
            "gain": "ramp",
            "n_chains": 8,
            "shared_weights": True,
            "n_partitions": 20,
        },
        "mh": {
            "algo": "mh",
            "config": "configs/mh.yaml",
            "proposal_std": 0.5,
            "temperature": 1.0,
        },
        "pt": {
            "algo": "pt",
            "config": "configs/pt.yaml",
            "n_replicas": 16,
            "t_max": 10,
        },
    },
    "rastrigin": {
        "samc": {
            "algo": "samc",
            "config": "configs/samc.yaml",
            "proposal_std": 0.5,
            "gain": "ramp",
            "n_chains": 16,
            "shared_weights": True,
        },
        "mh": {
            "algo": "mh",
            "config": "configs/mh.yaml",
            "proposal_std": 1.0,
        },
        "pt": {
            "algo": "pt",
            "config": "configs/pt.yaml",
            "n_replicas": 16,
            "t_max": 10,
        },
    },
}


def update_best_configs_from_ablation():
    """Scan ablation results and update BEST_CONFIGS with the actual best."""
    ablation_root = Path("outputs/ablation")
    if not ablation_root.exists():
        print("No ablation results found, using default BEST_CONFIGS")
        return

    for model in MODELS:
        for algo in ALGOS:
            best_energy = float("inf")
            best_cfg = None
            # Scan all groups for this algo
            for results_json in ablation_root.rglob(f"**/{model}/**/results.json"):
                try:
                    with open(results_json) as f:
                        data = json.load(f)
                    config_yaml = results_json.parent / "config.yaml"
                    if not config_yaml.exists():
                        continue
                    import yaml

                    with open(config_yaml) as f:
                        cfg = yaml.safe_load(f)
                    if cfg.get("algo") != algo:
                        continue
                    if cfg.get("model") != model:
                        continue
                    e = data.get("best_energy", float("inf"))
                    if e < best_energy:
                        best_energy = e
                        # Extract relevant config params
                        best_cfg = {"algo": algo, "config": f"configs/{algo}.yaml"}
                        for k in [
                            "proposal_std",
                            "gain",
                            "gain_t0",
                            "n_partitions",
                            "n_chains",
                            "shared_weights",
                            "temperature",
                            "n_replicas",
                            "t_max",
                            "t_min",
                            "swap_interval",
                        ]:
                            if k in cfg and cfg[k] is not None:
                                best_cfg[k] = cfg[k]
                except Exception:
                    continue

            if best_cfg is not None:
                BEST_CONFIGS.setdefault(model, {})[algo] = best_cfg
                print(f"  Best {algo} on {model}: E={best_energy:.4f} -> {best_cfg}")


def _param_str(cfg: dict) -> str:
    skip = {"algo", "config"}
    parts = []
    for k, v in sorted(cfg.items()):
        if k not in skip:
            v_str = str(v).replace("/", "_over_")
            parts.append(f"{k}={v_str}")
    return "_".join(parts) if parts else "default"


def generate_commands() -> dict[str, list[str]]:
    """Generate train.py commands for all best configs x 10 seeds."""
    all_commands = {}
    for model in MODELS:
        for algo in ALGOS:
            if model not in BEST_CONFIGS or algo not in BEST_CONFIGS[model]:
                continue
            cfg = BEST_CONFIGS[model][algo]
            group_name = f"{model}_{algo}"
            cmds = []
            for seed in SEEDS:
                param_str = _param_str(cfg)
                output_dir = BASE_OUTPUT / model / algo / param_str / f"seed_{seed}"
                parts = [
                    sys.executable,
                    "train.py",
                    f"--algo={cfg['algo']}",
                    f"--model={model}",
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
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=7200)
    return cmd, result.returncode, result.stderr[:500] if result.returncode != 0 else ""


def run_experiments(parallel: int = 4, dry_run: bool = False):
    """Run all cross-comparison experiments."""
    print("\n=== Updating best configs from ablation results ===")
    update_best_configs_from_ablation()

    all_commands = generate_commands()
    total = sum(len(v) for v in all_commands.values())
    print(f"\nTotal: {total} runs across {len(all_commands)} groups")

    if dry_run:
        for group_name, cmds in all_commands.items():
            print(f"\n--- {group_name} ({len(cmds)} runs) ---")
            for cmd in cmds:
                print(cmd)
        return

    overall_success = 0
    overall_fail = 0

    for group_name, cmds in all_commands.items():
        print(f"\n{'=' * 60}")
        print(f"  Group: {group_name} ({len(cmds)} runs)")
        print(f"{'=' * 60}")

        if parallel <= 1:
            results = []
            for i, cmd in enumerate(cmds):
                print(f"  [{i + 1}/{len(cmds)}] Running...")
                results.append(run_command(cmd))
        else:
            results = []
            with ProcessPoolExecutor(max_workers=parallel) as pool:
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


# ────────────────────────────────────────────────────────
# Analysis and figure generation
# ────────────────────────────────────────────────────────


def load_results() -> dict[str, dict[str, list[dict]]]:
    """Load all cross-comparison results organized by model -> algo -> [runs]."""
    data: dict[str, dict[str, list[dict]]] = {}
    for model in MODELS:
        data[model] = {}
        for algo in ALGOS:
            model_algo_dir = BASE_OUTPUT / model / algo
            if not model_algo_dir.exists():
                continue
            runs = []
            for rfile in sorted(model_algo_dir.rglob("results.json")):
                with open(rfile) as f:
                    runs.append(json.load(f))
            if runs:
                data[model][algo] = runs
    return data


def _stats(values: list[float]) -> tuple[float, float]:
    """Return (mean, std) of a list."""
    import statistics

    if not values:
        return float("inf"), 0.0
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return m, s


def generate_figures(data: dict[str, dict[str, list[dict]]]):
    """Generate all Step 28 figures."""
    if not HAS_MPL:
        print("matplotlib not available, skipping figures.")
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    algo_colors = {"samc": "#2196F3", "mh": "#FF5722", "pt": "#4CAF50"}
    algo_labels = {"samc": "SAMC", "mh": "MH", "pt": "PT"}
    model_dims = {"rosenbrock": 2, "10d": 10, "rastrigin": 20}

    # ── Figure 1: Bar chart with error bars (best energy per algo per problem) ──
    fig, axes = plt.subplots(1, len(MODELS), figsize=(4 * len(MODELS), 5), sharey=False)
    if len(MODELS) == 1:
        axes = [axes]
    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        algos_present = [a for a in ALGOS if a in data.get(model, {})]
        means, stds, colors, labels = [], [], [], []
        for algo in algos_present:
            runs = data[model][algo]
            energies = [r["best_energy"] for r in runs]
            m, s = _stats(energies)
            means.append(m)
            stds.append(s)
            colors.append(algo_colors[algo])
            labels.append(algo_labels[algo])
        x = range(len(algos_present))
        ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel("Best Energy (mean +/- std)" if idx == 0 else "")
        ax.set_title(f"{model} (dim={model_dims.get(model, '?')})", fontsize=12)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Algorithm Comparison at Best Hyperparameters", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_algo_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIG_DIR / 'fig1_algo_comparison.png'}")

    # ── Figure 2: Scaling with dimensionality ──
    fig, ax = plt.subplots(figsize=(7, 5))
    for algo in ALGOS:
        dims, means, stds = [], [], []
        for model in MODELS:
            if algo in data.get(model, {}):
                runs = data[model][algo]
                energies = [r["best_energy"] for r in runs]
                m, s = _stats(energies)
                dims.append(model_dims[model])
                means.append(m)
                stds.append(s)
        if dims:
            ax.errorbar(
                dims,
                means,
                yerr=stds,
                marker="o",
                capsize=5,
                label=algo_labels[algo],
                color=algo_colors[algo],
                linewidth=2,
                markersize=8,
            )
    ax.set_xlabel("Dimensionality", fontsize=12)
    ax.set_ylabel("Best Energy (mean +/- std)", fontsize=12)
    ax.set_title("Scaling with Dimensionality", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_scaling_dimensionality.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIG_DIR / 'fig2_scaling_dimensionality.png'}")

    # ── Figure 3: Robustness to proposal_std (SAMC vs MH) ──
    # Load proposal_std sweep results from ablation
    fig, axes = plt.subplots(1, len(MODELS), figsize=(4.5 * len(MODELS), 5), sharey=False)
    if len(MODELS) == 1:
        axes = [axes]
    ablation_root = Path("outputs/ablation")
    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        for algo in ["samc", "mh"]:
            group_dir = ablation_root / f"{algo}_proposal_std" / model
            if not group_dir.exists():
                continue
            std_vals: dict[float, list[float]] = {}
            for rfile in group_dir.rglob("results.json"):
                cfg_path = rfile.parent / "config.yaml"
                if not cfg_path.exists():
                    continue
                import yaml

                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f)
                if cfg.get("algo") != algo or cfg.get("model") != model:
                    continue
                ps = float(cfg.get("proposal_std", 0))
                e = json.load(open(rfile))["best_energy"]
                std_vals.setdefault(ps, []).append(e)
            if std_vals:
                xs = sorted(std_vals.keys())
                ms = [_stats(std_vals[x])[0] for x in xs]
                ss = [_stats(std_vals[x])[1] for x in xs]
                ax.errorbar(
                    xs,
                    ms,
                    yerr=ss,
                    marker="o",
                    capsize=4,
                    label=algo_labels[algo],
                    color=algo_colors[algo],
                    linewidth=2,
                )
        ax.set_xlabel("proposal_std", fontsize=11)
        ax.set_ylabel("Best Energy" if idx == 0 else "", fontsize=11)
        ax.set_title(f"{model}", fontsize=12)
        ax.set_xscale("log")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Robustness to Proposal Step Size", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_robustness_proposal_std.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIG_DIR / 'fig3_robustness_proposal_std.png'}")

    # ── Figure 4: SAMC gain schedule convergence ──
    fig, axes = plt.subplots(1, len(MODELS), figsize=(4.5 * len(MODELS), 5), sharey=False)
    if len(MODELS) == 1:
        axes = [axes]
    gain_colors = {"ramp": "#2196F3", "1/t": "#FF9800", "1_over_t": "#FF9800"}
    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        group_dir = ablation_root / "samc_gain_schedule" / model
        if not group_dir.exists():
            continue
        gain_vals: dict[str, list[float]] = {}
        for rfile in group_dir.rglob("results.json"):
            cfg_path = rfile.parent / "config.yaml"
            if not cfg_path.exists():
                continue
            import yaml

            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            gain = cfg.get("gain", "unknown")
            e = json.load(open(rfile))["best_energy"]
            gain_vals.setdefault(gain, []).append(e)
        if gain_vals:
            gains = sorted(gain_vals.keys())
            for g in gains:
                m, s = _stats(gain_vals[g])
                color = gain_colors.get(g, "#999999")
                ax.bar(
                    g,
                    m,
                    yerr=s,
                    capsize=5,
                    color=color,
                    alpha=0.85,
                    edgecolor="black",
                    label=g,
                )
        ax.set_ylabel("Best Energy" if idx == 0 else "", fontsize=11)
        ax.set_title(f"{model}", fontsize=12)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("SAMC Gain Schedule Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig4_gain_schedule.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIG_DIR / 'fig4_gain_schedule.png'}")

    # ── Figure 5: Compute efficiency Pareto front ──
    fig, ax = plt.subplots(figsize=(8, 6))
    for model in MODELS:
        for algo in ALGOS:
            if algo not in data.get(model, {}):
                continue
            runs = data[model][algo]
            evals = [r.get("total_energy_evals", r.get("n_iters", 0)) for r in runs]
            energies = [r["best_energy"] for r in runs]
            m_eval, _ = _stats(evals)
            m_e, s_e = _stats(energies)
            marker = {"samc": "o", "mh": "s", "pt": "^"}[algo]
            ax.errorbar(
                m_eval,
                m_e,
                yerr=s_e,
                marker=marker,
                markersize=10,
                capsize=4,
                color=algo_colors[algo],
                label=f"{algo_labels[algo]} ({model})",
                linewidth=0,
                elinewidth=1.5,
            )
    ax.set_xlabel("Total Energy Evaluations", fontsize=12)
    ax.set_ylabel("Best Energy (mean +/- std)", fontsize=12)
    ax.set_title("Compute Efficiency Pareto Front", fontsize=14)
    ax.set_xscale("log")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_pareto_front.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIG_DIR / 'fig5_pareto_front.png'}")


def generate_summary_table(data: dict[str, dict[str, list[dict]]]) -> str:
    """Generate a markdown summary table."""
    model_dims = {"rosenbrock": 2, "10d": 10, "rastrigin": 20}
    lines = [
        "| Problem | Dim | Algorithm | Best Energy (mean +/- std) | Acc Rate | "
        "Wall Time (s) | Energy Evals |",
        "|---------|-----|-----------|---------------------------|----------|"
        "--------------|-------------|",
    ]
    for model in MODELS:
        dim = model_dims.get(model, "?")
        for algo in ALGOS:
            if algo not in data.get(model, {}):
                lines.append(f"| {model} | {dim} | {algo.upper()} | -- | -- | -- | -- |")
                continue
            runs = data[model][algo]
            energies = [r["best_energy"] for r in runs]
            accs = [r["acceptance_rate"] for r in runs]
            times = [r["wall_time"] for r in runs]
            evals = [r.get("total_energy_evals", r.get("n_iters", 0)) for r in runs]
            m_e, s_e = _stats(energies)
            m_a, _ = _stats(accs)
            m_t, _ = _stats(times)
            m_ev, _ = _stats(evals)
            lines.append(
                f"| {model} | {dim} | {algo.upper()} | "
                f"{m_e:.4f} +/- {s_e:.4f} | "
                f"{m_a:.3f} | "
                f"{m_t:.1f} | "
                f"{m_ev:,.0f} |"
            )
    return "\n".join(lines)


def generate_report(data: dict[str, dict[str, list[dict]]]):
    """Generate the final comparison report."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table = generate_summary_table(data)

    report = f"""# Cross-Algorithm Comparison Report

> **Setup**: Best hyperparameters per algorithm per problem, 10 seeds each
> **Algorithms**: SAMC, Metropolis-Hastings (MH), Parallel Tempering (PT)
> **Problems**: Rosenbrock 2D, 10D Gaussian Mixture, Rastrigin 20D
> **Date**: 2026-04-01

---

## Summary Table

{table}

---

## Best Configs Used

"""
    for model in MODELS:
        report += f"### {model}\n\n"
        for algo in ALGOS:
            if model in BEST_CONFIGS and algo in BEST_CONFIGS[model]:
                cfg = BEST_CONFIGS[model][algo]
                report += f"**{algo.upper()}**: `{cfg}`\n\n"
        report += "\n"

    report += """---

## Figures

- **Figure 1**: `ablation/figures/fig1_algo_comparison.png` -- Bar chart with error bars
- **Figure 2**: `ablation/figures/fig2_scaling_dimensionality.png` -- Scaling with dimensionality
- **Figure 3**: `ablation/figures/fig3_robustness_proposal_std.png` -- Robustness to proposal_std
- **Figure 4**: `ablation/figures/fig4_gain_schedule.png` -- SAMC gain schedule comparison
- **Figure 5**: `ablation/figures/fig5_pareto_front.png` -- Compute efficiency Pareto front

---

## Key Findings

_To be filled after results are available._
"""

    report_path = REPORT_DIR / "final_comparison.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved report: {report_path}")

    # Also export CSV
    csv_path = REPORT_DIR / "cross_comparison.csv"
    all_rows = []
    for model in MODELS:
        for algo in ALGOS:
            if algo not in data.get(model, {}):
                continue
            for run in data[model][algo]:
                row = {"model": model, "algo": algo}
                row.update(run)
                all_rows.append(row)
    if all_rows:
        keys = sorted({k for r in all_rows for k in r})
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved CSV: {csv_path}")


def analyze():
    """Load results, generate figures and report."""
    print("\n=== Loading cross-comparison results ===")
    data = load_results()

    total_runs = sum(len(runs) for m in data.values() for runs in m.values())
    print(f"Loaded {total_runs} results")
    for model in MODELS:
        for algo in ALGOS:
            n = len(data.get(model, {}).get(algo, []))
            if n > 0:
                print(f"  {model}/{algo}: {n} runs")

    if total_runs == 0:
        print("No results found. Run experiments first.")
        return

    print("\n=== Generating figures ===")
    generate_figures(data)

    print("\n=== Generating report ===")
    generate_report(data)


def main():
    parser = argparse.ArgumentParser(description="Cross-algorithm comparison (Step 28).")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--parallel", type=int, default=4, help="Concurrent processes")
    parser.add_argument(
        "--run-only", action="store_true", help="Only run experiments, skip analysis"
    )
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze, skip running")
    args = parser.parse_args()

    if not args.analyze_only:
        run_experiments(parallel=args.parallel, dry_run=args.dry_run)

    if not args.run_only and not args.dry_run:
        analyze()


if __name__ == "__main__":
    main()
