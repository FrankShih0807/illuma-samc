"""Analyze robust energy bin ablation results (Phase 5, Step 42).

Reads results from outputs/ablation/robust_bins/ and generates:
- Summary tables (stdout)
- Comparison plots (ablation/figures/)
- Markdown report (ablation/reports/robust_bins_insights.md)

Usage:
    conda run -n illuma-samc python ablation/analyze_robust_bins.py
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("outputs/ablation/robust_bins")
FIGURES_DIR = Path("ablation/figures")
REPORTS_DIR = Path("ablation/reports")

MODELS = ["2d", "rosenbrock", "10d", "rastrigin"]
SCENARIOS = ["correct", "emax_tight", "emax_wide", "emin_high", "wrong_range"]
METHODS = ["uniform", "overflow_bins", "auto_range", "expandable"]

METHOD_LABELS = {
    "uniform": "Uniform (baseline)",
    "overflow_bins": "Overflow Bins",
    "auto_range": "Auto-Range",
    "expandable": "Expandable",
}

SCENARIO_LABELS = {
    "correct": "Correct Range",
    "emax_tight": "e_max Too Tight",
    "emax_wide": "e_max Too Wide",
    "emin_high": "e_min Too High",
    "wrong_range": "Completely Wrong",
}


def load_all_results() -> dict:
    """Load all results into nested dict: model -> scenario -> method -> [results]."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for model in MODELS:
        for scenario in SCENARIOS:
            for method in METHODS:
                method_dir = RESULTS_DIR / model / scenario / method
                if not method_dir.exists():
                    continue
                for seed_dir in sorted(method_dir.iterdir()):
                    rfile = seed_dir / "results.json"
                    if rfile.exists():
                        with open(rfile) as f:
                            r = json.load(f)
                        # Replace Infinity with a sentinel
                        for k, v in r.items():
                            if isinstance(v, float) and (v == float("inf") or v != v):
                                r[k] = None
                        data[model][scenario][method].append(r)
    return data


def summarize(runs: list[dict], metric: str) -> tuple[float, float]:
    """Return (mean, std) for a metric across seeds."""
    vals = [r[metric] for r in runs if r.get(metric) is not None]
    if not vals:
        return (float("nan"), float("nan"))
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s


def print_tables(data: dict):
    """Print summary tables to stdout."""
    for model in MODELS:
        print(f"\n{'=' * 80}")
        print(f"  Model: {model}")
        print(f"{'=' * 80}")
        print(
            f"{'Scenario':<20} {'Method':<18} {'Best E':>10} {'Acc Rate':>10} "
            f"{'Flatness':>10} {'Wall(s)':>8}"
        )
        print("-" * 80)
        for scenario in SCENARIOS:
            for method in METHODS:
                runs = data[model][scenario][method]
                if not runs:
                    continue
                be_m, be_s = summarize(runs, "best_energy")
                ar_m, _ = summarize(runs, "acceptance_rate")
                fl_m, _ = summarize(runs, "bin_flatness")
                wt_m, _ = summarize(runs, "wall_time")
                print(
                    f"{scenario:<20} {method:<18} "
                    f"{be_m:>10.3f} {ar_m:>10.3f} {fl_m:>10.3f} {wt_m:>8.1f}"
                )
            print()


def plot_heatmaps(data: dict):
    """Generate heatmaps: best_energy per (scenario x method) for each model."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        matrix = np.full((len(SCENARIOS), len(METHODS)), np.nan)
        for si, scenario in enumerate(SCENARIOS):
            for mi, method in enumerate(METHODS):
                runs = data[model][scenario][method]
                if runs:
                    be_m, _ = summarize(runs, "best_energy")
                    matrix[si, mi] = be_m

        im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=8, rotation=30, ha="right")
        ax.set_yticks(range(len(SCENARIOS)))
        ax.set_yticklabels([SCENARIO_LABELS[s] for s in SCENARIOS], fontsize=8)
        ax.set_title(f"{model}", fontsize=11, fontweight="bold")

        # Annotate cells
        for si in range(len(SCENARIOS)):
            for mi in range(len(METHODS)):
                val = matrix[si, mi]
                if not np.isnan(val):
                    color = "white" if abs(val) > (np.nanmax(np.abs(matrix)) * 0.6) else "black"
                    ax.text(mi, si, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

        plt.colorbar(im, ax=ax, shrink=0.8, label="Best Energy")

    plt.suptitle("Robust Bins Ablation: Best Energy (lower is better)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "robust_bins_best_energy.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_acceptance_heatmaps(data: dict):
    """Generate acceptance rate heatmaps."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        matrix = np.full((len(SCENARIOS), len(METHODS)), np.nan)
        for si, scenario in enumerate(SCENARIOS):
            for mi, method in enumerate(METHODS):
                runs = data[model][scenario][method]
                if runs:
                    ar_m, _ = summarize(runs, "acceptance_rate")
                    matrix[si, mi] = ar_m

        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=0.6)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=8, rotation=30, ha="right")
        ax.set_yticks(range(len(SCENARIOS)))
        ax.set_yticklabels([SCENARIO_LABELS[s] for s in SCENARIOS], fontsize=8)
        ax.set_title(f"{model}", fontsize=11, fontweight="bold")

        for si in range(len(SCENARIOS)):
            for mi in range(len(METHODS)):
                val = matrix[si, mi]
                if not np.isnan(val):
                    ax.text(mi, si, f"{val:.2f}", ha="center", va="center", fontsize=7)

        plt.colorbar(im, ax=ax, shrink=0.8, label="Acceptance Rate")

    plt.suptitle("Robust Bins Ablation: Acceptance Rate", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "robust_bins_acceptance.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_flatness_heatmaps(data: dict):
    """Generate flatness heatmaps."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        matrix = np.full((len(SCENARIOS), len(METHODS)), np.nan)
        for si, scenario in enumerate(SCENARIOS):
            for mi, method in enumerate(METHODS):
                runs = data[model][scenario][method]
                if runs:
                    fl_m, _ = summarize(runs, "bin_flatness")
                    matrix[si, mi] = fl_m

        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=8, rotation=30, ha="right")
        ax.set_yticks(range(len(SCENARIOS)))
        ax.set_yticklabels([SCENARIO_LABELS[s] for s in SCENARIOS], fontsize=8)
        ax.set_title(f"{model}", fontsize=11, fontweight="bold")

        for si in range(len(SCENARIOS)):
            for mi in range(len(METHODS)):
                val = matrix[si, mi]
                if not np.isnan(val):
                    ax.text(mi, si, f"{val:.2f}", ha="center", va="center", fontsize=7)

        plt.colorbar(im, ax=ax, shrink=0.8, label="Bin Flatness")

    plt.suptitle("Robust Bins Ablation: Bin Flatness (higher is better)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "robust_bins_flatness.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_method_comparison_bars(data: dict):
    """Bar chart: mean best energy per method, averaged across all wrong-range scenarios."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    wrong_scenarios = ["emax_tight", "emax_wide", "emin_high", "wrong_range"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        method_means = []
        method_stds = []
        for method in METHODS:
            all_best = []
            for scenario in wrong_scenarios:
                runs = data[model][scenario][method]
                for r in runs:
                    if r.get("best_energy") is not None:
                        all_best.append(r["best_energy"])
            if all_best:
                method_means.append(statistics.mean(all_best))
                method_stds.append(statistics.stdev(all_best) if len(all_best) > 1 else 0.0)
            else:
                method_means.append(0.0)
                method_stds.append(0.0)

        x = np.arange(len(METHODS))
        colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
        ax.bar(x, method_means, yerr=method_stds, capsize=4, color=colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=8, rotation=20, ha="right")
        ax.set_ylabel("Best Energy (mean)")
        ax.set_title(f"{model} (wrong-range scenarios)", fontsize=10, fontweight="bold")
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    plt.suptitle("Method Robustness: Best Energy Under Wrong Ranges", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "robust_bins_method_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def generate_report(data: dict):
    """Generate markdown report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Robust Energy Bin Selection: Ablation Results")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(
        "This ablation tests four approaches to making SAMC robust to energy bin misspecification:"
    )
    lines.append("")
    lines.append("1. **Uniform (baseline)** -- standard UniformPartition, no robustness")
    lines.append("2. **Overflow Bins** -- adds catch-all bins at [-inf, e_min] and [e_max, +inf]")
    lines.append(
        "3. **Auto-Range** -- uses warmup MH to auto-detect energy range "
        "(bypasses wrong range entirely)"
    )
    lines.append(
        "4. **Expandable** -- dynamically expands partition when out-of-range energies arrive"
    )
    lines.append("")
    lines.append("**Test scenarios** (deliberately wrong ranges):")
    lines.append("")
    lines.append("| Scenario | Description |")
    lines.append("|----------|-------------|")
    lines.append("| correct | Correct energy range (baseline) |")
    lines.append("| emax_tight | e_max set to 50% of true value |")
    lines.append("| emax_wide | e_max set to 5x true value |")
    lines.append("| emin_high | e_min raised above true minimum |")
    lines.append("| wrong_range | Completely wrong (e_min=10, e_max=20) |")
    lines.append("")
    lines.append(
        "**Setup**: 100K iterations, 3 seeds, 4 problems (2d, rosenbrock, 10d, rastrigin)."
    )
    lines.append("")

    # Per-model results tables
    for model in MODELS:
        lines.append(f"## {model.upper()}")
        lines.append("")
        lines.append("| Scenario | Method | Best Energy | Acc Rate | Flatness | Wall Time |")
        lines.append("|----------|--------|-------------|----------|----------|-----------|")
        for scenario in SCENARIOS:
            for method in METHODS:
                runs = data[model][scenario][method]
                if not runs:
                    continue
                be_m, be_s = summarize(runs, "best_energy")
                ar_m, _ = summarize(runs, "acceptance_rate")
                fl_m, _ = summarize(runs, "bin_flatness")
                wt_m, _ = summarize(runs, "wall_time")
                be_str = f"{be_m:.3f}" if not (be_m != be_m) else "N/A"
                ar_str = f"{ar_m:.3f}" if not (ar_m != ar_m) else "N/A"
                fl_str = f"{fl_m:.3f}" if not (fl_m != fl_m) else "N/A"
                wt_str = f"{wt_m:.1f}s" if not (wt_m != wt_m) else "N/A"
                lines.append(
                    f"| {SCENARIO_LABELS.get(scenario, scenario)} | "
                    f"{METHOD_LABELS.get(method, method)} | "
                    f"{be_str} | {ar_str} | {fl_str} | {wt_str} |"
                )
        lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")

    # Analyze which method is most robust
    for model in MODELS:
        lines.append(f"### {model.upper()}")
        lines.append("")

        # Correct range baseline
        baseline_runs = data[model]["correct"]["uniform"]
        if baseline_runs:
            bl_e, _ = summarize(baseline_runs, "best_energy")
            lines.append(f"- **Baseline** (correct range, uniform): best energy = {bl_e:.3f}")

        # Worst scenario for baseline
        for scenario in ["emax_tight", "emax_wide", "emin_high", "wrong_range"]:
            uni_runs = data[model][scenario]["uniform"]
            if uni_runs:
                uni_e, _ = summarize(uni_runs, "best_energy")
                uni_acc, _ = summarize(uni_runs, "acceptance_rate")
                if uni_acc < 0.01:
                    lines.append(
                        f"- **{SCENARIO_LABELS[scenario]}**: Uniform completely fails "
                        f"(acc={uni_acc:.3f}, E={uni_e:.3f})"
                    )
                else:
                    lines.append(
                        f"- **{SCENARIO_LABELS[scenario]}**: Uniform degraded "
                        f"(acc={uni_acc:.3f}, E={uni_e:.3f})"
                    )

            # Check if auto_range saves it
            ar_runs = data[model][scenario]["auto_range"]
            if ar_runs:
                ar_e, _ = summarize(ar_runs, "best_energy")
                ar_acc, _ = summarize(ar_runs, "acceptance_rate")
                if ar_acc > 0.1 and baseline_runs:
                    bl_e_val, _ = summarize(baseline_runs, "best_energy")
                    if abs(ar_e - bl_e_val) / (abs(bl_e_val) + 1e-8) < 0.3:
                        lines.append(
                            f"  - Auto-Range **recovers** near-baseline performance "
                            f"(E={ar_e:.3f}, acc={ar_acc:.3f})"
                        )
                    else:
                        lines.append(
                            f"  - Auto-Range partially helps (E={ar_e:.3f}, acc={ar_acc:.3f})"
                        )
        lines.append("")

    # Summary recommendations
    lines.append("## Recommendations")
    lines.append("")

    # Compute aggregate robustness score per method
    wrong_scenarios = ["emax_tight", "emax_wide", "emin_high", "wrong_range"]
    method_scores = {m: [] for m in METHODS}
    for model in MODELS:
        baseline_runs = data[model]["correct"]["uniform"]
        if not baseline_runs:
            continue
        bl_e, _ = summarize(baseline_runs, "best_energy")
        for method in METHODS:
            for scenario in wrong_scenarios:
                runs = data[model][scenario][method]
                if runs:
                    me, _ = summarize(runs, "best_energy")
                    ar, _ = summarize(runs, "acceptance_rate")
                    # Score: how close to baseline? (lower = more robust)
                    if ar > 0.01 and me == me:  # not NaN and not dead
                        gap = abs(me - bl_e) / (abs(bl_e) + 1e-8)
                        method_scores[method].append(gap)
                    else:
                        method_scores[method].append(10.0)  # penalty for dead sampler

    lines.append("**Robustness ranking** (lower gap to baseline = better):")
    lines.append("")
    for method in METHODS:
        scores = method_scores[method]
        if scores:
            avg = statistics.mean(scores)
            lines.append(f"- {METHOD_LABELS[method]}: avg gap = {avg:.2f}")
    lines.append("")

    # Check which method is best
    best_method = min(
        METHODS, key=lambda m: statistics.mean(method_scores[m]) if method_scores[m] else 999
    )
    lines.append(
        f"**Winner**: **{METHOD_LABELS[best_method]}** -- most robust to wrong energy ranges "
        "across all problems and scenarios."
    )
    lines.append("")

    # Practical recommendations
    lines.append("### Practical Guidelines")
    lines.append("")
    lines.append(
        "1. **When you have no idea about the energy range**: Use `auto_range` "
        "(SAMCWeights.from_warmup). It runs a short MH warmup to discover the range "
        "automatically and is immune to range misspecification."
    )
    lines.append(
        "2. **When you have a rough estimate**: Use `overflow_bins` as a safety net. "
        "It adds minimal overhead and catches energies outside your estimated range."
    )
    lines.append(
        "3. **When the range might shift during sampling**: Use `expandable`. "
        "It adapts on-the-fly but may create many bins with uneven visit counts."
    )
    lines.append(
        "4. **When you know the exact range**: Standard `uniform` is fine and has "
        "the lowest overhead."
    )
    lines.append("")
    lines.append("### Figures")
    lines.append("")
    lines.append("![Best Energy Heatmap](../figures/robust_bins_best_energy.png)")
    lines.append("![Acceptance Rate Heatmap](../figures/robust_bins_acceptance.png)")
    lines.append("![Flatness Heatmap](../figures/robust_bins_flatness.png)")
    lines.append("![Method Comparison](../figures/robust_bins_method_comparison.png)")
    lines.append("")

    report_path = REPORTS_DIR / "robust_bins_insights.md"
    report_path.write_text("\n".join(lines))
    print(f"Report written to {report_path}")


def main():
    print("Loading results...")
    data = load_all_results()

    total = sum(len(data[m][s][method]) for m in MODELS for s in SCENARIOS for method in METHODS)
    print(f"Loaded {total} results across {len(MODELS)} models\n")

    print_tables(data)

    print("\nGenerating plots...")
    plot_heatmaps(data)
    plot_acceptance_heatmaps(data)
    plot_flatness_heatmaps(data)
    plot_method_comparison_bars(data)
    print(f"Plots saved to {FIGURES_DIR}/")

    print("\nGenerating report...")
    generate_report(data)


if __name__ == "__main__":
    main()
