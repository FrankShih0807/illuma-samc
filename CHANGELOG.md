# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.0] - 2026-04-03

### Changed
- **Breaking:** `SAMC` multi-chain now defaults to independent weights (`shared_weights=False`). Pass `shared_weights=True` for previous behavior.
- **Breaking:** Default `bin_width` changed from 0.25 to 0.5 for both `SAMC` and `SAMCWeights`. Better performance on high-dimensional problems.
- Removed `AdaptivePartition`, `QuantilePartition`, `GrowingPartition` — only `UniformPartition` and `ExpandablePartition` remain.
- Removed `SAMCWeights.auto()` and `SAMCWeights.from_warmup()` factory methods — use `SAMCWeights()` with expand-on-demand instead.
- Simplified `benchmarks/three_way.py` to 3-way fair comparison (MH vs PT vs SAMC default).

### Added
- `shared_weights` parameter on `SAMC.__init__` to choose between independent and shared weight modes.
- 5-seed benchmark with tighter error bars.

## [0.2.0] - 2026-04-02

### Changed
- `SAMCWeights` now uses traditional `__init__` with zero-config defaults (lazy partition init on first energy seen)
- Default partition is `ExpandablePartition`: 201 uniform bins (100 above + center + 100 below first energy), expands on overflow
- `flatness()` now computes over visited bins only (avoids penalizing unreachable bins)
- Trimmed public API to `UniformPartition` and `ExpandablePartition` (removed `AdaptivePartition`, `QuantilePartition`)
- README rewritten for clarity: leads with practical value, simplified math, tighter FAQ

### Added
- `n_chains` parameter on `SAMC.__init__` for explicit multi-chain / batch size
- Batched tensor support in `SAMCWeights.correction()` and `.step()`
- Better error message when `energy_fn` returns non-scalar in single-chain mode
- Second reference: Liang (2009) in README and CITATION.bib

### Fixed
- `GrowingPartition` IndexError when partition expands mid-run (theta/counts now sync on resize)
- `flatness()` NaN on single visited bin

## [0.1.0] - 2026-04-01

### Added

#### Core
- `SAMC` sampler class with simple and fully-configurable modes
- `SAMCWeights` drop-in weight manager for existing MH loops (two-line integration)
- `SAMCResult` dataclass with best energy, acceptance rate, samples, and full history
- `GainSequence` with built-in schedules (`1/t`, `log`, `ramp`) and custom callable support
- Multi-chain sampling with shared or independent weights
- GPU (CUDA) support for single-chain and multi-chain sampling
- Input validation with descriptive errors for all public parameters
- Seed parameter on `SAMC.run()` for reproducibility
- Warnings for out-of-range initial states and low acceptance rates

#### Partitions
- `UniformPartition` with fixed energy range and equal-width bins
- `AdaptivePartition` that adjusts bin edges from observed energies
- `QuantilePartition` that places bin edges at energy quantiles
- `GrowingPartition` that adds bins on the fly (no energy range needed)
- `ExpandablePartition` that extends range when samples fall outside
- Overflow bins for robustness when energy range is misspecified
- Auto-range warmup (`from_warmup`) to discover energy range before fixed-bin sampling

#### Proposals
- `GaussianProposal` with configurable step size
- `LangevinProposal` (gradient-informed) with configurable step size

#### Diagnostics
- `plot_diagnostics` with 4-panel figure (weights, energy trace, bin visits, rolling acceptance)
- `plot_weight_diagnostics` for weight convergence analysis
- Energy mixing metrics and mode coverage analysis (`analysis.py`)

#### Benchmarks
- MH and PT baseline implementations (`baselines/` subpackage)
- `train.py` experiment CLI with YAML config system and full hyperparameter control
- `compare_results.py` for loading and ranking experiment results
- Structured output: `outputs/<model>/<algo>/<timestamp>/` with config, results, and plots
- Ablation sweep infrastructure (`ablation/sweep.py`, `ablation/analyze.py`)
- Ablation reports for 2D multimodal, Rosenbrock 2D, Gaussian 10D, Rastrigin 20D, and robust bins

#### CLI
- `--algo` flag for algorithm selection (samc, mh, pt)
- `--model` flag for problem selection (2d, 10d, rosenbrock_2d, rastrigin_20d)
- `--auto_range` flag for warmup-based energy range discovery
- `--growing` flag for growing partition mode
- `--n_chains` and `--shared_weights` flags for multi-chain control
- `--plot_energy` flag for energy trace visualization
- `--name` flag for named experiment output folders

#### Examples
- `demo_showcase.py` -- all-in-one feature showcase
- `gaussian_mixture.py` -- 4-mode Gaussian mixture demo
- `multimodal_2d.py` -- reproduces Liang's 2D experiment from the reference implementation
- `mh_vs_samc.ipynb` -- side-by-side MH vs SAMC comparison notebook
- `comparison_sample_code.py` -- validates API against reference `sample_code.py`

#### Problems
- 2D multimodal cost function (from Liang 2007)
- 10D Gaussian mixture (4 modes)
- 2D Rosenbrock (narrow valley)
- 20D Rastrigin (~10^20 local minima)
