# illuma-samc TODO

> Child Claude: work through these in order. Each task is a checkpoint — format, lint, test, commit, push after completing each one.

## Phase 1: Core MVP

### Step 1: Project Setup
- [x] Set up conda environment `illuma-samc` with Python 3.11, PyTorch, ruff, pytest
- [x] Finalize `pyproject.toml` with all dependencies (torch, matplotlib optional, tqdm optional)
- [x] Create `CITATION.bib` with Liang 2006 SAMC paper (JASA)
- [x] Verify `pip install -e ".[dev]"` works
- [x] Run `ruff check .` and `pytest` (empty test file is fine)

### Step 2: Gain Sequences
- [x] Implement `GainSequence` class in `src/illuma_samc/gain.py`
- [x] Support `"1/t"` schedule: γ_t = t0 / max(t, t0)
- [x] Support `"log"` schedule: γ_t = t0 / max(t * log(t+e), t0)
- [x] Support `"ramp"` schedule matching `sample_code.py`: rho * exp(-tau * log((t - offset) / step_scale)) with warmup phase
- [x] Support custom callable: any `(int) → float`
- [x] Add tests in `tests/test_gain.py`

### Step 3: Partitions
- [x] Implement `UniformPartition` in `src/illuma_samc/partitions.py` — linear energy binning matching `sample_code.py` style (e_min, e_max, n_bins)
- [x] Implement `AdaptivePartition` — recomputes boundaries from visited energies
- [x] Implement `QuantilePartition` — boundaries from warmup energy sample
- [x] All partitions have `.assign(energy: Tensor) → int` and `.n_partitions` property
- [x] Add tests in `tests/test_partitions.py`

### Step 4: Proposals
- [x] Implement `GaussianProposal` in `src/illuma_samc/proposals.py` — isotropic random walk with configurable step size
- [x] Implement `LangevinProposal` — MALA-style using autograd, with correct log proposal ratio
- [x] Both have `.propose(x) → x_new` and `.log_ratio(...)` methods
- [x] Add tests in `tests/test_proposals.py`

### Step 5: Core SAMC Sampler
- [x] Implement `SAMC` class in `src/illuma_samc/sampler.py` with two modes:
- [x] **Simple mode**: `SAMC(energy_fn=..., dim=..., n_partitions=...)` — user provides energy function, sampler handles proposal/partition/acceptance internally
- [x] **Flexible mode**: `SAMC(dim=..., n_partitions=..., proposal_fn=..., log_accept_fn=..., partition_fn=...)` — user provides custom functions for full control
- [x] Core loop matches `sample_code.py` logic: propose → compute acceptance with SAMC weight correction → accept/reject → update theta weights on EVERY iteration
- [x] `sampler.run(n_steps, x0)` returns `torch.Tensor` of samples
- [x] Track: `log_weights`, `acceptance_rate`, `energy_history`, `bin_counts`, `best_x`, `best_energy`
- [x] Support `device` parameter for GPU
- [x] Add progress bar via tqdm (optional)
- [x] Add tests in `tests/test_sampler.py` — test both simple and flexible modes, verify they produce equivalent results on the same problem

### Step 6: Diagnostics
- [x] Implement `diagnostics.py`: `plot_diagnostics(sampler)` function
- [x] Plot weight trajectory (theta over iterations)
- [x] Plot energy trace
- [x] Plot bin visit histogram
- [x] Plot acceptance rate over time (rolling window)
- [x] `sampler.plot_diagnostics()` convenience method
- [x] Add tests (just verify no errors, not plot correctness)

### Step 7: Examples & Verification
- [x] Create `examples/multimodal_2d.py` — reproduce `sample_code.py` results using the illuma-samc API (both SAMC and MH comparison)
- [x] Create `examples/gaussian_mixture.py` — classic multimodal Gaussian demo
- [x] Verify SAMC finds global minimum on the 2D cost function from `sample_code.py`
- [x] Verify weight vector converges to approximately flat on Gaussian mixture
- [x] Update `__init__.py` exports

### Step 8: README & Attribution
- [x] Create `README.md` with: Illuma branding, prominent Liang attribution ("Based on SAMC by Faming Liang, JASA 2006"), install instructions, quick start (simple + flexible API), example output plot
- [x] Update `PROJECT_REGISTRY.md` in root workspace to include illuma-samc

## Phase 2: GPU & Benchmarks

### Step 9: GPU Support
- [x] Ensure single-chain SAMC runs on CUDA when `device="cuda"` — energy_fn, proposals, and state all on GPU. Theta weights stay on CPU (small vector, updated every step).
- [x] NOTE: SAMC is inherently sequential (weight update depends on current theta). Do NOT implement naive batch parallelism — it would break correctness.
- [x] Implement **parallel chains with shared weights**: run N chains simultaneously, batch the energy evaluation (the expensive part), but synchronize theta updates after each step. API: `sampler.run(n_steps=10000, x0=torch.randn(N, dim))` where N is number of chains.
- [x] Each chain proposes independently, energy is evaluated as a batch, then each chain's accept/reject and weight update happens sequentially on the shared theta.
- [x] Return shape: `(N, n_steps, dim)` for multi-chain, `(n_steps, dim)` for single chain.
- [x] Add tests: verify GPU results match CPU results (same seed), verify multi-chain shared weights converge same as single chain.

### Step 9.5: Verify Correctness Against sample_code.py
- [x] Run `python sample_code.py` and save the output PNG (`samc_experiment.png`) as the ground truth
- [x] Run `python examples/multimodal_2d.py` with the same parameters (1M iterations, 42 bins, same gain schedule)
- [x] Compare results: the illuma-samc version MUST achieve flat bin visit histogram (all bins visited roughly equally) and find the global minimum (best energy ≈ -8.2)
- [x] If bin visits are NOT flat or best energy is significantly worse, debug and fix the core SAMC loop — the weight update or acceptance ratio is likely wrong
- [x] Save comparison plots side by side

### Step 10: Benchmarks
- [x] Create `benchmarks/vs_mh_pt.py` comparing SAMC vs MH vs parallel tempering on:
  - 2D multimodal cost function from `sample_code.py`
  - 10D Gaussian mixture (well-separated modes)
- [x] Metrics: best energy found, ESS (effective sample size), acceptance rate, wall-clock time
- [x] Generate comparison plots (save as PNG)
- [x] Add benchmark results to README

## Phase 2.5: Demo, Visualization & Experiment Infrastructure

### Step 11: Comprehensive Demo & README Visuals
- [x] Run `python sample_code.py` and verify `samc_experiment.png` shows flat bin visits and global minimum found
- [x] Run all existing examples and verify they work: `examples/multimodal_2d.py`, `examples/gaussian_mixture.py`
- [x] Create `examples/demo_showcase.py` — a polished all-in-one demo that:
  - Shows Simple API usage (energy-based, Gaussian mixture with 5+ modes)
  - Shows Flexible API usage (custom proposal and acceptance)
  - Generates a single high-quality figure with subplots: (1) target distribution contour, (2) SAMC samples overlaid, (3) weight convergence over time, (4) bin visit histogram showing flat exploration, (5) SAMC vs MH comparison showing MH stuck in one mode while SAMC explores all
  - Save as `assets/demo_showcase.png` (300 DPI, publication quality)
- [x] Create `assets/` directory for README images
- [x] Update `README.md` to embed `assets/demo_showcase.png` prominently — this is the selling point visual
- [x] Add a "Why SAMC?" section to README showing the SAMC vs MH comparison: "MH gets stuck. SAMC doesn't."
- [x] Run full test suite: `pytest -v` and ensure 100% pass
- [x] Commit and push

### Step 12: Fix README Benchmark Section
- [x] Remove ESS column from the benchmark results table in README.md
- [x] Update benchmark numbers with current results (proposal_std=0.05 for 2D, e_max=20 for 10D)
- [x] Remove ESS-related "key takeaways" text
- [x] Rerun `python benchmarks/vs_mh_pt.py` to regenerate plots, verify README matches

### Step 13: Normalize PT Benchmark Fairness
- [x] PT runs 8 replicas per iteration = 8x energy evaluations vs SAMC/MH
- [x] Add "Total Energy Evals" column to benchmark table to make compute cost transparent
- [x] Update README table and analysis text accordingly
- [x] Rerun benchmarks and verify

### Step 14: Experiment CLI (`train.py`)
- [x] Create `train.py` at project root with argparse CLI
- [x] `python train.py --algo samc --model 2d --proposal_std 0.05 ...`
- [x] Supported `--algo` choices: `samc`, `mh`, `pt`
- [x] Supported `--model` choices: `2d`, `10d` (extensible to new models)
- [x] All hyperparams as CLI args (proposal_std, n_iters, n_partitions, e_min, e_max, gain schedule, etc.)
- [x] CLI args override YAML config defaults (Step 15)
- [x] Each invocation runs a single experiment — cluster-ready (one run per command)
- [x] Prints summary metrics on completion

### Step 15: YAML Config System
- [x] Create `configs/` directory
- [x] `configs/samc.yaml` — default tuned settings keyed by model
- [x] `configs/mh.yaml` — MH defaults keyed by model (proposal_std, temperature, n_iters)
- [x] `configs/pt.yaml` — PT defaults keyed by model (n_replicas, t_min, t_max, swap_interval, etc.)
- [x] `train.py` loads the appropriate YAML, then CLI args override any matching keys

### Step 16: Output & Results Management
- [x] Output folder structure: `outputs/<model>/<algo>/<timestamp>/`
- [x] Each run saves into its timestamped folder:
  - `config.yaml` — full config snapshot (YAML defaults + CLI overrides merged)
  - `results.json` — best energy, acceptance rate, wall time, total energy evals
  - Diagnostic plots (energy trace, weight trajectory if SAMC, etc.)
- [x] Comparison utility: load all runs for a given model, rank by best energy, print comparison table
- [x] Example usage: `python compare_results.py --model 2d` prints ranked table of all runs

### Step 17: Update WORKLOG
- [x] Log partition boundary fix, ESS removal, benchmark tuning from current session
- [x] Log experiment infrastructure additions

## Phase 2.75: Bug Fixes, Tests & UX (after Phase 2.5)

> Address bugs, missing tests, and UX issues found during code review.

### Bug-Fix Procedure

Every bug follows this pipeline. Do NOT skip steps or batch multiple bugs into one commit.

```
For each bug:
  1. REPRODUCE  — Write a minimal script or test that triggers the bug. Confirm it fails.
  2. TEST       — Add a failing test to the test suite that captures the expected behavior.
                  Run `pytest` — the new test MUST fail (red).
  3. FIX        — Implement the minimal fix in the source code. Do not refactor unrelated code.
  4. VERIFY     — Run `pytest` — ALL tests must pass (green), including the new one.
  5. REGRESS    — Run the full test suite (`pytest -v`) to confirm no regressions.
  6. COMMIT     — `ruff format . && ruff check . && pytest` then commit with message:
                  "Fix: <one-line description of bug>"
                  Include which file(s) changed and why in the commit body.
```

For UX improvements and input validation, the same pipeline applies:
- Step 1 becomes "demonstrate the bad UX" (e.g., show the unhelpful error)
- Step 2 becomes "write a test for the expected behavior" (e.g., `pytest.raises(ValueError)`)

### Step 18: Bug Fixes
- [x] Bug A: Out-of-range samples saved with wrong bin index — fixed `max(cur_bin, 0)` in single/multi-chain
- [x] Bug B: `importance_weights` NaN when all -inf — returns zeros with warning
- [x] Bug C: AdaptivePartition unbounded memory — uses deque(maxlen=50_000)
- [x] Bug D: AdaptivePartition records out-of-range energies — only appends in-range
- [x] Bug E: UniformPartition.edges allocates per call — cached in __init__

### Step 19: Input Validation
- [x] `e_min >= e_max` → ValueError
- [x] `n_bins <= 0` → ValueError
- [x] `dim <= 0` → ValueError
- [x] `proposal_std <= 0` → ValueError
- [x] `n_steps <= 0` in `run()` → ValueError

### Step 20: UX Warnings & Improvements
- [x] Warn on out-of-range initial state
- [x] Warn on low acceptance rate (< 1%)
- [x] `seed` parameter on `run()`
- [x] `edges` abstract property on base `Partition` class
- [x] `n_bins` alias in SAMC constructor

### Step 21: Additional Test Coverage
- [x] Energy function returning (energy, in_region) tuple
- [x] Very short run (n_steps < save_every)
- [x] save_every=1
- [x] LangevinProposal end-to-end
- [x] Multi-chain `plot_diagnostics` (found and fixed crash)

### Step 22: Codebase Restructure
- [x] Extract energy functions into `src/illuma_samc/problems/` (multimodal_2d, gaussian_10d, registry)
- [x] Extract baselines into `src/illuma_samc/baselines/` (metropolis_hastings, parallel_tempering)
- [x] Slim down `benchmarks/vs_mh_pt.py` to orchestration + plotting only
- [x] Update `train.py` to import from new modules
- [x] Update `benchmarks/debug_10d.py` to import from new modules
- [x] Update `__init__.py` with subpackage re-exports
- [x] All tests pass, ruff clean, CLI commands verified

## Phase 3: Ablation Study & Production Hardening

> Progressive difficulty: learn tuning intuitions on cheap models, apply to harder ones.

### Step 23: Infrastructure Prep (3A)
- [x] Add Rosenbrock 2D problem (`src/illuma_samc/problems/rosenbrock_2d.py`) — narrow curved valley, global min at (1,1) with E=0
- [x] Add Rastrigin 20D problem (`src/illuma_samc/problems/rastrigin_20d.py`) — ~10^20 local minima, global min at origin with E=0
- [x] Register new problems in `problems/__init__.py` and `train.py` MODELS dict
- [x] Add default configs for `rosenbrock` and `rastrigin` models in `configs/samc.yaml`, `mh.yaml`, `pt.yaml`
- [x] Add mode coverage metric: count distinct modes visited per problem (post-hoc analysis)
- [x] Add bin visit flatness metric: `1 - std(bin_counts) / mean(bin_counts)`, standardize in results.json
- [x] Create `ablation/sweep.py` — reads YAML sweep spec, generates `train.py` commands, supports `--dry-run` and `--parallel N`
- [x] Create `ablation/analyze.py` — loads results from a group dir, computes derived metrics, generates plots + CSV
- [x] Add `--partition_type {uniform,adaptive,quantile}`, `--n_chains`, `--gain_t0` args to `train.py`
- [x] Verify infra with a quick smoke test: 3 algos x 2 models x 1 seed

### Step 24: 2D Multimodal Ablations (3B) — cheapest, full sweeps
> Run ALL ablation groups on 2D only. Each group varies ONE factor, 3 seeds each.

- [x] SAMC gain schedule: 1/t, log, ramp (9 runs)
- [x] SAMC gain t0: 100, 500, 1K, 5K, 10K with gain=1/t (15 runs)
- [x] SAMC n_bins: 10, 20, 42, 80, 150 (15 runs)
- [x] SAMC energy range: e_max in {-2, 0, 5, 10, 20} with e_min=-8.2 (15 runs)
- [x] SAMC proposal_std: 0.01, 0.05, 0.1, 0.5, 1.0, 2.0 (18 runs)
- [x] SAMC partition type: uniform, adaptive, quantile (9 runs)
- [x] SAMC multi-chain: 1, 2, 4, 8 chains (12 runs)
- [x] MH proposal_std: 0.01, 0.05, 0.1, 0.5, 1.0, 2.0 (18 runs)
- [x] MH temperature: 0.1, 0.5, 1.0, 2.0, 5.0 (15 runs)
- [x] PT n_replicas: 2, 4, 8, 16, 32 (15 runs)
- [x] PT t_max: 2, 5, 10, 20, 50 (15 runs)
- [x] PT swap_interval: 1, 5, 10, 50, 100 (15 runs)
- [x] **Deliverable**: `ablation/reports/2d_insights.md` — sensitivity ranking, optimal ranges, tuning heuristics, SAMC vs MH vs PT analysis

### Step 25: Rosenbrock 2D Ablations (3C) — cheap, different geometry
> Apply 2D heuristics as starting defaults. Narrower sweeps guided by Step 24 insights.

- [x] Set Rosenbrock defaults from 2D insights
- [x] Run narrowed sweeps on most impactful params (guided by 2D sensitivity ranking)
- [x] **Deliverable**: `ablation/reports/rosenbrock_insights.md` — validate/refine heuristics, note where narrow valley breaks assumptions

### Step 26: 10D Gaussian Mixture Ablations (3D) — dimensionality scaling
> Focus on params that mattered in 2D. Add scaling-specific questions.

- [x] Apply refined heuristics to set 10D defaults
- [x] Sweep only high-impact params from 2D analysis
- [x] Test scaling questions: does n_bins scale with dim? does proposal_std ~ 1/sqrt(dim)?
- [x] **Deliverable**: `ablation/reports/10d_insights.md` — scaling rules, dimensionality-specific heuristics

### Step 27: Rastrigin 20D Ablations (3E) — hardest, minimal sweeps
> Apply all accumulated heuristics. Only sweep 2-3 most impactful params.

- [x] Apply all heuristics to set Rastrigin defaults
- [x] Focus: can SAMC find global min? How many chains needed? Does gain schedule matter at this scale?
- [x] **Deliverable**: `ablation/reports/rastrigin_insights.md` — limits of SAMC, where it breaks down

### Step 28: Cross-Algorithm Comparison (3F)
- [x] Best config per algo per problem (from ablation results), 10 seeds for tight error bars
- [x] Figure 1: Algo comparison at best hyperparams (bar chart with error bars)
- [x] Figure 2: Scaling with dimensionality (2D → 10D → 20D)
- [x] Figure 3: Robustness to proposal_std (SAMC vs MH overlay)
- [x] Figure 4: SAMC gain schedule convergence curves
- [x] Figure 5: Compute efficiency Pareto front (energy evals vs best energy)
- [x] Summary table for README: problem x algo x metrics
- [x] **Deliverable**: `ablation/reports/final_comparison.md` + all figures in `ablation/figures/`

### Step 29: Production Hardening (after ablation insights)
> Informed by ablation results — implement the features that actually matter.

- [x] Adaptive proposal tuning: dual-averaging on GaussianProposal, `adapt_proposal=True` on SAMC, converges to ~0.05-0.11 from any starting point
- [x] Update README with ablation results, tuning guide, and recommendations
- [x] MPS compatibility: theta/counts moved to CPU (MPS lacks float64), 6 smoke tests passing (single/multi-chain, Langevin, adaptive, SAMCWeights)

## Phase 4: SAMCWeights Product Polish

> SAMCWeights is the main product. These improvements make it production-ready.

### Step 30: CI/CD (P0)
- [x] Add GitHub Actions workflow: pytest + ruff on every push/PR
- [x] Add badge to README (build status)

### Step 31: Bin Visit History in SAMCWeights (P0)
- [x] Add `bin_counts_history` to `SAMCWeights` — record `counts.clone()` every N steps (configurable `record_every` param, default=100)
- [x] Add `flatness_history()` method — returns flatness at each recorded snapshot
- [x] Lightweight: only store snapshots, not per-iteration (memory bounded)
- [x] Update `state_dict` / `load_state_dict` to include history
- [x] Add tests for history recording and flatness convergence curve

### Step 32: Diagnostics for SAMCWeights (P1)
- [x] Add `plot_diagnostics()` method or standalone function that works with `SAMCWeights`
- [x] Panels: bin visit histogram, flatness over time (from Step 31), theta trajectory, theta bar chart
- [x] Users should not need to build these plots manually from `wm.counts` and `wm.theta`
- [x] Add tests (no-error smoke tests)

### Step 33: Vectorize `importance_log_weights` (P1)
- [x] Replace Python loop with vectorized `partition.assign_batch` (added to Partition base class + UniformPartition)
- [x] Should handle 500K+ samples without Python-loop bottleneck
- [x] Add benchmark test to verify speedup
- [x] Same for `importance_weights` and `resample` (they call `importance_log_weights`)

### Step 34: Acceptance Rate Guidance (P1)
- [x] Add warning in `SAMCWeights.step()` if acceptance rate (tracked internally) falls outside 15-50% after warmup
- [x] Add `tracked_acceptance_rate` property for user inspection
- [x] Keep it advisory — SAMCWeights doesn't own the proposal, just nudges the user

### Step 35: High-Dim Benchmarks (P2)
- [x] Add 50D and 100D Gaussian mixture problems
- [x] Run SAMC vs MH vs PT with 3 seeds each on 10D/50D/100D
- [x] SAMC 4.5x better than MH at 50D, 5.5x at 100D; results in README

### Step 36: README & Discoverability (P2)
- [x] Link `examples/mh_vs_samc.ipynb` from README (already linked in two places)
- [x] Add "Why SAMC?" conceptual section (already present as "How It Works" section)

### Step 37: API Docs & Config (P3)
- [x] Generate API docs with Sphinx, `.readthedocs.yaml` configured, builds cleanly
- [x] `SAMCConfig` dataclass: `from_yaml()`, `build()` → SAMCWeights, `build_sampler()` → SAMC

## Phase 5: Robust Energy Bin Selection

> Energy range (e_min/e_max) is SAMC's #1 failure mode — wrong range = dead sampler.
> Three approaches to make SAMC robust, then ablate them.

### Step 39: Overflow Bins
- [x] Add `overflow_bins=False` parameter to `UniformPartition` (backward-compatible)
- [x] Two catch-all bins: `[-inf, e_min]` (bin 0) and `[e_max, +inf]` (last bin)
- [x] Update `assign()`, `assign_batch()`, `edges`, `n_partitions`
- [x] SAMCWeights works with overflow bins
- [x] Tests in `tests/test_partitions.py`

### Step 40: Auto-Range Warmup
- [x] Add `SAMCWeights.from_warmup()` classmethod
- [x] Runs warmup MH steps with no weight correction to estimate energy range
- [x] Sets e_min/e_max from observed min/max with margin
- [x] Tests in `tests/test_weight_manager.py`

### Step 41: Dynamic Bin Expansion
- [x] Add `ExpandablePartition` in `partitions.py`
- [x] Extends range when out-of-range energy arrives, up to `max_bins`
- [x] `SAMCWeights._resize_for_partition()` handles theta/counts resize
- [x] Tests in `tests/test_partitions.py`

### Step 42: Ablation — Robust Bins
- [x] Add `--overflow_bins`, `--auto_range`, `--expandable` flags to `train.py`
- [x] Run ablations on all 4 problems x 5 scenarios x 4 methods x 3 seeds (240 runs)
- [x] Write analysis to `ablation/reports/robust_bins_insights.md`

## Phase 5.5: Robust Defaults Validation & README Comparison

> Validate robust defaults, then produce a fair 4-way comparison for README.
> SAMC class deferred partition now aligned with SAMCWeights defaults:
> bin_width=0.5, n_bins_per_side=100, max_bins=1000.

### Step 43: Create Benchmark Script (Worker)

Create `ablation/robust_defaults.py` that runs two configs per problem:

**Config A — "Robust Defaults" (zero tuning):**
Uses `SAMC` class with `adapt_proposal=True` and NO explicit `e_min`/`e_max` (auto-range warmup).
User only specifies `energy_fn`, `dim`, and `n_chains=4`. Everything else is default.

```python
sampler = SAMC(
    energy_fn=fn, dim=dim, n_chains=4,
    adapt_proposal=True, adapt_warmup=2000,
    # no e_min, e_max, proposal_std, gain — all defaults
)
```

**Config B — "Hand-Tuned Ablation Winner" (from ablation reports):**
Best SAMC config per problem from Steps 24-28:

| Problem | proposal_std | gain | n_partitions | e_min | e_max | n_chains | shared |
|---------|-------------|------|-------------|-------|-------|----------|--------|
| 2d | 0.1 | ramp | 42 | -8.2 | 0.0 | 4 | shared |
| rosenbrock | 0.1 | ramp | 30 | 0.0 | 500 | 8 | independent |
| 10d | 1.0 | ramp | 30 | 0.0 | 20 | 4 | shared |
| rastrigin | 0.5 | ramp | 40 | 0.0 | 500 | 16 | shared |
| 50d | 0.5 | ramp | 40 | 0.0 | 60 | 4 | shared |
| 100d | 0.3 | ramp | 50 | 0.0 | 60 | 4 | shared |

**Run parameters:**
- 5 seeds: 42, 123, 456, 789, 999
- Iterations: 500K for 2d/rosenbrock, 200K for 10d/50d/100d/rastrigin
- Metrics per run: best_energy, acceptance_rate, flatness, final_proposal_std (for adaptive), wall_time

**Output:** Save all results to `ablation/outputs/robust_defaults/` as JSON per run.

Worker checklist:
- [x] Create `ablation/robust_defaults.py` with both configs
- [x] Verify it runs on 2d with 1 seed (smoke test) before full run
- [x] Run all 6 problems x 2 configs x 5 seeds = 60 runs
- [x] Print summary table to stdout

### Step 44: Analyze Results and Write Report (Worker)

Create `ablation/analyze_robust_defaults.py` that:
- [x] Loads all JSON results from `ablation/outputs/robust_defaults/`
- [x] Computes mean +/- std per (problem, config) group
- [x] Generates comparison table: robust defaults vs hand-tuned per problem
- [x] Computes "gap" = (robust_energy - tuned_energy) / |tuned_energy| as % degradation
- [x] Writes `ablation/reports/robust_defaults.md` with:
  - Summary table (problem x config x metrics)
  - Per-problem analysis: where does auto-tuning match? Where does it fall short?
  - Final verdict: what (if anything) do users still need to tune?
  - Recommended "minimal tuning" guide: which 1-2 params matter if defaults aren't enough

Worker checklist:
- [x] Script runs cleanly on the Step 43 outputs
- [x] Report includes all 6 problems
- [x] Report has a clear verdict section

### Step 45: Update README Tuning Guide (Worker)

Based on the Step 44 report:
- [x] If robust defaults match within 10%: simplify tuning guide to "just use defaults"
- [x] If some problems need tuning: list only the params that matter
- [x] Update the sensitivity ranking to reflect that #1 and #2 are now auto-handled
- [x] Add a "Zero-Config Quick Start" example using robust defaults
- [x] Remove or demote advice that's no longer relevant

Worker checklist:
- [x] README tuning section is updated
- [x] No stale advice remains from pre-robust ablations

### Step 46: Inspector Review

Inspector verifies:
- [x] `ablation/robust_defaults.py` uses correct hand-tuned configs (cross-check against ablation reports)
- [x] Robust defaults config truly uses NO manual tuning (no hardcoded e_min/e_max/proposal_std)
- [x] Results are reproducible (rerun 1 problem, 1 seed, verify match)
- [x] Report conclusions are supported by the data (no cherry-picking)
- [x] README tuning guide is consistent with report findings
- [x] All tests still pass (`pytest -v`)
- [x] Ruff clean (`ruff check .`)

### Step 47: Re-run 4-Way Benchmark with Aligned Defaults (Worker)

**Context:** SAMC class `_init_partition_from_energy` was misaligned with SAMCWeights.
Now both use: `bin_width=0.5, n_bins_per_side=100, max_bins=1000`.
Quick test shows massive improvement: 10D went from 33.1 → 0.66, 50D from 37.4 → 7.7.

**Task:** Re-run `benchmarks/four_way.py` with the aligned code and update README.

Worker checklist:
- [x] Delete old cached results: `rm -rf benchmarks/outputs/four_way`
- [x] Run full benchmark: `python benchmarks/four_way.py` (4 problems x 4 algos x 3 seeds = 48 runs, 200K iters)
- [x] Collect results into summary table
- [x] Update README comparison table at line ~224 with new numbers
- [x] Bold the best result per row
- [x] Update narrative text below the table
- [x] Run `ruff format . && ruff check .`
- [x] Run `pytest -x -q` to verify no regressions
- [x] Commit: "Update 4-way comparison with aligned partition defaults"

### Step 48: Fair 3-Way Benchmark (Worker)

**Context:** Inspector found major fairness issues in the 4-way benchmark:
- Unequal energy evals (MH: 200K, SAMC: 800K, PT: 1.6M)
- Different starting points (SAMC: origin, MH/PT: random)
- SAMC had adaptive proposal, MH/PT had fixed
- SAMC(tuned) was not the real ablation winner, just a modified version

**Decision:** Drop SAMC(tuned), keep only 3 algos: **MH vs PT vs SAMC (default/zero-config)**.
All algos get identical compute, starting points, and adaptive proposals.

**Task:** Rewrite `benchmarks/four_way.py` → `benchmarks/three_way.py` with these rules:

1. **3 algorithms:** MH, PT, SAMC (default zero-config only)
2. **4 chains/replicas each** — all algos get exactly `4 * n_iters` energy evals
   - MH: `n_chains=4` (already supported in `run_mh`)
   - PT: `n_replicas=4`, t_min=1.0, t_max=10.0, swap_interval=10
   - SAMC: `n_chains=4`, `adapt_proposal=True`, `adapt_warmup=2000`, no e_min/e_max
3. **Random starting point** — generate `x0 = torch.randn(dim)` ONCE per (problem, seed), pass same x0 to all algos
   - MH: pass `x0=x0` (single chain gets this; for multi-chain, each chain gets independent random but seeded)
   - PT: need to set `init_scale=1.0` (already default, starts from `torch.randn`)
   - SAMC: pass `x0=x0.unsqueeze(0).expand(4, -1).clone()` to give all 4 chains same start? Or let each chain start random? — Use same seed so all chains start consistently.
   - **Actually simplest:** just use `torch.manual_seed(seed)` before each algo call. Then MH multi-chain and PT both generate their own random starts from the same seed. SAMC with x0=None generates zeros — so explicitly pass `x0 = torch.randn(4, dim)` to SAMC after seeding.
4. **Adaptive proposal for MH/PT** — MH and PT baselines don't support adaptive proposals natively. Two options:
   - Option A: Write MH/PT loops using `GaussianProposal(adapt=True)` directly in the benchmark script
   - Option B: Keep fixed proposal but use a reasonable formula, and note it in the table
   - **Use Option A** — write simple MH and PT loops in the benchmark that use `GaussianProposal(step_size=1.0, adapt=True)` so all 3 algos auto-tune their step size
5. **Problems:** 10d, 50d, 100d, rastrigin (same 4 problems)
6. **Seeds:** 42, 123, 999 (3 seeds)
7. **Iterations:** 200K per chain/replica
8. **Output:** `benchmarks/outputs/three_way/`, JSON per run
9. **Summary table:** 3 columns (SAMC, MH, PT), markdown format for README

Worker checklist:
- [x] Create `benchmarks/three_way.py` with the above rules
- [x] For MH with adaptive: write a simple loop using `GaussianProposal(adapt=True)` + 4 independent chains, pick best
- [x] For PT with adaptive: write a loop using `GaussianProposal(adapt=True)` per replica + swap logic, 4 replicas
- [x] Smoke test with `--smoke` flag (1 problem, 1 seed, 10K iters)
- [x] Run full benchmark: `conda run -n illuma-samc python benchmarks/three_way.py`
- [x] Update README comparison table with new 3-column results
- [x] Update narrative text (remove SAMC tuned references, update multipliers)
- [x] `ruff format . && ruff check .`
- [x] `pytest -x -q`
- [x] Report full results table

**Fairness checklist (inspector will verify):**
- [x] All 3 algos: exactly 4 chains/replicas x 200K iters = 800K energy evals
- [x] All 3 algos: same random seed → same RNG state before each run
- [x] All 3 algos: `GaussianProposal(adapt=True)` for step-size tuning
- [x] SAMC uses zero-config defaults only (no e_min/e_max/n_partitions)
- [x] No algo starts at origin — all start from random or seeded random

## Phase 5.75: API Simplification

> Simplify SAMC and SAMCWeights based on benchmark findings.
> Remove dead code, align defaults, make independent weights the default.

### Step 49: Remove Unused Partition Types (Worker)

**What to remove:**
- `AdaptivePartition` — not used anywhere in sampler/weight_manager, only has tests
- `QuantilePartition` — not used anywhere, only has tests
- `GrowingPartition` — replaced by `ExpandablePartition`, only used in deprecated `SAMCWeights.auto()`

**What to keep:**
- `Partition` (base class)
- `UniformPartition` (explicit range)
- `ExpandablePartition` (auto-expand, the default)

Worker checklist:
- [x] Remove `AdaptivePartition`, `QuantilePartition`, `GrowingPartition` from `partitions.py`
- [x] Remove `SAMCWeights.auto()` classmethod (uses GrowingPartition, deprecated)
- [x] Remove `SAMCWeights.from_warmup()` classmethod (uses warmup MH, replaced by expand-on-demand)
- [x] Remove `GrowingPartition` from `__init__.py` exports
- [x] Remove tests for removed classes from `tests/test_partitions.py`
- [x] Remove any `from_warmup` tests from `tests/test_weight_manager.py`
- [x] Update `train.py` to remove references to removed partition types / factory methods
- [x] Update `ablation/` scripts if they reference removed classes
- [x] `ruff format . && ruff check .`
- [x] `pytest -x -q` — all remaining tests pass
- [x] Commit: "Remove unused partition types: Adaptive, Quantile, Growing"

### Step 50: Independent Weights as Default for SAMC (Worker)

**Current:** `SAMC(n_chains=4)` always shares weights across chains.

**Target:** `SAMC(n_chains=4)` defaults to independent weights. `shared_weights=True` for shared.

Independent = each chain has its own theta, counts, partition, and proposal.
Shared = all chains update one theta (current behavior).

Worker checklist:
- [x] Add `shared_weights: bool = False` parameter to `SAMC.__init__`
- [x] `n_chains > 1, shared_weights=False` (default): call `_run_single_chain` N times, each with its own state. Aggregate results: best_energy=min, acceptance_rate=avg, samples per-chain.
- [x] `n_chains > 1, shared_weights=True`: existing `_run_multi_chain` behavior
- [x] `n_chains=1`: unchanged (single chain, `shared_weights` ignored)
- [x] Update docstrings
- [x] Fix existing multi-chain tests: add `shared_weights=True` where they test shared behavior
- [x] Add test for independent mode
- [x] `ruff format . && ruff check . && pytest -x -q`
- [x] Commit: "Default to independent weights for multi-chain SAMC"

### Step 51: Align SAMC and SAMCWeights Bin Defaults (Worker)

**Current state:**
- SAMCWeights default: `bin_width=0.25, n_bins_per_side=100, max_bins=1000` (201 bins)
- SAMC class `_init_partition_from_energy`: also `bin_width=0.25, 201 bins, max_bins=1000` (aligned in this session)

**Benchmark finding:** `bin_width=0.5` outperforms 0.25 at 50D and 100D (3.92 vs 4.28 and 13.15 vs 19.12). 0.25 only better for Rastrigin.

**Decision:** Change default `bin_width` to 0.5 for both SAMC and SAMCWeights. Users who need finer resolution can set `bin_width=0.25`.

Worker checklist:
- [x] Change `SAMCWeights.__init__` default: `bin_width=0.5` (was 0.25)
- [x] Change `SAMC._init_partition_from_energy`: `bin_width=0.5` (was 0.25)
- [x] Keep `n_bins_per_side=100, max_bins=1000` (201 bins, same total coverage)
- [x] Update docstrings to reflect new default
- [x] Update tests if any hardcode bin_width=0.25 expectations
- [x] `ruff format . && ruff check .`
- [x] `pytest -x -q`
- [x] Commit: "Change default bin_width to 0.5 for better high-dim performance"

### Step 52: Update README and Docs (Worker)

After Steps 49-51:
- [x] Update README code examples (remove any GrowingPartition/auto/from_warmup references)
- [x] Update README comparison table with final benchmark numbers
- [x] Update tuning guide to reflect simplified API
- [x] Update `docs/api.rst` if it references removed classes
- [x] Verify all README code snippets still work
- [x] Commit: "Update README for simplified API"

### Step 53: Re-run Final Benchmark (Worker)

After Steps 49-51, re-run `benchmarks/three_way.py` with the new defaults to confirm:
- [x] Delete cached results: `rm -rf benchmarks/outputs/three_way`
- [x] Run full benchmark with 5 seeds (42, 123, 456, 789, 999) for tighter error bars
- [x] SAMC should now default to independent weights + bin_width=0.5
- [x] Update README table with final numbers
- [x] This is the definitive table for the README
- [x] Commit: "Final benchmark with simplified SAMC defaults"

## Phase 5.8: GPU Compatibility (dtype + device propagation)

> User-reported: API not GPU-compatible. dtype missing, device not propagated to all components.
> L2 already implemented core changes (config.py, sampler.py, weight_manager.py, partitions.py).
> L3 worker completes: tests, baselines, diagnostics, and docs.

### Step 55: Add dtype/device tests (Worker)

Add tests for the new `dtype` and `device` parameters. Test on CPU with explicit dtype,
and on MPS where available. No CUDA on this machine — skip CUDA tests.

**Important constraint:** MPS does NOT support float64 tensors. If `device="mps"` and
`dtype="float64"`, tensor creation will fail. Add a validation guard in `SAMC.__init__`
and `SAMCWeights.__init__`: if device is MPS and dtype is float64, raise `ValueError`
with a clear message. Then test that guard.

In `tests/test_sampler.py`, add a new test class `TestDtypeDevice`:
- [x] Test `SAMC(dtype="float64")` — verify `result.samples.dtype == torch.float64`
- [x] Test `SAMC(dtype=torch.float32)` — verify samples are float32 (default)
- [x] Test `SAMC(dtype="float64")` with `x0=torch.randn(2)` (float32 x0) — verify x0 is cast to float64
- [x] Test `SAMC(dtype="float64", n_chains=2)` — multi-chain dtype propagation
- [x] Test that `result.log_weights.dtype == torch.float64` always (internal accumulation stays float64 regardless of user dtype)
- [x] Test that `result.energy_history` is on the correct device

In `tests/test_weight_manager.py`, add tests:
- [x] Test `SAMCWeights(dtype="float64")` — verify `wm._dtype == torch.float64`
- [x] Test `SAMCWeights(dtype=torch.float32)` — verify `wm._dtype == torch.float32`
- [x] Test that `wm.theta.dtype == torch.float64` always (internal, not user-controlled)

In `tests/test_config.py` (or existing config tests), add:
- [x] Test `SAMCConfig(dtype="float64").build()` — verify the built SAMCWeights has `_dtype == torch.float64`
- [x] Test `SAMCConfig(dtype="float64").build_sampler(energy_fn=..., dim=2)` — verify sampler has `_dtype == torch.float64`
- [x] Test `SAMCConfig.from_yaml()` with a YAML that includes `dtype: float64` — verify it's parsed

In `tests/test_partitions.py`, add tests:
- [x] Test `UniformPartition(0, 10, 20, device="cpu")` — verify `edges.device.type == "cpu"`
- [x] Test `ExpandablePartition(0, 10, 20, device="cpu")` — verify `edges.device.type == "cpu"`
- [x] Test that `assign_batch` result is on the same device as the input energies tensor

In `tests/test_mps.py`, add MPS-specific dtype tests (skipped if no MPS):
- [x] Test `SAMC(device="mps", dtype="float32")` — verify samples are float32 on MPS
- [x] Test `SAMCWeights(device="mps", dtype="float32")` — verify dtype stored
- [x] Test `SAMC(device="mps", dtype="float64")` raises `ValueError` (MPS lacks float64)
- [x] Test `SAMCWeights(device="mps", dtype="float64")` raises `ValueError`

Run: `ruff format . && ruff check . && pytest -x -q`
[x] Commit: "Add dtype/device parameter tests"

### Step 56: Fix baselines device/dtype (Worker)

Add `device` and `dtype` parameters to baseline samplers so they work on GPU.

In `src/illuma_samc/baselines/metropolis_hastings.py`:
- [x] Add `device` and `dtype` params to `run_mh()` and `_run_single_mh()`
- [x] Fix `torch.randn(dim)` → `torch.randn(dim, device=device, dtype=dtype)` in x0 init and proposals
- [x] Fix `torch.tensor(energies)` → `torch.tensor(energies, device=device)`
- [x] Fix `torch.empty(0, dim)` → `torch.empty(0, dim, device=device, dtype=dtype)`

In `src/illuma_samc/baselines/parallel_tempering.py`:
- [x] Add `device` and `dtype` params to `run_parallel_tempering()`
- [x] Fix `torch.randn(dim)` → `torch.randn(dim, device=device, dtype=dtype)` in state init and proposals
- [x] Fix `torch.tensor(energies_list[0])` and `torch.empty(0, dim)` to use device/dtype

Run: `ruff format . && ruff check . && pytest -x -q`
[x] Commit: "Add device/dtype to MH and PT baselines"

### Step 57: Fix diagnostics device safety (Worker)

In `src/illuma_samc/diagnostics.py`:
- [x] Fix `torch.zeros(rolling_window - 1)` → add `.to(changed.device)` or explicit device
- [x] Fix `torch.ones(rolling_window)` → same
- [x] These only matter when energy_history is on GPU; ensure `.cpu()` is called before numpy conversion (already done for most tensors, verify completeness)

Run: `ruff format . && ruff check . && pytest -x -q`
[x] Commit: "Fix diagnostics tensor device for GPU compatibility"

### Step 58: Update docs and examples (Worker)

- [x] Update docstrings in `SAMCConfig`, `SAMC`, `SAMCWeights` to document `dtype` parameter
- [x] Add a GPU usage example to `examples/` or update existing example with `device`/`dtype` usage comment
- [x] Update `docs/quickstart.rst` if it exists — mention `device` and `dtype`
- [x] Verify all README code snippets still work with the new params

Run: `ruff format . && ruff check . && pytest -x -q`
[x] Commit: "Document dtype/device parameters"

### Step 59: Inspector review (Inspector)

Inspector verifies Steps 55-58:
- [x] All new tests actually test what they claim (read test code, not just names)
- [x] `dtype` parameter works end-to-end: config → sampler → result tensors
- [x] `device` parameter works end-to-end: config → partitions → sampler → result tensors
- [x] Baselines accept and propagate device/dtype correctly
- [x] No hardcoded `torch.float32` or `torch.float64` in tensor creation that should use the user's dtype
- [x] Internal accumulation (theta, counts) stays float64 regardless of user dtype
- [x] All tests pass: `pytest -x -q`
- [x] Lint clean: `ruff check .`
- [x] Existing MPS tests still pass (if MPS available)

**Inspector found:** `sample_log_weights` hardcoded to float32 in 3 code paths (sampler.py lines 594, 601, 883, 893). L2 fixed + added 4 dtype tests. 183 tests pass.

## Phase 6: PyPI Release

> Ship only after everything else is done and API is stable.

### Step 54: PyPI Release
- [ ] Add `python -m build` + twine upload to CI (or GitHub trusted publishing)
- [ ] Verify `pip install illuma-samc` works from PyPI
- [ ] Add installation badge to README
