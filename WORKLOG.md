# illuma-samc Work Log

## [2026-04-03] Steps 50-51: Independent weights default + bin_width alignment
- **State:** generalizing
- **Status:** done
- **Summary:**
  - Step 50: `SAMC(n_chains=N)` now defaults to independent chains (`shared_weights=False`). Each chain gets a fresh partition and proposal via `_reset_partition_and_proposal()`. New `_run_independent_chains()` runs N serial single-chain runs, aggregates to same output shapes as shared mode. Added `shared_weights=True` to `test_multi_chain_shared_weights_converge` and two new independent-chain tests.
  - Step 51: Changed default `bin_width` from 0.25 to 0.5 in both `SAMCWeights.__init__` and `SAMC._init_partition_from_energy`. Updated docstring. No test changes needed (no tests asserted specific bin counts from the default).
- **Decisions made:** Used Option A (reset partition/proposal state before each chain). Output shapes `(N, n_saved, dim)`, `(n_steps, N)` are identical for both modes, so most existing tests required no change.

## [2026-04-03] Step 49: Remove unused partition types
- **State:** generalizing
- **Status:** done
- **Summary:**
  - Removed `AdaptivePartition`, `QuantilePartition`, `GrowingPartition` from `partitions.py`.
  - Removed `SAMCWeights.auto()` and `SAMCWeights.from_warmup()` from `weight_manager.py`.
  - Simplified `_resize_for_partition` by removing `GrowingPartition`-specific low-side insertion logic.
  - Updated `__init__.py` exports: replaced `GrowingPartition` with `ExpandablePartition`.
  - Removed 21 tests covering removed classes/methods across `test_partitions.py` and `test_weight_manager.py`.
  - Updated `train.py` and `test_sampler.py` to remove all references to removed types.
- **Decisions made:** Kept `_resize_for_partition` method — still needed for `ExpandablePartition`; kept `ExpandablePartition` as the sole dynamic partition type.

## [2026-04-03] Steps 43-45: Robust Defaults Validation
- **State:** generalizing
- **Status:** done
- **Summary:**
  - Step 43: Created and ran `ablation/robust_defaults.py` — 60 runs (6 problems x 2 configs x 5 seeds). Both configs: (A) zero-config with `adapt_proposal=True`, (B) hand-tuned ablation winners from Steps 24-28.
  - Step 44: Created `ablation/analyze_robust_defaults.py` and generated `ablation/reports/robust_defaults.md`. Key finding: zero-config matches hand-tuned on 2D, Rosenbrock, Rastrigin (0% gap); but is 49%/114%/305% worse on 10D/50D/100D Gaussian mixtures where energy range matters.
  - Step 45: Updated README Tuning Guide — added "Zero-Config Quick Start" section with validation table, updated sensitivity ranking to reflect `proposal_std` and low-dim energy range are now auto-handled.
- **Decisions made:**
  - Energy range threshold: define "near-zero" as < 1e-4 (not 1e-8) to avoid floating point noise on problems with E≈0 showing spurious -90%+ gaps.
  - Kept the 10D gap in report (49%) as "NEEDS TUNING" even though it's marginal — the threshold is 10% and should stay conservative.
- **Blocked on:** Nothing.

## [2026-04-02] v0.2.0: Batch support, zero-config defaults, API cleanup
- **Phase:** generalizing
- **Status:** done
- **Summary:**
  - Added `n_chains` param to `SAMC.__init__` for explicit batch/multi-chain support (fixes user-reported `.item()` error on batched energy)
  - Refactored `SAMCWeights` to zero-config `__init__` with lazy partition init (ExpandablePartition centered on first energy, 201 bins default)
  - Batched tensor support in `correction()` and `step()`
  - Trimmed public partition API to UniformPartition + GrowingPartition
  - Changed flatness to visited-only metric (avoids penalizing unreachable bins)
  - Rewrote README for user attraction: practical-first ordering, simplified math, added Liang 2009 reference
  - Updated mh_vs_samc.ipynb with new API, moved to repo root
  - Published to public repo
- **Decisions made:**
  - ExpandablePartition as default over GrowingPartition (wide uniform + expand on overflow beats grow-from-one-bin)
  - Visited-only flatness after experiments showed over-specified bins unfairly penalize the metric
  - bin_width=0.25, 100 bins per side as defaults after ablation comparison
- **Blocked on:** Nothing.
- **Affects:** Public API breaking change (SAMCWeights.auto() deprecated, __init__ is now zero-config).

## [2026-04-01] Phase 3: Ablation Study Steps 25-28
- **Phase:** generalizing
- **Status:** done
- **Summary:**
  - Step 25: Ran 78 Rosenbrock 2D ablation runs. Key finding: independent multi-chain beats shared weights on unimodal problems; gain_t0=5000 critical.
  - Step 26: Ran 72 10D Gaussian Mixture ablation runs. Key finding: shared 8-chain SAMC best (E=0.280); proposal_std scales with problem not dimension; n_bins does NOT scale with dim.
  - Step 27: Ran 45 Rastrigin 20D ablation runs. Fixed random initialization (was starting at global min). Key finding: SAMC 16-chain (E=82.4) beats PT (E=92.5) and MH (E=106.8) on hardest problem.
  - Step 28: Ran 90 cross-comparison runs (3 algos x 3 problems x 10 seeds). Generated 5 comparison figures and final report.
- **Decisions made:**
  - Changed initialization from `torch.zeros` to `torch.randn * init_scale` in all runners (SAMC, MH, PT) to avoid starting at global minimum on Rastrigin.
  - Added `init_scale` parameter to train.py CLI, YAML configs, and PT/MH baselines.
  - Fixed `compute_energy_mixing` early return missing `n_round_trips` key.
  - Increased Rastrigin e_max from 200 to 500 to capture random initialization energy.
- **Blocked on:** Nothing.
- **Affects:** None.

## [2026-04-01] Phase 5: Robust Energy Bin Selection (Steps 39-42)
- **Phase:** prototyping
- **Status:** done
- **Summary:**
  - Steps 39-41 (overflow bins, auto-range warmup, expandable partition) were already implemented in prior sessions. Added Phase 5 to TODO.md and marked them complete.
  - Step 42: Ran 240 ablation runs (4 problems x 5 wrong-range scenarios x 4 methods x 3 seeds) at 100K iterations each.
  - Created `ablation/analyze_robust_bins.py` for analysis with heatmaps and bar charts.
  - Wrote comprehensive `ablation/reports/robust_bins_insights.md`.
- **Decisions made:**
  - Auto-Range (from_warmup) is the safest default -- recovers baseline performance in every scenario.
  - Overflow Bins are surprisingly effective even with completely wrong ranges and should be the recommended safety net.
  - Wrong energy range kills vanilla SAMC (0% acceptance). e_max too wide is benign; e_min too high is subtle but harmful.
  - Recommendation: always use overflow_bins=True unless you know the exact range.
- **Blocked on:** Nothing.
- **Affects:** Results inform future default parameter choices. Consider making overflow_bins=True the default in a future version.

## [2026-03-31] Step 24: 2D Multimodal Ablations (Phase 3B)
- **Phase:** prototyping
- **Status:** done
- **Summary:**
  - Ran all 12 ablation groups (171 total runs) on 2D multimodal: SAMC gain schedule, gain t0, n_bins, energy range, proposal_std, partition type, multi-chain; MH proposal_std, temperature; PT n_replicas, t_max, swap_interval. Each with 3 seeds (42, 123, 456) at 500K iterations.
  - Created `ablation/run_2d_ablations.py` runner script with parallel execution support.
  - Ran `ablation/analyze.py` on all 12 groups, generating CSV summaries and comparison plots.
  - Fixed bugs in analyze.py: handle missing bin_flatness for MH/PT results, handle inf energies in stdev computation.
  - Wrote comprehensive `ablation/reports/2d_insights.md` with sensitivity ranking, optimal ranges, tuning heuristics, and SAMC vs MH vs PT comparison.
- **Decisions made:** proposal_std (0.05-0.1) and energy range are the most critical SAMC parameters. Uniform partitions outperform adaptive/quantile. The `log` gain schedule should be avoided (poor flatness). All three algorithms find the global min on 2D, but SAMC provides unique flat exploration guarantee.
- **Blocked on:** Nothing.
- **Affects:** Insights feed into Steps 25-27 for higher-dimensional ablations.

## [2026-03-31] Step 23: Infrastructure Prep (Phase 3A)
- **Phase:** prototyping
- **Status:** done
- **Summary:**
  - Added Rosenbrock 2D (narrow valley, min at (1,1)) and Rastrigin 20D (~10^20 local minima, min at origin) problems
  - Created `analysis.py` with mode coverage (per-problem) and bin flatness metrics; integrated flatness into train.py results
  - Built `ablation/sweep.py` (YAML spec -> train.py commands, --dry-run, --parallel N) and `ablation/analyze.py` (load results, stats, plots, CSV)
  - Added --partition_type, --n_chains, --gain_t0 CLI args to train.py with full wiring
  - Smoke-tested all 6 combos (3 algos x 2 models) successfully
- **Decisions made:** QuantilePartition in CLI uses random warmup samples since it needs energy data upfront. Kept mode coverage simple (threshold-based distance to known modes).
- **Affects:** None — all new code, no breaking changes.

## [2026-03-31] Repo Organization Cleanup
- **Phase:** maintenance
- **Status:** done
- **Summary:**
  - Removed 4 stray PNGs from project root (regenerable outputs).
  - Moved `sample_code.py` to `reference/sample_code.py` and updated all references (CLAUDE.md, pyproject.toml, examples/comparison_sample_code.py).
  - Deleted `benchmarks/__init__.py` (nothing imports from benchmarks as a package) and `tests/test_placeholder.py` (redundant with real test suite).
  - Updated .gitignore with patterns for logs, caches, and generated PNGs.
  - Fixed all example scripts to save PNGs into their own directory instead of project root.
- **Decisions made:** Examples save PNGs to `examples/` via `os.path.dirname(__file__)`. Reference script saves to `reference/`. Both directories' PNGs are gitignored.
- **Blocked on:** Nothing.
- **Affects:** None.

## [2026-03-30] Phase 1 MVP Complete
- **Phase:** prototyping
- **Status:** done
- **Summary:**
  - Implemented full SAMC library: gain sequences (1/t, log, ramp, custom), partitions (uniform, adaptive, quantile), proposals (Gaussian, Langevin), and core sampler with simple + flexible modes.
  - Added diagnostics module with 4-panel plots (weights, energy trace, bin visits, rolling acceptance rate).
  - Created two examples: multimodal_2d.py (reproduces sample_code.py results) and gaussian_mixture.py (4-mode demo). SAMC found global minimum at -8.12 vs MH's -2.25 on the 2D cost function; all 4 modes visited on Gaussian mixture.
  - 46 tests passing, full lint clean (ruff check + format).
- **Decisions made:** Used ramp gain as default (matches sample_code.py). Kept partition.assign() accepting Tensor for consistency. Used dataclass for SAMCResult.
- **Blocked on:** Nothing — Phase 1 complete, ready for Frank's review.
- **Affects:** None.

## [2026-03-31] Phase 2.5: Benchmark fixes and experiment infrastructure (Steps 12-17)
- **Phase:** generalizing
- **Status:** done
- **Summary:**
  - Step 12: Removed ESS columns from README benchmark table (ESS was already removed from code). Updated benchmark numbers to current results (proposal_std=0.05 for 2D, proposal_std=1.0 for 10D, e_max=20 for 10D SAMC). Regenerated all benchmark plots.
  - Step 13: Added "Energy Evals" column to benchmark table to make PT's 8x compute cost transparent. Updated analysis text to highlight compute fairness.
  - Step 14: Created `train.py` CLI with argparse for running single experiments. Supports --algo (samc/mh/pt), --model (2d/10d), all hyperparams as CLI args. Reuses energy functions and runners from benchmarks/vs_mh_pt.py.
  - Step 15: Created YAML config system in `configs/` with tuned defaults per model. train.py auto-loads matching config, CLI args override YAML values.
  - Step 16: Added `compare_results.py` to load and rank all runs for a model. Output structure: `outputs/<model>/<algo>/<timestamp>/` with config.yaml, results.json, and diagnostic plots. Added .gitignore for outputs/.
- **Decisions made:** Added PyYAML as a core dependency. Created benchmarks/__init__.py to allow importing energy functions/runners. Used UTC timestamps for run directories.
- **Blocked on:** Nothing.
- **Affects:** None.

## [2026-03-31] Phase 2.75: Bug Fixes, Input Validation, UX, Tests (Steps 18-21)
- **Phase:** bugfix
- **Status:** done
- **Summary:**
  - Step 18: Fixed 5 bugs (one commit each): out-of-range sample bin assignment in multi-chain, importance_weights NaN on all-inf, AdaptivePartition unbounded memory (now deque), AdaptivePartition recording outliers, UniformPartition.edges allocation per call.
  - Step 19: Added input validation for e_min/e_max, n_bins, dim, proposal_std, n_steps with descriptive ValueErrors.
  - Step 20: Added 5 UX improvements (one commit each): warn on out-of-range initial state, warn on low acceptance rate, seed parameter on run(), abstract edges on Partition base class, n_bins alias in SAMC constructor.
  - Step 21: Added 5 coverage tests. Found and fixed multi-chain plot_diagnostics crash (energy_history was list[Tensor] not list[float]).
  - Test suite: 72 passed, 3 skipped (CUDA), 0 failed.
- **Decisions made:** Used deque(maxlen=50_000) for AdaptivePartition history. importance_weights returns zeros with warning on all-inf. n_bins is alias for n_partitions (backward compat kept).
- **Blocked on:** Nothing — Phase 2.75 complete.
- **Affects:** None.

## [2026-03-31] Codebase Restructure (Step 22)
- **Phase:** maintenance
- **Status:** done
- **Summary:**
  - Extracted energy functions into `src/illuma_samc/problems/` with registry dict and per-problem modules (multimodal_2d.py, gaussian_10d.py).
  - Extracted MH and PT baselines into `src/illuma_samc/baselines/` as standalone modules.
  - Slimmed `benchmarks/vs_mh_pt.py` from 629 lines to ~330 (orchestration + plotting only).
  - Updated `train.py` and `benchmarks/debug_10d.py` to import from new canonical locations.
  - Added subpackage re-exports in `__init__.py`.
  - All 72 tests pass, ruff clean, `train.py` and `compare_results.py` verified working.
- **Decisions made:** Kept legacy aliases (cost_2d, gaussian_mixture_10d) as re-exports for backward compatibility. Problems module uses registry dict pattern for extensibility.
- **Blocked on:** Nothing.
- **Affects:** None.

## [2026-03-30] Phase 2 Complete: GPU, Parallel Chains, Verification, Benchmarks
- **Phase:** generalizing
- **Status:** done
- **Summary:**
  - Step 9: Added GPU support (single-chain CUDA verified) and parallel chains with shared weights. Multi-chain API: `sampler.run(n_steps, x0=torch.randn(N, dim))`. Batched proposals + energy eval, sequential accept/reject per chain for correctness.
  - Step 9.5: Verified illuma-samc matches sample_code.py — reference best_E=-8.123 vs API best_E=-8.125, identical flat bin visits and learned weights.
  - Step 10: Benchmarked SAMC vs MH vs Parallel Tempering on 2D multimodal and 10D Gaussian mixture. SAMC achieves 2x ESS of MH on 2D at similar cost. Added results to README.
- **Decisions made:** Theta stays on CPU for multi-chain (small vector, frequent updates). Return shape: (N, n_saved, dim) for multi-chain. Energy history: (n_steps, N) for multi-chain.
- **Blocked on:** Nothing — Phase 2 complete.
- **Affects:** None.
