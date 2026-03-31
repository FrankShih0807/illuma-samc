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

Work through each bug **one at a time** using the procedure above.

#### Bug A: Out-of-range samples saved with wrong bin index
- **Where:** `sampler.py:331,473` — `max(cur_bin, 0)` assigns bin 0 when state is out of range
- **Impact:** Wrong importance weights on out-of-range samples
- **Reproduce:** Run SAMC with e_max smaller than initial energy → samples get bin 0 with wrong weights
- **Expected:** Out-of-range samples should have log-weight = -inf (or be skipped)
- **Test:** `test_out_of_range_sample_bin_not_zero` — verify sample_log_weights is -inf when sample is out of range
- **Fix:** Store bin=-1 for out-of-range samples, set log-weight to -inf in the importance weight computation

#### Bug B: `importance_weights` produces NaN when all weights are -inf
- **Where:** `SAMCResult.importance_weights` property
- **Impact:** `logsumexp(-inf) = -inf`, then `-inf - (-inf) = NaN` → NaN weights
- **Reproduce:** Create SAMCResult with all sample_log_weights = -inf, call `.importance_weights`
- **Expected:** Return zeros (no valid samples) or raise a clear error
- **Test:** `test_importance_weights_all_inf` — verify no NaN, returns zeros or raises
- **Fix:** Check if all log-weights are -inf before normalization; return zeros with warning

#### Bug C: AdaptivePartition unbounded memory growth
- **Where:** `partitions.py:89` — `_history` list grows by 1 float per call
- **Impact:** 1M iterations = ~8MB, 100M = ~800MB
- **Reproduce:** Create AdaptivePartition, call `.assign()` 1M times, check `len(self._history)`
- **Expected:** Memory usage bounded regardless of iteration count
- **Test:** `test_adaptive_partition_memory_bounded` — verify history size stays under limit after many calls
- **Fix:** Use a fixed-size deque or only keep last `max_history` samples (default 50_000)

#### Bug D: AdaptivePartition records out-of-range energies
- **Where:** `partitions.py:96` — `self._history.append(e)` happens before range check
- **Impact:** Out-of-range outlier energies skew the adapted bin edges
- **Reproduce:** Feed mostly in-range values + a few extreme outliers → edges expand to cover outliers
- **Expected:** Only in-range energies influence adaptation
- **Test:** `test_adaptive_partition_ignores_outliers` — verify edges don't expand to cover out-of-range values
- **Fix:** Only append to history if `_bin_for(e) >= 0`

#### Bug E: `UniformPartition.edges` allocates on every call
- **Where:** `partitions.py:57` — `torch.linspace(...)` creates new tensor each call
- **Impact:** Wasteful allocation in loops (diagnostics, plotting)
- **Reproduce:** Call `.edges` 1000 times, measure allocation
- **Expected:** Same tensor returned each time
- **Test:** `test_uniform_partition_edges_cached` — verify `p.edges is p.edges` (same object)
- **Fix:** Compute once in `__init__`, return cached tensor from property

### Step 19: Input Validation

Same procedure: write failing test first, then add validation, then verify.

- [ ] `e_min >= e_max` → `pytest.raises(ValueError, match="e_min must be less than e_max")`
- [ ] `n_bins <= 0` → `pytest.raises(ValueError, match="n_bins must be positive")`
- [ ] `dim <= 0` → `pytest.raises(ValueError, match="dim must be positive")`
- [ ] `proposal_std <= 0` → `pytest.raises(ValueError, match="proposal_std must be positive")`
- [ ] `n_steps <= 0` in `run()` → `pytest.raises(ValueError, match="n_steps must be positive")`

### Step 20: UX Warnings & Improvements

Each item follows: demonstrate problem → write test → implement → verify.

- [ ] **Warn on out-of-range initial state:** `warnings.warn("Initial energy {e} is outside partition range [{e_min}, {e_max}]. Chain may not mix.")`
  - Test: `pytest.warns(UserWarning, match="outside partition range")`
- [ ] **Warn on low acceptance rate:** After `run()` completes, if acceptance_rate < 0.01, warn.
  - Test: `pytest.warns(UserWarning, match="acceptance rate")`
- [ ] **`seed` parameter on `run()`:** `run(n_steps=1000, seed=42)` calls `torch.manual_seed(seed)`.
  - Test: Two runs with same seed produce identical results.
- [ ] **`edges` on base `Partition` class:** Add abstract property so users don't need `sampler._partition.edges`.
  - Test: All partition subclasses have `.edges` returning a tensor.
- [ ] **Naming consistency:** Pick `n_bins` for all partition constructors (it's more intuitive for energy bins). Keep `n_partitions` as the property name for backward compat, but accept `n_bins` in SAMC constructor too.
  - Test: `SAMC(..., n_bins=10)` works same as `SAMC(..., n_partitions=10)`.

### Step 21: Additional Test Coverage

These are tests for existing behavior that lacks coverage. Write them after Steps 18-20.

- [ ] **Energy function returning (energy, in_region) tuple:** Verify sampler handles `cost_2d` style
- [ ] **Very short run (n_steps < save_every):** Verify empty samples tensor, no crash
- [ ] **save_every=1:** Verify all samples saved (samples.shape[0] == n_steps)
- [ ] **LangevinProposal in full sampler loop:** End-to-end test with gradient-informed proposals
- [ ] **Multi-chain `plot_diagnostics`:** Verify no crash after multi-chain run

## Phase 3: Production Hardening (PARKED — do not start)

> Recorded for future planning. Do not begin until Phase 2.5 is complete and Frank approves.

### Checkpointing & Serialization
- Save/load sampler state (theta, partition edges, config) to resume interrupted runs
- Export results to common formats

### Adaptive Proposal Tuning
- Auto-tune proposal_std based on acceptance rate (target ~0.234 for high-dim)
- Dual averaging or Robbins-Monro style adaptation during warmup

### Convergence Diagnostics
- Weight stabilization test (are theta values converging?)
- Multi-chain R-hat diagnostic
- Geweke test on energy trace

### Scaling Study
- Benchmark SAMC vs MH vs PT across dimensions: 2, 10, 50, 100
- Identify where SAMC shines vs breaks down
- Guide recommendations for when to use SAMC
