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
- [ ] Ensure single-chain SAMC runs on CUDA when `device="cuda"` — energy_fn, proposals, and state all on GPU. Theta weights stay on CPU (small vector, updated every step).
- [ ] NOTE: SAMC is inherently sequential (weight update depends on current theta). Do NOT implement naive batch parallelism — it would break correctness.
- [ ] Implement **parallel chains with shared weights**: run N chains simultaneously, batch the energy evaluation (the expensive part), but synchronize theta updates after each step. API: `sampler.run(n_steps=10000, x0=torch.randn(N, dim))` where N is number of chains.
- [ ] Each chain proposes independently, energy is evaluated as a batch, then each chain's accept/reject and weight update happens sequentially on the shared theta.
- [ ] Return shape: `(N, n_steps, dim)` for multi-chain, `(n_steps, dim)` for single chain.
- [ ] Add tests: verify GPU results match CPU results (same seed), verify multi-chain shared weights converge same as single chain.

### Step 9.5: Verify Correctness Against sample_code.py
- [ ] Run `python sample_code.py` and save the output PNG (`samc_experiment.png`) as the ground truth
- [ ] Run `python examples/multimodal_2d.py` with the same parameters (1M iterations, 42 bins, same gain schedule)
- [ ] Compare results: the illuma-samc version MUST achieve flat bin visit histogram (all bins visited roughly equally) and find the global minimum (best energy ≈ -8.2)
- [ ] If bin visits are NOT flat or best energy is significantly worse, debug and fix the core SAMC loop — the weight update or acceptance ratio is likely wrong
- [ ] Save comparison plots side by side

### Step 10: Benchmarks
- [ ] Create `benchmarks/vs_mh_pt.py` comparing SAMC vs MH vs parallel tempering on:
  - 2D multimodal cost function from `sample_code.py`
  - 10D Gaussian mixture (well-separated modes)
- [ ] Metrics: best energy found, ESS (effective sample size), acceptance rate, wall-clock time
- [ ] Generate comparison plots (save as PNG)
- [ ] Add benchmark results to README
