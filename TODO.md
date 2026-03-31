# illuma-samc TODO

> Child Claude: work through these in order. Each task is a checkpoint — format, lint, test, commit, push after completing each one.

## Phase 1: Core MVP

### Step 1: Project Setup
- [ ] Set up conda environment `illuma-samc` with Python 3.11, PyTorch, ruff, pytest
- [ ] Finalize `pyproject.toml` with all dependencies (torch, matplotlib optional, tqdm optional)
- [ ] Create `CITATION.bib` with Liang 2006 SAMC paper (JASA)
- [ ] Verify `pip install -e ".[dev]"` works
- [ ] Run `ruff check .` and `pytest` (empty test file is fine)

### Step 2: Gain Sequences
- [ ] Implement `GainSequence` class in `src/illuma_samc/gain.py`
- [ ] Support `"1/t"` schedule: γ_t = t0 / max(t, t0)
- [ ] Support `"log"` schedule: γ_t = t0 / max(t * log(t+e), t0)
- [ ] Support `"ramp"` schedule matching `sample_code.py`: rho * exp(-tau * log((t - offset) / step_scale)) with warmup phase
- [ ] Support custom callable: any `(int) → float`
- [ ] Add tests in `tests/test_gain.py`

### Step 3: Partitions
- [ ] Implement `UniformPartition` in `src/illuma_samc/partitions.py` — linear energy binning matching `sample_code.py` style (e_min, e_max, n_bins)
- [ ] Implement `AdaptivePartition` — recomputes boundaries from visited energies
- [ ] Implement `QuantilePartition` — boundaries from warmup energy sample
- [ ] All partitions have `.assign(energy: Tensor) → int` and `.n_partitions` property
- [ ] Add tests in `tests/test_partitions.py`

### Step 4: Proposals
- [ ] Implement `GaussianProposal` in `src/illuma_samc/proposals.py` — isotropic random walk with configurable step size
- [ ] Implement `LangevinProposal` — MALA-style using autograd, with correct log proposal ratio
- [ ] Both have `.propose(x) → x_new` and `.log_ratio(...)` methods
- [ ] Add tests in `tests/test_proposals.py`

### Step 5: Core SAMC Sampler
- [ ] Implement `SAMC` class in `src/illuma_samc/sampler.py` with two modes:
- [ ] **Simple mode**: `SAMC(energy_fn=..., dim=..., n_partitions=...)` — user provides energy function, sampler handles proposal/partition/acceptance internally
- [ ] **Flexible mode**: `SAMC(dim=..., n_partitions=..., proposal_fn=..., log_accept_fn=..., partition_fn=...)` — user provides custom functions for full control
- [ ] Core loop matches `sample_code.py` logic: propose → compute acceptance with SAMC weight correction → accept/reject → update theta weights on EVERY iteration
- [ ] `sampler.run(n_steps, x0)` returns `torch.Tensor` of samples
- [ ] Track: `log_weights`, `acceptance_rate`, `energy_history`, `bin_counts`, `best_x`, `best_energy`
- [ ] Support `device` parameter for GPU
- [ ] Add progress bar via tqdm (optional)
- [ ] Add tests in `tests/test_sampler.py` — test both simple and flexible modes, verify they produce equivalent results on the same problem

### Step 6: Diagnostics
- [ ] Implement `diagnostics.py`: `plot_diagnostics(sampler)` function
- [ ] Plot weight trajectory (theta over iterations)
- [ ] Plot energy trace
- [ ] Plot bin visit histogram
- [ ] Plot acceptance rate over time (rolling window)
- [ ] `sampler.plot_diagnostics()` convenience method
- [ ] Add tests (just verify no errors, not plot correctness)

### Step 7: Examples & Verification
- [ ] Create `examples/multimodal_2d.py` — reproduce `sample_code.py` results using the illuma-samc API (both SAMC and MH comparison)
- [ ] Create `examples/gaussian_mixture.py` — classic multimodal Gaussian demo
- [ ] Verify SAMC finds global minimum on the 2D cost function from `sample_code.py`
- [ ] Verify weight vector converges to approximately flat on Gaussian mixture
- [ ] Update `__init__.py` exports

### Step 8: README & Attribution
- [ ] Create `README.md` with: Illuma branding, prominent Liang attribution ("Based on SAMC by Faming Liang, JASA 2006"), install instructions, quick start (simple + flexible API), example output plot
- [ ] Update `PROJECT_REGISTRY.md` in root workspace to include illuma-samc
