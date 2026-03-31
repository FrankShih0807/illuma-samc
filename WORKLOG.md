# illuma-samc Work Log

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
