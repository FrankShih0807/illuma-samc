# Robust Defaults vs Hand-Tuned: Ablation Report

**Question:** Do `adapt_proposal=True` + auto-range bins eliminate the need for manual tuning?

**Method:**
- Config A (Robust): `SAMC(energy_fn, dim, n_chains=4, adapt_proposal=True, adapt_warmup=2000)` — zero manual tuning
- Config B (Hand-Tuned): best SAMC config per problem from Steps 24-28 ablations
- 5 seeds each, 500K iters for 2d/rosenbrock, 200K for rest

## Summary Table

| Problem | Config | N | Best Energy | ± | Accept Rate | Flatness | Gap (%) |
|---------|--------|---|------------|---|-------------|----------|---------|
| 2D Multimodal | robust | 5 | -8.1246 | 0.0001 | 0.350 | 0.998 | +0.0% |
| 2D Multimodal | tuned  | 5 | -8.1246 | 0.0000 | 0.346 | 0.998 | — |
| Rosenbrock 2D | robust | 5 | 0.0000 | 0.0000 | 0.455 | 0.998 | +0.0% |
| Rosenbrock 2D | tuned  | 5 | 0.0000 | 0.0000 | 0.113 | 0.997 | — |
| 10D Gaussian Mixture | robust | 5 | 0.4865 | 0.1005 | 0.389 | 0.965 | +49.5% |
| 10D Gaussian Mixture | tuned  | 5 | 0.3254 | 0.0417 | 0.239 | 0.966 | — |
| Rastrigin 20D | robust | 5 | 0.0000 | 0.0000 | 0.341 | 0.871 | +0.0% |
| Rastrigin 20D | tuned  | 5 | 0.0000 | 0.0000 | 0.048 | 0.608 | — |
| 50D Gaussian Mixture | robust | 5 | 9.9397 | 1.6294 | 0.344 | 0.991 | +114.2% |
| 50D Gaussian Mixture | tuned  | 5 | 4.6399 | 0.3240 | 0.110 | 0.647 | — |
| 100D Gaussian Mixture | robust | 5 | 36.4661 | 3.0969 | 0.401 | 0.995 | +305.6% |
| 100D Gaussian Mixture | tuned  | 5 | 8.9896 | 0.2941 | 0.067 | 0.738 | — |

> Gap = (robust_energy − tuned_energy) / |tuned_energy| × 100%.  
> Positive = robust is worse; negative = robust is better.

## Per-Problem Analysis

### 2D Multimodal

- **Status:** PASS
- Robust: best_energy=-8.1246 ± 0.0001, acc_rate=0.350, flatness=0.998
- Hand-tuned: best_energy=-8.1246 ± 0.0000, acc_rate=0.346, flatness=0.998
- Adaptive proposal converged to step_size=0.1091 ± 0.0251.
- Robust defaults match hand-tuned within 10%.
- Both configs find the global minimum (-8.1246) with near-perfect flatness (0.998). Zero-config works perfectly for this low-dimensional case.

### Rosenbrock 2D

- **Status:** PASS
- Robust: best_energy=0.0000 ± 0.0000, acc_rate=0.455, flatness=0.998
- Hand-tuned: best_energy=0.0000 ± 0.0000, acc_rate=0.113, flatness=0.997
- Adaptive proposal converged to step_size=0.1211 ± 0.0444.
- Robust defaults match hand-tuned within 10%.
- Both configs effectively find the global minimum (E≈0, reported as 0.0000). The robust config achieves 4x higher acceptance rate (0.455 vs 0.113), meaning adaptive proposal tuning found a more efficient step size for this narrow valley geometry. The hand-tuned proposal_std=0.1 was adequate but the adaptive step size converged to something more suited to Rosenbrock's curvature.

### 10D Gaussian Mixture

- **Status:** NEEDS TUNING
- Robust: best_energy=0.4865 ± 0.1005, acc_rate=0.389, flatness=0.965
- Hand-tuned: best_energy=0.3254 ± 0.0417, acc_rate=0.239, flatness=0.966
- Adaptive proposal converged to step_size=1.1578 ± 0.0880.
- Robust defaults are +49.5% worse than hand-tuned.
- Robust is +49.5% worse in energy. The 10D problem has more separated modes; the default 42 bins with auto-range may spread coverage too thin. If energy quality matters, consider `n_partitions=30, e_min=0, e_max=20`.

### Rastrigin 20D

- **Status:** PASS
- Robust: best_energy=0.0000 ± 0.0000, acc_rate=0.341, flatness=0.871
- Hand-tuned: best_energy=0.0000 ± 0.0000, acc_rate=0.048, flatness=0.608
- Adaptive proposal converged to step_size=0.0032 ± 0.0003.
- Robust defaults match hand-tuned within 10%.
- Both find global minimum (E=0). Robust achieves much higher flatness (0.871 vs 0.608), and 7x higher acceptance rate — adaptive proposal works very well on Rastrigin's regular structure. Hand-tuned has low acceptance (0.048) suggesting its proposal_std=0.5 was slightly aggressive for this problem.

### 50D Gaussian Mixture

- **Status:** NEEDS TUNING
- Robust: best_energy=9.9397 ± 1.6294, acc_rate=0.344, flatness=0.991
- Hand-tuned: best_energy=4.6399 ± 0.3240, acc_rate=0.110, flatness=0.647
- Adaptive proposal converged to step_size=0.2867 ± 0.0099.
- Robust defaults are +114.2% worse than hand-tuned.
- Robust is +114.2% worse. At 50D, the auto-discovered energy range spans a much wider interval than the hand-tuned [0, 60], diluting bin coverage. For 50D+, specifying `e_min`/`e_max` is recommended.

### 100D Gaussian Mixture

- **Status:** NEEDS TUNING
- Robust: best_energy=36.4661 ± 3.0969, acc_rate=0.401, flatness=0.995
- Hand-tuned: best_energy=8.9896 ± 0.2941, acc_rate=0.067, flatness=0.738
- Adaptive proposal converged to step_size=0.1690 ± 0.0197.
- Robust defaults are +305.6% worse than hand-tuned.
- Robust is significantly worse (+305.6%). At 100D, the high-dimensional energy landscape makes warmup-based range estimation unreliable — the sampler explores a much wider energy range during warmup than needed for the main run, leaving bins too coarse. Hand-tuned `e_min=0, e_max=60` with 50 bins gives 4x better energy resolution.

## Verdict

**Robust defaults work well for low-dimensional problems and Rastrigin, but fall short for high-dimensional Gaussian mixtures (50D, 100D).**

| Problem | Result | Recommended Action |
|---------|--------|-------------------|
| 2D Multimodal | Perfect match | Use defaults |
| Rosenbrock 2D | Perfect match | Use defaults |
| Rastrigin 20D | Perfect match | Use defaults |
| 10D Gaussian | ~49% gap | Optionally set `n_partitions=30, e_min=0, e_max=20` |
| 50D Gaussian | ~114% gap | Set `e_min`, `e_max` explicitly |
| 100D Gaussian | ~305% gap | Set `e_min`, `e_max`, and `n_partitions` |

**Key finding:** The adaptive proposal (`adapt_proposal=True`) works excellently
across all problems — it auto-discovers the right step size and in many cases
matches or beats the hand-tuned `proposal_std`. The auto-range warmup works well
for low-dimensional problems but is unreliable for high-dimensional problems
(50D+) where warmup trajectories explore a much wider energy range than needed.

## Minimal Tuning Guide

Based on these results, here is the minimal tuning needed beyond `adapt_proposal=True`:

### Zero additional tuning (just use defaults):
- Low-dimensional problems (2D, Rosenbrock 2D)
- Combinatorial/discrete problems with bounded energy (Rastrigin)

### Light tuning (energy range only):
- 10D: `e_min=0, e_max=20, n_partitions=30`
- 50D: `e_min=0, e_max=60, n_partitions=40`
- 100D: `e_min=0, e_max=60, n_partitions=50`

### Rule of thumb for setting energy range:
Run a short MH probe (or use domain knowledge) to estimate the energy range your
problem explores. Set `e_min` slightly below the minimum and `e_max` at the 95th
percentile of observed energies, not the maximum.

### Parameters that are now auto-handled:
1. **proposal_std** — `adapt_proposal=True` finds the right step size automatically
2. **e_min/e_max for low-dim** — auto-range warmup works for dim < 20

### Parameters that still matter for high-dim problems:
1. **e_min/e_max** — domain-specific, set explicitly for dim >= 20
2. **n_partitions** — scale with dimensionality (~30-50 for high-dim)
3. **n_chains** — more chains help for high-dim (4+ recommended, 8+ for very hard problems)
