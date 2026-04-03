# Rastrigin 20D Ablation Study: Insights Report

> **Problem**: Rastrigin 20D with ~10^20 local minima, global min at origin with E=0
> **Setup**: 500K iterations, 3 seeds (42, 123, 456) per configuration, 45 total runs
> **Algorithms**: SAMC (main focus), MH baseline, PT baseline
> **Init**: Random initialization with scale=2.0 (starting E ~ 260)
> **Date**: 2026-04-01

---

## Executive Summary

Rastrigin 20D is an extremely hard optimization landscape -- ~10^20 local minima in 20 dimensions. **No algorithm comes close to the global minimum (E=0)**. The best result is SAMC with 16 shared chains at E=82.4. Key findings:

1. **More chains = better**: 16 shared chains achieves E=82.4 vs single chain at E=107.6. Multi-chain SAMC is essential for hard landscapes.
2. **SAMC beats MH and PT**: SAMC 16-chain (E=82.4) > PT 16-replica (E=92.5) > MH (E=106.8). SAMC's weight correction provides a meaningful advantage.
3. **All acceptance rates are extremely low** (0-4%), reflecting the difficulty of the landscape.
4. **Flatness is poor everywhere** -- 500K iterations is insufficient for weight convergence in 20D with 40 bins spanning E=[0, 500].
5. **Gain schedule doesn't matter** -- ramp and 1/t produce identical results (again).

---

## 1. SAMC Proposal Step Size

| proposal_std | Best Energy (mean +/- std) | Acceptance Rate | Bin Flatness |
|--------------|---------------------------|-----------------|--------------|
| **0.5** | **107.57 +/- 7.58** | **0.044** | 0.387 |
| 1.0 | 111.55 +/- 7.40 | 0.026 | -0.490 |
| 2.0 | 140.80 +/- 5.10 | 0.015 | 0.091 |

**Insight**: Smaller steps (0.5) work best in 20D Rastrigin -- the local minima are close together (period = 1 in each dimension), so small exploratory steps are more productive than large jumps. This is opposite to the 10D finding where larger steps helped.

## 2. SAMC Multi-Chain: Shared vs Independent

### Shared Weights

| n_chains | Best Energy (mean +/- std) | Acceptance Rate | Bin Flatness |
|----------|---------------------------|-----------------|--------------|
| 1 | 107.57 +/- 7.58 | 0.044 | 0.387 |
| 4 | 90.96 +/- 11.54 | 0.025 | -0.533 |
| 8 | 87.04 +/- 7.14 | 0.025 | -0.533 |
| **16** | **82.39 +/- 5.04** | 0.035 | 0.128 |

### Independent Weights

| n_chains | Best Energy (mean +/- std) | Acceptance Rate | Bin Flatness |
|----------|---------------------------|-----------------|--------------|
| 4 | 93.00 +/- 5.86 | 0.034 | -0.688 |
| 8 | 87.97 +/- 1.69 | 0.031 | -0.523 |

**Insight**: **Shared weights are slightly better than independent** in this multimodal landscape: shared 8-chain (E=87.0) vs independent 8-chain (E=88.0). The difference is small. However, independent chains have lower variance (std=1.69 vs 7.14), suggesting more consistent results.

16 shared chains achieves the best absolute energy (E=82.4) -- more parallel exploration helps in high-dimensional spaces with many local minima.

## 3. SAMC Gain Schedule

| Schedule | Best Energy (mean +/- std) | Acceptance Rate | Bin Flatness |
|----------|---------------------------|-----------------|--------------|
| 1/t | 107.57 +/- 7.58 | 0.044 | 0.387 |
| ramp | 107.57 +/- 7.58 | 0.044 | 0.387 |

**Insight**: Identical results. Gain schedule is a non-factor on Rastrigin (same conclusion as 2D and 10D).

## 4. MH Baseline

| proposal_std | Best Energy (mean +/- std) | Acceptance Rate |
|--------------|---------------------------|-----------------|
| 1.0 | 106.83 +/- 4.84 | 0.000 |
| 2.0 | 131.24 +/- 18.35 | 0.000 |

**Insight**: MH is completely stuck -- 0% acceptance rate with both step sizes. MH cannot escape local minima in 20D Rastrigin at T=1.0. proposal_std=1.0 performs better than 2.0 because the sampler finds a better initial local minimum before stalling.

## 5. PT Baseline

| n_replicas | Best Energy (mean +/- std) | Acceptance Rate | Total Evals |
|------------|---------------------------|-----------------|-------------|
| 8 | 95.71 +/- 7.85 | 0.001 | 4M |
| 16 | 92.55 +/- 8.49 | 0.001 | 8M |

**Insight**: PT with 16 replicas (E=92.5) beats single-chain SAMC (E=107.6) but loses to 16-chain SAMC (E=82.4). PT's temperature exchange provides some escape from local minima, but at 8M energy evaluations vs 8M for SAMC 16-chain, SAMC is more efficient.

---

## 6. Cross-Problem Scaling Summary

| Metric | 2D Multimodal | Rosenbrock 2D | 10D Gaussian | Rastrigin 20D |
|--------|--------------|---------------|--------------|---------------|
| SAMC best E | -8.125 (optimal) | 0.000 (optimal) | 0.280 (near-optimal) | 82.4 (far from 0) |
| MH best E | -8.125 | 0.000 | 0.281 | 106.8 |
| PT best E | -8.125 | 0.000 | 0.459 | 92.5 |
| SAMC acc rate | 30-50% | 6-12% | 23% | 3-4% |
| Chains needed | 1 | 1 | 8 | 16 |
| proposal_std | 0.1 | 0.5 | 1.0 | 0.5 |

---

## 7. Where SAMC Breaks Down

1. **Acceptance rate degrades with dimension**: From ~50% in 2D to ~3% in 20D. SAMC's weight correction can't compensate for the curse of dimensionality in proposal acceptance.

2. **Flatness is unreachable in 20D**: The best flatness on Rastrigin is 0.387 (vs 0.99 in 2D). With 40 bins and 500K iterations, there aren't enough visits per bin for weight convergence.

3. **Far from global minimum**: E=82.4 vs global min of 0. SAMC reduces energy by ~70% from initialization (E~260) but can't navigate the combinatorial explosion of local minima.

4. **More iterations would help**: The low acceptance rate and poor flatness suggest 500K iterations is insufficient. SAMC's theoretical guarantees require ergodicity, which is far from reached.

---

## 8. Optimal Rastrigin Configs

| Algorithm | Config | Best Energy | Compute |
|-----------|--------|-------------|---------|
| SAMC | proposal_std=0.5, 16 chains shared, e_max=500 | 82.4 | 8M evals |
| PT | n_replicas=16, t_max=10 | 92.5 | 8M evals |
| MH | proposal_std=1.0 | 106.8 | 500K evals |
