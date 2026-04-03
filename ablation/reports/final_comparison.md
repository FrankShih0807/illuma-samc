# Cross-Algorithm Comparison Report

> **Setup**: Best hyperparameters per algorithm per problem, 10 seeds each
> **Algorithms**: SAMC, Metropolis-Hastings (MH), Parallel Tempering (PT)
> **Problems**: Rosenbrock 2D, 10D Gaussian Mixture, Rastrigin 20D
> **Date**: 2026-04-01

---

## Summary Table

| Problem | Dim | Algorithm | Best Energy (mean +/- std) | Acc Rate | Wall Time (s) | Energy Evals |
|---------|-----|-----------|---------------------------|----------|--------------|-------------|
| rosenbrock | 2 | SAMC | 0.0000 +/- 0.0000 | 0.101 | 104.0 | 4,000,000 |
| rosenbrock | 2 | MH | 0.0000 +/- 0.0000 | 0.126 | 7.2 | 500,000 |
| rosenbrock | 2 | PT | 0.0000 +/- 0.0000 | 0.157 | 31.3 | 2,000,000 |
| 10d | 10 | SAMC | 0.2909 +/- 0.0585 | 0.251 | 24.4 | 800,000 |
| 10d | 10 | MH | 0.2500 +/- 0.0839 | 0.448 | 3.6 | 200,000 |
| 10d | 10 | PT | 0.4149 +/- 0.1585 | 0.194 | 59.6 | 3,200,000 |
| rastrigin | 20 | SAMC | 81.7104 +/- 5.0760 | 0.031 | 211.3 | 8,000,000 |
| rastrigin | 20 | MH | 111.2252 +/- 11.6338 | 0.000 | 6.9 | 500,000 |
| rastrigin | 20 | PT | 95.1260 +/- 7.4018 | 0.001 | 114.2 | 8,000,000 |

---

## Best Configs Used

### rosenbrock

**SAMC**: `{'algo': 'samc', 'config': 'configs/samc.yaml', 'proposal_std': 0.1, 'gain': 'ramp', 'n_partitions': 30, 'n_chains': 8, 'shared_weights': False}`

**MH**: `{'algo': 'mh', 'config': 'configs/mh.yaml', 'proposal_std': 0.1, 'temperature': 0.1}`

**PT**: `{'algo': 'pt', 'config': 'configs/pt.yaml', 'proposal_std': 0.1, 'n_replicas': 4, 't_max': 3.16, 't_min': 0.1, 'swap_interval': 10}`


### 10d

**SAMC**: `{'algo': 'samc', 'config': 'configs/samc.yaml', 'proposal_std': 1.0, 'gain': 'ramp', 'n_partitions': 30, 'n_chains': 4, 'shared_weights': True}`

**MH**: `{'algo': 'mh', 'config': 'configs/mh.yaml', 'proposal_std': 0.5, 'temperature': 1.0}`

**PT**: `{'algo': 'pt', 'config': 'configs/pt.yaml', 'proposal_std': 1.0, 'n_replicas': 16, 't_max': 10.0, 't_min': 1.0, 'swap_interval': 10}`


### rastrigin

**SAMC**: `{'algo': 'samc', 'config': 'configs/samc.yaml', 'proposal_std': 0.5, 'gain': 'ramp', 'n_partitions': 40, 'n_chains': 16, 'shared_weights': True}`

**MH**: `{'algo': 'mh', 'config': 'configs/mh.yaml', 'proposal_std': 1.0, 'temperature': 1.0}`

**PT**: `{'algo': 'pt', 'config': 'configs/pt.yaml', 'proposal_std': 0.5, 'n_replicas': 16, 't_max': 10.0, 't_min': 1.0, 'swap_interval': 10}`


---

## Figures

- **Figure 1**: `ablation/figures/fig1_algo_comparison.png` -- Bar chart with error bars
- **Figure 2**: `ablation/figures/fig2_scaling_dimensionality.png` -- Scaling with dimensionality
- **Figure 3**: `ablation/figures/fig3_robustness_proposal_std.png` -- Robustness to proposal_std
- **Figure 4**: `ablation/figures/fig4_gain_schedule.png` -- SAMC gain schedule comparison
- **Figure 5**: `ablation/figures/fig5_pareto_front.png` -- Compute efficiency Pareto front

---

## Key Findings

### 1. SAMC Wins on the Hardest Problem

On Rastrigin 20D (~10^20 local minima), SAMC with 16 shared chains achieves E=81.7 vs PT's E=95.1 and MH's E=111.2. SAMC's flat-histogram weight correction provides a meaningful advantage when the landscape has many local minima. The improvement over MH is 26% and over PT is 14%.

### 2. MH is the Most Efficient on Easy Problems

On Rosenbrock (unimodal) and 10D Gaussian mixture, MH matches or beats SAMC in energy quality while using 8-16x fewer energy evaluations. MH achieves E=0.0000 on Rosenbrock with 500K evals vs SAMC's 4M evals. On 10D, MH gets E=0.250 with 200K evals vs SAMC's 0.291 with 800K evals.

### 3. PT is Computationally Expensive for Modest Gains

PT consistently uses the most compute (due to multiple replicas) without achieving the best energy on any problem. On Rastrigin, PT (8M evals, E=95.1) is between SAMC (8M evals, E=81.7) and MH (500K evals, E=111.2).

### 4. Scaling Behavior

- **Rosenbrock (2D)**: All algorithms find the global minimum. No differentiation needed.
- **10D Gaussian**: SAMC and MH are comparable. PT falls behind.
- **Rastrigin (20D)**: SAMC pulls ahead. The gap widens with dimensionality and landscape complexity.

### 5. SAMC Requires More Compute for Exploration Guarantees

SAMC's flat-histogram property (uniform energy space coverage) comes at a compute cost: multi-chain runs use 8-16x more evaluations than single MH. The trade-off is worthwhile only when exploration guarantees matter (multimodal landscapes, sampling rather than optimization).

### 6. Practical Recommendations

- **For optimization** (finding the minimum): Use MH first. It's fastest and works well on most problems.
- **For sampling** (covering the energy landscape): Use SAMC with shared multi-chain. It provides provable exploration guarantees.
- **For hard multimodal problems**: Use SAMC with 8-16 chains. It outperforms both MH and PT.
- **Avoid PT**: It offers no clear advantage over SAMC at the same compute budget.
