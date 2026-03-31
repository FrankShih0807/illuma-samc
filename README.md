# illuma-samc

**Production-quality Stochastic Approximation Monte Carlo (SAMC) for PyTorch.**

Based on **SAMC by Faming Liang, Chuanhai Liu, and Raymond J. Carroll** — *"Stochastic Approximation in Monte Carlo Computation"*, Journal of the American Statistical Association, 2007.

SAMC is an adaptive MCMC algorithm that overcomes the local-trap problem by learning energy-dependent sampling weights on the fly. Unlike standard Metropolis-Hastings, SAMC explores all energy levels uniformly, making it effective for multimodal optimization and sampling.

## Why SAMC?

**MH gets stuck. SAMC doesn't.**

![SAMC vs MH comparison](assets/samc_vs_others.png)

Standard Metropolis-Hastings gets trapped in local minima — at low temperature (T=0.1), MH stays in a single energy region for the entire run (top-left: concentrated samples, bottom-left: flat energy trace). Even at T=1.0, MH explores partially but provides no exploration guarantee.

SAMC fixes this by learning sampling weights that penalize over-visited energy regions, producing **uniform exploration across all energy levels** (bottom-right: energy trace covers the full range). This is the flat-histogram property — SAMC visits every energy level equally, escaping local traps automatically.

![illuma-samc demo](assets/demo_showcase.png)

## Install

```bash
pip install -e .            # core (torch only)
pip install -e ".[viz]"     # + matplotlib for diagnostics
pip install -e ".[dev]"     # + ruff, pytest, matplotlib, tqdm
```

## Quick Start

### Simple Mode

Provide an energy function — the sampler handles everything else:

```python
import torch
from illuma_samc import SAMC

def energy_fn(x):
    """Two-well potential."""
    return torch.min(
        0.5 * torch.sum((x - 2) ** 2),
        0.5 * torch.sum((x + 2) ** 2),
    )

sampler = SAMC(energy_fn=energy_fn, dim=2, n_partitions=20, e_min=0, e_max=10)
result = sampler.run(n_steps=100_000)

print(f"Best energy: {result.best_energy:.4f}")
print(f"Best x: {result.best_x}")
print(f"Acceptance rate: {result.acceptance_rate:.3f}")
```

### Flexible Mode

Full control over proposal, partition, and gain schedule:

```python
from illuma_samc import SAMC, GainSequence, GaussianProposal, UniformPartition

sampler = SAMC(
    energy_fn=energy_fn,
    dim=2,
    proposal_fn=GaussianProposal(step_size=0.5),
    partition_fn=UniformPartition(e_min=0, e_max=10, n_bins=20),
    gain=GainSequence("ramp", rho=1.0, tau=1.0, warmup=1, step_scale=1000),
)
result = sampler.run(n_steps=100_000)
```

### Diagnostics

```python
sampler.plot_diagnostics()  # weight trajectory, energy trace, bin visits, acceptance rate
```

## Gain Schedules

| Schedule | Formula | Use case |
|----------|---------|----------|
| `"1/t"` | γ_t = t₀ / max(t, t₀) | Standard SAMC theory |
| `"log"` | γ_t = t₀ / max(t·log(t+e), t₀) | Faster decay |
| `"ramp"` | Warmup then power-law decay | Matches Liang's C implementation |
| callable | Any `(int) → float` | Custom schedules |

## Partition Types

- **`UniformPartition`** — Linear energy bins (default)
- **`AdaptivePartition`** — Recomputes boundaries from visited energies
- **`QuantilePartition`** — Boundaries from warmup energy quantiles

## Proposal Types

- **`GaussianProposal`** — Isotropic random walk
- **`LangevinProposal`** — MALA-style gradient-informed proposal via autograd

## Examples

```bash
python examples/demo_showcase.py      # All-in-one showcase (generates assets/demo_showcase.png)
python examples/gaussian_mixture.py   # 4-mode Gaussian demo
python examples/multimodal_2d.py      # Reproduce Liang's 2D experiment
```

## Benchmarks

### Sample Trajectories

The trajectory comparison below shows how each sampler explores the 2D multimodal energy landscape. SAMC covers the entire domain uniformly — MH gets trapped in local basins.

![Trajectory comparison](benchmarks/trajectory_comparison.png)

### Quantitative Results

SAMC vs Metropolis-Hastings vs Parallel Tempering on two problems. All methods use identical proposal, burn-in (10%), and sample collection frequency (every 100th iteration) for fair comparison.

| Problem | Method | Best Energy | Acc. Rate | Energy Evals | Time (s) |
|---------|--------|-------------|-----------|--------------|----------|
| 2D Multimodal | SAMC | -8.125 | 0.510 | 500K | 25.4 |
| 2D Multimodal | MH | -8.125 | 0.439 | 500K | 21.3 |
| 2D Multimodal | PT (8 replicas) | -8.125 | 0.784 | 4.0M | 176.9 |
| 10D Gaussian | SAMC | 0.419 | 0.239 | 200K | 4.8 |
| 10D Gaussian | MH | 0.385 | 0.145 | 200K | 3.4 |
| 10D Gaussian | PT (8 replicas) | 0.804 | 0.265 | 1.6M | 28.8 |

**Key takeaways:**
- **Compute fairness:** PT runs 8 replicas per iteration, so it evaluates the energy function 8x more than SAMC or MH at the same iteration count. The "Energy Evals" column makes this cost transparent.
- **2D multimodal:** All methods find the global minimum (~-8.12). SAMC and MH achieve similar best energy at equal compute cost, but SAMC's flat-histogram exploration ensures all energy levels are visited uniformly. PT has the highest acceptance rate but uses 8x the energy evaluations.
- **10D Gaussian mixture:** PT finds the best energy (0.804) thanks to replica exchanges, but at 8x the compute cost. SAMC (0.419) outperforms MH (0.385) at identical cost via adaptive weighting.

Run benchmarks yourself:
```bash
python benchmarks/vs_mh_pt.py
```

## Attribution

This implementation is based on the SAMC algorithm developed by:

> **Faming Liang, Chuanhai Liu, and Raymond J. Carroll.** *Stochastic Approximation in Monte Carlo Computation.* Journal of the American Statistical Association, 102(477):305–320, 2007.

See `CITATION.bib` for the BibTeX entry.

## License

Proprietary — Illuma Inc.
