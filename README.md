# illuma-samc

**Production-quality Stochastic Approximation Monte Carlo (SAMC) for PyTorch.**

SAMC is an adaptive MCMC algorithm that overcomes the local-trap problem by learning energy-dependent sampling weights on the fly. Unlike standard Metropolis-Hastings, SAMC explores all energy levels uniformly, making it effective for multimodal optimization and sampling.

> Based on **Liang, Liu & Carroll** — *Stochastic Approximation in Monte Carlo Computation*, JASA 2007.

## MH Gets Stuck. SAMC Doesn't.

All three algorithms face the same challenge at low temperature (T = 0.1). Same proposal, same energy landscape, same compute budget.

![SAMC vs MH comparison](assets/samc_vs_others.png)

**Metropolis-Hastings** gets trapped in a single energy basin — the energy trace flatlines. **Parallel Tempering** (8 replicas, 8× the compute) slowly escapes but only covers a limited range. **SAMC** learns sampling weights that overcome energy barriers, traversing the full energy landscape uniformly despite the low temperature.

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
    gain=GainSequence("1/t", t0=1000),
    temperature=0.1,  # low temperature for optimization
)
result = sampler.run(n_steps=100_000)
```

### Diagnostics

```python
sampler.plot_diagnostics()  # weight trajectory, energy trace, bin visits, acceptance rate
```

## Gain Schedules

The gain (step-size) sequence controls how quickly SAMC's log-weights adapt. The general form is:

$$\gamma(t) = \frac{\gamma_0}{(\gamma_1 + t)^\alpha}$$

The classic `"1/t"` schedule is the special case γ₀ = t₀, γ₁ = 0, α = 1, which satisfies the convergence conditions from Liang (2007).

| Schedule | Parameters | Description |
|----------|-----------|-------------|
| `"1/t"` (default) | `t0=1000` | γ₀/t — standard SAMC convergence guarantee |
| `"1/t"` | `gamma0, gamma1, alpha` | General power-law γ₀/(γ₁+t)^α |
| `"ramp"` | `rho, tau, warmup, step_scale` | Constant warmup then power-law decay |
| callable | any `(int) → float` | Custom schedule |

```python
# Standard 1/t with warmup at t0=500
GainSequence("1/t", t0=500)

# Slower decay: α = 0.6
GainSequence("1/t", gamma0=100, gamma1=10, alpha=0.6)

# Custom callable
GainSequence(lambda t: 1.0 / t**0.8)
```

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

## Attribution

> **Faming Liang, Chuanhai Liu, and Raymond J. Carroll.** *Stochastic Approximation in Monte Carlo Computation.* Journal of the American Statistical Association, 102(477):305–320, 2007.

See `CITATION.bib` for the BibTeX entry.

## License

Proprietary — Illuma Inc.
