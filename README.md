# illuma-samc

**Production-quality Stochastic Approximation Monte Carlo (SAMC) for PyTorch.**

Based on **SAMC by Faming Liang, Chuanhai Liu, and Raymond J. Carroll** — *"Stochastic Approximation in Monte Carlo Computation"*, Journal of the American Statistical Association, 2007.

SAMC is an adaptive MCMC algorithm that overcomes the local-trap problem by learning energy-dependent sampling weights on the fly. Unlike standard Metropolis-Hastings, SAMC explores all energy levels uniformly, making it effective for multimodal optimization and sampling.

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
python examples/gaussian_mixture.py   # 4-mode Gaussian demo
python examples/multimodal_2d.py      # Reproduce Liang's 2D experiment
```

## Attribution

This implementation is based on the SAMC algorithm developed by:

> **Faming Liang, Chuanhai Liu, and Raymond J. Carroll.** *Stochastic Approximation in Monte Carlo Computation.* Journal of the American Statistical Association, 102(477):305–320, 2007.

See `CITATION.bib` for the BibTeX entry.

## License

Proprietary — Illuma Inc.
