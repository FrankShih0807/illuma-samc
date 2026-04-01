# illuma-samc

**Your MH sampler gets stuck at low temperature. Fix it with two lines.**

![SAMC vs MH comparison](assets/samc_vs_others.png)

Same energy landscape. Same proposal. Same compute budget. At T=0.1, **MH gets trapped** in a single basin. **SAMC explores everything**.

## Two Lines. That's It.

If you already have a Metropolis-Hastings loop, add two lines to get SAMC:

```python
from illuma_samc import SAMCWeights, UniformPartition, GainSequence

wm = SAMCWeights(                                          # ← new
    partition=UniformPartition(e_min=0, e_max=10, n_bins=20),
    gain=GainSequence("1/t", t0=1000),
)

for t in range(1, n_steps + 1):
    x_new = propose(x)
    fy = energy_fn(x_new)

    log_r = (-fy + fx) / T + wm.correction(fx, fy)        # ← add correction
    if log_r > 0 or math.log(random()) < log_r:
        x, fx = x_new, fy

    wm.step(t, fx)                                         # ← update weights
```

Your loop stays yours. `SAMCWeights` just manages the bin weights that let SAMC overcome energy barriers.

### The Payoff

|                  |         MH |       SAMC |
|------------------|-----------:|-----------:|
| Accept rate      |      0.047 |      0.436 |
| Bins visited     |      24/42 |      42/42 |
| Bin flatness     |        N/A |      0.951 |
| Extra code       |    0 lines |    2 lines |

*2D multimodal benchmark, 500K steps, T=0.1. See `examples/mh_vs_samc.ipynb` for the full comparison.*

## Install

```bash
pip install -e .            # core (torch only)
pip install -e ".[dev]"     # + ruff, pytest, matplotlib, tqdm
```

## Starting from Scratch?

If you don't have an existing MH loop, use the `SAMC` class directly:

```python
import torch
from illuma_samc import SAMC

def energy_fn(x):
    return torch.min(
        0.5 * torch.sum((x - 2) ** 2),
        0.5 * torch.sum((x + 2) ** 2),
    )

sampler = SAMC(energy_fn=energy_fn, dim=2, n_partitions=20, e_min=0, e_max=10)
result = sampler.run(n_steps=100_000)

print(f"Best energy: {result.best_energy:.4f}")
print(f"Acceptance rate: {result.acceptance_rate:.3f}")
```

Full control over proposal, partition, and gain schedule:

```python
from illuma_samc import SAMC, GainSequence, GaussianProposal, UniformPartition

sampler = SAMC(
    energy_fn=energy_fn,
    dim=2,
    proposal_fn=GaussianProposal(step_size=0.5),
    partition_fn=UniformPartition(e_min=0, e_max=10, n_bins=20),
    gain=GainSequence("1/t", t0=1000),
    temperature=0.1,
)
result = sampler.run(n_steps=100_000)
sampler.plot_diagnostics()
```

## Gain Schedules

The gain (step-size) controls how quickly SAMC adapts its weights:

$$\gamma(t) = \frac{\gamma_0}{(\gamma_1 + t)^\alpha}$$

| Schedule | Parameters | Description |
|----------|-----------|-------------|
| `"1/t"` (default) | `t0=1000` | γ₀/t — standard convergence guarantee |
| `"1/t"` | `gamma0, gamma1, alpha` | General power-law γ₀/(γ₁+t)^α |
| `"ramp"` | `rho, tau, warmup, step_scale` | Constant warmup then power-law decay |
| callable | any `(int) → float` | Custom schedule |

## Partition Types

- **`UniformPartition`** — Linear energy bins (default)
- **`AdaptivePartition`** — Recomputes boundaries from visited energies
- **`QuantilePartition`** — Boundaries from warmup energy quantiles

## Examples

```bash
python examples/demo_showcase.py      # All-in-one showcase
python examples/gaussian_mixture.py   # 4-mode Gaussian demo
python examples/multimodal_2d.py      # Reproduce Liang's 2D experiment
```

See `examples/mh_vs_samc.ipynb` for a side-by-side MH vs SAMC comparison.

## How It Works

SAMC (Stochastic Approximation Monte Carlo) learns energy-dependent sampling weights that flatten the energy histogram. Where MH gets trapped because it can't cross energy barriers, SAMC's learned weights provide the "boost" needed to escape — giving you uniform exploration across all energy levels.

> **Faming Liang, Chuanhai Liu, and Raymond J. Carroll.** *Stochastic Approximation in Monte Carlo Computation.* Journal of the American Statistical Association, 102(477):305–320, 2007.

See `CITATION.bib` for the BibTeX entry.

## License

Free for academic and non-commercial research use. Commercial use requires permission.
