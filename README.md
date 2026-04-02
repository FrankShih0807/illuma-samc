# illuma-samc

[![CI](https://github.com/FrankShih0807/illuma-samc/actions/workflows/ci.yml/badge.svg)](https://github.com/FrankShih0807/illuma-samc/actions/workflows/ci.yml)

**Your MH sampler gets stuck at low temperature. Fix it with two lines.**

![SAMC vs MH comparison](assets/samc_vs_others.png)

Same energy landscape. Same proposal. Same compute budget. At T=0.1, **MH gets trapped** in a single basin. **SAMC explores everything**.

## Two Lines. That's It.

If you already have a Metropolis-Hastings loop, add two lines to get SAMC:

```python
from illuma_samc import SAMCWeights

wm = SAMCWeights()                                        # <- new

for t in range(1, n_steps + 1):
    x_new = propose(x)
    fy = energy_fn(x_new)

    log_r = (-fy + fx) / T + wm.correction(fx, fy)        # <- add correction
    if log_r > 0 or math.log(random()) < log_r:
        x, fx = x_new, fy

    wm.step(t, fx)                                         # <- update weights
```

That's it. No energy range needed -- bins are created automatically on the first step and expand as the sampler explores.

Works batched too -- pass tensors instead of scalars:

```python
wm = SAMCWeights()

# z: (N, dim) -- batch of N states
# energy, energy_prop: (N,) -- one scalar energy per state
for t in range(1, n_steps + 1):
    z_prop = z + 0.25 * torch.randn_like(z)       # (N, dim)
    energy_prop = energy_fn(z_prop)                # (N,)

    log_alpha = (-energy_prop + energy) / T + wm.correction(energy, energy_prop)  # (N,)
    accept = torch.rand_like(log_alpha).log() < log_alpha                         # (N,)

    z = torch.where(accept.unsqueeze(-1), z_prop, z)   # (N, dim)
    energy = torch.where(accept, energy_prop, energy)   # (N,)
    wm.step(t, energy)                                  # (N,)
```

If you know your energy range, pass a `UniformPartition` for precise control:

```python
from illuma_samc import SAMCWeights, UniformPartition, GainSequence

wm = SAMCWeights(
    partition=UniformPartition(e_min=0, e_max=10, n_bins=40),
    gain=GainSequence("1/t", t0=1000),
)
```

### The Payoff

|                  |         MH |       SAMC |
|------------------|-----------:|-----------:|
| Accept rate      |      0.047 |      0.412 |
| Bin flatness     |        N/A |      0.990 |
| Best energy      |     -8.125 |     -8.125 |
| Extra code       |    0 lines |    2 lines |

*2D multimodal benchmark, 500K steps, T=0.1. See [`mh_vs_samc.ipynb`](mh_vs_samc.ipynb) for the full comparison with plots.*

## Install

Requires PyTorch >= 2.0. See [pytorch.org](https://pytorch.org) for installation instructions.

```bash
pip install -e ".[dev]"
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

sampler = SAMC(energy_fn=energy_fn, dim=2)
result = sampler.run(n_steps=100_000)

print(f"Best energy: {result.best_energy:.4f}")
print(f"Acceptance rate: {result.acceptance_rate:.3f}")
```

Multi-chain with shared weights:

```python
sampler = SAMC(energy_fn=energy_fn, dim=2, n_chains=4)
result = sampler.run(n_steps=100_000)
# result.samples has shape (4, n_saved, dim)
```

Full control over proposal, partition, and gain:

```python
from illuma_samc import SAMC, GainSequence, GaussianProposal, UniformPartition

sampler = SAMC(
    energy_fn=energy_fn,
    dim=2,
    proposal_fn=GaussianProposal(step_size=0.5),
    partition_fn=UniformPartition(e_min=0, e_max=10, n_bins=40),
    gain=GainSequence("1/t", t0=1000),
    temperature=0.1,
)
result = sampler.run(n_steps=100_000, burn_in=100)
sampler.plot_diagnostics()
```

## How It Works

SAMC partitions the energy space into $m$ subregions and learns **log-density-of-states estimates** $\theta_t$ that flatten the energy histogram. The MH acceptance ratio gains a weight correction:

$$\alpha = \min\left(1,\; \exp\left(\theta_{J(\mathbf{x})} - \theta_{J(\mathbf{y})} - \frac{U(\mathbf{y}) - U(\mathbf{x})}{T}\right)\right)$$

where $J(\mathbf{x})$ is the subregion index. After each step, the weights update via stochastic approximation:

$$\theta_{t+1} = \theta_t + \gamma_{t+1}(\mathbf{e}_{t+1} - \boldsymbol{\pi})$$

where $\gamma_t$ is a decreasing gain sequence, $\mathbf{e}_{t+1}$ is the indicator for the occupied subregion, and $\boldsymbol{\pi}$ is the desired visit frequency (uniform by default). As $\theta$ converges, the sampler visits all energy levels equally, escaping any local trap.

**Recovering the target distribution.** SAMC samples from a flattened distribution. Reweight each sample by $\exp(\theta_{J(\mathbf{x})})$ to recover the original target, or call `resample()` for unweighted draws.

## FAQ

**Q: How do I choose the energy range?**

A: You don't have to. `SAMCWeights()` auto-creates bins centered on the first energy it sees and expands as needed. If you know the range, pass `partition=UniformPartition(...)` for tighter bins.

**Q: Can I use SAMC for Bayesian posterior sampling?**

A: Yes. Set `energy_fn = -log p(x | data) - log p(x)`. Call `resample()` to recover unweighted posterior draws. Especially useful for multimodal posteriors where standard MCMC gets trapped.

**Q: What temperature should I use?**

A: Start with $T = 1.0$ for exploration, lower it (e.g., $T = 0.1$) to concentrate around the best solutions. Low temperature is exactly where MH gets trapped and SAMC shines.

**Q: `SAMC` vs `SAMCWeights`?**

A: **`SAMCWeights`** drops into your existing MH loop -- you control the proposal and acceptance. **`SAMC`** is batteries-included -- handles the loop, proposals, diagnostics, and burn-in.

## References

> **Faming Liang, Chuanhai Liu, and Raymond J. Carroll.** *Stochastic Approximation in Monte Carlo Computation.* Journal of the American Statistical Association, 102(477):305-320, 2007.

> **Faming Liang.** *On the Use of Stochastic Approximation Monte Carlo for Monte Carlo Integration.* Statistics & Probability Letters, 79(5):581-587, 2009.

See `CITATION.bib` for BibTeX entries.

## Acknowledgments

This is the first scientific research project under the [Illuma](https://github.com/FrankShih0807) umbrella -- and it was built in collaboration with [Claude](https://claude.ai), Anthropic's AI assistant. From architecture decisions to ablation studies to this README, Claude served as a hands-on research and engineering partner throughout the project.

## License

Free for academic and non-commercial research use. Commercial use requires permission.
