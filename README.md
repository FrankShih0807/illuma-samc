# illuma-samc

[![CI](https://github.com/FrankShih0807/illuma-samc/actions/workflows/ci.yml/badge.svg)](https://github.com/FrankShih0807/illuma-samc/actions/workflows/ci.yml)

**Your MH sampler gets stuck at low temperature. Fix it with two lines.**

![SAMC vs MH comparison](assets/samc_vs_others.png)

Same energy landscape. Same proposal. Same compute budget. At T=0.1, **MH gets trapped** in a single basin. **SAMC explores everything**.

## How It Works

Let $p(\mathbf{x}) = c \, p_0(\mathbf{x})$ be the target distribution, where $U(\mathbf{x}) = -\log p_0(\mathbf{x})$ is the energy function. Standard Metropolis-Hastings samples from $p(\mathbf{x})$ directly, but at low temperature, energy barriers become insurmountable and the chain gets trapped in a single mode.

SAMC partitions the energy space into $m$ disjoint subregions $E_1, \ldots, E_m$ and learns **log-density-of-states estimates** $\theta_t = (\theta_{t1}, \ldots, \theta_{tm})$ that flatten the energy histogram. At iteration $t$, the sampler targets the *working density* (Eq. 3 in Liang et al.):

$$p_{\theta_t}(\mathbf{x}) = \frac{1}{Z_t} \sum_{i=1}^{m} \frac{\psi(\mathbf{x})}{e^{\theta_{ti}}} I(\mathbf{x} \in E_i)$$

where $\psi(\mathbf{x})$ is a biasing function (typically set to $p_0(\mathbf{x})$). As $\theta$ converges, $p_{\theta}$ becomes approximately uniform across energy levels -- the sampler visits all subregions equally, escaping any local trap.

The MH acceptance ratio under the working density gains a weight correction:

$$\alpha = \min\left(1,\; \exp\left(\theta_{J(\mathbf{x})} - \theta_{J(\mathbf{y})} - \frac{U(\mathbf{y}) - U(\mathbf{x})}{T}\right)\right)$$

where $J(\mathbf{x})$ denotes the subregion index for state $\mathbf{x}$. After each step, the weights are updated via stochastic approximation (the SAMC update rule):

$$\theta_{t+1} = \theta_t + \gamma_{t+1}(\mathbf{e}_{t+1} - \boldsymbol{\pi})$$

where $\gamma_t = t_0 / \max(t_0, t)$ is a gain sequence satisfying $\sum \gamma_t = \infty$ and $\sum \gamma_t^\zeta < \infty$ for some $\zeta \in (1, 2)$, $\mathbf{e}_{t+1}$ is the indicator vector for the occupied subregion, and $\boldsymbol{\pi} = (\pi_1, \ldots, \pi_m)$ is the desired sampling frequency (uniform by default).

**Recovering the target distribution.** Since SAMC samples from the flattened $p_{\theta}$, not the original $p$, you recover correct samples via binary resampling: keep each sample $\mathbf{x}$ with probability $\propto \exp(\theta_{J(\mathbf{x})})$. The resampled set is an unweighted draw from the target $p$.

> **Faming Liang, Chuanhai Liu, and Raymond J. Carroll.** *Stochastic Approximation in Monte Carlo Computation.* Journal of the American Statistical Association, 102(477):305-320, 2007.

See `CITATION.bib` for the BibTeX entry.

## Two Lines. That's It.

If you already have a Metropolis-Hastings loop, add two lines to get SAMC:

```python
from illuma_samc import SAMCWeights

wm = SAMCWeights.auto(bin_width=0.2)                      # <- new

for t in range(1, n_steps + 1):
    x_new = propose(x)
    fy = energy_fn(x_new)

    log_r = (-fy + fx) / T + wm.correction(fx, fy)        # <- add correction
    if log_r > 0 or math.log(random()) < log_r:
        x, fx = x_new, fy

    wm.step(t, fx)                                         # <- update weights
```

Your loop stays yours. `SAMCWeights` just manages the bin weights that let SAMC overcome energy barriers. Bins grow automatically -- no energy range needed.

**`auto()` is the recommended default** -- zero config, works on any problem. If you know your energy range, `UniformPartition` gives better bin flatness:

```python
from illuma_samc import SAMCWeights, UniformPartition, GainSequence

wm = SAMCWeights(
    partition=UniformPartition(e_min=0, e_max=10, n_bins=20),
    gain=GainSequence("1/t", t0=1000),
)
```

The benchmark plots below use `UniformPartition` with tuned ranges to show SAMC at its best.

### `SAMCWeights` API

| Method | Description |
|--------|-------------|
| `SAMCWeights(partition, gain)` | Constructor. Takes a `Partition` and `GainSequence`. |
| `SAMCWeights.auto(bin_width=0.2)` | Zero-config factory. Bins grow automatically — no energy range needed. |
| `SAMCWeights.from_warmup(energy_fn, dim)` | Runs a short MH warmup to discover the energy range, then creates fixed bins. |
| `wm.correction(fx, fy) -> float` | SAMC weight correction to add to your log acceptance ratio. |
| `wm.step(t, energy)` | Update weights after accept/reject. Call once per iteration. |
| `wm.flatness() -> float` | Bin visit uniformity: 1.0 = perfectly flat, lower = uneven. |
| `wm.importance_weights(energies)` | Normalized importance weights for resampling to target distribution. |
| `wm.resample(samples, energies)` | Importance resampling — recover unweighted draws from the target. |
| `wm.state_dict()` / `wm.load_state_dict(d)` | Checkpoint and restore weights for long runs. |

### The Payoff

|                  |         MH |       SAMC |
|------------------|-----------:|-----------:|
| Accept rate      |      0.047 |      0.436 |
| Bin flatness     |        N/A |      0.951 |
| Bins visited     |        N/A |      42/42 |
| Extra code       |    0 lines |    2 lines |

*2D multimodal benchmark, 500K steps, T=0.1. See `examples/mh_vs_samc.ipynb` for the full comparison.*

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

sampler = SAMC(energy_fn=energy_fn, dim=2, n_partitions=20, e_min=0, e_max=10)
result = sampler.run(n_steps=100_000)

print(f"Best energy: {result.best_energy:.4f}")
print(f"Acceptance rate: {result.acceptance_rate:.3f}")
```

If you don't know the energy range, omit `e_min`/`e_max` and SAMC will auto-detect it via a short warmup:

```python
sampler = SAMC(energy_fn=energy_fn, dim=2)
result = sampler.run(n_steps=100_000)
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
result = sampler.run(n_steps=100_000, burn_in=100)
sampler.plot_diagnostics()
```

## Gain Schedule

The gain $\gamma_t$ controls how quickly SAMC adapts its weights. The default `"1/t"` schedule (Liang 2007) is all you need:

```python
GainSequence("1/t", t0=1000)  # gain = 1.0 for first t0 steps, then decays as t0/t
```

`t0` is the only parameter that matters -- set it to `n_iters / 500` to `n_iters / 100`. You can also pass your own schedule:

```python
GainSequence(lambda t: min(1.0, 1000 / t))  # any (int) -> float callable
```

## Multi-Chain SAMC

Run multiple chains with shared weights for faster bin flattening:

```python
sampler = SAMC(energy_fn=energy_fn, dim=2, n_chains=4)
result = sampler.run(n_steps=100_000)
# result.samples has shape (4, n_saved, dim)
```

Or from the CLI:

```bash
python train.py --algo samc --model 2d --n_chains 4                        # shared weights (default)
python train.py --algo samc --model 2d --n_chains 4 --shared_weights false # independent weights
```

Shared weights: all chains update the same `SAMCWeights` interleaved, flattening bins faster.

## Experiment CLI

```bash
python train.py --algo samc --model 2d                    # basic run
python train.py --algo samc --model 2d --plot_energy      # show energy trace
python train.py --algo samc --model 2d --auto_range       # warmup to discover range, then fixed bins (best flatness)
python train.py --algo samc --model 2d --growing          # bins grow on the fly, no range needed (bin_width=0.2)
python train.py --algo samc --model 2d --growing --bin_width 0.5  # custom bin width
python train.py --algo samc --model 10d --n_chains 4      # multi-chain
python train.py --algo mh --model 2d                      # MH baseline
python train.py --algo pt --model 2d                      # PT baseline
python train.py --algo samc --model 2d --name my_experiment  # named output folder
```

Results saved to `outputs/<model>/<algo>/<name or timestamp>/` with `config.yaml`, `results.json`, and diagnostic plots.

## Examples

```bash
python examples/demo_showcase.py      # All-in-one showcase
python examples/gaussian_mixture.py   # 4-mode Gaussian demo
python examples/multimodal_2d.py      # Reproduce Liang's 2D experiment
```

See `examples/mh_vs_samc.ipynb` for a side-by-side MH vs SAMC comparison.

## FAQ

**Q: How do I choose `e_min` and `e_max`?**

A: If you don't know your energy range, you have three options:
- **`SAMCWeights.auto()`** -- bins grow on the fly, zero config.
- **`SAMCWeights.from_warmup()`** -- runs a quick MH warmup to discover the range, then creates fixed bins.
- **`SAMC(energy_fn=..., dim=...)`** -- omit `e_min`/`e_max` and the sampler auto-detects via warmup.

If you know the range, explicit `UniformPartition` gives the best bin flatness.

**Q: Can I use SAMC for Bayesian posterior sampling?**

A: Yes. Set your energy function to the negative log-posterior: `energy_fn = -log p(x | data) - log p(x)`. SAMC samples from a flattened distribution; call `resample()` to recover unweighted draws from your posterior. This is especially useful for multimodal posteriors where standard MCMC gets trapped.

**Q: What temperature should I use?**

A: Temperature $T$ scales the energy in the Boltzmann factor $\exp(-U(\mathbf{x})/T)$. Low temperature ($T < 1$) sharpens the distribution around minima, making the landscape rugged -- exactly where MH gets trapped and SAMC shines. Start with $T = 1.0$ for exploration, lower it (e.g., $T = 0.1$) to concentrate around the best solutions.

**Q: When should I use `SAMC` vs `SAMCWeights`?**

A: They serve different users:
- **`SAMCWeights`** -- Drop-in for your existing MH loop. You control the proposal, acceptance, and loop. Maximum flexibility for researchers and engineers.
- **`SAMC`** -- Batteries-included sampler. Handles the loop, proposals, diagnostics, and burn-in. Best for getting started quickly.

**Q: How do I choose `t0` in the gain schedule?**

A: Rule of thumb: set `t0` between `n_steps / 500` and `n_steps / 100`. Too small and weights oscillate; too large and adaptation is slow. The default works well for most problems.

**Q: Why is my bin flatness low with `auto()`?**

A: `auto()` uses growing bins that expand on the fly. It needs more steps to achieve the same flatness as a fixed partition because each new bin starts with zero visits. For best flatness, use `UniformPartition` with a known energy range. The tradeoff: `auto()` requires zero configuration but needs longer mixing time.

**Q: What does `from_warmup()` do?**

A: It runs a short MH warmup (default 2000 steps) to discover your energy range, then creates a fixed `UniformPartition` from the observed range. It is the middle ground between `auto()` (zero config, growing bins) and manual range specification.

## Acknowledgments

This is the first scientific research project under the [Illuma](https://github.com/FrankShih0807) umbrella -- and it was built in collaboration with [Claude](https://claude.ai), Anthropic's AI assistant. From architecture decisions to ablation studies to this README, Claude served as a hands-on research and engineering partner throughout the project.

## License

Free for academic and non-commercial research use. Commercial use requires permission.
