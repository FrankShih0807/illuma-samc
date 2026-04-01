# illuma-samc

[![CI](https://github.com/FrankShih0807/illuma-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/FrankShih0807/illuma-lab/actions/workflows/ci.yml)

**Your MH sampler gets stuck at low temperature. Fix it with two lines.**

![SAMC vs MH comparison](assets/samc_vs_others.png)

Same energy landscape. Same proposal. Same compute budget. At T=0.1, **MH gets trapped** in a single basin. **SAMC explores everything**.

## Two Lines. That's It.

If you already have a Metropolis-Hastings loop, add two lines to get SAMC:

```python
from illuma_samc import SAMCWeights

wm = SAMCWeights.auto(bin_width=0.2)                      # ← new

for t in range(1, n_steps + 1):
    x_new = propose(x)
    fy = energy_fn(x_new)

    log_r = (-fy + fx) / T + wm.correction(fx, fy)        # ← add correction
    if log_r > 0 or math.log(random()) < log_r:
        x, fx = x_new, fy

    wm.step(t, fx)                                         # ← update weights
```

Your loop stays yours. `SAMCWeights` just manages the bin weights that let SAMC overcome energy barriers. Bins grow automatically — no energy range needed.

**`auto()` is the recommended default** — zero config, works on any problem. If you know your energy range, `UniformPartition` gives better bin flatness:

```python
from illuma_samc import SAMCWeights, UniformPartition, GainSequence

wm = SAMCWeights(
    partition=UniformPartition(e_min=0, e_max=10, n_bins=20),
    gain=GainSequence("1/t", t0=1000),
)
```

The benchmark plots below use `UniformPartition` with tuned ranges to show SAMC at its best.

### The Payoff

|                  |         MH |       SAMC |
|------------------|-----------:|-----------:|
| Accept rate      |      0.047 |      0.436 |
| Bin flatness     |        N/A |      0.951 |
| Bins visited     |        N/A |      42/42 |
| Extra code       |    0 lines |    2 lines |

*2D multimodal benchmark, 500K steps, T=0.1. See `examples/mh_vs_samc.ipynb` for the full comparison.*

## Install

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

## Gain Schedule

The gain controls how quickly SAMC adapts its weights. The default `"1/t"` schedule (Liang 2007) is all you need:

```python
GainSequence("1/t", t0=1000)  # gain = 1.0 for first t0 steps, then decays as t0/t
```

`t0` is the only parameter that matters — set it to `n_iters / 500` to `n_iters / 100`. You can also pass your own schedule:

```python
GainSequence(lambda t: min(1.0, 1000 / t))  # any (int) -> float callable
```

## Multi-Chain SAMC

Run multiple chains with shared weights for faster bin flattening:

```python
# In train.py
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

## How It Works

SAMC (Stochastic Approximation Monte Carlo) learns energy-dependent sampling weights that flatten the energy histogram. Where MH gets trapped because it can't cross energy barriers, SAMC's learned weights provide the "boost" needed to escape — giving you uniform exploration across all energy levels.

> **Faming Liang, Chuanhai Liu, and Raymond J. Carroll.** *Stochastic Approximation in Monte Carlo Computation.* Journal of the American Statistical Association, 102(477):305-320, 2007.

See `CITATION.bib` for the BibTeX entry.

## License

Free for academic and non-commercial research use. Commercial use requires permission.
