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

sampler = SAMC(energy_fn=energy_fn, dim=2, n_chains=4, adapt_proposal=True)
result = sampler.run(n_steps=100_000)

print(f"Best energy: {result.best_energy:.4f}")
print(f"Acceptance rate: {result.acceptance_rate:.3f}")
```

## High-Dimensional Comparison

Best energy found, 200K iterations x 4 chains, 5 seeds. All algorithms use identical compute budgets (800K energy evals), same starting points, and adaptive proposals. SAMC is zero-config.

| Problem | Dim | MH | PT | SAMC (default) |
|---------|-----|----|----|----------------|
| 10D Gaussian Mixture | 10 | **0.24** | 0.27 | 0.31 |
| 50D Gaussian Mixture | 50 | 8.99 | 9.61 | **3.43** |
| 100D Gaussian Mixture | 100 | 24.64 | 27.20 | **13.40** |
| Rastrigin 20D | 20 | 22.86 | **19.34** | 21.29 |

Lower is better. Bold = best per row. SAMC achieves **2.6x better energy than MH at 50D** and **1.8x better at 100D**, without any tuning. See `benchmarks/three_way.py` for the reproducible benchmark.

## Documentation

For full documentation including tutorials, tuning guide, and API reference:

- [**Quickstart**](docs/quickstart.rst) -- usage patterns for `SAMC` and `SAMCWeights`
- [**Tuning Guide**](docs/tuning.rst) -- parameter sensitivity, energy range selection, when to use SAMC vs MH
- [**How It Works**](docs/how_it_works.rst) -- the SAMC algorithm explained with math
- [**FAQ**](docs/faq.rst) -- energy range, Bayesian posteriors, temperature, multi-chain
- **Tutorials:** [Drop-in SAMCWeights](docs/tutorials/drop_in.rst) | [Multi-chain](docs/tutorials/multi_chain.rst) | [Bayesian inference](docs/tutorials/bayesian.rst)
- [**API Reference**](docs/api.rst) -- full class and method documentation

## References

> **Faming Liang, Chuanhai Liu, and Raymond J. Carroll.** *Stochastic Approximation in Monte Carlo Computation.* Journal of the American Statistical Association, 102(477):305-320, 2007.

> **Faming Liang.** *On the Use of Stochastic Approximation Monte Carlo for Monte Carlo Integration.* Statistics & Probability Letters, 79(5):581-587, 2009.

See `CITATION.bib` for BibTeX entries.

## Acknowledgments

This is the first scientific research project under the [Illuma](https://github.com/FrankShih0807) umbrella -- and it was built in collaboration with [Claude](https://claude.ai), Anthropic's AI assistant. From architecture decisions to ablation studies to this README, Claude served as a hands-on research and engineering partner throughout the project.

## License

Free for academic and non-commercial research use. Commercial use requires permission.
