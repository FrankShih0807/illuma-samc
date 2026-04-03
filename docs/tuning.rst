Tuning Guide
============

This guide walks you through tuning :class:`SAMC` for your specific problem.
Start from the top and stop as soon as things work — most problems need no
more than a handful of adjustments.

.. note::

   All recommendations here are based on ablation studies across six problems
   (2D to 100D), twelve parameter groups, and over 600 runs, plus a dedicated
   robust-defaults validation across 60 runs.


Start Here: Zero-Config Mode
-----------------------------

For the majority of problems up to ~20 dimensions, the defaults work out of
the box. Enable adaptive proposals and let the sampler handle itself:

.. code-block:: python

   from illuma_samc import SAMC

   sampler = SAMC(
       energy_fn=energy_fn,
       dim=dim,
       n_chains=4,
       adapt_proposal=True,
       adapt_warmup=2000,
   )
   result = sampler.run(n_steps=200_000)

With ``adapt_proposal=True``, the Gaussian step size is tuned automatically
via dual averaging — you can start with any initial value and it will converge
to the right size. The energy range is also handled automatically for dim < 20:
an expandable partition is centered on the first energy seen and grows as
needed.

**When zero-config is enough:**

- Continuous problems in less than ~20 dimensions
- Energy landscape with modes that are reasonably reachable from a random start
- No strong prior knowledge about the energy range

If you see poor flatness, slow mixing, or a very low acceptance rate (< 10%),
continue reading.


When to Tune
------------

Use the following decision tree to decide what to tune:

**My goal is optimization (finding the lowest energy):**
   Start with plain MH — it is fastest for optimization. Use SAMC only if
   MH gets trapped in local minima.

**My goal is sampling (covering the energy landscape):**
   Use SAMC. It provides flat-histogram exploration guarantees that MH cannot
   match in high dimensions.

**My problem is high-dimensional (dim >= 20) with widely separated modes:**
   Set ``e_min`` / ``e_max`` explicitly (see `Energy Range for High Dimensions`_).
   Auto-range breaks down here.

**My acceptance rate is very low (< 10%):**
   Increase ``proposal_std`` or enable ``adapt_proposal=True``.

**My bin visit histogram is uneven (low flatness):**
   First check the energy range — bins outside the true energy support are
   never visited, which skews the histogram. Then consider increasing ``n_chains``.

**My weights have not converged after many steps:**
   Increase ``t0`` in the gain schedule (see `Gain Schedule and t0`_).


Parameter Sensitivity Ranking
------------------------------

Listed from most impactful to least, based on ablation results:

1. **Energy range** (``e_min`` / ``e_max``) — dominant for dim >= 20
2. **Gain schedule** (``gain``) — ``"ramp"`` vs ``"1/t"``
3. **Gain t0** (``gain_kwargs={"t0": ...}``) — convergence speed
4. **Number of bins** (``n_partitions``) — resolution of energy partition
5. **Number of chains** (``n_chains``) — exploration diversity

Each is described in detail below.


Energy Range for High Dimensions
---------------------------------

Auto-range (the default when ``e_min`` / ``e_max`` are not specified) works
well for dim < 20. In higher dimensions, warmup trajectories over-explore
the energy space and produce a partition that is too wide. Bins in regions
with no probability mass are never visited, which breaks the flatness
guarantee.

**How auto-range fails in high dimensions:**

When the partition spans a wider range than the true energy support, most
bins accumulate zero visits. The weight vector ``theta`` cannot converge
because it never receives gradient signal from those empty bins. The result
is poor mixing and misleadingly low flatness scores.

**How to set the range manually:**

Run a short MH probe first to get an empirical energy distribution:

.. code-block:: python

   import torch
   from illuma_samc import SAMC

   # Short MH probe — no SAMC weights, just plain MH
   probe = SAMC(
       energy_fn=energy_fn,
       dim=dim,
       n_chains=1,
       adapt_proposal=True,
   )
   probe_result = probe.run(n_steps=10_000)
   energies = probe_result.energy_history

   e_min = energies.min().item() * 0.95   # a bit below the observed min
   e_max = torch.quantile(energies, 0.92).item()   # 90–95th percentile

   sampler = SAMC(
       energy_fn=energy_fn,
       dim=dim,
       n_chains=4,
       n_partitions=40,
       e_min=e_min,
       e_max=e_max,
       adapt_proposal=True,
   )
   result = sampler.run(n_steps=200_000)

.. tip::

   Use the **90th–95th percentile** of observed energies as ``e_max``, not
   the maximum. Extreme tail energies appear rarely and consume bins that
   contribute almost nothing to the estimate. A range that is too wide hurts
   nearly as much as one that is too narrow.

**Rule of thumb:** set ``e_min`` slightly below the observed minimum
(~5% margin) and ``e_max`` at the 90th–95th percentile.


Gain Schedule and t0
~~~~~~~~~~~~~~~~~~~~

The gain schedule controls how fast the log-weights converge. Two presets
are available:

- ``"ramp"`` (default) — constant gain during a warmup phase, then power-law
  decay. Matches the reference implementation and works well in practice.
- ``"1/t"`` — standard SAMC schedule from Liang (2007). Theoretically
  guarantees convergence but can be slower in the warmup phase.

The most important gain parameter is ``t0`` — the iteration at which the
schedule begins to decay. Setting ``t0`` too small (e.g., 100) causes the
gain to shrink before the weights have had time to converge, leading to
poor flatness:

.. code-block:: python

   from illuma_samc import SAMC

   # Slower decay — better for difficult problems
   sampler = SAMC(
       energy_fn=energy_fn,
       dim=dim,
       n_chains=4,
       gain="1/t",
       gain_kwargs={"t0": 5000},   # default is 1000
   )

.. tip::

   Use ``t0 >= 1000``. For problems where weight convergence is slow
   (check ``result.log_weights`` variance over time), try ``t0 = 5000``
   or higher.


Number of Bins
~~~~~~~~~~~~~~

``n_partitions`` (alias ``n_bins``) controls the resolution of the energy
partition. More bins give finer-grained weight estimates but require more
iterations to populate:

- **Default:** 42 bins (works for most problems)
- **Low-dim (< 10D):** 20–30 bins is sufficient
- **High-dim (>= 20D):** 40–50 bins recommended; scale up with dimensionality

.. code-block:: python

   sampler = SAMC(
       energy_fn=energy_fn,
       dim=50,
       n_partitions=45,   # or equivalently: n_bins=45
       e_min=0.0,
       e_max=60.0,
   )

Do not set ``n_partitions`` higher than roughly ``n_steps / 1000`` — each
bin needs at least ~1000 visits to produce a reliable weight estimate.


Number of Chains
~~~~~~~~~~~~~~~~

Multiple chains with independent weight vectors improve exploration diversity:

.. code-block:: python

   sampler = SAMC(
       energy_fn=energy_fn,
       dim=dim,
       n_chains=8,       # independent chains by default
       adapt_proposal=True,
   )

- **Default:** 1 chain
- **Recommended:** 4–8 chains for most problems
- **Hard multimodal (20D+):** 8–16 chains

By default (``shared_weights=False``), each chain runs independently with
its own weight vector and partition. The result aggregates across chains:
``best_energy`` is the minimum across all chains, ``samples`` has shape
``(n_chains, n_saved, dim)``.

.. tip::

   Independent chains (default) outperform shared weights on flatness in
   ablation experiments. Use ``shared_weights=True`` only if you need a
   single converged weight vector — for example, when you plan to reuse
   ``log_weights`` as a warm start.


Cross-Algorithm Comparison
---------------------------

The table below compares :class:`SAMC`, standard Metropolis-Hastings (MH),
and Parallel Tempering (PT) on best energy found. All methods use identical
compute budgets (800K energy evaluations: 200K iterations × 4 chains),
the same random starting points, and adaptive Gaussian proposals.
SAMC uses zero-config defaults — no hand-tuned energy range or partition.
Lower is better.

.. list-table:: Best energy found (mean over 5 seeds)
   :header-rows: 1
   :widths: 30 10 15 15 20

   * - Problem
     - Dim
     - MH
     - PT
     - SAMC (default)
   * - Gaussian Mixture
     - 10
     - **0.24**
     - 0.27
     - 0.31
   * - Gaussian Mixture
     - 50
     - 8.99
     - 9.61
     - **3.43**
   * - Gaussian Mixture
     - 100
     - 24.64
     - 27.20
     - **13.40**
   * - Rastrigin
     - 20
     - 22.86
     - **19.34**
     - 21.29

**Reading the table:**

At low dimension (10D), all three algorithms perform similarly — MH converges
quickly and SAMC's flat-histogram overhead is a slight disadvantage. SAMC's
advantage grows sharply with dimension: 2.6x better energy than MH at 50D
and 1.8x better at 100D, without any tuning. PT tracks MH and never escapes
the high-dimensional curse.

On Rastrigin (highly multimodal, structured), PT has a slight edge at 20D.
Setting ``e_min`` / ``e_max`` explicitly would improve SAMC's result here
(see `Energy Range for High Dimensions`_).

See ``benchmarks/three_way.py`` for the reproducible benchmark and
``ablation/reports/`` for full figures.


SAMC vs MH vs PT Decision Guide
---------------------------------

**Use MH when:**

- Your goal is optimization (not sampling) and the problem is < 20D
- You need maximum speed with minimal setup
- The energy landscape is unimodal or has well-separated modes you don't
  need to cover

**Use PT when:**

- You need sampling on a structured multimodal landscape in < 20D
- You have a good temperature ladder and the modes are known
- Runtime is not a bottleneck (PT requires multiple temperature replicas)

**Use SAMC when:**

- You need reliable flat-histogram sampling across all energy levels
- The problem is >= 20D with many local minima
- You want a single algorithm that works without hand-tuning temperature
  ladders or restart schedules
- You are doing Bayesian posterior sampling with a multimodal posterior
  (set ``energy_fn = -log_likelihood - log_prior``; call
  ``result.importance_weights`` to recover target-distribution samples)

.. note::

   :class:`SAMCWeights` is the lower-level building block: it drops into any
   existing MH loop with two lines of code. Use :class:`SAMC` for a
   batteries-included sampler that handles the loop, proposals, diagnostics,
   and burn-in automatically.
