Frequently Asked Questions
==========================

How do I choose the energy range?
----------------------------------

You usually do not need to. When no explicit range is given, :class:`SAMC` and
:class:`SAMCWeights` both use *auto-range mode*: on the first iteration the
sampler creates an :class:`~illuma_samc.partitions.ExpandablePartition` centered
on the initial energy, with 100 bins on each side (201 bins total, width 0.5
each). If the chain wanders outside this range, the partition expands
automatically up to 1000 bins.

.. code-block:: python

    from illuma_samc import SAMC

    # Auto-range: no e_min/e_max needed
    sampler = SAMC(energy_fn=my_energy, dim=2)
    result = sampler.run(n_steps=100_000)

If you already know the energy range (e.g., from a prior run), pass it
explicitly with :class:`~illuma_samc.partitions.UniformPartition` for tighter,
fixed bins:

.. code-block:: python

    from illuma_samc import SAMC
    from illuma_samc.partitions import UniformPartition

    partition = UniformPartition(e_min=0.0, e_max=20.0, n_bins=40)
    sampler = SAMC(energy_fn=my_energy, dim=2, partition_fn=partition)
    result = sampler.run(n_steps=100_000)

Or equivalently via the shorthand constructor arguments:

.. code-block:: python

    sampler = SAMC(energy_fn=my_energy, dim=2, e_min=0.0, e_max=20.0, n_partitions=40)

.. tip::

    Start with auto-range for exploratory runs. Once you know the typical
    energy range from ``result.energy_history``, switch to a fixed
    :class:`~illuma_samc.partitions.UniformPartition` for production runs —
    fixed bins waste no capacity on rarely-visited extremes.


Can I use SAMC for Bayesian posterior sampling?
------------------------------------------------

Yes. SAMC's energy function is just the negative log of whatever distribution
you want to sample from. For a Bayesian model, set::

    energy(x) = -log p(x | data) - log p(x)

where the first term is the negative log-likelihood and the second is the
negative log-prior. SAMC will explore the full posterior, including well-separated
modes that standard MH would miss.

.. code-block:: python

    import torch
    from illuma_samc import SAMC

    def posterior_energy(x):
        log_likelihood = ...   # -log p(data | x)
        log_prior = ...        # -log p(x)
        return log_likelihood + log_prior

    sampler = SAMC(energy_fn=posterior_energy, dim=d)
    result = sampler.run(n_steps=200_000)

    # Recover unweighted posterior samples via importance resampling
    from illuma_samc import SAMCWeights
    wm = SAMCWeights()
    # (if using SAMCWeights directly, call wm.resample after the loop)

    # Or with SAMCResult directly:
    weights = result.importance_weights          # normalized, sums to 1
    idx = torch.multinomial(weights, num_samples=1000, replacement=True)
    posterior_samples = result.samples[idx]

SAMC is especially useful for multimodal posteriors (mixture models, Bayesian
neural networks with symmetries, etc.) where standard MCMC gets trapped in a
single mode.

.. tip::

    Use :meth:`SAMCResult.importance_weights <illuma_samc.sampler.SAMCResult.importance_weights>`
    to compute weighted expectations directly, or call
    :meth:`SAMCWeights.resample` to obtain an unweighted subset drawn from the
    target distribution.


What temperature should I use?
--------------------------------

Temperature ``T`` controls the sharpness of the Boltzmann acceptance
ratio:

.. math::

    \log r = \theta[k_x] - \theta[k_y] + \frac{U(x) - U(x')}{T}

- **T = 1.0** (default): standard Boltzmann acceptance. Good starting point
  for exploration and for problems where the energy scale is not extreme.
- **T < 1** (e.g., 0.1): concentrates sampling near low-energy regions. This
  is exactly the regime where MH gets trapped — SAMC's weight correction
  compensates by penalizing over-visited bins, so it still escapes.
- **T > 1**: smooths the energy landscape, accepting uphill moves more freely.
  Useful as a first pass to locate all basins before running at low T.

.. code-block:: python

    # Exploration at high temperature first
    sampler = SAMC(energy_fn=my_energy, dim=2, temperature=2.0)
    result_warm = sampler.run(n_steps=50_000)

    # Concentrate at low temperature using the same partition
    sampler_cold = SAMC(energy_fn=my_energy, dim=2, temperature=0.1)
    result_cold = sampler_cold.run(n_steps=200_000)

.. tip::

    Low temperature (T = 0.1 or smaller) is where SAMC most outperforms MH.
    If you are running an optimization-style search for the global minimum,
    lower T gives sharper results while SAMC's flat-density mechanism keeps
    the chain from getting trapped.


SAMC vs SAMCWeights — when to use which?
------------------------------------------

:class:`~illuma_samc.sampler.SAMC` is the batteries-included sampler. It owns
the full loop: proposal, energy evaluation, accept/reject, weight update, sample
storage, diagnostics, and burn-in. Use it when you are starting from scratch or
want minimal boilerplate.

.. code-block:: python

    from illuma_samc import SAMC

    sampler = SAMC(energy_fn=my_energy, dim=2, n_partitions=30)
    result = sampler.run(n_steps=100_000, burn_in=1000)
    print(result.acceptance_rate, result.best_energy)

:class:`~illuma_samc.weight_manager.SAMCWeights` is a drop-in correction for
an *existing* MH loop. It only manages the theta vector; you keep full control
of proposals, acceptance logic, and storage. Use it when you already have a
working sampler and want to add SAMC's flat-density property without a rewrite.

.. code-block:: python

    import math
    from illuma_samc import SAMCWeights

    wm = SAMCWeights()

    for t in range(1, n_steps + 1):
        x_new = my_proposal(x)
        fy = energy_fn(x_new)

        log_r = (-fy + fx) / T + wm.correction(fx, fy)   # add correction
        if log_r > 0 or math.log(random()) < log_r:
            x, fx = x_new, fy
        wm.step(t, fx)                                     # update weights

Summary:

+-------------------+-------------------------------------------+
| :class:`SAMC`     | New code, full sampler, rich diagnostics  |
+-------------------+-------------------------------------------+
| :class:`SAMCWeights` | Drop-in for existing MH, two lines     |
+-------------------+-------------------------------------------+


Independent vs shared weights — what is the difference?
---------------------------------------------------------

*New in v0.3.0.*

When you run :class:`SAMC` with ``n_chains > 1``, you can choose between two
multi-chain modes via the ``shared_weights`` parameter.

**Independent chains** (default, ``shared_weights=False``): each chain runs
its own SAMC loop with its own theta vector and partition. Chains do not
communicate. Results are aggregated at the end: ``samples`` has shape
``(n_chains, n_saved, dim)``, ``best_energy`` is the minimum across chains.

.. code-block:: python

    sampler = SAMC(energy_fn=my_energy, dim=2, n_chains=4)
    result = sampler.run(n_steps=100_000)
    # result.samples: (4, n_saved, 2)
    # result.energy_history: (n_steps, 4) — one column per chain

Use independent chains when you want parallel exploration and are happy to
pick the best result afterward. This is the recommended default — it is
trivially parallelizable and the per-chain diagnostics are clean.

**Shared weights** (``shared_weights=True``): all chains share a single theta
vector. Proposals and energy evaluations are batched; accept/reject and weight
updates are sequential per chain within each step to maintain correctness.

.. code-block:: python

    sampler = SAMC(energy_fn=my_energy, dim=2, n_chains=4, shared_weights=True)
    result = sampler.run(n_steps=100_000)

Use shared weights when you want the chains to collectively learn the energy
partition faster (more total updates per step) and do not need per-chain
diagnostics.

.. tip::

    When in doubt, use independent chains (the default). They are easier to
    diagnose and the aggregated ``best_energy`` still benefits from running
    multiple chains.


How many chains should I run?
-------------------------------

For most problems, **4 chains** is a good default. Each chain starts from a
different position and explores independently, so the chance that at least one
chain finds the global minimum increases with the number of chains.

.. code-block:: python

    sampler = SAMC(energy_fn=my_energy, dim=2, n_chains=4)
    result = sampler.run(n_steps=100_000)

Guidelines by problem difficulty:

- **Simple, unimodal** (known good starting region): 1–2 chains.
- **Multimodal, low-dimensional** (d < 20): 4 chains.
- **High-dimensional or deeply multimodal** (d ≥ 20, many basins): 8–16 chains.
- **Bayesian posterior, unknown number of modes**: start with 8 chains and
  check that at least two chains converge to each distinct mode.

.. tip::

    After a run, inspect ``result.energy_history`` per chain. If chains
    disagree substantially on ``best_energy``, add more chains or increase
    ``n_steps``. If all chains agree, fewer chains suffice on the next run.


My sampler has 0% acceptance rate — what is wrong?
----------------------------------------------------

A near-zero acceptance rate typically has one of two causes.

**1. Energy range too narrow.**
If you specified ``e_min`` / ``e_max`` explicitly and most proposals land
outside that range, they are rejected immediately. Check
``result.energy_history`` to see where energies actually fall.

.. code-block:: python

    import torch
    result = sampler.run(n_steps=10_000)
    print(result.energy_history.min(), result.energy_history.max())

Widen the range or switch to auto-range mode (omit ``e_min`` / ``e_max``).

**2. Step size too large.**
If ``proposal_std`` is much larger than the scale of the energy landscape,
nearly all proposals will move to very high-energy regions and be rejected.

.. code-block:: python

    # Too large — nearly all proposals rejected
    sampler = SAMC(energy_fn=my_energy, dim=2, proposal_std=10.0)

    # Use adaptive step size instead
    sampler = SAMC(energy_fn=my_energy, dim=2, proposal_std=1.0, adapt_proposal=True)

The sampler will emit a warning when the acceptance rate falls below 1%::

    UserWarning: Very low acceptance rate (0.0023). Consider increasing
    proposal_std or widening the partition range.

.. tip::

    Target an acceptance rate of 15–50% for Gaussian proposals in continuous
    spaces. Enable ``adapt_proposal=True`` to let the sampler tune
    ``proposal_std`` automatically during a warmup period.


How do I recover the original target distribution?
----------------------------------------------------

SAMC deliberately samples from a *flattened* distribution that visits all
energy bins equally. To recover samples from the true target
:math:`\pi(x) \propto \exp(-U(x)/T)`, you need importance reweighting or
resampling.

**Using SAMCResult (SAMC class)**

:attr:`SAMCResult.importance_weights <illuma_samc.sampler.SAMCResult.importance_weights>`
returns normalized weights that sum to 1:

.. code-block:: python

    result = sampler.run(n_steps=100_000)

    # Weighted expectation under the target
    w = result.importance_weights          # shape (n_saved,)
    mean_x = (w.unsqueeze(-1) * result.samples).sum(0)

    # Unweighted resample from the target
    idx = torch.multinomial(w, num_samples=500, replacement=True)
    target_samples = result.samples[idx]

**Using SAMCWeights (drop-in mode)**

:meth:`SAMCWeights.resample` takes the flat samples and their energies
and returns an unweighted subset:

.. code-block:: python

    # After your MH loop with SAMCWeights:
    flat_samples = torch.stack(collected_samples)   # (n_samples, dim)
    flat_energies = torch.tensor(collected_energies)

    # Recover target samples (acceptance-rejection resampling)
    target_samples = wm.resample(flat_samples, flat_energies)

    # Or compute weighted expectations manually:
    weights = wm.importance_weights(flat_energies)  # sums to 1
    mean_x = (weights.unsqueeze(-1) * flat_samples).sum(0)

The underlying math: SAMC samples from
:math:`q(x) \propto \exp(-U(x)/T - \theta[k(x)])`. The importance weight for
a sample in bin :math:`k` is :math:`w \propto \exp(\theta[k])`, which cancels
the theta correction and recovers :math:`\pi(x)`.

.. tip::

    For optimization (finding the global minimum), you do not need to
    reweight — use ``result.best_x`` and ``result.best_energy`` directly.
    Importance weighting is needed only when computing posterior expectations
    or drawing samples from the true target distribution.
