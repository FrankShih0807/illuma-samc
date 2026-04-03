Running Multiple Chains
=======================

.. contents:: On this page
   :local:
   :depth: 2

Why multiple chains?
--------------------

A single SAMC chain can get stuck — spending too long in one region of the
energy landscape before the weight vector pushes it elsewhere. Running several
chains in parallel addresses this in three ways:

* **Escape local traps.** Chains start from different initial positions, so at
  least one is likely to find a low-energy basin early. The best solution found
  across all chains is returned.
* **Statistical independence.** Samples from separate chains are uncorrelated
  (independent weights mode). This makes mixing diagnostics meaningful and
  posterior estimates more reliable.
* **Better coverage.** Multiple chains collectively visit more of the energy
  landscape per wall-clock run, especially in high-dimensional or multimodal
  problems.

:class:`SAMC` supports two multi-chain modes, selected via the
``shared_weights`` parameter.


Independent weights (default)
------------------------------

When ``n_chains > 1`` and ``shared_weights=False`` (the default), each chain
runs its own full SAMC sampler — separate ``theta`` vector, separate partition,
separate adaptive proposal. The chains never communicate during the run.
After all chains finish, results are aggregated:

* ``result.samples`` shape: ``(n_chains, n_saved, dim)``
* ``result.best_energy``: minimum across all chains
* ``result.best_x``: point that achieved ``best_energy``
* ``result.log_weights`` / ``result.bin_counts``: from the best chain
* ``result.acceptance_rate``: mean over all chains
* ``result.energy_history`` shape: ``(n_steps, n_chains)``

This is the right mode for most problems.

.. code-block:: python

    import torch
    from illuma_samc import SAMC

    def energy_fn(x: torch.Tensor) -> torch.Tensor:
        # 2D mixture: two wells at (-3, 0) and (3, 0)
        d1 = ((x - torch.tensor([-3.0, 0.0])) ** 2).sum()
        d2 = ((x - torch.tensor([ 3.0, 0.0])) ** 2).sum()
        return -torch.log(torch.exp(-0.5 * d1) + torch.exp(-0.5 * d2))

    sampler = SAMC(energy_fn=energy_fn, dim=2, n_chains=4, adapt_proposal=True)
    result = sampler.run(n_steps=100_000)

    print(result.samples.shape)        # torch.Size([4, 1000, 2])
    print(f"Best energy: {result.best_energy:.4f}")
    print(f"Mean acceptance: {result.acceptance_rate:.3f}")

Each chain adapts its proposal step size independently, so one chain that
happens to start in a flat region will not drag down a chain that is already
mixing well.


Shared weights
--------------

When ``shared_weights=True``, all chains update a single shared ``theta``
vector. Proposals and energy evaluations are batched across chains, but the
weight update is applied sequentially per chain within each iteration to keep
the stochastic approximation correct.

Use this mode when you want aggressive weight convergence — for instance, when
the energy range is wide and a single chain would need many iterations before
``theta`` flattens out. All chains effectively pool their bin visits into one
weight vector.

.. code-block:: python

    sampler = SAMC(
        energy_fn=energy_fn,
        dim=2,
        n_chains=4,
        shared_weights=True,
        adapt_proposal=True,
    )
    result = sampler.run(n_steps=100_000)

    # Same output shape as independent mode:
    print(result.samples.shape)   # torch.Size([4, 1000, 2])
    print(result.energy_history.shape)  # torch.Size([100000, 4])

The ``energy_history`` tensor has shape ``(n_steps, n_chains)`` — one column
per chain — regardless of which mode is used.

.. note::

   In shared-weights mode the chains are *coupled*: one chain accepting a
   proposal and updating ``theta`` immediately changes the acceptance
   probability for the next chain in the same iteration. Exploration is less
   statistically independent, but the weight vector converges faster.


When to use which
-----------------

.. list-table::
   :header-rows: 1
   :widths: 35 32 33

   * - Situation
     - Recommendation
     - Why
   * - Typical multimodal sampling
     - Independent (default)
     - Uncorrelated chains, robust diagnostics
   * - Very high-dimensional space, wide energy range
     - Shared weights
     - Faster flattening of ``theta``
   * - Need credible intervals or ESS estimates
     - Independent
     - Per-chain samples are statistically independent
   * - GPU / batched energy function
     - Shared weights
     - Energy evaluation is already batched; shared mode exploits that


Inspecting per-chain results
----------------------------

All per-chain data is accessible from the returned :class:`SAMCResult`.

**Samples for a single chain:**

.. code-block:: python

    chain_0_samples = result.samples[0]   # shape: (n_saved, dim)
    chain_2_samples = result.samples[2]   # shape: (n_saved, dim)

**Energy history per chain:**

.. code-block:: python

    # energy_history shape: (n_steps, n_chains) for multi-chain runs
    chain_energies = result.energy_history[:, 0]   # chain 0, shape (n_steps,)

    import matplotlib.pyplot as plt
    for i in range(result.samples.shape[0]):
        plt.plot(result.energy_history[:, i].numpy(), alpha=0.6, label=f"chain {i}")
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.legend()

**Best solution across all chains:**

.. code-block:: python

    print(f"Best energy: {result.best_energy:.4f}")
    print(f"Best x:      {result.best_x}")     # shape: (dim,)

**Importance-weighted mean (reweights to target distribution):**

.. code-block:: python

    # Flatten chains for global normalization
    all_samples = result.samples.reshape(-1, dim)           # (n_chains * n_saved, dim)
    all_log_w = result.sample_log_weights.reshape(-1)       # (n_chains * n_saved,)

    # Normalize globally with log-sum-exp
    log_w_norm = all_log_w - torch.logsumexp(all_log_w, dim=0)
    w = log_w_norm.exp()                                    # (n_chains * n_saved,)

    # Weighted mean across all chains
    mean_x = (w.unsqueeze(-1) * all_samples).sum(dim=0)    # (dim,)


How many chains?
----------------

The default is ``n_chains=1`` (single chain). A reasonable starting point for
multi-chain runs:

* **4 chains** — good balance of coverage and compute for most 2D–20D problems.
* **8–16 chains** — hard multimodal problems with many well-separated basins,
  or when you need variance estimates and want many independent samples.
* More than 16 — rarely needed; usually better to increase ``n_steps`` instead.

Each additional chain multiplies the compute per run proportionally — there is
no free lunch. With independent weights, chains run sequentially (Python loop),
so wall-clock time scales linearly with ``n_chains``. If your ``energy_fn``
supports batched inputs (shape ``(N, dim) -> (N,)``), consider shared-weights
mode for better hardware utilization.

.. code-block:: python

    # Hard 10-mode mixture: start with 8 chains
    sampler = SAMC(
        energy_fn=hard_energy,
        dim=10,
        n_chains=8,
        adapt_proposal=True,
        proposal_std=0.5,
    )
    result = sampler.run(n_steps=200_000)
    print(result.samples.shape)   # torch.Size([8, 2000, 10])
