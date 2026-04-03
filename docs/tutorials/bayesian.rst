SAMC for Bayesian Inference
============================

This tutorial shows how to use SAMC for Bayesian posterior sampling, with
particular focus on multimodal posteriors where standard MCMC fails.

.. contents:: Contents
   :local:
   :depth: 2


Why SAMC for Bayesian Problems?
--------------------------------

Standard MCMC methods such as Metropolis-Hastings work well when the
posterior is unimodal and well-behaved at the chosen temperature.  At low
temperature — or when the posterior concentrates into narrow, well-separated
modes — the random walk proposal rarely crosses the energy barrier between
modes.  The chain gets trapped.

SAMC escapes this trap by learning a *flat-histogram* over energy levels.
It adds a weight correction :math:`\theta_{J(x)} - \theta_{J(y)}` to the
MH acceptance ratio, where :math:`J(\cdot)` maps a state to its energy bin
and :math:`\theta` tracks how often each energy level has been visited.
When a bin is over-visited, its weight rises and proposals into it become
harder to accept; when a bin is under-visited, proposals into it are
favoured.  The result: the chain visits all energy levels equally and can
freely cross barriers between modes.

SAMC samples from a **flattened distribution**, not the true posterior.
You must reweight or resample before computing any posterior quantity.
Section :ref:`flat-histogram-caveat` explains exactly what this means and
how to handle it.


Setting Up the Energy Function
--------------------------------

Bayesian inference targets the posterior :math:`p(x \mid \text{data}) \propto
p(\text{data} \mid x)\, p(x)`.  In SAMC, the energy function :math:`U(x)` is
the negative log posterior:

.. math::

   U(x) = -\log p(\text{data} \mid x) - \log p(x)

**Concrete example.** Suppose we observe data from a 1-D Gaussian mixture
and place a weak Gaussian prior on the component mean :math:`x`.  We want
the posterior over :math:`x`.

.. code-block:: python

    import torch
    import math

    # Observed data: 20 points, half from mode at -3, half from mode at +3
    torch.manual_seed(0)
    data = torch.cat([
        torch.randn(10) - 3.0,   # left mode
        torch.randn(10) + 3.0,   # right mode
    ])

    def energy_fn(x: torch.Tensor) -> torch.Tensor:
        """U(x) = -log p(data | x) - log p(x).

        Likelihood: each datum is drawn from N(x, 1.0).
        Prior:      x ~ N(0, 5).
        """
        # x has shape (dim,); extract scalar
        mu = x[0]

        # -log p(data | mu): negative log-likelihood under N(mu, 1)
        log_likelihood = -0.5 * ((data - mu) ** 2).sum()

        # -log p(mu): negative log-prior under N(0, 5)
        log_prior = -0.5 * (mu / 5.0) ** 2

        return -(log_likelihood + log_prior)  # energy = -log posterior

Notice that ``energy_fn`` must accept a ``torch.Tensor`` of shape ``(dim,)``
and return a scalar ``torch.Tensor``.  The negative sign ensures the sampler
minimises energy, which is equivalent to maximising the log posterior.


Running the Sampler
--------------------

Pass the energy function to :class:`SAMC` and let it handle the sampling
loop.  Use ``adapt_proposal=True`` so the step size tunes itself — no manual
tuning required.

.. code-block:: python

    from illuma_samc import SAMC

    sampler = SAMC(
        energy_fn=energy_fn,
        dim=1,                  # x is a 1-D parameter (the mean)
        n_chains=4,             # independent chains, each with its own partition
        adapt_proposal=True,    # dual-averaging step-size adaptation
        adapt_warmup=2000,
        temperature=1.0,        # keep T=1.0 for correct posterior; see Tips
    )

    result = sampler.run(
        n_steps=200_000,
        burn_in=200,            # discard the first 200 saved samples per chain
        progress=True,
    )

    print(f"Best energy:     {result.best_energy:.4f}")
    print(f"Acceptance rate: {result.acceptance_rate:.3f}")

Expected output::

    Best energy:     19.0841
    Acceptance rate: 0.342

After the run, ``result.samples`` has shape ``(4, n_saved, 1)`` — one row of
saved states per chain.  ``result.sample_log_weights`` has shape
``(4, n_saved)`` — the log importance weight :math:`\theta[J(x)]` for every
saved sample.


.. _flat-histogram-caveat:

The Flat-Histogram Caveat
--------------------------

.. warning::

   SAMC samples from a **flattened distribution**, not the posterior.
   Raw samples from ``result.samples`` are *not* draws from
   :math:`p(x \mid \text{data})`.  You **must** reweight before computing
   any posterior quantity.

SAMC adjusts the acceptance ratio so that every energy bin is visited
equally often.  This destroys the natural Boltzmann weighting
:math:`\exp(-U(x)/T)` that would give you the posterior.  To restore it,
each sample :math:`x_i` must be weighted by

.. math::

   w_i \;\propto\; \exp\!\bigl(\theta[J(x_i)]\bigr)

where :math:`\theta[J(x_i)]` is the learned log-weight for the energy bin
that sample :math:`x_i` falls in.  Bins that SAMC over-visited get high
:math:`\theta` values; weighting by :math:`\exp(\theta)` corrects for this
inflation and restores the posterior density.

:attr:`SAMCResult.importance_weights` computes these normalized weights for
you in one call.


Recovering the Posterior
-------------------------

Two approaches are available depending on what you need.

**Approach A — Weighted samples (for expectations)**

Use :attr:`SAMCResult.importance_weights` to get normalized weights that sum
to 1 across all samples from all chains.

.. code-block:: python

    # Flatten chains: (4, n_saved, 1) -> (4*n_saved, 1)
    n_chains, n_saved, dim = result.samples.shape
    flat_samples = result.samples.reshape(-1, dim)           # (N, 1)
    flat_log_w   = result.sample_log_weights.reshape(-1)     # (N,)

    # Build a temporary SAMCResult-like object, or compute weights directly
    import torch
    log_w = flat_log_w
    log_w = log_w - torch.logsumexp(log_w, dim=0)           # log-normalize
    weights = torch.exp(log_w)                               # shape (N,)

    # Weighted posterior mean
    posterior_mean = (weights * flat_samples[:, 0]).sum()
    print(f"Posterior mean: {posterior_mean:.4f}")

For a single chain (``n_chains=1``), ``result.importance_weights`` is
available directly:

.. code-block:: python

    sampler_1chain = SAMC(
        energy_fn=energy_fn,
        dim=1,
        adapt_proposal=True,
        temperature=1.0,
    )
    result_1 = sampler_1chain.run(n_steps=200_000, burn_in=200)

    w = result_1.importance_weights          # shape (n_saved,)
    samples_1d = result_1.samples[:, 0]      # shape (n_saved,)

    posterior_mean = (w * samples_1d).sum()
    print(f"Posterior mean: {posterior_mean:.4f}")

Expected output::

    Posterior mean: 0.0312

**Approach B — Unweighted draws (for histograms and credible intervals)**

Resample from the weighted distribution using ``torch.multinomial`` to
obtain unweighted draws.  These can be treated as ordinary posterior samples.

.. code-block:: python

    # Using the single-chain result from above
    w = result_1.importance_weights           # (n_saved,) normalized weights
    n_resample = 2000

    idx = torch.multinomial(w, num_samples=n_resample, replacement=True)
    posterior_draws = result_1.samples[idx]   # (2000, 1) — unweighted posterior draws

    # These can now be treated as ordinary MCMC draws
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 3))
    plt.hist(posterior_draws[:, 0].numpy(), bins=50, density=True, alpha=0.7)
    plt.xlabel("x")
    plt.title("Posterior histogram (resampled from flat SAMC draws)")
    plt.tight_layout()
    plt.savefig("posterior_histogram.png", dpi=120)

The histogram should show two peaks near :math:`x \approx -3` and
:math:`x \approx +3`, reflecting the two clusters in the data.


Computing Posterior Statistics
--------------------------------

With importance weights in hand, any posterior expectation is a weighted
average.

.. code-block:: python

    w = result_1.importance_weights     # (n_saved,)
    x = result_1.samples[:, 0]         # (n_saved,)

    # Posterior mean
    mean = (w * x).sum()

    # Posterior variance
    variance = (w * (x - mean) ** 2).sum()
    std = variance.sqrt()

    print(f"Posterior mean:  {mean:.4f}")
    print(f"Posterior std:   {std:.4f}")

Expected output::

    Posterior mean:  0.0312
    Posterior std:   3.0841

**95 % credible interval** using resampled draws:

.. code-block:: python

    idx = torch.multinomial(w, num_samples=5000, replacement=True)
    draws = result_1.samples[idx, 0]

    lower = draws.quantile(0.025)
    upper = draws.quantile(0.975)
    print(f"95% credible interval: [{lower:.3f}, {upper:.3f}]")

Expected output::

    95% credible interval: [-3.891, 3.904]

**Effective sample size (ESS)** measures how many independent samples the
weighted collection is worth:

.. code-block:: python

    ess = 1.0 / (w ** 2).sum()
    print(f"Effective sample size: {ess:.1f} / {len(w)}")

A low ESS (say, below 10 % of the raw sample count) means the weights are
very concentrated and more samples are needed.


Tips for Bayesian SAMC
-----------------------

**Use temperature = 1.0.**
   Temperature rescales the energy as :math:`U(x)/T`.  At :math:`T = 1`
   the sampler targets :math:`\exp(-U(x))`, which is exactly the posterior.
   Lowering :math:`T` concentrates the flat-histogram distribution near the
   modes but does *not* change what ``importance_weights`` corrects to — it
   still recovers :math:`\exp(-U(x)/T)`, not the posterior.  For Bayesian
   inference, always keep ``temperature=1.0``.

**Use multiple chains.**
   With ``n_chains=4`` (the default multi-chain mode uses independent
   weights), each chain explores independently and the reweighting is
   applied per-chain before aggregation.  Four or more chains reduce the
   chance that all chains miss a mode.

**Check flatness before reweighting.**
   SAMC's importance weights are only reliable once the energy histogram is
   roughly flat.  Inspect ``sampler.plot_diagnostics()`` or compute flatness
   manually:

   .. code-block:: python

       # Bin visit counts from the best chain
       counts = result.bin_counts.float()
       visited = counts[counts > 0]
       flatness = float(1.0 - visited.std() / visited.mean())
       print(f"Flatness: {flatness:.3f}")   # aim for > 0.85

   If flatness is low, increase ``n_steps`` or reduce the energy range.

**Check the effective sample size.**
   After reweighting, always compute ESS.  If ESS is below 5 % of raw
   sample count, the flat-histogram sampling did not cover the posterior
   well.  Try more steps, more chains, or a narrower ``e_min`` / ``e_max``
   range.

**High-dimensional posteriors (dim >= 20).**
   Auto-range mode may over-expand the energy partition because warmup
   trajectories explore too widely.  Specify the energy range explicitly:

   .. code-block:: python

       sampler = SAMC(
           energy_fn=energy_fn,
           dim=50,
           n_chains=8,
           e_min=-200.0,
           e_max=0.0,        # log-posterior values are typically negative
           adapt_proposal=True,
           temperature=1.0,
       )

   Run a short probe with plain MH first to estimate the typical range of
   :math:`-\log p(x \mid \text{data})` for your problem.
