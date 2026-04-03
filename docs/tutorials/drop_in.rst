Drop-In Guide: Adding SAMC to an Existing MH Loop
===================================================

This tutorial is for you if you already have a working
Metropolis-Hastings (MH) sampler and want to add SAMC's flat-histogram
property without rewriting your loop from scratch.

.. contents:: On this page
   :local:
   :depth: 2


The Premise
-----------

Standard MH samples by proposing a new state :math:`\mathbf{y}` from the
current state :math:`\mathbf{x}` and accepting it with probability:

.. math::

   \alpha_{\text{MH}} = \min\!\left(1,\; \exp\!\left(-\frac{U(\mathbf{y}) - U(\mathbf{x})}{T}\right)\right)

This works well at high temperature, but at low :math:`T` any uphill move
is exponentially suppressed. A chain that falls into a low-energy basin
rarely escapes — even if other basins with identical energy exist elsewhere.

SAMC fixes this by adding a learned weight correction to the acceptance
ratio. The correction nudges the sampler away from heavily visited energy
levels toward less-visited ones, so every energy level ends up visited
equally — a *flat histogram*. The algorithm learns these weights
on-the-fly, with no knowledge of the energy landscape required up front.

The good news: you do not need to rewrite anything. The entire SAMC
machinery is packaged in :class:`SAMCWeights`, which slots into your
existing loop with **two lines of code**.


Step 1: Add SAMCWeights
-----------------------

Start by importing and constructing a :class:`SAMCWeights` instance:

.. code-block:: python

    from illuma_samc import SAMCWeights

    wm = SAMCWeights()

That is all the setup required. Note what you did *not* have to specify:

- **Energy range.** :class:`SAMCWeights` uses *deferred initialization*:
  it waits to see the first energy value before creating any bins. On
  that first call it places a wide uniform partition centered on the
  observed energy.
- **Number of bins.** The default is 201 bins (100 on each side of the
  starting energy, each 0.5 energy units wide).
- **Auto-expansion.** If the sampler wanders outside the initial range,
  bins are added automatically up to a configurable maximum.

The weight vector ``wm.theta`` and visit counter ``wm.counts`` both
start empty and grow as the sampler explores. You never need to
pre-allocate or resize them yourself.


Step 2: Add the Correction
--------------------------

Here is a typical MH loop **before** SAMC:

.. code-block:: python

    import math
    from random import random

    T = 0.1
    x = initial_state()
    fx = energy_fn(x)

    for t in range(1, n_steps + 1):
        x_new = propose(x)
        fy = energy_fn(x_new)

        log_r = (-fy + fx) / T
        if log_r > 0 or math.log(random()) < log_r:
            x, fx = x_new, fy

The only change for SAMC is to call :meth:`SAMCWeights.correction` and
add the result to ``log_r``:

.. code-block:: python

    import math
    from random import random
    from illuma_samc import SAMCWeights

    T = 0.1
    wm = SAMCWeights()                                          # <- new
    x = initial_state()
    fx = energy_fn(x)

    for t in range(1, n_steps + 1):
        x_new = propose(x)
        fy = energy_fn(x_new)

        log_r = (-fy + fx) / T + wm.correction(fx, fy)         # <- add correction
        if log_r > 0 or math.log(random()) < log_r:
            x, fx = x_new, fy

:meth:`SAMCWeights.correction` returns :math:`\theta[k_x] - \theta[k_y]`,
where :math:`k_x` and :math:`k_y` are the bin indices for the current and
proposed energies. This modifies the effective acceptance probability to:

.. math::

   \alpha_{\text{SAMC}} = \min\!\left(1,\; \exp\!\left(\theta_{k_x} - \theta_{k_y} - \frac{U(\mathbf{y}) - U(\mathbf{x})}{T}\right)\right)

When the current bin has a high weight (visited often) and the proposed
bin has a lower weight (visited rarely), the correction is positive —
making the move *more* likely to be accepted. As weights converge, all
bins become equally attractive and the histogram flattens.

Out-of-range energies are handled automatically: the
:class:`~illuma_samc.ExpandablePartition` grows to accommodate new energies
(up to ``max_bins``). Only when expansion has reached its limit does the
correction return ``-inf`` (rejecting the proposal) or ``+inf`` (pulling the
chain back into range).


Step 3: Update Weights
----------------------

After every accept/reject decision, call :meth:`SAMCWeights.step` with
the current iteration number and the energy of the **current state**
(post-accept/reject, not the proposal):

.. code-block:: python

    import math
    from random import random
    from illuma_samc import SAMCWeights

    T = 0.1
    wm = SAMCWeights()
    x = initial_state()
    fx = energy_fn(x)

    for t in range(1, n_steps + 1):
        x_new = propose(x)
        fy = energy_fn(x_new)

        log_r = (-fy + fx) / T + wm.correction(fx, fy)
        if log_r > 0 or math.log(random()) < log_r:
            x, fx = x_new, fy

        wm.step(t, fx)                                          # <- update weights

:meth:`SAMCWeights.step` performs the stochastic approximation update:

.. math::

   \theta_{t+1} = \theta_t + \gamma_{t+1}\!\left(\mathbf{e}_{t+1} - \boldsymbol{\pi}\right)

where :math:`\gamma_t` is a decreasing gain (step size), :math:`\mathbf{e}_{t+1}`
is the indicator vector for the occupied bin, and :math:`\boldsymbol{\pi}`
is the target visit frequency (uniform by default). The current bin's
weight increases; all other bins decrease slightly. Over many iterations
the weights converge so that all energy levels are visited equally.

The iteration number ``t`` (1-indexed) controls the gain schedule.
Pass the actual loop counter — do not reset it or start from zero.

That is the complete drop-in. Two lines added; the rest of your loop is
unchanged.


Adaptive Step Size
------------------

If you do not know a good proposal standard deviation, :class:`GaussianProposal`
with ``adapt=True`` will tune it automatically via dual averaging. Pass any
starting guess — it will converge regardless.

.. code-block:: python

    import math
    from random import random
    from illuma_samc import SAMCWeights, GaussianProposal

    T = 0.1
    proposal = GaussianProposal(step_size=1.0, adapt=True)     # any starting guess
    wm = SAMCWeights()
    x = initial_state()
    fx = energy_fn(x)

    for t in range(1, n_steps + 1):
        x_new = proposal.propose(x)
        fy = energy_fn(x_new)

        log_r = (-fy + fx) / T + wm.correction(fx, fy)
        accepted = log_r > 0 or math.log(random()) < log_r
        if accepted:
            x, fx = x_new, fy

        proposal.report_accept(accepted)                        # tune step size
        wm.step(t, fx)

    print(f"Tuned step size: {proposal.step_size:.4f}")

The key difference from the basic loop is that you need the ``accepted``
flag as a separate variable so you can pass it to
:meth:`GaussianProposal.report_accept`. Dual averaging runs for
``adapt_warmup`` steps (default 1000) and then freezes the step size.
After that, ``proposal.step_size`` holds the tuned value.

The adaptive proposal targets a 35% acceptance rate by default, which is
near-optimal for Gaussian random-walk proposals in moderate dimensions.
Empirically, starting step sizes from 0.01 to 5.0 all converge to a
well-tuned value with no manual intervention.


Batched Mode
------------

If you want to run multiple chains in parallel, pass tensors instead of
scalars. :class:`SAMCWeights` supports batched inputs throughout — all
``N`` chains share one weight vector and update it together each step.

.. code-block:: python

    import torch
    from illuma_samc import SAMCWeights

    T = 0.1
    N = 8          # number of parallel chains
    dim = 2
    wm = SAMCWeights()

    # z: (N, dim) -- batch of N states
    # energy: (N,) -- one scalar energy per state
    z = torch.randn(N, dim)
    energy = energy_fn(z)                   # must return shape (N,)

    for t in range(1, n_steps + 1):
        z_prop = z + 0.25 * torch.randn_like(z)        # (N, dim)
        energy_prop = energy_fn(z_prop)                # (N,)

        log_alpha = (-energy_prop + energy) / T + wm.correction(energy, energy_prop)
        accept = torch.rand_like(log_alpha).log() < log_alpha   # (N,) bool

        z = torch.where(accept.unsqueeze(-1), z_prop, z)        # (N, dim)
        energy = torch.where(accept, energy_prop, energy)       # (N,)
        wm.step(t, energy)                                      # (N,)

The shapes to keep in mind:

- ``z``: ``(N, dim)`` — the full state tensor for all chains.
- ``energy`` / ``energy_prop``: ``(N,)`` — one energy value per chain.
- ``wm.correction(energy, energy_prop)``: returns ``(N,)`` — one
  correction per chain.
- ``accept``: ``(N,)`` bool tensor — one accept/reject decision per chain.
- ``wm.step(t, energy)``: receives the ``(N,)`` post-step energies.

All ``N`` chains update the same :attr:`SAMCWeights.theta` vector.
If you want independent weights per chain, construct one
:class:`SAMCWeights` per chain and loop over them explicitly.


Inspecting Results
------------------

After running, :class:`SAMCWeights` exposes several diagnostic properties
and methods.

**Flatness.** The primary quality metric is how evenly the sampler
visited all energy bins. A value of 1.0 means perfectly uniform visits;
values above 0.9 indicate good exploration.

.. code-block:: python

    print(f"Flatness: {wm.flatness():.3f}")

**Weight vector.** The learned log-density-of-states estimates are
stored in ``wm.theta`` (a 1-D ``torch.Tensor``). Each element
corresponds to one energy bin. When the sampler has converged,
``wm.theta`` encodes the full density of states of your energy function.

.. code-block:: python

    print(f"Theta range: [{wm.theta.min():.3f}, {wm.theta.max():.3f}]")

**Visit counts.** ``wm.counts`` records how many times each bin was
visited. Bins with zero counts were never reached.

.. code-block:: python

    n_visited = (wm.counts > 0).sum().item()
    print(f"Bins visited: {n_visited} / {wm.n_bins}")

**Diagnostic plot.** For a visual overview of weight convergence and
bin visit history, call :meth:`SAMCWeights.plot_diagnostics`:

.. code-block:: python

    wm.plot_diagnostics()

This calls :func:`illuma_samc.diagnostics.plot_weight_diagnostics`
internally and requires matplotlib.


Explicit Partition
------------------

The default auto-expanding partition is convenient, but it has a cost:
the bin boundaries shift as the partition grows. If you know the energy
range of your problem in advance, use :class:`UniformPartition` for
fixed, predictable bins:

.. code-block:: python

    import math
    from random import random
    from illuma_samc import SAMCWeights, UniformPartition

    wm = SAMCWeights(
        partition=UniformPartition(e_min=0.0, e_max=20.0, n_bins=40),
    )

With an explicit partition:

- Bins are fixed at construction time — no expansion occurs.
- Proposals with energies outside ``[e_min, e_max]`` are rejected
  (the correction returns ``-inf``).
- The bin width is ``(e_max - e_min) / n_bins``. Choose ``n_bins`` so
  each bin is narrow enough to resolve the energy landscape but not so
  narrow that bins go unvisited.

You can also pass a custom gain schedule at the same time:

.. code-block:: python

    from illuma_samc import SAMCWeights, UniformPartition, GainSequence

    wm = SAMCWeights(
        partition=UniformPartition(e_min=0.0, e_max=20.0, n_bins=40),
        gain=GainSequence("1/t", t0=1000),
    )

The ``t0`` parameter delays the gain from decaying until iteration
``t0``, giving the weights time to take large exploratory steps early
in the run. The default gain schedule (``"ramp"``) works well in most
cases; ``"1/t"`` is an alternative for long runs where you want a
smoother decay.

**When to use explicit partitions:**

- High-dimensional problems (dim >= 20) where the auto-range warmup
  tends to over-explore and produce a range that is too wide.
- Problems where you have a good prior estimate of the energy range from
  a short preliminary MH run.
- Reproducibility: explicit partitions give identical bin boundaries
  across runs, making results easier to compare.

For low-dimensional problems (dim < 20), the default ``SAMCWeights()``
with no arguments is usually sufficient.
