How It Works
============

This page explains the SAMC algorithm from first principles: why standard
Metropolis-Hastings fails on multimodal problems, what SAMC does differently,
how the math works, and how to recover the target distribution from SAMC samples.

.. contents:: On this page
   :local:
   :depth: 2


The Problem: Local Traps
------------------------

Metropolis-Hastings (MH) explores a distribution by proposing local moves
and accepting or rejecting them based on the energy difference. At high
temperature this works well — the sampler accepts most moves and roams freely.
But at low temperature (small :math:`T`), the acceptance ratio for a move from
energy :math:`U(\mathbf{x})` to :math:`U(\mathbf{y})` is:

.. math::

   \alpha_{\text{MH}} = \min\!\left(1,\; \exp\!\left(-\frac{U(\mathbf{y}) - U(\mathbf{x})}{T}\right)\right)

When :math:`T` is small, any uphill move is exponentially unlikely to be
accepted. If two low-energy modes are separated by a high-energy barrier,
a chain that falls into one mode will almost never cross the barrier to
find the other. The sampler gets *trapped*.

This is not a tuning problem — it is a fundamental property of MH. No choice
of proposal distribution or step size will help once the barrier height exceeds
a few multiples of :math:`T`. The only escape is to somehow make barrier
regions less costly to visit.


SAMC Solution: Flat-Histogram Sampling
---------------------------------------

SAMC (Stochastic Approximation Monte Carlo) escapes local traps by learning
and compensating for the *density of states* — how many configurations exist
at each energy level.

The key idea: partition the energy axis into :math:`m` subregions
:math:`E_1, E_2, \ldots, E_m`. Assign each subregion a weight
:math:`\theta_k` (a log-density-of-states estimate). Modify the acceptance
ratio so that the sampler is *penalized* for returning to heavily visited
regions and *rewarded* for visiting sparse ones. As the weights adapt, every
energy level gets visited with equal frequency — the histogram of visited
energies becomes flat.

Once flat, the sampler moves freely across the entire energy landscape. A high
barrier between two modes is no longer a trap: the weights learn to compensate
for it, making uphill moves as likely as downhill ones.


The Algorithm
-------------

Step 1 — Partition the energy space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Divide the energy axis into :math:`m` non-overlapping subregions
:math:`E_1 < E_2 < \cdots < E_m`. Each visited state :math:`\mathbf{x}` is
assigned to subregion :math:`J(\mathbf{x}) \in \{1, \ldots, m\}` by its
energy :math:`U(\mathbf{x})`.

In practice, the subregions are uniform-width bins. You can specify the range
explicitly with :class:`~illuma_samc.partitions.UniformPartition`, or let
:class:`~illuma_samc.SAMCWeights` auto-create bins centered on the first
energy it sees and expand as needed.


Step 2 — Modified acceptance ratio
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The weight vector :math:`\boldsymbol{\theta}_t = (\theta_{t,1}, \ldots, \theta_{t,m})`
tracks the estimated log-density of each subregion. The modified acceptance
ratio for a proposed move from :math:`\mathbf{x}` (current) to
:math:`\mathbf{y}` (proposed) is:

.. math::

   \alpha = \min\!\left(1,\; \exp\!\left(
       \theta_{t,J(\mathbf{x})} - \theta_{t,J(\mathbf{y})}
       - \frac{U(\mathbf{y}) - U(\mathbf{x})}{T}
   \right)\right)

The term :math:`\theta_{t,J(\mathbf{x})} - \theta_{t,J(\mathbf{y})}` is the
**weight correction**. It down-weights moves into already-dense regions and
up-weights moves into sparse ones. When :math:`\boldsymbol{\theta}` has
converged to the true log-density of states, this correction exactly cancels
the energy barrier, and all subregions become equally accessible.

In code this is the :meth:`~illuma_samc.SAMCWeights.correction` method::

    log_r = (-fy + fx) / T + wm.correction(fx, fy)


Step 3 — Weight update via stochastic approximation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After each accept/reject step, update the weights using the stochastic
approximation rule:

.. math::

   \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t
       + \gamma_{t+1}\!\left(\mathbf{e}_{t+1} - \boldsymbol{\pi}\right)

where:

- :math:`\gamma_{t+1}` is the gain (step size) at iteration :math:`t+1`
- :math:`\mathbf{e}_{t+1}` is the indicator vector for the subregion occupied
  after step :math:`t+1` — it is 1 in the occupied bin and 0 elsewhere
- :math:`\boldsymbol{\pi}` is the target visit frequency, uniform by default
  (:math:`\pi_k = 1/m` for all :math:`k`)

Concretely: the occupied bin gains :math:`\gamma_{t+1}(1 - 1/m)` and every
other bin loses :math:`\gamma_{t+1}/m`. Bins that are visited too often
accumulate high :math:`\theta_k`, which makes them harder to enter. Bins that
are rarely visited accumulate low :math:`\theta_k`, which makes them easier
to enter. Over time the visit histogram flattens.

In code this is the :meth:`~illuma_samc.SAMCWeights.step` method::

    wm.step(t, fx)   # call once per iteration with the current energy


Step 4 — Gain sequence decay
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The gain :math:`\gamma_t` must decrease over time so that the weights converge
rather than oscillate. The standard choice (Liang et al. 2007) is:

.. math::

   \gamma(t) = \frac{\gamma_0}{(\gamma_1 + t)^\alpha}

with :math:`\gamma_0 = t_0`, :math:`\gamma_1 = 0`, :math:`\alpha = 1`,
giving the familiar :math:`t_0 / \max(t, t_0)` schedule. The parameter
:math:`t_0` is the *warmup period* — the gain is capped at 1 for the first
:math:`t_0` iterations, allowing the weights to move quickly in early
exploration before slowing down for stable convergence.

The default schedule in ``illuma-samc`` is ``"ramp"``, which holds a constant
gain during warmup and then decays as a power law, matching the behavior of
Liang's reference implementation.


Key Components
--------------

Partition
~~~~~~~~~

A :class:`~illuma_samc.partitions.Partition` maps an energy value to a bin
index. Two implementations are provided:

:class:`~illuma_samc.partitions.UniformPartition`
    Fixed uniform bins over a specified energy range ``[e_min, e_max]``.
    Best when you know the energy range in advance. Any sample outside the
    range returns -1 (rejected by the weight correction).

:class:`~illuma_samc.partitions.ExpandablePartition`
    Starts as a uniform partition centered on the first energy seen and
    expands in 5-bin increments when samples land outside the current range
    (up to ``max_bins`` total). Used internally by :class:`~illuma_samc.SAMCWeights`
    for zero-configuration startup.

The partition width and number of bins trade off resolution against statistical
noise. As a rule of thumb, 20–80 bins works well for most problems; scale
toward the higher end for high-dimensional problems where the energy range is
wider.

Gain Sequence
~~~~~~~~~~~~~

:class:`~illuma_samc.gain.GainSequence` produces the step-size schedule
:math:`\gamma_t`. The available presets are:

``"1/t"``
    Classical SAMC gain from Liang et al. (2007). Guarantees convergence
    under mild conditions. Equivalent to :math:`t_0 / \max(t, t_0)`.

``"ramp"``
    Constant gain during a warmup phase, then power-law decay. Default.
    More aggressive early exploration than ``"1/t"``.

Custom schedules can be passed as any callable ``(int) -> float``.

The gain controls the speed/accuracy tradeoff. A large gain adapts quickly
but may never fully settle. A gain that decays too fast freezes the weights
before they have converged. The default ``t0=1000`` is a safe starting point;
increase it (e.g., ``t0=5000``) for higher-dimensional problems.

Weight Correction
~~~~~~~~~~~~~~~~~

The weight correction :math:`\theta_{t,J(\mathbf{x})} - \theta_{t,J(\mathbf{y})}`
is the single change that turns MH into SAMC. It is computed by
:meth:`~illuma_samc.SAMCWeights.correction` and added to the MH log
acceptance ratio before the accept/reject decision.

When the weights have not yet converged, the correction is approximate — the
sampler still explores but with some bias. As :math:`\boldsymbol{\theta}`
converges to the true log-density of states, the correction becomes exact and
the visit histogram is flat. The :meth:`~illuma_samc.SAMCWeights.flatness`
diagnostic measures how close to flat the histogram is at any point.


Recovering the Target Distribution
------------------------------------

SAMC samples from a *flattened* distribution, not the original target
:math:`p(\mathbf{x}) \propto \exp(-U(\mathbf{x})/T)`. The flattened distribution
assigns extra probability mass to high-energy (low-density) regions so that
the sampler visits them.

To recover the target, apply importance reweighting. Each sample
:math:`\mathbf{x}_i` in bin :math:`k_i` receives log-weight
:math:`\theta_{k_i}`, which corrects for the flat-histogram bias:

.. math::

   w_i \propto \exp\!\left(\theta_{k_i}\right)

After normalizing so :math:`\sum_i w_i = 1`, you can compute weighted
expectations over the target:

.. math::

   \mathbb{E}_p[f] \approx \sum_i w_i f(\mathbf{x}_i)

For unweighted samples drawn from the target, use
:meth:`~illuma_samc.SAMCWeights.resample`. This performs acceptance–rejection
with acceptance probability proportional to :math:`\exp(\theta_{k_i})`,
returning a subset of the collected samples that are distributed according
to the target.

In code::

    # Weighted expectation
    weights = wm.importance_weights(energies)          # normalized, sums to 1
    weighted_mean = (weights * values).sum()

    # Unweighted draws from the target
    target_samples = wm.resample(flat_samples, energies)


References
----------

.. [Liang2007] Faming Liang, Chuanhai Liu, and Raymond J. Carroll.
   *Stochastic Approximation in Monte Carlo Computation.*
   Journal of the American Statistical Association, 102(477):305–320, 2007.

.. [Liang2009] Faming Liang.
   *On the Use of Stochastic Approximation Monte Carlo for Monte Carlo Integration.*
   Statistics & Probability Letters, 79(5):581–587, 2009.

See ``CITATION.bib`` in the repository root for BibTeX entries.
