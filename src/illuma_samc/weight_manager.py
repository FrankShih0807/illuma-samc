"""Lightweight SAMC weight manager — drop into any MH loop.

SAMC is just MH with a weight correction. If you already have an MH loop,
add two lines to get SAMC:

**Before (standard MH):**

.. code-block:: python

    for t in range(1, n_steps + 1):
        x_new = propose(x)
        fy = energy_fn(x_new)

        log_r = (-fy + fx) / T
        if log_r > 0 or torch.rand(1).item() < math.exp(log_r):
            x, fx = x_new, fy

**After (SAMC):**

.. code-block:: python

    wm = SAMCWeights()                                        # <-- new

    for t in range(1, n_steps + 1):
        x_new = propose(x)
        fy = energy_fn(x_new)

        log_r = (-fy + fx) / T + wm.correction(fx, fy)       # <-- add correction
        if log_r > 0 or torch.rand(1).item() < math.exp(log_r):
            x, fx = x_new, fy
        wm.step(t, fx)                                        # <-- update weights

That's it. Your MH sampler now explores all energy levels uniformly.

**With adaptive step size** — don't know the right proposal_std? Let it tune itself:

.. code-block:: python

    from illuma_samc.proposals import GaussianProposal

    proposal = GaussianProposal(step_size=1.0, adapt=True)  # <-- any starting guess
    wm = SAMCWeights()

    for t in range(1, n_steps + 1):
        x_new = proposal.propose(x)
        fy = energy_fn(x_new)

        log_r = (-fy + fx) / T + wm.correction(fx, fy)
        accepted = log_r > 0 or torch.rand(1).item() < math.exp(log_r)
        if accepted:
            x, fx = x_new, fy
        proposal.report_accept(accepted)                     # <-- tune step size
        wm.step(t, fx)

    print(f"Tuned step size: {proposal.step_size:.4f}")      # see what it found
"""

from __future__ import annotations

import torch

from illuma_samc.gain import GainSequence
from illuma_samc.partitions import Partition


class SAMCWeights:
    r"""Drop-in SAMC weight correction for any Metropolis-Hastings loop.

    Manages the bin weights (theta) that make SAMC overcome energy barriers.
    Two methods are all you need:

    - :meth:`correction` — returns :math:`\theta[k_x] - \theta[k_y]`,
      the term you add to your MH log acceptance ratio.
    - :meth:`step` — updates the weights after accept/reject.

    By default, bins are initialized on the first energy seen: a wide
    uniform partition is centered on that energy and expands if the
    sampler wanders outside. For explicit control, pass ``partition``.

    Parameters
    ----------
    bin_width : float
        Width of each energy bin. Default 0.5.
    n_bins_per_side : int
        Number of bins above and below the starting energy.
        Default 100 (201 total bins: 100 below + 1 center + 100 above).
    max_bins : int
        Maximum bins after expansion. Default 1000.
    partition : Partition, optional
        Explicit energy-space partition. Overrides auto-initialization.
    gain : GainSequence or str, optional
        Step-size schedule. Default ``"ramp"``.
    gain_kwargs : dict, optional
        Extra kwargs for :class:`GainSequence` (when ``gain`` is a string).
    device : str or torch.device
        Device for theta and counts tensors. Default ``"cpu"``.
    dtype : str or torch.dtype
        Dtype for sample-facing tensors (e.g. ``"float32"``, ``torch.float64``).
        Internal accumulation (theta, counts) always uses float64 for precision.
    """

    def __init__(
        self,
        *,
        bin_width: float = 0.5,
        n_bins_per_side: int = 100,
        max_bins: int = 1000,
        partition: Partition | None = None,
        gain: GainSequence | str | None = None,
        gain_kwargs: dict | None = None,
        device: torch.device | str = "cpu",
        dtype: str | torch.dtype = torch.float32,
        record_every: int = 100,
    ) -> None:
        self._device = torch.device(device)
        self._dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
        if self._device.type == "mps" and self._dtype == torch.float64:
            raise ValueError(
                "MPS does not support float64. Use dtype='float32' (or omit dtype)"
                " with device='mps'."
            )

        if partition is not None:
            self.partition = partition
            self._deferred_init = False
        else:
            # Defer partition creation until first energy is seen
            self.partition = None  # type: ignore[assignment]
            self._deferred_init = True
            self._bin_width = bin_width
            self._n_bins_per_side = n_bins_per_side
            self._max_bins = max_bins

        if isinstance(gain, GainSequence):
            self.gain = gain
        else:
            default_gain_kwargs = {
                "rho": 1.0,
                "tau": 1.0,
                "warmup": 1,
                "step_scale": 1000,
            }
            self.gain = GainSequence(gain or "ramp", **(gain_kwargs or default_gain_kwargs))

        # Theta/counts always on CPU (small vectors; MPS lacks float64)
        if not self._deferred_init:
            n = self.partition.n_partitions
            self.theta = torch.zeros(n, dtype=torch.float64)
            self.counts = torch.zeros(n, dtype=torch.float64)
            self._refden = 1.0 / n
        else:
            self.theta = torch.empty(0, dtype=torch.float64)
            self.counts = torch.empty(0, dtype=torch.float64)
            self._refden = 0.0
        self._t = 0

        # History tracking
        self._record_every = record_every
        self.bin_counts_history: list[torch.Tensor] = []

        # Acceptance rate tracking
        self._n_steps = 0
        self._n_accepted = 0
        self._last_energy: float | None = None
        self._warmup_done = False
        self._acceptance_warned = False

    def _init_partition_from_energy(self, energy: float) -> None:
        """Create the partition centered on the first energy seen.

        Places ``n_bins_per_side`` bins on each side of the starting
        energy, giving ``2 * n_bins_per_side + 1`` bins total (default 201).
        """
        from illuma_samc.partitions import make_expandable_partition

        self.partition = make_expandable_partition(
            energy,
            bin_width=self._bin_width,
            n_bins_per_side=self._n_bins_per_side,
            max_bins=self._max_bins,
        )
        self._deferred_init = False

        n = self.partition.n_partitions
        self.theta = torch.zeros(n, dtype=torch.float64)
        self.counts = torch.zeros(n, dtype=torch.float64)
        self._refden = 1.0 / n

    # ------------------------------------------------------------------
    # Partition resize support
    # ------------------------------------------------------------------

    def _resize_for_partition(self) -> None:
        """Resize theta and counts to match current partition size.

        Called after the partition expands (e.g., :class:`ExpandablePartition`).
        New bins get theta = 0 and counts = 0.
        """
        n = self.partition.n_partitions
        old_n = self.theta.shape[0]
        if n == old_n:
            return

        if n > old_n:
            extra = n - old_n
            self.theta = torch.cat([self.theta, torch.zeros(extra, dtype=torch.float64)])
            self.counts = torch.cat([self.counts, torch.zeros(extra, dtype=torch.float64)])
        else:
            self.theta = self.theta[:n]
            self.counts = self.counts[:n]

        self._refden = 1.0 / n

    def _maybe_resize(self) -> None:
        """Check if partition expanded and resize theta/counts if needed."""
        if hasattr(self.partition, "expanded") and self.partition.expanded:
            self._resize_for_partition()
            self.partition.expanded = False
        elif self.partition.n_partitions != self.theta.shape[0]:
            self._resize_for_partition()

    @property
    def n_bins(self) -> int:
        """Number of energy bins (0 if partition not yet initialized)."""
        if self._deferred_init:
            return 0
        return self.partition.n_partitions

    @property
    def log_weights(self) -> torch.Tensor:
        """Current theta vector (log partition weights)."""
        return self.theta

    def correction(
        self,
        energy_current: float | torch.Tensor,
        energy_proposed: float | torch.Tensor,
    ) -> float | torch.Tensor:
        r"""SAMC weight correction to add to your MH log acceptance ratio.

        Returns :math:`\theta[k_x] - \theta[k_y]` where :math:`k_x, k_y`
        are the bin indices for the current and proposed energies.

        Add this to your existing MH log ratio::

            log_r = (-fy + fx) / T + wm.correction(fx, fy)

        Returns ``-inf`` if proposed energy is out of partition range
        (reject the proposal). Returns ``+inf`` if current energy is
        out of range but proposed is in range (accept to get back in range).

        Supports batched inputs: if energies are 1-D tensors with shape
        ``(N,)``, returns a tensor of corrections with the same shape.

        Parameters
        ----------
        energy_current : float or Tensor
            Energy of current state(s). Scalar or shape ``(N,)``.
        energy_proposed : float or Tensor
            Energy of proposed state(s). Scalar or shape ``(N,)``.

        Returns
        -------
        float or Tensor
            The correction term :math:`\theta[k_x] - \theta[k_y]`.
        """
        # Lazy init: create partition centered on first energy
        if self._deferred_init:
            if isinstance(energy_current, torch.Tensor) and energy_current.dim() >= 1:
                self._init_partition_from_energy(energy_current[0].item())
            else:
                self._init_partition_from_energy(float(energy_current))

        # Batched path
        if isinstance(energy_current, torch.Tensor) and energy_current.dim() >= 1:
            kx = self.partition.assign_batch(energy_current)
            self._maybe_resize()
            ky = self.partition.assign_batch(energy_proposed)
            self._maybe_resize()

            result = torch.zeros(
                energy_current.shape[0],
                dtype=self.theta.dtype,
                device=energy_current.device,
            )
            both_in = (kx >= 0) & (ky >= 0)
            result[both_in] = self.theta[kx[both_in]] - self.theta[ky[both_in]]
            result[(kx >= 0) & (ky < 0)] = float("-inf")
            result[(kx < 0) & (ky >= 0)] = float("inf")
            result[(kx < 0) & (ky < 0)] = 0.0
            return result

        # Scalar path
        ex = float(energy_current)
        ey = float(energy_proposed)

        kx = self.partition.assign(torch.tensor(ex, device=self._device))
        self._maybe_resize()
        ky = self.partition.assign(torch.tensor(ey, device=self._device))
        self._maybe_resize()

        if ky < 0:
            return float("-inf")
        if kx < 0:
            return float("inf")

        return self.theta[kx].item() - self.theta[ky].item()

    def step(self, t: int, energy: float | torch.Tensor) -> None:
        """Update bin weights after accept/reject.

        Call once per iteration with the energy of the current state
        (after accept/reject).

        Supports batched inputs: if energy is a 1-D tensor with shape
        ``(N,)``, all elements contribute to the weight update in a
        single step.

        Parameters
        ----------
        t : int
            Iteration number (1-indexed). Used to compute the gain.
        energy : float or Tensor
            Energy of the current state(s). Scalar or shape ``(N,)``.
        """
        self._t = t

        # Lazy init: create partition centered on first energy
        if self._deferred_init:
            if isinstance(energy, torch.Tensor) and energy.dim() >= 1:
                self._init_partition_from_energy(energy[0].item())
            else:
                self._init_partition_from_energy(float(energy))

        # Batched path
        if isinstance(energy, torch.Tensor) and energy.dim() >= 1:
            ks = self.partition.assign_batch(energy)
            self._maybe_resize()

            valid = ks >= 0
            if valid.any():
                delta = self.gain(t)
                valid_ks = ks[valid]
                n_valid = valid_ks.shape[0]
                self.theta -= delta * self._refden * n_valid
                ones = torch.ones(n_valid, dtype=self.theta.dtype)
                self.theta.scatter_add_(0, valid_ks, ones * delta)
                self.counts.scatter_add_(0, valid_ks, ones)

            # Track acceptance rate (skip for batched — not meaningful per-element)
            self._n_steps += 1

            if t % self._record_every == 0:
                self.bin_counts_history.append(self.counts.clone())
            return

        # Scalar path
        e = float(energy)

        # Track acceptance rate (energy change implies acceptance)
        self._n_steps += 1
        if self._last_energy is not None and e != self._last_energy:
            self._n_accepted += 1
        self._last_energy = e

        # Warn once if acceptance rate outside 15-50% after warmup
        warmup_threshold = 1000
        if self._n_steps == warmup_threshold:
            self._warmup_done = True
        if (
            self._warmup_done
            and not self._acceptance_warned
            and self._n_steps % warmup_threshold == 0
        ):
            rate = self._n_accepted / self._n_steps
            if rate < 0.15 or rate > 0.50:
                import warnings

                self._acceptance_warned = True
                if rate < 0.15:
                    msg = (
                        f"SAMCWeights: acceptance rate is {rate:.3f} (below 15%). "
                        "Consider increasing proposal_std or temperature."
                    )
                else:
                    msg = (
                        f"SAMCWeights: acceptance rate is {rate:.3f} (above 50%). "
                        "Consider decreasing proposal_std or temperature."
                    )
                warnings.warn(msg, stacklevel=2)

        k = self.partition.assign(torch.tensor(e, device=self._device))
        self._maybe_resize()

        if k < 0:
            return

        delta = self.gain(t)
        self.theta -= delta * self._refden
        self.theta[k] += delta
        self.counts[k] += 1

        # Record history snapshot
        if t % self._record_every == 0:
            self.bin_counts_history.append(self.counts.clone())

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def tracked_acceptance_rate(self) -> float:
        """Acceptance rate tracked internally from energy changes.

        Returns 0.0 if no steps have been taken.
        """
        if self._n_steps == 0:
            return 0.0
        return self._n_accepted / self._n_steps

    def flatness(self) -> float:
        """Bin visit flatness over visited bins: ``1 - std / mean``.

        Only bins with at least one visit are included.
        Returns 1.0 for perfectly flat visits (or single visited bin),
        lower for uneven visits.  Returns 0.0 if no bins have been visited.
        """
        visited = self.counts[self.counts > 0].float()
        if len(visited) <= 1:
            return 1.0 if len(visited) == 1 else 0.0
        mean = visited.mean()
        return float(1.0 - visited.std() / mean)

    def flatness_history(self) -> list[float]:
        """Flatness at each recorded snapshot.

        Returns a list of flatness values, one per snapshot recorded
        every ``record_every`` steps. Computed over visited bins only.
        """
        result = []
        for snapshot in self.bin_counts_history:
            visited = snapshot[snapshot > 0].float()
            if len(visited) == 0:
                result.append(0.0)
            else:
                result.append(float(1.0 - visited.std() / visited.mean()))
        return result

    def importance_log_weights(
        self,
        energies: torch.Tensor,
    ) -> torch.Tensor:
        r"""Per-sample importance log-weights for reweighting to the target.

        SAMC samples from a flattened distribution proportional to
        :math:`\exp(-E(x)/T - \theta[k(x)])`. To recover the original
        target :math:`\exp(-E(x)/T)`, weight each sample by
        :math:`\exp(\theta[k_i])`.

        Parameters
        ----------
        energies : Tensor
            1-D tensor of energies for each collected sample.

        Returns
        -------
        Tensor
            Log importance weights (1-D, same length as *energies*).
            Samples from unvisited bins get ``-inf``.
        """
        bins = self.partition.assign_batch(energies)
        in_range = bins >= 0
        # Use clamped bins for indexing (out-of-range get index 0, but masked later)
        safe_bins = bins.clamp(min=0)

        log_w = self.theta[safe_bins].clone()
        visited = self.counts[safe_bins] > 0
        log_w[~(in_range & visited)] = float("-inf")
        return log_w

    def importance_weights(
        self,
        energies: torch.Tensor,
    ) -> torch.Tensor:
        """Normalized importance weights for reweighting to the target.

        Returns ``exp(theta[k]) / max(exp(theta))`` per sample, then
        normalized to sum to 1. Use for resampling or weighted expectations::

            # Weighted expectation
            weighted_mean = (weights * values).sum()

            # Resample to get unweighted target samples
            idx = torch.multinomial(weights, n, replacement=True)
            target_samples = flat_samples[idx]

        Parameters
        ----------
        energies : Tensor
            1-D tensor of energies for each collected sample.

        Returns
        -------
        Tensor
            Normalized importance weights (1-D, sums to 1).
            Returns zeros if all log-weights are ``-inf``.
        """
        log_w = self.importance_log_weights(energies)
        if (log_w == float("-inf")).all():
            return torch.zeros_like(log_w)
        # Stable: subtract max before exp
        log_w_shifted = log_w - log_w[log_w > float("-inf")].max()
        w = torch.exp(log_w_shifted)
        return w / w.sum()

    def resample(
        self,
        samples: torch.Tensor,
        energies: torch.Tensor,
    ) -> torch.Tensor:
        """Resample from flat SAMC samples to recover the target distribution.

        For each sample in bin *k*, keep it with probability
        ``exp(theta[k]) / max(exp(theta))``, discard otherwise.
        The kept samples are unweighted draws from the target.

        Parameters
        ----------
        samples : Tensor
            Collected samples, shape ``(n_samples, dim)``.
        energies : Tensor
            1-D tensor of energies for each sample.

        Returns
        -------
        Tensor
            Subset of samples drawn from the target distribution.
        """
        log_w = self.importance_log_weights(energies)
        valid = log_w > float("-inf")
        if not valid.any():
            return samples[:0].clone()  # empty

        # p_i = exp(theta[k_i]) / max(exp(theta)) = exp(theta[k_i] - max(theta))
        log_w_shifted = log_w.clone()
        log_w_shifted[~valid] = float("-inf")
        log_w_shifted = log_w_shifted - log_w[valid].max()
        p = torch.exp(log_w_shifted)

        keep = torch.rand(len(samples), device=samples.device) < p
        return samples[keep]

    def plot_diagnostics(self, **kwargs: object) -> object:
        """Plot diagnostic panels for this weight manager.

        Requires matplotlib. See :func:`diagnostics.plot_weight_diagnostics`
        for details on the panels and keyword arguments.

        Returns
        -------
        matplotlib.figure.Figure
            The diagnostic figure.
        """
        from illuma_samc.diagnostics import plot_weight_diagnostics

        return plot_weight_diagnostics(self, **kwargs)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return serializable state for checkpointing."""
        return {
            "theta": self.theta.clone(),
            "counts": self.counts.clone(),
            "t": self._t,
            "bin_counts_history": [h.clone() for h in self.bin_counts_history],
            "n_steps": self._n_steps,
            "n_accepted": self._n_accepted,
            "last_energy": self._last_energy,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from a checkpoint."""
        self.theta.copy_(state["theta"])
        self.counts.copy_(state["counts"])
        self._t = state["t"]
        if "bin_counts_history" in state:
            self.bin_counts_history = [h.clone() for h in state["bin_counts_history"]]
        if "n_steps" in state:
            self._n_steps = state["n_steps"]
            self._n_accepted = state["n_accepted"]
            self._last_energy = state["last_energy"]
            self._warmup_done = self._n_steps >= 1000
