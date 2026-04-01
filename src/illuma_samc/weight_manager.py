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

    wm = SAMCWeights(partition, gain)                         # <-- new

    for t in range(1, n_steps + 1):
        x_new = propose(x)
        fy = energy_fn(x_new)

        log_r = (-fy + fx) / T + wm.correction(fx, fy)       # <-- add correction
        if log_r > 0 or torch.rand(1).item() < math.exp(log_r):
            x, fx = x_new, fy
        wm.step(t, fx)                                        # <-- update weights

That's it. Your MH sampler now explores all energy levels uniformly.
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

    Parameters
    ----------
    partition : Partition
        Energy-space partition (defines bins).
    gain : GainSequence
        Step-size schedule for weight updates.
    device : str or torch.device
        Device for theta and counts tensors. Default ``"cpu"``.
    """

    def __init__(
        self,
        partition: Partition,
        gain: GainSequence,
        *,
        device: torch.device | str = "cpu",
        record_every: int = 100,
    ) -> None:
        self.partition = partition
        self.gain = gain

        n = partition.n_partitions
        self.theta = torch.zeros(n, device=device, dtype=torch.float64)
        self.counts = torch.zeros(n, device=device, dtype=torch.float64)
        self._refden = 1.0 / n
        self._t = 0

        # History tracking
        self._record_every = record_every
        self.bin_counts_history: list[torch.Tensor] = []

        # Acceptance rate tracking
        self._n_steps = 0
        self._n_accepted = 0
        self._last_energy: float | None = None
        self._warmup_done = False

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_warmup(
        cls,
        energy_fn,
        dim: int,
        n_bins: int = 42,
        warmup_steps: int = 5000,
        proposal_std: float = 0.25,
        margin: float = 0.1,
        gain: str | None = None,
        gain_kwargs: dict | None = None,
        *,
        overflow_bins: bool = False,
        device: torch.device | str = "cpu",
        record_every: int = 100,
    ) -> "SAMCWeights":
        """Create SAMCWeights with auto-detected energy range from warmup MH.

        Runs ``warmup_steps`` of vanilla Metropolis-Hastings (no weight
        correction) to estimate the energy range, then constructs a
        :class:`UniformPartition` with ``margin`` padding on each side.

        Parameters
        ----------
        energy_fn : callable
            ``Tensor -> Tensor`` or ``Tensor -> (Tensor, bool)`` energy function.
        dim : int
            Dimensionality of the sample space.
        n_bins : int
            Number of core energy bins. Default 42.
        warmup_steps : int
            Number of warmup MH steps. Default 5000.
        proposal_std : float
            Proposal standard deviation for warmup. Default 0.25.
        margin : float
            Fractional margin to pad the observed range. Default 0.1
            (10% on each side).
        gain : str or None
            Gain schedule name. Default ``None`` uses ``"1/t"`` with ``t0=1000``.
        gain_kwargs : dict or None
            Extra kwargs for :class:`GainSequence`.
        overflow_bins : bool
            Whether to add overflow bins. Default ``False``.
        device : str or torch.device
            Device for tensors.
        record_every : int
            Snapshot interval for bin counts history.

        Returns
        -------
        SAMCWeights
            Initialized weight manager with auto-detected energy range.
        """
        from illuma_samc.partitions import UniformPartition

        # Run warmup MH with no weight correction
        x = torch.zeros(dim, device=device)
        raw = energy_fn(x)
        if isinstance(raw, tuple):
            fx = float(raw[0])
        else:
            fx = float(raw)

        energies_seen = [fx]
        for _ in range(warmup_steps):
            x_new = x + proposal_std * torch.randn(dim, device=device)
            raw = energy_fn(x_new)
            if isinstance(raw, tuple):
                fy, in_r = float(raw[0]), bool(raw[1])
            else:
                fy, in_r = float(raw), True

            # Simple MH acceptance (no weight correction, T=1)
            import math

            log_r = -fy + fx
            if in_r and (log_r > 0 or math.log(torch.rand(1).item() + 1e-300) < log_r):
                x, fx = x_new.clone(), fy

            energies_seen.append(fx)

        e_min_obs = min(energies_seen)
        e_max_obs = max(energies_seen)
        span = e_max_obs - e_min_obs
        if span < 1e-8:
            span = 1.0  # avoid degenerate range

        e_min = e_min_obs - margin * span
        e_max = e_max_obs + margin * span

        partition = UniformPartition(
            e_min=e_min, e_max=e_max, n_bins=n_bins, overflow_bins=overflow_bins
        )

        # Build gain
        if gain is None:
            gain_schedule = "1/t"
            gkw = {"t0": 1000}
        else:
            gain_schedule = gain
            gkw = gain_kwargs or {}

        if gain_kwargs is not None and gain is not None:
            gkw = gain_kwargs

        gain_obj = GainSequence(gain_schedule, **gkw)

        return cls(
            partition=partition,
            gain=gain_obj,
            device=device,
            record_every=record_every,
        )

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
            device = self.theta.device
            self.theta = torch.cat(
                [self.theta, torch.zeros(extra, device=device, dtype=torch.float64)]
            )
            self.counts = torch.cat(
                [self.counts, torch.zeros(extra, device=device, dtype=torch.float64)]
            )
        else:
            self.theta = self.theta[:n]
            self.counts = self.counts[:n]
        self._refden = 1.0 / n

    def _maybe_resize(self) -> None:
        """Check if partition expanded and resize theta/counts if needed."""
        if hasattr(self.partition, "expanded") and self.partition.expanded:
            self._resize_for_partition()
            self.partition.expanded = False

    @property
    def n_bins(self) -> int:
        """Number of energy bins."""
        return self.partition.n_partitions

    @property
    def log_weights(self) -> torch.Tensor:
        """Current theta vector (log partition weights)."""
        return self.theta

    def correction(
        self,
        energy_current: float | torch.Tensor,
        energy_proposed: float | torch.Tensor,
    ) -> float:
        r"""SAMC weight correction to add to your MH log acceptance ratio.

        Returns :math:`\theta[k_x] - \theta[k_y]` where :math:`k_x, k_y`
        are the bin indices for the current and proposed energies.

        Add this to your existing MH log ratio::

            log_r = (-fy + fx) / T + wm.correction(fx, fy)

        Returns ``-inf`` if proposed energy is out of partition range
        (reject the proposal). Returns ``+inf`` if current energy is
        out of range but proposed is in range (accept to get back in range).

        Parameters
        ----------
        energy_current : float or Tensor
            Energy of current state.
        energy_proposed : float or Tensor
            Energy of proposed state.

        Returns
        -------
        float
            The correction term :math:`\theta[k_x] - \theta[k_y]`.
        """
        ex = float(energy_current)
        ey = float(energy_proposed)

        kx = self.partition.assign(torch.tensor(ex))
        self._maybe_resize()
        ky = self.partition.assign(torch.tensor(ey))
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

        Parameters
        ----------
        t : int
            Iteration number (1-indexed). Used to compute the gain.
        energy : float or Tensor
            Energy of the current state.
        """
        self._t = t
        e = float(energy)

        # Track acceptance rate (energy change implies acceptance)
        self._n_steps += 1
        if self._last_energy is not None and e != self._last_energy:
            self._n_accepted += 1
        self._last_energy = e

        # Warn if acceptance rate outside 15-50% after warmup
        warmup_threshold = 1000
        if self._n_steps == warmup_threshold:
            self._warmup_done = True
        if self._warmup_done and self._n_steps % warmup_threshold == 0:
            rate = self._n_accepted / self._n_steps
            if rate < 0.15 or rate > 0.50:
                import warnings

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

        k = self.partition.assign(torch.tensor(e))
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
        """Bin visit flatness: ``1 - std(counts) / mean(counts)``.

        Returns 1.0 for perfectly flat visits, lower for uneven visits.
        Values can be negative if visits are extremely skewed.
        Returns 0.0 if no bins have been visited.
        """
        counts = self.counts.float()
        mean = counts.mean()
        if mean == 0:
            return 0.0
        return float(1.0 - counts.std() / mean)

    def flatness_history(self) -> list[float]:
        """Flatness at each recorded snapshot.

        Returns a list of flatness values, one per snapshot recorded
        every ``record_every`` steps.
        """
        result = []
        for snapshot in self.bin_counts_history:
            c = snapshot.float()
            mean = c.mean()
            if mean == 0:
                result.append(0.0)
            else:
                result.append(float(1.0 - c.std() / mean))
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

        keep = torch.rand(len(samples)) < p
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
