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
    ) -> None:
        self.partition = partition
        self.gain = gain

        n = partition.n_partitions
        self.theta = torch.zeros(n, device=device, dtype=torch.float64)
        self.counts = torch.zeros(n, device=device, dtype=torch.float64)
        self._refden = 1.0 / n
        self._t = 0

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
        ky = self.partition.assign(torch.tensor(ey))

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
        k = self.partition.assign(torch.tensor(e))

        if k < 0:
            return

        delta = self.gain(t)
        self.theta -= delta * self._refden
        self.theta[k] += delta
        self.counts[k] += 1

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

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

    def importance_log_weights(
        self,
        energies: torch.Tensor,
    ) -> torch.Tensor:
        r"""Per-sample importance log-weights for reweighting to the target.

        SAMC samples from a flattened distribution. To recover expectations
        under the true (Boltzmann) target, weight each sample by
        :math:`\exp(-\theta[k_i])` (unnormalized).

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
        log_w = torch.full(energies.shape, float("-inf"), dtype=torch.float64)
        for i, e in enumerate(energies):
            k = self.partition.assign(e)
            if k >= 0 and self.counts[k] > 0:
                log_w[i] = -self.theta[k].item()
        return log_w

    def importance_weights(
        self,
        energies: torch.Tensor,
    ) -> torch.Tensor:
        """Normalized importance weights that sum to 1.

        Use these to compute expectations under the target distribution::

            weighted_mean = (weights * values).sum()

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
        # Stable softmax-style normalization
        log_w_shifted = log_w - log_w[log_w > float("-inf")].max()
        w = torch.exp(log_w_shifted)
        return w / w.sum()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return serializable state for checkpointing."""
        return {
            "theta": self.theta.clone(),
            "counts": self.counts.clone(),
            "t": self._t,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from a checkpoint."""
        self.theta.copy_(state["theta"])
        self.counts.copy_(state["counts"])
        self._t = state["t"]
