"""Lightweight SAMC weight manager — torch.optim style.

The user owns the sampling loop (proposal, energy evaluation, accept/reject).
This class only manages the bin weights (theta) and provides the SAMC
acceptance correction.

Usage::

    wm = SAMCWeights(
        partition=UniformPartition(e_min=0, e_max=10, n_bins=50),
        gain=GainSequence("1/t", t0=1000),
    )

    x = initial_state()
    fx = energy_fn(x)

    for t in range(1, n_steps + 1):
        x_new = propose(x)
        fy = energy_fn(x_new)

        log_r = wm.log_accept_ratio(fx, fy)
        accept = log_r > 0 or torch.rand(1).item() < math.exp(log_r)

        if accept:
            wm.step(t, fy)
            x, fx = x_new, fy
        else:
            wm.step(t, fx)
"""

from __future__ import annotations

import torch

from illuma_samc.gain import GainSequence
from illuma_samc.partitions import Partition


class SAMCWeights:
    r"""Manages SAMC bin weights (theta) with a torch.optim-style interface.

    This class handles only the weight bookkeeping:

    1. **Log acceptance ratio** — the SAMC correction term
       :math:`\theta[k_x] - \theta[k_y] + (-E_y + E_x) / T`
    2. **Weight update** — after accept/reject, update theta and bin counts

    The user controls everything else: proposal, energy evaluation, accept/reject.

    Parameters
    ----------
    partition : Partition
        Energy-space partition (defines bins).
    gain : GainSequence
        Step-size schedule for weight updates.
    temperature : float
        Temperature for the energy term in acceptance ratio. Default 1.0.
    device : str or torch.device
        Device for theta and counts tensors. Default "cpu".
    """

    def __init__(
        self,
        partition: Partition,
        gain: GainSequence,
        *,
        temperature: float = 1.0,
        device: torch.device | str = "cpu",
    ) -> None:
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")

        self.partition = partition
        self.gain = gain
        self.temperature = temperature

        n = partition.n_partitions
        self.theta = torch.zeros(n, device=device, dtype=torch.float64)
        self.counts = torch.zeros(n, device=device, dtype=torch.float64)
        self._refden = 1.0 / n
        self._t = 0  # current iteration (updated on each step)

    @property
    def n_bins(self) -> int:
        """Number of energy bins."""
        return self.partition.n_partitions

    @property
    def log_weights(self) -> torch.Tensor:
        """Current theta vector (log partition weights)."""
        return self.theta

    def log_accept_ratio(
        self,
        energy_current: float | torch.Tensor,
        energy_proposed: float | torch.Tensor,
    ) -> float:
        """Compute the SAMC log acceptance ratio.

        .. math::
            \\log r = \\theta[k_x] - \\theta[k_y] + (-E_y + E_x) / T

        Returns ``-inf`` if the proposed energy is out of partition range.
        Returns ``+inf`` if current energy is out of range but proposed is in range.

        Parameters
        ----------
        energy_current : float or Tensor
            Energy of current state.
        energy_proposed : float or Tensor
            Energy of proposed state.

        Returns
        -------
        float
            Log acceptance ratio (includes SAMC weight correction).
        """
        ex = float(energy_current)
        ey = float(energy_proposed)

        kx = self.partition.assign(torch.tensor(ex))
        ky = self.partition.assign(torch.tensor(ey))

        if ky < 0:
            return float("-inf")
        if kx < 0:
            return float("inf")

        return self.theta[kx].item() - self.theta[ky].item() + (-ey + ex) / self.temperature

    def step(self, t: int, energy: float | torch.Tensor) -> None:
        """Update bin weights after accept/reject.

        Call this once per iteration with the energy of the *current* state
        (i.e., the accepted state after accept/reject).

        Parameters
        ----------
        t : int
            Current iteration number (1-indexed). Used to compute gain.
        energy : float or Tensor
            Energy of the current state after accept/reject.
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
