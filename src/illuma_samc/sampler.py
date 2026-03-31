"""Core SAMC sampler."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

from illuma_samc.gain import GainSequence
from illuma_samc.partitions import Partition, UniformPartition
from illuma_samc.proposals import GaussianProposal, Proposal

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]


@dataclass
class SAMCResult:
    """Container for SAMC run results."""

    samples: torch.Tensor
    log_weights: torch.Tensor
    energy_history: torch.Tensor
    bin_counts: torch.Tensor
    acceptance_rate: float
    best_x: torch.Tensor
    best_energy: float


class SAMC:
    """Stochastic Approximation Monte Carlo sampler.

    Two usage modes:

    **Simple mode** — provide an energy function, sampler handles the rest::

        sampler = SAMC(energy_fn=my_energy, dim=2, n_partitions=30)
        result = sampler.run(n_steps=100_000)

    **Flexible mode** — provide custom components::

        sampler = SAMC(
            dim=2,
            n_partitions=30,
            proposal_fn=my_proposal,
            log_accept_fn=my_accept,
            partition_fn=my_partition,
        )
        result = sampler.run(n_steps=100_000)

    Parameters
    ----------
    energy_fn : callable, optional
        Energy function ``Tensor -> Tensor`` (scalar). Required for simple mode.
    dim : int
        Dimensionality of the sample space.
    n_partitions : int
        Number of energy bins.
    e_min : float
        Lower energy bound for uniform partition (simple mode).
    e_max : float
        Upper energy bound for uniform partition (simple mode).
    proposal_std : float
        Gaussian proposal step size (simple mode).
    gain : GainSequence or str
        Gain schedule. Defaults to ``"ramp"`` matching sample_code.py.
    device : str or torch.device
        Device for tensors.
    proposal_fn : Proposal, optional
        Custom proposal (flexible mode).
    log_accept_fn : callable, optional
        Custom log-acceptance: ``(x, x_new, energy_x, energy_new) -> float``.
        If provided, replaces the default Boltzmann acceptance.
    partition_fn : Partition, optional
        Custom partition (flexible mode).
    gain_kwargs : dict, optional
        Extra keyword arguments forwarded to :class:`GainSequence`.
    """

    def __init__(
        self,
        energy_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        *,
        dim: int,
        n_partitions: int = 42,
        e_min: float = -8.2,
        e_max: float = 0.0,
        proposal_std: float = 0.25,
        gain: GainSequence | str = "ramp",
        device: str | torch.device = "cpu",
        proposal_fn: Proposal | None = None,
        log_accept_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], float]
        | None = None,
        partition_fn: Partition | None = None,
        gain_kwargs: dict | None = None,
    ) -> None:
        self._device = torch.device(device)
        self._dim = dim

        # --- Energy function ---
        self._energy_fn = energy_fn

        # --- Proposal ---
        if proposal_fn is not None:
            self._proposal = proposal_fn
        else:
            self._proposal = GaussianProposal(step_size=proposal_std)

        # --- Acceptance ---
        self._log_accept_fn = log_accept_fn

        # --- Partition ---
        if partition_fn is not None:
            self._partition = partition_fn
        else:
            self._partition = UniformPartition(e_min, e_max, n_partitions)

        # --- Gain ---
        if isinstance(gain, GainSequence):
            self._gain = gain
        else:
            kw = gain_kwargs or {}
            self._gain = GainSequence(gain, **kw)

        self._n_partitions = self._partition.n_partitions

        # --- State (populated by run()) ---
        self.log_weights: torch.Tensor | None = None
        self.energy_history: list[float] = []
        self.bin_counts: torch.Tensor | None = None
        self.acceptance_rate: float = 0.0
        self.best_x: torch.Tensor | None = None
        self.best_energy: float = float("inf")
        self._samples: list[torch.Tensor] = []

    def _compute_energy(self, x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Compute energy. Returns (energy_scalar, in_region)."""
        if self._energy_fn is None:
            raise RuntimeError("No energy_fn provided and no log_accept_fn — cannot compute energy")
        result = self._energy_fn(x)
        if isinstance(result, tuple):
            energy, in_region = result
            if isinstance(in_region, torch.Tensor):
                in_region = in_region.item()
            return energy.squeeze(), bool(in_region)
        return result.squeeze(), True

    def run(
        self,
        n_steps: int,
        x0: torch.Tensor | None = None,
        *,
        save_every: int = 100,
        progress: bool = True,
    ) -> SAMCResult:
        """Run the SAMC sampler.

        Parameters
        ----------
        n_steps : int
            Number of MCMC iterations.
        x0 : Tensor, optional
            Initial state. Defaults to zeros.
        save_every : int
            Store a sample every this many iterations.
        progress : bool
            Show a tqdm progress bar (if tqdm is installed).

        Returns
        -------
        SAMCResult
        """
        device = self._device

        # Initialize state
        x = x0.to(device).clone() if x0 is not None else torch.zeros(self._dim, device=device)
        fx, in_region = self._compute_energy(x)
        fx_val = fx.item()

        # Theta (log weights) and counts
        theta = torch.zeros(self._n_partitions, device=device, dtype=torch.float64)
        counts = torch.zeros(self._n_partitions, device=device, dtype=torch.float64)
        refden = 1.0 / self._n_partitions

        best_x = x.clone()
        best_energy = fx_val
        samples: list[torch.Tensor] = []
        energy_history: list[float] = []
        accept_count = 0

        iterator = range(1, n_steps + 1)
        if progress and tqdm is not None:
            iterator = tqdm(iterator, desc="SAMC")

        for it in iterator:
            delta = self._gain(it)
            k1 = self._partition.assign(fx)

            # Propose
            x_new = self._proposal.propose(x)
            fy, in_reg = self._compute_energy(x_new)
            fy_val = fy.item()
            k2 = self._partition.assign(fy)

            # Log acceptance ratio
            if self._log_accept_fn is not None:
                log_r = self._log_accept_fn(x, x_new, fx, fy)
            else:
                log_r = theta[k1].item() - theta[k2].item() - fy_val + fx_val

            # Proposal correction (for asymmetric proposals)
            if hasattr(self._proposal, "log_ratio"):
                log_r += self._proposal.log_ratio(x, x_new)

            # Accept/reject
            if not in_reg:
                accept = False
            elif log_r > 0:
                accept = True
            else:
                accept = torch.rand(1).item() < math.exp(log_r)

            if accept:
                x = x_new.clone()
                fx = fy
                fx_val = fy_val
                theta -= delta * refden
                theta[k2] += delta
                counts[k2] += 1
                accept_count += 1
            else:
                theta -= delta * refden
                theta[k1] += delta
                counts[k1] += 1

            if fx_val < best_energy:
                best_energy = fx_val
                best_x = x.clone()

            energy_history.append(fx_val)

            if it % save_every == 0:
                samples.append(x.clone())

        self.log_weights = theta
        self.energy_history = energy_history
        self.bin_counts = counts
        self.acceptance_rate = accept_count / n_steps
        self.best_x = best_x
        self.best_energy = best_energy

        samples_tensor = (
            torch.stack(samples) if samples else torch.empty(0, self._dim, device=device)
        )

        return SAMCResult(
            samples=samples_tensor,
            log_weights=theta.clone(),
            energy_history=torch.tensor(energy_history, device=device),
            bin_counts=counts.clone(),
            acceptance_rate=accept_count / n_steps,
            best_x=best_x.clone(),
            best_energy=best_energy,
        )

    def plot_diagnostics(self, **kwargs) -> None:
        """Convenience wrapper around :func:`diagnostics.plot_diagnostics`."""
        from illuma_samc.diagnostics import plot_diagnostics

        return plot_diagnostics(self, **kwargs)
