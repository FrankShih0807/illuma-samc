"""Core SAMC sampler."""

from __future__ import annotations

import math
import warnings
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
    """Container for SAMC run results.

    Attributes
    ----------
    samples : Tensor
        Raw SAMC samples (drawn from the flattened distribution).
    log_weights : Tensor
        Final theta vector (log partition weights per bin).
    sample_log_weights : Tensor
        Per-sample importance log-weights for reweighting to the target
        distribution: ``theta[bin(sample)]``. Use ``self.importance_weights``
        for normalized weights that sum to 1.
    energy_history : Tensor
        Energy at every iteration.
    bin_counts : Tensor
        Number of visits per energy bin.
    acceptance_rate : float
    best_x : Tensor
    best_energy : float
    """

    samples: torch.Tensor
    log_weights: torch.Tensor
    sample_log_weights: torch.Tensor
    energy_history: torch.Tensor
    bin_counts: torch.Tensor
    acceptance_rate: float
    best_x: torch.Tensor
    best_energy: float

    @property
    def importance_weights(self) -> torch.Tensor:
        """Normalized importance weights for recovering the target distribution.

        SAMC samples from a flattened distribution. To compute expectations
        under the true target, weight each sample by these values::

            w = result.importance_weights
            mean_x = (w.unsqueeze(-1) * result.samples).sum(0)

        Returns zeros if all log-weights are ``-inf`` (no valid samples).
        """
        import warnings

        log_w = self.sample_log_weights
        if torch.all(log_w == float("-inf")):
            warnings.warn(
                "All sample log-weights are -inf — no valid samples for importance weighting. "
                "Returning zero weights.",
                stacklevel=2,
            )
            return torch.zeros_like(log_w)
        log_w = log_w - torch.logsumexp(log_w, dim=0)
        return torch.exp(log_w)


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
        Number of energy bins (simple mode with explicit range).
    e_min : float, optional
        Lower energy bound for uniform partition (simple mode). Default 0.0.
    e_max : float, optional
        Upper energy bound for uniform partition (simple mode). Default 20.0.
        If neither ``e_min``, ``e_max``, nor ``partition_fn`` are specified,
        auto-range mode is used (growing bins, no range needed).
    proposal_std : float
        Gaussian proposal step size (simple mode).
    temperature : float
        Temperature for the Boltzmann acceptance ratio. Higher values
        increase exploration (smoother energy landscape). Default 1.0.
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

    _UNSET = object()  # sentinel for detecting unspecified e_min/e_max

    def __init__(
        self,
        energy_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        *,
        dim: int,
        n_partitions: int = 42,
        n_bins: int | None = None,
        e_min: float = _UNSET,
        e_max: float = _UNSET,
        proposal_std: float = 0.25,
        temperature: float = 1.0,
        gain: GainSequence | str = "ramp",
        device: str | torch.device = "cpu",
        proposal_fn: Proposal | None = None,
        log_accept_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], float]
        | None = None,
        partition_fn: Partition | None = None,
        gain_kwargs: dict | None = None,
    ) -> None:
        # n_bins is an alias for n_partitions
        if n_bins is not None:
            n_partitions = n_bins

        # Detect whether user explicitly set e_min/e_max
        user_set_range = e_min is not self._UNSET or e_max is not self._UNSET
        if e_min is self._UNSET:
            e_min = 0.0
        if e_max is self._UNSET:
            e_max = 20.0

        # Detect whether user explicitly set n_partitions/n_bins
        user_set_bins = n_bins is not None or n_partitions != 42

        # --- Input validation ---
        if dim <= 0:
            raise ValueError("dim must be positive")
        if proposal_fn is None and proposal_std <= 0:
            raise ValueError("proposal_std must be positive")
        if partition_fn is None:
            if user_set_bins and n_partitions <= 0:
                raise ValueError("n_bins must be positive")
            if user_set_range and e_min >= e_max:
                raise ValueError("e_min must be less than e_max")

        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self._device = torch.device(device)
        self._dim = dim
        self._temperature = temperature

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
        self._auto_range = False
        if partition_fn is not None:
            self._partition = partition_fn
        elif user_set_range:
            self._partition = UniformPartition(e_min, e_max, n_partitions)
        else:
            # Auto-range: will run a warmup to discover energy range before main loop
            self._auto_range = True
            self._auto_n_partitions = n_partitions
            self._partition = UniformPartition(e_min, e_max, n_partitions)  # placeholder

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
        """Compute energy for a single state. Returns (energy_scalar, in_region)."""
        if self._energy_fn is None:
            raise RuntimeError("No energy_fn provided and no log_accept_fn — cannot compute energy")
        result = self._energy_fn(x)
        if isinstance(result, tuple):
            energy, in_region = result
            if isinstance(in_region, torch.Tensor):
                in_region = in_region.item()
            return energy.squeeze(), bool(in_region)
        return result.squeeze(), True

    def _compute_energy_batch(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute energy for a batch of states.

        Parameters
        ----------
        x : Tensor
            Shape ``(N, dim)``.

        Returns
        -------
        energies : Tensor
            Shape ``(N,)``.
        in_region : Tensor
            Boolean tensor, shape ``(N,)``.
        """
        if self._energy_fn is None:
            raise RuntimeError("No energy_fn provided — cannot compute energy")
        result = self._energy_fn(x)
        if isinstance(result, tuple):
            energy, in_region = result
            if not isinstance(in_region, torch.Tensor):
                in_region = torch.tensor(in_region, dtype=torch.bool, device=x.device)
            return energy.view(-1), in_region.view(-1).bool()
        return result.view(-1), torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

    def run(
        self,
        n_steps: int,
        x0: torch.Tensor | None = None,
        *,
        save_every: int = 100,
        burn_in: int = 0,
        progress: bool = True,
        seed: int | None = None,
    ) -> SAMCResult:
        """Run the SAMC sampler.

        Parameters
        ----------
        n_steps : int
            Number of MCMC iterations.
        x0 : Tensor, optional
            Initial state. Shape ``(dim,)`` for single chain or ``(N, dim)``
            for *N* parallel chains with shared weights.
        save_every : int
            Store a sample every this many iterations.
        burn_in : int
            Number of initial samples to discard from the result. Applied
            after the run: the first ``burn_in`` saved samples are dropped.
        progress : bool
            Show a tqdm progress bar (if tqdm is installed).

        Returns
        -------
        SAMCResult
            For single chain, ``samples`` has shape ``(n_saved, dim)``.
            For multi-chain, ``samples`` has shape ``(N, n_saved, dim)``.
        """
        if seed is not None:
            torch.manual_seed(seed)
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if burn_in < 0:
            raise ValueError("burn_in must be non-negative")
        if x0 is not None and x0.dim() == 2:
            result = self._run_multi_chain(n_steps, x0, save_every=save_every, progress=progress)
        else:
            result = self._run_single_chain(n_steps, x0, save_every=save_every, progress=progress)

        if burn_in > 0 and result.samples.shape[-2] > burn_in:
            # For multi-chain: shape (N, n_saved, dim), for single: (n_saved, dim)
            result = SAMCResult(
                samples=result.samples[..., burn_in:, :],
                log_weights=result.log_weights,
                sample_log_weights=result.sample_log_weights[..., burn_in:],
                energy_history=result.energy_history,
                bin_counts=result.bin_counts,
                acceptance_rate=result.acceptance_rate,
                best_x=result.best_x,
                best_energy=result.best_energy,
            )
        return result

    def _run_single_chain(
        self,
        n_steps: int,
        x0: torch.Tensor | None = None,
        *,
        save_every: int = 100,
        progress: bool = True,
    ) -> SAMCResult:
        device = self._device

        # Initialize state
        x = x0.to(device).clone() if x0 is not None else torch.zeros(self._dim, device=device)

        # Auto-range: run a short MH warmup to discover energy range
        if self._auto_range:
            warmup_energies = []
            x_warmup = x.clone()
            fx_warmup, _ = self._compute_energy(x_warmup)
            warmup_energies.append(fx_warmup.item())
            for _ in range(2000):
                x_new = self._proposal.propose(x_warmup)
                fy, in_reg = self._compute_energy(x_new)
                log_r = (-fy.item() + fx_warmup.item()) / self._temperature
                if in_reg and (log_r > 0 or torch.rand(1).item() < math.exp(min(log_r, 0))):
                    x_warmup = x_new
                    fx_warmup = fy
                warmup_energies.append(fx_warmup.item())
            e_min_auto = min(warmup_energies)
            e_max_auto = max(warmup_energies)
            margin = max(0.1 * (e_max_auto - e_min_auto), 0.5)
            self._partition = UniformPartition(
                e_min_auto - margin, e_max_auto + margin, self._auto_n_partitions
            )
            self._n_partitions = self._partition.n_partitions

        fx, in_region = self._compute_energy(x)
        fx_val = fx.item()

        # Warn if initial state is out of partition range
        if self._partition.assign(fx) < 0:
            e_edges = self._partition.edges
            warnings.warn(
                f"Initial energy {fx_val:.4g} is outside partition range "
                f"[{e_edges[0].item():.4g}, {e_edges[-1].item():.4g}]. "
                f"Chain may not mix.",
                stacklevel=3,
            )

        # Theta (log weights) and counts
        theta = torch.zeros(self._n_partitions, device=device, dtype=torch.float64)
        counts = torch.zeros(self._n_partitions, device=device, dtype=torch.float64)
        refden = 1.0 / self._n_partitions

        best_x = x.clone()
        best_energy = fx_val
        samples: list[torch.Tensor] = []
        sample_bins: list[int] = []
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

            # Reject if proposal lands outside partition range
            if k2 < 0:
                in_reg = False

            # Log acceptance ratio
            if self._log_accept_fn is not None:
                log_r = self._log_accept_fn(x, x_new, fx, fy)
            elif k1 < 0:
                # Current state out of range — always accept in-range proposals
                log_r = float("inf") if k2 >= 0 else float("-inf")
            else:
                log_r = theta[k1].item() - theta[k2].item() + (-fy_val + fx_val) / self._temperature

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
            elif k1 >= 0:
                theta -= delta * refden
                theta[k1] += delta
                counts[k1] += 1

            if fx_val < best_energy:
                best_energy = fx_val
                best_x = x.clone()

            energy_history.append(fx_val)

            if it % save_every == 0:
                cur_bin = self._partition.assign(fx)
                samples.append(x.clone())
                sample_bins.append(cur_bin)  # keep -1 for out-of-range

        acc_rate = accept_count / n_steps
        self.log_weights = theta
        self.energy_history = energy_history
        self.bin_counts = counts
        self.acceptance_rate = acc_rate
        self.best_x = best_x
        self.best_energy = best_energy

        if acc_rate < 0.01:
            warnings.warn(
                f"Very low acceptance rate ({acc_rate:.4f}). Consider increasing "
                f"proposal_std or widening the partition range.",
                stacklevel=2,
            )

        samples_tensor = (
            torch.stack(samples) if samples else torch.empty(0, self._dim, device=device)
        )

        # Per-sample importance log-weights: +theta[bin] reweights from flat to target.
        # SAMC samples from exp(-E/T - theta[k]), so weight by exp(+theta[k]) to
        # recover the target exp(-E/T). Only include samples from bins with positive
        # visit counts (unvisited bins have meaningless theta values).
        # Out-of-range samples (bin == -1) always get -inf.
        if sample_bins:
            bin_idx = torch.tensor(sample_bins, dtype=torch.long)
            in_range = bin_idx >= 0
            sample_log_w = torch.full((len(sample_bins),), float("-inf"), device=device)
            if in_range.any():
                valid_bins = bin_idx[in_range]
                visited_mask = counts[valid_bins] > 0
                # Create a sub-mask for in-range samples that were also visited
                in_range_indices = in_range.nonzero(as_tuple=True)[0]
                visited_indices = in_range_indices[visited_mask]
                sample_log_w[visited_indices] = theta[bin_idx[visited_indices]].float().to(device)
        else:
            sample_log_w = torch.empty(0, device=device)

        return SAMCResult(
            samples=samples_tensor,
            log_weights=theta.clone(),
            sample_log_weights=sample_log_w,
            energy_history=torch.tensor(energy_history, device=device),
            bin_counts=counts.clone(),
            acceptance_rate=accept_count / n_steps,
            best_x=best_x.clone(),
            best_energy=best_energy,
        )

    def _run_multi_chain(
        self,
        n_steps: int,
        x0: torch.Tensor,
        *,
        save_every: int = 100,
        progress: bool = True,
    ) -> SAMCResult:
        """Run N parallel chains with shared theta weights.

        Proposals and energy evaluations are batched across chains.
        Accept/reject and weight updates are sequential per chain to
        maintain correctness of the shared theta vector.
        """
        device = self._device
        n_chains = x0.shape[0]

        # Initialize states — shape (N, dim)
        x = x0.to(device).clone()

        # Auto-range: run a short MH warmup on first chain to discover energy range
        if self._auto_range:
            warmup_energies = []
            x_warmup = x[0].clone()
            fx_warmup, _ = self._compute_energy(x_warmup)
            warmup_energies.append(fx_warmup.item())
            for _ in range(2000):
                x_new = self._proposal.propose(x_warmup)
                fy, in_reg = self._compute_energy(x_new)
                log_r = (-fy.item() + fx_warmup.item()) / self._temperature
                if in_reg and (log_r > 0 or torch.rand(1).item() < math.exp(min(log_r, 0))):
                    x_warmup = x_new
                    fx_warmup = fy
                warmup_energies.append(fx_warmup.item())
            e_min_auto = min(warmup_energies)
            e_max_auto = max(warmup_energies)
            margin = max(0.1 * (e_max_auto - e_min_auto), 0.5)
            self._partition = UniformPartition(
                e_min_auto - margin, e_max_auto + margin, self._auto_n_partitions
            )
            self._n_partitions = self._partition.n_partitions

        fx, in_region = self._compute_energy_batch(x)  # (N,), (N,)

        # Warn if any initial state is out of partition range
        for c in range(n_chains):
            if self._partition.assign(fx[c]) < 0:
                e_edges = self._partition.edges
                warnings.warn(
                    f"Initial energy {fx[c].item():.4g} is outside partition range "
                    f"[{e_edges[0].item():.4g}, {e_edges[-1].item():.4g}]. "
                    f"Chain may not mix.",
                    stacklevel=3,
                )
                break  # one warning is enough

        # Shared theta on CPU (small vector, updated every step)
        theta = torch.zeros(self._n_partitions, dtype=torch.float64)
        counts = torch.zeros(self._n_partitions, dtype=torch.float64)
        refden = 1.0 / self._n_partitions

        best_x = x[0].clone()
        best_energy = fx.min().item()
        best_idx = fx.argmin().item()
        best_x = x[best_idx].clone()

        # Per-chain sample storage: list of (N, dim) snapshots
        samples: list[torch.Tensor] = []
        sample_bins_per_step: list[list[int]] = []  # list of [N] bin indices per save
        energy_history: list[torch.Tensor] = []
        accept_count = 0
        total_decisions = 0

        iterator = range(1, n_steps + 1)
        if progress and tqdm is not None:
            iterator = tqdm(iterator, desc=f"SAMC ({n_chains} chains)")

        for it in iterator:
            delta = self._gain(it)

            # --- Batch propose across all chains ---
            x_new = self._proposal.propose(x)  # (N, dim)

            # --- Batch energy evaluation ---
            fy, in_reg = self._compute_energy_batch(x_new)  # (N,), (N,)

            # --- Sequential accept/reject per chain (shared theta) ---
            for c in range(n_chains):
                fx_c = fx[c].item()
                fy_c = fy[c].item()
                k1 = self._partition.assign(fx[c])
                k2 = self._partition.assign(fy[c])

                in_reg_c = in_reg[c].item()
                # Reject if proposal lands outside partition range
                if k2 < 0:
                    in_reg_c = False

                # Log acceptance ratio
                if self._log_accept_fn is not None:
                    log_r = self._log_accept_fn(x[c], x_new[c], fx[c], fy[c])
                elif k1 < 0:
                    log_r = float("inf") if k2 >= 0 else float("-inf")
                else:
                    log_r = theta[k1].item() - theta[k2].item() + (-fy_c + fx_c) / self._temperature

                # Proposal correction (for asymmetric proposals)
                if hasattr(self._proposal, "log_ratio"):
                    log_r += self._proposal.log_ratio(x[c], x_new[c])

                # Accept/reject
                if not in_reg_c:
                    accept = False
                elif log_r > 0:
                    accept = True
                else:
                    accept = torch.rand(1).item() < math.exp(log_r)

                if accept:
                    x[c] = x_new[c]
                    fx[c] = fy[c]
                    theta -= delta * refden
                    theta[k2] += delta
                    counts[k2] += 1
                    accept_count += 1
                elif k1 >= 0:
                    theta -= delta * refden
                    theta[k1] += delta
                    counts[k1] += 1

                total_decisions += 1

                if fx[c].item() < best_energy:
                    best_energy = fx[c].item()
                    best_x = x[c].clone()

            # Store per-step energy for all chains
            energy_history.append(fx.clone().cpu())

            if it % save_every == 0:
                samples.append(x.clone())  # (N, dim)
                sample_bins_per_step.append(
                    [self._partition.assign(fx[c]) for c in range(n_chains)]
                )

        acc_rate = accept_count / total_decisions if total_decisions > 0 else 0.0

        self.log_weights = theta.to(device)
        self.energy_history = energy_history
        self.bin_counts = counts.to(device)
        self.acceptance_rate = acc_rate
        self.best_x = best_x
        self.best_energy = best_energy

        if acc_rate < 0.01:
            warnings.warn(
                f"Very low acceptance rate ({acc_rate:.4f}). Consider increasing "
                f"proposal_std or widening the partition range.",
                stacklevel=2,
            )

        # samples: list of (N, dim) → (N, n_saved, dim)
        if samples:
            stacked = torch.stack(samples)  # (n_saved, N, dim)
            samples_tensor = stacked.permute(1, 0, 2)  # (N, n_saved, dim)
        else:
            samples_tensor = torch.empty(n_chains, 0, self._dim, device=device)

        # Per-sample importance log-weights: +theta[bin] → (N, n_saved)
        # SAMC samples from exp(-E/T - theta[k]), weight by exp(+theta[k]).
        # Only include samples from bins with positive visit counts.
        # Out-of-range samples (bin == -1) always get -inf.
        if sample_bins_per_step:
            # (n_saved, N)
            bin_idx = torch.tensor(sample_bins_per_step, dtype=torch.long)
            in_range = bin_idx >= 0
            sample_log_w = torch.full_like(
                bin_idx, float("-inf"), dtype=torch.float32, device=device
            )
            if in_range.any():
                valid_bins = bin_idx[in_range]
                visited_mask = counts[valid_bins] > 0
                # Indices of in-range entries that also have positive counts
                in_range_flat = in_range.nonzero(as_tuple=False)
                visited_flat = in_range_flat[visited_mask]
                for idx in visited_flat:
                    r, c = idx[0].item(), idx[1].item()
                    sample_log_w[r, c] = theta[bin_idx[r, c]].float().to(device)
            sample_log_w = sample_log_w.permute(1, 0)  # (N, n_saved)
        else:
            sample_log_w = torch.empty(n_chains, 0, device=device)

        # energy_history: list of (N,) → (n_steps, N) on device
        energy_tensor = torch.stack(energy_history).to(device)  # (n_steps, N)

        return SAMCResult(
            samples=samples_tensor,
            log_weights=theta.to(device).clone(),
            sample_log_weights=sample_log_w,
            energy_history=energy_tensor,
            bin_counts=counts.to(device).clone(),
            acceptance_rate=acc_rate,
            best_x=best_x.clone(),
            best_energy=best_energy,
        )

    def plot_diagnostics(self, **kwargs) -> None:
        """Convenience wrapper around :func:`diagnostics.plot_diagnostics`."""
        from illuma_samc.diagnostics import plot_diagnostics

        return plot_diagnostics(self, **kwargs)
