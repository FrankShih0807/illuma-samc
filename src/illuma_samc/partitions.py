"""Energy-space partitions for SAMC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

import torch


class Partition(ABC):
    """Base class for energy-space partitions."""

    @abstractmethod
    def assign(self, energy: torch.Tensor) -> int:
        """Return the bin index for a scalar energy value."""

    def assign_batch(self, energies: torch.Tensor) -> torch.Tensor:
        """Vectorized bin assignment for a 1-D tensor of energies.

        Returns a LongTensor of bin indices. Out-of-range energies get -1.
        Default implementation calls :meth:`assign` in a loop; subclasses
        may override for efficiency.
        """
        result = torch.empty(energies.shape[0], dtype=torch.long)
        for i in range(energies.shape[0]):
            result[i] = self.assign(energies[i])
        return result

    @property
    @abstractmethod
    def n_partitions(self) -> int:
        """Number of bins."""

    @property
    @abstractmethod
    def edges(self) -> torch.Tensor:
        """Bin edges as a 1-D tensor of length ``n_partitions + 1``."""


class UniformPartition(Partition):
    """Linearly-spaced energy bins matching ``sample_code.py`` style.

    Parameters
    ----------
    e_min : float
        Lower energy bound.
    e_max : float
        Upper energy bound.
    n_bins : int
        Number of bins.
    overflow_bins : bool
        If ``True``, add two catch-all bins: ``[-inf, e_min]`` (bin 0) and
        ``[e_max, +inf]`` (last bin). The original bins are shifted by 1.
        Default ``False`` for backward compatibility.
    """

    def __init__(
        self,
        e_min: float,
        e_max: float,
        n_bins: int,
        overflow_bins: bool = False,
    ) -> None:
        self._e_min = e_min
        self._e_max = e_max
        self._n_bins_core = n_bins
        self._overflow_bins = overflow_bins
        self._scale = n_bins / (e_max - e_min)

        inner_edges = torch.linspace(e_min, e_max, n_bins + 1)
        if overflow_bins:
            self._edges = torch.cat(
                [torch.tensor([float("-inf")]), inner_edges, torch.tensor([float("inf")])]
            )
        else:
            self._edges = inner_edges

    def assign(self, energy: torch.Tensor) -> int:
        """Return bin index, or -1 if energy is outside range (no overflow bins)."""
        e = energy.item() if isinstance(energy, torch.Tensor) else float(energy)
        if self._overflow_bins:
            if e < self._e_min:
                return 0  # low overflow bin
            if e > self._e_max:
                return self._n_bins_core + 1  # high overflow bin
            idx = int((e - self._e_min) * self._scale)
            return min(idx, self._n_bins_core - 1) + 1  # shift by 1 for low overflow
        else:
            if e > self._e_max or e < self._e_min:
                return -1
            idx = int((e - self._e_min) * self._scale)
            return min(idx, self._n_bins_core - 1)

    def assign_batch(self, energies: torch.Tensor) -> torch.Tensor:
        """Vectorized bin assignment using the same formula as :meth:`assign`."""
        idx = ((energies.double() - self._e_min) * self._scale).long()
        idx = idx.clamp(0, self._n_bins_core - 1)

        if self._overflow_bins:
            idx = idx + 1  # shift for low overflow bin
            below = energies < self._e_min
            above = energies > self._e_max
            idx[below] = 0
            idx[above] = self._n_bins_core + 1
        else:
            out_of_range = (energies < self._e_min) | (energies > self._e_max)
            idx[out_of_range] = -1
        return idx

    @property
    def n_partitions(self) -> int:
        if self._overflow_bins:
            return self._n_bins_core + 2
        return self._n_bins_core

    @property
    def edges(self) -> torch.Tensor:
        """Bin edges as a 1-D tensor of length ``n_partitions + 1``."""
        return self._edges


class AdaptivePartition(Partition):
    """Partition that recomputes boundaries from visited energies.

    Starts with uniform bins and adapts after collecting enough samples.

    Parameters
    ----------
    e_min : float
        Initial lower energy bound.
    e_max : float
        Initial upper energy bound.
    n_bins : int
        Number of bins.
    adapt_interval : int
        Recompute boundaries every this many calls to :meth:`record_and_assign`.
    min_samples : int
        Minimum recorded energies before adapting.
    """

    def __init__(
        self,
        e_min: float,
        e_max: float,
        n_bins: int,
        adapt_interval: int = 5000,
        min_samples: int = 1000,
        max_history: int = 50_000,
    ) -> None:
        self._n_bins = n_bins
        self._edges = torch.linspace(e_min, e_max, n_bins + 1, dtype=torch.float64)
        self._history: deque[float] = deque(maxlen=max_history)
        self._adapt_interval = adapt_interval
        self._min_samples = min_samples
        self._call_count = 0

    def assign(self, energy: torch.Tensor) -> int:
        e = energy.item() if isinstance(energy, torch.Tensor) else float(energy)
        b = self._bin_for(e)
        if b >= 0:
            self._history.append(e)
        self._call_count += 1
        if self._call_count % self._adapt_interval == 0 and len(self._history) >= self._min_samples:
            self._adapt()
        return b

    def _bin_for(self, e: float) -> int:
        if e < self._edges[0].item() or e > self._edges[-1].item():
            return -1
        idx = int(torch.searchsorted(self._edges, torch.tensor(e, dtype=torch.float64)).item()) - 1
        return max(0, min(idx, self._n_bins - 1))

    def _adapt(self) -> None:
        t = torch.tensor(self._history, dtype=torch.float64)
        lo, hi = t.min().item(), t.max().item()
        if lo < hi:
            self._edges = torch.linspace(lo, hi, self._n_bins + 1, dtype=torch.float64)

    @property
    def n_partitions(self) -> int:
        return self._n_bins

    @property
    def edges(self) -> torch.Tensor:
        return self._edges


class QuantilePartition(Partition):
    """Partition whose boundaries are quantiles of a warmup energy sample.

    Parameters
    ----------
    energies : torch.Tensor
        1-D tensor of energy values from a warmup run.
    n_bins : int
        Number of bins.
    """

    def __init__(self, energies: torch.Tensor, n_bins: int) -> None:
        self._n_bins = n_bins
        quantiles = torch.linspace(0, 1, n_bins + 1, dtype=torch.float64)
        self._edges = torch.quantile(energies.double(), quantiles)
        # Ensure strictly increasing edges
        self._edges[-1] = self._edges[-1] + 1e-6

    def assign(self, energy: torch.Tensor) -> int:
        e = energy.item() if isinstance(energy, torch.Tensor) else float(energy)
        if e < self._edges[0].item() or e > self._edges[-1].item():
            return -1
        idx = int(torch.searchsorted(self._edges, torch.tensor(e, dtype=torch.float64)).item()) - 1
        return max(0, min(idx, self._n_bins - 1))

    @property
    def n_partitions(self) -> int:
        return self._n_bins

    @property
    def edges(self) -> torch.Tensor:
        return self._edges


class ExpandablePartition(Partition):
    """Uniform partition that dynamically expands when out-of-range energies arrive.

    When :meth:`assign` receives an energy outside the current range, the
    partition extends in that direction by ``expand_step`` bins, up to
    ``max_bins`` total. The bin width stays the same as the original partition.

    After expansion, callers must call ``wm._resize_for_partition()`` to sync
    the theta/counts vectors. The :attr:`expanded` flag indicates whether an
    expansion occurred since the last reset.

    Parameters
    ----------
    e_min : float
        Initial lower energy bound.
    e_max : float
        Initial upper energy bound.
    n_bins : int
        Initial number of bins.
    expand_step : int
        Number of bins to add per expansion. Default 5.
    max_bins : int
        Maximum total bins. Default 200.
    """

    def __init__(
        self,
        e_min: float,
        e_max: float,
        n_bins: int,
        expand_step: int = 5,
        max_bins: int = 200,
    ) -> None:
        self._e_min = e_min
        self._e_max = e_max
        self._n_bins = n_bins
        self._expand_step = expand_step
        self._max_bins = max_bins
        self._bin_width = (e_max - e_min) / n_bins
        self._edges = torch.linspace(e_min, e_max, n_bins + 1)
        self.expanded = False  # flag for callers to check

    def _expand_low(self, energy: float) -> None:
        """Expand partition to cover energy below current e_min."""
        if self._n_bins >= self._max_bins:
            return
        add = min(self._expand_step, self._max_bins - self._n_bins)
        self._e_min = self._e_min - add * self._bin_width
        self._n_bins += add
        self._edges = torch.linspace(self._e_min, self._e_max, self._n_bins + 1)
        self.expanded = True

    def _expand_high(self, energy: float) -> None:
        """Expand partition to cover energy above current e_max."""
        if self._n_bins >= self._max_bins:
            return
        add = min(self._expand_step, self._max_bins - self._n_bins)
        self._e_max = self._e_max + add * self._bin_width
        self._n_bins += add
        self._edges = torch.linspace(self._e_min, self._e_max, self._n_bins + 1)
        self.expanded = True

    def assign(self, energy: torch.Tensor) -> int:
        """Return bin index. May trigger expansion for out-of-range energies."""
        e = energy.item() if isinstance(energy, torch.Tensor) else float(energy)
        if e < self._e_min:
            self._expand_low(e)
        elif e > self._e_max:
            self._expand_high(e)
        # Now assign (after potential expansion)
        if e < self._e_min or e > self._e_max:
            return -1  # still out of range (hit max_bins)
        scale = self._n_bins / (self._e_max - self._e_min)
        idx = int((e - self._e_min) * scale)
        return min(idx, self._n_bins - 1)

    def assign_batch(self, energies: torch.Tensor) -> torch.Tensor:
        """Vectorized assignment. May trigger expansion."""
        e_min_val = energies.min().item()
        e_max_val = energies.max().item()
        if e_min_val < self._e_min:
            self._expand_low(e_min_val)
        if e_max_val > self._e_max:
            self._expand_high(e_max_val)

        scale = self._n_bins / (self._e_max - self._e_min)
        idx = ((energies.double() - self._e_min) * scale).long()
        idx = idx.clamp(0, self._n_bins - 1)
        out_of_range = (energies < self._e_min) | (energies > self._e_max)
        idx[out_of_range] = -1
        return idx

    @property
    def n_partitions(self) -> int:
        return self._n_bins

    @property
    def edges(self) -> torch.Tensor:
        return self._edges
