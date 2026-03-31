"""Gain (step-size) sequences for SAMC weight updates."""

from __future__ import annotations

import math
from typing import Callable, Literal

import torch


class GainSequence:
    """Produces the gain (step-size) γ_t used to update SAMC log-weights.

    Parameters
    ----------
    schedule : str or callable
        One of ``"1/t"``, ``"log"``, ``"ramp"``, or a callable ``(int) -> float``.
    t0 : float
        Warm-up / scaling parameter for the ``"1/t"`` and ``"log"`` schedules.
    rho : float
        Initial gain magnitude for the ``"ramp"`` schedule.
    tau : float
        Decay exponent for the ``"ramp"`` schedule.
    warmup : int
        Number of warm-up steps at constant gain for the ``"ramp"`` schedule.
    step_scale : int
        Iteration scaling factor for the ``"ramp"`` schedule.
    """

    def __init__(
        self,
        schedule: Literal["1/t", "log", "ramp"] | Callable[[int], float] = "1/t",
        *,
        t0: float = 1000.0,
        rho: float = 1.0,
        tau: float = 1.0,
        warmup: int = 1,
        step_scale: int = 1000,
    ) -> None:
        if callable(schedule) and not isinstance(schedule, str):
            self._fn = schedule
            self._kind = "custom"
        elif schedule == "1/t":
            self._kind = "1/t"
            self._t0 = t0
        elif schedule == "log":
            self._kind = "log"
            self._t0 = t0
        elif schedule == "ramp":
            self._kind = "ramp"
            self._rho = rho
            self._tau = tau
            self._warmup = warmup
            self._step_scale = step_scale
        else:
            raise ValueError(
                f"Unknown schedule {schedule!r}. Use '1/t', 'log', 'ramp', or a callable."
            )

    def __call__(self, t: int) -> float:
        """Return the gain value at iteration *t* (1-indexed)."""
        if self._kind == "custom":
            return self._fn(t)

        if self._kind == "1/t":
            return self._t0 / max(t, self._t0)

        if self._kind == "log":
            return self._t0 / max(t * math.log(t + math.e), self._t0)

        # ramp — matches sample_code.py logic
        if t <= self._warmup * self._step_scale:
            return self._rho
        offset = (self._warmup - 1) * self._step_scale
        return self._rho * math.exp(-self._tau * math.log((t - offset) / self._step_scale))

    def as_tensor(self, n_steps: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """Pre-compute gains for iterations 1 .. *n_steps* as a 1-D tensor."""
        return torch.tensor(
            [self(t) for t in range(1, n_steps + 1)],
            dtype=torch.float64,
            device=device,
        )
