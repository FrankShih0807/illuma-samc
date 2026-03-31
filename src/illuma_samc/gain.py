"""Gain (step-size) sequences for SAMC weight updates.

The default gain schedule is the power-law form:

    γ(t) = γ₀ / (γ₁ + t)^α

where ``1/t`` is the special case ``γ₀ = t₀, γ₁ = 0, α = 1`` (with
a warmup floor at ``t₀``).  Custom callables are also supported.
"""

from __future__ import annotations

import math
from typing import Callable, Literal

import torch


class GainSequence:
    r"""Produces the gain (step-size) :math:`\gamma_t` used to update SAMC log-weights.

    The general form is:

    .. math::
        \gamma(t) = \frac{\gamma_0}{(\gamma_1 + t)^\alpha}

    with a warmup floor so that :math:`\gamma(t) \le 1`.

    Preset schedules
    ~~~~~~~~~~~~~~~~

    ``"1/t"`` (default)
        :math:`\gamma_0 = t_0,\; \gamma_1 = 0,\; \alpha = 1`

        Equivalent to :math:`t_0 / \max(t, t_0)`.  This is the standard
        SAMC gain from Liang (2007) that guarantees convergence.

    ``"ramp"``
        Constant gain during a warmup phase, then power-law decay
        matching Liang's reference C implementation.

    Parameters
    ----------
    schedule : str or callable
        ``"1/t"`` (default), ``"ramp"``, or a callable ``(int) -> float``.
    gamma0 : float
        Numerator scale (γ₀).  For ``"1/t"`` this equals *t0*.
    gamma1 : float
        Denominator offset (γ₁).  Default 0.
    alpha : float
        Decay exponent (α).  Default 1.
    t0 : float
        Legacy alias for *gamma0* when using the ``"1/t"`` schedule.
    rho, tau, warmup, step_scale : float/int
        Parameters for the ``"ramp"`` schedule (kept for backward compatibility).
    """

    def __init__(
        self,
        schedule: Literal["1/t", "ramp"] | Callable[[int], float] = "1/t",
        *,
        gamma0: float | None = None,
        gamma1: float = 0.0,
        alpha: float = 1.0,
        t0: float = 1000.0,
        rho: float = 1.0,
        tau: float = 1.0,
        warmup: int = 1,
        step_scale: int = 1000,
    ) -> None:
        # Handle legacy "log" as power-law with alpha=1 (same as 1/t)
        if schedule == "log":
            schedule = "1/t"

        if callable(schedule) and not isinstance(schedule, str):
            self._fn = schedule
            self._kind = "custom"
        elif schedule == "1/t":
            self._kind = "power"
            self._gamma0 = gamma0 if gamma0 is not None else t0
            self._gamma1 = gamma1
            self._alpha = alpha
        elif schedule == "ramp":
            self._kind = "ramp"
            self._rho = rho
            self._tau = tau
            self._warmup = warmup
            self._step_scale = step_scale
        else:
            raise ValueError(
                f"Unknown schedule {schedule!r}. Use '1/t', 'ramp', or a callable."
            )

    def __call__(self, t: int) -> float:
        """Return the gain value at iteration *t* (1-indexed)."""
        if self._kind == "custom":
            return self._fn(t)

        if self._kind == "power":
            # γ(t) = γ₀ / (γ₁ + t)^α, clamped to [0, 1]
            val = self._gamma0 / (self._gamma1 + t) ** self._alpha
            return min(val, 1.0)

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
