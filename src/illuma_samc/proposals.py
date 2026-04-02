"""Proposal distributions for SAMC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch


class Proposal(ABC):
    """Base class for MCMC proposal distributions."""

    @abstractmethod
    def propose(self, x: torch.Tensor) -> torch.Tensor:
        """Generate a proposal given the current state *x*."""

    @abstractmethod
    def log_ratio(self, x: torch.Tensor, x_new: torch.Tensor) -> float:
        """Return log q(x | x_new) - log q(x_new | x).

        For symmetric proposals this is 0.
        """


class GaussianProposal(Proposal):
    """Isotropic Gaussian random-walk proposal.

    Parameters
    ----------
    step_size : float
        Standard deviation of the Gaussian perturbation.
    """

    def __init__(self, step_size: float = 0.25) -> None:
        self._step_size = step_size

    def propose(self, x: torch.Tensor) -> torch.Tensor:
        return x + self._step_size * torch.randn_like(x)

    def log_ratio(self, x: torch.Tensor, x_new: torch.Tensor) -> float:
        return 0.0  # symmetric


class LangevinProposal(Proposal):
    """MALA-style Langevin proposal using autograd.

    Proposes: x_new = x + (step_size^2 / 2) * grad_log_pi(x) + step_size * noise

    Parameters
    ----------
    energy_fn : callable
        Energy function ``Tensor -> Tensor`` (scalar). The proposal uses
        ``-energy_fn`` as the log-density gradient.
    step_size : float
        Controls both drift and noise magnitude.
    """

    def __init__(
        self,
        energy_fn: Callable[[torch.Tensor], torch.Tensor],
        step_size: float = 0.01,
    ) -> None:
        self._energy_fn = energy_fn
        self._step_size = step_size

    def _grad_neg_energy(self, x: torch.Tensor) -> torch.Tensor:
        x_req = x.detach().requires_grad_(True)
        e = self._energy_fn(x_req)
        (grad,) = torch.autograd.grad(e, x_req)
        return -grad  # gradient of -energy = log-density direction

    def _mean(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.5 * self._step_size**2 * self._grad_neg_energy(x)

    def propose(self, x: torch.Tensor) -> torch.Tensor:
        mean = self._mean(x)
        return mean + self._step_size * torch.randn_like(x)

    def log_ratio(self, x: torch.Tensor, x_new: torch.Tensor) -> float:
        # log q(x | x_new) - log q(x_new | x)
        mu_fwd = self._mean(x)
        mu_rev = self._mean(x_new)
        ss2 = self._step_size**2

        log_q_rev = -0.5 * torch.sum((x - mu_rev) ** 2).item() / ss2
        log_q_fwd = -0.5 * torch.sum((x_new - mu_fwd) ** 2).item() / ss2

        return log_q_rev - log_q_fwd
