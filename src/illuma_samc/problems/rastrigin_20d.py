"""Rastrigin 20D problem — ~10^20 local minima, global min at origin."""

import math

import torch

NAME = "Rastrigin 20D"
DIM = 20

# Known mode: global minimum at the origin with E=0
MODES = torch.zeros(1, 20)


def energy_fn(z: torch.Tensor) -> torch.Tensor:
    """Batch-compatible Rastrigin energy.

    E(x) = 10*d + sum(xi^2 - 10*cos(2*pi*xi))
    Global minimum at the origin with E = 0, dim = 20.
    """
    if z.dim() == 1:
        z = z.unsqueeze(0)
    d = z.shape[-1]
    energy = 10 * d + torch.sum(z**2 - 10 * torch.cos(2 * math.pi * z), dim=-1)
    return energy.squeeze()
