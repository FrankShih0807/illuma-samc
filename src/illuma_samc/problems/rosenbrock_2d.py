"""Rosenbrock 2D problem — narrow curved valley, global min at (1,1)."""

import torch

NAME = "Rosenbrock 2D"
DIM = 2

# Known mode: global minimum at (1, 1) with E=0
MODES = torch.tensor([[1.0, 1.0]])


def energy_fn(z: torch.Tensor) -> torch.Tensor:
    """Batch-compatible Rosenbrock energy.

    E(x) = (1 - x1)^2 + 100 * (x2 - x1^2)^2
    Global minimum at (1, 1) with E = 0.
    """
    if z.dim() == 1:
        z = z.unsqueeze(0)
    x1, x2 = z[:, 0], z[:, 1]
    energy = (1 - x1) ** 2 + 100 * (x2 - x1**2) ** 2
    return energy.squeeze()
