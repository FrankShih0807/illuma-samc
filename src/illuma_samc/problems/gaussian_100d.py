"""100D Gaussian mixture with well-separated modes."""

import torch

NAME = "100D Gaussian Mixture"
DIM = 100

# 8 modes at distance 10 in different axis-aligned directions
_MODES = torch.zeros(8, 100)
_MODES[0, 0] = 10.0
_MODES[1, 0] = -10.0
_MODES[2, 1] = 10.0
_MODES[3, 1] = -10.0
_MODES[4, 2] = 10.0
_MODES[5, 2] = -10.0
_MODES[6, 3] = 10.0
_MODES[7, 3] = -10.0


def energy_fn(z: torch.Tensor) -> torch.Tensor:
    """Batch-compatible 100D Gaussian mixture energy.

    E(x) = -log sum_k exp(-0.5 * ||x - mu_k||^2)
    """
    if z.dim() == 1:
        z = z.unsqueeze(0)
    diffs = z.unsqueeze(1) - _MODES.unsqueeze(0)
    log_components = -0.5 * torch.sum(diffs**2, dim=-1)
    energy = -torch.logsumexp(log_components, dim=-1)
    return energy.squeeze()
