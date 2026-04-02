"""10D Gaussian mixture with well-separated modes."""

import torch

NAME = "10D Gaussian Mixture"
DIM = 10

# 4 modes at distance 10 from origin in different directions
_MODES_10D = torch.zeros(4, 10)
_MODES_10D[0, 0] = 10.0
_MODES_10D[1, 0] = -10.0
_MODES_10D[2, 1] = 10.0
_MODES_10D[3, 1] = -10.0


def energy_fn(z: torch.Tensor) -> torch.Tensor:
    """Batch-compatible 10D Gaussian mixture energy.

    E(x) = -log sum_k exp(-0.5 * ||x - mu_k||^2)
    """
    if z.dim() == 1:
        z = z.unsqueeze(0)
    # z: (N, 10), modes: (4, 10) -> diffs: (N, 4, 10)
    diffs = z.unsqueeze(1) - _MODES_10D.unsqueeze(0)
    log_components = -0.5 * torch.sum(diffs**2, dim=-1)  # (N, 4)
    energy = -torch.logsumexp(log_components, dim=-1)  # (N,)
    return energy.squeeze()


# Convenience alias matching the legacy name
gaussian_mixture_10d = energy_fn
