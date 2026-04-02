"""2D multimodal cost function from sample_code.py."""

import torch

NAME = "2D Multimodal"
DIM = 2


def energy_fn(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch-compatible 2D multimodal cost. Returns (energy, in_region)."""
    if z.dim() == 1:
        z = z.unsqueeze(0)
    x1, x2 = z[:, 0], z[:, 1]
    out_of_bounds = (x1 < -1.1) | (x1 > 1.1) | (x2 < -1.1) | (x2 > 1.1)

    d1 = x1 * torch.sin(20 * x2) + x2 * torch.sin(20 * x1)
    sum1 = d1**2 * torch.cosh(torch.sin(10 * x1) * x1)
    d2 = x1 * torch.cos(10 * x2) - x2 * torch.sin(10 * x1)
    sum2 = d2**2 * torch.cosh(torch.cos(20 * x2) * x2)

    energy = -sum1 - sum2
    energy = torch.where(out_of_bounds, torch.tensor(1e100), energy)
    in_region = ~out_of_bounds
    return energy.squeeze(), in_region.squeeze()


# Convenience alias matching the legacy name
cost_2d = energy_fn
