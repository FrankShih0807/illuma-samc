"""Benchmark problem registry."""

from illuma_samc.problems.gaussian_10d import energy_fn as _energy_10d
from illuma_samc.problems.gaussian_10d import gaussian_mixture_10d
from illuma_samc.problems.multimodal_2d import cost_2d
from illuma_samc.problems.multimodal_2d import energy_fn as _energy_2d

PROBLEMS = {
    "2d": {"energy_fn": _energy_2d, "dim": 2, "name": "2D Multimodal"},
    "10d": {"energy_fn": _energy_10d, "dim": 10, "name": "10D Gaussian Mixture"},
}

__all__ = ["PROBLEMS", "cost_2d", "gaussian_mixture_10d"]
