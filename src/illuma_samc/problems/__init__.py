"""Benchmark problem registry."""

from illuma_samc.problems.gaussian_10d import energy_fn as _energy_10d
from illuma_samc.problems.gaussian_10d import gaussian_mixture_10d
from illuma_samc.problems.multimodal_2d import cost_2d
from illuma_samc.problems.multimodal_2d import energy_fn as _energy_2d
from illuma_samc.problems.rastrigin_20d import energy_fn as _energy_rastrigin
from illuma_samc.problems.rosenbrock_2d import energy_fn as _energy_rosenbrock

PROBLEMS = {
    "2d": {"energy_fn": _energy_2d, "dim": 2, "name": "2D Multimodal"},
    "10d": {"energy_fn": _energy_10d, "dim": 10, "name": "10D Gaussian Mixture"},
    "rosenbrock": {"energy_fn": _energy_rosenbrock, "dim": 2, "name": "Rosenbrock 2D"},
    "rastrigin": {"energy_fn": _energy_rastrigin, "dim": 20, "name": "Rastrigin 20D"},
}

__all__ = [
    "PROBLEMS",
    "cost_2d",
    "gaussian_mixture_10d",
]
