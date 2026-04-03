"""Benchmark problem registry."""

from illuma_samc.problems.gaussian_10d import energy_fn as _energy_10d
from illuma_samc.problems.gaussian_10d import gaussian_mixture_10d
from illuma_samc.problems.gaussian_50d import energy_fn as _energy_50d
from illuma_samc.problems.gaussian_100d import energy_fn as _energy_100d
from illuma_samc.problems.multimodal_2d import cost_2d
from illuma_samc.problems.multimodal_2d import energy_fn as _energy_2d
from illuma_samc.problems.rastrigin_20d import energy_fn as _energy_rastrigin
from illuma_samc.problems.rosenbrock_2d import energy_fn as _energy_rosenbrock

PROBLEMS = {
    "2d": {"energy_fn": _energy_2d, "dim": 2, "name": "2D Multimodal"},
    "10d": {"energy_fn": _energy_10d, "dim": 10, "name": "10D Gaussian Mixture"},
    "50d": {"energy_fn": _energy_50d, "dim": 50, "name": "50D Gaussian Mixture"},
    "100d": {"energy_fn": _energy_100d, "dim": 100, "name": "100D Gaussian Mixture"},
    "rosenbrock": {"energy_fn": _energy_rosenbrock, "dim": 2, "name": "Rosenbrock 2D"},
    "rastrigin": {"energy_fn": _energy_rastrigin, "dim": 20, "name": "Rastrigin 20D"},
}

__all__ = [
    "PROBLEMS",
    "cost_2d",
    "gaussian_mixture_10d",
]
