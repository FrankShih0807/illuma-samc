"""illuma-samc: Production-quality SAMC for PyTorch."""

__version__ = "0.1.0"

# Subpackage re-exports for convenience
from illuma_samc import baselines, problems  # noqa: F401
from illuma_samc.diagnostics import plot_diagnostics, plot_weight_diagnostics
from illuma_samc.gain import GainSequence
from illuma_samc.partitions import (
    AdaptivePartition,
    ExpandablePartition,
    QuantilePartition,
    UniformPartition,
)
from illuma_samc.proposals import GaussianProposal, LangevinProposal
from illuma_samc.sampler import SAMC, SAMCResult
from illuma_samc.weight_manager import SAMCWeights

__all__ = [
    "SAMC",
    "SAMCResult",
    "SAMCWeights",
    "GainSequence",
    "UniformPartition",
    "AdaptivePartition",
    "ExpandablePartition",
    "QuantilePartition",
    "GaussianProposal",
    "LangevinProposal",
    "plot_diagnostics",
    "plot_weight_diagnostics",
    "baselines",
    "problems",
]
