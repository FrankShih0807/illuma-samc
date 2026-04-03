"""illuma-samc: Production-quality SAMC for PyTorch."""

__version__ = "0.3.0"

# Subpackage re-exports for convenience
from illuma_samc import analysis, baselines, problems  # noqa: F401
from illuma_samc.config import SAMCConfig
from illuma_samc.diagnostics import plot_diagnostics, plot_weight_diagnostics
from illuma_samc.gain import GainSequence
from illuma_samc.partitions import (
    ExpandablePartition,
    Partition,
    UniformPartition,
)
from illuma_samc.proposals import GaussianProposal, LangevinProposal, Proposal
from illuma_samc.sampler import SAMC, SAMCResult
from illuma_samc.weight_manager import SAMCWeights

__all__ = [
    "SAMC",
    "SAMCConfig",
    "SAMCResult",
    "SAMCWeights",
    "GainSequence",
    "Partition",
    "UniformPartition",
    "ExpandablePartition",
    "Proposal",
    "GaussianProposal",
    "LangevinProposal",
    "plot_diagnostics",
    "plot_weight_diagnostics",
    "analysis",
    "baselines",
    "problems",
]
