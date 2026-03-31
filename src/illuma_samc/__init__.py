"""illuma-samc: Production-quality SAMC for PyTorch."""

__version__ = "0.1.0"

# Subpackage re-exports for convenience
from illuma_samc import baselines, problems  # noqa: F401
from illuma_samc.diagnostics import plot_diagnostics
from illuma_samc.gain import GainSequence
from illuma_samc.partitions import AdaptivePartition, QuantilePartition, UniformPartition
from illuma_samc.proposals import GaussianProposal, LangevinProposal
from illuma_samc.sampler import SAMC, SAMCResult

__all__ = [
    "SAMC",
    "SAMCResult",
    "GainSequence",
    "UniformPartition",
    "AdaptivePartition",
    "QuantilePartition",
    "GaussianProposal",
    "LangevinProposal",
    "plot_diagnostics",
    "baselines",
    "problems",
]
