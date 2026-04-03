"""SAMCConfig: reduce partition + gain + proposal boilerplate.

Usage::

    from illuma_samc import SAMCConfig, SAMCWeights

    # From a YAML file
    cfg = SAMCConfig.from_yaml("configs/samc.yaml", model="2d")
    wm = cfg.build()

    # Or programmatically
    cfg = SAMCConfig(n_bins=40, e_min=0, e_max=10, gain="1/t", gain_t0=1000)
    wm = cfg.build()

    # Or build a full SAMC sampler
    sampler = cfg.build_sampler(energy_fn=my_energy, dim=2)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import yaml

from illuma_samc.gain import GainSequence
from illuma_samc.partitions import UniformPartition
from illuma_samc.weight_manager import SAMCWeights


@dataclass
class SAMCConfig:
    """Configuration for SAMC weight manager and sampler.

    Parameters
    ----------
    n_bins : int
        Number of energy bins.
    e_min : float or None
        Lower energy bound. If None, uses auto-expanding bins.
    e_max : float or None
        Upper energy bound. If None, uses auto-expanding bins.
    gain : str
        Gain schedule name: ``"1/t"``, ``"ramp"``, or ``"log"``.
    gain_t0 : int
        Gain schedule ``t0`` parameter.
    gain_kwargs : dict
        Additional gain kwargs (rho, tau, warmup, step_scale).
    proposal_std : float
        Gaussian proposal step size.
    adapt_proposal : bool
        Enable dual-averaging step size adaptation.
    adapt_warmup : int
        Number of steps for adaptation warmup.
    target_accept_rate : float
        Target acceptance rate for adaptation.
    n_chains : int
        Number of parallel chains (for ``build_sampler``).
    temperature : float
        Boltzmann temperature.
    n_iters : int
        Number of MCMC iterations.
    overflow_bins : bool
        Add overflow bins at partition edges.
    device : str
        Torch device.
    dtype : str
        Torch dtype for sample tensors (e.g. ``"float32"``, ``"float64"``).
    """

    n_bins: int = 42
    e_min: float | None = None
    e_max: float | None = None
    gain: str = "ramp"
    gain_t0: int = 1000
    gain_kwargs: dict = field(
        default_factory=lambda: {
            "rho": 1.0,
            "tau": 1.0,
            "warmup": 1,
            "step_scale": 1000,
        }
    )
    proposal_std: float = 0.25
    adapt_proposal: bool = False
    adapt_warmup: int = 1000
    target_accept_rate: float = 0.35
    n_chains: int = 1
    temperature: float = 1.0
    n_iters: int = 100_000
    overflow_bins: bool = False
    device: str = "cpu"
    dtype: str = "float32"

    @classmethod
    def from_yaml(cls, path: str | Path, model: str | None = None) -> "SAMCConfig":
        """Load config from a YAML file.

        Parameters
        ----------
        path : str or Path
            Path to YAML config file.
        model : str, optional
            If the YAML has model-keyed sections (e.g., ``2d:``, ``10d:``),
            select this model's config. If None, uses the top-level keys.
        """
        with open(path) as f:
            raw = yaml.safe_load(f)

        if model is not None:
            if model not in raw:
                raise KeyError(
                    f"Model '{model}' not found in {path}. Available: {list(raw.keys())}"
                )
            raw = raw[model]

        # Map YAML keys to SAMCConfig fields
        kwargs: dict = {}
        key_map = {
            "n_partitions": "n_bins",
            "n_bins": "n_bins",
            "e_min": "e_min",
            "e_max": "e_max",
            "gain": "gain",
            "proposal_std": "proposal_std",
            "n_iters": "n_iters",
            "temperature": "temperature",
            "n_chains": "n_chains",
            "overflow_bins": "overflow_bins",
            "device": "device",
            "dtype": "dtype",
            "adapt_proposal": "adapt_proposal",
            "adapt_warmup": "adapt_warmup",
            "target_accept_rate": "target_accept_rate",
        }
        for yaml_key, config_key in key_map.items():
            if yaml_key in raw:
                kwargs[config_key] = raw[yaml_key]

        # Handle gain_kwargs
        if "gain_kwargs" in raw:
            kwargs["gain_kwargs"] = raw["gain_kwargs"]
            if "t0" in raw["gain_kwargs"]:
                kwargs["gain_t0"] = raw["gain_kwargs"]["t0"]

        return cls(**kwargs)

    def _build_gain(self) -> GainSequence:
        """Build GainSequence from config."""
        kw = dict(self.gain_kwargs)
        if self.gain in ("1/t", "log") and "t0" not in kw:
            kw["t0"] = self.gain_t0
        return GainSequence(self.gain, **kw)

    def build(self, **overrides) -> SAMCWeights:
        """Build a SAMCWeights instance from this config.

        Parameters
        ----------
        **overrides
            Override any SAMCWeights constructor kwargs.
        """
        gain = self._build_gain()

        if self.e_min is not None and self.e_max is not None:
            partition = UniformPartition(
                self.e_min,
                self.e_max,
                self.n_bins,
                overflow_bins=self.overflow_bins,
                device=self.device,
            )
            return SAMCWeights(
                partition=partition,
                gain=gain,
                device=self.device,
                dtype=self.dtype,
                **overrides,
            )
        else:
            # Auto-expanding bins
            return SAMCWeights(
                gain=gain,
                device=self.device,
                dtype=self.dtype,
                **overrides,
            )

    def build_sampler(
        self,
        energy_fn: Callable[[torch.Tensor], torch.Tensor],
        dim: int,
        **overrides,
    ):
        """Build a full SAMC sampler from this config.

        Parameters
        ----------
        energy_fn : callable
            Energy function.
        dim : int
            Sample space dimensionality.
        **overrides
            Override any SAMC constructor kwargs.
        """
        from illuma_samc.sampler import SAMC

        kwargs: dict = {
            "energy_fn": energy_fn,
            "dim": dim,
            "n_partitions": self.n_bins,
            "proposal_std": self.proposal_std,
            "adapt_proposal": self.adapt_proposal,
            "adapt_warmup": self.adapt_warmup,
            "target_accept_rate": self.target_accept_rate,
            "temperature": self.temperature,
            "gain": self._build_gain(),
            "device": self.device,
            "dtype": self.dtype,
            "n_chains": self.n_chains,
        }
        if self.e_min is not None and self.e_max is not None:
            kwargs["e_min"] = self.e_min
            kwargs["e_max"] = self.e_max

        kwargs.update(overrides)
        return SAMC(**kwargs)
