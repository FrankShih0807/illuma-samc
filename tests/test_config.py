"""Tests for SAMCConfig."""

import tempfile
from pathlib import Path

import torch
import yaml

from illuma_samc.config import SAMCConfig


def _quadratic(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return 0.5 * torch.sum(x**2)
    return 0.5 * torch.sum(x**2, dim=-1)


class TestSAMCConfig:
    def test_defaults(self):
        cfg = SAMCConfig()
        assert cfg.n_bins == 42
        assert cfg.gain == "ramp"
        assert cfg.e_min is None

    def test_build_weights_auto(self):
        """Build SAMCWeights with auto-expanding bins."""
        cfg = SAMCConfig()
        wm = cfg.build()
        assert wm.n_bins == 0  # deferred until first energy

    def test_build_weights_explicit_range(self):
        cfg = SAMCConfig(n_bins=20, e_min=0.0, e_max=10.0)
        wm = cfg.build()
        assert wm.n_bins == 20

    def test_build_sampler(self):
        cfg = SAMCConfig(n_bins=10, e_min=0.0, e_max=5.0, proposal_std=0.5)
        sampler = cfg.build_sampler(energy_fn=_quadratic, dim=2)
        result = sampler.run(n_steps=100, progress=False)
        assert result.samples.shape[-1] == 2

    def test_build_sampler_with_adapt(self):
        cfg = SAMCConfig(
            n_bins=10,
            e_min=0.0,
            e_max=5.0,
            adapt_proposal=True,
            adapt_warmup=50,
        )
        sampler = cfg.build_sampler(energy_fn=_quadratic, dim=2)
        assert sampler._proposal._adapt is True

    def test_from_yaml(self):
        data = {
            "2d": {
                "n_partitions": 30,
                "e_min": -8.0,
                "e_max": 0.0,
                "proposal_std": 0.1,
                "gain": "1/t",
                "gain_kwargs": {"t0": 500},
                "n_iters": 100000,
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            path = f.name

        cfg = SAMCConfig.from_yaml(path, model="2d")
        assert cfg.n_bins == 30
        assert cfg.e_min == -8.0
        assert cfg.gain == "1/t"
        assert cfg.gain_t0 == 500
        assert cfg.proposal_std == 0.1

        Path(path).unlink()

    def test_from_yaml_missing_model(self):
        data = {"2d": {"n_partitions": 10}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            path = f.name

        import pytest

        with pytest.raises(KeyError, match="10d"):
            SAMCConfig.from_yaml(path, model="10d")

        Path(path).unlink()

    def test_overrides(self):
        cfg = SAMCConfig(n_bins=20, e_min=0.0, e_max=10.0)
        wm = cfg.build(record_every=50)
        assert wm._record_every == 50


class TestDtype:
    """Tests for dtype propagation through SAMCConfig."""

    def test_build_weights_dtype_float64(self):
        """SAMCConfig(dtype='float64').build() produces SAMCWeights with float64 dtype."""
        cfg = SAMCConfig(dtype="float64")
        wm = cfg.build()
        assert wm._dtype == torch.float64

    def test_build_sampler_dtype_float64(self):
        """SAMCConfig(dtype='float64').build_sampler() produces SAMC with float64 dtype."""
        cfg = SAMCConfig(dtype="float64")
        sampler = cfg.build_sampler(energy_fn=_quadratic, dim=2)
        assert sampler._dtype == torch.float64

    def test_from_yaml_dtype(self):
        """SAMCConfig.from_yaml() parses dtype field."""
        import tempfile
        from pathlib import Path

        import yaml

        data = {"model": {"n_partitions": 10, "e_min": 0.0, "e_max": 10.0, "dtype": "float64"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            path = f.name

        cfg = SAMCConfig.from_yaml(path, model="model")
        assert cfg.dtype == "float64"
        Path(path).unlink()
