# Project: illuma-samc

> Inherits workspace rules from the root `CLAUDE.md`. This file defines project-specific context.

## Purpose
Production-quality PyTorch implementation of Stochastic Approximation Monte Carlo (SAMC). Designed for commercial use — easy API, GPU support, extensible.

## Setup & Run
```bash
conda activate illuma-samc
pip install -e ".[dev]"

# Format & lint
ruff format .
ruff check .

# Test
pytest

# Run example
python examples/gaussian_mixture.py
```

## Tech Stack
- **Language:** Python 3.11+
- **Framework:** PyTorch (tensors, autograd, GPU)
- **Testing:** pytest
- **Packaging:** src layout, pip-installable

## Conventions
- All tensors use `torch.Tensor`, not numpy
- Energy functions: callable `Tensor → Tensor` (scalar output)
- Device handling: accept `device` param, default to CPU
- Type hints on all public APIs
- No unnecessary dependencies — torch only for core

## Current Status
- **Phase:** 1 — Core MVP
- **Waiting on:** Frank's reference SAMC script for algorithm implementation
