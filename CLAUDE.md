# Project: illuma-samc

> Inherits workspace rules from the root `CLAUDE.md`. This file defines project-specific context.

## Your Role
Read `../AGENT_PLAYBOOK.md` for the full agent hierarchy. To determine your level:
- **If Frank is talking to you interactively** → you are **L2 Project Lead**. You own this project's direction, make decisions, update STATUS.md and WORKLOG.md.
- **If you were launched by `watch_todo.sh` or as a subagent** → you are **L3 Worker**. Follow TODO.md steps in order, write WORKLOG.md, create BLOCKED.md if stuck.

Follow the session start/end checklists for your level (AGENT_PLAYBOOK.md Section 5).

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

## Reference Implementation
`reference/sample_code.py` is the ground truth SAMC implementation. Use it as:
- **Algorithm reference** — the core SAMC loop, weight update, gain schedule, and acceptance ratio are all correct
- **Regression test** — the new API should reproduce the same results (best energy, acceptance rate, weight convergence) on the same 2D multimodal cost function
- **Test case** — port the `cost()` function and `run_samc()` parameters into the test suite as a known-good benchmark

## Current Status
- **Phase:** 3 — Generalizing (ablation complete, robust bins complete)
