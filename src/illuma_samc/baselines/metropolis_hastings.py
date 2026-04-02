"""Standard Metropolis-Hastings sampler.

Supports multi-chain mode: pass ``n_chains > 1`` to run independent
chains in parallel.  Each chain runs the same MH loop with independent
RNG streams; results are aggregated across chains.
"""

from __future__ import annotations

import math

import torch


def _run_single_mh(
    energy_fn,
    dim: int,
    n_iters: int,
    proposal_std: float,
    temperature: float,
    x0: torch.Tensor | None,
    burn_in: int,
    save_every: int,
) -> dict:
    """Run a single MH chain. Internal helper."""
    x = x0.clone() if x0 is not None else torch.randn(dim)
    result = energy_fn(x)
    if isinstance(result, tuple):
        fx, _ = result
        fx = fx.item()
    else:
        fx = result.item()

    best_x, best_e = x.clone(), fx
    accept_count = 0
    energies = []
    samples = []

    for it in range(1, n_iters + 1):
        y = x + proposal_std * torch.randn(dim)
        result = energy_fn(y)
        if isinstance(result, tuple):
            fy, in_r = result
            fy_val = fy.item()
            if isinstance(in_r, torch.Tensor):
                in_r = in_r.item()
        else:
            fy_val = result.item()
            in_r = True

        log_r = (-fy_val + fx) / temperature

        if not in_r:
            accept = False
        elif log_r > 0:
            accept = True
        else:
            accept = torch.rand(1).item() < math.exp(log_r)

        if accept:
            x = y.clone()
            fx = fy_val
            accept_count += 1
        if fx < best_e:
            best_e = fx
            best_x = x.clone()
        energies.append(fx)

        if it > burn_in and it % save_every == 0:
            samples.append(x.clone())

    return {
        "best_energy": best_e,
        "best_x": best_x,
        "acceptance_rate": accept_count / n_iters,
        "energies": torch.tensor(energies),
        "samples": torch.stack(samples) if samples else torch.empty(0, dim),
    }


def run_mh(
    energy_fn,
    dim: int,
    n_iters: int,
    proposal_std: float = 0.25,
    temperature: float = 1.0,
    x0: torch.Tensor | None = None,
    burn_in: int = 0,
    save_every: int = 1,
    n_chains: int = 1,
) -> dict:
    """Run standard MH. Returns dict of metrics + samples.

    Parameters
    ----------
    n_chains : int
        Number of independent chains. Default 1 (single chain).
        When > 1, runs independent chains and reports the best result.
        ``energies`` uses the best chain's trace; ``samples`` from the
        best chain; ``acceptance_rate`` averaged across chains.
    """
    if n_chains <= 1:
        return _run_single_mh(
            energy_fn, dim, n_iters, proposal_std, temperature, x0, burn_in, save_every
        )

    # Multi-chain: run independent chains, aggregate
    chains = []
    for c in range(n_chains):
        chain_result = _run_single_mh(
            energy_fn, dim, n_iters, proposal_std, temperature, None, burn_in, save_every
        )
        chains.append(chain_result)

    # Best chain by energy
    best_idx = min(range(n_chains), key=lambda i: chains[i]["best_energy"])
    best_chain = chains[best_idx]
    avg_acc = sum(c["acceptance_rate"] for c in chains) / n_chains

    return {
        "best_energy": best_chain["best_energy"],
        "best_x": best_chain["best_x"],
        "acceptance_rate": avg_acc,
        "energies": best_chain["energies"],
        "samples": best_chain["samples"],
        "n_chains": n_chains,
    }
