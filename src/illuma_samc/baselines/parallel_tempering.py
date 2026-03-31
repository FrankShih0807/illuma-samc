"""Parallel Tempering sampler."""

from __future__ import annotations

import math

import torch


def run_parallel_tempering(
    energy_fn,
    dim: int,
    n_iters: int,
    n_replicas: int = 8,
    proposal_std: float = 0.25,
    t_min: float = 1.0,
    t_max: float = 10.0,
    swap_interval: int = 10,
    burn_in: int = 0,
    save_every: int = 1,
) -> dict:
    """Parallel tempering with geometric temperature ladder.

    Coldest replica runs at t_min (default 1.0 for fair comparison with MH).
    """
    temps = torch.logspace(math.log10(t_min), math.log10(t_max), n_replicas)

    # Initialize replicas
    states = [torch.zeros(dim) for _ in range(n_replicas)]
    energies_list: list[list[float]] = [[] for _ in range(n_replicas)]

    # Compute initial energies
    fxs = []
    for i in range(n_replicas):
        result = energy_fn(states[i])
        if isinstance(result, tuple):
            e, _ = result
            fxs.append(e.item())
        else:
            fxs.append(result.item())

    best_e = min(fxs)
    best_x = states[fxs.index(best_e)].clone()
    accept_counts = [0] * n_replicas
    swap_count = 0
    swap_attempts = 0
    cold_samples = []

    for it in range(1, n_iters + 1):
        # MH step for each replica
        for i in range(n_replicas):
            y = states[i] + proposal_std * torch.randn(dim)
            result = energy_fn(y)
            if isinstance(result, tuple):
                fy, in_r = result
                fy_val = fy.item()
                if isinstance(in_r, torch.Tensor):
                    in_r = in_r.item()
            else:
                fy_val = result.item()
                in_r = True

            log_r = (-fy_val + fxs[i]) / temps[i].item()

            if not in_r:
                accept = False
            elif log_r > 0:
                accept = True
            else:
                accept = torch.rand(1).item() < math.exp(log_r)

            if accept:
                states[i] = y.clone()
                fxs[i] = fy_val
                accept_counts[i] += 1

            if fxs[i] < best_e:
                best_e = fxs[i]
                best_x = states[i].clone()

            energies_list[i].append(fxs[i])

        # Replica swaps
        if it % swap_interval == 0:
            for i in range(n_replicas - 1):
                swap_attempts += 1
                delta = (1.0 / temps[i].item() - 1.0 / temps[i + 1].item()) * (fxs[i + 1] - fxs[i])
                if delta > 0 or torch.rand(1).item() < math.exp(delta):
                    states[i], states[i + 1] = states[i + 1], states[i]
                    fxs[i], fxs[i + 1] = fxs[i + 1], fxs[i]
                    swap_count += 1

        # Collect sample from coldest replica (after burn-in, at save_every)
        if it > burn_in and it % save_every == 0:
            cold_samples.append(states[0].clone())

    # Return coldest replica stats
    return {
        "best_energy": best_e,
        "best_x": best_x,
        "acceptance_rate": accept_counts[0] / n_iters,
        "swap_rate": swap_count / max(swap_attempts, 1),
        "energies": torch.tensor(energies_list[0]),
        "samples": (torch.stack(cold_samples) if cold_samples else torch.empty(0, dim)),
    }
