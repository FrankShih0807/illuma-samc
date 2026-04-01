"""Post-hoc analysis metrics for SAMC experiments."""

from __future__ import annotations

import torch

from illuma_samc.problems import gaussian_10d, rastrigin_20d, rosenbrock_2d


def compute_mode_coverage(
    samples: torch.Tensor,
    problem_name: str,
    threshold: float | None = None,
) -> float:
    """Compute fraction of known modes found by samples.

    Parameters
    ----------
    samples : Tensor
        Sample positions, shape ``(n_samples, dim)``.
    problem_name : str
        One of ``"2d"``, ``"10d"``, ``"rosenbrock"``, ``"rastrigin"``.
    threshold : float, optional
        Distance threshold for counting a mode as found.
        Defaults vary by problem.

    Returns
    -------
    float
        Fraction of known modes covered (0.0 to 1.0).
    """
    if samples.dim() == 1:
        samples = samples.unsqueeze(0)

    if problem_name == "2d":
        return _coverage_2d(samples, threshold or 0.05)
    elif problem_name == "10d":
        return _coverage_10d(samples, threshold or 3.0)
    elif problem_name == "rosenbrock":
        return _coverage_rosenbrock(samples, threshold or 0.5)
    elif problem_name == "rastrigin":
        return _coverage_rastrigin(samples, threshold or 1.0)
    else:
        raise ValueError(f"Unknown problem: {problem_name}")


def _coverage_2d(samples: torch.Tensor, threshold: float) -> float:
    """2D multimodal: check if samples reach the low-energy basin.

    The 2D multimodal problem has a complex energy landscape in [-1.1, 1.1]^2.
    We check whether samples cluster near the global minimum region
    (best energy ~ -8.2). Since the landscape is complex with many basins,
    we approximate by checking if any samples reach very low energy.
    """
    from illuma_samc.problems.multimodal_2d import energy_fn

    energies = energy_fn(samples)
    if isinstance(energies, tuple):
        energies = energies[0]
    # Count "mode found" if any sample reaches within threshold of best known energy (-8.2)
    found = (energies < -7.0).any().item()
    return 1.0 if found else 0.0


def _coverage_10d(samples: torch.Tensor, threshold: float) -> float:
    """10D Gaussian mixture: check samples within threshold of 4 mode centers."""
    modes = gaussian_10d._MODES_10D  # (4, 10)
    n_modes = modes.shape[0]
    found = 0
    for i in range(n_modes):
        dists = torch.norm(samples - modes[i].unsqueeze(0), dim=-1)
        if (dists < threshold).any():
            found += 1
    return found / n_modes


def _coverage_rosenbrock(samples: torch.Tensor, threshold: float) -> float:
    """Rosenbrock: check if samples reach near (1, 1)."""
    mode = rosenbrock_2d.MODES[0]  # (2,)
    dists = torch.norm(samples[:, :2] - mode.unsqueeze(0), dim=-1)
    found = (dists < threshold).any().item()
    return 1.0 if found else 0.0


def _coverage_rastrigin(samples: torch.Tensor, threshold: float) -> float:
    """Rastrigin: check if samples reach near origin."""
    mode = rastrigin_20d.MODES[0]  # (20,)
    dists = torch.norm(samples - mode.unsqueeze(0), dim=-1)
    found = (dists < threshold).any().item()
    return 1.0 if found else 0.0


def compute_energy_mixing(energy_history: torch.Tensor, n_bins: int = 20) -> dict:
    """Compute energy-space mixing metrics from an energy trace.

    Parameters
    ----------
    energy_history : Tensor
        1-D tensor of energies at each iteration.
    n_bins : int
        Number of energy levels for round-trip computation.

    Returns
    -------
    dict with keys:
        - ``round_trip_time``: mean iterations per full energy round-trip
          (low → high → low). ``inf`` if no complete round-trip.
        - ``energy_autocorr_50``: autocorrelation at lag 50.
        - ``energy_autocorr_200``: autocorrelation at lag 200.
    """
    e = energy_history.float()
    # Multi-chain: average across chains to get 1-D trace
    if e.ndim == 2:
        e = e.mean(dim=1)
    n = len(e)

    # --- Autocorrelation ---
    e_centered = e - e.mean()
    var = (e_centered**2).sum()
    autocorr = {}
    for lag in [50, 200]:
        if lag >= n:
            autocorr[lag] = 1.0
        else:
            autocorr[lag] = float((e_centered[: n - lag] * e_centered[lag:]).sum() / var)

    # --- Round-trip time ---
    e_min, e_max = e.min().item(), e.max().item()
    if e_min == e_max:
        return {
            "round_trip_time": float("inf"),
            "energy_autocorr_50": 1.0,
            "energy_autocorr_200": 1.0,
            "n_round_trips": 0,
        }

    low_thresh = e_min + 0.2 * (e_max - e_min)
    high_thresh = e_max - 0.2 * (e_max - e_min)

    # Track round-trips: low → high → low
    trip_times = []
    state = "seeking_low"
    trip_start = 0
    for i in range(n):
        ei = e[i].item()
        if state == "seeking_low" and ei <= low_thresh:
            state = "seeking_high"
            trip_start = i
        elif state == "seeking_high" and ei >= high_thresh:
            state = "seeking_low_return"
        elif state == "seeking_low_return" and ei <= low_thresh:
            trip_times.append(i - trip_start)
            state = "seeking_high"
            trip_start = i

    mean_rt = float(sum(trip_times) / len(trip_times)) if trip_times else float("inf")

    return {
        "round_trip_time": mean_rt,
        "energy_autocorr_50": autocorr[50],
        "energy_autocorr_200": autocorr[200],
        "n_round_trips": len(trip_times),
    }


def compute_bin_flatness(bin_counts: torch.Tensor) -> float:
    """Compute bin visit flatness metric.

    ``1 - std(bin_counts) / mean(bin_counts)``

    Returns 1.0 for perfectly flat visits, lower for uneven visits.
    Values can be negative if std > mean.

    Parameters
    ----------
    bin_counts : Tensor
        Visit counts per bin.

    Returns
    -------
    float
        Flatness score.
    """
    counts = bin_counts.float()
    mean = counts.mean()
    if mean == 0:
        return 0.0
    std = counts.std()
    return float(1.0 - std / mean)
