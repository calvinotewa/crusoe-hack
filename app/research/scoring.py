"""Loss-based scoring: gain over the starting point, noise threshold, fitness."""

from __future__ import annotations

import statistics

NOISE_FLOOR_FRACTION = 0.005


def gain(l0: float, l_best: float) -> float:
    return (l0 - l_best) / l0


def noise_threshold(l0: float, seed_losses: list[float]) -> float:
    """Larger of 2 x seed spread and 0.5% of L0, from the starting train.py on several seeds."""
    spread = statistics.stdev(seed_losses) if len(seed_losses) > 1 else 0.0
    return max(2 * spread, NOISE_FLOOR_FRACTION * l0)


def beats(candidate: float, best: float, threshold: float) -> bool:
    return candidate < best - threshold


def fitness(gains_by_seed: list[float]) -> float:
    return statistics.fmean(gains_by_seed)
