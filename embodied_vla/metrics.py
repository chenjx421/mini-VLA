from __future__ import annotations

import math


def wilson_score_interval(
    successes: int,
    trials: int,
    *,
    z_value: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a Bernoulli proportion."""

    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie between zero and trials")
    if z_value <= 0.0:
        raise ValueError("z_value must be positive")
    proportion = successes / trials
    z_squared = z_value**2
    denominator = 1.0 + z_squared / trials
    center = (proportion + z_squared / (2.0 * trials)) / denominator
    radius = (
        z_value
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z_squared / (4.0 * trials**2)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)
