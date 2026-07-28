from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def depth_to_rgb(
    depth: NDArray[np.floating],
    *,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> NDArray[np.uint8]:
    """Convert metric depth to a robust near-warm, far-cool RGB preview."""

    values = np.asarray(depth, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("depth must be a 2D array")
    valid = np.isfinite(values) & (values > 0.0)
    if not valid.any():
        raise ValueError("depth must contain at least one finite positive value")
    low, high = np.percentile(
        values[valid],
        (lower_percentile, upper_percentile),
    )
    normalized = np.clip(
        (values - low) / max(float(high - low), 1e-6),
        0.0,
        1.0,
    )
    closeness = 1.0 - normalized
    red = np.clip(2.0 * closeness - 0.5, 0.0, 1.0)
    green = np.clip(1.5 - np.abs(2.0 * closeness - 1.0), 0.0, 1.0)
    blue = np.clip(1.5 - 2.0 * closeness, 0.0, 1.0)
    rgb = np.uint8(np.stack((red, green, blue), axis=-1) * 255.0)
    rgb[~valid] = 0
    return rgb
