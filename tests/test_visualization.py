from __future__ import annotations

import numpy as np
import pytest

from embodied_vla.image_utils import depth_to_rgb


def test_depth_to_rgb_preserves_shape_and_depth_order() -> None:
    depth = np.linspace(0.5, 3.0, 100, dtype=np.float32).reshape(10, 10)

    rgb = depth_to_rgb(depth, lower_percentile=0.0, upper_percentile=100.0)

    assert rgb.shape == (10, 10, 3)
    assert rgb.dtype == np.uint8
    assert rgb[0, 0, 0] > rgb[-1, -1, 0]
    assert rgb[0, 0, 2] < rgb[-1, -1, 2]


def test_depth_to_rgb_rejects_no_valid_depth() -> None:
    with pytest.raises(ValueError, match="finite positive"):
        depth_to_rgb(np.full((4, 4), np.nan, dtype=np.float32))
