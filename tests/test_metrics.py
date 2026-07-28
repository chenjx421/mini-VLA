from __future__ import annotations

import pytest

from embodied_vla.metrics import wilson_score_interval


def test_wilson_interval_contains_observed_proportion() -> None:
    low, high = wilson_score_interval(15, 20)
    assert low < 0.75 < high
    assert low == pytest.approx(0.531299, abs=1e-6)
    assert high == pytest.approx(0.888139, abs=1e-6)


def test_wilson_interval_validates_counts() -> None:
    with pytest.raises(ValueError):
        wilson_score_interval(2, 1)
    with pytest.raises(ValueError):
        wilson_score_interval(0, 0)
