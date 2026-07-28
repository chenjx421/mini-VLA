from __future__ import annotations

import unittest

import numpy as np

from embodied_vla.control import damped_least_squares


class DampedLeastSquaresTest(unittest.TestCase):
    def test_identity_jacobian_tracks_displacement(self) -> None:
        displacement = np.array([0.2, -0.1, 0.05])
        joint_delta = damped_least_squares(
            np.eye(3),
            displacement,
            damping=1e-4,
        )
        np.testing.assert_allclose(joint_delta, displacement, atol=1e-6)

    def test_singular_jacobian_stays_finite(self) -> None:
        jacobian = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        result = damped_least_squares(
            jacobian,
            np.array([1.0, 1.0, 1.0]),
            damping=0.05,
        )
        self.assertTrue(np.isfinite(result).all())

    def test_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            damped_least_squares(np.eye(3), np.ones(2))


if __name__ == "__main__":
    unittest.main()
