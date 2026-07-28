from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from embodied_vla.grounding_calibration import (
    CALIBRATION_SCHEMA_VERSION,
    AffinePixelGroundingCalibration,
)


def test_affine_pixel_calibration_applies_role_specific_transforms() -> None:
    calibration = AffinePixelGroundingCalibration(
        {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "transforms": {
                "target": {
                    "matrix": [
                        [0.1, -0.1],
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ]
                },
                "goal": {
                    "matrix": [
                        [0.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ]
                },
            },
        }
    )

    corrected = calibration.correct(
        np.array([[0.25, 0.75], [0.4, 0.6]], dtype=np.float32)
    )

    np.testing.assert_allclose(corrected[0], [0.35, 0.65])
    np.testing.assert_allclose(corrected[1], [0.4, 0.6])


def test_language_conditioned_calibration_uses_instruction_features() -> None:
    calibration = AffinePixelGroundingCalibration(
        {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "transforms": {
                "target": {
                    "features": ["bias", "u", "v", "is_blue", "is_right"],
                    "matrix": [
                        [0.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [0.1, 0.0],
                        [0.0, -0.1],
                    ],
                },
                "goal": {
                    "matrix": [
                        [0.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ]
                },
            },
        }
    )

    corrected = calibration.correct(
        np.array([[0.25, 0.75], [0.4, 0.6]], dtype=np.float32),
        target_color="blue",
        goal_side="right",
    )

    np.testing.assert_allclose(corrected[0], [0.35, 0.65])


def test_calibration_rejects_a_different_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"selected checkpoint")
    calibration = AffinePixelGroundingCalibration(
        {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "checkpoint_sha256": hashlib.sha256(b"selected checkpoint").hexdigest(),
            "transforms": {
                role: {
                    "matrix": [
                        [0.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0],
                    ]
                }
                for role in ("target", "goal")
            },
        }
    )

    calibration.verify_checkpoint(checkpoint_path)
    checkpoint_path.write_bytes(b"different checkpoint")

    with pytest.raises(ValueError, match="checkpoint mismatch"):
        calibration.verify_checkpoint(checkpoint_path)
