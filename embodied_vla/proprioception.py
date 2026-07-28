from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

JOINT_PROPRIO_DIM = 12
CARTESIAN_PROPRIO_DIM = 15
END_EFFECTOR_STATE_SLICE = slice(12, 15)
TARGET_WORLD_STATE_SLICE = slice(15, 18)
GOAL_WORLD_STATE_SLICE = slice(21, 24)
END_EFFECTOR_POSITION_SCALE = 0.5


def uses_end_effector_position(proprio_dim: int) -> bool:
    if proprio_dim == JOINT_PROPRIO_DIM:
        return False
    if proprio_dim == CARTESIAN_PROPRIO_DIM:
        return True
    raise ValueError(
        "proprio_dim must be 12 (joint position/velocity) or "
        "15 (joint position/velocity plus normalized end-effector XYZ)"
    )


def assemble_model_proprio(
    joint_proprio: NDArray[np.floating],
    *,
    expected_dim: int,
    normalized_end_effector_position: NDArray[np.floating] | None = None,
) -> NDArray[np.float32]:
    joint = np.asarray(joint_proprio, dtype=np.float32)
    if joint.shape != (JOINT_PROPRIO_DIM,):
        raise ValueError(
            f"joint_proprio must have shape ({JOINT_PROPRIO_DIM},), got {joint.shape}"
        )
    if not uses_end_effector_position(expected_dim):
        return joint.copy()
    if normalized_end_effector_position is None:
        raise ValueError("15D proprioception requires an end-effector position")
    end_effector = np.asarray(normalized_end_effector_position, dtype=np.float32)
    if end_effector.shape != (3,):
        raise ValueError(
            "normalized_end_effector_position must have shape (3,), "
            f"got {end_effector.shape}"
        )
    return np.concatenate((joint, end_effector)).astype(np.float32, copy=False)
