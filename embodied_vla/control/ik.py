from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def damped_least_squares(
    jacobian: NDArray[np.floating],
    displacement: NDArray[np.floating],
    *,
    damping: float = 0.05,
) -> NDArray[np.float64]:
    """Map a Cartesian displacement to joint displacement with damped IK.

    The right-damped pseudoinverse

        dq = J^T (J J^T + lambda^2 I)^-1 dx

    remains finite close to kinematic singularities. It also avoids explicitly
    forming a matrix inverse by solving the linear system.
    """

    jacobian = np.asarray(jacobian, dtype=np.float64)
    displacement = np.asarray(displacement, dtype=np.float64)
    if jacobian.ndim != 2:
        raise ValueError(f"jacobian must be rank 2, got shape {jacobian.shape}")
    if displacement.shape != (jacobian.shape[0],):
        raise ValueError(
            "displacement dimension must match the Jacobian task dimension: "
            f"{displacement.shape} vs {jacobian.shape}"
        )
    if damping <= 0:
        raise ValueError("damping must be positive")

    task_matrix = jacobian @ jacobian.T
    regularizer = (damping**2) * np.eye(task_matrix.shape[0], dtype=np.float64)
    return jacobian.T @ np.linalg.solve(task_matrix + regularizer, displacement)
