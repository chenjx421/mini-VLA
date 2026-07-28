from __future__ import annotations

import re

import mujoco
import numpy as np
from geometry_msgs.msg import Quaternion
from numpy.typing import NDArray


def frame_name(name: str | None, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_/]", "_", name or fallback)
    return cleaned.strip("/") or fallback


def quaternion_from_matrix(rotation: NDArray[np.floating]) -> Quaternion:
    quaternion_wxyz = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion_wxyz, np.asarray(rotation).reshape(-1))
    message = Quaternion()
    message.w = float(quaternion_wxyz[0])
    message.x = float(quaternion_wxyz[1])
    message.y = float(quaternion_wxyz[2])
    message.z = float(quaternion_wxyz[3])
    return message


def planar_quaternion(yaw: float) -> Quaternion:
    message = Quaternion()
    message.z = float(np.sin(yaw / 2.0))
    message.w = float(np.cos(yaw / 2.0))
    return message
