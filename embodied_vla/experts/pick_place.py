from __future__ import annotations

from enum import IntEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from embodied_vla.envs.config import SOArmEnvConfig

APPROACH_HEIGHT_OFFSET = 0.085


class ExpertPhase(IntEnum):
    APPROACH = 0
    DESCEND_GRASP = 1
    CLOSE_GRIPPER = 2
    LIFT = 3
    TRANSPORT = 4
    DESCEND_RELEASE = 5
    OPEN_GRIPPER = 6
    RETREAT = 7
    DONE = 8


class PickPlaceExpert:
    """Privileged waypoint expert used only to generate demonstrations.

    It reads object and goal positions from ``info``. A learned policy never
    receives those privileged coordinates in multimodal mode.
    """

    def __init__(self, config: SOArmEnvConfig) -> None:
        self.config = config
        self.phase = ExpertPhase.APPROACH
        self.phase_steps = 0
        self.retries = 0

    def reset(self) -> None:
        self.phase = ExpertPhase.APPROACH
        self.phase_steps = 0
        self.retries = 0

    def act(self, info: dict[str, Any]) -> NDArray[np.float32]:
        end_effector = np.asarray(info["end_effector_position"], dtype=np.float64)
        target = np.asarray(info["target_position"], dtype=np.float64)
        goal = np.asarray(info["goal_position"], dtype=np.float64)
        self.phase_steps += 1

        jaw = self._jaw_action(self.config.pregrasp_jaw)
        waypoint = end_effector.copy()

        if self.phase == ExpertPhase.APPROACH:
            waypoint = target + np.array([0.0, 0.0, APPROACH_HEIGHT_OFFSET])
            if self._close_xyz(end_effector, waypoint, xy=0.006, z=0.008):
                self._transition(ExpertPhase.DESCEND_GRASP)

        elif self.phase == ExpertPhase.DESCEND_GRASP:
            waypoint = target + np.array([0.0, 0.0, 0.004])
            centered_between_fingers = (
                bool(info["bilateral_contact"])
                or (
                    np.linalg.norm(end_effector[:2] - target[:2]) < 0.012
                    and 0.012 < end_effector[2] - target[2] < 0.032
                )
            )
            if centered_between_fingers:
                self._transition(ExpertPhase.CLOSE_GRIPPER)

        elif self.phase == ExpertPhase.CLOSE_GRIPPER:
            jaw = -1.0
            if bool(info["bilateral_contact"]) and self.phase_steps >= 5:
                self._transition(ExpertPhase.LIFT)
            elif self.phase_steps >= 35:
                self.retries += 1
                self._transition(ExpertPhase.APPROACH)

        elif self.phase == ExpertPhase.LIFT:
            jaw = -1.0
            waypoint = np.array([target[0], target[1], 0.145])
            if self._lost_object(info, target):
                self._retry()
            elif target[2] > 0.105:
                self._transition(ExpertPhase.TRANSPORT)

        elif self.phase == ExpertPhase.TRANSPORT:
            jaw = -1.0
            waypoint = np.array([goal[0], goal[1], 0.145])
            if self._lost_object(info, target):
                self._retry()
            elif np.linalg.norm(target[:2] - goal[:2]) < 0.025:
                self._transition(ExpertPhase.DESCEND_RELEASE)

        elif self.phase == ExpertPhase.DESCEND_RELEASE:
            jaw = -1.0
            waypoint = np.array([goal[0], goal[1], 0.058])
            if target[2] < 0.070 and np.linalg.norm(target[:2] - goal[:2]) < 0.030:
                self._transition(ExpertPhase.OPEN_GRIPPER)

        elif self.phase == ExpertPhase.OPEN_GRIPPER:
            jaw = 1.0
            if self.phase_steps >= 10:
                self._transition(ExpertPhase.RETREAT)

        elif self.phase == ExpertPhase.RETREAT:
            jaw = 1.0
            waypoint = np.array([goal[0], goal[1], 0.135])
            if end_effector[2] > 0.115 and not bool(info["bilateral_contact"]):
                self._transition(ExpertPhase.DONE)

        action = np.zeros(5, dtype=np.float32)
        if self.phase != ExpertPhase.DONE:
            delta = waypoint - end_effector
            action[:3] = np.clip(
                0.6 * delta / self.config.cartesian_action_scale,
                -1.0,
                1.0,
            )
        action[4] = jaw
        return action

    def _transition(self, next_phase: ExpertPhase) -> None:
        self.phase = next_phase
        self.phase_steps = 0

    def _retry(self) -> None:
        self.retries += 1
        self._transition(ExpertPhase.APPROACH)

    def _lost_object(self, info: dict[str, Any], target: NDArray[np.float64]) -> bool:
        contact_force = float(np.asarray(info["contact_force"]).sum())
        return bool(
            self.phase_steps > 8
            and target[2] < self.config.table_height + 0.045
            and contact_force < 0.05
        )

    def _jaw_action(self, desired_jaw: float) -> float:
        alpha = (desired_jaw - self.config.closed_jaw) / (
            self.config.open_jaw - self.config.closed_jaw
        )
        return float(np.clip(2.0 * alpha - 1.0, -1.0, 1.0))

    @staticmethod
    def _close_xyz(
        current: NDArray[np.float64],
        desired: NDArray[np.float64],
        *,
        xy: float,
        z: float,
    ) -> bool:
        return bool(
            np.linalg.norm(current[:2] - desired[:2]) < xy
            and abs(float(current[2] - desired[2])) < z
        )
