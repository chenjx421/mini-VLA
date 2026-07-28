from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from embodied_vla.envs.config import SOArmEnvConfig
from embodied_vla.experts import ExpertPhase
from embodied_vla.experts.pick_place import APPROACH_HEIGHT_OFFSET


class GroundedVisualServoController:
    """Calibrated low-level controller driven by learned target/goal estimates.

    The controller may read robot proprioception and contact sensing, but never
    reads simulator target or goal coordinates. Those positions are supplied by
    the caller after learned pixel grounding and camera unprojection.
    """

    def __init__(
        self,
        config: SOArmEnvConfig,
        *,
        smoothing_alpha: float = 0.35,
        recovery_search_radius_m: float = 0.0,
        close_retry_steps: int = 35,
    ) -> None:
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must lie in (0, 1]")
        if recovery_search_radius_m < 0.0:
            raise ValueError("recovery_search_radius_m cannot be negative")
        if close_retry_steps <= 0:
            raise ValueError("close_retry_steps must be positive")
        self.config = config
        self.smoothing_alpha = smoothing_alpha
        self.recovery_search_radius_m = recovery_search_radius_m
        self.close_retry_steps = close_retry_steps
        self.phase = ExpertPhase.APPROACH
        self.phase_steps = 0
        self.retries = 0
        self._target_estimate: NDArray[np.float64] | None = None
        self._goal_estimate: NDArray[np.float64] | None = None
        self._locked_target: NDArray[np.float64] | None = None

    def reset(self) -> None:
        self.phase = ExpertPhase.APPROACH
        self.phase_steps = 0
        self.retries = 0
        self._target_estimate = None
        self._goal_estimate = None
        self._locked_target = None

    def act(
        self,
        info: dict[str, Any],
        *,
        target_position_estimate: NDArray[np.floating],
        goal_position_estimate: NDArray[np.floating],
    ) -> NDArray[np.float32]:
        end_effector = np.asarray(info["end_effector_position"], dtype=np.float64)
        target_measurement = self._validate_position(
            target_position_estimate,
            "target_position_estimate",
        )
        goal_measurement = self._validate_position(
            goal_position_estimate,
            "goal_position_estimate",
        )
        self._target_estimate = self._smooth(
            self._target_estimate,
            target_measurement,
        )
        self._goal_estimate = self._smooth(
            self._goal_estimate,
            goal_measurement,
        )
        target = (
            self._locked_target
            if self._locked_target is not None
            else self._target_estimate + self.search_offset
        )
        goal = self._goal_estimate
        self.phase_steps += 1

        jaw = self._jaw_action(self.config.pregrasp_jaw)
        waypoint = end_effector.copy()

        if self.phase == ExpertPhase.APPROACH:
            waypoint = target + np.array([0.0, 0.0, APPROACH_HEIGHT_OFFSET])
            if self._close_xyz(end_effector, waypoint, xy=0.008, z=0.010):
                self._locked_target = target.copy()
                self._transition(ExpertPhase.DESCEND_GRASP)

        elif self.phase == ExpertPhase.DESCEND_GRASP:
            waypoint = target + np.array([0.0, 0.0, 0.004])
            centered_between_fingers = (
                bool(info["bilateral_contact"])
                or (
                    np.linalg.norm(end_effector[:2] - target[:2]) < 0.013
                    and 0.012 < end_effector[2] - target[2] < 0.032
                )
            )
            if centered_between_fingers:
                self._transition(ExpertPhase.CLOSE_GRIPPER)

        elif self.phase == ExpertPhase.CLOSE_GRIPPER:
            jaw = -1.0
            if bool(info["bilateral_contact"]) and self.phase_steps >= 5:
                self._transition(ExpertPhase.LIFT)
            elif self.phase_steps >= self.close_retry_steps:
                self._retry()

        elif self.phase == ExpertPhase.LIFT:
            jaw = -1.0
            waypoint = np.array([target[0], target[1], 0.145])
            if self._lost_grasp(info):
                self._retry()
            elif end_effector[2] > 0.125:
                self._transition(ExpertPhase.TRANSPORT)

        elif self.phase == ExpertPhase.TRANSPORT:
            jaw = -1.0
            waypoint = np.array([goal[0], goal[1], 0.145])
            if self._lost_grasp(info):
                self._retry()
            elif np.linalg.norm(end_effector[:2] - goal[:2]) < 0.022:
                self._transition(ExpertPhase.DESCEND_RELEASE)

        elif self.phase == ExpertPhase.DESCEND_RELEASE:
            jaw = -1.0
            waypoint = np.array([goal[0], goal[1], 0.058])
            if end_effector[2] < 0.067:
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

    @property
    def search_offset(self) -> NDArray[np.float64]:
        """Return the current tactile-recovery offset in the table plane."""

        if self.retries == 0 or self.recovery_search_radius_m == 0.0:
            return np.zeros(3, dtype=np.float64)
        directions = (
            (-1.0, 0.0),
            (1.0, 0.0),
            (0.0, -1.0),
            (0.0, 1.0),
            (-1.0, -1.0),
            (1.0, 1.0),
            (-1.0, 1.0),
            (1.0, -1.0),
        )
        direction = directions[(self.retries - 1) % len(directions)]
        ring = 1 + (self.retries - 1) // len(directions)
        offset = (
            self.recovery_search_radius_m
            * ring
            * np.asarray(direction, dtype=np.float64)
        )
        return np.array([offset[0], offset[1], 0.0], dtype=np.float64)

    def _retry(self) -> None:
        self.retries += 1
        self._locked_target = None
        self._transition(ExpertPhase.APPROACH)

    def _lost_grasp(self, info: dict[str, Any]) -> bool:
        return bool(
            self.phase_steps > 8
            and not bool(info["bilateral_contact"])
            and not bool(info["assisted_grasp_active"])
        )

    def _transition(self, next_phase: ExpertPhase) -> None:
        self.phase = next_phase
        self.phase_steps = 0

    def _jaw_action(self, desired_jaw: float) -> float:
        alpha = (desired_jaw - self.config.closed_jaw) / (
            self.config.open_jaw - self.config.closed_jaw
        )
        return float(np.clip(2.0 * alpha - 1.0, -1.0, 1.0))

    def _smooth(
        self,
        previous: NDArray[np.float64] | None,
        measurement: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if previous is None:
            return measurement.copy()
        return (
            (1.0 - self.smoothing_alpha) * previous
            + self.smoothing_alpha * measurement
        )

    @staticmethod
    def _validate_position(
        position: NDArray[np.floating],
        name: str,
    ) -> NDArray[np.float64]:
        value = np.asarray(position, dtype=np.float64)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"{name} must be a finite vector with shape (3,)")
        return value

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
