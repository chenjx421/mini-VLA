from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from embodied_vla.control import damped_least_squares
from embodied_vla.envs.config import SOArmEnvConfig
from embodied_vla.language import make_instruction, tokenize
from embodied_vla.proprioception import (
    CARTESIAN_PROPRIO_DIM,
    END_EFFECTOR_POSITION_SCALE,
    JOINT_PROPRIO_DIM,
    assemble_model_proprio,
)

COLORS = ("red", "green", "blue")
SIDES = ("left", "right")
ARM_JOINTS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll")
ALL_JOINTS = ARM_JOINTS + ("Jaw",)
FINGER_GEOMS = {
    "fixed": tuple(f"fixed_jaw_pad_{index}" for index in range(1, 5)),
    "moving": tuple(f"moving_jaw_pad_{index}" for index in range(1, 5)),
}


@dataclass(frozen=True)
class ContactState:
    fixed: bool
    moving: bool
    fixed_force: float
    moving_force: float

    @property
    def bilateral(self) -> bool:
        return self.fixed and self.moving


class SOArmPickPlaceEnv(gym.Env):
    """Language-conditioned SO-ARM100 manipulation with MuJoCo contacts.

    The policy emits normalized Cartesian delta actions:

    ``[dx, dy, dz, wrist_roll, jaw]``.

    Cartesian deltas are converted to the five arm joint targets with damped
    least-squares inverse kinematics. ``jaw=-1`` means close and ``jaw=+1``
    means open.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}
    STATE_DIM = 37
    PROPRIO_DIM = JOINT_PROPRIO_DIM
    LANGUAGE_LENGTH = 16

    def __init__(
        self,
        config: SOArmEnvConfig | None = None,
        *,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or SOArmEnvConfig()
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        self.render_mode = render_mode

        xml_path = (
            Path(__file__).resolve().parents[1]
            / "assets"
            / "so_arm100"
            / "pick_place.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.model.opt.timestep = self.config.physics_timestep
        self.data = mujoco.MjData(self.model)
        self._renderer: mujoco.Renderer | None = None

        self._joint_ids = np.array(
            [self._name_to_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in ALL_JOINTS],
            dtype=np.int32,
        )
        self._qpos_adrs = self.model.jnt_qposadr[self._joint_ids]
        self._dof_adrs = self.model.jnt_dofadr[self._joint_ids]
        self._arm_dof_adrs = self._dof_adrs[: len(ARM_JOINTS)]
        self._actuator_ids = np.array(
            [self._name_to_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ALL_JOINTS],
            dtype=np.int32,
        )
        self._grip_site_id = self._name_to_id(mujoco.mjtObj.mjOBJ_SITE, "grip_site")
        self._render_camera_id = self._name_to_id(
            mujoco.mjtObj.mjOBJ_CAMERA,
            self.config.render_camera,
        )

        self._cube_body_ids = np.array(
            [self._name_to_id(mujoco.mjtObj.mjOBJ_BODY, f"{color}_cube") for color in COLORS],
            dtype=np.int32,
        )
        self._cube_geom_ids = np.array(
            [self._name_to_id(mujoco.mjtObj.mjOBJ_GEOM, f"{color}_cube_geom") for color in COLORS],
            dtype=np.int32,
        )
        self._cube_joint_ids = np.array(
            [
                self._name_to_id(mujoco.mjtObj.mjOBJ_JOINT, f"{color}_cube_joint")
                for color in COLORS
            ],
            dtype=np.int32,
        )
        self._cube_qpos_adrs = self.model.jnt_qposadr[self._cube_joint_ids]
        self._cube_dof_adrs = self.model.jnt_dofadr[self._cube_joint_ids]
        self._goal_site_ids = {
            side: self._name_to_id(mujoco.mjtObj.mjOBJ_SITE, f"{side}_goal_site")
            for side in SIDES
        }
        self._finger_geom_ids = {
            side: {
                self._name_to_id(mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in names
            }
            for side, names in FINGER_GEOMS.items()
        }
        self._assist_eq_ids = np.array(
            [
                self._name_to_id(mujoco.mjtObj.mjOBJ_EQUALITY, f"assist_{color}")
                for color in COLORS
            ],
            dtype=np.int32,
        )

        self._joint_ranges = self.model.jnt_range[self._joint_ids].copy()
        self._base_light_diffuse = self.model.light_diffuse.copy()
        self._base_geom_friction = self.model.geom_friction.copy()
        self._home_qpos = np.array(
            [0.0, -1.57, 1.57, 1.57, -1.57, self.config.pregrasp_jaw]
        )
        self._joint_targets = self._home_qpos.copy()
        self._target_index = 0
        self._goal_side = "left"
        self._instruction = ""
        self._language_tokens = np.zeros(self.LANGUAGE_LENGTH, dtype=np.int64)
        self._language_mask = np.zeros(self.LANGUAGE_LENGTH, dtype=np.int8)
        self._step_count = 0
        self._has_grasped = False
        self._has_lifted = False
        self._previous_potential = 0.0
        self._last_contact = ContactState(False, False, 0.0, 0.0)
        self._assisted_grasp_active = False

        self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        if self.config.observation_mode == "state":
            self.observation_space = spaces.Box(
                low=-2.0,
                high=2.0,
                shape=(self.STATE_DIM,),
                dtype=np.float32,
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "rgb": spaces.Box(
                        0,
                        255,
                        shape=(self.config.image_size, self.config.image_size, 3),
                        dtype=np.uint8,
                    ),
                    "proprio": spaces.Box(
                        -1.0,
                        1.0,
                        shape=(self.proprio_dim,),
                        dtype=np.float32,
                    ),
                    "language": spaces.Box(
                        0,
                        np.iinfo(np.int32).max,
                        shape=(self.LANGUAGE_LENGTH,),
                        dtype=np.int64,
                    ),
                    "language_mask": spaces.Box(
                        0,
                        1,
                        shape=(self.LANGUAGE_LENGTH,),
                        dtype=np.int8,
                    ),
                }
            )

    @property
    def target_color(self) -> str:
        return COLORS[self._target_index]

    @property
    def instruction(self) -> str:
        return self._instruction

    @property
    def proprio_dim(self) -> int:
        return (
            CARTESIAN_PROPRIO_DIM
            if self.config.include_end_effector_position_in_proprio
            else JOINT_PROPRIO_DIM
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32] | dict[str, NDArray[Any]], dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        mujoco.mj_resetData(self.model, self.data)

        target_color = options.get("target_color")
        goal_side = options.get("goal_side")
        if target_color is not None and target_color not in COLORS:
            raise ValueError(f"target_color must be one of {COLORS}")
        if goal_side is not None and goal_side not in SIDES:
            raise ValueError(f"goal_side must be one of {SIDES}")
        self._target_index = (
            COLORS.index(target_color)
            if target_color is not None
            else int(self.np_random.integers(len(COLORS)))
        )
        self._goal_side = (
            str(goal_side)
            if goal_side is not None
            else SIDES[int(self.np_random.integers(len(SIDES)))]
        )
        self._instruction = make_instruction(
            self.target_color,
            self._goal_side,
            template_index=int(self.np_random.integers(3)),
        )
        self._language_tokens, self._language_mask = tokenize(
            self._instruction,
            max_length=self.LANGUAGE_LENGTH,
        )

        self.data.qpos[self._qpos_adrs] = self._home_qpos
        self.data.qvel[:] = 0.0
        self._joint_targets = self._home_qpos.copy()
        self.data.ctrl[self._actuator_ids] = self._joint_targets
        self.data.eq_active[self._assist_eq_ids] = 0
        self._assisted_grasp_active = False
        self.model.light_diffuse[:] = self._base_light_diffuse
        self.model.geom_friction[:] = self._base_geom_friction
        self._randomize_cube_poses()
        self._apply_domain_randomization()
        mujoco.mj_forward(self.model, self.data)

        for _ in range(50):
            self.data.ctrl[self._actuator_ids] = self._joint_targets
            mujoco.mj_step(self.model, self.data)

        self._step_count = 0
        self._has_grasped = False
        self._has_lifted = False
        self._last_contact = self._contact_state()
        self._previous_potential = self._task_potential()
        observation = self._observation()
        info = self._info(success=False, reason=None)
        return observation, info

    def step(
        self,
        action: NDArray[np.floating],
    ) -> tuple[
        NDArray[np.float32] | dict[str, NDArray[Any]],
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.action_space.shape:
            raise ValueError(f"expected action shape {self.action_space.shape}, got {action.shape}")
        action = np.clip(action, -1.0, 1.0)

        self._apply_action(action)
        self._step_count += 1
        contact = self._contact_state()
        newly_grasped = contact.bilateral and not self._has_grasped
        self._has_grasped = self._has_grasped or contact.bilateral

        target_position = self._target_position()
        lift_threshold = (
            self.config.table_height
            + 2.0 * self.config.cube_half_size
            + 0.035
        )
        newly_lifted = target_position[2] > lift_threshold and not self._has_lifted
        self._has_lifted = self._has_lifted or newly_lifted
        self._last_contact = contact

        success = self._is_success()
        failed = self._is_failure()
        terminated = bool(success or failed)
        truncated = bool(
            not terminated and self._step_count >= self.config.max_episode_steps
        )

        potential = self._task_potential()
        reward = potential - self._previous_potential
        reward -= 0.002 * float(np.square(action[:4]).mean())
        reward += 0.5 if newly_grasped else 0.0
        reward += 1.0 if newly_lifted else 0.0
        reward += self._terminal_reward(success=success, failed=failed)
        self._previous_potential = potential

        reason = "success" if success else "object_out_of_bounds" if failed else None
        observation = self._observation()
        info = self._info(success=success, reason=reason)
        info["reward_terms"] = {
            "potential": float(potential),
            "new_grasp": float(newly_grasped) * 0.5,
            "new_lift": float(newly_lifted),
            "terminal": self._terminal_reward(success=success, failed=failed),
        }
        return observation, float(reward), terminated, truncated, info

    def render(self) -> NDArray[np.uint8]:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model,
                height=self.config.image_size,
                width=self.config.image_size,
            )
        self._renderer.update_scene(self.data, camera=self.config.render_camera)
        return np.asarray(self._renderer.render(), dtype=np.uint8).copy()

    def render_depth(self) -> NDArray[np.float32]:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model,
                height=self.config.image_size,
                width=self.config.image_size,
            )
        self._renderer.update_scene(self.data, camera=self.config.render_camera)
        self._renderer.enable_depth_rendering()
        try:
            return np.asarray(self._renderer.render(), dtype=np.float32).copy()
        finally:
            self._renderer.disable_depth_rendering()

    @property
    def joint_names(self) -> tuple[str, ...]:
        return ALL_JOINTS

    @property
    def joint_positions(self) -> NDArray[np.float32]:
        return self.data.qpos[self._qpos_adrs].astype(np.float32).copy()

    @property
    def joint_velocities(self) -> NDArray[np.float32]:
        return self.data.qvel[self._dof_adrs].astype(np.float32).copy()

    def camera_intrinsics(self) -> NDArray[np.float64]:
        focal = 0.5 * self.config.image_size / np.tan(
            np.deg2rad(float(self.model.cam_fovy[self._render_camera_id])) / 2.0
        )
        center = (self.config.image_size - 1) / 2.0
        return np.array(
            [
                [focal, 0.0, center],
                [0.0, focal, center],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def camera_pose(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return render-camera position and rotation in the MuJoCo world frame."""

        position = self.data.cam_xpos[self._render_camera_id].copy()
        rotation = self.data.cam_xmat[self._render_camera_id].reshape(3, 3).copy()
        return position, rotation

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def privileged_state(self) -> NDArray[np.float32]:
        """Return simulator state for critics, experts, and labels only."""

        return self._state_observation().copy()

    def project_world_point(
        self,
        world_point: NDArray[np.floating],
    ) -> tuple[NDArray[np.float32], bool]:
        """Project a 3D world point into normalized camera coordinates."""

        point = np.asarray(world_point, dtype=np.float64)
        if point.shape != (3,):
            raise ValueError(f"world_point must have shape (3,), got {point.shape}")
        camera_position = self.data.cam_xpos[self._render_camera_id]
        camera_rotation = self.data.cam_xmat[self._render_camera_id].reshape(3, 3)
        camera_point = camera_rotation.T @ (point - camera_position)
        depth = -float(camera_point[2])
        if depth <= 1e-6:
            return np.array([-1.0, -1.0], dtype=np.float32), False

        focal_scale = 1.0 / np.tan(
            np.deg2rad(float(self.model.cam_fovy[self._render_camera_id])) / 2.0
        )
        x_ndc = float(camera_point[0]) * focal_scale / depth
        y_ndc = float(camera_point[1]) * focal_scale / depth
        normalized = np.array(
            [(x_ndc + 1.0) / 2.0, (1.0 - y_ndc) / 2.0],
            dtype=np.float32,
        )
        visible = bool(
            depth > 0.0
            and 0.0 <= normalized[0] <= 1.0
            and 0.0 <= normalized[1] <= 1.0
        )
        return normalized, visible

    def unproject_normalized_pixel_to_plane(
        self,
        normalized_pixel: NDArray[np.floating],
        *,
        world_z: float,
    ) -> NDArray[np.float32]:
        """Intersect a calibrated camera ray with a horizontal world plane."""

        pixel = np.asarray(normalized_pixel, dtype=np.float64)
        if pixel.shape != (2,):
            raise ValueError(
                f"normalized_pixel must have shape (2,), got {pixel.shape}"
            )
        x_ndc = 2.0 * float(pixel[0]) - 1.0
        y_ndc = 1.0 - 2.0 * float(pixel[1])
        focal_scale = 1.0 / np.tan(
            np.deg2rad(float(self.model.cam_fovy[self._render_camera_id])) / 2.0
        )
        camera_ray = np.array(
            [x_ndc / focal_scale, y_ndc / focal_scale, -1.0],
            dtype=np.float64,
        )
        camera_position = self.data.cam_xpos[self._render_camera_id]
        camera_rotation = self.data.cam_xmat[self._render_camera_id].reshape(3, 3)
        world_ray = camera_rotation @ camera_ray
        if abs(float(world_ray[2])) < 1e-9:
            raise ValueError("camera ray is parallel to the requested world plane")
        ray_scale = (float(world_z) - float(camera_position[2])) / float(
            world_ray[2]
        )
        if ray_scale <= 0.0:
            raise ValueError("requested plane lies behind the camera ray")
        world_point = camera_position + ray_scale * world_ray
        world_point[2] = float(world_z)
        return world_point.astype(np.float32)

    def _name_to_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo object not found: {object_type.name} {name!r}")
        return object_id

    def _randomize_cube_poses(self) -> None:
        positions: list[NDArray[np.float64]] = []
        for cube_index, qpos_address in enumerate(self._cube_qpos_adrs):
            for _ in range(200):
                position = np.array(
                    [
                        self.np_random.uniform(-0.13, 0.13),
                        self.np_random.uniform(-0.285, -0.175),
                        self.config.table_height + self.config.cube_half_size + 0.003,
                    ],
                    dtype=np.float64,
                )
                clears_other_cubes = all(
                    np.linalg.norm(position[:2] - old[:2]) >= 0.055 for old in positions
                )
                clears_goals = all(
                    np.linalg.norm(
                        position[:2] - self.model.site_pos[site_id, :2]
                    )
                    >= self.config.success_radius + 0.025
                    for site_id in self._goal_site_ids.values()
                )
                if clears_other_cubes and clears_goals:
                    break
            else:
                raise RuntimeError("could not sample non-overlapping cube positions")
            positions.append(position)
            yaw = float(self.np_random.uniform(-np.pi, np.pi))
            quaternion = np.array(
                [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)],
                dtype=np.float64,
            )
            self.data.qpos[qpos_address : qpos_address + 3] = position
            self.data.qpos[qpos_address + 3 : qpos_address + 7] = quaternion
            dof_address = self._cube_dof_adrs[cube_index]
            self.data.qvel[dof_address : dof_address + 6] = 0.0

    def _apply_domain_randomization(self) -> None:
        if not self.config.domain_randomization:
            return
        light_scale = float(self.np_random.uniform(0.75, 1.25))
        self.model.light_diffuse[:] = np.clip(
            self.model.light_diffuse * light_scale,
            0.1,
            1.0,
        )
        friction = float(self.np_random.uniform(0.8, 1.6))
        self.model.geom_friction[self._cube_geom_ids, 0] = friction

    def _apply_action(self, action: NDArray[np.float64]) -> None:
        if self.config.grasp_mode == "contact_assisted":
            if action[4] < -0.5 and not self._assisted_grasp_active:
                if self._contact_state().bilateral:
                    self.data.eq_active[self._assist_eq_ids[self._target_index]] = 1
                    self._assisted_grasp_active = True
            elif action[4] > 0.0 and self._assisted_grasp_active:
                self.data.eq_active[self._assist_eq_ids[self._target_index]] = 0
                self._assisted_grasp_active = False

        jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(
            self.model,
            self.data,
            jacobian_position,
            jacobian_rotation,
            self._grip_site_id,
        )
        task_jacobian = jacobian_position[:, self._arm_dof_adrs]
        displacement = action[:3] * self.config.cartesian_action_scale
        joint_delta = damped_least_squares(
            task_jacobian,
            displacement,
            damping=self.config.ik_damping,
        )
        joint_delta = np.clip(joint_delta, -0.12, 0.12)
        current_arm_positions = self.data.qpos[self._qpos_adrs[: len(ARM_JOINTS)]]
        self._joint_targets[: len(ARM_JOINTS)] = current_arm_positions + joint_delta
        self._joint_targets[4] += action[3] * self.config.wrist_action_scale

        jaw_alpha = (float(action[4]) + 1.0) / 2.0
        jaw_target = (
            (1.0 - jaw_alpha) * self.config.closed_jaw
            + jaw_alpha * self.config.open_jaw
        )
        self._joint_targets[5] = jaw_target
        self._joint_targets = np.clip(
            self._joint_targets,
            self._joint_ranges[:, 0],
            self._joint_ranges[:, 1],
        )

        for _ in range(self.config.control_substeps):
            self.data.ctrl[self._actuator_ids] = self._joint_targets
            mujoco.mj_step(self.model, self.data)

    def _contact_state(self) -> ContactState:
        target_geom_id = int(self._cube_geom_ids[self._target_index])
        touched = {"fixed": False, "moving": False}
        forces = {"fixed": 0.0, "moving": 0.0}
        contact_force = np.zeros(6, dtype=np.float64)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if target_geom_id not in (contact.geom1, contact.geom2):
                continue
            other_geom = contact.geom2 if contact.geom1 == target_geom_id else contact.geom1
            side = next(
                (
                    candidate
                    for candidate, geom_ids in self._finger_geom_ids.items()
                    if other_geom in geom_ids
                ),
                None,
            )
            if side is None:
                continue
            mujoco.mj_contactForce(self.model, self.data, contact_index, contact_force)
            normal_force = max(0.0, float(contact_force[0]))
            touched[side] = True
            forces[side] += normal_force
        return ContactState(
            fixed=touched["fixed"],
            moving=touched["moving"],
            fixed_force=forces["fixed"],
            moving_force=forces["moving"],
        )

    def _observation(self) -> NDArray[np.float32] | dict[str, NDArray[Any]]:
        if self.config.observation_mode == "state":
            return self._state_observation()
        return {
            "rgb": self.render(),
            "proprio": self._proprioception(),
            "language": self._language_tokens.copy(),
            "language_mask": self._language_mask.copy(),
        }

    def _proprioception(self) -> NDArray[np.float32]:
        joint_proprio = self._joint_proprioception()
        if not self.config.include_end_effector_position_in_proprio:
            return joint_proprio
        return assemble_model_proprio(
            joint_proprio,
            expected_dim=CARTESIAN_PROPRIO_DIM,
            normalized_end_effector_position=(
                self._end_effector_position() / END_EFFECTOR_POSITION_SCALE
            ),
        )

    def _joint_proprioception(self) -> NDArray[np.float32]:
        joint_positions = self.data.qpos[self._qpos_adrs]
        joint_velocities = self.data.qvel[self._dof_adrs]
        normalized_positions = 2.0 * (
            (joint_positions - self._joint_ranges[:, 0])
            / (self._joint_ranges[:, 1] - self._joint_ranges[:, 0])
        ) - 1.0
        normalized_velocities = np.tanh(joint_velocities / 4.0)
        return np.concatenate([normalized_positions, normalized_velocities]).astype(np.float32)

    def _state_observation(self) -> NDArray[np.float32]:
        target_position = self._target_position()
        target_velocity = self.data.qvel[
            self._cube_dof_adrs[self._target_index] : self._cube_dof_adrs[self._target_index] + 3
        ]
        end_effector = self._end_effector_position()
        goal = self._goal_position()
        color_one_hot = np.eye(len(COLORS), dtype=np.float64)[self._target_index]
        side_one_hot = np.eye(len(SIDES), dtype=np.float64)[SIDES.index(self._goal_side)]
        contact = np.array(
            [self._last_contact.fixed, self._last_contact.moving],
            dtype=np.float64,
        )
        state = np.concatenate(
            [
                self._joint_proprioception().astype(np.float64),
                end_effector / 0.5,
                target_position / 0.5,
                np.tanh(target_velocity / 2.0),
                goal / 0.5,
                (target_position - end_effector) / 0.5,
                (goal - target_position) / 0.5,
                color_one_hot,
                side_one_hot,
                contact,
            ]
        )
        if state.shape != (self.STATE_DIM,):
            raise RuntimeError(f"internal state contract changed: {state.shape}")
        return state.astype(np.float32)

    def _end_effector_position(self) -> NDArray[np.float64]:
        return self.data.site_xpos[self._grip_site_id].copy()

    def _target_position(self) -> NDArray[np.float64]:
        return self.data.xpos[self._cube_body_ids[self._target_index]].copy()

    def _goal_position(self) -> NDArray[np.float64]:
        return self.data.site_xpos[self._goal_site_ids[self._goal_side]].copy()

    def _task_potential(self) -> float:
        end_effector = self._end_effector_position()
        target = self._target_position()
        goal = self._goal_position()
        reach_distance = float(np.linalg.norm(end_effector - target))
        table_rest_z = self.config.table_height + self.config.cube_half_size
        lift_progress = float(np.clip((target[2] - table_rest_z) / 0.10, 0.0, 1.0))
        goal_distance = float(np.linalg.norm(goal[:2] - target[:2]))

        if self.config.task_level == "reach":
            return -4.0 * reach_distance
        if self.config.task_level == "pick":
            return -2.0 * reach_distance + 4.0 * lift_progress
        place_term = -5.0 * goal_distance if self._has_lifted else 0.0
        return -2.0 * reach_distance + 4.0 * lift_progress + place_term

    def _is_success(self) -> bool:
        target = self._target_position()
        if self.config.task_level == "reach":
            return bool(np.linalg.norm(self._end_effector_position() - target) < 0.025)
        if self.config.task_level == "pick":
            return bool(
                target[2]
                > self.config.table_height + 2.0 * self.config.cube_half_size + 0.035
            )

        goal = self._goal_position()
        in_goal = np.linalg.norm(target[:2] - goal[:2]) < self.config.success_radius
        released = self._joint_targets[5] > 0.20 and not self._last_contact.bilateral
        settled_height = target[2] < self.config.table_height + 0.065
        return bool(self._has_lifted and in_goal and released and settled_height)

    def _is_failure(self) -> bool:
        target = self._target_position()
        return bool(
            target[2] < -0.02
            or abs(target[0]) > 0.34
            or target[1] < -0.52
            or target[1] > 0.05
        )

    def _terminal_reward(self, *, success: bool, failed: bool) -> float:
        if success:
            return 5.0 if self.config.task_level == "reach" else 15.0
        if failed:
            return -5.0
        return 0.0

    def _info(self, *, success: bool, reason: str | None) -> dict[str, Any]:
        target = self._target_position()
        goal = self._goal_position()
        end_effector = self._end_effector_position()
        return {
            "step": self._step_count,
            "instruction": self._instruction,
            "target_color": self.target_color,
            "goal_side": self._goal_side,
            "target_position": target.astype(np.float32),
            "goal_position": goal.astype(np.float32),
            "end_effector_position": end_effector.astype(np.float32),
            "reach_distance": float(np.linalg.norm(end_effector - target)),
            "goal_distance": float(np.linalg.norm(target[:2] - goal[:2])),
            "bilateral_contact": self._last_contact.bilateral,
            "contact_force": np.array(
                [self._last_contact.fixed_force, self._last_contact.moving_force],
                dtype=np.float32,
            ),
            "has_grasped": self._has_grasped,
            "has_lifted": self._has_lifted,
            "grasp_mode": self.config.grasp_mode,
            "assisted_grasp_active": self._assisted_grasp_active,
            "success": success,
            "termination_reason": reason,
        }
