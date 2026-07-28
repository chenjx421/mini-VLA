from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files

import mujoco
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MobileSlamConfig:
    control_dt: float = 0.05
    physics_timestep: float = 0.002
    lidar_beams: int = 180
    lidar_min_range: float = 0.08
    lidar_max_range: float = 6.0
    lidar_noise_std: float = 0.005
    odometry_linear_noise_std: float = 0.003
    odometry_angular_noise_std: float = 0.002
    max_linear_velocity: float = 0.45
    max_angular_velocity: float = 1.2
    render_size: int = 512

    def __post_init__(self) -> None:
        ratio = self.control_dt / self.physics_timestep
        if ratio < 1 or not np.isclose(ratio, round(ratio)):
            raise ValueError("control_dt must be an integer multiple of physics_timestep")
        if self.lidar_beams < 16:
            raise ValueError("lidar_beams must be at least 16")
        if not 0.0 < self.lidar_min_range < self.lidar_max_range:
            raise ValueError("invalid lidar range")

    @property
    def control_substeps(self) -> int:
        return round(self.control_dt / self.physics_timestep)


class MobileSlamEnv:
    """MuJoCo mobile sensor platform for ROS 2 SLAM experiments."""

    def __init__(self, config: MobileSlamConfig | None = None) -> None:
        self.config = config or MobileSlamConfig()
        model_path = files("embodied_vla").joinpath("assets", "slam", "warehouse.xml")
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.model.opt.timestep = self.config.physics_timestep
        self.data = mujoco.MjData(self.model)

        self._joint_ids = np.asarray(
            [
                self._name_to_id(mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in ("base_x", "base_y", "base_yaw")
            ],
            dtype=np.int32,
        )
        self._qpos_addresses = self.model.jnt_qposadr[self._joint_ids]
        self._actuator_ids = np.asarray(
            [
                self._name_to_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in ("x_velocity", "y_velocity", "yaw_velocity")
            ],
            dtype=np.int32,
        )
        self._base_body_id = self._name_to_id(
            mujoco.mjtObj.mjOBJ_BODY,
            "mobile_base",
        )
        self._lidar_site_id = self._name_to_id(mujoco.mjtObj.mjOBJ_SITE, "lidar")
        self._camera_id = self._name_to_id(mujoco.mjtObj.mjOBJ_CAMERA, "overhead")
        self._angles = np.linspace(
            -np.pi,
            np.pi,
            self.config.lidar_beams,
            endpoint=False,
            dtype=np.float64,
        )
        self._rng = np.random.default_rng()
        self._odom_pose = np.zeros(3, dtype=np.float64)
        self._last_true_pose = np.zeros(3, dtype=np.float64)
        self._linear_bias = 0.0
        self._angular_bias = 0.0
        self._renderer: mujoco.Renderer | None = None

    @property
    def angle_min(self) -> float:
        return -np.pi

    @property
    def angle_increment(self) -> float:
        return 2.0 * np.pi / self.config.lidar_beams

    def reset(self, *, seed: int | None = None) -> dict[str, NDArray[np.float32]]:
        self._rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._qpos_addresses] = np.array([-2.25, -2.25, 0.0])
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._last_true_pose = self.true_pose.astype(np.float64)
        self._odom_pose = self._last_true_pose.copy()
        self._linear_bias = float(self._rng.normal(0.0, 0.0015))
        self._angular_bias = float(self._rng.normal(0.0, 0.001))
        return self.observation()

    def step(
        self,
        command: NDArray[np.floating],
    ) -> dict[str, NDArray[np.float32]]:
        command = np.asarray(command, dtype=np.float64)
        if command.shape != (2,):
            raise ValueError(f"expected [linear, angular] command, got {command.shape}")
        linear = float(
            np.clip(
                command[0],
                -self.config.max_linear_velocity,
                self.config.max_linear_velocity,
            )
        )
        angular = float(
            np.clip(
                command[1],
                -self.config.max_angular_velocity,
                self.config.max_angular_velocity,
            )
        )
        yaw = float(self.true_pose[2])
        self.data.ctrl[self._actuator_ids] = np.array(
            [linear * np.cos(yaw), linear * np.sin(yaw), angular],
            dtype=np.float64,
        )
        for _ in range(self.config.control_substeps):
            mujoco.mj_step(self.model, self.data)
        self._update_odometry()
        return self.observation()

    @property
    def true_pose(self) -> NDArray[np.float32]:
        pose = self.data.qpos[self._qpos_addresses].copy()
        pose[2] = _wrap_angle(float(pose[2]))
        return pose.astype(np.float32)

    @property
    def odometry_pose(self) -> NDArray[np.float32]:
        return self._odom_pose.astype(np.float32).copy()

    def observation(self) -> dict[str, NDArray[np.float32]]:
        return {
            "scan": self.lidar_scan(),
            "true_pose": self.true_pose,
            "odometry_pose": self.odometry_pose,
        }

    def lidar_scan(self) -> NDArray[np.float32]:
        origin = self.data.site_xpos[self._lidar_site_id].copy()
        yaw = float(self.true_pose[2])
        distances = np.empty(self.config.lidar_beams, dtype=np.float64)
        geom_id = np.empty(1, dtype=np.int32)
        for index, relative_angle in enumerate(self._angles):
            angle = yaw + relative_angle
            direction = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float64)
            distance = mujoco.mj_ray(
                self.model,
                self.data,
                origin,
                direction,
                None,
                True,
                self._base_body_id,
                geom_id,
            )
            distances[index] = (
                self.config.lidar_max_range if distance < 0.0 else distance
            )
        if self.config.lidar_noise_std > 0.0:
            distances += self._rng.normal(
                0.0,
                self.config.lidar_noise_std,
                size=distances.shape,
            )
        return np.clip(
            distances,
            self.config.lidar_min_range,
            self.config.lidar_max_range,
        ).astype(np.float32)

    def render(self) -> NDArray[np.uint8]:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model,
                height=self.config.render_size,
                width=self.config.render_size,
            )
        self._renderer.update_scene(self.data, camera=self._camera_id)
        return self._renderer.render().copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _update_odometry(self) -> None:
        current_true = self.true_pose.astype(np.float64)
        previous_yaw = float(self._last_true_pose[2])
        delta_world = current_true[:2] - self._last_true_pose[:2]
        rotation_world_to_body = np.array(
            [
                [np.cos(previous_yaw), np.sin(previous_yaw)],
                [-np.sin(previous_yaw), np.cos(previous_yaw)],
            ]
        )
        delta_body = rotation_world_to_body @ delta_world
        noisy_delta_body = delta_body * (
            1.0
            + self._linear_bias
            + self._rng.normal(0.0, self.config.odometry_linear_noise_std)
        )
        delta_yaw = _wrap_angle(float(current_true[2] - previous_yaw))
        noisy_delta_yaw = (
            delta_yaw
            + self._angular_bias * self.config.control_dt
            + float(self._rng.normal(0.0, self.config.odometry_angular_noise_std))
        )

        odom_yaw = float(self._odom_pose[2])
        rotation_body_to_odom = np.array(
            [
                [np.cos(odom_yaw), -np.sin(odom_yaw)],
                [np.sin(odom_yaw), np.cos(odom_yaw)],
            ]
        )
        self._odom_pose[:2] += rotation_body_to_odom @ noisy_delta_body
        self._odom_pose[2] = _wrap_angle(odom_yaw + noisy_delta_yaw)
        self._last_true_pose = current_true

    def _name_to_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return object_id


def _wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi
