from __future__ import annotations

import json
from typing import Any

import mujoco
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float32MultiArray, String
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

from embodied_vla.envs import SOArmEnvConfig, SOArmPickPlaceEnv
from embodied_vla_ros.transforms import frame_name, quaternion_from_matrix


class ArmBridge(Node):
    def __init__(self) -> None:
        super().__init__("embodied_vla_arm_bridge")
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("image_size", 128)
        self.declare_parameter("seed", 42)
        self.declare_parameter("grasp_mode", "contact_assisted")
        self.declare_parameter("auto_reset", True)

        self._rate_hz = float(self.get_parameter("rate_hz").value)
        self._seed = int(self.get_parameter("seed").value)
        image_size = int(self.get_parameter("image_size").value)
        grasp_mode = str(self.get_parameter("grasp_mode").value)
        self._auto_reset = bool(self.get_parameter("auto_reset").value)
        self._env = SOArmPickPlaceEnv(
            SOArmEnvConfig(
                observation_mode="multimodal",
                task_level="pick_place",
                grasp_mode=grasp_mode,
                image_size=image_size,
                max_episode_steps=300,
            )
        )
        self._observation, self._info = self._env.reset(seed=self._seed)
        self._action = np.zeros(5, dtype=np.float32)
        self._reset_count = 0
        self._episode_done = False

        self._joint_publisher = self.create_publisher(JointState, "/joint_states", 10)
        self._rgb_publisher = self.create_publisher(
            Image,
            "/camera/color/image_raw",
            qos_profile_sensor_data,
        )
        self._depth_publisher = self.create_publisher(
            Image,
            "/camera/depth/image_raw",
            qos_profile_sensor_data,
        )
        self._camera_info_publisher = self.create_publisher(
            CameraInfo,
            "/camera/color/camera_info",
            qos_profile_sensor_data,
        )
        self._task_publisher = self.create_publisher(String, "/vla/task", 10)
        self._metadata_publisher = self.create_publisher(String, "/vla/task_metadata", 10)
        self._tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(
            Float32MultiArray,
            "/vla/action",
            self._action_callback,
            10,
        )
        self.create_service(Trigger, "/vla/reset", self._reset_callback)
        self._timer = self.create_timer(1.0 / self._rate_hz, self._tick)
        self.get_logger().info(
            f"SO-ARM100 bridge ready at {self._rate_hz:.1f} Hz, image={image_size}"
        )

    def _action_callback(self, message: Float32MultiArray) -> None:
        action = np.asarray(message.data, dtype=np.float32)
        if action.shape != (5,):
            self.get_logger().warning(
                f"ignored /vla/action with shape {action.shape}; expected five values"
            )
            return
        self._action = np.clip(action, -1.0, 1.0)

    def _reset_callback(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        del request
        self._reset_environment()
        response.success = True
        response.message = str(self._info["instruction"])
        return response

    def _tick(self) -> None:
        if self._episode_done and self._auto_reset:
            self._reset_environment()
        ended_this_tick = False
        if not self._episode_done:
            self._observation, _, terminated, truncated, self._info = self._env.step(
                self._action
            )
            ended_this_tick = bool(terminated or truncated)
            self._episode_done = ended_this_tick
        stamp = self.get_clock().now().to_msg()
        self._publish_joint_state(stamp)
        self._publish_images(stamp)
        self._publish_task()
        self._publish_tf(stamp)
        if ended_this_tick:
            self.get_logger().info(
                f"episode ended success={self._info['success']} "
                f"reason={self._info['termination_reason']}"
            )

    def _reset_environment(self) -> None:
        self._reset_count += 1
        self._observation, self._info = self._env.reset(
            seed=self._seed + self._reset_count
        )
        self._action.fill(0.0)
        self._episode_done = False

    def _publish_joint_state(self, stamp: Any) -> None:
        message = JointState()
        message.header.stamp = stamp
        message.header.frame_id = "world"
        message.name = list(self._env.joint_names)
        message.position = self._env.joint_positions.astype(float).tolist()
        message.velocity = self._env.joint_velocities.astype(float).tolist()
        self._joint_publisher.publish(message)

    def _publish_images(self, stamp: Any) -> None:
        rgb = np.ascontiguousarray(self._observation["rgb"], dtype=np.uint8)
        rgb_message = Image()
        rgb_message.header.stamp = stamp
        rgb_message.header.frame_id = "camera_color_optical_frame"
        rgb_message.height, rgb_message.width = rgb.shape[:2]
        rgb_message.encoding = "rgb8"
        rgb_message.is_bigendian = False
        rgb_message.step = int(rgb_message.width * 3)
        rgb_message.data = rgb.tobytes()
        self._rgb_publisher.publish(rgb_message)

        depth = np.ascontiguousarray(self._env.render_depth(), dtype=np.float32)
        depth_message = Image()
        depth_message.header = rgb_message.header
        depth_message.height, depth_message.width = depth.shape
        depth_message.encoding = "32FC1"
        depth_message.is_bigendian = False
        depth_message.step = int(depth_message.width * 4)
        depth_message.data = depth.tobytes()
        self._depth_publisher.publish(depth_message)

        intrinsics = self._env.camera_intrinsics()
        camera_info = CameraInfo()
        camera_info.header = rgb_message.header
        camera_info.height = rgb_message.height
        camera_info.width = rgb_message.width
        camera_info.distortion_model = "plumb_bob"
        camera_info.d = [0.0] * 5
        camera_info.k = intrinsics.reshape(-1).tolist()
        camera_info.r = np.eye(3).reshape(-1).tolist()
        camera_info.p = [
            float(intrinsics[0, 0]),
            0.0,
            float(intrinsics[0, 2]),
            0.0,
            0.0,
            float(intrinsics[1, 1]),
            float(intrinsics[1, 2]),
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
        self._camera_info_publisher.publish(camera_info)

    def _publish_task(self) -> None:
        task = String()
        task.data = str(self._info["instruction"])
        self._task_publisher.publish(task)
        metadata = String()
        metadata.data = json.dumps(
            {
                "target_color": self._info["target_color"],
                "goal_side": self._info["goal_side"],
                "success": bool(self._info["success"]),
                "has_lifted": bool(self._info["has_lifted"]),
            }
        )
        self._metadata_publisher.publish(metadata)

    def _publish_tf(self, stamp: Any) -> None:
        transforms: list[TransformStamped] = []
        for body_id in range(1, self._env.model.nbody):
            parent_id = int(self._env.model.body_parentid[body_id])
            body_name = frame_name(
                mujoco.mj_id2name(self._env.model, mujoco.mjtObj.mjOBJ_BODY, body_id),
                f"body_{body_id}",
            )
            parent_name = (
                "world"
                if parent_id == 0
                else frame_name(
                    mujoco.mj_id2name(
                        self._env.model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        parent_id,
                    ),
                    f"body_{parent_id}",
                )
            )
            world_position = self._env.data.xpos[body_id]
            world_rotation = self._env.data.xmat[body_id].reshape(3, 3)
            if parent_id == 0:
                relative_position = world_position
                relative_rotation = world_rotation
            else:
                parent_position = self._env.data.xpos[parent_id]
                parent_rotation = self._env.data.xmat[parent_id].reshape(3, 3)
                relative_position = parent_rotation.T @ (
                    world_position - parent_position
                )
                relative_rotation = parent_rotation.T @ world_rotation
            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = parent_name
            transform.child_frame_id = body_name
            transform.transform.translation.x = float(relative_position[0])
            transform.transform.translation.y = float(relative_position[1])
            transform.transform.translation.z = float(relative_position[2])
            transform.transform.rotation = quaternion_from_matrix(relative_rotation)
            transforms.append(transform)

        camera_position, camera_rotation = self._env.camera_pose()
        optical_rotation = camera_rotation @ np.diag([1.0, -1.0, -1.0])
        camera_transform = TransformStamped()
        camera_transform.header.stamp = stamp
        camera_transform.header.frame_id = "world"
        camera_transform.child_frame_id = "camera_color_optical_frame"
        camera_transform.transform.translation.x = float(camera_position[0])
        camera_transform.transform.translation.y = float(camera_position[1])
        camera_transform.transform.translation.z = float(camera_position[2])
        camera_transform.transform.rotation = quaternion_from_matrix(optical_rotation)
        transforms.append(camera_transform)
        self._tf_broadcaster.sendTransform(transforms)

    def destroy_node(self) -> bool:
        self._env.close()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ArmBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
