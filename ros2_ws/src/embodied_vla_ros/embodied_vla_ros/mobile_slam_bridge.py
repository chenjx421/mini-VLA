from __future__ import annotations

from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from embodied_vla.envs import MobileSlamConfig, MobileSlamEnv
from embodied_vla_ros.transforms import planar_quaternion


class MobileSlamBridge(Node):
    def __init__(self) -> None:
        super().__init__("embodied_vla_mobile_slam_bridge")
        self.declare_parameter("seed", 7)
        self.declare_parameter("autopilot", True)
        self.declare_parameter("lidar_beams", 180)
        self.declare_parameter("control_dt", 0.05)

        seed = int(self.get_parameter("seed").value)
        self._autopilot = bool(self.get_parameter("autopilot").value)
        self._env = MobileSlamEnv(
            MobileSlamConfig(
                lidar_beams=int(self.get_parameter("lidar_beams").value),
                control_dt=float(self.get_parameter("control_dt").value),
            )
        )
        self._observation = self._env.reset(seed=seed)
        self._command = np.zeros(2, dtype=np.float32)
        self._waypoints = np.asarray(
            [
                [-2.25, 2.35],
                [2.25, 2.35],
                [2.25, -2.25],
                [-2.25, -2.25],
            ],
            dtype=np.float32,
        )
        self._waypoint_index = 0

        self._scan_publisher = self.create_publisher(
            LaserScan,
            "/scan",
            qos_profile_sensor_data,
        )
        self._odom_publisher = self.create_publisher(Odometry, "/odom", 20)
        self._truth_publisher = self.create_publisher(
            PoseStamped,
            "/ground_truth_pose",
            10,
        )
        self._tf_broadcaster = TransformBroadcaster(self)
        self._static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.create_subscription(Twist, "/cmd_vel", self._command_callback, 10)
        self._publish_static_tf()
        self._timer = self.create_timer(self._env.config.control_dt, self._tick)
        self.get_logger().info(
            f"mobile SLAM bridge ready, autopilot={self._autopilot}, "
            f"scan beams={self._env.config.lidar_beams}"
        )

    def _command_callback(self, message: Twist) -> None:
        if self._autopilot:
            return
        self._command[:] = [message.linear.x, message.angular.z]

    def _tick(self) -> None:
        if self._autopilot:
            self._command = self._autopilot_command()
        self._observation = self._env.step(self._command)
        stamp = self.get_clock().now().to_msg()
        self._publish_scan(stamp)
        self._publish_odometry(stamp)
        self._publish_ground_truth(stamp)

    def _autopilot_command(self) -> np.ndarray:
        pose = self._observation["true_pose"]
        target = self._waypoints[self._waypoint_index]
        delta = target - pose[:2]
        distance = float(np.linalg.norm(delta))
        if distance < 0.18:
            self._waypoint_index = (self._waypoint_index + 1) % len(self._waypoints)
            target = self._waypoints[self._waypoint_index]
            delta = target - pose[:2]
        desired_heading = float(np.arctan2(delta[1], delta[0]))
        heading_error = (desired_heading - float(pose[2]) + np.pi) % (
            2.0 * np.pi
        ) - np.pi
        angular = float(np.clip(2.2 * heading_error, -1.1, 1.1))
        linear = 0.34 * max(0.0, np.cos(heading_error))
        if abs(heading_error) > 0.8:
            linear = 0.0
        return np.asarray([linear, angular], dtype=np.float32)

    def _publish_scan(self, stamp: Any) -> None:
        message = LaserScan()
        message.header.stamp = stamp
        message.header.frame_id = "laser"
        message.angle_min = float(self._env.angle_min)
        message.angle_increment = float(self._env.angle_increment)
        message.angle_max = float(
            self._env.angle_min
            + (self._env.config.lidar_beams - 1) * self._env.angle_increment
        )
        message.time_increment = float(
            self._env.config.control_dt / self._env.config.lidar_beams
        )
        message.scan_time = float(self._env.config.control_dt)
        message.range_min = float(self._env.config.lidar_min_range)
        message.range_max = float(self._env.config.lidar_max_range)
        message.ranges = self._observation["scan"].astype(float).tolist()
        self._scan_publisher.publish(message)

    def _publish_odometry(self, stamp: Any) -> None:
        pose = self._observation["odometry_pose"]
        message = Odometry()
        message.header.stamp = stamp
        message.header.frame_id = "odom"
        message.child_frame_id = "base_link"
        message.pose.pose.position.x = float(pose[0])
        message.pose.pose.position.y = float(pose[1])
        message.pose.pose.orientation = planar_quaternion(float(pose[2]))
        message.pose.covariance[0] = 0.02
        message.pose.covariance[7] = 0.02
        message.pose.covariance[35] = 0.04
        message.twist.twist.linear.x = float(self._command[0])
        message.twist.twist.angular.z = float(self._command[1])
        self._odom_publisher.publish(message)

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = "odom"
        transform.child_frame_id = "base_link"
        transform.transform.translation.x = float(pose[0])
        transform.transform.translation.y = float(pose[1])
        transform.transform.rotation = planar_quaternion(float(pose[2]))
        self._tf_broadcaster.sendTransform(transform)

    def _publish_ground_truth(self, stamp: Any) -> None:
        pose = self._observation["true_pose"]
        message = PoseStamped()
        message.header.stamp = stamp
        message.header.frame_id = "world"
        message.pose.position.x = float(pose[0])
        message.pose.position.y = float(pose[1])
        message.pose.orientation = planar_quaternion(float(pose[2]))
        self._truth_publisher.publish(message)

    def _publish_static_tf(self) -> None:
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "base_link"
        transform.child_frame_id = "laser"
        transform.transform.translation.z = 0.23
        transform.transform.rotation.w = 1.0
        self._static_tf_broadcaster.sendTransform(transform)

    def destroy_node(self) -> bool:
        self._env.close()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MobileSlamBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
