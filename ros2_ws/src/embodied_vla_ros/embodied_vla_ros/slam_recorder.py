from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Quaternion
from nav_msgs.msg import OccupancyGrid, Odometry
from PIL import Image, ImageDraw
from rclpy.node import Node


class SlamRecorder(Node):
    def __init__(self) -> None:
        super().__init__("embodied_vla_slam_recorder")
        self.map_message: OccupancyGrid | None = None
        self.truth_trajectory: list[tuple[float, float, float, float]] = []
        self.odom_trajectory: list[tuple[float, float, float, float]] = []
        self.slam_trajectory: list[tuple[float, float, float, float]] = []
        self.create_subscription(OccupancyGrid, "/map", self._map_callback, 10)
        self.create_subscription(PoseStamped, "/ground_truth_pose", self._truth_callback, 20)
        self.create_subscription(Odometry, "/odom", self._odom_callback, 20)
        self.create_subscription(
            PoseWithCovarianceStamped,
            "/pose",
            self._slam_callback,
            20,
        )

    def _map_callback(self, message: OccupancyGrid) -> None:
        self.map_message = message

    def _truth_callback(self, message: PoseStamped) -> None:
        self.truth_trajectory.append(_stamped_pose_tuple(message.header.stamp, message.pose))

    def _odom_callback(self, message: Odometry) -> None:
        self.odom_trajectory.append(
            _stamped_pose_tuple(message.header.stamp, message.pose.pose)
        )

    def _slam_callback(self, message: PoseWithCovarianceStamped) -> None:
        self.slam_trajectory.append(
            _stamped_pose_tuple(message.header.stamp, message.pose.pose)
        )

    def save(self, output_dir: Path) -> dict[str, Any]:
        if self.map_message is None:
            raise RuntimeError("no /map message was received")
        output_dir.mkdir(parents=True, exist_ok=True)
        message = self.map_message
        width = int(message.info.width)
        height = int(message.info.height)
        occupancy = np.asarray(message.data, dtype=np.int16).reshape(height, width)
        image_array = np.empty((height, width, 3), dtype=np.uint8)
        image_array[occupancy < 0] = (135, 140, 145)
        image_array[occupancy == 0] = (245, 246, 244)
        occupied_intensity = np.uint8(
            np.clip(240 - np.maximum(occupancy, 0) * 2.15, 20, 240)
        )
        known = occupancy > 0
        image_array[known] = occupied_intensity[known, None]
        image_array = np.flipud(image_array)
        map_image = Image.fromarray(image_array, mode="RGB")
        map_image.save(output_dir / "occupancy_map.png")

        scale = 4
        header_height = 64
        canvas = Image.new(
            "RGB",
            (width * scale, height * scale + header_height),
            color=(22, 24, 28),
        )
        canvas.paste(
            map_image.resize(
                (width * scale, height * scale),
                resample=Image.Resampling.NEAREST,
            ),
            (0, header_height),
        )
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 6), "MuJoCo + ROS 2 Jazzy + slam_toolbox", fill=(245, 245, 245))
        draw.text(
            (8, 23),
            "ground truth  |  wheel odometry  |  SLAM corrected pose",
            fill=(215, 215, 215),
        )
        legend = [
            ("ground truth", (220, 52, 52)),
            ("odometry", (245, 155, 35)),
            ("SLAM", (28, 170, 205)),
        ]
        x_offset = 8
        for label, color in legend:
            draw.line((x_offset, 48, x_offset + 18, 48), fill=color, width=3)
            draw.text((x_offset + 23, 42), label, fill=(230, 230, 230))
            x_offset += 105

        origin_x = float(message.info.origin.position.x)
        origin_y = float(message.info.origin.position.y)
        resolution = float(message.info.resolution)
        self._draw_trajectory(
            draw,
            self.truth_trajectory,
            color=(220, 52, 52),
            origin_x=origin_x,
            origin_y=origin_y,
            resolution=resolution,
            height=height,
            scale=scale,
            y_offset=header_height,
        )
        self._draw_trajectory(
            draw,
            self.odom_trajectory,
            color=(245, 155, 35),
            origin_x=origin_x,
            origin_y=origin_y,
            resolution=resolution,
            height=height,
            scale=scale,
            y_offset=header_height,
        )
        self._draw_trajectory(
            draw,
            self.slam_trajectory,
            color=(28, 170, 205),
            origin_x=origin_x,
            origin_y=origin_y,
            resolution=resolution,
            height=height,
            scale=scale,
            y_offset=header_height,
        )
        canvas.save(output_dir / "slam_trajectory_comparison.png")

        truth = np.asarray(self.truth_trajectory, dtype=np.float32)
        odometry = np.asarray(self.odom_trajectory, dtype=np.float32)
        slam = np.asarray(self.slam_trajectory, dtype=np.float32)
        np.savez_compressed(
            output_dir / "slam_capture.npz",
            occupancy=occupancy.astype(np.int8),
            truth=truth,
            odometry=odometry,
            slam=slam,
            resolution=np.float32(resolution),
            origin=np.asarray([origin_x, origin_y], dtype=np.float32),
        )
        summary = {
            "map_width": width,
            "map_height": height,
            "resolution": resolution,
            "truth_samples": len(truth),
            "odometry_samples": len(odometry),
            "slam_samples": len(slam),
            "final_odometry_position_error": _final_position_error(truth, odometry),
            "final_slam_position_error": _final_position_error(truth, slam),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        return summary

    @staticmethod
    def _draw_trajectory(
        draw: ImageDraw.ImageDraw,
        trajectory: list[tuple[float, float, float, float]],
        *,
        color: tuple[int, int, int],
        origin_x: float,
        origin_y: float,
        resolution: float,
        height: int,
        scale: int,
        y_offset: int,
    ) -> None:
        if len(trajectory) < 2:
            return
        pixels = []
        for _, x, y, _ in trajectory:
            pixel_x = (x - origin_x) / resolution * scale
            pixel_y = (height - 1 - (y - origin_y) / resolution) * scale + y_offset
            pixels.append((pixel_x, pixel_y))
        draw.line(pixels, fill=color, width=3)


def _stamped_pose_tuple(stamp: Any, pose: Any) -> tuple[float, float, float, float]:
    return (
        float(stamp.sec) + float(stamp.nanosec) * 1e-9,
        float(pose.position.x),
        float(pose.position.y),
        _yaw_from_quaternion(pose.orientation),
    )


def _yaw_from_quaternion(quaternion: Quaternion) -> float:
    sin_yaw = 2.0 * (
        float(quaternion.w) * float(quaternion.z)
        + float(quaternion.x) * float(quaternion.y)
    )
    cos_yaw = 1.0 - 2.0 * (
        float(quaternion.y) ** 2 + float(quaternion.z) ** 2
    )
    return float(np.arctan2(sin_yaw, cos_yaw))


def _final_position_error(
    truth: np.ndarray,
    estimate: np.ndarray,
) -> float | None:
    if len(truth) == 0 or len(estimate) == 0:
        return None
    estimate_final = estimate[-1]
    truth_index = int(np.argmin(np.abs(truth[:, 0] - estimate_final[0])))
    return float(np.linalg.norm(truth[truth_index, 1:3] - estimate_final[1:3]))


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Record ROS SLAM map and trajectories.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=60.0)
    parsed, ros_args = parser.parse_known_args(args)
    rclpy.init(args=ros_args)
    node = SlamRecorder()
    deadline = time.monotonic() + parsed.duration
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        summary = node.save(parsed.output_dir)
        node.get_logger().info(json.dumps(summary, ensure_ascii=True))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
