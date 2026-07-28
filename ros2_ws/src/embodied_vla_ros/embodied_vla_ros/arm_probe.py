from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from PIL import Image as PillowImage
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

from embodied_vla.image_utils import depth_to_rgb


class ArmProbe(Node):
    def __init__(self) -> None:
        super().__init__("embodied_vla_arm_probe")
        self.joint_state: JointState | None = None
        self.rgb: Image | None = None
        self.depth: Image | None = None
        self.camera_info: CameraInfo | None = None
        self.task: String | None = None
        self.metadata: String | None = None
        self.tf_frames: set[tuple[str, str]] = set()
        self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_callback,
            10,
        )
        self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self._rgb_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            "/camera/depth/image_raw",
            self._depth_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            "/camera/color/camera_info",
            self._camera_info_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(String, "/vla/task", self._task_callback, 10)
        self.create_subscription(
            String,
            "/vla/task_metadata",
            self._metadata_callback,
            10,
        )
        self.create_subscription(TFMessage, "/tf", self._tf_callback, 20)

    @property
    def complete(self) -> bool:
        return all(
            message is not None
            for message in (
                self.joint_state,
                self.rgb,
                self.depth,
                self.camera_info,
                self.task,
                self.metadata,
            )
        ) and bool(self.tf_frames)

    def _joint_callback(self, message: JointState) -> None:
        self.joint_state = message

    def _rgb_callback(self, message: Image) -> None:
        self.rgb = message

    def _depth_callback(self, message: Image) -> None:
        self.depth = message

    def _camera_info_callback(self, message: CameraInfo) -> None:
        self.camera_info = message

    def _task_callback(self, message: String) -> None:
        self.task = message

    def _metadata_callback(self, message: String) -> None:
        self.metadata = message

    def _tf_callback(self, message: TFMessage) -> None:
        for transform in message.transforms:
            self.tf_frames.add(
                (transform.header.frame_id, transform.child_frame_id)
            )

    def save(self, output_dir: Path) -> dict[str, Any]:
        if not self.complete:
            raise RuntimeError("arm probe did not receive every required topic")
        assert self.joint_state is not None
        assert self.rgb is not None
        assert self.depth is not None
        assert self.camera_info is not None
        assert self.task is not None
        assert self.metadata is not None
        output_dir.mkdir(parents=True, exist_ok=True)

        rgb = np.frombuffer(self.rgb.data, dtype=np.uint8).reshape(
            self.rgb.height,
            self.rgb.width,
            3,
        )
        depth = np.frombuffer(self.depth.data, dtype=np.float32).reshape(
            self.depth.height,
            self.depth.width,
        )
        PillowImage.fromarray(rgb, mode="RGB").save(output_dir / "arm_rgb.png")
        _save_depth_visualization(depth, output_dir / "arm_depth.png")

        metadata = json.loads(self.metadata.data)
        checks = {
            "six_joint_names": len(self.joint_state.name) == 6,
            "six_joint_positions": len(self.joint_state.position) == 6,
            "finite_joint_positions": bool(
                np.isfinite(np.asarray(self.joint_state.position)).all()
            ),
            "rgb8_encoding": self.rgb.encoding == "rgb8",
            "depth_32fc1_encoding": self.depth.encoding == "32FC1",
            "matching_image_shape": rgb.shape[:2] == depth.shape,
            "finite_positive_depth": bool(
                np.isfinite(depth).all() and np.all(depth > 0.0)
            ),
            "positive_camera_focal_length": bool(
                self.camera_info.k[0] > 0.0 and self.camera_info.k[4] > 0.0
            ),
            "nonempty_task": bool(self.task.data.strip()),
            "valid_task_metadata": (
                metadata.get("target_color") in {"red", "green", "blue"}
                and metadata.get("goal_side") in {"left", "right"}
            ),
            "camera_tf_present": (
                "world",
                "camera_color_optical_frame",
            )
            in self.tf_frames,
        }
        summary = {
            "all_checks_passed": all(checks.values()),
            "checks": checks,
            "rgb_shape": list(rgb.shape),
            "depth_shape": list(depth.shape),
            "depth_min": float(depth.min()),
            "depth_max": float(depth.max()),
            "task": self.task.data,
            "metadata": metadata,
            "joint_names": list(self.joint_state.name),
            "tf_frame_count": len(self.tf_frames),
            "tf_frames": [list(frame_pair) for frame_pair in sorted(self.tf_frames)],
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        if not summary["all_checks_passed"]:
            failed = [name for name, passed in checks.items() if not passed]
            raise RuntimeError(f"arm probe checks failed: {', '.join(failed)}")
        return summary


def _save_depth_visualization(depth: np.ndarray, path: Path) -> None:
    PillowImage.fromarray(depth_to_rgb(depth), mode="RGB").save(path)


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate ROS arm topics and save RGB-D evidence."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=10.0)
    parsed, ros_args = parser.parse_known_args(args)
    rclpy.init(args=ros_args)
    node = ArmProbe()
    deadline = time.monotonic() + parsed.timeout
    try:
        while time.monotonic() < deadline and not node.complete:
            rclpy.spin_once(node, timeout_sec=0.1)
        summary = node.save(parsed.output_dir)
        node.get_logger().info(json.dumps(summary, ensure_ascii=True))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
