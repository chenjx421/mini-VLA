#!/usr/bin/env bash

set -eo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
workspace="${project_root}/ros2_ws"
output_dir="${1:-${project_root}/outputs/ros_arm_probe_wsl_jazzy}"
bridge_log="${project_root}/outputs/ros_arm_bridge_wsl.log"

source /opt/ros/jazzy/setup.bash
source "${workspace}/install/setup.bash"
set -u
export MUJOCO_GL="${MUJOCO_GL:-egl}"

cd "${workspace}"
ros2 launch embodied_vla_ros arm_bridge.launch.py \
  image_size:=128 \
  rate_hz:=10.0 \
  >"${bridge_log}" 2>&1 &
bridge_pid=$!

cleanup() {
  if ! kill -0 "${bridge_pid}" 2>/dev/null; then
    return
  fi
  kill -INT "${bridge_pid}" 2>/dev/null || true
  for _ in $(seq 1 20); do
    if ! kill -0 "${bridge_pid}" 2>/dev/null; then
      wait "${bridge_pid}" 2>/dev/null || true
      return
    fi
    sleep 0.25
  done
  kill -TERM "${bridge_pid}" 2>/dev/null || true
  wait "${bridge_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 5
ros2 run embodied_vla_ros arm_probe \
  --output-dir "${output_dir}" \
  --timeout 20.0
