FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive
ENV MUJOCO_GL=egl

RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 \
    libgl1 \
    python3-colcon-common-extensions \
    python3-pip \
    ros-jazzy-slam-toolbox \
    ros-jazzy-tf2-ros \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY pyproject.toml LICENSE README.md THIRD_PARTY.md ./
COPY embodied_vla ./embodied_vla

RUN python3 -m pip install --break-system-packages \
    gymnasium \
    imageio \
    mujoco \
    numpy \
    Pillow \
    PyYAML \
    && python3 -m pip install --break-system-packages --no-deps -e .

COPY ros2_ws ./ros2_ws
RUN /bin/bash -c \
    "source /opt/ros/jazzy/setup.bash && cd /workspace/ros2_ws && colcon build --symlink-install"

COPY docker/ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["ros2", "launch", "embodied_vla_ros", "mobile_slam.launch.py"]
