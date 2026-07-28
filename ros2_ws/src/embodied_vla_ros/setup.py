from glob import glob

from setuptools import find_packages, setup

package_name = "embodied_vla_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="chenjx421",
    maintainer_email="chenjx421@users.noreply.github.com",
    description="ROS 2 bridges for EmbodiedVLA MuJoCo labs.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "arm_bridge = embodied_vla_ros.arm_bridge:main",
            "arm_probe = embodied_vla_ros.arm_probe:main",
            "mobile_slam_bridge = embodied_vla_ros.mobile_slam_bridge:main",
            "slam_recorder = embodied_vla_ros.slam_recorder:main",
        ],
    },
)
