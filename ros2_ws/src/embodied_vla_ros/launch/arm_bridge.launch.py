from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("image_size", default_value="128"),
            DeclareLaunchArgument("seed", default_value="42"),
            DeclareLaunchArgument("grasp_mode", default_value="contact_assisted"),
            DeclareLaunchArgument("rate_hz", default_value="10.0"),
            DeclareLaunchArgument("auto_reset", default_value="true"),
            Node(
                package="embodied_vla_ros",
                executable="arm_bridge",
                name="arm_bridge",
                output="screen",
                parameters=[
                    {
                        "image_size": ParameterValue(
                            LaunchConfiguration("image_size"),
                            value_type=int,
                        ),
                        "seed": ParameterValue(
                            LaunchConfiguration("seed"),
                            value_type=int,
                        ),
                        "grasp_mode": LaunchConfiguration("grasp_mode"),
                        "rate_hz": ParameterValue(
                            LaunchConfiguration("rate_hz"),
                            value_type=float,
                        ),
                        "auto_reset": ParameterValue(
                            LaunchConfiguration("auto_reset"),
                            value_type=bool,
                        ),
                    }
                ],
            ),
        ]
    )
