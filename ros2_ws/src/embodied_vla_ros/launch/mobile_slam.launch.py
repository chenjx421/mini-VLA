from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    slam_parameters = PathJoinSubstitution(
        [FindPackageShare("embodied_vla_ros"), "config", "slam_toolbox.yaml"]
    )
    slam_launch = PathJoinSubstitution(
        [FindPackageShare("slam_toolbox"), "launch", "online_async_launch.py"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("autopilot", default_value="true"),
            DeclareLaunchArgument("seed", default_value="7"),
            Node(
                package="embodied_vla_ros",
                executable="mobile_slam_bridge",
                name="mobile_slam_bridge",
                output="screen",
                parameters=[
                    {
                        "autopilot": ParameterValue(
                            LaunchConfiguration("autopilot"),
                            value_type=bool,
                        ),
                        "seed": ParameterValue(
                            LaunchConfiguration("seed"),
                            value_type=int,
                        ),
                    }
                ],
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(slam_launch),
                launch_arguments={
                    "slam_params_file": slam_parameters,
                    "use_sim_time": "false",
                }.items(),
            ),
        ]
    )
