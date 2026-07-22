#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    use_sim_time = LaunchConfiguration("use_sim_time")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use simulation clock.",
            ),
            Node(
                package="tb3_cone_perception",
                executable="cone_perception_node",
                name="cone_perception_node",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "front_fov_deg": 180.0,
                    }
                ],
            ),
            Node(
                package="tb3_reference",
                executable="reference_node",
                name="reference_node",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "input_timeout_sec": 0.5,
                        "publish_rate_hz": 10.0,
                    }
                ],
            ),
            Node(
                package="tb3_mpc_controller",
                executable="mpc_controller_node",
                name="mpc_controller_node",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "control_rate_hz": 10.0,
                        "input_timeout_sec": 0.5,
                    }
                ],
            ),
            Node(
                package="tb3_safety",
                executable="safety_node",
                name="safety_node",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time,
                        "publish_rate_hz": 20.0,
                        "max_linear_speed": 0.15,
                        "max_angular_speed": 1.0,
                        "stop_distance": 0.20,
                    }
                ],
            ),
        ]
    )
