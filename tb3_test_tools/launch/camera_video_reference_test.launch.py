"""Run the offline camera-video-to-reference test pipeline."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    """Create the video, lane-perception, and reference test pipeline."""
    video_file = LaunchConfiguration('video_file')
    rate_hz = LaunchConfiguration('rate_hz')
    loop = LaunchConfiguration('loop')
    publish_test_tf = LaunchConfiguration('publish_test_tf')

    reference_share = get_package_share_directory('tb3_reference')
    reference_parameters = os.path.join(
        reference_share,
        'config',
        'camera_reference.yaml',
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'video_file',
                default_value='sample_camera.mp4',
                description=(
                    'File in tb3_test_tools/data/camera or an absolute path.'
                ),
            ),
            DeclareLaunchArgument(
                'rate_hz',
                default_value='10.0',
                description='Publish rate; zero uses the video FPS.',
            ),
            DeclareLaunchArgument(
                'loop',
                default_value='true',
                description='Restart the video when it reaches the end.',
            ),
            DeclareLaunchArgument(
                'publish_test_tf',
                default_value='true',
                description=(
                    'Publish identity odom -> base_footprint TF for an '
                    'offline test. Set false when robot bringup is running.'
                ),
            ),
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name='camera_test_static_tf',
                output='screen',
                condition=IfCondition(publish_test_tf),
                arguments=[
                    '--x', '0',
                    '--y', '0',
                    '--z', '0',
                    '--yaw', '0',
                    '--pitch', '0',
                    '--roll', '0',
                    '--frame-id', 'odom',
                    '--child-frame-id', 'base_footprint',
                ],
            ),
            Node(
                package='tb3_test_tools',
                executable='video_sample_publisher',
                name='video_sample_publisher',
                output='screen',
                parameters=[
                    {
                        'file_name': video_file,
                        'output_topic': '/camera/image_raw',
                        'rate_hz': ParameterValue(rate_hz, value_type=float),
                        'loop': ParameterValue(loop, value_type=bool),
                    }
                ],
            ),
            Node(
                package='tb3_lane_perception',
                executable='lane_perception_node',
                name='lane_perception_node',
                output='screen',
                parameters=[{'image_topic': '/camera/image_raw'}],
            ),
            Node(
                package='tb3_reference',
                executable='camera_reference_node',
                name='camera_reference_node',
                output='screen',
                parameters=[reference_parameters],
            ),
        ]
    )