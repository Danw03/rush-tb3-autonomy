"""Launch the cone reference generator with its default parameters."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Create the cone reference node launch description."""
    package_share = get_package_share_directory('tb3_reference')
    parameters = os.path.join(
        package_share,
        'config',
        'cone_reference.yaml',
    )

    return LaunchDescription(
        [
            Node(
                package='tb3_reference',
                executable='reference_generator_node',
                name='reference_node',
                output='screen',
                parameters=[parameters],
            )
        ]
    )
