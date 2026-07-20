#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class ConePerceptionNode(Node):

    def __init__(self) -> None:
        super().__init__('cone_perception_node')

        self.closest_distance = None
        self.scan_received = False

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data,
        )

        # /scan Topic 10Hz print 5 Hz --> Feasible
        self.print_timer = self.create_timer(
            0.2,
            self.print_closest_distance,
        )

        self.get_logger().info('Cone perception node started.')
        self.get_logger().info('Waiting for /scan...')

    def scan_callback(self, msg: LaserScan) -> None:
        self.scan_received = True

        valid_ranges = [
            distance
            for distance in msg.ranges
            if (
                math.isfinite(distance)
                and msg.range_min <= distance <= msg.range_max
            )
        ]

        if not valid_ranges:
            self.closest_distance = None
            return

        self.closest_distance = min(valid_ranges)

    def print_closest_distance(self) -> None:
        if not self.scan_received:
            self.get_logger().info('No /scan message received yet.')
            return

        if self.closest_distance is None:
            self.get_logger().warning(
                'No valid LiDAR measurement found.'
            )
            return

        self.get_logger().info(
            f'Closest distance: {self.closest_distance:.3f} m'
        )


def main(args=None) -> None:
    rclpy.init(args=args)

    node = ConePerceptionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()