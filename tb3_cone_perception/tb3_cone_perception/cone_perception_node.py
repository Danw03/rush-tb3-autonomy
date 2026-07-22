#!/usr/bin/env python3

import math
from typing import Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray


class ConePerceptionNode(Node):
    """
    Temporary cone-perception pipeline.

    Current placeholder behavior:
      /scan -> nearest valid LiDAR point -> /cone_path
            -> simple numeric features -> /cone_features

    Team TODO:
      1. Convert scan to 2-D points.
      2. Cluster points and reject non-cone objects.
      3. Split cones into left/right boundaries.
      4. Generate a center path.
      5. Replace the placeholder feature vector with final features.
    """

    FEATURE_LAYOUT = [
        "path_valid",
        "minimum_distance_m",
        "nearest_angle_rad",
        "nearest_x_m",
        "nearest_y_m",
        "valid_scan_ratio",
        "valid_point_count",
    ]

    def __init__(self) -> None:
        super().__init__("cone_perception_node")

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("path_topic", "/cone_path")
        self.declare_parameter("features_topic", "/cone_features")
        self.declare_parameter("front_fov_deg", 180.0)

        scan_topic = self.get_parameter("scan_topic").value
        path_topic = self.get_parameter("path_topic").value
        features_topic = self.get_parameter("features_topic").value

        self.front_fov_rad = math.radians(
            float(self.get_parameter("front_fov_deg").value)
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            qos_profile_sensor_data,
        )

        self.path_pub = self.create_publisher(Path, path_topic, 10)
        self.features_pub = self.create_publisher(
            Float32MultiArray,
            features_topic,
            10,
        )

        self.get_logger().info(
            f"Started: {scan_topic} -> {path_topic}, {features_topic}"
        )
        self.get_logger().info(
            "cone_features layout: " + ", ".join(self.FEATURE_LAYOUT)
        )

    def scan_callback(self, msg: LaserScan) -> None:
        points, ranges, angles, valid_ratio = self.preprocess_scan(msg)

        if len(ranges) == 0:
            self.publish_empty_result(msg, valid_ratio)
            return

        # PLACEHOLDER:
        # Publish the nearest valid LiDAR point for end-to-end testing.
        # Replace with clustering -> cone detection -> left/right split ->
        # center-path generation.
        nearest_index = int(np.argmin(ranges))
        minimum_distance = float(ranges[nearest_index])
        nearest_angle = float(angles[nearest_index])
        nearest_x = float(points[nearest_index, 0])
        nearest_y = float(points[nearest_index, 1])

        path_msg = self.build_placeholder_path(
            msg=msg,
            x=nearest_x,
            y=nearest_y,
        )

        feature_msg = Float32MultiArray()
        feature_msg.data = [
            1.0,
            minimum_distance,
            nearest_angle,
            nearest_x,
            nearest_y,
            float(valid_ratio),
            float(len(ranges)),
        ]

        self.path_pub.publish(path_msg)
        self.features_pub.publish(feature_msg)

    def preprocess_scan(
        self,
        msg: LaserScan,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Convert valid LaserScan samples into robot-frame 2-D points."""
        ranges_all = np.asarray(msg.ranges, dtype=np.float32)

        if ranges_all.size == 0:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
                0.0,
            )

        indices = np.arange(ranges_all.size, dtype=np.float32)
        angles_all = msg.angle_min + indices * msg.angle_increment

        valid = np.isfinite(ranges_all)
        valid &= ranges_all >= float(msg.range_min)
        valid &= ranges_all <= float(msg.range_max)

        wrapped_angles = np.arctan2(
            np.sin(angles_all),
            np.cos(angles_all),
        )
        valid &= np.abs(wrapped_angles) <= self.front_fov_rad / 2.0

        valid_ratio = float(np.count_nonzero(valid)) / float(ranges_all.size)

        ranges = ranges_all[valid]
        angles = wrapped_angles[valid]

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        points = np.column_stack((x, y)).astype(np.float32)

        return points, ranges, angles, valid_ratio

    def build_placeholder_path(
        self,
        msg: LaserScan,
        x: float,
        y: float,
    ) -> Path:
        """Build a one-point Path for end-to-end topic testing."""
        path_msg = Path()
        path_msg.header.stamp = msg.header.stamp
        path_msg.header.frame_id = msg.header.frame_id

        pose = PoseStamped()
        pose.header = path_msg.header
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0

        path_msg.poses.append(pose)
        return path_msg

    def publish_empty_result(
        self,
        msg: LaserScan,
        valid_ratio: float,
    ) -> None:
        path_msg = Path()
        path_msg.header.stamp = msg.header.stamp
        path_msg.header.frame_id = msg.header.frame_id

        feature_msg = Float32MultiArray()
        feature_msg.data = [
            0.0,
            float("inf"),
            0.0,
            0.0,
            0.0,
            float(valid_ratio),
            0.0,
        ]

        self.path_pub.publish(path_msg)
        self.features_pub.publish(feature_msg)

    def cluster_lidar_points(self, points: np.ndarray):
        """TODO: DBSCAN or distance-gap clustering."""
        raise NotImplementedError

    def detect_cones(self, clusters):
        """TODO: Reject clusters that do not match expected cone geometry."""
        raise NotImplementedError

    def split_left_right(self, cones):
        """TODO: Classify or connect left/right cone boundaries."""
        raise NotImplementedError

    def generate_center_path(self, msg: LaserScan, left_cones, right_cones):
        """TODO: Generate and smooth the center path."""
        raise NotImplementedError

    def extract_features(self, cones, path):
        """TODO: Return features used by the Reference node classifier."""
        raise NotImplementedError


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ConePerceptionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
