#!/usr/bin/env python3
"""Convert a camera lane path into an odom-frame MPC reference."""

import copy
import math
from typing import List, Optional, Tuple

from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Float32, String
from tb3_reference.path_utils import (
    crop_polyline,
    heading_yaws,
    remove_near_duplicates,
    resample_polyline,
    smooth_polyline,
)
from tf2_ros import Buffer, TransformException, TransformListener


Point2 = Tuple[float, float]


class CameraReferenceNode(Node):
    """Create a fresh MPC reference from ``/lane_path``."""

    def __init__(self) -> None:
        """Initialize parameters, TF, ROS interfaces, and publish timer."""
        super().__init__('camera_reference_node')

        self.declare_parameter('lane_path_topic', '/lane_path')
        self.declare_parameter('reference_path_topic', '/reference_path')
        self.declare_parameter('reference_speed_topic', '/reference_speed')
        self.declare_parameter('driving_mode_topic', '/driving_mode')

        self.declare_parameter('target_frame', 'odom')
        self.declare_parameter('input_timeout_sec', 0.5)
        self.declare_parameter('transform_timeout_sec', 0.05)
        self.declare_parameter('allow_latest_transform_fallback', True)
        self.declare_parameter('publish_rate_hz', 10.0)

        self.declare_parameter('prepend_robot_origin', True)
        self.declare_parameter('minimum_point_separation_m', 0.01)
        self.declare_parameter('minimum_path_length_m', 0.05)
        self.declare_parameter('maximum_path_length_m', 1.0)
        self.declare_parameter('resample_spacing_m', 0.03)
        self.declare_parameter('smoothing_window', 3)
        self.declare_parameter('reference_speed_mps', 0.03)

        self._load_parameters()

        self.latest_lane_path: Optional[Path] = None
        self.last_lane_path_time: Optional[Time] = None
        self.last_mode: Optional[str] = None
        self.last_warning_times = {}

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.lane_path_sub = self.create_subscription(
            Path,
            self.lane_path_topic,
            self._lane_path_callback,
            10,
        )
        self.reference_path_pub = self.create_publisher(
            Path,
            self.reference_path_topic,
            10,
        )
        self.reference_speed_pub = self.create_publisher(
            Float32,
            self.reference_speed_topic,
            10,
        )
        self.driving_mode_pub = self.create_publisher(
            String,
            self.driving_mode_topic,
            10,
        )

        self.timer = self.create_timer(
            1.0 / self.publish_rate_hz,
            self._timer_callback,
        )

        self.get_logger().info(
            'Camera reference ready: '
            f'{self.lane_path_topic} -> '
            f'{self.reference_path_topic} [{self.target_frame}]'
        )

    def _load_parameters(self) -> None:
        self.lane_path_topic = str(
            self.get_parameter('lane_path_topic').value
        )
        self.reference_path_topic = str(
            self.get_parameter('reference_path_topic').value
        )
        self.reference_speed_topic = str(
            self.get_parameter('reference_speed_topic').value
        )
        self.driving_mode_topic = str(
            self.get_parameter('driving_mode_topic').value
        )

        self.target_frame = str(self.get_parameter('target_frame').value)
        self.input_timeout_sec = float(
            self.get_parameter('input_timeout_sec').value
        )
        self.transform_timeout_sec = float(
            self.get_parameter('transform_timeout_sec').value
        )
        self.allow_latest_transform_fallback = bool(
            self.get_parameter('allow_latest_transform_fallback').value
        )
        self.publish_rate_hz = float(
            self.get_parameter('publish_rate_hz').value
        )

        self.prepend_robot_origin = bool(
            self.get_parameter('prepend_robot_origin').value
        )
        self.minimum_point_separation_m = float(
            self.get_parameter('minimum_point_separation_m').value
        )
        self.minimum_path_length_m = float(
            self.get_parameter('minimum_path_length_m').value
        )
        self.maximum_path_length_m = float(
            self.get_parameter('maximum_path_length_m').value
        )
        self.resample_spacing_m = float(
            self.get_parameter('resample_spacing_m').value
        )
        self.smoothing_window = int(
            self.get_parameter('smoothing_window').value
        )
        self.reference_speed_mps = float(
            self.get_parameter('reference_speed_mps').value
        )

        if not self.target_frame:
            raise ValueError('target_frame must not be empty')
        if self.input_timeout_sec <= 0.0:
            raise ValueError('input_timeout_sec must be positive')
        if self.transform_timeout_sec < 0.0:
            raise ValueError('transform_timeout_sec must be non-negative')
        if self.publish_rate_hz <= 0.0:
            raise ValueError('publish_rate_hz must be positive')
        if self.minimum_point_separation_m < 0.0:
            raise ValueError('minimum_point_separation_m must be non-negative')
        if self.minimum_path_length_m <= 0.0:
            raise ValueError('minimum_path_length_m must be positive')
        if self.maximum_path_length_m < self.minimum_path_length_m:
            raise ValueError(
                'maximum_path_length_m must be >= minimum_path_length_m'
            )
        if self.resample_spacing_m <= 0.0:
            raise ValueError('resample_spacing_m must be positive')
        if self.smoothing_window < 1:
            raise ValueError('smoothing_window must be at least 1')
        if self.smoothing_window % 2 == 0:
            self.smoothing_window += 1
            self.get_logger().warning(
                'smoothing_window must be odd; using '
                f'{self.smoothing_window}'
            )
        if self.reference_speed_mps < 0.0:
            raise ValueError('reference_speed_mps must be non-negative')

    def _lane_path_callback(self, message: Path) -> None:
        self.latest_lane_path = copy.deepcopy(message)
        self.last_lane_path_time = self.get_clock().now()

    def _timer_callback(self) -> None:
        invalid_reason = self._invalid_input_reason()
        if invalid_reason is not None:
            self._publish_stop(invalid_reason)
            return

        try:
            reference_path = self._generate_reference_path(
                self.latest_lane_path
            )
        except TransformException as error:
            self._warn_throttled('transform', f'TF unavailable: {error}')
            self._publish_stop('transform unavailable')
            return
        except ValueError as error:
            self._warn_throttled('path', f'Invalid lane path: {error}')
            self._publish_stop(str(error))
            return

        self.reference_path_pub.publish(reference_path)
        self._publish_speed(self.reference_speed_mps)
        self._publish_mode('LANE')

    def _invalid_input_reason(self) -> Optional[str]:
        if self.latest_lane_path is None:
            return f'waiting for {self.lane_path_topic}'
        if self.last_lane_path_time is None:
            return 'waiting for lane path timestamp'

        age = (
            self.get_clock().now() - self.last_lane_path_time
        ).nanoseconds / 1.0e9

        if age > self.input_timeout_sec:
            return f'lane path stale ({age:.2f}s)'
        if len(self.latest_lane_path.poses) < 2:
            return 'lane path has fewer than two poses'
        return None

    def _generate_reference_path(self, source_path: Path) -> Path:
        source_frame = source_path.header.frame_id.strip()
        if not source_frame:
            raise ValueError('lane path frame_id is empty')

        transform = self._lookup_transform(source_path, source_frame)
        points: List[Point2] = []

        if self.prepend_robot_origin:
            points.append(self._transform_point(0.0, 0.0, transform))

        for source_pose in source_path.poses:
            position = source_pose.pose.position
            if not (
                math.isfinite(position.x) and math.isfinite(position.y)
            ):
                continue
            points.append(
                self._transform_point(
                    float(position.x),
                    float(position.y),
                    transform,
                )
            )

        points = remove_near_duplicates(
            points,
            self.minimum_point_separation_m,
        )
        points = crop_polyline(points, self.maximum_path_length_m)
        points = smooth_polyline(points, self.smoothing_window)
        points = resample_polyline(points, self.resample_spacing_m)

        if len(points) < 2:
            raise ValueError('reference path has fewer than two points')

        length = sum(
            math.hypot(
                points[index][0] - points[index - 1][0],
                points[index][1] - points[index - 1][1],
            )
            for index in range(1, len(points))
        )
        if length < self.minimum_path_length_m:
            raise ValueError(
                f'reference path is too short ({length:.3f}m)'
            )

        return self._build_path_message(points)

    def _lookup_transform(
        self,
        source_path: Path,
        source_frame: str,
    ) -> Optional[TransformStamped]:
        if source_frame == self.target_frame:
            return None

        path_stamp = Time.from_msg(source_path.header.stamp)
        timeout = Duration(seconds=self.transform_timeout_sec)

        try:
            return self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                path_stamp,
                timeout=timeout,
            )
        except TransformException as exact_error:
            if not self.allow_latest_transform_fallback:
                raise exact_error

            self._warn_throttled(
                'latest_transform',
                'Exact-time TF unavailable; using the latest transform',
            )
            return self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                Time(),
                timeout=timeout,
            )

    @staticmethod
    def _transform_point(
        x_value: float,
        y_value: float,
        transform: Optional[TransformStamped],
    ) -> Point2:
        if transform is None:
            return float(x_value), float(y_value)

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        yaw = math.atan2(
            2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
            1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
        )

        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        return (
            float(translation.x + cosine * x_value - sine * y_value),
            float(translation.y + sine * x_value + cosine * y_value),
        )

    def _build_path_message(self, points: List[Point2]) -> Path:
        output = Path()
        output.header.stamp = self.get_clock().now().to_msg()
        output.header.frame_id = self.target_frame

        for point, yaw in zip(points, heading_yaws(points)):
            pose = PoseStamped()
            pose.header = output.header
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = 0.0
            pose.pose.orientation.z = math.sin(0.5 * yaw)
            pose.pose.orientation.w = math.cos(0.5 * yaw)
            output.poses.append(pose)

        return output

    def _publish_stop(self, reason: str) -> None:
        entering_stop = self.last_mode != 'STOP'
        empty_path = Path()
        empty_path.header.stamp = self.get_clock().now().to_msg()
        empty_path.header.frame_id = self.target_frame
        self.reference_path_pub.publish(empty_path)
        self._publish_speed(0.0)
        self._publish_mode('STOP', log_change=False)

        if entering_stop:
            self.get_logger().info(f'Driving mode: STOP ({reason})')

    def _publish_speed(self, speed: float) -> None:
        message = Float32()
        message.data = float(speed)
        self.reference_speed_pub.publish(message)

    def _publish_mode(self, mode: str, log_change: bool = True) -> None:
        message = String()
        message.data = mode
        self.driving_mode_pub.publish(message)

        if mode != self.last_mode:
            if log_change:
                self.get_logger().info(f'Driving mode: {mode}')
            self.last_mode = mode

    def _warn_throttled(
        self,
        key: str,
        message: str,
        period_sec: float = 2.0,
    ) -> None:
        now_seconds = self.get_clock().now().nanoseconds * 1.0e-9
        last_seconds = self.last_warning_times.get(key, -math.inf)
        if now_seconds - last_seconds < period_sec:
            return
        self.last_warning_times[key] = now_seconds
        self.get_logger().warning(message)


def main(args=None) -> None:
    """Run the camera reference generator node."""
    rclpy.init(args=args)
    node = CameraReferenceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
