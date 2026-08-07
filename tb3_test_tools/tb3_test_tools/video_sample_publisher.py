#!/usr/bin/env python3
"""Publish a sample video as a ROS camera image stream."""

import math
from pathlib import Path

import cv2
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class VideoSamplePublisher(Node):
    """Publish frames from one video as ``sensor_msgs/Image`` messages."""

    def __init__(self) -> None:
        super().__init__('video_sample_publisher')

        self.declare_parameter('file_name', 'sample_camera.mp4')
        self.declare_parameter('output_topic', '/camera/image_raw')
        self.declare_parameter('frame_id', 'camera_link')
        self.declare_parameter('rate_hz', 0.0)
        self.declare_parameter('loop', True)

        file_name = str(self.get_parameter('file_name').value)
        output_topic = str(self.get_parameter('output_topic').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        requested_rate_hz = float(self.get_parameter('rate_hz').value)
        self.loop = bool(self.get_parameter('loop').value)

        if requested_rate_hz < 0.0:
            raise ValueError('rate_hz must be zero or greater.')

        self.video_path = self.resolve_video_path(file_name)
        self.capture = cv2.VideoCapture(str(self.video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f'Failed to open video: {self.video_path}')

        source_rate_hz = float(self.capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(source_rate_hz) or source_rate_hz <= 0.0:
            source_rate_hz = 30.0
            self.get_logger().warning(
                'Video FPS is unavailable; using 30.0 Hz.'
            )

        self.rate_hz = (
            requested_rate_hz if requested_rate_hz > 0.0 else source_rate_hz
        )
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(
            Image,
            output_topic,
            qos_profile_sensor_data,
        )
        self.timer = self.create_timer(
            1.0 / self.rate_hz,
            self.publish_next_frame,
        )

        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.get_logger().info(f'Video file: {self.video_path}')
        self.get_logger().info(
            f'Publishing {width}x{height}, {frame_count} frames '
            f'to {output_topic} at {self.rate_hz:.2f} Hz '
            f'(loop={self.loop})'
        )

    @staticmethod
    def resolve_video_path(file_name: str) -> Path:
        """Resolve an absolute path or a package camera-data file."""
        supplied_path = Path(file_name).expanduser()
        if supplied_path.is_absolute():
            video_path = supplied_path
        else:
            package_share = Path(
                get_package_share_directory('tb3_test_tools')
            )
            video_path = package_share / 'data' / 'camera' / supplied_path

        if not video_path.is_file():
            raise FileNotFoundError(
                f'Video sample not found: {video_path}'
            )
        return video_path

    def publish_next_frame(self) -> None:
        """Read and publish the next frame, rewinding at end of file."""
        success, frame = self.capture.read()

        if not success and self.loop:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
            success, frame = self.capture.read()

        if not success:
            self.timer.cancel()
            if self.loop:
                self.get_logger().error('Failed to rewind and read the video.')
            else:
                self.get_logger().info('Reached the end of the video.')
            return

        message = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        self.publisher.publish(message)

    def destroy_node(self) -> bool:
        """Release the video file before destroying the ROS node."""
        if hasattr(self, 'capture'):
            self.capture.release()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None

    try:
        node = VideoSamplePublisher()
        rclpy.spin(node)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f'[video_sample_publisher] ERROR: {error}')
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()