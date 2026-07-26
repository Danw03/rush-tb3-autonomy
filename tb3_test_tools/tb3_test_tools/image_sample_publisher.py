#!/usr/bin/env python3

from pathlib import Path

import cv2
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class ImageSamplePublisher(Node):
    """Publish one image file repeatedly as sensor_msgs/Image."""

    def __init__(self) -> None:
        super().__init__('image_sample_publisher')

        self.declare_parameter('file_name', 'sample_camera.jpg')
        self.declare_parameter('output_topic', '/camera/image_raw')
        self.declare_parameter('frame_id', 'camera_link')
        self.declare_parameter('rate_hz', 10.0)

        file_name = str(self.get_parameter('file_name').value)
        output_topic = str(self.get_parameter('output_topic').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        rate_hz = float(self.get_parameter('rate_hz').value)

        if rate_hz <= 0.0:
            raise ValueError('rate_hz must be greater than zero.')

        self.image_path = self.resolve_image_path(file_name)

        self.cv_image = cv2.imread(
            str(self.image_path),
            cv2.IMREAD_COLOR,
        )

        if self.cv_image is None:
            raise RuntimeError(
                f'Failed to load image: {self.image_path}'
            )

        self.bridge = CvBridge()

        self.publisher = self.create_publisher(
            Image,
            output_topic,
            qos_profile_sensor_data,
        )

        self.timer = self.create_timer(
            1.0 / rate_hz,
            self.publish_image,
        )

        height, width = self.cv_image.shape[:2]

        self.get_logger().info(
            f'Image file: {self.image_path}'
        )
        self.get_logger().info(
            f'Publishing {width}x{height} BGR image '
            f'to {output_topic} at {rate_hz:.1f} Hz'
        )

    @staticmethod
    def resolve_image_path(file_name: str) -> Path:
        """Resolve an absolute path or a package data file."""

        supplied_path = Path(file_name).expanduser()

        if supplied_path.is_absolute():
            image_path = supplied_path
        else:
            package_share = Path(
                get_package_share_directory('tb3_test_tools')
            )

            image_path = (
                package_share
                / 'data'
                / 'camera'
                / supplied_path
            )

        if not image_path.is_file():
            raise FileNotFoundError(
                f'Image sample not found: {image_path}'
            )

        return image_path

    def publish_image(self) -> None:
        message = self.bridge.cv2_to_imgmsg(
            self.cv_image,
            encoding='bgr8',
        )

        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id

        self.publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None

    try:
        node = ImageSamplePublisher()
        rclpy.spin(node)

    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f'[image_sample_publisher] ERROR: {error}')

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()