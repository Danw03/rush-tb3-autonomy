#!/usr/bin/env python3

import math
from pathlib import Path
from typing import Any

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


def convert_float(value: Any) -> float:
    """Convert YAML values, including inf and nan strings, to float."""

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized == '...':
            raise ValueError(
                'The YAML contains "...". '
                'Capture it again using ros2 topic echo --full-length.'
            )

        if normalized in {'inf', '+inf', '.inf', '+.inf'}:
            return math.inf

        if normalized in {'-inf', '-.inf'}:
            return -math.inf

        if normalized in {'nan', '.nan'}:
            return math.nan

    return float(value)


def load_scan_yaml(path: Path) -> dict[str, Any]:
    """Load the first LaserScan document from a YAML file."""

    with path.open('r', encoding='utf-8') as file:
        for document in yaml.safe_load_all(file):
            if isinstance(document, dict) and 'ranges' in document:
                return document

    raise ValueError(
        f'No LaserScan message found in: {path}'
    )


class ScanSamplePublisher(Node):
    """Publish one LaserScan YAML sample repeatedly."""

    def __init__(self) -> None:
        super().__init__('scan_sample_publisher')

        self.declare_parameter('file_name', 'sample_lidar.yaml')
        self.declare_parameter('output_topic', '/scan')
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('frame_id_override', '')

        file_name = str(self.get_parameter('file_name').value)
        output_topic = str(self.get_parameter('output_topic').value)
        rate_hz = float(self.get_parameter('rate_hz').value)
        frame_id_override = str(
            self.get_parameter('frame_id_override').value
        )

        if rate_hz <= 0.0:
            raise ValueError('rate_hz must be greater than zero.')

        self.yaml_path = self.resolve_yaml_path(file_name)
        source = load_scan_yaml(self.yaml_path)

        self.message = self.create_scan_message(
            source,
            frame_id_override,
        )

        self.publisher = self.create_publisher(
            LaserScan,
            output_topic,
            qos_profile_sensor_data,
        )

        self.timer = self.create_timer(
            1.0 / rate_hz,
            self.publish_scan,
        )

        valid_count = sum(
            1
            for distance in self.message.ranges
            if (
                math.isfinite(distance)
                and self.message.range_min
                <= distance
                <= self.message.range_max
            )
        )

        zero_count = sum(
            distance == 0.0
            for distance in self.message.ranges
        )

        angle_span = math.degrees(
            self.message.angle_max
            - self.message.angle_min
        )

        self.get_logger().info(
            f'LaserScan file: {self.yaml_path}'
        )
        self.get_logger().info(
            f'Publishing {len(self.message.ranges)} points '
            f'to {output_topic} at {rate_hz:.1f} Hz'
        )
        self.get_logger().info(
            f'frame_id={self.message.header.frame_id}, '
            f'angle_span={angle_span:.2f} deg, '
            f'valid={valid_count}, zeros={zero_count}'
        )

    @staticmethod
    def resolve_yaml_path(file_name: str) -> Path:
        """Resolve an absolute path or package data file."""

        supplied_path = Path(file_name).expanduser()

        if supplied_path.is_absolute():
            yaml_path = supplied_path
        else:
            package_share = Path(
                get_package_share_directory('tb3_test_tools')
            )

            yaml_path = (
                package_share
                / 'data'
                / 'lidar'
                / supplied_path
            )

        if not yaml_path.is_file():
            raise FileNotFoundError(
                f'LaserScan sample not found: {yaml_path}'
            )

        return yaml_path

    @staticmethod
    def create_scan_message(
        source: dict[str, Any],
        frame_id_override: str,
    ) -> LaserScan:
        message = LaserScan()

        source_header = source.get('header', {})

        original_frame_id = str(
            source_header.get('frame_id', 'base_scan')
        )

        message.header.frame_id = (
            frame_id_override
            if frame_id_override
            else original_frame_id
        )

        message.angle_min = convert_float(source['angle_min'])
        message.angle_max = convert_float(source['angle_max'])
        message.angle_increment = convert_float(
            source['angle_increment']
        )
        message.time_increment = convert_float(
            source.get('time_increment', 0.0)
        )
        message.scan_time = convert_float(
            source.get('scan_time', 0.1)
        )
        message.range_min = convert_float(source['range_min'])
        message.range_max = convert_float(source['range_max'])

        message.ranges = [
            convert_float(value)
            for value in source.get('ranges', [])
        ]

        message.intensities = [
            convert_float(value)
            for value in source.get('intensities', [])
        ]

        if not message.ranges:
            raise ValueError('The ranges array is empty.')

        return message

    def publish_scan(self) -> None:
        # 저장 당시 stamp 대신 현재 시간을 사용한다.
        self.message.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None

    try:
        node = ScanSamplePublisher()
        rclpy.spin(node)

    except (
        FileNotFoundError,
        KeyError,
        RuntimeError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        print(f'[scan_sample_publisher] ERROR: {error}')

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()