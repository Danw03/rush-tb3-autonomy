#!/usr/bin/env python3

import math
import collections
from collections import deque

import cv2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray


class LanePerceptionNode(Node):
    """
    Camera-based lane perception skeleton.

    Input
    -----
    /camera/image_raw : sensor_msgs/msg/Image

    Output
    ------
    /lane_features : std_msgs/msg/Float32MultiArray

        data[0] = lane_detected
                  1.0: detected
                  0.0: not detected

        data[1] = normalized lateral error
                  positive: lane is to robot's left
                  negative: lane is to robot's right

        data[2] = heading error [rad]
                  positive: lane direction turns left
                  negative: lane direction turns right

        data[3] = detected contour area / ROI area

    /lane_path : nav_msgs/msg/Path

    /lane_debug_image : sensor_msgs/msg/Image
    """

    def __init__(self) -> None:
        super().__init__('lane_perception_node')

        # Topic parameters
        self.declare_parameter(
            'image_topic',
            '/camera/image_raw',
        )
        self.declare_parameter(
            'features_topic',
            '/lane_features',
        )
        self.declare_parameter(
            'path_topic',
            '/lane_path',
        )
        self.declare_parameter(
            'debug_image_topic',
            '/lane_debug_image',
        )

        # Processing parameters
        self.declare_parameter(
            'frame_id',
            'base_link',
        )
        self.declare_parameter(
            'roi_start_ratio',
            0.55,
        )
        self.declare_parameter(
            'min_contour_area',
            300.0,
        )

        # Temporary pixel-to-path conversion parameters
        self.declare_parameter(
            'path_length_m',
            1.0,
        )
        self.declare_parameter(
            'path_point_count',
            10,
        )
        self.declare_parameter(
            'max_lateral_offset_m',
            0.25,
        )

        image_topic = str(
            self.get_parameter('image_topic').value
        )
        features_topic = str(
            self.get_parameter('features_topic').value
        )
        path_topic = str(
            self.get_parameter('path_topic').value
        )
        debug_image_topic = str(
            self.get_parameter('debug_image_topic').value
        )

        self.frame_id = str(
            self.get_parameter('frame_id').value
        )
        self.roi_start_ratio = float(
            self.get_parameter('roi_start_ratio').value
        )
        self.min_contour_area = float(
            self.get_parameter('min_contour_area').value
        )
        self.path_length_m = float(
            self.get_parameter('path_length_m').value
        )
        self.path_point_count = int(
            self.get_parameter('path_point_count').value
        )
        self.max_lateral_offset_m = float(
            self.get_parameter(
                'max_lateral_offset_m'
            ).value
        )

        self.bridge = CvBridge()

        self.image_subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.features_publisher = self.create_publisher(
            Float32MultiArray,
            features_topic,
            10,
        )

        self.path_publisher = self.create_publisher(
            Path,
            path_topic,
            10,
        )

        self.debug_image_publisher = self.create_publisher(
            Image,
            debug_image_topic,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f'Subscribing to: {image_topic}'
        )
        self.get_logger().info(
            f'Publishing features: {features_topic}'
        )
        self.get_logger().info(
            f'Publishing path: {path_topic}'
        )
        self.get_logger().info(
            f'Publishing debug image: {debug_image_topic}'
        )

    def image_callback(self, image_msg: Image) -> None:
        """Process one camera image."""

        try:
            image = self.bridge.imgmsg_to_cv2(
                image_msg,
                desired_encoding='bgr8',
            )
        except Exception as error:
            self.get_logger().error(
                f'Failed to convert image: {error}'
            )
            return

        if image is None or image.size == 0:
            self.get_logger().warning('Received an empty image.')
            return

        image_height, image_width = image.shape[:2]

        roi_start_y = int(
            image_height * self.roi_start_ratio
        )

        roi_start_y = max(
            0,
            min(roi_start_y, image_height - 1),
        )

        roi = image[roi_start_y:image_height, :]

        debug_image = image.copy()

        cv2.rectangle(
            debug_image,
            (0, roi_start_y),
            (image_width - 1, image_height - 1),
            (255, 0, 0),
            2,
        )

        # ---------------------------------------------------------
        # TODO 1
        # 현재는 밝은 바닥 위의 검은색 라인을 가정한다.
        # 실제 환경에 따라 HSV threshold, adaptive threshold,
        # Canny, bird's-eye view 등을 적용하도록 수정한다.
        # ---------------------------------------------------------

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY,
        )

        blurred = cv2.GaussianBlur(
            gray,
            (5, 5),
            0,
        )

        _, binary = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 5),
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
        )

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        valid_contours = [
            contour
            for contour in contours
            if cv2.contourArea(contour)
            >= self.min_contour_area
        ]

        if not valid_contours:
            self.publish_not_detected(
                image_msg,
                debug_image,
            )
            return

        lane_contour = max(
            valid_contours,
            key=cv2.contourArea,
        )

        contour_area = float(
            cv2.contourArea(lane_contour)
        )

        moments = cv2.moments(lane_contour)

        if abs(moments['m00']) < 1.0e-6:
            self.publish_not_detected(
                image_msg,
                debug_image,
            )
            return

        center_x = int(
            moments['m10'] / moments['m00']
        )

        center_y_roi = int(
            moments['m01'] / moments['m00']
        )

        center_y_image = center_y_roi + roi_start_y

        # Contour의 주 방향 계산
        line = cv2.fitLine(
            lane_contour,
            cv2.DIST_L2,
            0,
            0.01,
            0.01,
        )

        vx, vy, _, _ = [
            float(value)
            for value in line.flatten()
        ]

        # fitLine 방향이 아래쪽을 향하면 뒤집어서
        # 항상 영상 위쪽을 향하게 만든다.
        if vy > 0.0:
            vx = -vx
            vy = -vy

        # 영상 좌표:
        # 오른쪽 +x, 아래쪽 +y
        image_heading_error = math.atan2(
            vx,
            -vy,
        )

        # ROS base_link:
        # 왼쪽 +y, 반시계 방향 +
        #
        # 영상 오른쪽은 로봇 오른쪽이므로 부호를 반대로 한다.
        lateral_error_normalized = -(
            center_x - image_width * 0.5
        ) / (image_width * 0.5)

        heading_error = -image_heading_error

        roi_area = float(
            max(1, roi.shape[0] * roi.shape[1])
        )

        contour_area_ratio = contour_area / roi_area

        lateral_error_normalized = max(
            -1.0,
            min(1.0, lateral_error_normalized),
        )

        self.publish_features(
            detected=True,
            lateral_error=lateral_error_normalized,
            heading_error=heading_error,
            area_ratio=contour_area_ratio,
        )

        lane_path = self.create_lane_path(
            lateral_error=lateral_error_normalized,
            heading_error=heading_error,
        )

        self.path_publisher.publish(lane_path)

        # Debug 표시를 위해 ROI contour를 원본 영상 좌표로 이동
        contour_for_debug = lane_contour.copy()
        contour_for_debug[:, 0, 1] += roi_start_y

        cv2.drawContours(
            debug_image,
            [contour_for_debug],
            -1,
            (0, 255, 0),
            2,
        )

        cv2.circle(
            debug_image,
            (center_x, center_y_image),
            6,
            (0, 0, 255),
            -1,
        )

        line_length = 80

        line_end_x = int(
            center_x + vx * line_length
        )
        line_end_y = int(
            center_y_image + vy * line_length
        )

        cv2.line(
            debug_image,
            (center_x, center_y_image),
            (line_end_x, line_end_y),
            (0, 255, 255),
            3,
        )

        cv2.putText(
            debug_image,
            (
                f'lat={lateral_error_normalized:+.3f} '
                f'heading={heading_error:+.3f}'
            ),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        self.publish_debug_image(
            image_msg,
            debug_image,
        )

    def publish_not_detected(
        self,
        input_image_msg: Image,
        debug_image,
    ) -> None:
        """Publish a safe output when no lane is detected."""

        self.publish_features(
            detected=False,
            lateral_error=0.0,
            heading_error=0.0,
            area_ratio=0.0,
        )

        empty_path = Path()
        empty_path.header.stamp = (
            self.get_clock().now().to_msg()
        )
        empty_path.header.frame_id = self.frame_id

        self.path_publisher.publish(empty_path)

        cv2.putText(
            debug_image,
            'LANE NOT DETECTED',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        self.publish_debug_image(
            input_image_msg,
            debug_image,
        )

    def publish_features(
        self,
        detected: bool,
        lateral_error: float,
        heading_error: float,
        area_ratio: float,
    ) -> None:
        """Publish raw lane features without confidence scoring."""

        message = Float32MultiArray()

        message.data = [
            1.0 if detected else 0.0,
            float(lateral_error),
            float(heading_error),
            float(area_ratio),
        ]

        self.features_publisher.publish(message)

    def create_lane_path(
        self,
        lateral_error: float,
        heading_error: float,
    ) -> Path:
        """
        Create a temporary path from image-space features.

        TODO:
        Replace this with a calibrated inverse perspective mapping
        or bird's-eye-view path generator.
        """

        path = Path()

        stamp = self.get_clock().now().to_msg()

        path.header.stamp = stamp
        path.header.frame_id = self.frame_id

        lateral_offset_m = (
            lateral_error
            * self.max_lateral_offset_m
        )

        limited_heading = max(
            -0.7,
            min(0.7, heading_error),
        )

        point_count = max(
            2,
            self.path_point_count,
        )

        for index in range(1, point_count + 1):
            ratio = index / point_count

            x_position = (
                self.path_length_m * ratio
            )

            y_position = (
                lateral_offset_m
                + math.tan(limited_heading)
                * x_position
            )

            pose = PoseStamped()

            pose.header.stamp = stamp
            pose.header.frame_id = self.frame_id

            pose.pose.position.x = x_position
            pose.pose.position.y = y_position
            pose.pose.position.z = 0.0

            pose.pose.orientation.z = math.sin(
                limited_heading * 0.5
            )
            pose.pose.orientation.w = math.cos(
                limited_heading * 0.5
            )

            path.poses.append(pose)

        return path

    def publish_debug_image(
        self,
        input_image_msg: Image,
        debug_image,
    ) -> None:
        debug_message = self.bridge.cv2_to_imgmsg(
            debug_image,
            encoding='bgr8',
        )

        debug_message.header = input_image_msg.header

        self.debug_image_publisher.publish(
            debug_message
        )


def main(args=None) -> None:
    rclpy.init(args=args)

    node = LanePerceptionNode()

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
