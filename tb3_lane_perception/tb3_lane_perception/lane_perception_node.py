#!/usr/bin/env python3

import collections
import math
from collections import deque

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class LanePerceptionNode(Node):
    """
    Camera-based lane perception node using 2nd-order curve fitting and state machine.
    * Includes Canny edge detection, smart start points, and dynamic margin search.
    """

    def __init__(self) -> None:
        super().__init__('lane_perception_node')

        # Topic parameters
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('path_topic', '/lane_path')
        self.declare_parameter('path_frame', 'base_footprint')
        self.declare_parameter('debug_image_topic', '/lane_debug_image')
        self.declare_parameter('debug_enabled', True)
        self.declare_parameter('debug_publish_rate_hz', 10.0)

        # Calibrated camera-pixel -> base_footprint ground-plane transform.
        # The default matrix was fitted from nine measured ground points in a
        # 640x360 image captured with camera_ros orientation=180:
        # x = {0.35, 0.45, 0.55} m and y = {+0.10, 0.00, -0.10} m.
        # ROS convention: +x forward, +y left.
        self.declare_parameter('use_ground_homography', True)
        self.declare_parameter('calibration_image_width', 640)
        self.declare_parameter('calibration_image_height', 360)
        self.declare_parameter(
            'camera_to_ground_homography',
            [
                -2.17988523147e-05,
                -2.82921592876e-04,
                -3.51409662208e-01,
                6.41847053732e-04,
                8.37437547011e-06,
                -1.96711769375e-01,
                1.02191172386e-04,
                -6.93535449431e-03,
                1.0,
            ],
        )
        self.declare_parameter('minimum_forward_path_m', 0.20)
        self.declare_parameter('maximum_forward_path_m', 0.75)
        self.declare_parameter('maximum_abs_lateral_path_m', 0.40)

        # Legacy bird's-eye pixel scaling. These parameters are used only when
        # use_ground_homography is false.
        self.declare_parameter('forward_m_per_pixel', 0.001)
        self.declare_parameter('lateral_m_per_pixel', 0.001)
        self.declare_parameter('lateral_offset_m', 0.0)
        self.declare_parameter('max_lost_frames', 3)

        # Tuning parameters (알고리즘 튜닝용)
        self.declare_parameter('intersection_threshold', 30000)
        self.declare_parameter('ema_alpha', 0.7)
        self.declare_parameter('lane_width_offset', 250)
        self.declare_parameter('turn_end_margin', 40)
        self.declare_parameter('min_hist_thresh', 20)

        # New Tuning parameters for Robust Tracking
        self.declare_parameter('expected_lane_width', 500)
        self.declare_parameter('dynamic_margin', 150)

        image_topic = str(self.get_parameter('image_topic').value)
        path_topic = str(self.get_parameter('path_topic').value)
        debug_image_topic = str(
            self.get_parameter('debug_image_topic').value
        )

        # State machine variables
        self.in_intersection_mode = False
        self.MIN_RECOVERY_PTS = 5
        self.turn_direction = 'STRAIGHT'
        self.intersection_y = 0
        self.turn_history = deque(maxlen=5)
        self.prev_fit = None
        self.lost_frame_count = 0
        self.last_debug_publish_ns = 0

        # Smart tracking memory variables
        self.prev_leftx_base = None
        self.prev_rightx_base = None

        self.bridge = CvBridge()
        self.load_ground_calibration()

        # ROS 2 Subscribers and Publishers
        self.image_subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.path_publisher = self.create_publisher(
            Path,
            path_topic,
            10,
        )
        self.debug_image_publisher = self.create_publisher(
            Image,
            debug_image_topic,
            2,
        )

        self.get_logger().info(f'Subscribing to: {image_topic}')
        self.get_logger().info(f'Publishing lane path to: {path_topic}')
        self.get_logger().info(
            f'Publishing recognition view to: {debug_image_topic}'
        )
        if self.use_ground_homography:
            self.get_logger().info(
                'Ground homography enabled for '
                f'{self.calibration_image_width}x'
                f'{self.calibration_image_height} images; '
                'keep camera_ros orientation=180'
            )

    def load_ground_calibration(self) -> None:
        """Load and validate the measured camera-to-ground calibration."""
        self.use_ground_homography = bool(
            self.get_parameter('use_ground_homography').value
        )
        self.calibration_image_width = int(
            self.get_parameter('calibration_image_width').value
        )
        self.calibration_image_height = int(
            self.get_parameter('calibration_image_height').value
        )
        homography_values = list(
            self.get_parameter('camera_to_ground_homography').value
        )
        self.minimum_forward_path_m = float(
            self.get_parameter('minimum_forward_path_m').value
        )
        self.maximum_forward_path_m = float(
            self.get_parameter('maximum_forward_path_m').value
        )
        self.maximum_abs_lateral_path_m = float(
            self.get_parameter('maximum_abs_lateral_path_m').value
        )

        if self.calibration_image_width <= 0:
            raise ValueError('calibration_image_width must be positive')
        if self.calibration_image_height <= 0:
            raise ValueError('calibration_image_height must be positive')
        if len(homography_values) != 9:
            raise ValueError(
                'camera_to_ground_homography must contain 9 values'
            )
        if not all(math.isfinite(float(value)) for value in homography_values):
            raise ValueError(
                'camera_to_ground_homography contains a non-finite value'
            )
        if self.minimum_forward_path_m < 0.0:
            raise ValueError('minimum_forward_path_m must be non-negative')
        if self.maximum_forward_path_m <= self.minimum_forward_path_m:
            raise ValueError(
                'maximum_forward_path_m must exceed minimum_forward_path_m'
            )
        if self.maximum_abs_lateral_path_m <= 0.0:
            raise ValueError(
                'maximum_abs_lateral_path_m must be positive'
            )

        self.camera_to_ground_homography = np.asarray(
            homography_values,
            dtype=np.float64,
        ).reshape(3, 3)

    def pixel_path_to_metric(
        self,
        pixel_path,
        image_width,
        image_height,
    ):
        """Convert ordered bird's-eye pixels to base_footprint metres."""
        if not pixel_path:
            return []

        if not self.use_ground_homography:
            forward_scale = float(
                self.get_parameter('forward_m_per_pixel').value
            )
            lateral_scale = float(
                self.get_parameter('lateral_m_per_pixel').value
            )
            lateral_offset = float(
                self.get_parameter('lateral_offset_m').value
            )
            return [
                (
                    max(
                        0.0,
                        (float(image_height) - pixel_y) * forward_scale,
                    ),
                    -(
                        pixel_x - 0.5 * float(image_width)
                    ) * lateral_scale + lateral_offset,
                )
                for pixel_x, pixel_y in pixel_path
            ]

        _, inverse_matrix = self.birdseye_matrices(
            image_width,
            image_height,
        )
        camera_points = self.project_to_camera(
            pixel_path,
            inverse_matrix,
        )

        scale_u = self.calibration_image_width / float(image_width)
        scale_v = self.calibration_image_height / float(image_height)
        metric_points = []
        for camera_u, camera_v in camera_points:
            calibration_point = np.asarray(
                [camera_u * scale_u, camera_v * scale_v, 1.0],
                dtype=np.float64,
            )
            projected = (
                self.camera_to_ground_homography @ calibration_point
            )
            denominator = float(projected[2])
            if abs(denominator) < 1.0e-9:
                continue

            forward_x = float(projected[0] / denominator)
            lateral_y = float(projected[1] / denominator)
            if not (
                math.isfinite(forward_x)
                and math.isfinite(lateral_y)
            ):
                continue
            if not (
                self.minimum_forward_path_m
                <= forward_x
                <= self.maximum_forward_path_m
            ):
                continue
            if abs(lateral_y) > self.maximum_abs_lateral_path_m:
                continue
            metric_points.append((forward_x, lateral_y))

        return metric_points

    def make_path_message(
        self,
        pixel_path,
        image_width,
        image_height,
        stamp,
    ):
        """Convert ordered bird's-eye pixels into a robot-frame Path."""
        path_msg = Path()
        path_msg.header.stamp = stamp
        path_msg.header.frame_id = str(
            self.get_parameter('path_frame').value
        )

        if not pixel_path:
            return path_msg

        metric_points = self.pixel_path_to_metric(
            pixel_path,
            image_width,
            image_height,
        )

        if len(metric_points) < 2:
            return path_msg

        yaws = []
        for index in range(len(metric_points)):
            if index < len(metric_points) - 1:
                start = metric_points[index]
                end = metric_points[index + 1]
            else:
                start = metric_points[index - 1]
                end = metric_points[index]

            delta_x = end[0] - start[0]
            delta_y = end[1] - start[1]
            yaws.append(math.atan2(delta_y, delta_x))

        for (forward_x, lateral_y), yaw in zip(metric_points, yaws):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = forward_x
            pose.pose.position.y = lateral_y
            pose.pose.position.z = 0.0
            pose.pose.orientation.z = math.sin(0.5 * yaw)
            pose.pose.orientation.w = math.cos(0.5 * yaw)
            path_msg.poses.append(pose)

        return path_msg

    def filter_colors(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 100, 255])
        color_mask = cv2.inRange(hsv, lower_white, upper_white)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edge_mask = cv2.Canny(blur, 50, 150)

        combined_mask = cv2.bitwise_or(color_mask, edge_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        return combined_mask

    @staticmethod
    def birdseye_matrices(image_width, image_height):
        """Return camera-to-birdseye and birdseye-to-camera matrices."""
        w = float(image_width)
        h = float(image_height)
        pts_src = np.float32([
            [w * 0.0, h * 0.70],
            [w * 1.0, h * 0.70],
            [-w * 0.2, h * 1.0],
            [w * 1.2, h * 1.0]
        ])
        pts_dst = np.float32([
            [0, 0], [w, 0],
            [0, h], [w, h]
        ])
        matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
        inverse_matrix = cv2.getPerspectiveTransform(pts_dst, pts_src)
        return matrix, inverse_matrix

    def warp_birdseye(self, img):
        h, w = img.shape[:2]
        matrix, _ = self.birdseye_matrices(w, h)
        warped_img = cv2.warpPerspective(img, matrix, (w, h))
        return warped_img

    @staticmethod
    def project_to_camera(pixel_points, inverse_matrix):
        """Project birdseye pixel points back onto the camera image."""
        if not pixel_points:
            return []

        points = np.asarray(pixel_points, dtype=np.float32).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(points, inverse_matrix)
        return [tuple(point) for point in projected.reshape(-1, 2)]

    def publish_debug_image(
        self,
        camera_image,
        warped_mask,
        detected_centers,
        path_points,
        image_msg,
    ):
        """Publish the live camera image with recognition overlays."""
        if not bool(self.get_parameter('debug_enabled').value):
            return

        debug_rate = float(
            self.get_parameter('debug_publish_rate_hz').value
        )
        if debug_rate <= 0.0:
            return

        now_ns = self.get_clock().now().nanoseconds
        minimum_period_ns = int(1.0e9 / debug_rate)
        if now_ns - self.last_debug_publish_ns < minimum_period_ns:
            return
        self.last_debug_publish_ns = now_ns

        height, width = camera_image.shape[:2]
        _, inverse_matrix = self.birdseye_matrices(width, height)
        output = camera_image.copy()

        source_roi = np.array(
            [
                [0, int(height * 0.70)],
                [width - 1, int(height * 0.70)],
                [width - 1, height - 1],
                [0, height - 1],
            ],
            dtype=np.int32,
        )
        cv2.polylines(output, [source_roi], True, (0, 255, 255), 1)

        projected_centers = self.project_to_camera(
            detected_centers,
            inverse_matrix,
        )
        for point_x, point_y in projected_centers:
            if math.isfinite(point_x) and math.isfinite(point_y):
                cv2.circle(
                    output,
                    (int(round(point_x)), int(round(point_y))),
                    3,
                    (255, 255, 0),
                    -1,
                )

        projected_path = self.project_to_camera(
            path_points,
            inverse_matrix,
        )
        drawable_path = []
        for point_x, point_y in projected_path:
            if math.isfinite(point_x) and math.isfinite(point_y):
                drawable_path.append(
                    (int(round(point_x)), int(round(point_y)))
                )

        if len(drawable_path) >= 2:
            cv2.polylines(
                output,
                [np.asarray(drawable_path, dtype=np.int32)],
                False,
                (0, 255, 0),
                3,
            )
        for point in drawable_path:
            cv2.circle(output, point, 4, (0, 0, 255), -1)

        # Small birdseye inset: white pixels are the perception mask and the
        # green/red line is exactly the pixel path converted into /lane_path.
        inset_width = max(160, width // 3)
        inset_height = max(90, height // 3)
        inset = cv2.cvtColor(warped_mask, cv2.COLOR_GRAY2BGR)
        inset = cv2.resize(inset, (inset_width, inset_height))
        scale_x = inset_width / float(width)
        scale_y = inset_height / float(height)
        inset_path = [
            (int(point[0] * scale_x), int(point[1] * scale_y))
            for point in path_points
        ]
        if len(inset_path) >= 2:
            cv2.polylines(
                inset,
                [np.asarray(inset_path, dtype=np.int32)],
                False,
                (0, 255, 0),
                2,
            )
        for point in inset_path:
            cv2.circle(inset, point, 2, (0, 0, 255), -1)

        inset_x = width - inset_width - 8
        inset_y = 8
        output[
            inset_y:inset_y + inset_height,
            inset_x:inset_x + inset_width,
        ] = inset
        cv2.rectangle(
            output,
            (inset_x, inset_y),
            (inset_x + inset_width, inset_y + inset_height),
            (255, 255, 255),
            1,
        )

        lane_ok = len(path_points) >= 2
        state_text = 'LANE OK' if lane_ok else 'LANE LOST'
        state_color = (0, 255, 0) if lane_ok else (0, 0, 255)
        cv2.putText(
            output,
            state_text,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            state_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            f'mode={self.turn_direction} points={len(path_points)} '
            f'lost={self.lost_frame_count}',
            (12, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        metric_path = self.pixel_path_to_metric(
            path_points,
            width,
            height,
        )
        if metric_path:
            first_x, first_y = metric_path[0]
            last_x, last_y = metric_path[-1]
            cv2.putText(
                output,
                (
                    f'base: ({first_x:.2f},{first_y:+.2f}) -> '
                    f'({last_x:.2f},{last_y:+.2f}) m'
                ),
                (12, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        debug_message = self.bridge.cv2_to_imgmsg(output, encoding='bgr8')
        debug_message.header = image_msg.header
        self.debug_image_publisher.publish(debug_message)

    def detect_intersection_and_direction(self, binary_warped, fit_center):
        threshold = self.get_parameter('intersection_threshold').value
        h, w = binary_warped.shape
        roi_top = h // 4
        roi_bottom = h
        midpoint = w // 2

        n_windows = 5
        window_height = (roi_bottom - roi_top) // n_windows

        for i in range(n_windows):
            win_y_low = roi_top + i * window_height
            win_y_high = roi_top + (i + 1) * window_height if i < n_windows - 1 else roi_bottom

            window_slice = binary_warped[win_y_low:win_y_high, :]
            horizontal_hist = np.sum(window_slice, axis=1)

            if np.max(horizontal_hist) > threshold:
                if fit_center is not None:
                    y_bottom = h
                    x_bottom = (
                        fit_center[0] * (y_bottom**2)
                        + fit_center[1] * y_bottom
                        + fit_center[2]
                    )
                    y_target = win_y_low
                    x_target = (
                        fit_center[0] * (y_target**2)
                        + fit_center[1] * y_target
                        + fit_center[2]
                    )
                    turn_direction = (
                        'LEFT' if x_target < x_bottom else 'RIGHT'
                    )
                else:
                    left_density = np.count_nonzero(window_slice[:, :midpoint])
                    right_density = np.count_nonzero(window_slice[:, midpoint:])
                    turn_direction = (
                        'RIGHT' if left_density > right_density else 'LEFT'
                    )

                return True, i, turn_direction

        return False, -1, 'STRAIGHT'

    def find_lane_points(self, binary_warped):
        min_hist_thresh = self.get_parameter('min_hist_thresh').value
        expected_lane_width = self.get_parameter('expected_lane_width').value
        dynamic_margin = self.get_parameter('dynamic_margin').value

        h, w = binary_warped.shape
        midpoint = w // 2

        search_bottom = int(h * 0.8)
        histogram = np.sum(binary_warped[search_bottom:, :], axis=0)

        left_hist = histogram[:midpoint]
        right_hist = histogram[midpoint:]

        left_found = np.max(left_hist) > min_hist_thresh
        right_found = np.max(right_hist) > min_hist_thresh

        # --- [1. 스마트 시작점 추정] ---
        if left_found:
            leftx_base = np.argmax(left_hist)
        else:
            leftx_base = None

        if right_found:
            rightx_base = np.argmax(right_hist) + midpoint
        else:
            rightx_base = None

        if left_found and not right_found:
            rightx_base = leftx_base + expected_lane_width
        elif right_found and not left_found:
            leftx_base = rightx_base - expected_lane_width
        elif not left_found and not right_found:
            if self.prev_leftx_base is not None and self.prev_rightx_base is not None:
                leftx_base = self.prev_leftx_base
                rightx_base = self.prev_rightx_base
            else:
                leftx_base = w // 4
                rightx_base = w * 3 // 4

        nwindows = 10
        window_height = int(h / nwindows)
        margin = 50
        minpix = 50

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        valid_center_pts = []

        for window in range(nwindows):
            win_y_low = h - (window + 1) * window_height
            win_y_high = h - window * window_height

            # 1차 탐색
            win_xleft_low, win_xleft_high = leftx_current - margin, leftx_current + margin
            win_xright_low, win_xright_high = rightx_current - margin, rightx_current + margin

            good_left_inds = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_xleft_low)
                & (nonzerox < win_xleft_high)
            ).nonzero()[0]
            good_right_inds = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_xright_low)
                & (nonzerox < win_xright_high)
            ).nonzero()[0]

            # --- [2. 동적 마진 (Dynamic Margin) 적용] ---
            if len(good_left_inds) < minpix:
                win_xleft_low = leftx_current - dynamic_margin
                win_xleft_high = leftx_current + dynamic_margin
                good_left_inds = (
                    (nonzeroy >= win_y_low)
                    & (nonzeroy < win_y_high)
                    & (nonzerox >= win_xleft_low)
                    & (nonzerox < win_xleft_high)
                ).nonzero()[0]

            if len(good_right_inds) < minpix:
                win_xright_low = rightx_current - dynamic_margin
                win_xright_high = rightx_current + dynamic_margin
                good_right_inds = (
                    (nonzeroy >= win_y_low)
                    & (nonzeroy < win_y_high)
                    & (nonzerox >= win_xright_low)
                    & (nonzerox < win_xright_high)
                ).nonzero()[0]

            found_lane_in_window = False

            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
                found_lane_in_window = True

            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
                found_lane_in_window = True

            if found_lane_in_window:
                center_x = int((leftx_current + rightx_current) / 2)
                center_y = int((win_y_low + win_y_high) / 2)
                valid_center_pts.append((center_x, center_y))

        if len(valid_center_pts) >= 3:
            pts_center = np.array(valid_center_pts)
            fit_center = np.polyfit(pts_center[:, 1], pts_center[:, 0], 2)
        elif len(valid_center_pts) == 2:
            pts_center = np.array(valid_center_pts)
            fit_1d = np.polyfit(pts_center[:, 1], pts_center[:, 0], 1)
            fit_center = np.array([0.0, fit_1d[0], fit_1d[1]])
        else:
            fit_center = None

        return valid_center_pts, fit_center, leftx_base, rightx_base

    def image_callback(self, image_msg: Image) -> None:
        """Process one camera image and publish MPC path."""
        try:
            img = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as error:
            self.get_logger().error(f'Failed to convert image: {error}')
            return

        if img is None or img.size == 0:
            self.get_logger().warning('Received an empty image.')
            return

        img_height, img_width = img.shape[:2]

        # 1. Image Processing & Lane Finding
        white_mask = self.filter_colors(img)
        warped_mask = self.warp_birdseye(white_mask)
        (
            valid_center_pts,
            raw_fit_center,
            current_leftx,
            current_rightx,
        ) = self.find_lane_points(warped_mask)

        # State Update: 시작점 메모리 저장
        self.prev_leftx_base = current_leftx
        self.prev_rightx_base = current_rightx

        # 2. EMA Filtering for Curve Fitting
        alpha = self.get_parameter('ema_alpha').value
        if raw_fit_center is not None:
            self.lost_frame_count = 0
            if self.prev_fit is not None:
                fit_center = alpha * self.prev_fit + (1.0 - alpha) * raw_fit_center
            else:
                fit_center = raw_fit_center
            self.prev_fit = fit_center
        else:
            self.lost_frame_count += 1
            max_lost_frames = int(
                self.get_parameter('max_lost_frames').value
            )
            if self.lost_frame_count <= max_lost_frames:
                fit_center = self.prev_fit
            else:
                fit_center = None
                self.prev_fit = None

        # 3. State Machine (Intersection Detection)
        (
            is_crossroad_detected,
            detected_win_idx,
            current_turn_dir,
        ) = self.detect_intersection_and_direction(warped_mask, fit_center)

        intersection_flag_value = 0.0

        if not self.in_intersection_mode:
            if is_crossroad_detected:
                self.turn_history.append(current_turn_dir)
                if len(self.turn_history) >= 2:
                    counter = collections.Counter(self.turn_history)
                    final_turn_dir = counter.most_common(1)[0][0]

                    self.in_intersection_mode = True
                    self.turn_direction = final_turn_dir

                    roi_top = img_height // 2
                    window_height = (img_height - roi_top) // 5
                    self.intersection_y = roi_top + int((detected_win_idx + 1) * window_height)

                    intersection_flag_value = 1.0
                    self.get_logger().info(
                        f'[{self.turn_direction} turn confirmed]'
                    )
                    self.turn_history.clear()
            else:
                if len(self.turn_history) > 0:
                    self.turn_history.clear()
        else:
            is_lane_recovered = (len(valid_center_pts) >= self.MIN_RECOVERY_PTS)
            if is_lane_recovered and not is_crossroad_detected:
                self.in_intersection_mode = False
                self.turn_direction = 'STRAIGHT'
                self.get_logger().info('[Lane recovered]')
            else:
                intersection_flag_value = 1.0

        # 4. Path Generation
        mpc_path = []
        num_waypoints = 10
        lane_width_offset = self.get_parameter('lane_width_offset').value
        turn_end_margin = self.get_parameter('turn_end_margin').value

        if intersection_flag_value == 0.0:
            if fit_center is not None:
                y_points = np.linspace(img_height, 0, num_waypoints)
                for y in y_points:
                    x = fit_center[0] * (y**2) + fit_center[1] * y + fit_center[2]
                    mpc_path.append((float(x), float(y)))
        else:
            if fit_center is not None:
                P0_y = img_height
                P0_x = fit_center[0] * (P0_y**2) + fit_center[1] * P0_y + fit_center[2]
                P0 = (P0_x, P0_y)

                P1_y = self.intersection_y + turn_end_margin
                P1_x = fit_center[0] * (P1_y**2) + fit_center[1] * P1_y + fit_center[2]

                if self.turn_direction == 'RIGHT':
                    P2_x = P1_x + lane_width_offset
                else:
                    P2_x = P1_x - lane_width_offset
                P2_y = P1_y

                t_values = np.linspace(0.0, 1.0, num_waypoints)
                for t in t_values:
                    bx = float((1-t)**2 * P0[0] + 2*(1-t)*t * P1_x + t**2 * P2_x)
                    by = float((1-t)**2 * P0[1] + 2*(1-t)*t * P1_y + t**2 * P2_y)
                    mpc_path.append((bx, by))

        # 5. Publish nav_msgs/Path in robot coordinates.
        # An empty Path explicitly marks lane loss for downstream nodes.
        path_msg = self.make_path_message(
            mpc_path,
            img_width,
            img_height,
            image_msg.header.stamp,
        )
        self.path_publisher.publish(path_msg)
        self.publish_debug_image(
            img,
            warped_mask,
            valid_center_pts,
            mpc_path,
            image_msg,
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
