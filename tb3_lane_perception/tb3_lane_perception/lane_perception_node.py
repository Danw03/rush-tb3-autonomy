#!/usr/bin/env python3

#!/usr/bin/env python3

import collections
import math
from collections import deque

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray


class LanePerceptionNode(Node):
    """
    Camera-based lane perception node using 2nd-order curve fitting and state machine.

    Input
    -----
    /camera/image_raw : sensor_msgs/msg/Image

    Output
    ------
    /lane_features : std_msgs/msg/Float32MultiArray
        - data[0:20] : 10 waypoints (x1, y1, x2, y2, ... x10, y10) of MPC path
    """

    def __init__(self) -> None:
        super().__init__('lane_perception_node')

        # Topic parameters
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('features_topic', '/lane_features')

        # Tuning parameters (알고리즘 튜닝용)
        self.declare_parameter('intersection_threshold', 20000)
        self.declare_parameter('ema_alpha', 0.7)
        self.declare_parameter('lane_width_offset', 250)
        self.declare_parameter('turn_end_margin', 40)
        self.declare_parameter('min_hist_thresh', 20)

        image_topic = str(self.get_parameter('image_topic').value)
        features_topic = str(self.get_parameter('features_topic').value)

        # State machine variables
        self.in_intersection_mode = False
        self.MIN_RECOVERY_PTS = 5
        self.turn_direction = "STRAIGHT"
        self.intersection_y = 0
        self.turn_history = deque(maxlen=5)
        self.prev_fit = None

        self.bridge = CvBridge()

        # ROS 2 Subscribers and Publishers
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

        self.get_logger().info(f'Subscribing to: {image_topic}')
        self.get_logger().info(f'Publishing MPC path features to: {features_topic}')

    def filter_colors(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 100, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        return mask

    def warp_birdseye(self, img):
        h, w = img.shape[:2]
        pts_src = np.float32([
            [w * 0.0, h * 0.80],
            [w * 1.0, h * 0.80],
            [-w * 0.2, h * 1.0],
            [w * 1.2, h * 1.0]
        ])
        pts_dst = np.float32([
            [0, 0], [w, 0],
            [0, h], [w, h]
        ])
        matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped_img = cv2.warpPerspective(img, matrix, (w, h))
        return warped_img

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
                    x_bottom = fit_center[0] * (y_bottom**2) + fit_center[1] * y_bottom + fit_center[2]
                    
                    y_target = win_y_low
                    x_target = fit_center[0] * (y_target**2) + fit_center[1] * y_target + fit_center[2]
                    
                    turn_direction = "LEFT" if x_target < x_bottom else "RIGHT"
                else:
                    left_density = np.count_nonzero(window_slice[:, :midpoint])
                    right_density = np.count_nonzero(window_slice[:, midpoint:])
                    turn_direction = "RIGHT" if left_density > right_density else "LEFT"
                    
                return True, i, turn_direction
                
        return False, -1, "STRAIGHT"

    def find_lane_points(self, binary_warped):
        min_hist_thresh = self.get_parameter('min_hist_thresh').value
        h, w = binary_warped.shape
        midpoint = w // 2
        
        search_bottom = int(h * 0.8)
        histogram = np.sum(binary_warped[search_bottom:, :], axis=0)
        
        left_hist = histogram[:midpoint]
        right_hist = histogram[midpoint:]
        
        if np.max(left_hist) > min_hist_thresh:
            leftx_base = np.argmax(left_hist)
        else:
            leftx_base = w // 4
            
        if np.max(right_hist) > min_hist_thresh:
            rightx_base = np.argmax(right_hist) + midpoint
        else:
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
            
            win_xleft_low, win_xleft_high = leftx_current - margin, leftx_current + margin
            win_xright_low, win_xright_high = rightx_current - margin, rightx_current + margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
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
            return valid_center_pts, fit_center
        elif len(valid_center_pts) == 2:
            pts_center = np.array(valid_center_pts)
            fit_1d = np.polyfit(pts_center[:, 1], pts_center[:, 0], 1)
            fit_center = np.array([0.0, fit_1d[0], fit_1d[1]])
            return valid_center_pts, fit_center
        else:
            return [], None

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
        
        # 1. Image Processing
        white_mask = self.filter_colors(img)
        warped_mask = self.warp_birdseye(white_mask)
        valid_center_pts, raw_fit_center = self.find_lane_points(warped_mask)
        
        # EMA Filtering
        alpha = self.get_parameter('ema_alpha').value
        if raw_fit_center is not None:
            if self.prev_fit is not None:
                fit_center = alpha * self.prev_fit + (1.0 - alpha) * raw_fit_center
            else:
                fit_center = raw_fit_center
            self.prev_fit = fit_center
        else:
            fit_center = self.prev_fit
            
        # 2. State Machine Update
        is_crossroad_detected, detected_win_idx, current_turn_dir = self.detect_intersection_and_direction(
            warped_mask, fit_center
        )
        
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
                    self.get_logger().info(f"🚨 [{self.turn_direction} 턴 진입 확정]")
                    self.turn_history.clear()
            else:
                if len(self.turn_history) > 0:
                    self.turn_history.clear()
        else:
            is_lane_recovered = (len(valid_center_pts) >= self.MIN_RECOVERY_PTS)
            if is_lane_recovered and not is_crossroad_detected:
                self.in_intersection_mode = False
                self.turn_direction = "STRAIGHT"
                self.get_logger().info("✅ [차선 복귀]")
            else:
                intersection_flag_value = 1.0

        # 3. Path Generation
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
                
                if self.turn_direction == "RIGHT":
                    P2_x = P1_x + lane_width_offset
                else:
                    P2_x = P1_x - lane_width_offset
                P2_y = P1_y
                
                t_values = np.linspace(0.0, 1.0, num_waypoints)
                for t in t_values:
                    bx = float((1-t)**2 * P0[0] + 2*(1-t)*t * P1_x + t**2 * P2_x)
                    by = float((1-t)**2 * P0[1] + 2*(1-t)*t * P1_y + t**2 * P2_y)
                    mpc_path.append((bx, by))

        # 4. Publish Features (Only Path Coordinates)
        if mpc_path:
            features_msg = Float32MultiArray()
            # 튜플 리스트 [(x1, y1), (x2, y2), ...]를 1차원 리스트 [x1, y1, x2, y2, ...]로 평탄화
            features_msg.data = [val for pt in mpc_path for val in pt]
            self.features_publisher.publish(features_msg)


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
