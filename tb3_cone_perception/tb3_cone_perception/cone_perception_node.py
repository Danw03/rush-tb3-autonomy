#!/usr/bin/env python3

import math
from typing import Dict, List, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from visualization_msgs.msg import Marker, MarkerArray


class ConePerceptionNode(Node):
    """
    Cone-perception pipeline (LiDAR team).

    /scan -> nan/inf 제거 -> 극좌표 클러스터링 -> 원 피팅 콘 판정
          -> /cone_path, /cone_features

    클러스터링은 극좌표에서 직접 수행한다. 두 점 사이 거리는 코사인법칙
        d = sqrt(r1^2 + r2^2 - 2*r1*r2*cos(dphi))
    으로 계산하므로 직교 변환 없이 정확한 값을 얻는다.

    설계 원칙:
      * 콘 판정은 파라미터 튜닝이 아니라 콘의 실제 물리 치수와의 일치로 한다.
      * cone_diameter_m 은 밑면이 아니라 **LiDAR 가 자르는 높이에서의 지름**이다.
        콘은 위로 갈수록 좁아지므로 두 값은 크게 다르다.
      * 검출 최대 거리는 각도 분해능에서 자동 유도한다. 파라미터를 하나
        바꿨을 때 다른 파라미터가 물리적으로 모순되는 일을 막는다.
      * 기존 토픽 이름과 FEATURE_LAYOUT(7개, 순서 동일)을 바꾸지 않는다.
        tb3_reference 와의 통신 계약을 유지하기 위함이다.
      * 디버깅 정보는 새 토픽으로만 추가한다(가산적 변경).

    남은 TODO:
      3. split_left_right     - 좌/우 경계 분류
      4. generate_center_path - 중심 경로 생성
      5. extract_features     - 최종 feature 벡터

      /cone_features 로 내보내는 숫자 7개 순서 약속

      0 path_valid 결과가 쓸만한가(double)
      1 minimum_distance_m 가장 가까운 콘까지의 거리
      2 nearest_angle_rad 그 콘의 방향
      3 nearest_x_m
      4 nearest_y_m
      5 valid_scan_ratio 쓸 만한 측정값 비율
      6 valid_point_count 쓸 만한 측정값 개수
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

    DETECTION_LAYOUT = ["x_m", "y_m", "radius_m", "point_count"]

    def __init__(self) -> None:
        super().__init__("cone_perception_node")

        # --- 기존 파라미터: 이름과 기본값을 바꾸지 않는다 ---
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("path_topic", "/cone_path")
        self.declare_parameter("features_topic", "/cone_features")
        self.declare_parameter("front_fov_deg", 180.0)

        # --- 콘 기하 ---
        # 실측 스캔 3장에서 원 피팅으로 얻은 지름은 0.085 +- 0.017 m 였다.
        # 콘을 바꾸거나 LiDAR 높이가 달라지면 반드시 다시 재야 한다.
        # calibration_mode 를 켜면 모든 후보의 지름이 로그로 나온다.
        # 지름 0.085을 기준으로 +-42%, 즉 0.049~0.121m 사이면 콘으로 인정
        # cone_width_margin 덩어리 폭이 콘 지름의 1.6배를 넘으면 벽
        # cone_point_tolerance 이 거리에서 콘이라면 찍혀야 할 점의 개수의 절반에도 못 미치면 노이즈로 보고 기각
        self.declare_parameter("cone_diameter_m", 0.085)
        self.declare_parameter("cone_diameter_tolerance", 0.42)
        self.declare_parameter("cone_fit_rms_max_m", 0.010)
        self.declare_parameter("cone_width_margin", 1.60)
        self.declare_parameter("cone_point_tolerance", 0.50)

        # --- 거리 게이트 ---
        # range_gate_max_m 을 0 이하로 두면 각도 분해능에서 자동 계산한다.
        # 아래 effective_range_max() 라는 별도 함수에서 정함
        self.declare_parameter("range_gate_min_m", 0.12)
        self.declare_parameter("range_gate_max_m", 0.0)

        # --- 극좌표 클러스터링 ---
        # 
        self.declare_parameter("cluster_max_beam_gap", 2)
        self.declare_parameter("cluster_gap_base_m", 0.04)
        self.declare_parameter("cluster_gap_span_factor", 2.0)
        self.declare_parameter("cluster_min_points", 3) # 원을 그리려면 최소 3개의 점이 필요함

        # --- 진단 ---
        self.declare_parameter("log_period_s", 1.0)
        self.declare_parameter("calibration_mode", False)
        self.declare_parameter("publish_markers", True)
        self.declare_parameter("detections_topic", "/cone_detections")
        self.declare_parameter("markers_topic", "/cone_markers")

        scan_topic = str(self.get_parameter("scan_topic").value)
        path_topic = str(self.get_parameter("path_topic").value)
        features_topic = str(self.get_parameter("features_topic").value)
        detections_topic = str(self.get_parameter("detections_topic").value)
        markers_topic = str(self.get_parameter("markers_topic").value)

        self.angle_increment = 0.0
        self.last_log_time = 0.0

        self.scan_sub = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scan_callback,
            qos_profile_sensor_data,
        )

        # 기존 퍼블리셔: 이름·타입 그대로 유지
        self.path_pub = self.create_publisher(Path, path_topic, 10)
        self.features_pub = self.create_publisher(
            Float32MultiArray,
            features_topic,
            10,
        ) 

        # 추가 퍼블리셔: 디버깅 전용, 기존 계약에 영향 없음
        self.detections_pub = self.create_publisher(
            Float32MultiArray,
            detections_topic,
            10,
        )
        self.markers_pub = self.create_publisher(MarkerArray, markers_topic, 10)

        self.get_logger().info(
            f"Started: {scan_topic} -> {path_topic}, {features_topic}"
        )
        self.get_logger().info(
            "cone_features layout: " + ", ".join(self.FEATURE_LAYOUT)
        )
        self.get_logger().info(
            f"debug topics: {detections_topic} "
            f"[{', '.join(self.DETECTION_LAYOUT)}], {markers_topic}"
        )

    # ------------------------------------------------------------------
    # 파라미터
    # ------------------------------------------------------------------

    def reload_parameters(self) -> None:
        """콜백마다 파라미터를 다시 읽어 ros2 param set 을 즉시 반영한다."""
        self.front_fov_rad = math.radians(
            float(self.get_parameter("front_fov_deg").value)
        )
        self.cone_diameter_m = float(
            self.get_parameter("cone_diameter_m").value
        )
        self.cone_diameter_tolerance = float(
            self.get_parameter("cone_diameter_tolerance").value
        )
        self.cone_fit_rms_max_m = float(
            self.get_parameter("cone_fit_rms_max_m").value
        )
        self.cone_width_margin = float(
            self.get_parameter("cone_width_margin").value
        )
        self.cone_point_tolerance = float(
            self.get_parameter("cone_point_tolerance").value
        )
        self.range_gate_min_m = float(
            self.get_parameter("range_gate_min_m").value
        )
        self.range_gate_max_param = float(
            self.get_parameter("range_gate_max_m").value
        )
        self.cluster_max_beam_gap = max(
            1, int(self.get_parameter("cluster_max_beam_gap").value)
        )
        self.cluster_gap_base_m = float(
            self.get_parameter("cluster_gap_base_m").value
        )
        self.cluster_gap_span_factor = float(
            self.get_parameter("cluster_gap_span_factor").value
        )
        self.cluster_min_points = max(
            3, int(self.get_parameter("cluster_min_points").value)
        )
        self.log_period_s = float(self.get_parameter("log_period_s").value)
        self.calibration_mode = bool(
            self.get_parameter("calibration_mode").value
        )
        self.publish_markers = bool(
            self.get_parameter("publish_markers").value
        )

    def effective_range_max(self) -> float:
        """검출 가능한 최대 거리.

        콘에 최소 cluster_min_points 개의 빔이 맞아야 원을 그릴 수 있다.
        콘이 차지하는 각폭이 2*atan(R/d) 이므로

            2*atan(R/d) >= min_points * angle_increment
            d <= R / tan(min_points * angle_increment / 2)

        이 값을 넘는 거리는 물리적으로 검출이 불가능하므로 미리 잘라낸다.
        """
        if self.range_gate_max_param > 0.0:
            return self.range_gate_max_param

        if self.angle_increment <= 0.0:
            return 3.5

        half = self.cluster_min_points * self.angle_increment / 2.0
        if half <= 0.0 or half >= math.pi / 2.0:
            return 3.5

        return (self.cone_diameter_m / 2.0) / math.tan(half)

    # ------------------------------------------------------------------
    # 메인 콜백
    # ------------------------------------------------------------------

    def scan_callback(self, msg: LaserScan) -> None:
        self.reload_parameters()
        self.angle_increment = float(msg.angle_increment)

        points, ranges, angles, valid_ratio = self.preprocess_scan(msg)

        if len(ranges) == 0:
            self.publish_empty_result(msg, valid_ratio)
            self.publish_detections(msg, [])
            if self.publish_markers:
                self.publish_cone_markers(msg, [])
            self.log_summary(0, 0, None, valid_ratio, 0)
            return

        clusters = self.cluster_lidar_points(points, ranges, angles)
        cones = self.detect_cones(clusters)

        self.publish_detections(msg, cones)
        if self.publish_markers:
            self.publish_cone_markers(msg, cones)

        if not cones:
            self.publish_empty_result(msg, valid_ratio)
            self.log_summary(len(clusters), 0, None, valid_ratio, len(ranges))
            return

        nearest = min(cones, key=lambda c: c["distance"])

        path_msg = self.build_placeholder_path(
            msg=msg,
            x=float(nearest["center"][0]),
            y=float(nearest["center"][1]),
        )

        feature_msg = Float32MultiArray()
        feature_msg.data = [
            1.0,
            float(nearest["distance"]),
            float(math.atan2(nearest["center"][1], nearest["center"][0])),
            float(nearest["center"][0]),
            float(nearest["center"][1]),
            float(valid_ratio),
            float(len(ranges)),
        ]

        self.path_pub.publish(path_msg)
        self.features_pub.publish(feature_msg)

        self.log_summary(
            len(clusters), len(cones), nearest, valid_ratio, len(ranges)
        )

    # ------------------------------------------------------------------
    # 전처리 (원본 유지)
    # ------------------------------------------------------------------

    def preprocess_scan(
        self,
        msg: LaserScan,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Convert valid LaserScan samples into robot-frame 2-D points.

        np.isfinite 가 inf / -inf / nan 을 모두 제거하고,
        range_min 이 0.0 무효 측정값을 제거한다.
        """
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

    # ------------------------------------------------------------------
    # TODO 1: 극좌표 클러스터링
    # ------------------------------------------------------------------

    def cluster_lidar_points(
        self,
        points: np.ndarray,
        ranges: np.ndarray = None,
        angles: np.ndarray = None,
    ) -> List[np.ndarray]:
        """극좌표에서 두 개의 관문으로 클러스터를 자른다.

        관문 1 (각도): 이웃한 두 점의 각도 차이가 angle_increment 의 몇 배인지
          세어, 정해진 개수 이상 빔이 비어 있으면 자른다. 두 물체 사이로
          레이저가 빠져나가면 그 방향 빔은 아무것도 맞히지 못해 nan 이 되고,
          nan 을 제거한 뒤에는 그 공백이 좌표상으로 보이지 않는다.
          이 관문이 없으면 나란히 선 두 콘이 하나로 합쳐지고,
          합쳐진 덩어리는 폭 검사에서 기각되어 둘 다 사라진다.

        관문 2 (거리): 코사인법칙으로 두 점의 실제 거리를 구해 임계값과 비교한다.
              d = sqrt(r1^2 + r2^2 - 2*r1*r2*cos(dphi))
          점 간격이 r * angle_increment 로 거리에 비례해 벌어지므로
          임계값도 거리에 비례해 키운다.

        거리 차이 |r2 - r1| 만으로 자르는 순수 ABD 방식은 쓰지 않는다.
        콘 표면의 가장자리는 빔과 거의 평행해서 인접 빔 간 거리가 급변하는데,
        ABD 는 이를 물체 경계로 오인해 콘을 조각낸다.

        ranges / angles 를 주지 않으면 points 에서 복원한다.
        기존 호출 방식(points 하나만 전달)과 호환된다.
        """
        if points.shape[0] == 0:
            return []

        if ranges is None or angles is None:
            work = points.astype(np.float64)
            radius = np.linalg.norm(work, axis=1)
            phi = np.arctan2(work[:, 1], work[:, 0])
        else:
            radius = ranges.astype(np.float64)
            phi = angles.astype(np.float64)

        range_max = self.effective_range_max()
        in_gate = (radius >= self.range_gate_min_m) & (radius <= range_max)
        radius = radius[in_gate]
        phi = phi[in_gate]

        if radius.size < self.cluster_min_points:
            return []

        # preprocess_scan 의 마스킹 결과는 각도 순이 아니다.
        # FOV 필터가 wrap 된 각도를 쓰므로 배열 중간에 각도 점프가 생기고,
        # 배열의 첫 점과 마지막 점이 실제로는 이웃이 된다.
        # 정렬하지 않으면 0도 방향에 놓인 물체가 두 조각으로 찢어진다.
        order = np.argsort(phi)
        radius = radius[order]
        phi = phi[order]

        delta_phi = np.diff(phi)

        # 관문 1: 각도 간격
        if self.angle_increment > 0.0:
            beam_gap = np.rint(delta_phi / self.angle_increment).astype(np.int64)
            break_angle = beam_gap > self.cluster_max_beam_gap
        else:
            break_angle = np.zeros(radius.size - 1, dtype=bool)

        # 관문 2: 코사인법칙 거리
        near = radius[:-1]
        far = radius[1:]
        squared = near ** 2 + far ** 2 - 2.0 * near * far * np.cos(delta_phi)
        separation = np.sqrt(np.maximum(squared, 0.0))
        threshold = (
            self.cluster_gap_base_m
            + self.cluster_gap_span_factor
            * np.maximum(near, far)
            * self.angle_increment
        )
        break_range = separation > threshold

        split_at = np.flatnonzero(break_angle | break_range) + 1
        index_groups = np.split(np.arange(radius.size), split_at)

        clusters: List[np.ndarray] = []
        for group in index_groups:
            if group.size < self.cluster_min_points:
                continue
            group_r = radius[group]
            group_p = phi[group]
            clusters.append(
                np.column_stack(
                    (group_r * np.cos(group_p), group_r * np.sin(group_p))
                )
            )
        return clusters

    # ------------------------------------------------------------------
    # TODO 2: 콘 판정 (벽 배제)
    # ------------------------------------------------------------------

    @staticmethod
    def fit_circle(cluster: np.ndarray) -> Tuple[float, float, float, float]:
        """Kasa 대수적 원 피팅. (cx, cy, radius, rms 잔차) 반환.

        원의 방정식 (x-a)^2 + (y-b)^2 = R^2 을 전개하면
            x^2 + y^2 = 2ax + 2by + (R^2 - a^2 - b^2)
        이 되어 미지수에 대해 선형이므로 최소제곱 한 번으로 풀린다.
        """
        x = cluster[:, 0]
        y = cluster[:, 1]

        design = np.column_stack((x, y, np.ones(x.size)))
        target = x ** 2 + y ** 2

        try:
            solution, *_ = np.linalg.lstsq(design, target, rcond=None)
        except np.linalg.LinAlgError:
            return 0.0, 0.0, float("inf"), float("inf")

        cx = float(solution[0]) / 2.0
        cy = float(solution[1]) / 2.0
        squared = float(solution[2]) + cx ** 2 + cy ** 2

        if not math.isfinite(squared) or squared <= 0.0:
            return cx, cy, float("inf"), float("inf")

        radius = math.sqrt(squared)
        residual = np.abs(np.hypot(x - cx, y - cy) - radius)
        rms = float(np.sqrt(np.mean(residual ** 2)))

        return cx, cy, radius, rms

    def detect_cones(self, clusters: List[np.ndarray]) -> List[Dict]:
        """콘의 실제 기하와 일치하는 클러스터만 남긴다.

        네 단계 검사:
          1. 폭     - 콘 지름보다 크게 벗어나면 벽
          2. 점 개수 - 이 거리에서 콘이라면 찍혀야 할 개수와 비교
          3. 반지름  - 원을 피팅해 콘 반지름과 일치하는지
          4. 잔차   - 실제로 원 모양인지 (점 4개 이상일 때만 유효)
        """
        cones: List[Dict] = []
        radius_expected = self.cone_diameter_m / 2.0
        radius_low = radius_expected * (1.0 - self.cone_diameter_tolerance)
        radius_high = radius_expected * (1.0 + self.cone_diameter_tolerance)
        width_max = self.cone_diameter_m * self.cone_width_margin

        for cluster in clusters:
            count = int(cluster.shape[0])
            centroid = cluster.mean(axis=0)
            distance = float(np.linalg.norm(centroid))
            width = float(np.linalg.norm(cluster[0] - cluster[-1]))

            cx, cy, radius, rms = self.fit_circle(cluster)

            if self.calibration_mode:
                self.get_logger().info(
                    f"  [cal] n={count:3d} d={distance:5.2f}m "
                    f"width={width:.3f}m fit_diameter={2.0 * radius:.3f}m "
                    f"rms={rms:.4f}m"
                )

            if distance <= 1e-6 or width > width_max:
                continue

            if self.angle_increment > 0.0:
                expected = (
                    2.0
                    * math.atan2(radius_expected, distance)
                    / self.angle_increment
                )
                if count < expected * self.cone_point_tolerance:
                    continue

            if not (radius_low <= radius <= radius_high):
                continue

            # 점이 3개면 원이 정확히 하나로 결정되어 잔차가 항상 0이 된다.
            # 이 경우 잔차 검사는 정보가 없으므로 건너뛴다.
            if count >= 4 and rms > self.cone_fit_rms_max_m:
                continue

            cones.append(
                {
                    "center": np.array([cx, cy], dtype=np.float64),
                    "radius": radius,
                    "rms": rms,
                    "width": width,
                    "count": count,
                    "distance": float(math.hypot(cx, cy)),
                }
            )

        cones.sort(key=lambda c: c["distance"])
        return cones

    # ------------------------------------------------------------------
    # 출력
    # ------------------------------------------------------------------

    def build_placeholder_path(
        self,
        msg: LaserScan,
        x: float,
        y: float,
    ) -> Path:
        """Build a one-point Path.

        TODO 4 (generate_center_path) 가 구현되면 이 메서드는
        중심 경로 전체를 담은 Path 로 대체된다.
        """
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

    def publish_detections(self, msg: LaserScan, cones: List[Dict]) -> None:
        """콘 목록을 (N x 4) 행렬로 발행한다. ros2 topic echo 로 읽을 수 있다."""
        detection_msg = Float32MultiArray()

        rows = MultiArrayDimension()
        rows.label = "cone"
        rows.size = len(cones)
        rows.stride = len(cones) * len(self.DETECTION_LAYOUT)

        cols = MultiArrayDimension()
        cols.label = ",".join(self.DETECTION_LAYOUT)
        cols.size = len(self.DETECTION_LAYOUT)
        cols.stride = len(self.DETECTION_LAYOUT)

        detection_msg.layout.dim = [rows, cols]
        detection_msg.layout.data_offset = 0

        flat: List[float] = []
        for cone in cones:
            flat.extend(
                [
                    float(cone["center"][0]),
                    float(cone["center"][1]),
                    float(cone["radius"]),
                    float(cone["count"]),
                ]
            )
        detection_msg.data = flat

        self.detections_pub.publish(detection_msg)

    def publish_cone_markers(self, msg: LaserScan, cones: List[Dict]) -> None:
        """RViz 시각화용 MarkerArray. 매 프레임 전체를 교체한다."""
        marker_array = MarkerArray()

        clear_marker = Marker()
        clear_marker.header.frame_id = msg.header.frame_id
        clear_marker.header.stamp = msg.header.stamp
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        for index, cone in enumerate(cones):
            marker = Marker()
            marker.header.frame_id = msg.header.frame_id
            marker.header.stamp = msg.header.stamp
            marker.ns = "cones"
            marker.id = index
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = float(cone["center"][0])
            marker.pose.position.y = float(cone["center"][1])
            marker.pose.position.z = 0.10
            marker.pose.orientation.w = 1.0
            marker.scale.x = float(cone["radius"]) * 2.0
            marker.scale.y = float(cone["radius"]) * 2.0
            marker.scale.z = 0.20
            marker.color.r = 1.0
            marker.color.g = 0.45
            marker.color.b = 0.0
            marker.color.a = 0.85
            marker_array.markers.append(marker)

        self.markers_pub.publish(marker_array)

    def log_summary(
        self,
        cluster_count: int,
        cone_count: int,
        nearest,
        valid_ratio: float,
        point_count: int,
    ) -> None:
        """log_period_s 간격으로만 요약을 출력한다. 10 Hz 도배를 막는다."""
        now = self.get_clock().now().nanoseconds * 1e-9

        if self.log_period_s > 0.0 and not self.calibration_mode:
            if now - self.last_log_time < self.log_period_s:
                return
        self.last_log_time = now

        head = (
            f"points={point_count} ratio={valid_ratio:.3f} "
            f"clusters={cluster_count} cones={cone_count} "
            f"gate=[{self.range_gate_min_m:.2f},"
            f"{self.effective_range_max():.2f}]m"
        )

        if nearest is None:
            self.get_logger().info(head + " nearest=none")
            return

        self.get_logger().info(
            head
            + f" nearest=({nearest['center'][0]:.3f},{nearest['center'][1]:.3f})"
            + f" d={nearest['distance']:.3f}m"
            + f" dia={2.0 * nearest['radius']:.3f}m"
            + f" rms={nearest['rms']:.4f}m"
            + f" n={nearest['count']}"
        )

    # ------------------------------------------------------------------
    # 남은 TODO
    # ------------------------------------------------------------------

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