"""extract_features() 오프라인 단위 테스트 + scan_callback 연동 테스트.

extract_features() 는 split_left_right() 이전의 원시 cones 리스트(거리순)를
받아 최근접 콘 정보를 계산해야 한다. split_left_right()/ConeSideTracker 는
정면 근처(데드밴드)에 새로 나타난 콘을 좌우 어느 쪽에도 배정하지 않고
버릴 수 있으므로(cone_perception_node.py 의 "알려진 한계" 참고), 분류
이후 리스트에서 nearest 를 뽑으면 안전 관련 값인 minimum_distance_m 이
실제보다 더 먼 콘을 가리키거나 아예 값을 놓칠 수 있다. 이 파일은 그
회귀를 막는다.
"""

import math

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan

from tb3_cone_perception.cone_perception_node import ConePerceptionNode

Y_DEADBAND = 0.03


def _cone(x: float, y: float) -> dict:
    return {
        "center": np.array([x, y], dtype=np.float64),
        "radius": 0.042,
        "rms": 0.001,
        "width": 0.08,
        "count": 5,
        "distance": float(math.hypot(x, y)),
    }


def _path_with_poses(n: int) -> Path:
    path = Path()
    for i in range(n):
        pose = PoseStamped()
        pose.pose.position.x = float(i)
        path.poses.append(pose)
    return path


def _make_empty_scan() -> LaserScan:
    """유효 측정값이 하나도 없는 스캔. preprocess_scan() 단계에서 걸린다."""
    msg = LaserScan()
    msg.header.frame_id = "base_scan"
    msg.angle_min = -math.pi
    msg.angle_max = math.pi
    msg.angle_increment = math.radians(1.0)
    msg.range_min = 0.12
    msg.range_max = 3.5
    msg.ranges = []
    return msg


def _make_wall_scan() -> LaserScan:
    """정면 전체가 벽처럼 넓은 물체 하나뿐인 스캔.

    유효 점은 있지만(preprocess_scan 통과) 폭이 cone_width_margin 을 훨씬
    넘어 detect_cones() 의 폭 게이트에서 전부 걸러져 cones=[] 가 된다.
    """
    n = 360
    msg = LaserScan()
    msg.header.frame_id = "base_scan"
    msg.angle_min = -math.pi
    msg.angle_max = math.pi - math.radians(1.0)
    msg.angle_increment = math.radians(1.0)
    msg.range_min = 0.12
    msg.range_max = 3.5
    msg.ranges = [1.0] * n
    return msg


@pytest.fixture(scope="module", autouse=True)
def _rclpy_context():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture()
def node():
    n = ConePerceptionNode()
    # split_left_right() 를 scan_callback 을 거치지 않고 직접 호출하는
    # 테스트가 있으므로, reload_parameters() 가 정상적으로 채워줄 속성을
    # test_split_and_path.py 와 동일하게 미리 세팅해 둔다.
    n.y_deadband_m = Y_DEADBAND
    yield n
    n.destroy_node()


# ----------------------------------------------------------------------
# nearest_* 는 split 이전 원시 cones 기준이어야 한다
# ----------------------------------------------------------------------


def test_nearest_uses_raw_cones_not_split_result(node):
    """데드밴드에 걸려 split 결과에서는 사라지는 정면 콘이 실제로는 가장
    가까운 콘인 시나리오. extract_features 는 split 을 거치지 않고 원시
    cones 에서 직접 nearest 를 뽑아야 한다."""
    near_deadband_cone = _cone(0.5, 0.01)  # |y|=0.01 < y_deadband_m(0.03)
    far_left_cone = _cone(1.2, 0.20)

    cones = sorted([near_deadband_cone, far_left_cone], key=lambda c: c["distance"])

    # split 결과에서는 정면 콘이 보류(None)돼 사라진다는 걸 먼저 확인한다.
    left_out, right_out = node.split_left_right(cones)
    assert len(left_out) + len(right_out) == 1
    assert all(c is not near_deadband_cone for c in left_out + right_out)

    path = _path_with_poses(1)
    features = node.extract_features(cones, path, 1.0, 360)

    assert features[0] == 1.0  # path_valid
    assert features[1] == pytest.approx(math.hypot(0.5, 0.01))  # minimum_distance_m
    assert features[2] == pytest.approx(math.atan2(0.01, 0.5))  # nearest_angle_rad
    assert features[3] == pytest.approx(0.5)  # nearest_x_m
    assert features[4] == pytest.approx(0.01)  # nearest_y_m
    assert features[5] == pytest.approx(1.0)  # valid_scan_ratio
    assert features[6] == pytest.approx(360.0)  # valid_point_count


def test_nearest_picks_min_distance_regardless_of_list_order(node):
    cones = [_cone(2.0, 0.3), _cone(0.8, -0.1), _cone(1.5, 0.5)]
    path = _path_with_poses(1)

    features = node.extract_features(cones, path, 0.9, 300)

    assert features[1] == pytest.approx(math.hypot(0.8, -0.1))
    assert features[3] == pytest.approx(0.8)
    assert features[4] == pytest.approx(-0.1)


# ----------------------------------------------------------------------
# path_valid: len(path.poses) 로만 판단
# (기존 인라인 코드는 이 분기에 진입하면 무조건 1.0 을 넣던 버그였다)
# ----------------------------------------------------------------------


def test_path_valid_false_when_cones_exist_but_path_empty(node):
    """콘이 있어도 generate_center_path 가 빈 Path 를 반환하면 path_valid
    는 0 이어야 한다. nearest 값 자체는 cones 기준이므로 여전히 유효해야
    한다 - path_valid 와 nearest 는 서로 독립적인 값이다."""
    cones = [_cone(0.5, 0.2)]
    empty_path = Path()

    features = node.extract_features(cones, empty_path, 1.0, 360)

    assert features[0] == 0.0
    assert features[1] == pytest.approx(math.hypot(0.5, 0.2))


def test_path_valid_true_when_path_has_poses(node):
    cones = [_cone(0.5, 0.2)]
    path = _path_with_poses(3)

    features = node.extract_features(cones, path, 1.0, 360)

    assert features[0] == 1.0


# ----------------------------------------------------------------------
# 콘 0개: extract_features 자체의 폴백 값 + scan_callback 조기 리턴
# ----------------------------------------------------------------------


def test_extract_features_fallback_when_cones_empty(node):
    """extract_features 를 직접 콘 0개로 호출했을 때의 폴백 값. 실제
    scan_callback 은 이 경우 아래 두 테스트처럼 extract_features 를 아예
    호출하지 않지만, 메서드 자체의 계약도 별도로 보증해 둔다."""
    features = node.extract_features([], Path(), 0.5, 100)

    assert features[0] == 0.0
    assert features[1] == float("inf")
    assert features[2] == 0.0
    assert features[3] == 0.0
    assert features[4] == 0.0
    assert features[5] == pytest.approx(0.5)
    assert features[6] == pytest.approx(100.0)


def test_scan_callback_skips_extract_features_when_scan_has_no_valid_points(
    node, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        node, "extract_features", lambda *a, **k: calls.append(1)
    )

    node.scan_callback(_make_empty_scan())

    assert calls == []


def test_scan_callback_skips_extract_features_when_no_cones_detected(
    node, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        node, "extract_features", lambda *a, **k: calls.append(1)
    )

    node.scan_callback(_make_wall_scan())

    assert calls == []


# ----------------------------------------------------------------------
# resolve() 는 프레임당 정확히 한 번만 불려야 한다
# ----------------------------------------------------------------------


def test_scan_callback_calls_resolve_and_extract_features_exactly_once(
    node, monkeypatch
):
    """extract_features 연동이 ConeSideTracker.resolve() 를 프레임당 두
    번 호출하게 만들지 않았는지 확인한다 (이력 오염 방지).

    detect_cones() 를 고정된 콘 목록으로 갈아끼워, 실제 콘 형상을 만족하는
    스캔을 합성하는 수고 없이 scan_callback 의 나머지 흐름만 검증한다.
    """
    fixed_cones = [_cone(1.0, 0.2)]
    monkeypatch.setattr(node, "detect_cones", lambda clusters: fixed_cones)

    resolve_calls = []
    original_resolve = node.cone_tracker.resolve

    def spy_resolve(*args, **kwargs):
        resolve_calls.append(1)
        return original_resolve(*args, **kwargs)

    monkeypatch.setattr(node.cone_tracker, "resolve", spy_resolve)

    extract_calls = []
    original_extract = node.extract_features

    def spy_extract(*args, **kwargs):
        extract_calls.append(1)
        return original_extract(*args, **kwargs)

    monkeypatch.setattr(node, "extract_features", spy_extract)

    node.scan_callback(_make_wall_scan())

    assert len(resolve_calls) == 1
    assert len(extract_calls) == 1
