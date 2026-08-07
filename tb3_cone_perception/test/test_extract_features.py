"""extract_features() / ConeCountStabilityTracker 오프라인 단위 테스트.

2026-08-06, MPC 팀 요구사항 반영으로 FEATURE_LAYOUT 이 전면 교체됐다
(breaking change). 새 계약:

    0 cone_count           len(cones)
    1 nearest_distance_m   가장 가까운 콘까지 거리
    2 mean_x_m             콘들의 x좌표 평균
    3 mean_y_m             콘들의 y좌표 평균
    4 std_x_m              콘들의 x좌표 표준편차 (ddof=0)
    5 std_y_m              콘들의 y좌표 표준편차 (ddof=0)
    6 stable_frame_count   최근 10프레임 중 콘 개수>=2였던 프레임 수

기준 리스트는 split_left_right() 이전의 원시 cones(거리순)다.
split_left_right()/ConeSideTracker 는 정면 근처(데드밴드)에 새로 나타난
콘을 좌우 어느 쪽에도 배정하지 않고 버릴 수 있으므로(cone_perception_node
의 "알려진 한계" 참고), split 이후 리스트로 통계를 뽑으면 콘 하나를
통째로 놓칠 수 있다. 이 파일은 그 회귀를 막는다 - 기존 nearest_* 전용
설계 원칙을 cone_count/mean/std 에도 동일하게 적용한다.

path_valid 는 새 계약에 없다 - `/cone_path` 가 비어있으면(poses=0) 그
자체로 경로 무효를 나타내므로 features 에서 중복 제공하지 않는다는
가정이다(TODO: 이 가정을 MPC 팀과 재확인할 것).
"""

import math

import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import LaserScan

from tb3_cone_perception.cone_perception_node import (
    ConeCountStabilityTracker,
    ConePerceptionNode,
)


def _cone(x: float, y: float) -> dict:
    return {
        "center": np.array([x, y], dtype=np.float64),
        "radius": 0.042,
        "rms": 0.001,
        "width": 0.08,
        "count": 5,
        "distance": float(math.hypot(x, y)),
    }


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
    yield n
    n.destroy_node()


# ----------------------------------------------------------------------
# 콘 여러 개: cone_count / mean / std
# ----------------------------------------------------------------------


def test_multiple_cones_stats(node):
    cones = [_cone(1.0, 0.2), _cone(2.0, -0.4), _cone(1.5, 0.6)]
    xs = [1.0, 2.0, 1.5]
    ys = [0.2, -0.4, 0.6]

    features = node.extract_features(cones, stable_frame_count=3)

    assert features[0] == pytest.approx(3.0)  # cone_count
    assert features[1] == pytest.approx(math.hypot(1.0, 0.2))  # nearest_distance_m
    assert features[2] == pytest.approx(sum(xs) / 3.0)  # mean_x_m
    assert features[3] == pytest.approx(sum(ys) / 3.0)  # mean_y_m
    assert features[4] == pytest.approx(np.std(xs))  # std_x_m (ddof=0)
    assert features[5] == pytest.approx(np.std(ys))  # std_y_m
    assert features[6] == pytest.approx(3.0)  # stable_frame_count passthrough


def test_nearest_and_stats_pick_min_distance_regardless_of_list_order(node):
    cones = [_cone(2.0, 0.3), _cone(0.8, -0.1), _cone(1.5, 0.5)]

    features = node.extract_features(cones, stable_frame_count=0)

    assert features[1] == pytest.approx(math.hypot(0.8, -0.1))


def test_uses_raw_cones_not_split_result(node):
    """정면 콘도 cone_count/nearest/mean/std 모두에 반영돼야 한다.

    split 이후 사라지는 콘이라도 예외가 아니다 - 기존 nearest_* 설계
    원칙을 새 통계 지표에도 동일하게 적용한다.
    """
    near_deadband_cone = _cone(0.5, 0.01)  # |y|=0.01 < y_deadband_m(기본 0.03)
    far_left_cone = _cone(1.2, 0.20)
    cones = sorted(
        [near_deadband_cone, far_left_cone], key=lambda c: c["distance"]
    )

    node.reload_parameters()
    left_out, right_out = node.split_left_right(cones)
    assert len(left_out) + len(right_out) == 1  # 정면 콘은 split에서 사라짐

    features = node.extract_features(cones, stable_frame_count=0)

    assert features[0] == pytest.approx(2.0)  # cone_count: split과 무관
    assert features[1] == pytest.approx(math.hypot(0.5, 0.01))
    assert features[2] == pytest.approx((0.5 + 1.2) / 2.0)  # mean_x_m
    assert features[3] == pytest.approx((0.01 + 0.20) / 2.0)  # mean_y_m


# ----------------------------------------------------------------------
# 콘 1개: std = 0.0 (정상 계산 결과, 특별 처리 없음)
# ----------------------------------------------------------------------


def test_single_cone_std_is_zero(node):
    cones = [_cone(1.0, 0.3)]

    features = node.extract_features(cones, stable_frame_count=1)

    assert features[0] == pytest.approx(1.0)
    assert features[1] == pytest.approx(math.hypot(1.0, 0.3))
    assert features[2] == pytest.approx(1.0)  # mean_x_m
    assert features[3] == pytest.approx(0.3)  # mean_y_m
    assert features[4] == pytest.approx(0.0)  # std_x_m
    assert features[5] == pytest.approx(0.0)  # std_y_m


# ----------------------------------------------------------------------
# 콘 0개: sentinel 값
# ----------------------------------------------------------------------


def test_empty_cones_sentinel(node):
    features = node.extract_features([], stable_frame_count=7)

    assert features[0] == 0.0
    assert features[1] == float("inf")
    assert features[2] == 0.0
    assert features[3] == 0.0
    assert features[4] == 0.0
    assert features[5] == 0.0
    # stable_frame_count 는 sentinel 대상이 아니다 - 트래커 실측값 그대로.
    assert features[6] == pytest.approx(7.0)


def test_scan_callback_publishes_sentinel_when_no_valid_points(node):
    published = []
    node.features_pub.publish = lambda msg: published.append(list(msg.data))

    node.scan_callback(_make_empty_scan())

    assert len(published) == 1
    data = published[0]
    assert data[0] == 0.0
    assert data[1] == float("inf")
    assert data[2:6] == [0.0, 0.0, 0.0, 0.0]
    # 첫 콜백이자 콘<2 인 프레임이므로 stable_frame_count=0
    assert data[6] == pytest.approx(0.0)


def test_scan_callback_publishes_sentinel_when_no_cones_detected(node):
    published = []
    node.features_pub.publish = lambda msg: published.append(list(msg.data))

    node.scan_callback(_make_wall_scan())

    assert len(published) == 1
    data = published[0]
    assert data[0] == 0.0
    assert data[1] == float("inf")


# ----------------------------------------------------------------------
# ConeCountStabilityTracker: 부분 윈도우 / 슬라이딩(오래된 프레임 밀려남)
# ROS 의존 없는 순수 python 이라 노드 없이 직접 테스트한다.
# ----------------------------------------------------------------------


def test_stability_tracker_partial_window_before_full():
    """window(10) 이 채워지기 전에는 그때까지 쌓인 프레임만으로 계산한다."""
    tracker = ConeCountStabilityTracker(window=10)
    counts = [0, 1, 2, 3, 2, 0]  # 6프레임만 흘려보냄 (10 미만)

    results = [tracker.update(c) for c in counts]

    # >=2 인 프레임: 2, 3, 2 -> 3개
    assert results[-1] == 3
    assert results == [0, 0, 1, 2, 3, 3]


def test_stability_tracker_sliding_window_evicts_old_frames():
    """가장 오래된 프레임은 window(deque maxlen) 를 넘어서면 밀려난다."""
    tracker = ConeCountStabilityTracker(window=10)

    for _ in range(10):
        tracker.update(2)  # 안정 프레임 10개로 window 를 가득 채운다
    assert tracker.update(2) == 10  # 11번째도 안정 -> 여전히 10 (창 크기 고정)

    last = None
    for _ in range(9):
        last = tracker.update(0)  # 불안정 프레임 9개로 교체
    assert last == 1  # 안정 프레임이 1개(가장 최근 것)만 창에 남아 있어야 함

    final = tracker.update(0)  # 10번째 불안정 프레임 -> 마지막 안정 프레임도 밀려남
    assert final == 0


def test_stability_tracker_boundary_is_ge_two():
    tracker = ConeCountStabilityTracker(window=10)
    assert tracker.update(0) == 0
    assert tracker.update(1) == 0  # 1개는 불안정
    assert tracker.update(2) == 1  # 2개부터 안정


# ----------------------------------------------------------------------
# stability_tracker.update() 는 프레임당 정확히 한 번만 불려야 한다
# (ConeSideTracker.resolve() 와 동일한 이유 - 이력/카운트 이중 반영 방지)
# ----------------------------------------------------------------------


def test_stability_tracker_updated_exactly_once_per_callback(node, monkeypatch):
    calls = []
    original_update = node.stability_tracker.update

    def spy_update(cone_count):
        calls.append(cone_count)
        return original_update(cone_count)

    monkeypatch.setattr(node.stability_tracker, "update", spy_update)

    node.scan_callback(_make_empty_scan())
    assert calls == [0]

    calls.clear()
    node.scan_callback(_make_wall_scan())
    assert calls == [0]

    calls.clear()
    fixed_cones = [_cone(1.0, 0.2)]
    monkeypatch.setattr(node, "detect_cones", lambda clusters: fixed_cones)
    node.scan_callback(_make_wall_scan())
    assert calls == [1]


def test_scan_callback_calls_resolve_and_extract_features_exactly_once(
    node, monkeypatch
):
    """ConeSideTracker.resolve() 가 프레임당 정확히 한 번만 불리는지 확인한다.

    extract_features 연동이 이력 오염(resolve 중복 호출)을 일으키지
    않았는지 보는 회귀 테스트다. detect_cones() 를 고정된 콘 목록으로
    갈아끼워, 실제 콘 형상을 만족하는 스캔을 합성하는 수고 없이
    scan_callback 의 나머지 흐름만 검증한다.
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
