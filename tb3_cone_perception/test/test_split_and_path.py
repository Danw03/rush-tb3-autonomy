"""split_left_right / generate_center_path 오프라인 시나리오 테스트.

LiDAR_샘플_데이터.txt 나 실기 없이, detect_cones() 가 반환할 법한 콘 목록을
직접 합성해 두 메서드만 단위로 검증한다. 시나리오: 직선, 완만한 좌/우 커브,
급한 좌/우 커브, 헤어핀(내측 반경 3종), 편측만 보임(좌/우), 콘 1개, 콘 0개.
판정 기준: 경계혼합(좌우 콘 개수 불일치), 자기교차, 급점프, 중심선이탈.

주의 -- y-부호 분류의 시야각 한계 (구현 버그 아님, 실측으로 확인한 특성):
  split_left_right() 는 REP103 y-부호(반평면)로 좌/우를 나눈다. 로봇 헤딩
  기준으로 트랙이 일정 각도 이상 휘면, "물리적으로 오른쪽"인 콘의 좌표가
  로봇 프레임에서 y>0 으로 넘어가 왼쪽으로 잘못 분류된다. 이 한계각은
  반경이 작을수록(곡률이 클수록) 좁아진다 (track_width_m=0.6 기준 실측:
  R=3.0m -> 약 23도, R=1.0m -> 약 34도, R=0.5m -> 약 34도).
  아래 트랙 생성 파라미터는 전부 이 한계 안에서 골랐다. 이 한계를 넘는
  스윕은 애초에 한 프레임에서 y-부호로 올바르게 분류될 수 없으므로
  "실패해야 정상"이며, 이 파일에서 검증 대상이 아니다.
"""

import math

import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import LaserScan

from tb3_cone_perception.cone_perception_node import ConePerceptionNode

TRACK_WIDTH = 0.6
CONE_SPACING = 0.5
Y_DEADBAND = 0.03

# 인접 웨이포인트 간 최대 허용 간격. 콘 간격의 3배를 넘으면 급점프로 본다.
MAX_GAP_M = CONE_SPACING * 3.0
# 양쪽이 다 보일 때는 좌우 콘의 (인덱스) 페어링만으로 중점이 정확히 계산되므로
# (원호를 만든 방식 그대로 역산되어 부동소수점 오차 수준) 허용오차를 작게 둔다.
CENTERLINE_TOL_M = 0.01


def _sagitta_tol(spacing: float, radius: float, margin: float = 1.6) -> float:
    """편측만 보일 때(가상 콘 미러링) 기대되는 근사오차 상한.

    체인 끝점 접선은 중앙차분이 아니라 편차분(전진/후진차분)으로 근사하므로,
    원호 곡률이 있으면 실제 접선과 어긋난다. 이 어긋남은 활 모양 오차
    (sagitta) 공식 spacing^2 / (2*radius) 크기와 같은 order 로 나타난다
    (직접 측정해 확인함: R=3.0/spacing=0.3 -> 실측 1.5cm, 공식 1.5cm 등).
    margin 은 원호 위치에 따른 편차를 흡수하기 위한 여유.
    """
    return margin * (spacing ** 2) / (2.0 * radius)


# ----------------------------------------------------------------------
# 합성 트랙 생성
#
# cone_perception_node._mirror_centers() 와 동일한 회전 규약을 쓴다:
#   left_perp  = (-t_y, t_x)  (진행방향을 +90도 회전, 왼쪽)
#   right_perp = ( t_y, -t_x) (진행방향을 -90도 회전, 오른쪽)
# 이렇게 만든 좌/우 콘으로부터 역산한 중심선이 생성 결과와 정확히 겹쳐야
# 알고리즘이 올바르다고 볼 수 있다.
# ----------------------------------------------------------------------


def _cone(x: float, y: float) -> dict:
    return {
        "center": np.array([x, y], dtype=np.float64),
        "radius": 0.042,
        "rms": 0.001,
        "width": 0.08,
        "count": 5,
        "distance": float(math.hypot(x, y)),
    }


def _straight_track(n_pairs: int, spacing: float = CONE_SPACING, width: float = TRACK_WIDTH):
    """직선 트랙. centerline, left_cones, right_cones 를 반환한다."""
    centerline = np.array(
        [[spacing * (i + 1), 0.0] for i in range(n_pairs)], dtype=np.float64
    )
    left = [(_cone(x, width / 2.0)) for x, _ in centerline]
    right = [(_cone(x, -width / 2.0)) for x, _ in centerline]
    return centerline, left, right


def _arc_track(
    radius: float,
    turn: str,
    n_pairs: int,
    spacing: float = CONE_SPACING,
    width: float = TRACK_WIDTH,
):
    """좌/우로 굽는 원호 트랙. turn 은 'left' 또는 'right'."""
    theta_step = spacing / radius
    thetas = np.array([theta_step * (i + 1) for i in range(n_pairs)])

    if turn == "left":
        center_c = np.array([0.0, radius])
        points = center_c + radius * np.column_stack((np.sin(thetas), -np.cos(thetas)))
        tangents = np.column_stack((np.cos(thetas), np.sin(thetas)))
    else:
        center_c = np.array([0.0, -radius])
        points = center_c + radius * np.column_stack((np.sin(thetas), np.cos(thetas)))
        tangents = np.column_stack((np.cos(thetas), -np.sin(thetas)))

    left_perp = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    right_perp = np.column_stack((tangents[:, 1], -tangents[:, 0]))

    left = [_cone(x, y) for x, y in points + (width / 2.0) * left_perp]
    right = [_cone(x, y) for x, y in points + (width / 2.0) * right_perp]
    return points, left, right


def _hairpin_track(inner_radius: float, n_pairs: int, turn: str = "left"):
    """헤어핀: 반경이 작아 곡률이 큰 트랙.

    spacing=0.15m 로 촘촘히 찍는다. 한 프레임에서 y-부호로 안전하게
    분류되는 스윕각은 반경이 작을수록 좁아지므로(아래 모듈 docstring 참고),
    n_pairs 는 반경별로 그 한계 안에서 고른 값이다.
    """
    return _arc_track(inner_radius, turn, n_pairs, spacing=0.15)


def _sorted_by_distance(*cone_lists):
    """detect_cones() 출력 계약(거리순 정렬)을 흉내낸다."""
    merged = [c for lst in cone_lists for c in lst]
    merged.sort(key=lambda c: c["distance"])
    return merged


def _waypoints(path_msg) -> np.ndarray:
    return np.array(
        [[p.pose.position.x, p.pose.position.y] for p in path_msg.poses],
        dtype=np.float64,
    )


# ----------------------------------------------------------------------
# 판정 기준 헬퍼
# ----------------------------------------------------------------------


def assert_no_sudden_jump(points: np.ndarray, max_gap: float = MAX_GAP_M) -> None:
    if points.shape[0] < 2:
        return
    gaps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert np.all(gaps <= max_gap), f"급점프 발생: gaps={gaps}"


def _segments_intersect(p1, p2, p3, p4) -> bool:
    def orient(a, b, c):
        val = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if val > 1e-12:
            return 1
        if val < -1e-12:
            return -1
        return 0

    o1, o2 = orient(p1, p2, p3), orient(p1, p2, p4)
    o3, o4 = orient(p3, p4, p1), orient(p3, p4, p2)
    return o1 != o2 and o3 != o4 and o1 != 0 and o2 != 0 and o3 != 0 and o4 != 0


def assert_no_self_intersection(points: np.ndarray) -> None:
    n = points.shape[0]
    for i in range(n - 1):
        for j in range(i + 2, n - 1):
            if _segments_intersect(points[i], points[i + 1], points[j], points[j + 1]):
                raise AssertionError(f"자기교차 발생: segment {i}-{i+1} x {j}-{j+1}")


def assert_close_to_centerline(
    points: np.ndarray,
    centerline: np.ndarray,
    tol: float = CENTERLINE_TOL_M,
) -> None:
    for pt in points:
        dists = np.linalg.norm(centerline - pt, axis=1)
        nearest = float(np.min(dists))
        assert nearest <= tol, f"중심선이탈: point={pt} nearest_dist={nearest:.4f}"


# ----------------------------------------------------------------------
# 픽스처
# ----------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _rclpy_context():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture()
def node():
    n = ConePerceptionNode()
    n.y_deadband_m = Y_DEADBAND
    n.track_width_m = TRACK_WIDTH
    yield n
    n.destroy_node()


@pytest.fixture()
def scan_msg():
    msg = LaserScan()
    msg.header.frame_id = "base_scan"
    return msg


# ----------------------------------------------------------------------
# 시나리오: 양쪽 다 보임
# ----------------------------------------------------------------------


def _run_both_sides(node, scan_msg, centerline, left, right, tol=CENTERLINE_TOL_M):
    cones = _sorted_by_distance(left, right)
    left_out, right_out = node.split_left_right(cones)

    assert len(left_out) == len(left)
    assert len(right_out) == len(right)
    assert all(c["center"][1] > 0.0 for c in left_out)
    assert all(c["center"][1] < 0.0 for c in right_out)

    path_msg = node.generate_center_path(scan_msg, left_out, right_out)
    points = _waypoints(path_msg)

    assert points.shape[0] == len(centerline)
    assert_no_sudden_jump(points)
    assert_no_self_intersection(points)
    assert_close_to_centerline(points, centerline, tol=tol)
    return points


def test_straight(node, scan_msg):
    centerline, left, right = _straight_track(n_pairs=6)
    _run_both_sides(node, scan_msg, centerline, left, right)


def test_gentle_curve_left(node, scan_msg):
    centerline, left, right = _arc_track(radius=3.0, turn="left", n_pairs=4, spacing=0.3)
    _run_both_sides(node, scan_msg, centerline, left, right)


def test_gentle_curve_right(node, scan_msg):
    centerline, left, right = _arc_track(radius=3.0, turn="right", n_pairs=4, spacing=0.3)
    _run_both_sides(node, scan_msg, centerline, left, right)


def test_sharp_curve_left(node, scan_msg):
    centerline, left, right = _arc_track(radius=1.0, turn="left", n_pairs=4, spacing=0.15)
    _run_both_sides(node, scan_msg, centerline, left, right)


def test_sharp_curve_right(node, scan_msg):
    centerline, left, right = _arc_track(radius=1.0, turn="right", n_pairs=4, spacing=0.15)
    _run_both_sides(node, scan_msg, centerline, left, right)


@pytest.mark.parametrize(
    "inner_radius, n_pairs",
    [(0.5, 2), (0.9, 4), (1.3, 5)],
)
def test_hairpin_left(node, scan_msg, inner_radius, n_pairs):
    centerline, left, right = _hairpin_track(inner_radius, n_pairs, turn="left")
    # 양쪽 다 보이면 페어링만으로 정확히 계산되므로 허용오차는 그대로 둔다.
    _run_both_sides(node, scan_msg, centerline, left, right)


# ----------------------------------------------------------------------
# 시나리오: 경계혼합 (좌우 콘 개수 불일치)
# ----------------------------------------------------------------------


def test_mixed_boundary_more_left(node, scan_msg):
    centerline, left, right = _straight_track(n_pairs=6)
    right = right[:4]  # 오른쪽 콘 2개가 가려졌다고 가정

    cones = _sorted_by_distance(left, right)
    left_out, right_out = node.split_left_right(cones)
    path_msg = node.generate_center_path(scan_msg, left_out, right_out)
    points = _waypoints(path_msg)

    assert points.shape[0] == len(left)
    assert_no_sudden_jump(points)
    assert_no_self_intersection(points)
    # 페어링된 앞부분 4개는 정확히 중심선과 일치해야 한다.
    assert_close_to_centerline(points[:4], centerline[:4])
    # 나머지는 가상 미러링 결과이므로 실제 중심선과 정확히 일치해야 한다
    # (직선이라 tangent 가 일정해 미러링 결과 = 원래 중심선).
    assert_close_to_centerline(points[4:], centerline[4:])


def test_mixed_boundary_more_right(node, scan_msg):
    centerline, left, right = _straight_track(n_pairs=6)
    left = left[:3]  # 왼쪽 콘 3개가 가려졌다고 가정

    cones = _sorted_by_distance(left, right)
    left_out, right_out = node.split_left_right(cones)
    path_msg = node.generate_center_path(scan_msg, left_out, right_out)
    points = _waypoints(path_msg)

    assert points.shape[0] == len(right)
    assert_no_sudden_jump(points)
    assert_no_self_intersection(points)
    assert_close_to_centerline(points, centerline)


# ----------------------------------------------------------------------
# 시나리오: 편측만 보임
# ----------------------------------------------------------------------


def test_left_only_visible(node, scan_msg):
    radius, spacing = 3.0, 0.3
    centerline, left, right = _arc_track(radius=radius, turn="left", n_pairs=4, spacing=spacing)

    cones = _sorted_by_distance(left)
    left_out, right_out = node.split_left_right(cones)
    assert len(left_out) == len(left)
    assert len(right_out) == 0

    path_msg = node.generate_center_path(scan_msg, left_out, right_out)
    points = _waypoints(path_msg)

    assert points.shape[0] == len(left)
    assert_no_sudden_jump(points)
    assert_no_self_intersection(points)
    # 가상 콘 미러링은 접선을 편차분으로 근사하므로 sagitta 오차만큼 벗어난다.
    assert_close_to_centerline(points, centerline, tol=_sagitta_tol(spacing, radius))


def test_right_only_visible(node, scan_msg):
    radius, spacing = 3.0, 0.3
    centerline, left, right = _arc_track(radius=radius, turn="right", n_pairs=4, spacing=spacing)

    cones = _sorted_by_distance(right)
    left_out, right_out = node.split_left_right(cones)
    assert len(left_out) == 0
    assert len(right_out) == len(right)

    path_msg = node.generate_center_path(scan_msg, left_out, right_out)
    points = _waypoints(path_msg)

    assert points.shape[0] == len(right)
    assert_no_sudden_jump(points)
    assert_no_self_intersection(points)
    assert_close_to_centerline(points, centerline, tol=_sagitta_tol(spacing, radius))


# ----------------------------------------------------------------------
# 시나리오: 콘 1개 / 콘 0개
# ----------------------------------------------------------------------


def test_single_cone_left(node, scan_msg):
    cones = [_cone(1.0, 0.4)]
    left_out, right_out = node.split_left_right(cones)
    assert len(left_out) == 1
    assert len(right_out) == 0

    path_msg = node.generate_center_path(scan_msg, left_out, right_out)
    points = _waypoints(path_msg)
    assert points.shape[0] == 1

    # 진행방향은 원점 -> 콘 방향. 그 방향의 오른쪽 수직으로 track_width_m/2
    # 만큼 이동한 위치가 중심점이어야 한다.
    cone_xy = np.array([1.0, 0.4])
    heading = cone_xy / np.linalg.norm(cone_xy)
    right_perp = np.array([heading[1], -heading[0]])
    expected = cone_xy + (TRACK_WIDTH / 2.0) * right_perp
    assert np.allclose(points[0], expected, atol=1e-9)


def test_no_cones(node, scan_msg):
    left_out, right_out = node.split_left_right([])
    assert left_out == []
    assert right_out == []

    path_msg = node.generate_center_path(scan_msg, left_out, right_out)
    assert len(path_msg.poses) == 0


# ----------------------------------------------------------------------
# y_deadband_m 경계 처리
# ----------------------------------------------------------------------


def test_deadband_holds_cone_unassigned(node, scan_msg):
    cones = [
        _cone(1.0, 0.02),  # |y| < 0.03 -> 보류
        _cone(1.0, 0.05),  # 왼쪽
        _cone(1.0, -0.05),  # 오른쪽
        _cone(1.0, -0.01),  # |y| < 0.03 -> 보류
    ]
    left_out, right_out = node.split_left_right(cones)

    assert len(left_out) == 1
    assert left_out[0]["center"][1] == pytest.approx(0.05)
    assert len(right_out) == 1
    assert right_out[0]["center"][1] == pytest.approx(-0.05)


# ----------------------------------------------------------------------
# 시나리오: 이력 기반 완화 (ConeSideTracker)
# ----------------------------------------------------------------------


def test_gradual_flip_is_held_then_accepted(node):
    """실제 커브 진입 로그에서 관찰된 패턴 재현: 왼쪽 콘의 y가 서서히
    줄어 데드밴드에 잠깐 걸리고, 이후 반대쪽(오른쪽) 부호로 넘어간다.

    override_grace_s(기본 0.3s) 동안 반대쪽 판정이 연속으로 나오지
    않으면 직전 소속(왼쪽)을 유지해야 하고, 그 시간을 다 채우면 그제서야
    오른쪽으로 확정돼야 한다. x 는 물리적으로 같은 콘임을 나타내기 위해
    거의 고정해 둔다. 불일치 구간의 dt 는 0.3s 경계에 딱 걸리지 않도록
    (부동소수점 오차로 인한 flaky 방지) 0.12s 간격으로 잡았다.
    """
    x = 1.0
    frames = [
        (0.00, 0.050),   # 최초 관측 -> 왼쪽
        (0.20, 0.048),   # 왼쪽과 일치 -> 왼쪽 유지
        (0.40, 0.039),   # 왼쪽과 일치 -> 왼쪽 유지
        (0.60, 0.020),   # |y| < y_deadband_m(0.03) -> 판단 보류, 왼쪽 유지
        (0.72, -0.032),  # 반대쪽 판정 시작. disagree_s=0.12s < grace
        (0.84, -0.035),  # disagree_s=0.24s < grace, 아직 왼쪽 유지
        (0.96, -0.038),  # disagree_s=0.36s >= grace(0.3) -> 오른쪽으로 확정
        (1.08, -0.040),  # 오른쪽과 일치 -> 오른쪽 유지
    ]
    expected_side = ["L", "L", "L", "L", "L", "L", "R", "R"]

    for (t, y), side in zip(frames, expected_side):
        left_out, right_out = node.split_left_right([_cone(x, y)], timestamp_s=t)
        if side == "L":
            assert len(left_out) == 1, f"t={t}: 아직 왼쪽이어야 함"
            assert len(right_out) == 0, f"t={t}: 오른쪽엔 없어야 함"
        else:
            assert len(right_out) == 1, f"t={t}: 오른쪽으로 확정돼야 함"
            assert len(left_out) == 0, f"t={t}: 왼쪽엔 없어야 함"
