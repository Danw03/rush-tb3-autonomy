"""detect_cones() 의 n<=4 하이브리드 판정 오프라인 단위 테스트.

배경(cone_perception_node.py 의 fit_fixed_radius / detect_cones docstring과
동일한 실측 근거): calibration_mode 로그를 트랙 bag(lidar_drive_01/02)로
30초 이상 수집해 확인한 결과,

  * n=3 은 세 점이 원을 유일하게 결정하므로 fit_circle()(자유 반지름)
    잔차가 항상 0이다 - 반지름이 아무리 틀려도 자유 반지름 잔차만으로는
    절대 걸러낼 수 없다.
  * n=4 도 반지름이 허용 범위를 벗어난 후보 중 47.9%가 자유 반지름
    잔차만으로는 낮게(cone_fit_rms_max_m 이하로) 나와 걸러지지 않았다.

이 파일은 n<=4 일 때 fit_fixed_radius()(반지름 고정) 잔차로 판정이
바뀐 뒤 이 두 실패 유형이 실제로 걸러지는지, 그리고 n>=5 판정은 여전히
자유 반지름 잔차만 쓰고 fit_fixed_radius() 를 호출조차 하지 않는지를
검증한다.

클러스터는 실제 LaserScan/클러스터링 없이, detect_cones() 가 받는
(N, 2) numpy 배열을 원의 방정식으로 직접 합성한다. 로봇 원점을 향한
원호 조각 모양이라 실제 클러스터와 동일한 형태다.
"""

import math

import numpy as np
import pytest
import rclpy

from tb3_cone_perception.cone_perception_node import ConePerceptionNode

CONE_DIAMETER_M = 0.085
RADIUS_EXPECTED = CONE_DIAMETER_M / 2.0


def _arc_cluster(radius: float, center_x: float, half_angle: float, n: int) -> np.ndarray:
    """반지름 radius 인 원의 로봇 정면 호에서 n 개 점을 균등 각도로 뽑는다.

    center_x 는 원 중심의 x 좌표(정면으로 center_x 만큼), half_angle 은
    원 중심 기준 호의 절반각(라디안)이다. 점들은 그 원 위에 정확히
    놓이므로, free-fit(Kasa)이 이 원을 그대로(잔차 0에 가깝게) 복원한다 -
    노이즈 없는 이상적인 경우로도 n<=4 함정을 재현하기에 충분하다.
    """
    thetas = np.linspace(-half_angle, half_angle, n) if n > 1 else np.array([0.0])
    center = np.array([center_x, 0.0])
    return np.array(
        [center + radius * np.array([-math.cos(t), math.sin(t)]) for t in thetas]
    )


# 실측(위 docstring)으로 확인한 "반지름은 허용 범위 안인데 실제로는 틀림"
# 함정을 재현하는 파라미터. radius_low/high=[0.02465, 0.06035] 안쪽 끝에
# 걸치는 0.060 을 자유 반지름으로 갖는 원에서 뽑은 점들은 free-fit 잔차가
# 거의 0이지만, 반지름을 진짜 콘 반지름(0.0425)으로 고정해 피팅하면
# 잔차가 cone_fit_rms_max_m(기본 0.010) 을 넘는다.
_WRONG_RADIUS = 0.060
_WRONG_HALF_ANGLE = 1.35
_WRONG_CENTER_X = 1.0

_TRUE_HALF_ANGLE = 0.4
_TRUE_CENTER_X = 1.0


@pytest.fixture(scope="module", autouse=True)
def _rclpy_context():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture()
def node():
    n = ConePerceptionNode()
    n.reload_parameters()
    n.angle_increment = math.radians(1.0)  # LDS-01 1도 분해능
    yield n
    n.destroy_node()


# ----------------------------------------------------------------------
# n<=4: 반지름이 맞는 진짜 콘은 여전히 통과해야 한다 (회귀 방지)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [3, 4])
def test_true_cone_accepted_for_low_n(node, n):
    cluster = _arc_cluster(RADIUS_EXPECTED, _TRUE_CENTER_X, _TRUE_HALF_ANGLE, n)

    cones = node.detect_cones([cluster])

    assert len(cones) == 1
    assert cones[0]["count"] == n
    assert cones[0]["radius"] == pytest.approx(RADIUS_EXPECTED, abs=1e-6)


# ----------------------------------------------------------------------
# n<=4: 반지름이 실제로는 틀렸지만 free-fit 잔차가 낮은 함정 케이스
# ----------------------------------------------------------------------


def test_n3_wrong_radius_low_free_rms_is_rejected(node):
    """실측으로 관찰된 'n=3, 잘못된 반지름, free rms~0' 케이스의 재현.

    자유 반지름 피팅으로는 반지름=0.060m(허용 범위 안), 잔차~0 이 나와
    옛 판정(n=3 이면 잔차 검사 자체를 건너뜀)이라면 무조건 통과했을
    클러스터다. 새 판정은 fit_fixed_radius() 잔차로 걸러야 한다.
    """
    cluster = _arc_cluster(_WRONG_RADIUS, _WRONG_CENTER_X, _WRONG_HALF_ANGLE, 3)

    # 함정 성립 전제: free-fit 이 실제로 허용 범위 안의 반지름을, 거의
    # 0에 가까운 잔차로 내놓는지 먼저 확인한다(이게 깨지면 아래 검증은
    # 의미가 없다).
    cx, cy, radius, rms = node.fit_circle(cluster)
    radius_low = RADIUS_EXPECTED * (1.0 - node.cone_diameter_tolerance)
    radius_high = RADIUS_EXPECTED * (1.0 + node.cone_diameter_tolerance)
    assert radius_low <= radius <= radius_high
    assert rms == pytest.approx(0.0, abs=1e-6)

    cones = node.detect_cones([cluster])

    assert cones == []


def test_n4_wrong_radius_low_free_rms_is_rejected(node):
    """같은 함정을 n=4 로 재현. 실측 트랙 bag 기준 이 유형이 47.9%였다."""
    cluster = _arc_cluster(_WRONG_RADIUS, _WRONG_CENTER_X, _WRONG_HALF_ANGLE, 4)

    cx, cy, radius, rms = node.fit_circle(cluster)
    radius_low = RADIUS_EXPECTED * (1.0 - node.cone_diameter_tolerance)
    radius_high = RADIUS_EXPECTED * (1.0 + node.cone_diameter_tolerance)
    assert radius_low <= radius <= radius_high
    assert rms == pytest.approx(0.0, abs=1e-6)

    cones = node.detect_cones([cluster])

    assert cones == []


def test_n4_wrong_radius_fixed_rms_exceeds_threshold(node):
    """위 함정에서 실제로 fixed_rms 가 임계값을 넘는지 직접 확인한다."""
    cluster = _arc_cluster(_WRONG_RADIUS, _WRONG_CENTER_X, _WRONG_HALF_ANGLE, 4)

    _, _, fixed_rms = node.fit_fixed_radius(cluster, RADIUS_EXPECTED)

    assert fixed_rms > node.cone_fit_rms_max_m


# ----------------------------------------------------------------------
# n>=5: 기존 자유 반지름 판정을 그대로 쓴다 - fit_fixed_radius() 를
# 호출조차 하지 않아야 한다 (calibration_mode=False 기준).
# ----------------------------------------------------------------------


def test_n5_true_cone_still_accepted(node):
    cluster = _arc_cluster(RADIUS_EXPECTED, _TRUE_CENTER_X, _TRUE_HALF_ANGLE, 5)

    cones = node.detect_cones([cluster])

    assert len(cones) == 1
    assert cones[0]["count"] == 5


def test_n5_judgment_does_not_call_fit_fixed_radius(node, monkeypatch):
    """n>=5 구간은 실측 검증이 아직 없으므로 건드리지 않는다는 계약을
    코드 차원에서 고정한다. fit_fixed_radius() 가 예외를 던지도록 만들어,
    n=5 판정 경로가 그 함수를 전혀 쓰지 않음을 증명한다."""

    def _boom(*args, **kwargs):
        raise AssertionError("n>=5 판정에서 fit_fixed_radius() 가 호출됨")

    monkeypatch.setattr(node, "fit_fixed_radius", _boom)
    assert node.calibration_mode is False

    cluster = _arc_cluster(RADIUS_EXPECTED, _TRUE_CENTER_X, _TRUE_HALF_ANGLE, 5)

    cones = node.detect_cones([cluster])

    assert len(cones) == 1
