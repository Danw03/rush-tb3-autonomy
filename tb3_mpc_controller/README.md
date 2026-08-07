# tb3_mpc_controller

TurtleBot3용 선형화 반복형 convex MPC 경로 추종 패키지입니다.

## 인터페이스

구독:

- `/odom` (`nav_msgs/msg/Odometry`)
- `/reference_path` (`nav_msgs/msg/Path`)
- `/reference_speed` (`std_msgs/msg/Float32`, 선택 사항)

발행:

- `/cmd_vel_raw` (`geometry_msgs/msg/TwistStamped`)
- `/mpc_predicted_path` (`nav_msgs/msg/Path`)
- `/mpc_reference_horizon` (`nav_msgs/msg/Path`)

모든 토픽 이름은 `config/mpc_controller.yaml`에서 바꿀 수 있습니다.
`/reference_speed`가 오지 않으면 `reference_speed_default`를 사용합니다.

## MPC 구성

- 상태: `[x, y, yaw]`
- 입력: `[v, omega]`
- 모델: exact-discretized unicycle model
- 기본 제어주기: `0.1 s`
- 기본 horizon: `20 steps`
- 기본 선형화 반복: `2`
- QP: Eigen LDLT 기반 dense ADMM solver
- 입력 제한, 각속도 제한, 선속도/각속도 변화율 제한 포함

## 빌드

```bash
cd ~/tb3_project
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select tb3_mpc_controller
source install/setup.bash
```

## 실행

```bash
ros2 launch tb3_mpc_controller mpc_controller.launch.py
```

직접 실행하려면:

```bash
ros2 run tb3_mpc_controller mpc_controller_node \
  --ros-args \
  --params-file ~/tb3_project/src/tb3_autonomy/tb3_mpc_controller/config/mpc_controller.yaml
```

## 입력 조건

- reference path의 `header.frame_id`는 odometry의 `header.frame_id`와 같아야 합니다.
- reference path는 pose가 최소 2개여야 합니다.
- 기본 `input_timeout`이 `0.5 s`이므로 `/odom`과 `/reference_path`는 2 Hz보다 빠르게 갱신해야 합니다.
- `/reference_speed`를 한 번이라도 발행하면 이후에도 timeout보다 빠르게 갱신해야 합니다. 속도 토픽을 쓰지 않으면 기본 속도가 계속 적용됩니다.
- 경로의 각 `PoseStamped.orientation`에는 해당 지점의 진행 방향 yaw가 들어 있어야 곡선 추종이 정상 동작합니다.

입력이 없거나 오래되었거나 frame이 다르거나 solver 예외가 발생하면 0 속도를 발행합니다.
