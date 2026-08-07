# tb3_reference

`tb3_cone_perception`의 로봇 좌표계 중심 경로를 `odom` 좌표계 MPC 기준
경로로 변환한다.

## ROS interface

| Direction | Topic | Type |
| --- | --- | --- |
| Subscribe | `/cone_path` | `nav_msgs/msg/Path` |
| Subscribe | `/cone_features` | `std_msgs/msg/Float32MultiArray` |
| Publish | `/reference_path` | `nav_msgs/msg/Path` |
| Publish | `/reference_speed` | `std_msgs/msg/Float32` |
| Publish | `/driving_mode` | `std_msgs/msg/String` |

`/cone_features`는 다음 7개 순서를 사용한다.

1. `cone_count`
2. `nearest_distance_m`
3. `mean_x_m`
4. `mean_y_m`
5. `std_x_m`
6. `std_y_m`
7. `stable_frame_count`

빈 `/cone_path`, 콘 0개, 입력 timeout, TF 실패 시 `STOP`을 발행한다.
유효한 경로는 robot origin을 앞에 붙이고, TF로 `odom`에 변환한 뒤,
중복 제거, 길이 제한, 평활화, 등간격 재표본화, tangent yaw 계산을 거쳐
`/reference_path`로 발행한다.

## Run

```bash
ros2 launch tb3_reference cone_reference.launch.py
```
