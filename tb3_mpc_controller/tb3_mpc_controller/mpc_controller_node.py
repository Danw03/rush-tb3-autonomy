#!/usr/bin/env python3

import math
from typing import Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node

try:
    import osqp
    from scipy import sparse

    QP_LIBRARIES_AVAILABLE = True
except ImportError:
    osqp = None
    sparse = None
    QP_LIBRARIES_AVAILABLE = False


class MpcControllerNode(Node):
    """
    MPC communication skeleton.

    Current behavior:
      /reference_path + /odom -> zero /cmd_vel_raw at a fixed rate

    Zero output is deliberate because the temporary cone path points to the
    nearest LiDAR return and must not be followed by the real robot.
    """

    def __init__(self) -> None:
        super().__init__("mpc_controller_node")

        self.declare_parameter("reference_path_topic", "/reference_path")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_vel_raw_topic", "/cmd_vel_raw")
        self.declare_parameter("control_rate_hz", 10.0)
        self.declare_parameter("input_timeout_sec", 0.5)
        self.declare_parameter("command_frame", "base_link")

        reference_path_topic = self.get_parameter(
            "reference_path_topic"
        ).value
        odom_topic = self.get_parameter("odom_topic").value
        cmd_vel_raw_topic = self.get_parameter("cmd_vel_raw_topic").value

        control_rate_hz = float(
            self.get_parameter("control_rate_hz").value
        )
        self.input_timeout_sec = float(
            self.get_parameter("input_timeout_sec").value
        )
        self.command_frame = str(
            self.get_parameter("command_frame").value
        )

        self.latest_reference_path: Optional[Path] = None
        self.latest_odom: Optional[Odometry] = None
        self.last_reference_time = None
        self.last_odom_time = None

        self.reference_sub = self.create_subscription(
            Path,
            reference_path_topic,
            self.reference_path_callback,
            10,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            10,
        )

        self.cmd_pub = self.create_publisher(
            TwistStamped,
            cmd_vel_raw_topic,
            10,
        )

        self.timer = self.create_timer(
            1.0 / max(control_rate_hz, 1.0),
            self.control_callback,
        )

        self.get_logger().info(
            f"Started: {reference_path_topic}, {odom_topic} "
            f"-> {cmd_vel_raw_topic}"
        )
        self.get_logger().info(
            "Placeholder controller publishes zero velocity."
        )

        if not QP_LIBRARIES_AVAILABLE:
            self.get_logger().warning(
                "OSQP/SciPy are unavailable. This is fine for the skeleton; "
                "install them before implementing the QP controller."
            )

    def reference_path_callback(self, msg: Path) -> None:
        self.latest_reference_path = msg
        self.last_reference_time = self.get_clock().now()

    def odom_callback(self, msg: Odometry) -> None:
        self.latest_odom = msg
        self.last_odom_time = self.get_clock().now()

    def control_callback(self) -> None:
        if not self.inputs_are_ready():
            self.publish_control(0.0, 0.0)
            return

        linear_velocity, angular_velocity = self.solve_mpc(
            self.latest_odom,
            self.latest_reference_path,
        )
        self.publish_control(linear_velocity, angular_velocity)

    def inputs_are_ready(self) -> bool:
        if self.latest_reference_path is None or self.latest_odom is None:
            return False

        if len(self.latest_reference_path.poses) == 0:
            return False

        if self.last_reference_time is None or self.last_odom_time is None:
            return False

        now = self.get_clock().now()
        reference_age = (
            now - self.last_reference_time
        ).nanoseconds / 1e9
        odom_age = (now - self.last_odom_time).nanoseconds / 1e9

        return (
            reference_age <= self.input_timeout_sec
            and odom_age <= self.input_timeout_sec
        )

    def solve_mpc(
        self,
        odom: Odometry,
        reference_path: Path,
    ) -> Tuple[float, float]:
        """
        TODO: Implement LTV-MPC as a convex QP.

        Suggested flow:
          state = extract_state(odom)
          reference = build_reference_states(reference_path)
          A, B = linearize_model(state, reference)
          P, q, A_qp, lower, upper = build_qp(...)
          solution = OSQP.solve(...)
          return first v, first omega
        """
        del odom
        del reference_path
        return 0.0, 0.0

    def publish_control(
        self,
        linear_velocity: float,
        angular_velocity: float,
    ) -> None:
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = self.command_frame
        cmd.twist.linear.x = float(linear_velocity)
        cmd.twist.angular.z = float(angular_velocity)
        self.cmd_pub.publish(cmd)

    def extract_state(self, odom: Odometry) -> np.ndarray:
        """Return [x, y, yaw] as the basic unicycle state."""
        position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation

        siny_cosp = 2.0 * (
            orientation.w * orientation.z
            + orientation.x * orientation.y
        )
        cosy_cosp = 1.0 - 2.0 * (
            orientation.y**2 + orientation.z**2
        )
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return np.asarray(
            [position.x, position.y, yaw],
            dtype=np.float64,
        )

    def build_reference_states(self, path: Path):
        """TODO: Convert nav_msgs/Path into MPC horizon arrays."""
        raise NotImplementedError

    def linearize_model(self, state, reference):
        """TODO: Linearize the unicycle model along the reference."""
        raise NotImplementedError

    def build_qp(self, state, reference, dynamics):
        """TODO: Assemble sparse QP cost, dynamics, and constraints."""
        raise NotImplementedError


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MpcControllerNode()

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
