#!/usr/bin/env python3

import copy
from pathlib import Path as FilePath
from typing import List, Optional

import numpy as np
import rclpy
from nav_msgs.msg import Path
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

try:
    import joblib
except ImportError:
    joblib = None


class ReferenceNode(Node):
    """
    Temporary Reference Generator.

    Current behavior:
      fresh, valid /cone_path -> CONE mode -> republish as /reference_path
      otherwise               -> STOP mode -> publish an empty Path

    Future behavior:
      /lane_features + /cone_features
          -> scaler + logistic-regression model
          -> hysteresis/state machine
          -> LANE, CONE, or STOP
          -> selected path smoothing and resampling
    """

    def __init__(self) -> None:
        super().__init__("reference_node")

        self.declare_parameter("cone_path_topic", "/cone_path")
        self.declare_parameter("cone_features_topic", "/cone_features")
        self.declare_parameter("reference_path_topic", "/reference_path")
        self.declare_parameter("driving_mode_topic", "/driving_mode")
        self.declare_parameter("input_timeout_sec", 0.5)
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("model_path", "")

        cone_path_topic = self.get_parameter("cone_path_topic").value
        cone_features_topic = self.get_parameter("cone_features_topic").value
        reference_path_topic = self.get_parameter("reference_path_topic").value
        driving_mode_topic = self.get_parameter("driving_mode_topic").value

        self.input_timeout_sec = float(
            self.get_parameter("input_timeout_sec").value
        )
        publish_rate_hz = float(
            self.get_parameter("publish_rate_hz").value
        )

        self.latest_cone_path: Optional[Path] = None
        self.latest_cone_features: Optional[List[float]] = None
        self.last_cone_path_time = None
        self.last_mode = None

        self.model = self.load_model(
            str(self.get_parameter("model_path").value)
        )

        self.cone_path_sub = self.create_subscription(
            Path,
            cone_path_topic,
            self.cone_path_callback,
            10,
        )
        self.cone_features_sub = self.create_subscription(
            Float32MultiArray,
            cone_features_topic,
            self.cone_features_callback,
            10,
        )

        self.reference_path_pub = self.create_publisher(
            Path,
            reference_path_topic,
            10,
        )
        self.driving_mode_pub = self.create_publisher(
            String,
            driving_mode_topic,
            10,
        )

        self.timer = self.create_timer(
            1.0 / max(publish_rate_hz, 1.0),
            self.timer_callback,
        )

        self.get_logger().info(
            f"Started: {cone_path_topic}, {cone_features_topic} "
            f"-> {reference_path_topic}, {driving_mode_topic}"
        )

    def cone_path_callback(self, msg: Path) -> None:
        self.latest_cone_path = msg
        self.last_cone_path_time = self.get_clock().now()

    def cone_features_callback(self, msg: Float32MultiArray) -> None:
        self.latest_cone_features = list(msg.data)

    def timer_callback(self) -> None:
        mode = self.select_driving_mode()

        if mode == "CONE" and self.latest_cone_path is not None:
            reference_path = self.generate_reference_path(
                self.latest_cone_path
            )
        else:
            reference_path = self.build_empty_path()

        mode_msg = String()
        mode_msg.data = mode

        self.reference_path_pub.publish(reference_path)
        self.driving_mode_pub.publish(mode_msg)

        if mode != self.last_mode:
            self.get_logger().info(f"Driving mode: {mode}")
            self.last_mode = mode

    def select_driving_mode(self) -> str:
        """Select the current mode with safety checks before ML inference."""
        if not self.cone_input_is_fresh():
            return "STOP"

        if self.latest_cone_path is None or len(self.latest_cone_path.poses) == 0:
            return "STOP"

        if not self.cone_feature_is_valid():
            return "STOP"

        predicted_mode = self.predict_mode_with_model()
        if predicted_mode in {"CONE", "STOP"}:
            return predicted_mode

        return "CONE"

    def cone_input_is_fresh(self) -> bool:
        if self.last_cone_path_time is None:
            return False

        age = (
            self.get_clock().now() - self.last_cone_path_time
        ).nanoseconds / 1e9

        return age <= self.input_timeout_sec

    def cone_feature_is_valid(self) -> bool:
        if not self.latest_cone_features:
            return False

        path_valid = self.latest_cone_features[0]
        return np.isfinite(path_valid) and path_valid >= 0.5

    def generate_reference_path(self, source_path: Path) -> Path:
        """
        Placeholder: copy /cone_path directly.

        TODO:
          - choose lane/cone path using the classifier
          - transform to the controller frame
          - smooth
          - resample at a fixed spatial interval
          - compute heading/curvature/speed references
        """
        output = copy.deepcopy(source_path)
        output.header.stamp = self.get_clock().now().to_msg()

        for pose in output.poses:
            pose.header = output.header

        return output

    def build_empty_path(self) -> Path:
        output = Path()
        output.header.stamp = self.get_clock().now().to_msg()

        if self.latest_cone_path is not None:
            output.header.frame_id = self.latest_cone_path.header.frame_id
        else:
            output.header.frame_id = "base_scan"

        return output

    def load_model(self, model_path: str):
        if not model_path:
            self.get_logger().info(
                "No logistic-regression model configured; "
                "using the placeholder mode rule."
            )
            return None

        if joblib is None:
            self.get_logger().warning(
                "joblib is not installed; model loading is disabled."
            )
            return None

        path = FilePath(model_path).expanduser()
        if not path.is_file():
            self.get_logger().warning(f"Model file not found: {path}")
            return None

        try:
            model = joblib.load(path)
            self.get_logger().info(f"Loaded mode model: {path}")
            return model
        except Exception as exc:
            self.get_logger().error(f"Failed to load model: {exc}")
            return None

    def predict_mode_with_model(self) -> Optional[str]:
        if self.model is None or not self.latest_cone_features:
            return None

        features = np.asarray(
            self.latest_cone_features,
            dtype=np.float64,
        ).reshape(1, -1)

        if not np.all(np.isfinite(features)):
            return "STOP"

        try:
            prediction = self.model.predict(features)[0]
        except Exception as exc:
            self.get_logger().error(f"Mode inference failed: {exc}")
            return "STOP"

        prediction_text = str(prediction).upper()
        if prediction_text in {"CONE", "STOP"}:
            return prediction_text

        self.get_logger().warning(
            f"Unsupported model output '{prediction_text}'; "
            "using the placeholder rule."
        )
        return None

    def combine_lane_and_cone_features(self):
        """TODO: Build the final classifier input in a fixed feature order."""
        raise NotImplementedError

    def apply_mode_hysteresis(self, probabilities):
        """TODO: Require sufficient confidence and consecutive frames."""
        raise NotImplementedError

    def smooth_and_resample_path(self, path: Path):
        """TODO: Create a uniformly sampled MPC reference path."""
        raise NotImplementedError


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ReferenceNode()

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
