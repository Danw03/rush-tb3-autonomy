#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32.hpp"

#include "tb3_mpc_controller/mpc_solver.hpp"

namespace tb3_mpc_controller
{
namespace
{

double quaternion_to_yaw(const geometry_msgs::msg::Quaternion & q)
{
  const double sin_yaw = 2.0 * (q.w * q.z + q.x * q.y);
  const double cos_yaw = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(sin_yaw, cos_yaw);
}

double wrap_angle(double angle)
{
  return std::atan2(std::sin(angle), std::cos(angle));
}

geometry_msgs::msg::Quaternion yaw_to_quaternion(double yaw)
{
  geometry_msgs::msg::Quaternion q;
  q.z = std::sin(0.5 * yaw);
  q.w = std::cos(0.5 * yaw);
  return q;
}

Eigen::Matrix3d diagonal3(
  const std::vector<double> & values,
  const std::string & parameter_name)
{
  if (values.size() != 3) {
    throw std::runtime_error(
            parameter_name + " must contain exactly 3 values.");
  }

  Eigen::Matrix3d matrix = Eigen::Matrix3d::Zero();
  matrix.diagonal() << values[0], values[1], values[2];
  return matrix;
}

Eigen::Matrix2d diagonal2(
  const std::vector<double> & values,
  const std::string & parameter_name)
{
  if (values.size() != 2) {
    throw std::runtime_error(
            parameter_name + " must contain exactly 2 values.");
  }

  Eigen::Matrix2d matrix = Eigen::Matrix2d::Zero();
  matrix.diagonal() << values[0], values[1];
  return matrix;
}

}  // namespace

class MpcControllerNode : public rclcpp::Node
{
public:
  MpcControllerNode()
  : Node("mpc_controller")
  {
    declare_parameters();
    load_parameters();

    command_publisher_ =
      create_publisher<geometry_msgs::msg::TwistStamped>(command_topic_, 10);

    prediction_publisher_ =
      create_publisher<nav_msgs::msg::Path>("/mpc_predicted_path", 10);

    reference_horizon_publisher_ =
      create_publisher<nav_msgs::msg::Path>("/mpc_reference_horizon", 10);

    odometry_subscription_ =
      create_subscription<nav_msgs::msg::Odometry>(
      odometry_topic_,
      rclcpp::QoS(20),
      [this](nav_msgs::msg::Odometry::SharedPtr message)
      {
        odometry_ = *message;
        odometry_received_ = true;
        last_odometry_time_ = now();
      });

    path_subscription_ =
      create_subscription<nav_msgs::msg::Path>(
      reference_path_topic_,
      rclcpp::QoS(10),
      [this](nav_msgs::msg::Path::SharedPtr message)
      {
        set_reference_path(*message);
      });

    speed_subscription_ =
      create_subscription<std_msgs::msg::Float32>(
      reference_speed_topic_,
      rclcpp::QoS(10),
      [this](std_msgs::msg::Float32::SharedPtr message)
      {
        reference_speed_ = std::max(
          0.0,
          static_cast<double>(message->data));
        speed_received_ = true;
        last_speed_time_ = now();
      });

    const auto timer_period = std::chrono::duration<double>(config_.dt);
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(timer_period),
      [this]() {control_step();});

    RCLCPP_INFO(
      get_logger(),
      "MPC ready: dt=%.3f, N=%d, output=%s",
      config_.dt,
      config_.horizon_steps,
      command_topic_.c_str());
  }

private:
  struct PathSample
  {
    Eigen::Vector2d position{Eigen::Vector2d::Zero()};
    double yaw{0.0};
  };

  void declare_parameters()
  {
    declare_parameter("control_period", 0.1);
    declare_parameter("horizon_steps", 20);
    declare_parameter("linearization_iterations", 2);

    declare_parameter("q", std::vector<double>{3.0, 80.0, 15.0});
    declare_parameter(
      "q_terminal", std::vector<double>{10.0, 150.0, 30.0});
    declare_parameter("r", std::vector<double>{2.0, 0.2});
    declare_parameter("r_delta", std::vector<double>{4.0, 0.5});

    declare_parameter("v_min", 0.0);
    declare_parameter("v_max", 0.06);
    declare_parameter("omega_max", 0.8);
    declare_parameter("acceleration_v", 0.15);
    declare_parameter("acceleration_omega", 1.5);
    declare_parameter("regularization", 1.0e-8);

    declare_parameter("qp_max_iterations", 250);
    declare_parameter("qp_rho", 1.0);
    declare_parameter("qp_sigma", 1.0e-6);
    declare_parameter("qp_eps_abs", 1.0e-4);
    declare_parameter("qp_eps_rel", 1.0e-4);

    declare_parameter("reference_speed_default", 0.03);
    declare_parameter("input_timeout", 0.5);

    declare_parameter("odom_topic", "/odom");
    declare_parameter("reference_path_topic", "/reference_path");
    declare_parameter("reference_speed_topic", "/reference_speed");
    declare_parameter("cmd_topic", "/cmd_vel_raw");
  }

  void load_parameters()
  {
    config_.dt = get_parameter("control_period").as_double();
    config_.horizon_steps =
      static_cast<int>(get_parameter("horizon_steps").as_int());
    config_.linearization_iterations =
      static_cast<int>(get_parameter("linearization_iterations").as_int());

    config_.q = diagonal3(get_parameter("q").as_double_array(), "q");
    config_.q_terminal = diagonal3(
      get_parameter("q_terminal").as_double_array(), "q_terminal");
    config_.r = diagonal2(get_parameter("r").as_double_array(), "r");
    config_.r_delta = diagonal2(
      get_parameter("r_delta").as_double_array(), "r_delta");

    config_.v_min = get_parameter("v_min").as_double();
    config_.v_max = get_parameter("v_max").as_double();
    config_.omega_max = get_parameter("omega_max").as_double();
    config_.acceleration_v = get_parameter("acceleration_v").as_double();
    config_.acceleration_omega =
      get_parameter("acceleration_omega").as_double();
    config_.regularization = get_parameter("regularization").as_double();

    config_.qp.max_iterations =
      static_cast<int>(get_parameter("qp_max_iterations").as_int());
    config_.qp.rho = get_parameter("qp_rho").as_double();
    config_.qp.sigma = get_parameter("qp_sigma").as_double();
    config_.qp.eps_abs = get_parameter("qp_eps_abs").as_double();
    config_.qp.eps_rel = get_parameter("qp_eps_rel").as_double();

    reference_speed_ =
      get_parameter("reference_speed_default").as_double();
    input_timeout_ = get_parameter("input_timeout").as_double();

    odometry_topic_ = get_parameter("odom_topic").as_string();
    reference_path_topic_ =
      get_parameter("reference_path_topic").as_string();
    reference_speed_topic_ =
      get_parameter("reference_speed_topic").as_string();
    command_topic_ = get_parameter("cmd_topic").as_string();

    solver_ = std::make_unique<ConvexMpcSolver>(config_);
  }

  void set_reference_path(const nav_msgs::msg::Path & path)
  {
    if (path.poses.size() < 2) {
      RCLCPP_WARN(get_logger(), "Reference path has fewer than 2 poses.");
      return;
    }

    reference_path_ = path;
    path_arc_length_.assign(path.poses.size(), 0.0);
    path_yaw_.assign(path.poses.size(), 0.0);

    for (std::size_t i = 0; i < path.poses.size(); ++i) {
      path_yaw_[i] = quaternion_to_yaw(path.poses[i].pose.orientation);

      if (i == 0) {
        continue;
      }

      const double dx =
        path.poses[i].pose.position.x -
        path.poses[i - 1].pose.position.x;
      const double dy =
        path.poses[i].pose.position.y -
        path.poses[i - 1].pose.position.y;

      path_arc_length_[i] = path_arc_length_[i - 1] + std::hypot(dx, dy);
    }

    for (std::size_t i = 1; i < path_yaw_.size(); ++i) {
      path_yaw_[i] =
        path_yaw_[i - 1] + wrap_angle(path_yaw_[i] - path_yaw_[i - 1]);
    }

    path_received_ = true;
    last_path_time_ = now();
  }

  PathSample pose_to_sample(std::size_t index) const
  {
    PathSample sample;
    sample.position <<
      reference_path_.poses[index].pose.position.x,
      reference_path_.poses[index].pose.position.y;
    sample.yaw = path_yaw_[index];
    return sample;
  }

  PathSample sample_path(double query_arc_length) const
  {
    query_arc_length = std::clamp(
      query_arc_length, 0.0, path_arc_length_.back());

    const auto upper_iterator = std::lower_bound(
      path_arc_length_.begin(), path_arc_length_.end(), query_arc_length);

    if (upper_iterator == path_arc_length_.begin()) {
      return pose_to_sample(0);
    }
    if (upper_iterator == path_arc_length_.end()) {
      return pose_to_sample(reference_path_.poses.size() - 1);
    }

    const std::size_t upper_index =
      static_cast<std::size_t>(
      std::distance(path_arc_length_.begin(), upper_iterator));
    const std::size_t lower_index = upper_index - 1;

    const double s0 = path_arc_length_[lower_index];
    const double s1 = path_arc_length_[upper_index];
    const double ratio =
      (query_arc_length - s0) / std::max(1.0e-9, s1 - s0);

    PathSample sample;
    const auto & p0 = reference_path_.poses[lower_index].pose.position;
    const auto & p1 = reference_path_.poses[upper_index].pose.position;

    sample.position.x() = p0.x + ratio * (p1.x - p0.x);
    sample.position.y() = p0.y + ratio * (p1.y - p0.y);
    sample.yaw =
      path_yaw_[lower_index] +
      ratio * (path_yaw_[upper_index] - path_yaw_[lower_index]);

    return sample;
  }

  double nearest_path_progress(const Eigen::Vector2d & robot_position) const
  {
    double best_distance_squared = std::numeric_limits<double>::infinity();
    double best_progress = 0.0;

    for (std::size_t i = 0; i + 1 < reference_path_.poses.size(); ++i) {
      const Eigen::Vector2d p0(
        reference_path_.poses[i].pose.position.x,
        reference_path_.poses[i].pose.position.y);
      const Eigen::Vector2d p1(
        reference_path_.poses[i + 1].pose.position.x,
        reference_path_.poses[i + 1].pose.position.y);

      const Eigen::Vector2d segment = p1 - p0;
      const double segment_squared_norm = segment.squaredNorm();
      double ratio = 0.0;

      if (segment_squared_norm > 1.0e-12) {
        ratio = std::clamp(
          (robot_position - p0).dot(segment) / segment_squared_norm,
          0.0,
          1.0);
      }

      const Eigen::Vector2d projection = p0 + ratio * segment;
      const double distance_squared =
        (robot_position - projection).squaredNorm();

      if (distance_squared < best_distance_squared) {
        best_distance_squared = distance_squared;
        best_progress =
          path_arc_length_[i] +
          ratio * (path_arc_length_[i + 1] - path_arc_length_[i]);
      }
    }

    return best_progress;
  }

  void build_reference(
    Eigen::MatrixXd & reference_states,
    Eigen::MatrixXd & reference_controls,
    nav_msgs::msg::Path & reference_horizon)
  {
    const double robot_yaw = quaternion_to_yaw(odometry_.pose.pose.orientation);
    const Eigen::Vector2d robot_position(
      odometry_.pose.pose.position.x,
      odometry_.pose.pose.position.y);

    Eigen::Matrix2d world_to_robot;
    world_to_robot <<
      std::cos(robot_yaw), std::sin(robot_yaw),
      -std::sin(robot_yaw), std::cos(robot_yaw);

    const double start_progress = nearest_path_progress(robot_position);

    reference_states.resize(3, config_.horizon_steps + 1);
    reference_controls.resize(2, config_.horizon_steps);

    reference_horizon.header.stamp = now();
    reference_horizon.header.frame_id = reference_path_.header.frame_id;
    reference_horizon.poses.clear();

    double previous_local_yaw = 0.0;

    for (int k = 0; k <= config_.horizon_steps; ++k) {
      const double query_progress = std::min(
        path_arc_length_.back(),
        start_progress +
        static_cast<double>(k) * reference_speed_ * config_.dt);

      const PathSample sample = sample_path(query_progress);
      const Eigen::Vector2d local_position =
        world_to_robot * (sample.position - robot_position);

      double local_yaw = sample.yaw - robot_yaw;
      if (k == 0) {
        local_yaw = wrap_angle(local_yaw);
      } else {
        local_yaw =
          previous_local_yaw + wrap_angle(local_yaw - previous_local_yaw);
      }
      previous_local_yaw = local_yaw;

      reference_states.col(k) <<
        local_position.x(), local_position.y(), local_yaw;

      geometry_msgs::msg::PoseStamped pose;
      pose.header = reference_horizon.header;
      pose.pose.position.x = sample.position.x();
      pose.pose.position.y = sample.position.y();
      pose.pose.orientation = yaw_to_quaternion(sample.yaw);
      reference_horizon.poses.push_back(pose);

      if (k < config_.horizon_steps) {
        const bool path_continues =
          query_progress < path_arc_length_.back() - 1.0e-4;
        reference_controls.col(k) <<
          (path_continues ? reference_speed_ : 0.0),
          0.0;
      }
    }
  }

  void publish_prediction(const Eigen::MatrixXd & local_states)
  {
    nav_msgs::msg::Path prediction;
    prediction.header.stamp = now();
    prediction.header.frame_id = reference_path_.header.frame_id;

    const double robot_yaw = quaternion_to_yaw(odometry_.pose.pose.orientation);
    const Eigen::Vector2d robot_position(
      odometry_.pose.pose.position.x,
      odometry_.pose.pose.position.y);

    Eigen::Matrix2d robot_to_world;
    robot_to_world <<
      std::cos(robot_yaw), -std::sin(robot_yaw),
      std::sin(robot_yaw), std::cos(robot_yaw);

    for (int k = 0; k < local_states.cols(); ++k) {
      const Eigen::Vector2d world_position =
        robot_position + robot_to_world * local_states.block<2, 1>(0, k);

      geometry_msgs::msg::PoseStamped pose;
      pose.header = prediction.header;
      pose.pose.position.x = world_position.x();
      pose.pose.position.y = world_position.y();
      pose.pose.orientation =
        yaw_to_quaternion(robot_yaw + local_states(2, k));
      prediction.poses.push_back(pose);
    }

    prediction_publisher_->publish(prediction);
  }

  bool inputs_ready() const
  {
    if (!odometry_received_ || !path_received_) {
      return false;
    }

    const rclcpp::Time current_time = now();

    if ((current_time - last_odometry_time_).seconds() > input_timeout_) {
      return false;
    }
    if ((current_time - last_path_time_).seconds() > input_timeout_) {
      return false;
    }
    if (
      speed_received_ &&
      (current_time - last_speed_time_).seconds() > input_timeout_)
    {
      return false;
    }

    return true;
  }

  void control_step()
  {
    if (!inputs_ready()) {
      previous_command_.setZero();
      publish_command(previous_command_);
      return;
    }

    if (reference_path_.header.frame_id != odometry_.header.frame_id) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Path frame '%s' must match odom frame '%s'.",
        reference_path_.header.frame_id.c_str(),
        odometry_.header.frame_id.c_str());

      previous_command_.setZero();
      publish_command(previous_command_);
      return;
    }

    Eigen::MatrixXd reference_states;
    Eigen::MatrixXd reference_controls;
    nav_msgs::msg::Path reference_horizon;

    build_reference(
      reference_states,
      reference_controls,
      reference_horizon);

    try {
      const MpcResult result = solver_->solve(
        Eigen::Vector3d::Zero(),
        previous_command_,
        reference_states,
        reference_controls);

      previous_command_ = result.command;
      publish_command(previous_command_);
      publish_prediction(result.states);
      reference_horizon_publisher_->publish(reference_horizon);

      if (!result.converged) {
        RCLCPP_WARN_THROTTLE(
          get_logger(),
          *get_clock(),
          1000,
          "QP reached iteration limit: primal=%.3e dual=%.3e",
          result.primal_residual,
          result.dual_residual);
      }

      RCLCPP_INFO_THROTTLE(
        get_logger(),
        *get_clock(),
        1000,
        "cmd v=%.4f omega=%.4f | qp_iter=%d | residual=(%.2e, %.2e)",
        previous_command_.x(),
        previous_command_.y(),
        result.qp_iterations,
        result.primal_residual,
        result.dual_residual);
    } catch (const std::exception & exception) {
      RCLCPP_ERROR(get_logger(), "MPC solve failed: %s", exception.what());
      solver_->reset();
      previous_command_.setZero();
      publish_command(previous_command_);
    }
  }

  void publish_command(const Eigen::Vector2d & command)
  {
    geometry_msgs::msg::TwistStamped message;
    message.header.stamp = now();
    message.header.frame_id =
      odometry_received_ ? odometry_.child_frame_id : "base_footprint";
    message.twist.linear.x = command.x();
    message.twist.angular.z = command.y();
    command_publisher_->publish(message);
  }

  MpcConfig config_;
  std::unique_ptr<ConvexMpcSolver> solver_;

  std::string odometry_topic_;
  std::string reference_path_topic_;
  std::string reference_speed_topic_;
  std::string command_topic_;

  double reference_speed_{0.03};
  double input_timeout_{0.5};

  nav_msgs::msg::Odometry odometry_;
  nav_msgs::msg::Path reference_path_;
  std::vector<double> path_arc_length_;
  std::vector<double> path_yaw_;

  Eigen::Vector2d previous_command_{Eigen::Vector2d::Zero()};

  bool odometry_received_{false};
  bool path_received_{false};
  bool speed_received_{false};

  rclcpp::Time last_odometry_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_path_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_speed_time_{0, 0, RCL_ROS_TIME};

  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr
    command_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr prediction_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr
    reference_horizon_publisher_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr
    odometry_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_subscription_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr
    speed_subscription_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace tb3_mpc_controller

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<tb3_mpc_controller::MpcControllerNode>());
  rclcpp::shutdown();
  return 0;
}
