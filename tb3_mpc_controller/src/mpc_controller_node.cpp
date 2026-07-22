#include <chrono>
#include <functional>
#include <memory>

#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"

using namespace std::chrono_literals;

class MpcControllerNode : public rclcpp::Node
{
public:
  MpcControllerNode()
  : Node("mpc_controller_node")
  {
    reference_sub_ = this->create_subscription<nav_msgs::msg::Path>(
      "/reference_path",
      10,
      std::bind(
        &MpcControllerNode::reference_callback,
        this,
        std::placeholders::_1));

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/odom",
      10,
      std::bind(
        &MpcControllerNode::odom_callback,
        this,
        std::placeholders::_1));

    cmd_pub_ =
      this->create_publisher<geometry_msgs::msg::TwistStamped>(
      "/cmd_vel_raw",
      10);

    timer_ = this->create_wall_timer(
      100ms,
      std::bind(
        &MpcControllerNode::control_callback,
        this));

    RCLCPP_INFO(
      this->get_logger(),
      "MPC controller started");

    RCLCPP_INFO(
      this->get_logger(),
      "/reference_path + /odom -> /cmd_vel_raw");
  }

private:
  void reference_callback(
    const nav_msgs::msg::Path::SharedPtr msg)
  {
    latest_reference_ = *msg;
    reference_received_ = !msg->poses.empty();
  }

  void odom_callback(
    const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    latest_odom_ = *msg;
    odom_received_ = true;
  }

  void control_callback()
  {
    double linear_velocity = 0.0;
    double angular_velocity = 0.0;

    if (reference_received_ && odom_received_) {
      solve_mpc(
        linear_velocity,
        angular_velocity);
    }

    publish_command(
      linear_velocity,
      angular_velocity);
  }

  void solve_mpc(
    double & linear_velocity,
    double & angular_velocity)
  {
    // TODO:
    // 1. latest_odom_에서 현재 상태 추출
    // 2. latest_reference_에서 horizon 생성
    // 3. Unicycle model 선형화
    // 4. Convex QP 구성
    // 5. QP solver 실행
    // 6. 첫 번째 제어입력 v, omega 반환

    // 현재는 전체 통신 검증을 위해 정지 명령만 출력
    linear_velocity = 0.0;
    angular_velocity = 0.0;
  }

  void publish_command(
    const double linear_velocity,
    const double angular_velocity)
  {
    geometry_msgs::msg::TwistStamped cmd;

    cmd.header.stamp = this->now();
    cmd.header.frame_id = "base_link";

    cmd.twist.linear.x = linear_velocity;
    cmd.twist.angular.z = angular_velocity;

    cmd_pub_->publish(cmd);
  }

  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr
    reference_sub_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr
    odom_sub_;

  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr
    cmd_pub_;

  rclcpp::TimerBase::SharedPtr timer_;

  nav_msgs::msg::Path latest_reference_;
  nav_msgs::msg::Odometry latest_odom_;

  bool reference_received_{false};
  bool odom_received_{false};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<MpcControllerNode>();

  rclcpp::spin(node);

  rclcpp::shutdown();

  return 0;
}