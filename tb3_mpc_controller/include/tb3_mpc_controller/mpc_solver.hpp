#ifndef TB3_MPC_CONTROLLER__MPC_SOLVER_HPP_
#define TB3_MPC_CONTROLLER__MPC_SOLVER_HPP_

#include <limits>

#include <Eigen/Core>

#include "tb3_mpc_controller/dense_qp_solver.hpp"

namespace tb3_mpc_controller
{

struct MpcConfig
{
  double dt{0.1};
  int horizon_steps{20};
  int linearization_iterations{2};

  Eigen::Matrix3d q{
    (Eigen::Vector3d(3.0, 80.0, 15.0)).asDiagonal()};
  Eigen::Matrix3d q_terminal{
    (Eigen::Vector3d(10.0, 150.0, 30.0)).asDiagonal()};
  Eigen::Matrix2d r{
    (Eigen::Vector2d(2.0, 0.2)).asDiagonal()};
  Eigen::Matrix2d r_delta{
    (Eigen::Vector2d(4.0, 0.5)).asDiagonal()};

  double v_min{0.0};
  double v_max{0.06};
  double omega_max{0.8};
  double acceleration_v{0.15};
  double acceleration_omega{1.5};
  double regularization{1.0e-8};

  QpSettings qp;
};

struct MpcResult
{
  Eigen::Vector2d command{Eigen::Vector2d::Zero()};
  Eigen::MatrixXd states;
  Eigen::MatrixXd controls;
  bool converged{false};
  int qp_iterations{0};
  double primal_residual{std::numeric_limits<double>::infinity()};
  double dual_residual{std::numeric_limits<double>::infinity()};
};

class ConvexMpcSolver
{
public:
  explicit ConvexMpcSolver(const MpcConfig & config);

  MpcResult solve(
    const Eigen::Vector3d & initial_state,
    const Eigen::Vector2d & previous_command,
    const Eigen::MatrixXd & reference_states,
    const Eigen::MatrixXd & reference_controls);

  void reset();

private:
  Eigen::Vector3d dynamics(
    const Eigen::Vector3d & state,
    const Eigen::Vector2d & input) const;

  void dynamics_jacobians(
    const Eigen::Vector3d & state,
    const Eigen::Vector2d & input,
    Eigen::Matrix3d & a,
    Eigen::Matrix<double, 3, 2> & b) const;

  Eigen::MatrixXd rollout(
    const Eigen::Vector3d & initial_state,
    const Eigen::MatrixXd & controls) const;

  Eigen::MatrixXd shifted_warm_start(
    const Eigen::MatrixXd & reference_controls) const;

  MpcConfig config_;
  Eigen::MatrixXd previous_solution_;
  bool has_previous_solution_{false};
};

}  // namespace tb3_mpc_controller

#endif  // TB3_MPC_CONTROLLER__MPC_SOLVER_HPP_
