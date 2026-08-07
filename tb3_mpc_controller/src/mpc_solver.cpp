#include "tb3_mpc_controller/mpc_solver.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace tb3_mpc_controller
{
namespace
{

Eigen::MatrixXd block_diagonal(
  const Eigen::MatrixXd & block,
  int repeat_count)
{
  Eigen::MatrixXd result = Eigen::MatrixXd::Zero(
    block.rows() * repeat_count,
    block.cols() * repeat_count);

  for (int i = 0; i < repeat_count; ++i) {
    result.block(
      i * block.rows(),
      i * block.cols(),
      block.rows(),
      block.cols()) = block;
  }

  return result;
}

Eigen::MatrixXd difference_matrix(int horizon_steps)
{
  constexpr int input_size = 2;
  Eigen::MatrixXd difference = Eigen::MatrixXd::Zero(
    input_size * horizon_steps,
    input_size * horizon_steps);

  for (int k = 0; k < horizon_steps; ++k) {
    difference.block<input_size, input_size>(
      input_size * k,
      input_size * k) = Eigen::Matrix2d::Identity();

    if (k > 0) {
      difference.block<input_size, input_size>(
        input_size * k,
        input_size * (k - 1)) = -Eigen::Matrix2d::Identity();
    }
  }

  return difference;
}

Eigen::VectorXd flatten_controls(const Eigen::MatrixXd & controls)
{
  Eigen::VectorXd vector(controls.size());

  for (int k = 0; k < controls.cols(); ++k) {
    vector.segment<2>(2 * k) = controls.col(k);
  }

  return vector;
}

Eigen::MatrixXd unflatten_controls(
  const Eigen::VectorXd & vector,
  int horizon_steps)
{
  Eigen::MatrixXd controls(2, horizon_steps);

  for (int k = 0; k < horizon_steps; ++k) {
    controls.col(k) = vector.segment<2>(2 * k);
  }

  return controls;
}

Eigen::VectorXd flatten_reference_states(
  const Eigen::MatrixXd & reference_states)
{
  const int horizon_steps =
    static_cast<int>(reference_states.cols()) - 1;

  Eigen::VectorXd vector(3 * horizon_steps);

  for (int k = 0; k < horizon_steps; ++k) {
    vector.segment<3>(3 * k) = reference_states.col(k + 1);
  }

  return vector;
}

}  // namespace

ConvexMpcSolver::ConvexMpcSolver(const MpcConfig & config)
: config_(config)
{
  if (config_.horizon_steps <= 0 || config_.dt <= 0.0) {
    throw std::invalid_argument("MPC horizon and dt must be positive.");
  }
  if (
    config_.linearization_iterations <= 0 ||
    config_.v_min > config_.v_max ||
    config_.omega_max <= 0.0 ||
    config_.acceleration_v <= 0.0 ||
    config_.acceleration_omega <= 0.0)
  {
    throw std::invalid_argument("MPC limits and iteration counts are invalid.");
  }
}

void ConvexMpcSolver::reset()
{
  previous_solution_.resize(0, 0);
  has_previous_solution_ = false;
}

Eigen::Vector3d ConvexMpcSolver::dynamics(
  const Eigen::Vector3d & state,
  const Eigen::Vector2d & input) const
{
  const double x = state.x();
  const double y = state.y();
  const double yaw = state.z();
  const double v = input.x();
  const double omega = input.y();

  Eigen::Vector3d next;

  if (std::abs(omega) > 1.0e-6) {
    const double yaw_next = yaw + omega * config_.dt;

    next.x() =
      x + (v / omega) * (std::sin(yaw_next) - std::sin(yaw));

    next.y() =
      y - (v / omega) * (std::cos(yaw_next) - std::cos(yaw));

    next.z() = yaw_next;
  } else {
    next.x() = x + v * config_.dt * std::cos(yaw);
    next.y() = y + v * config_.dt * std::sin(yaw);
    next.z() = yaw + omega * config_.dt;
  }

  return next;
}

void ConvexMpcSolver::dynamics_jacobians(
  const Eigen::Vector3d & state,
  const Eigen::Vector2d & input,
  Eigen::Matrix3d & a,
  Eigen::Matrix<double, 3, 2> & b) const
{
  const double yaw = state.z();
  const double v = input.x();
  const double omega = input.y();
  const double dt = config_.dt;

  a = Eigen::Matrix3d::Identity();
  b.setZero();

  if (std::abs(omega) > 1.0e-6) {
    const double yaw_next = yaw + omega * dt;
    const double sin_difference = std::sin(yaw_next) - std::sin(yaw);
    const double cos_difference = std::cos(yaw_next) - std::cos(yaw);

    a(0, 2) =
      (v / omega) * (std::cos(yaw_next) - std::cos(yaw));

    a(1, 2) =
      (v / omega) * (std::sin(yaw_next) - std::sin(yaw));

    b(0, 0) = sin_difference / omega;
    b(1, 0) = -cos_difference / omega;

    b(0, 1) =
      v *
      (omega * dt * std::cos(yaw_next) - sin_difference) /
      (omega * omega);

    b(1, 1) =
      v *
      (omega * dt * std::sin(yaw_next) + cos_difference) /
      (omega * omega);

    b(2, 1) = dt;
  } else {
    a(0, 2) = -v * dt * std::sin(yaw);
    a(1, 2) = v * dt * std::cos(yaw);

    b(0, 0) = dt * std::cos(yaw);
    b(1, 0) = dt * std::sin(yaw);
    b(0, 1) = -0.5 * v * dt * dt * std::sin(yaw);
    b(1, 1) = 0.5 * v * dt * dt * std::cos(yaw);
    b(2, 1) = dt;
  }
}

Eigen::MatrixXd ConvexMpcSolver::rollout(
  const Eigen::Vector3d & initial_state,
  const Eigen::MatrixXd & controls) const
{
  Eigen::MatrixXd states(3, config_.horizon_steps + 1);
  states.col(0) = initial_state;

  for (int k = 0; k < config_.horizon_steps; ++k) {
    states.col(k + 1) = dynamics(states.col(k), controls.col(k));
  }

  return states;
}

Eigen::MatrixXd ConvexMpcSolver::shifted_warm_start(
  const Eigen::MatrixXd & reference_controls) const
{
  if (!has_previous_solution_) {
    return reference_controls;
  }

  Eigen::MatrixXd shifted(2, config_.horizon_steps);
  shifted.leftCols(config_.horizon_steps - 1) =
    previous_solution_.rightCols(config_.horizon_steps - 1);
  shifted.col(config_.horizon_steps - 1) =
    reference_controls.col(config_.horizon_steps - 1);

  return shifted;
}

MpcResult ConvexMpcSolver::solve(
  const Eigen::Vector3d & initial_state,
  const Eigen::Vector2d & previous_command,
  const Eigen::MatrixXd & reference_states,
  const Eigen::MatrixXd & reference_controls)
{
  const int horizon = config_.horizon_steps;
  constexpr int state_size = 3;
  constexpr int input_size = 2;

  if (
    reference_states.rows() != state_size ||
    reference_states.cols() != horizon + 1 ||
    reference_controls.rows() != input_size ||
    reference_controls.cols() != horizon)
  {
    throw std::invalid_argument("MPC reference dimensions are invalid.");
  }

  Eigen::MatrixXd controls = shifted_warm_start(reference_controls);

  QpResult last_qp_result;
  bool all_iterations_converged = true;

  for (int outer = 0; outer < config_.linearization_iterations; ++outer) {
    const Eigen::MatrixXd nominal_states = rollout(initial_state, controls);

    std::vector<Eigen::Matrix3d> a_matrices(horizon);
    std::vector<Eigen::Matrix<double, 3, 2>> b_matrices(horizon);
    std::vector<Eigen::Vector3d> c_vectors(horizon);

    for (int k = 0; k < horizon; ++k) {
      dynamics_jacobians(
        nominal_states.col(k),
        controls.col(k),
        a_matrices[k],
        b_matrices[k]);

      c_vectors[k] =
        dynamics(nominal_states.col(k), controls.col(k)) -
        a_matrices[k] * nominal_states.col(k) -
        b_matrices[k] * controls.col(k);
    }

    Eigen::MatrixXd sx = Eigen::MatrixXd::Zero(
      state_size * horizon,
      state_size);

    Eigen::MatrixXd su = Eigen::MatrixXd::Zero(
      state_size * horizon,
      input_size * horizon);

    Eigen::VectorXd sc = Eigen::VectorXd::Zero(state_size * horizon);

    Eigen::Matrix3d state_transition = Eigen::Matrix3d::Identity();
    Eigen::MatrixXd input_transition = Eigen::MatrixXd::Zero(
      state_size,
      input_size * horizon);
    Eigen::Vector3d affine_transition = Eigen::Vector3d::Zero();

    for (int k = 0; k < horizon; ++k) {
      state_transition = a_matrices[k] * state_transition;
      input_transition = a_matrices[k] * input_transition;

      input_transition.block<state_size, input_size>(
        0,
        input_size * k) += b_matrices[k];

      affine_transition =
        a_matrices[k] * affine_transition + c_vectors[k];

      sx.block<state_size, state_size>(
        state_size * k,
        0) = state_transition;

      su.block(
        state_size * k,
        0,
        state_size,
        input_size * horizon) = input_transition;

      sc.segment<state_size>(state_size * k) = affine_transition;
    }

    Eigen::MatrixXd q_bar = block_diagonal(config_.q, horizon);
    q_bar.bottomRightCorner<state_size, state_size>() = config_.q_terminal;

    const Eigen::MatrixXd r_bar = block_diagonal(config_.r, horizon);
    const Eigen::MatrixXd r_delta_bar =
      block_diagonal(config_.r_delta, horizon);
    const Eigen::MatrixXd difference = difference_matrix(horizon);

    Eigen::VectorXd previous_offset =
      Eigen::VectorXd::Zero(input_size * horizon);
    previous_offset.segment<input_size>(0) = previous_command;

    const Eigen::VectorXd reference_state_vector =
      flatten_reference_states(reference_states);
    const Eigen::VectorXd reference_control_vector =
      flatten_controls(reference_controls);
    const Eigen::VectorXd affine_state = sx * initial_state + sc;

    Eigen::MatrixXd hessian =
      su.transpose() * q_bar * su +
      r_bar +
      difference.transpose() * r_delta_bar * difference;

    hessian =
      0.5 * (hessian + hessian.transpose()) +
      config_.regularization *
      Eigen::MatrixXd::Identity(
      input_size * horizon,
      input_size * horizon);

    const Eigen::VectorXd gradient =
      su.transpose() * q_bar *
      (affine_state - reference_state_vector) -
      r_bar * reference_control_vector -
      difference.transpose() * r_delta_bar * previous_offset;

    Eigen::MatrixXd constraint_matrix(
      2 * input_size * horizon,
      input_size * horizon);

    constraint_matrix.topRows(input_size * horizon) =
      Eigen::MatrixXd::Identity(
      input_size * horizon,
      input_size * horizon);
    constraint_matrix.bottomRows(input_size * horizon) = difference;

    Eigen::VectorXd lower_bound(2 * input_size * horizon);
    Eigen::VectorXd upper_bound(2 * input_size * horizon);

    for (int k = 0; k < horizon; ++k) {
      lower_bound.segment<input_size>(input_size * k) <<
        config_.v_min, -config_.omega_max;
      upper_bound.segment<input_size>(input_size * k) <<
        config_.v_max, config_.omega_max;
    }

    const Eigen::Vector2d maximum_delta(
      config_.acceleration_v * config_.dt,
      config_.acceleration_omega * config_.dt);

    lower_bound.segment<input_size>(input_size * horizon) =
      previous_command - maximum_delta;
    upper_bound.segment<input_size>(input_size * horizon) =
      previous_command + maximum_delta;

    for (int k = 1; k < horizon; ++k) {
      lower_bound.segment<input_size>(
        input_size * horizon + input_size * k) = -maximum_delta;
      upper_bound.segment<input_size>(
        input_size * horizon + input_size * k) = maximum_delta;
    }

    last_qp_result = solve_dense_qp(
      hessian,
      gradient,
      constraint_matrix,
      lower_bound,
      upper_bound,
      flatten_controls(controls),
      config_.qp);

    controls = unflatten_controls(last_qp_result.x, horizon);
    all_iterations_converged =
      all_iterations_converged && last_qp_result.converged;
  }

  previous_solution_ = controls;
  has_previous_solution_ = true;

  Eigen::Vector2d command = controls.col(0);
  command.x() = std::clamp(command.x(), config_.v_min, config_.v_max);
  command.y() = std::clamp(command.y(), -config_.omega_max, config_.omega_max);

  const Eigen::Vector2d maximum_delta(
    config_.acceleration_v * config_.dt,
    config_.acceleration_omega * config_.dt);

  Eigen::Vector2d delta = command - previous_command;
  delta.x() = std::clamp(delta.x(), -maximum_delta.x(), maximum_delta.x());
  delta.y() = std::clamp(delta.y(), -maximum_delta.y(), maximum_delta.y());
  command = previous_command + delta;
  controls.col(0) = command;

  MpcResult result;
  result.command = command;
  result.controls = controls;
  result.states = rollout(initial_state, controls);
  result.converged = all_iterations_converged;
  result.qp_iterations = last_qp_result.iterations;
  result.primal_residual = last_qp_result.primal_residual;
  result.dual_residual = last_qp_result.dual_residual;

  return result;
}

}  // namespace tb3_mpc_controller
