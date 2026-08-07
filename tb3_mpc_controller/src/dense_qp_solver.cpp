#include "tb3_mpc_controller/dense_qp_solver.hpp"

#include <algorithm>
#include <stdexcept>

#include <Eigen/Cholesky>

namespace tb3_mpc_controller
{
namespace
{

double infinity_norm(const Eigen::VectorXd & vector)
{
  return vector.size() == 0 ? 0.0 : vector.cwiseAbs().maxCoeff();
}

Eigen::VectorXd clamp_vector(
  const Eigen::VectorXd & value,
  const Eigen::VectorXd & lower,
  const Eigen::VectorXd & upper)
{
  return value.cwiseMax(lower).cwiseMin(upper);
}

}  // namespace

QpResult solve_dense_qp(
  const Eigen::MatrixXd & hessian,
  const Eigen::VectorXd & gradient,
  const Eigen::MatrixXd & constraint_matrix,
  const Eigen::VectorXd & lower_bound,
  const Eigen::VectorXd & upper_bound,
  const Eigen::VectorXd & warm_start,
  const QpSettings & settings)
{
  const int variable_count = static_cast<int>(hessian.rows());
  const int constraint_count = static_cast<int>(constraint_matrix.rows());

  if (
    hessian.cols() != variable_count ||
    gradient.size() != variable_count ||
    constraint_matrix.cols() != variable_count ||
    lower_bound.size() != constraint_count ||
    upper_bound.size() != constraint_count)
  {
    throw std::invalid_argument("Dense QP dimensions are inconsistent.");
  }

  if (settings.rho <= 0.0 || settings.sigma < 0.0) {
    throw std::invalid_argument("QP rho must be positive and sigma nonnegative.");
  }

  if ((lower_bound.array() > upper_bound.array()).any()) {
    throw std::invalid_argument("Dense QP lower bound exceeds upper bound.");
  }

  Eigen::VectorXd x = warm_start;
  if (x.size() != variable_count) {
    x = Eigen::VectorXd::Zero(variable_count);
  }

  Eigen::VectorXd z = clamp_vector(
    constraint_matrix * x, lower_bound, upper_bound);
  Eigen::VectorXd y = Eigen::VectorXd::Zero(constraint_count);

  const Eigen::MatrixXd system_matrix =
    hessian +
    settings.sigma * Eigen::MatrixXd::Identity(variable_count, variable_count) +
    settings.rho * constraint_matrix.transpose() * constraint_matrix;

  Eigen::LDLT<Eigen::MatrixXd> factorization(system_matrix);
  if (factorization.info() != Eigen::Success) {
    throw std::runtime_error("Dense QP factorization failed.");
  }

  QpResult result;
  result.x = x;

  for (int iteration = 1; iteration <= settings.max_iterations; ++iteration) {
    const Eigen::VectorXd right_hand_side =
      settings.sigma * x -
      gradient +
      constraint_matrix.transpose() * (settings.rho * z - y);

    x = factorization.solve(right_hand_side);
    if (factorization.info() != Eigen::Success || !x.allFinite()) {
      throw std::runtime_error("Dense QP linear solve failed.");
    }

    const Eigen::VectorXd z_previous = z;
    const Eigen::VectorXd ax = constraint_matrix * x;

    z = clamp_vector(
      ax + y / settings.rho,
      lower_bound,
      upper_bound);

    y += settings.rho * (ax - z);

    const double primal_residual = infinity_norm(ax - z);
    const double dual_residual = infinity_norm(
      settings.rho * constraint_matrix.transpose() * (z - z_previous));

    const double primal_tolerance =
      settings.eps_abs +
      settings.eps_rel * std::max(infinity_norm(ax), infinity_norm(z));

    const double dual_tolerance =
      settings.eps_abs +
      settings.eps_rel * infinity_norm(constraint_matrix.transpose() * y);

    result.x = x;
    result.iterations = iteration;
    result.primal_residual = primal_residual;
    result.dual_residual = dual_residual;

    if (
      primal_residual <= primal_tolerance &&
      dual_residual <= dual_tolerance)
    {
      result.converged = true;
      break;
    }
  }

  return result;
}

}  // namespace tb3_mpc_controller
