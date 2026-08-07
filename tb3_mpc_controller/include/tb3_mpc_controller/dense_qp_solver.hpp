#ifndef TB3_MPC_CONTROLLER__DENSE_QP_SOLVER_HPP_
#define TB3_MPC_CONTROLLER__DENSE_QP_SOLVER_HPP_

#include <limits>

#include <Eigen/Core>

namespace tb3_mpc_controller
{

struct QpSettings
{
  int max_iterations{250};
  double rho{1.0};
  double sigma{1.0e-6};
  double eps_abs{1.0e-4};
  double eps_rel{1.0e-4};
};

struct QpResult
{
  Eigen::VectorXd x;
  bool converged{false};
  int iterations{0};
  double primal_residual{std::numeric_limits<double>::infinity()};
  double dual_residual{std::numeric_limits<double>::infinity()};
};

QpResult solve_dense_qp(
  const Eigen::MatrixXd & hessian,
  const Eigen::VectorXd & gradient,
  const Eigen::MatrixXd & constraint_matrix,
  const Eigen::VectorXd & lower_bound,
  const Eigen::VectorXd & upper_bound,
  const Eigen::VectorXd & warm_start,
  const QpSettings & settings);

}  // namespace tb3_mpc_controller

#endif  // TB3_MPC_CONTROLLER__DENSE_QP_SOLVER_HPP_
