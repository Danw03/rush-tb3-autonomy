#include <gtest/gtest.h>

#include <Eigen/Core>

#include "tb3_mpc_controller/dense_qp_solver.hpp"

namespace tb3_mpc_controller
{

TEST(DenseQpSolver, SolvesBoxConstrainedQuadratic)
{
  Eigen::MatrixXd hessian(1, 1);
  hessian << 1.0;

  Eigen::VectorXd gradient(1);
  gradient << -2.0;

  Eigen::MatrixXd constraint(1, 1);
  constraint << 1.0;

  Eigen::VectorXd lower(1);
  lower << 0.0;

  Eigen::VectorXd upper(1);
  upper << 1.0;

  QpSettings settings;
  settings.max_iterations = 1000;
  settings.eps_abs = 1.0e-7;
  settings.eps_rel = 1.0e-7;

  const QpResult result = solve_dense_qp(
    hessian,
    gradient,
    constraint,
    lower,
    upper,
    Eigen::VectorXd::Zero(1),
    settings);

  EXPECT_TRUE(result.converged);
  ASSERT_EQ(result.x.size(), 1);
  EXPECT_NEAR(result.x.x(), 1.0, 1.0e-5);
}

TEST(DenseQpSolver, RejectsInvertedBounds)
{
  Eigen::MatrixXd hessian = Eigen::MatrixXd::Identity(1, 1);
  Eigen::VectorXd gradient = Eigen::VectorXd::Zero(1);
  Eigen::MatrixXd constraint = Eigen::MatrixXd::Identity(1, 1);

  Eigen::VectorXd lower(1);
  lower << 1.0;

  Eigen::VectorXd upper(1);
  upper << 0.0;

  EXPECT_THROW(
    solve_dense_qp(
      hessian,
      gradient,
      constraint,
      lower,
      upper,
      Eigen::VectorXd::Zero(1),
      QpSettings{}),
    std::invalid_argument);
}

}  // namespace tb3_mpc_controller
