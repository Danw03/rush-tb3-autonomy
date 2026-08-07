#include <gtest/gtest.h>

#include <cmath>

#include <Eigen/Core>

#include "tb3_mpc_controller/mpc_solver.hpp"

namespace tb3_mpc_controller
{

TEST(ConvexMpcSolver, TracksStraightReferenceWithinLimits)
{
  MpcConfig config;
  config.dt = 0.1;
  config.horizon_steps = 10;
  config.linearization_iterations = 2;

  ConvexMpcSolver solver(config);

  Eigen::MatrixXd reference_states(3, config.horizon_steps + 1);
  Eigen::MatrixXd reference_controls(2, config.horizon_steps);

  constexpr double reference_speed = 0.03;
  for (int k = 0; k <= config.horizon_steps; ++k) {
    reference_states.col(k) <<
      static_cast<double>(k) * reference_speed * config.dt,
      0.0,
      0.0;

    if (k < config.horizon_steps) {
      reference_controls.col(k) << reference_speed, 0.0;
    }
  }

  const MpcResult result = solver.solve(
    Eigen::Vector3d::Zero(),
    Eigen::Vector2d::Zero(),
    reference_states,
    reference_controls);

  ASSERT_EQ(result.states.rows(), 3);
  ASSERT_EQ(result.states.cols(), config.horizon_steps + 1);
  ASSERT_EQ(result.controls.rows(), 2);
  ASSERT_EQ(result.controls.cols(), config.horizon_steps);

  EXPECT_TRUE(result.command.allFinite());
  EXPECT_GE(result.command.x(), config.v_min - 1.0e-9);
  EXPECT_LE(result.command.x(), config.acceleration_v * config.dt + 1.0e-9);
  EXPECT_LE(std::abs(result.command.y()), config.omega_max + 1.0e-9);
  EXPECT_NEAR(result.command.y(), 0.0, 1.0e-3);
}

TEST(ConvexMpcSolver, RejectsReferenceWithWrongShape)
{
  MpcConfig config;
  config.horizon_steps = 5;
  ConvexMpcSolver solver(config);

  EXPECT_THROW(
    solver.solve(
      Eigen::Vector3d::Zero(),
      Eigen::Vector2d::Zero(),
      Eigen::MatrixXd::Zero(3, 5),
      Eigen::MatrixXd::Zero(2, 5)),
    std::invalid_argument);
}

}  // namespace tb3_mpc_controller
