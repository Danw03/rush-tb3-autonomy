cmake_minimum_required(VERSION 3.8)
project(tb3_mpc_controller)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
  )
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

add_executable(
  mpc_controller_node
  src/mpc_controller_node.cpp
)

target_compile_features(
  mpc_controller_node
  PUBLIC
  cxx_std_17
)

ament_target_dependencies(
  mpc_controller_node
  rclcpp
  nav_msgs
  geometry_msgs
)

install(
  TARGETS
  mpc_controller_node
  DESTINATION
  lib/${PROJECT_NAME}
)

ament_package()