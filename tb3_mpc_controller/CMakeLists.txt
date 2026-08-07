cmake_minimum_required(VERSION 3.8)
project(tb3_mpc_controller)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

add_library(mpc_core
  src/dense_qp_solver.cpp
  src/mpc_solver.cpp
)
target_include_directories(mpc_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(mpc_core PUBLIC Eigen3::Eigen)

add_executable(mpc_controller_node src/mpc_controller_node.cpp)
target_link_libraries(mpc_controller_node mpc_core Eigen3::Eigen)
ament_target_dependencies(mpc_controller_node
  geometry_msgs
  nav_msgs
  rclcpp
  std_msgs
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  TARGETS mpc_core mpc_controller_node
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
  INCLUDES DESTINATION include
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_dense_qp_solver test/test_dense_qp_solver.cpp)
  if(TARGET test_dense_qp_solver)
    target_link_libraries(test_dense_qp_solver mpc_core Eigen3::Eigen)
  endif()

  ament_add_gtest(test_mpc_solver test/test_mpc_solver.cpp)
  if(TARGET test_mpc_solver)
    target_link_libraries(test_mpc_solver mpc_core Eigen3::Eigen)
  endif()
endif()

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3)
ament_package()
