cmake_minimum_required(VERSION 3.14)
project(ga_ocp_ros2)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 OLD)
endif()
set(COAL_DISABLE_HPP_FCL_WARNINGS TRUE)
add_compile_definitions(COAL_DISABLE_HPP_FCL_WARNINGS)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ga_ocp_core REQUIRED)
find_package(TetraPGA CONFIG REQUIRED)
find_package(pinocchio REQUIRED)
find_package(crocoddyl REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(interactive_markers REQUIRED)
find_package(pluginlib REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets)
find_package(rviz_common REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(visualization_msgs REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(CASADI QUIET casadi)
option(GA_OCP_ROS2_ENABLE_CASADI "Enable CasADi backend for closed_loop_mpc_node" ON)
if(GA_OCP_ROS2_ENABLE_CASADI AND CASADI_FOUND)
  find_package(benchmark REQUIRED)
endif()
if(GA_OCP_ROS2_ENABLE_CASADI AND NOT CASADI_FOUND)
  message(WARNING "CasADi was not found; closed_loop_mpc_node will reject backend:=casadi")
endif()

include_directories(
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(target_interactive_marker_node src/target_interactive_marker_node.cpp)
ament_target_dependencies(target_interactive_marker_node
  ga_ocp_core
  rclcpp
  geometry_msgs
  interactive_markers
  visualization_msgs
)

add_executable(crocoddyl_motor_planner_node src/crocoddyl_motor_planner_node.cpp)
target_link_libraries(crocoddyl_motor_planner_node
  TetraPGA::TetraPGA
  crocoddyl::crocoddyl
  pinocchio::pinocchio
)
ament_target_dependencies(crocoddyl_motor_planner_node
  ament_index_cpp
  ga_ocp_core
  rclcpp
  geometry_msgs
  sensor_msgs
  std_msgs
  std_srvs
  trajectory_msgs
  visualization_msgs
)

add_executable(crocoddyl_collision_planner_node src/crocoddyl_collision_planner_node.cpp)
target_link_libraries(crocoddyl_collision_planner_node
  TetraPGA::TetraPGA
  crocoddyl::crocoddyl
  pinocchio::pinocchio
)
ament_target_dependencies(crocoddyl_collision_planner_node
  ga_ocp_core
  rclcpp
  geometry_msgs
  sensor_msgs
  std_msgs
  std_srvs
  trajectory_msgs
  visualization_msgs
)

add_executable(closed_loop_mpc_node src/closed_loop_mpc_node.cpp)
target_link_libraries(closed_loop_mpc_node
  TetraPGA::TetraPGA
  crocoddyl::crocoddyl
  pinocchio::pinocchio
)
if(GA_OCP_ROS2_ENABLE_CASADI AND CASADI_FOUND)
  target_link_libraries(closed_loop_mpc_node ${CASADI_LIBRARIES} benchmark::benchmark)
  target_compile_definitions(closed_loop_mpc_node PRIVATE GA_OCP_HAS_CASADI_BENCH=1)
  target_include_directories(closed_loop_mpc_node PRIVATE ${CASADI_INCLUDE_DIRS})
  target_link_directories(closed_loop_mpc_node PRIVATE ${CASADI_LIBRARY_DIRS})
endif()
ament_target_dependencies(closed_loop_mpc_node
  ament_index_cpp
  ga_ocp_core
  rclcpp
  sensor_msgs
  std_msgs
  trajectory_msgs
)

add_executable(static_obstacle_marker_node src/static_obstacle_marker_node.cpp)
ament_target_dependencies(static_obstacle_marker_node
  ga_ocp_core
  rclcpp
  geometry_msgs
  visualization_msgs
)

add_library(ga_ocp_planning_panel SHARED src/ga_ocp_planning_panel.cpp)
target_link_libraries(ga_ocp_planning_panel Qt5::Widgets)
ament_target_dependencies(ga_ocp_planning_panel
  ga_ocp_core
  rclcpp
  rviz_common
  pluginlib
  std_msgs
  std_srvs
)

set(GA_OCP_ROS2_TARGETS
  target_interactive_marker_node
  crocoddyl_motor_planner_node
  crocoddyl_collision_planner_node
  closed_loop_mpc_node
  static_obstacle_marker_node
)

install(TARGETS ${GA_OCP_ROS2_TARGETS}
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/joint_command_executor.py
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS ga_ocp_planning_panel
  LIBRARY DESTINATION lib
)

set(_ga_ocp_ros2_install_dirs
  config
  launch
  rviz
)

foreach(_ga_ocp_ros2_dir IN LISTS _ga_ocp_ros2_install_dirs)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_ga_ocp_ros2_dir}")
    install(
      DIRECTORY "${_ga_ocp_ros2_dir}/"
      DESTINATION share/${PROJECT_NAME}/${_ga_ocp_ros2_dir}
    )
  endif()
endforeach()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/plugin_description.xml")
  install(FILES plugin_description.xml DESTINATION share/${PROJECT_NAME})
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
  install(FILES README.md DESTINATION share/${PROJECT_NAME})
endif()

ament_package()
