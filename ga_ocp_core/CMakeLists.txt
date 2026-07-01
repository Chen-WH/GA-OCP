cmake_minimum_required(VERSION 3.14)
project(ga_ocp_core)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 OLD)
endif()
set(COAL_DISABLE_HPP_FCL_WARNINGS TRUE)
add_compile_definitions(COAL_DISABLE_HPP_FCL_WARNINGS)

# Debug configuration
# set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall")

# Release configuration
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -Wall -g")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(GA_OCP_BUILD_CROCODDYL_TESTS "Build Crocoddyl integration tests" ON)
option(GA_OCP_BUILD_BENCHMARKS "Build Crocoddyl benchmarks" OFF)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(TetraPGA CONFIG REQUIRED)
find_package(pinocchio REQUIRED)
find_package(crocoddyl REQUIRED)

if(NOT TetraPGA_ROBOT_ASSETS_DIR)
    message(FATAL_ERROR "TetraPGA_ROBOT_ASSETS_DIR was not exported by TetraPGAConfig.cmake")
endif()

if(NOT EXISTS "${TetraPGA_ROBOT_ASSETS_DIR}/ur10/urdf/ur10.urdf")
    message(FATAL_ERROR
        "TetraPGA_ROBOT_ASSETS_DIR must point to the robot-assets repository. "
        "Expected to find ur10/urdf/ur10.urdf under '${TetraPGA_ROBOT_ASSETS_DIR}'.")
endif()

if(GA_OCP_BUILD_CROCODDYL_TESTS OR GA_OCP_BUILD_BENCHMARKS)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(COAL REQUIRED IMPORTED_TARGET coal)
endif()

if(GA_OCP_BUILD_BENCHMARKS)
    find_package(benchmark REQUIRED)
    find_package(mim_solvers CONFIG REQUIRED)
    pkg_check_modules(CASADI REQUIRED casadi)
endif()

include_directories(
    include
    ${EIGEN3_INCLUDE_DIR}
)

set(GA_OCP_CROCODDYL_TEST_TARGETS)
set(GA_OCP_BENCHMARK_TARGETS)

function(ga_ocp_add_crocoddyl_test target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name}
        TetraPGA::TetraPGA
        crocoddyl::crocoddyl
        pinocchio::pinocchio
    )
    target_compile_definitions(${target_name}
        PRIVATE GA_OCP_ROBOT_ASSETS_DIR="${TetraPGA_ROBOT_ASSETS_DIR}")
    list(APPEND GA_OCP_CROCODDYL_TEST_TARGETS ${target_name})
    set(GA_OCP_CROCODDYL_TEST_TARGETS "${GA_OCP_CROCODDYL_TEST_TARGETS}" PARENT_SCOPE)
endfunction()

function(ga_ocp_add_benchmark target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name}
        TetraPGA::TetraPGA
        benchmark::benchmark
        crocoddyl::crocoddyl
        pinocchio::pinocchio
        ${CASADI_LIBRARIES}
    )
    target_compile_definitions(${target_name}
        PRIVATE
          GA_OCP_ROBOT_ASSETS_DIR="${TetraPGA_ROBOT_ASSETS_DIR}"
          GA_OCP_HAS_CASADI_BENCH=1)
    target_include_directories(${target_name} PRIVATE ${CASADI_INCLUDE_DIRS})
    target_link_directories(${target_name} PRIVATE ${CASADI_LIBRARY_DIRS})
    list(APPEND GA_OCP_BENCHMARK_TARGETS ${target_name})
    set(GA_OCP_BENCHMARK_TARGETS "${GA_OCP_BENCHMARK_TARGETS}" PARENT_SCOPE)
endfunction()

if(GA_OCP_BUILD_CROCODDYL_TESTS)
    ga_ocp_add_crocoddyl_test(Crocoddyl_fddp_test test/Crocoddyl_fddp_test.cpp)
    ga_ocp_add_crocoddyl_test(Crocoddyl_torLim_test test/Crocoddyl_torLim_test.cpp)
    ga_ocp_add_crocoddyl_test(Crocoddyl_velLim_test test/Crocoddyl_velLim_test.cpp)
    ga_ocp_add_crocoddyl_test(Crocoddyl_motor_test test/Crocoddyl_motor_test.cpp)
    ga_ocp_add_crocoddyl_test(Crocoddyl_collision_test test/Crocoddyl_collision_test.cpp)
    ga_ocp_add_crocoddyl_test(Crocoddyl_idmpc_test test/Crocoddyl_idmpc_test.cpp)
    target_link_libraries(Crocoddyl_collision_test PkgConfig::COAL)
endif()

if(GA_OCP_BUILD_BENCHMARKS)
    ga_ocp_add_benchmark(Crocoddyl_fddp_bench benchmark/Crocoddyl_fddp_bench.cpp)
    ga_ocp_add_benchmark(Crocoddyl_fddp_budget_bench benchmark/Crocoddyl_fddp_budget_bench.cpp)
    ga_ocp_add_benchmark(Crocoddyl_runtime_breakdown benchmark/Crocoddyl_runtime_breakdown.cpp)
    target_compile_definitions(Crocoddyl_runtime_breakdown PRIVATE NDEBUG)
    target_compile_options(Crocoddyl_runtime_breakdown PRIVATE -O3)
    ga_ocp_add_benchmark(Crocoddyl_sqp_bench benchmark/Crocoddyl_sqp_bench.cpp)
    ga_ocp_add_benchmark(Crocoddyl_obstacle_margin_sweep benchmark/Crocoddyl_obstacle_margin_sweep.cpp)
    ga_ocp_add_benchmark(Crocoddyl_robustness_sweep benchmark/Crocoddyl_robustness_sweep.cpp)
    target_link_libraries(Crocoddyl_sqp_bench mim_solvers::mim_solvers)
endif()

if(GA_OCP_BUILD_CROCODDYL_TESTS)
    install(TARGETS ${GA_OCP_CROCODDYL_TEST_TARGETS} DESTINATION lib/${PROJECT_NAME})
endif()

if(GA_OCP_BUILD_BENCHMARKS)
    install(TARGETS ${GA_OCP_BENCHMARK_TARGETS} DESTINATION lib/${PROJECT_NAME})
endif()

install(DIRECTORY include/ DESTINATION include)

set(GA_OCP_ROBOT_ASSET_PACKAGES
    franka_panda
    jaka_zu12
    kinect_dk
    leap_hand
    realsense_d435i
    reorientation_cube
    robotiq_2f85
    stanford_tidybot
    ur10)

foreach(_ga_ocp_robot_asset IN LISTS GA_OCP_ROBOT_ASSET_PACKAGES)
    install(
        DIRECTORY "${TetraPGA_ROBOT_ASSETS_DIR}/${_ga_ocp_robot_asset}"
        DESTINATION share/${PROJECT_NAME}/robot-assets
    )
endforeach()

ament_export_include_directories(include)
ament_export_dependencies(Eigen3 TetraPGA pinocchio crocoddyl)

if(BUILD_TESTING)
    find_package(ament_lint_auto REQUIRED)
    set(ament_cmake_cpplint_FOUND TRUE)
    ament_lint_auto_find_test_dependencies()
endif()

ament_package()
