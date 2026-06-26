#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <cstdlib>
#include <random>
#include <iomanip>
#include <limits>

// Crocoddyl Includes
#include <crocoddyl/core/state-base.hpp>
#include <crocoddyl/core/states/euclidean.hpp> // 使用 StateVector
#include <crocoddyl/core/solvers/box-fddp.hpp>
#include <crocoddyl/core/integrator/euler.hpp>
#include <crocoddyl/core/costs/cost-sum.hpp>
#include <crocoddyl/core/costs/residual.hpp>
#include <crocoddyl/core/activations/quadratic-barrier.hpp>
#include <crocoddyl/multibody/residuals/state.hpp>
#include <crocoddyl/multibody/residuals/frame-placement.hpp>
#include <crocoddyl/core/utils/callbacks.hpp>
#include <crocoddyl/core/numdiff/diff-action.hpp> // 用于验证导数

// Pinocchio Includes
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <crocoddyl/multibody/states/multibody.hpp>
#include <crocoddyl/multibody/actuations/full.hpp>
#include <crocoddyl/multibody/actions/free-fwddyn.hpp>
#include <pinocchio/algorithm/frames.hpp>

// TetraPGA Includes
#include "TetraPGA/ModelRepo.hpp"
#include "ga_ocp/CrocoddylActions.hpp"
#include "ga_ocp/CrocoddylResiduals.hpp"

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::milliseconds milliseconds;

int main() {
    double dt = 0.008; // 时间步长
    Model<double> ur_model = ur();
    const Eigen::VectorXd torque_lb = -ur_model.effortLimit;
    const Eigen::VectorXd torque_ub = ur_model.effortLimit;

    constexpr unsigned int kRandomSeed = 20260320;
    std::srand(kRandomSeed);
    
    // =========================================================================
    // Generate Random Initial State and Target State (for both methods)
    // =========================================================================
    const int dof = 6;
    Eigen::VectorXd shared_q0 = Eigen::VectorXd::Random(dof);
    Eigen::VectorXd shared_q_ref = Eigen::VectorXd::Random(dof);
    Eigen::VectorXd shared_x0(2 * dof);
    shared_x0.head(dof) = shared_q0;
    shared_x0.tail(dof).setZero(); // Set velocities to zero for initial state
    
    std::cout << "Shared q0: " << shared_q0.transpose() << std::endl;
    std::cout << "Shared q_ref: " << shared_q_ref.transpose() << std::endl;
    std::cout << "Shared x0: " << shared_x0.transpose() << std::endl;
    std::cout << "Random seed: " << kRandomSeed << std::endl;
    
    // =========================================================================
    // Part A: Pinocchio-based 对照组 (Baseline using Pinocchio)
    // =========================================================================
    std::cout << "========================================" << std::endl;
    std::cout << "Part A: Pinocchio Baseline Test" << std::endl;
    std::cout << "========================================" << std::endl;
    
    auto pin_start = Clock::now();
    
    // 1. 加载 Pinocchio 模型
    const std::string urdf_path = std::string(GA_OCP_ROBOT_ASSETS_DIR) + "/ur10/urdf/ur10.urdf";
    pinocchio::Model pin_model;
    pinocchio::urdf::buildModel(urdf_path, pin_model);
    
    std::cout << "[Pinocchio] Loaded model with " << pin_model.nv << " DOFs" << std::endl;
    
    // 2. 创建 Crocoddyl Multibody State
    auto pin_state = std::make_shared<crocoddyl::StateMultibody>(
        std::make_shared<pinocchio::Model>(pin_model));
    
    // 3. 定义 Cost 函数
    auto pin_running_cost = std::make_shared<crocoddyl::CostModelSum>(pin_state);
    auto pin_terminal_cost = std::make_shared<crocoddyl::CostModelSum>(pin_state);
    
    // Frame placement cost for end effector
    pinocchio::FrameIndex frame_id = pin_model.getFrameId("tool0");
    pinocchio::Data pin_data_ref(pin_model);
    pinocchio::forwardKinematics(pin_model, pin_data_ref, shared_q_ref);
    pinocchio::updateFramePlacements(pin_model, pin_data_ref);
    pinocchio::SE3 pin_placement_ref = pin_data_ref.oMf[frame_id];
    
    auto pin_placement_residual = std::make_shared<crocoddyl::ResidualModelFramePlacement>(
        pin_state, frame_id, pin_placement_ref);
    auto pin_placement_cost = std::make_shared<crocoddyl::CostModelResidual>(pin_state, pin_placement_residual);

    Eigen::VectorXd pin_x_zero = Eigen::VectorXd::Zero(pin_state->get_nx());
    auto pin_vel_residual = std::make_shared<crocoddyl::ResidualModelState>(
        pin_state, pin_x_zero, pin_model.nv);
    crocoddyl::ActivationBounds pin_vel_bounds;
    pin_vel_bounds.lb = Eigen::VectorXd::Constant(pin_state->get_nx(), -std::numeric_limits<double>::infinity());
    pin_vel_bounds.ub = Eigen::VectorXd::Constant(pin_state->get_nx(), std::numeric_limits<double>::infinity());
    for (int i = 0; i < pin_model.nv; ++i) {
      const double vlim = ur_model.velocityLimit[i];
      pin_vel_bounds.lb[pin_model.nv + i] = -vlim;
      pin_vel_bounds.ub[pin_model.nv + i] = vlim;
    }
    auto pin_vel_activation =
        std::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(pin_vel_bounds);
    auto pin_vel_cost = std::make_shared<crocoddyl::CostModelResidual>(
        pin_state, pin_vel_activation, pin_vel_residual);
    
    pin_running_cost->addCost("placement_reg", pin_placement_cost, 10.0);
    pin_running_cost->addCost("vel_limit", pin_vel_cost, 100.0);
    pin_terminal_cost->addCost("placement_reg", pin_placement_cost, 100.0);
    pin_terminal_cost->addCost("vel_limit", pin_vel_cost, 100.0);
    
    // 4. 创建 Actuation 和 Action Model
    auto pin_actuation = std::make_shared<crocoddyl::ActuationModelFull>(pin_state);
    pin_actuation->set_u_lb(torque_lb);
    pin_actuation->set_u_ub(torque_ub);
    auto pin_diff_model = std::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(
        pin_state, pin_actuation, pin_running_cost);
    auto pin_diff_model_term = std::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(
        pin_state, pin_actuation, pin_terminal_cost);
    
    // 5. 验证导数
    std::cout << "\n[Pinocchio Validation] Checking derivatives..." << std::endl;
    auto pin_num_diff = std::make_shared<crocoddyl::DifferentialActionModelNumDiff>(pin_diff_model);
    auto pin_num_data = pin_num_diff->createData();
    auto pin_diff_data = pin_diff_model->createData();
    
    Eigen::VectorXd pin_x_rand = pin_state->rand();
    Eigen::VectorXd pin_u_rand = Eigen::VectorXd::Random(pin_model.nv);
    
    pin_diff_model->calc(pin_diff_data, pin_x_rand, pin_u_rand);
    pin_diff_model->calcDiff(pin_diff_data, pin_x_rand, pin_u_rand);
    pin_num_diff->calc(pin_num_data, pin_x_rand, pin_u_rand);
    pin_num_diff->calcDiff(pin_num_data, pin_x_rand, pin_u_rand);
    
    // 6. 设置优化问题并求解
    auto pin_running_IAM = std::make_shared<crocoddyl::IntegratedActionModelEuler>(pin_diff_model, dt);
    auto pin_terminal_IAM = std::make_shared<crocoddyl::IntegratedActionModelEuler>(pin_diff_model_term, dt);
    
    long T_pin = 100;
    std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> pin_running_models(T_pin, pin_running_IAM);
    
    // 使用共享的初始状态
    Eigen::VectorXd pin_x0 = shared_x0;
    
    auto pin_problem = std::make_shared<crocoddyl::ShootingProblem>(pin_x0, pin_running_models, pin_terminal_IAM);
    crocoddyl::SolverBoxFDDP pin_solver(pin_problem);
    pin_solver.set_th_stop(1e-6);
    
    std::vector<std::shared_ptr<crocoddyl::CallbackAbstract>> pin_callbacks;
    auto pin_solve_start = Clock::now();
    const bool pin_converged = pin_solver.solve({}, {}, 300);
    auto pin_solve_end = Clock::now();
    
    auto pin_end = Clock::now();
    
    auto pin_total_ms = std::chrono::duration_cast<milliseconds>(pin_end - pin_start).count();
    auto pin_solve_ms = std::chrono::duration_cast<milliseconds>(pin_solve_end - pin_solve_start).count();
    
    std::cout << "\n[Pinocchio Results]" << std::endl;
    std::cout << "Final Cost: " << pin_solver.get_cost() << std::endl;
    std::cout << "Converged:  " << (pin_converged ? "YES" : "NO") << std::endl;
    std::cout << "Stop:       " << pin_solver.get_stop()
              << ", Iter: " << pin_solver.get_iter() << std::endl;
    std::cout << "Total Time: " << pin_total_ms << " ms, Solve Time: " << pin_solve_ms << " ms" << std::endl;
    
    // =========================================================================
    // Verify End-Effector Placement
    // =========================================================================
    std::cout << "\n[End-Effector Verification]" << std::endl;
    
    // 获取规划得到的最后一个状态
    Eigen::VectorXd pin_x_terminal = pin_solver.get_xs().back();
    
    // 使用 pinocchio 计算末端执行器位姿
    pinocchio::Data pin_data(pin_model);
    pinocchio::forwardKinematics(pin_model, pin_data, pin_x_terminal.head(pin_model.nq));
    pinocchio::updateFramePlacements(pin_model, pin_data);

    pinocchio::SE3 actual_placement = pin_data.oMf[frame_id];
    std::cout << "\nFinal joint angles (Pinocchio): "
              << pin_x_terminal.head(pin_model.nq).transpose() << std::endl;
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nTarget End-Effector Position: " << std::endl;
    std::cout << "  X: " << pin_placement_ref.translation()(0) << std::endl;
    std::cout << "  Y: " << pin_placement_ref.translation()(1) << std::endl;
    std::cout << "  Z: " << pin_placement_ref.translation()(2) << std::endl;
    
    std::cout << "\nActual End-Effector Position: " << std::endl;
    std::cout << "  X: " << actual_placement.translation()(0) << std::endl;
    std::cout << "  Y: " << actual_placement.translation()(1) << std::endl;
    std::cout << "  Z: " << actual_placement.translation()(2) << std::endl;
    
    std::cout << "\nPosition Error (Euclidean): " << std::endl;
    Eigen::Vector3d pos_error = actual_placement.translation() - pin_placement_ref.translation();
    std::cout << "  Error Norm: " << pos_error.norm() << std::endl;
    
    std::cout << "\nTarget End-Effector Rotation (Quaternion):" << std::endl;
    Eigen::Quaterniond quat_ref(pin_placement_ref.rotation());
    std::cout << "  qx: " << quat_ref.x() << ", qy: " << quat_ref.y() << ", qz: " << quat_ref.z() << ", qw: " << quat_ref.w() << std::endl;
    
    std::cout << "\nActual End-Effector Rotation (Quaternion):" << std::endl;
    Eigen::Quaterniond quat_actual(actual_placement.rotation());
    std::cout << "  qx: " << quat_actual.x() << ", qy: " << quat_actual.y() << ", qz: " << quat_actual.z() << ", qw: " << quat_actual.w() << std::endl;

    // =========================================================================
    // Part B: TetraPGA Placement Residual Test
    // =========================================================================
    std::cout << "\n========================================" << std::endl;
    std::cout << "Part B: TetraPGA Placement Residual Test" << std::endl;
    std::cout << "========================================" << std::endl;

    auto ga_start = Clock::now();

    // 1. 设置模型
    auto state = std::make_shared<crocoddyl::StateVector>(2 * ur_model.dof_a);

    // 2. 定义 Cost 函数
    auto running_cost_model = std::make_shared<crocoddyl::CostModelSum>(state);
    auto terminal_cost_model = std::make_shared<crocoddyl::CostModelSum>(state);

    Data<double> ga_ref_data(ur_model);
    forwardKinematics(ur_model, ga_ref_data, shared_q_ref);
    Data<double> ga_initial_data(ur_model);
    forwardKinematics(ur_model, ga_initial_data, shared_q0);
    const Motor3D<double> ga_M_ref = align_motor_hemisphere(
        ga_ref_data.M.col(ur_model.n - 1), ga_initial_data.M.col(ur_model.n - 1));
    std::cout << "Target Motor: " << ga_M_ref.transpose() << std::endl;

    auto ga_placement_residual = std::make_shared<ResidualModelTetraPGAFramePlacement<double>>(
        state, ur_model, ga_M_ref);
    auto ga_placement_cost = std::make_shared<crocoddyl::CostModelResidual>(state, ga_placement_residual);

    Eigen::VectorXd x_zero = Eigen::VectorXd::Zero(2 * ur_model.dof_a);
    auto vel_residual = std::make_shared<crocoddyl::ResidualModelState>(
        state, x_zero, ur_model.dof_a);
    crocoddyl::ActivationBounds vel_bounds;
    vel_bounds.lb = Eigen::VectorXd::Constant(2 * ur_model.dof_a, -std::numeric_limits<double>::infinity());
    vel_bounds.ub = Eigen::VectorXd::Constant(2 * ur_model.dof_a, std::numeric_limits<double>::infinity());
    for (int i = 0; i < ur_model.dof_a; ++i) {
      const double vlim = ur_model.velocityLimit[i];
      vel_bounds.lb[ur_model.dof_a + i] = -vlim;
      vel_bounds.ub[ur_model.dof_a + i] = vlim;
    }
    auto vel_activation =
        std::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(vel_bounds);
    auto vel_limit_cost = std::make_shared<crocoddyl::CostModelResidual>(
        state, vel_activation, vel_residual);

    running_cost_model->addCost("placement_reg", ga_placement_cost, 10.0);
    running_cost_model->addCost("vel_limit", vel_limit_cost, 100.0);
    terminal_cost_model->addCost("placement_reg", ga_placement_cost, 100.0);
    terminal_cost_model->addCost("vel_limit", vel_limit_cost, 100.0);

    // 3. 构建 Action Model
    auto ga_diff_model = std::make_shared<DifferentialActionModelTetraPGAForwardDynamics<double>>(
        state, ur_model, running_cost_model);
    auto ga_diff_model_term = std::make_shared<DifferentialActionModelTetraPGAForwardDynamics<double>>(
        state, ur_model, terminal_cost_model);
    ga_diff_model->set_u_lb(torque_lb);
    ga_diff_model->set_u_ub(torque_ub);
    ga_diff_model_term->set_u_lb(torque_lb);
    ga_diff_model_term->set_u_ub(torque_ub);

    // 4. 构建最优控制问题并求解
    auto ga_running_IAM = std::make_shared<crocoddyl::IntegratedActionModelEuler>(ga_diff_model, dt);
    auto ga_terminal_IAM = std::make_shared<crocoddyl::IntegratedActionModelEuler>(ga_diff_model_term, dt);

    long T_ga = 100;
    std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> ga_running_models(T_ga, ga_running_IAM);

    Eigen::VectorXd ga_x0 = shared_x0;
    auto ga_problem = std::make_shared<crocoddyl::ShootingProblem>(ga_x0, ga_running_models, ga_terminal_IAM);

    crocoddyl::SolverBoxFDDP ga_solver(ga_problem);
    ga_solver.set_th_stop(1e-6);

    auto ga_solve_start = Clock::now();
    const bool ga_converged = ga_solver.solve({}, {}, 300);
    auto ga_solve_end = Clock::now();

    auto ga_end = Clock::now();

    auto ga_total_ms = std::chrono::duration_cast<milliseconds>(ga_end - ga_start).count();
    auto ga_solve_ms = std::chrono::duration_cast<milliseconds>(ga_solve_end - ga_solve_start).count();

    std::cout << "\n[TetraPGA Placement Results]" << std::endl;
    std::cout << "Final Cost: " << ga_solver.get_cost() << std::endl;
    std::cout << "Converged:  " << (ga_converged ? "YES" : "NO") << std::endl;
    std::cout << "Stop:       " << ga_solver.get_stop()
              << ", Iter: " << ga_solver.get_iter() << std::endl;
    std::cout << "Total Time: " << ga_total_ms << " ms, Solve Time: " << ga_solve_ms << " ms" << std::endl;

    // =========================================================================
    // Verify End-Effector Placement (TetraPGA)
    // =========================================================================
    std::cout << "\n[End-Effector Verification - TetraPGA]" << std::endl;

    Eigen::VectorXd ga_x_terminal = ga_solver.get_xs().back();
    Data<double> ga_data(ur_model);
    forwardKinematics(ur_model, ga_data, ga_x_terminal.head(ur_model.dof_a));

    const Motor3D<double> ga_M_actual = ga_data.M.col(ur_model.n - 1);
    const Line3D<double> ga_r = ga_log(ga_mul(ga_rev(ga_M_ref), ga_M_actual));
    std::cout << "Actual Motor: " << ga_M_actual.transpose() << std::endl;
    std::cout << "Residual norm (log map): " << ga_r.norm() << std::endl;

    // =========================================================================
    // Part C: TetraPGA-based Motor Residual Test
    // =========================================================================
    std::cout << "\n========================================" << std::endl;
    std::cout << "Part C: TetraPGA Motor Residual Test" << std::endl;
    std::cout << "========================================" << std::endl;

    auto ga_motor_start = Clock::now();

    auto motor_running_cost_model = std::make_shared<crocoddyl::CostModelSum>(state);
    auto motor_terminal_cost_model = std::make_shared<crocoddyl::CostModelSum>(state);

    auto ga_motor_residual = std::make_shared<ResidualModelTetraPGAFrameMotor<double>>(
        state, ur_model, ga_M_ref);
    auto ga_motor_cost = std::make_shared<crocoddyl::CostModelResidual>(state, ga_motor_residual);

    motor_running_cost_model->addCost("motor_reg", ga_motor_cost, 10.0);
    motor_running_cost_model->addCost("vel_limit", vel_limit_cost, 100.0);
    motor_terminal_cost_model->addCost("motor_reg", ga_motor_cost, 100.0);
    motor_terminal_cost_model->addCost("vel_limit", vel_limit_cost, 100.0);

    auto ga_motor_diff_model = std::make_shared<DifferentialActionModelTetraPGAForwardDynamics<double>>(
        state, ur_model, motor_running_cost_model);
    auto ga_motor_diff_model_term = std::make_shared<DifferentialActionModelTetraPGAForwardDynamics<double>>(
        state, ur_model, motor_terminal_cost_model);
    ga_motor_diff_model->set_u_lb(torque_lb);
    ga_motor_diff_model->set_u_ub(torque_ub);
    ga_motor_diff_model_term->set_u_lb(torque_lb);
    ga_motor_diff_model_term->set_u_ub(torque_ub);

    auto ga_motor_running_IAM = std::make_shared<crocoddyl::IntegratedActionModelEuler>(
        ga_motor_diff_model, dt);
    auto ga_motor_terminal_IAM = std::make_shared<crocoddyl::IntegratedActionModelEuler>(
        ga_motor_diff_model_term, dt);

    std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> ga_motor_running_models(
        T_ga, ga_motor_running_IAM);

    auto ga_motor_problem = std::make_shared<crocoddyl::ShootingProblem>(
        ga_x0, ga_motor_running_models, ga_motor_terminal_IAM);

    crocoddyl::SolverBoxFDDP ga_motor_solver(ga_motor_problem);
    ga_motor_solver.set_th_stop(1e-6);

    auto ga_motor_solve_start = Clock::now();
    const bool ga_motor_converged = ga_motor_solver.solve({}, {}, 300);
    auto ga_motor_solve_end = Clock::now();

    auto ga_motor_end = Clock::now();

    auto ga_motor_total_ms = std::chrono::duration_cast<milliseconds>(ga_motor_end - ga_motor_start).count();
    auto ga_motor_solve_ms = std::chrono::duration_cast<milliseconds>(ga_motor_solve_end - ga_motor_solve_start).count();

    std::cout << "\n[TetraPGA Motor Results]" << std::endl;
    std::cout << "Final Cost: " << ga_motor_solver.get_cost() << std::endl;
    std::cout << "Converged:  " << (ga_motor_converged ? "YES" : "NO") << std::endl;
    std::cout << "Stop:       " << ga_motor_solver.get_stop()
              << ", Iter: " << ga_motor_solver.get_iter() << std::endl;
    std::cout << "Total Time: " << ga_motor_total_ms << " ms, Solve Time: " << ga_motor_solve_ms << " ms" << std::endl;

    Eigen::VectorXd ga_motor_x_terminal = ga_motor_solver.get_xs().back();
    Data<double> ga_motor_data(ur_model);
    forwardKinematics(ur_model, ga_motor_data, ga_motor_x_terminal.head(ur_model.dof_a));

    const Motor3D<double> ga_motor_actual = ga_motor_data.M.col(ur_model.n - 1);
    const Motor3D<double> ga_motor_r = ga_motor_actual - ga_M_ref;
    std::cout << "Actual Motor: " << ga_motor_actual.transpose() << std::endl;
    std::cout << "Residual norm (motor diff): " << ga_motor_r.norm() << std::endl;
    
    // =========================================================================
    // Summary
    // =========================================================================
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Complete" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
