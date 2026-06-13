#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <random>
#include <cmath>
#include <limits>

// Crocoddyl Includes
#include <crocoddyl/core/state-base.hpp>
#include <crocoddyl/core/states/euclidean.hpp> // 使用 StateVector
#include <crocoddyl/core/solvers/fddp.hpp>
#include <crocoddyl/core/integrator/euler.hpp>
#include <crocoddyl/core/costs/cost-sum.hpp>
#include <crocoddyl/core/costs/residual.hpp>
#include <crocoddyl/core/activations/quadratic-barrier.hpp>
#include <crocoddyl/multibody/residuals/state.hpp>
#include <crocoddyl/core/utils/callbacks.hpp>
#include <crocoddyl/core/numdiff/diff-action.hpp> // 用于验证导数

// Pinocchio Includes
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <crocoddyl/multibody/states/multibody.hpp>
#include <crocoddyl/multibody/actuations/full.hpp>
#include <crocoddyl/multibody/actions/free-fwddyn.hpp>

// Your Custom Library Includes
#include "TetraPGA/ModelRepo.hpp"
#include "ga_ocp/CrocoddylActions.hpp"
#include "ga_ocp/CrocoddylResiduals.hpp"

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::milliseconds milliseconds;

int main() {
    double dt = 0.008; // 时间步长
    Model<double> ur_model = ur();
    
    // Set random seed using current time for non-deterministic behavior
    srand((unsigned int) time(0));
    
    // =========================================================================
    // Generate Random Initial State and Target State (for both methods)
    // =========================================================================
    const int dof = 6;
    Eigen::VectorXd shared_x0 = Eigen::VectorXd::Random(2 * dof);
    shared_x0.tail(dof).setZero(); // Set velocities to zero for initial state
    Eigen::VectorXd shared_x_target = Eigen::VectorXd::Random(2 * dof);
    shared_x_target.tail(dof).setZero(); // Set velocities to zero for target
    
    std::cout << "Shared x0: " << shared_x0.transpose() << std::endl;
    std::cout << "Shared x_target: " << shared_x_target.transpose() << std::endl;
    
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
    
    // 使用共享的目标状态
    Eigen::VectorXd pin_x_target = shared_x_target;
    
    auto pin_state_residual = std::make_shared<crocoddyl::ResidualModelState>(pin_state, pin_x_target);
    auto pin_state_cost = std::make_shared<crocoddyl::CostModelResidual>(pin_state, pin_state_residual);
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
    
    pin_running_cost->addCost("state_reg", pin_state_cost, 1.0);
    pin_running_cost->addCost("vel_limit", pin_vel_cost, 100.0);
    pin_terminal_cost->addCost("state_reg", pin_state_cost, 1000.0);
    pin_terminal_cost->addCost("vel_limit", pin_vel_cost, 100.0);
    
    // 4. 创建 Actuation 和 Action Model
    auto pin_actuation = std::make_shared<crocoddyl::ActuationModelFull>(pin_state);
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
    crocoddyl::SolverFDDP pin_solver(pin_problem);
    pin_solver.set_th_stop(1e-6);
    
    std::vector<std::shared_ptr<crocoddyl::CallbackAbstract>> pin_callbacks;
    auto pin_solve_start = Clock::now();
    pin_solver.solve({}, {}, 100);
    auto pin_solve_end = Clock::now();
    
    auto pin_end = Clock::now();
    
    auto pin_total_ms = std::chrono::duration_cast<milliseconds>(pin_end - pin_start).count();
    auto pin_solve_ms = std::chrono::duration_cast<milliseconds>(pin_solve_end - pin_solve_start).count();

        Eigen::VectorXd pin_max_abs_vel = Eigen::VectorXd::Zero(pin_model.nv);
        const auto& pin_xs = pin_solver.get_xs();
        for (const auto& xk : pin_xs) {
            for (int j = 0; j < pin_model.nv; ++j) {
                pin_max_abs_vel[j] = std::max(pin_max_abs_vel[j], std::abs(xk[pin_model.nv + j]));
            }
        }
    
    std::cout << "\n[Pinocchio Results]" << std::endl;
    std::cout << "Final Cost: " << pin_solver.get_cost() << std::endl;
    std::cout << "Converged:  " << (pin_solver.get_stop() < 1e-5 ? "YES" : "NO") << std::endl;
    std::cout << "Total Time: " << pin_total_ms << " ms, Solve Time: " << pin_solve_ms << " ms" << std::endl;
    std::cout << "Max |joint velocity| over trajectory: " << pin_max_abs_vel.transpose() << std::endl;
    
    // =========================================================================
    // Part B: TetraPGA-based 测试 (TetraPGA Test)
    // =========================================================================
    std::cout << "\n========================================" << std::endl;
    std::cout << "Part B: TetraPGA Test" << std::endl;
    std::cout << "========================================" << std::endl;
    
    auto ga_start = Clock::now();
    
    // -------------------------------------------------------------------------
    // 1. 设置模型 (Setup Model)
    // -------------------------------------------------------------------------
    auto state = std::make_shared<crocoddyl::StateVector>(2 * ur_model.dof_a);

    // -------------------------------------------------------------------------
    // 2. 定义 Cost 函数 (Define Costs)
    // -------------------------------------------------------------------------
    auto running_cost_model = std::make_shared<crocoddyl::CostModelSum>(state);
    auto terminal_cost_model = std::make_shared<crocoddyl::CostModelSum>(state);

    // 使用共享的目标状态
    Eigen::VectorXd x_target = shared_x_target;

    auto state_residual = std::make_shared<crocoddyl::ResidualModelState>(state, x_target);
    auto state_cost = std::make_shared<crocoddyl::CostModelResidual>(state, state_residual);
    
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

    running_cost_model->addCost("state_reg", state_cost, 1.0);
    running_cost_model->addCost("vel_limit", vel_limit_cost, 100.0);
    terminal_cost_model->addCost("state_reg", state_cost, 1000.0);
    terminal_cost_model->addCost("vel_limit", vel_limit_cost, 100.0);

    // -------------------------------------------------------------------------
    // 3. 构建 Action Model (Build Action Model)
    // -------------------------------------------------------------------------
    auto diff_model = std::make_shared<DifferentialActionModelTetraPGAForwardDynamics<double>>(
        state, ur_model, running_cost_model
    );
    
    auto diff_model_term = std::make_shared<DifferentialActionModelTetraPGAForwardDynamics<double>>(
        state, ur_model, terminal_cost_model
    );

    // -------------------------------------------------------------------------
    // 4. 构建最优控制问题并求解 (Solve OCP)
    // -------------------------------------------------------------------------
    auto running_IAM = std::make_shared<crocoddyl::IntegratedActionModelEuler>(diff_model, dt);
    auto terminal_IAM = std::make_shared<crocoddyl::IntegratedActionModelEuler>(diff_model_term, dt);

    long T = 100;
    std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> running_models(T, running_IAM);

    // 使用共享的初始状态
    Eigen::VectorXd x0 = shared_x0;

    auto problem = std::make_shared<crocoddyl::ShootingProblem>(x0, running_models, terminal_IAM);

    crocoddyl::SolverFDDP solver(problem);
    solver.set_th_stop(1e-6);
    
    auto ga_solve_start = Clock::now();
    solver.solve({}, {}, 100);
    auto ga_solve_end = Clock::now();
    
    auto ga_end = Clock::now();
    
    auto ga_total_ms = std::chrono::duration_cast<milliseconds>(ga_end - ga_start).count();
    auto ga_solve_ms = std::chrono::duration_cast<milliseconds>(ga_solve_end - ga_solve_start).count();

        Eigen::VectorXd ga_max_abs_vel = Eigen::VectorXd::Zero(ur_model.dof_a);
        const auto& ga_xs = solver.get_xs();
        for (const auto& xk : ga_xs) {
            for (int j = 0; j < ur_model.dof_a; ++j) {
                ga_max_abs_vel[j] = std::max(ga_max_abs_vel[j], std::abs(xk[ur_model.dof_a + j]));
            }
        }

    std::cout << "\n[TetraPGA Results]" << std::endl;
    std::cout << "Final Cost: " << solver.get_cost() << std::endl;
    std::cout << "Converged:  " << (solver.get_stop() < 1e-5 ? "YES" : "NO") << std::endl;
    std::cout << "Total Time: " << ga_total_ms << " ms, Solve Time: " << ga_solve_ms << " ms" << std::endl;
    std::cout << "Max |joint velocity| over trajectory: " << ga_max_abs_vel.transpose() << std::endl;
    
    // =========================================================================
    // Summary: Performance Comparison
    // =========================================================================
    std::cout << "\n========================================" << std::endl;
    std::cout << "Performance Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Pinocchio Final Cost: " << pin_solver.get_cost() << std::endl;
    std::cout << "Pinocchio Converged: " << (pin_solver.get_stop() < 1e-5 ? "YES" : "NO") << std::endl;
    std::cout << "Pinocchio Total Time: " << pin_total_ms << " ms" << std::endl;
    std::cout << "Pinocchio Solve Time: " << pin_solve_ms << " ms" << std::endl;
    std::cout << "Pinocchio Max |joint velocity|: " << pin_max_abs_vel.transpose() << std::endl;
    
    std::cout << "\nTetraPGA Final Cost: " << solver.get_cost() << std::endl;
    std::cout << "TetraPGA Converged: " << (solver.get_stop() < 1e-5 ? "YES" : "NO") << std::endl;
    std::cout << "TetraPGA Total Time: " << ga_total_ms << " ms" << std::endl;
    std::cout << "TetraPGA Solve Time: " << ga_solve_ms << " ms" << std::endl;
    std::cout << "TetraPGA Max |joint velocity|: " << ga_max_abs_vel.transpose() << std::endl;

    return 0;
}
