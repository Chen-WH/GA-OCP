#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <random>

// Crocoddyl Includes
#include <crocoddyl/core/state-base.hpp>
#include <crocoddyl/core/states/euclidean.hpp> // 使用 StateVector
#include <crocoddyl/core/solvers/box-fddp.hpp>
#include <crocoddyl/core/integrator/euler.hpp>
#include <crocoddyl/core/costs/cost-sum.hpp>
#include <crocoddyl/core/costs/residual.hpp>
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
    const Eigen::VectorXd torque_lb = -ur_model.effortLimit;
    const Eigen::VectorXd torque_ub = ur_model.effortLimit;
    
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
    std::cout << "Part A: Pinocchio BoxFDDP Test" << std::endl;
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
    
    Eigen::VectorXd pin_x_target = shared_x_target;
    auto pin_state_residual = std::make_shared<crocoddyl::ResidualModelState>(pin_state, pin_x_target);
    auto pin_state_cost = std::make_shared<crocoddyl::CostModelResidual>(pin_state, pin_state_residual);
    pin_running_cost->addCost("state_reg", pin_state_cost, 1.0);
    pin_terminal_cost->addCost("state_reg", pin_state_cost, 1000.0);
    
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
    Eigen::VectorXd pin_x0 = shared_x0;
    
    auto pin_problem = std::make_shared<crocoddyl::ShootingProblem>(pin_x0, pin_running_models, pin_terminal_IAM);
    crocoddyl::SolverBoxFDDP pin_solver(pin_problem);
    pin_solver.set_th_stop(1e-6);
    
    auto pin_solve_start = Clock::now();
    pin_solver.solve({}, {}, 100);
    auto pin_solve_end = Clock::now();
    
    auto pin_end = Clock::now();
    
    auto pin_total_ms = std::chrono::duration_cast<milliseconds>(pin_end - pin_start).count();
    auto pin_solve_ms = std::chrono::duration_cast<milliseconds>(pin_solve_end - pin_solve_start).count();
    
    std::cout << "\n[Pinocchio Results]" << std::endl;
    std::cout << "Final Cost: " << pin_solver.get_cost() << std::endl;
    std::cout << "Converged:  " << (pin_solver.get_stop() < 1e-5 ? "YES" : "NO") << std::endl;
    std::cout << "Total Time: " << pin_total_ms << " ms, Solve Time: " << pin_solve_ms << " ms" << std::endl;
    
    // =========================================================================
    // Part B: TetraPGA-based 测试 (TetraPGA Test)
    // =========================================================================
    std::cout << "\n========================================" << std::endl;
    std::cout << "Part B: TetraPGA BoxFDDP Test" << std::endl;
    std::cout << "========================================" << std::endl;
    
    auto ga_start = Clock::now();
    auto state = std::make_shared<crocoddyl::StateVector>(2 * ur_model.dof_a);

    auto running_cost_model = std::make_shared<crocoddyl::CostModelSum>(state);
    auto terminal_cost_model = std::make_shared<crocoddyl::CostModelSum>(state);

    Eigen::VectorXd x_target = shared_x_target;
    auto state_residual = std::make_shared<crocoddyl::ResidualModelState>(state, x_target);
    auto state_cost = std::make_shared<crocoddyl::CostModelResidual>(state, state_residual);
    
    running_cost_model->addCost("state_reg", state_cost, 1.0);
    terminal_cost_model->addCost("state_reg", state_cost, 1000.0);

    auto diff_model = std::make_shared<DifferentialActionModelTetraPGAForwardDynamics<double>>(
        state, ur_model, running_cost_model
    );
    auto diff_model_term = std::make_shared<DifferentialActionModelTetraPGAForwardDynamics<double>>(
        state, ur_model, terminal_cost_model
    );
    diff_model->set_u_lb(torque_lb);
    diff_model->set_u_ub(torque_ub);
    diff_model_term->set_u_lb(torque_lb);
    diff_model_term->set_u_ub(torque_ub);

    auto running_IAM = std::make_shared<crocoddyl::IntegratedActionModelEuler>(diff_model, dt);
    auto terminal_IAM = std::make_shared<crocoddyl::IntegratedActionModelEuler>(diff_model_term, dt);

    long T = 100;
    std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> running_models(T, running_IAM);
    Eigen::VectorXd x0 = shared_x0;

    auto problem = std::make_shared<crocoddyl::ShootingProblem>(x0, running_models, terminal_IAM);

    crocoddyl::SolverBoxFDDP solver(problem);
    solver.set_th_stop(1e-6);
    
    auto ga_solve_start = Clock::now();
    solver.solve({}, {}, 100);
    auto ga_solve_end = Clock::now();
    
    auto ga_end = Clock::now();
    
    auto ga_total_ms = std::chrono::duration_cast<milliseconds>(ga_end - ga_start).count();
    auto ga_solve_ms = std::chrono::duration_cast<milliseconds>(ga_solve_end - ga_solve_start).count();

    std::cout << "\n[TetraPGA Results]" << std::endl;
    std::cout << "Final Cost: " << solver.get_cost() << std::endl;
    std::cout << "Converged:  " << (solver.get_stop() < 1e-5 ? "YES" : "NO") << std::endl;
    std::cout << "Total Time: " << ga_total_ms << " ms, Solve Time: " << ga_solve_ms << " ms" << std::endl;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Performance Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Pinocchio Final Cost: " << pin_solver.get_cost() << std::endl;
    std::cout << "Pinocchio Converged: " << (pin_solver.get_stop() < 1e-5 ? "YES" : "NO") << std::endl;
    std::cout << "Pinocchio Total Time: " << pin_total_ms << " ms" << std::endl;
    std::cout << "Pinocchio Solve Time: " << pin_solve_ms << " ms" << std::endl;
    
    std::cout << "\nTetraPGA Final Cost: " << solver.get_cost() << std::endl;
    std::cout << "TetraPGA Converged: " << (solver.get_stop() < 1e-5 ? "YES" : "NO") << std::endl;
    std::cout << "TetraPGA Total Time: " << ga_total_ms << " ms" << std::endl;
    std::cout << "TetraPGA Solve Time: " << ga_solve_ms << " ms" << std::endl;

    return 0;
}
