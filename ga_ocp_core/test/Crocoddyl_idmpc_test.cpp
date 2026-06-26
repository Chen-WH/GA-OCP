#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include <crocoddyl/core/activations/quadratic-barrier.hpp>
#include <crocoddyl/core/costs/cost-sum.hpp>
#include <crocoddyl/core/costs/residual.hpp>
#include <crocoddyl/core/integrator/euler.hpp>
#include <crocoddyl/core/numdiff/diff-action.hpp>
#include <crocoddyl/core/residuals/control.hpp>
#include <crocoddyl/core/residuals/joint-effort.hpp>
#include <crocoddyl/core/solvers/fddp.hpp>
#include <crocoddyl/core/states/euclidean.hpp>
#include <crocoddyl/multibody/actions/free-invdyn.hpp>
#include <crocoddyl/multibody/actuations/full.hpp>
#include <crocoddyl/multibody/states/multibody.hpp>
#include <crocoddyl/multibody/residuals/state.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include "ga_ocp/CrocoddylActions.hpp"
#include "ga_ocp/CrocoddylResiduals.hpp"
#include "TetraPGA/ModelRepo.hpp"

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::milliseconds milliseconds;

namespace {

Eigen::VectorXd computePinocchioMaxAbsTau(
    const pinocchio::Model& model,
    const std::vector<Eigen::VectorXd>& xs,
    const std::vector<Eigen::VectorXd>& us) {
  pinocchio::Data data(model);
  Eigen::VectorXd max_abs_tau = Eigen::VectorXd::Zero(model.nv);
  for (std::size_t i = 0; i < us.size(); ++i) {
    const Eigen::VectorXd tau =
        pinocchio::rnea(model, data, xs[i].head(model.nq), xs[i].tail(model.nv), us[i]);
    max_abs_tau = max_abs_tau.cwiseMax(tau.cwiseAbs());
  }
  return max_abs_tau;
}

Eigen::VectorXd computeTetraPGAMaxAbsTau(
    const Model<double>& model,
    const std::vector<Eigen::VectorXd>& xs,
    const std::vector<Eigen::VectorXd>& us) {
  Data<double> data(model);
  Eigen::VectorXd max_abs_tau = Eigen::VectorXd::Zero(model.dof_a);
  for (std::size_t i = 0; i < us.size(); ++i) {
    const Eigen::VectorXd tau =
        inverseDynamics(model, data, xs[i].head(model.dof_a), xs[i].tail(model.dof_a), us[i]);
    max_abs_tau = max_abs_tau.cwiseMax(tau.cwiseAbs());
  }
  return max_abs_tau;
}

}  // namespace

int main() {
  constexpr int dof = 6;
  const double dt = 0.008;
  const long horizon = 100;
  const std::size_t max_iterations = 100;

  Model<double> ur_model = ur();

  srand(static_cast<unsigned int>(time(0)));

  Eigen::VectorXd shared_x0 = Eigen::VectorXd::Random(2 * dof);
  shared_x0.tail(dof).setZero();

  Eigen::VectorXd shared_x_target = Eigen::VectorXd::Random(2 * dof);
  shared_x_target.tail(dof).setZero();

  std::cout << "Shared x0: " << shared_x0.transpose() << std::endl;
  std::cout << "Shared x_target: " << shared_x_target.transpose() << std::endl;

  std::cout << "========================================" << std::endl;
  std::cout << "Part A: Pinocchio FreeInvDynamics Test" << std::endl;
  std::cout << "========================================" << std::endl;

  auto pin_start = Clock::now();

  const std::string urdf_path = std::string(GA_OCP_ROBOT_ASSETS_DIR) + "/ur10/urdf/ur10.urdf";

  pinocchio::Model pin_model;
  pinocchio::urdf::buildModel(urdf_path, pin_model);
  std::cout << "[Pinocchio] Loaded model with " << pin_model.nv << " DOFs" << std::endl;

  auto pin_state = std::make_shared<crocoddyl::StateMultibody>(
      std::make_shared<pinocchio::Model>(pin_model));
  auto pin_actuation = std::make_shared<crocoddyl::ActuationModelFull>(pin_state);

  auto pin_running_cost = std::make_shared<crocoddyl::CostModelSum>(pin_state);
  auto pin_terminal_cost = std::make_shared<crocoddyl::CostModelSum>(pin_state);

  auto pin_state_residual =
      std::make_shared<crocoddyl::ResidualModelState>(pin_state, shared_x_target);
  auto pin_state_cost =
      std::make_shared<crocoddyl::CostModelResidual>(pin_state, pin_state_residual);

  auto pin_acc_residual =
      std::make_shared<crocoddyl::ResidualModelControl>(pin_state, pin_model.nv);
  auto pin_acc_cost =
      std::make_shared<crocoddyl::CostModelResidual>(pin_state, pin_acc_residual);

  auto pin_tau_residual = std::make_shared<crocoddyl::ResidualModelJointEffort>(
      pin_state, pin_actuation, Eigen::VectorXd::Zero(pin_model.nv));
  auto pin_tau_cost =
      std::make_shared<crocoddyl::CostModelResidual>(pin_state, pin_tau_residual);

  Eigen::VectorXd pin_zero_state = Eigen::VectorXd::Zero(pin_state->get_nx());
  auto pin_vel_residual =
      std::make_shared<crocoddyl::ResidualModelState>(pin_state, pin_zero_state, pin_model.nv);
  crocoddyl::ActivationBounds pin_vel_bounds;
  pin_vel_bounds.lb = Eigen::VectorXd::Constant(
      pin_state->get_nx(), -std::numeric_limits<double>::infinity());
  pin_vel_bounds.ub = Eigen::VectorXd::Constant(
      pin_state->get_nx(), std::numeric_limits<double>::infinity());
  for (int i = 0; i < pin_model.nv; ++i) {
    pin_vel_bounds.lb[pin_model.nv + i] = -ur_model.velocityLimit[i];
    pin_vel_bounds.ub[pin_model.nv + i] = ur_model.velocityLimit[i];
  }
  auto pin_vel_activation =
      std::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(pin_vel_bounds);
  auto pin_vel_cost = std::make_shared<crocoddyl::CostModelResidual>(
      pin_state, pin_vel_activation, pin_vel_residual);

  pin_running_cost->addCost("state_reg", pin_state_cost, 1.0);
  pin_running_cost->addCost("acc_reg", pin_acc_cost, 1e-2);
  pin_running_cost->addCost("tau_reg", pin_tau_cost, 1e-4);
  pin_running_cost->addCost("vel_limit", pin_vel_cost, 100.0);
  pin_terminal_cost->addCost("state_reg", pin_state_cost, 1000.0);
  pin_terminal_cost->addCost("vel_limit", pin_vel_cost, 100.0);

  auto pin_diff_model =
      std::make_shared<crocoddyl::DifferentialActionModelFreeInvDynamics>(
          pin_state, pin_actuation, pin_running_cost);
  auto pin_diff_model_term =
      std::make_shared<crocoddyl::DifferentialActionModelFreeInvDynamics>(
          pin_state, pin_actuation, pin_terminal_cost);

  std::cout << "\n[Pinocchio Validation] Checking derivatives..." << std::endl;
  auto pin_num_diff =
      std::make_shared<crocoddyl::DifferentialActionModelNumDiff>(pin_diff_model);
  auto pin_num_data = pin_num_diff->createData();
  auto pin_diff_data = pin_diff_model->createData();
  Eigen::VectorXd pin_x_rand = pin_state->rand();
  Eigen::VectorXd pin_u_rand = Eigen::VectorXd::Random(pin_model.nv);
  pin_diff_model->calc(pin_diff_data, pin_x_rand, pin_u_rand);
  pin_diff_model->calcDiff(pin_diff_data, pin_x_rand, pin_u_rand);
  pin_num_diff->calc(pin_num_data, pin_x_rand, pin_u_rand);
  pin_num_diff->calcDiff(pin_num_data, pin_x_rand, pin_u_rand);

  auto pin_running_iam =
      std::make_shared<crocoddyl::IntegratedActionModelEuler>(pin_diff_model, dt);
  auto pin_terminal_iam =
      std::make_shared<crocoddyl::IntegratedActionModelEuler>(pin_diff_model_term, dt);
  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> pin_running_models(
      horizon, pin_running_iam);
  auto pin_problem = std::make_shared<crocoddyl::ShootingProblem>(
      shared_x0, pin_running_models, pin_terminal_iam);

  crocoddyl::SolverFDDP pin_solver(pin_problem);
  pin_solver.set_th_stop(1e-6);

  auto pin_solve_start = Clock::now();
  pin_solver.solve({}, {}, max_iterations);
  auto pin_solve_end = Clock::now();
  auto pin_end = Clock::now();

  const auto pin_total_ms =
      std::chrono::duration_cast<milliseconds>(pin_end - pin_start).count();
  const auto pin_solve_ms =
      std::chrono::duration_cast<milliseconds>(pin_solve_end - pin_solve_start).count();
  const Eigen::VectorXd pin_terminal_error =
      pin_solver.get_xs().back() - shared_x_target;
  const Eigen::VectorXd pin_max_abs_tau =
      computePinocchioMaxAbsTau(pin_model, pin_solver.get_xs(), pin_solver.get_us());

  std::cout << "\n[Pinocchio Results]" << std::endl;
  std::cout << "Final Cost: " << pin_solver.get_cost() << std::endl;
  std::cout << "Converged:  " << (pin_solver.get_stop() < 1e-5 ? "YES" : "NO") << std::endl;
  std::cout << "Total Time: " << pin_total_ms << " ms, Solve Time: " << pin_solve_ms << " ms"
            << std::endl;
  std::cout << "Terminal Error Norm: " << pin_terminal_error.norm() << std::endl;
  std::cout << "Max |tau|: " << pin_max_abs_tau.transpose() << std::endl;

  std::cout << "\n========================================" << std::endl;
  std::cout << "Part B: TetraPGA FreeInvDynamics Test" << std::endl;
  std::cout << "========================================" << std::endl;

  auto ga_start = Clock::now();

  auto ga_state = std::static_pointer_cast<crocoddyl::StateAbstract>(
      std::make_shared<crocoddyl::StateVector>(2 * ur_model.dof_a));
  auto ga_running_cost = std::make_shared<crocoddyl::CostModelSum>(ga_state);
  auto ga_terminal_cost = std::make_shared<crocoddyl::CostModelSum>(ga_state);

  auto ga_state_residual =
      std::make_shared<crocoddyl::ResidualModelState>(ga_state, shared_x_target);
  auto ga_state_cost =
      std::make_shared<crocoddyl::CostModelResidual>(ga_state, ga_state_residual);

  auto ga_acc_residual =
      std::make_shared<crocoddyl::ResidualModelControl>(ga_state, ur_model.dof_a);
  auto ga_acc_cost =
      std::make_shared<crocoddyl::CostModelResidual>(ga_state, ga_acc_residual);

  auto ga_tau_residual = std::make_shared<ResidualModelTetraPGAJointTorque<double>>(
      ga_state, ur_model, Eigen::VectorXd::Zero(ur_model.dof_a));
  auto ga_tau_cost =
      std::make_shared<crocoddyl::CostModelResidual>(ga_state, ga_tau_residual);

  Eigen::VectorXd ga_zero_state = Eigen::VectorXd::Zero(2 * ur_model.dof_a);
  auto ga_vel_residual =
      std::make_shared<crocoddyl::ResidualModelState>(ga_state, ga_zero_state, ur_model.dof_a);
  crocoddyl::ActivationBounds ga_vel_bounds;
  ga_vel_bounds.lb = Eigen::VectorXd::Constant(
      2 * ur_model.dof_a, -std::numeric_limits<double>::infinity());
  ga_vel_bounds.ub = Eigen::VectorXd::Constant(
      2 * ur_model.dof_a, std::numeric_limits<double>::infinity());
  for (int i = 0; i < ur_model.dof_a; ++i) {
    ga_vel_bounds.lb[ur_model.dof_a + i] = -ur_model.velocityLimit[i];
    ga_vel_bounds.ub[ur_model.dof_a + i] = ur_model.velocityLimit[i];
  }
  auto ga_vel_activation =
      std::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(ga_vel_bounds);
  auto ga_vel_cost = std::make_shared<crocoddyl::CostModelResidual>(
      ga_state, ga_vel_activation, ga_vel_residual);

  ga_running_cost->addCost("state_reg", ga_state_cost, 1.0);
  ga_running_cost->addCost("acc_reg", ga_acc_cost, 1e-2);
  ga_running_cost->addCost("tau_reg", ga_tau_cost, 1e-4);
  ga_running_cost->addCost("vel_limit", ga_vel_cost, 100.0);
  ga_terminal_cost->addCost("state_reg", ga_state_cost, 1000.0);
  ga_terminal_cost->addCost("vel_limit", ga_vel_cost, 100.0);

  auto ga_diff_model =
      std::make_shared<DifferentialActionModelTetraPGAInverseDynamics<double>>(ga_state, ur_model, ga_running_cost);
  auto ga_diff_model_term =
      std::make_shared<DifferentialActionModelTetraPGAInverseDynamics<double>>(ga_state, ur_model, ga_terminal_cost);

  std::cout << "\n[TetraPGA Validation] Checking derivatives..." << std::endl;
  auto ga_num_diff =
      std::make_shared<crocoddyl::DifferentialActionModelNumDiff>(ga_diff_model);
  auto ga_num_data = ga_num_diff->createData();
  auto ga_diff_data = ga_diff_model->createData();
  Eigen::VectorXd ga_x_rand = ga_state->rand();
  Eigen::VectorXd ga_u_rand = Eigen::VectorXd::Random(ur_model.dof_a);
  ga_diff_model->calc(ga_diff_data, ga_x_rand, ga_u_rand);
  ga_diff_model->calcDiff(ga_diff_data, ga_x_rand, ga_u_rand);
  ga_num_diff->calc(ga_num_data, ga_x_rand, ga_u_rand);
  ga_num_diff->calcDiff(ga_num_data, ga_x_rand, ga_u_rand);

  auto ga_running_iam =
      std::make_shared<crocoddyl::IntegratedActionModelEuler>(ga_diff_model, dt);
  auto ga_terminal_iam =
      std::make_shared<crocoddyl::IntegratedActionModelEuler>(ga_diff_model_term, dt);
  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> ga_running_models(
      horizon, ga_running_iam);
  auto ga_problem = std::make_shared<crocoddyl::ShootingProblem>(
      shared_x0, ga_running_models, ga_terminal_iam);

  crocoddyl::SolverFDDP ga_solver(ga_problem);
  ga_solver.set_th_stop(1e-6);

  auto ga_solve_start = Clock::now();
  ga_solver.solve({}, {}, max_iterations);
  auto ga_solve_end = Clock::now();
  auto ga_end = Clock::now();

  const auto ga_total_ms =
      std::chrono::duration_cast<milliseconds>(ga_end - ga_start).count();
  const auto ga_solve_ms =
      std::chrono::duration_cast<milliseconds>(ga_solve_end - ga_solve_start).count();
  const Eigen::VectorXd ga_terminal_error =
      ga_solver.get_xs().back() - shared_x_target;
  const Eigen::VectorXd ga_max_abs_tau =
      computeTetraPGAMaxAbsTau(ur_model, ga_solver.get_xs(), ga_solver.get_us());

  std::cout << "\n[TetraPGA Results]" << std::endl;
  std::cout << "Final Cost: " << ga_solver.get_cost() << std::endl;
  std::cout << "Converged:  " << (ga_solver.get_stop() < 1e-5 ? "YES" : "NO") << std::endl;
  std::cout << "Total Time: " << ga_total_ms << " ms, Solve Time: " << ga_solve_ms << " ms"
            << std::endl;
  std::cout << "Terminal Error Norm: " << ga_terminal_error.norm() << std::endl;
  std::cout << "Max |tau|: " << ga_max_abs_tau.transpose() << std::endl;

  std::cout << "\n========================================" << std::endl;
  std::cout << "Performance Summary" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Pinocchio Final Cost: " << pin_solver.get_cost() << std::endl;
  std::cout << "Pinocchio Terminal Error Norm: " << pin_terminal_error.norm() << std::endl;
  std::cout << "Pinocchio Max |tau|: " << pin_max_abs_tau.transpose() << std::endl;
  std::cout << "Pinocchio Total Time: " << pin_total_ms << " ms" << std::endl;
  std::cout << "Pinocchio Solve Time: " << pin_solve_ms << " ms" << std::endl;

  std::cout << "\nTetraPGA Final Cost: " << ga_solver.get_cost() << std::endl;
  std::cout << "TetraPGA Terminal Error Norm: " << ga_terminal_error.norm() << std::endl;
  std::cout << "TetraPGA Max |tau|: " << ga_max_abs_tau.transpose() << std::endl;
  std::cout << "TetraPGA Total Time: " << ga_total_ms << " ms" << std::endl;
  std::cout << "TetraPGA Solve Time: " << ga_solve_ms << " ms" << std::endl;

  return 0;
}
