#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <crocoddyl/core/activations/quadratic-barrier.hpp>
#include <crocoddyl/core/costs/cost-sum.hpp>
#include <crocoddyl/core/costs/residual.hpp>
#include <crocoddyl/core/integrator/euler.hpp>
#include <crocoddyl/core/solvers/box-fddp.hpp>
#include <crocoddyl/core/state-base.hpp>
#include <crocoddyl/core/states/euclidean.hpp>
#include <crocoddyl/multibody/residuals/state.hpp>

#include "ga_ocp/CrocoddylActions.hpp"
#include "ga_ocp/CrocoddylResiduals.hpp"
#include "TetraPGA/ModelRepo.hpp"

namespace {

using Clock = std::chrono::high_resolution_clock;
using milliseconds = std::chrono::milliseconds;

struct SolverArtifacts {
    bool success{false};
    double cost{std::numeric_limits<double>::infinity()};
    double stop{std::numeric_limits<double>::infinity()};
    std::size_t iter{0};
    long solve_ms{0};
    std::vector<Eigen::VectorXd> xs;
    std::vector<Eigen::VectorXd> us;
};

struct TrajectoryCollisionSummary {
    double min_distance{std::numeric_limits<double>::infinity()};
    int violation_count{0};
};

std::vector<SSP<double>> makeObstacles() {
    std::vector<SSP<double>> obstacles;

    SSP<double> obs1;
    obs1.radius = 0.15;
    obs1.center = Point3D<double>(0.3, 0.5, 0.9, 1.0);
    obstacles.push_back(obs1);

    return obstacles;
}

Eigen::VectorXd sampleJointConfiguration(const Model<double>& ur_model,
                                         std::mt19937& rng,
                                         const double amplitude) {
    std::uniform_real_distribution<double> dist(-amplitude, amplitude);
    Eigen::VectorXd q(ur_model.dof_a);
    for (int i = 0; i < ur_model.dof_a; ++i) {
        q[i] = ur_model.qa0[i] + dist(rng);
    }
    return q;
}

template <typename ResidualModelT>
std::shared_ptr<crocoddyl::ShootingProblem> buildProblem(
    const std::shared_ptr<crocoddyl::StateVector>& state,
    const Model<double>& ur_model,
    const Environment<double>& env,
    const Eigen::VectorXd& x0,
    const Motor3D<double>& M_ref,
    const double dt,
    const long horizon,
    const double d_safe) {
    const int dof = ur_model.dof_a;
    const Eigen::VectorXd torque_lb = -ur_model.effortLimit;
    const Eigen::VectorXd torque_ub = ur_model.effortLimit;

    auto running_cost_model = std::make_shared<crocoddyl::CostModelSum>(state);
    auto terminal_cost_model = std::make_shared<crocoddyl::CostModelSum>(state);

    auto placement_residual =
        std::make_shared<ResidualModelTetraPGAFramePlacement<double>>(state, ur_model, M_ref);
    auto placement_cost =
        std::make_shared<crocoddyl::CostModelResidual>(state, placement_residual);

    const Eigen::VectorXd x_zero = Eigen::VectorXd::Zero(2 * dof);
    auto vel_residual =
        std::make_shared<crocoddyl::ResidualModelState>(state, x_zero, dof);
    crocoddyl::ActivationBounds vel_bounds;
    vel_bounds.lb =
        Eigen::VectorXd::Constant(2 * dof, -std::numeric_limits<double>::infinity());
    vel_bounds.ub =
        Eigen::VectorXd::Constant(2 * dof, std::numeric_limits<double>::infinity());
    for (int i = 0; i < dof; ++i) {
        const double vlim = ur_model.velocityLimit[i];
        vel_bounds.lb[dof + i] = -vlim;
        vel_bounds.ub[dof + i] = vlim;
    }
    auto vel_activation =
        std::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(vel_bounds);
    auto vel_limit_cost =
        std::make_shared<crocoddyl::CostModelResidual>(state, vel_activation, vel_residual);

    auto collision_residual =
        std::make_shared<ResidualModelT>(state, ur_model, env, d_safe);
    const int num_collision_pairs = ur_model.num_collision_ssl * env.num_static_sphere;
    crocoddyl::ActivationBounds collision_bounds;
    collision_bounds.lb = Eigen::VectorXd::Zero(num_collision_pairs);
    collision_bounds.ub = Eigen::VectorXd::Constant(
        num_collision_pairs, std::numeric_limits<double>::infinity());
    auto barrier_activation =
        std::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(collision_bounds);
    auto collision_cost = std::make_shared<crocoddyl::CostModelResidual>(
        state, barrier_activation, collision_residual);

    running_cost_model->addCost("placement", placement_cost, 10.0);
    running_cost_model->addCost("vel_limit", vel_limit_cost, 100.0);
    running_cost_model->addCost("collision", collision_cost, 100.0);

    terminal_cost_model->addCost("placement", placement_cost, 100.0);
    terminal_cost_model->addCost("vel_limit", vel_limit_cost, 100.0);
    terminal_cost_model->addCost("collision", collision_cost, 100.0);

    auto diff_model =
        std::make_shared<DifferentialActionModelTetraPGAForwardDynamics<double>>(state, ur_model, running_cost_model);
    auto diff_model_term = std::make_shared<DifferentialActionModelTetraPGAForwardDynamics<double>>(
        state, ur_model, terminal_cost_model);
    diff_model->set_u_lb(torque_lb);
    diff_model->set_u_ub(torque_ub);
    diff_model_term->set_u_lb(torque_lb);
    diff_model_term->set_u_ub(torque_ub);

    auto running_iam = std::make_shared<crocoddyl::IntegratedActionModelEuler>(diff_model, dt);
    auto terminal_iam =
        std::make_shared<crocoddyl::IntegratedActionModelEuler>(diff_model_term, dt);
    std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> running_models(
        static_cast<std::size_t>(horizon), running_iam);
    return std::make_shared<crocoddyl::ShootingProblem>(x0, running_models, terminal_iam);
}

SolverArtifacts solveProblem(const std::shared_ptr<crocoddyl::ShootingProblem>& problem,
                             const std::vector<Eigen::VectorXd>& init_xs,
                             const std::vector<Eigen::VectorXd>& init_us,
                             const std::size_t max_iter,
                             const double th_stop) {
    crocoddyl::SolverBoxFDDP solver(problem);
    solver.set_th_stop(th_stop);

    const auto start = Clock::now();
    const bool success = solver.solve(init_xs, init_us, max_iter);
    const auto end = Clock::now();

    SolverArtifacts result;
    result.success = success;
    result.cost = solver.get_cost();
    result.stop = solver.get_stop();
    result.iter = solver.get_iter();
    result.solve_ms = std::chrono::duration_cast<milliseconds>(end - start).count();
    result.xs = solver.get_xs();
    result.us = solver.get_us();
    return result;
}

double maxTrajectoryDifference(const std::vector<Eigen::VectorXd>& lhs,
                               const std::vector<Eigen::VectorXd>& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::runtime_error("Trajectory size mismatch during comparison.");
    }

    double max_err = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        max_err = std::max(max_err, (lhs[i] - rhs[i]).cwiseAbs().maxCoeff());
    }
    return max_err;
}

TrajectoryCollisionSummary summarizeConfigurationCollision(const Model<double>& ur_model,
                                                           const Environment<double>& env,
                                                           const Eigen::VectorXd& q) {
    Data<double> data(ur_model);
    EnvironmentData<double> env_data(ur_model, env);
    forwardKinematics(ur_model, data, q);
    computeDistance(ur_model, data, env, env_data);

    TrajectoryCollisionSummary summary;
    for (int i = 0; i < env_data.num_collision_pair; ++i) {
        summary.min_distance = std::min(summary.min_distance, env_data.distance[i]);
        if (env_data.distance[i] < 0.0) {
            ++summary.violation_count;
        }
    }
    return summary;
}

TrajectoryCollisionSummary summarizeTrajectoryCollision(const Model<double>& ur_model,
                                                        const Environment<double>& env,
                                                        const std::vector<Eigen::VectorXd>& xs) {
    TrajectoryCollisionSummary summary;
    for (const auto& x : xs) {
        const auto step_summary =
            summarizeConfigurationCollision(ur_model, env, x.head(ur_model.dof_a));
        summary.min_distance = std::min(summary.min_distance, step_summary.min_distance);
        summary.violation_count += step_summary.violation_count;
    }
    return summary;
}

}  // namespace

int main() {
    const double dt = 0.008;
    const double d_safe = 0.1;
    const long horizon = 50;
    const std::size_t max_iter = 50;
    constexpr std::uint32_t kRandomSeed = 20260320;

    const auto global_start = Clock::now();

    Model<double> ur_model = ur();
    auto state = std::make_shared<crocoddyl::StateVector>(2 * ur_model.dof_a);
    const Environment<double> env(makeObstacles());
    const int dof = ur_model.dof_a;
    std::mt19937 rng(kRandomSeed);

    const Eigen::VectorXd q0 = sampleJointConfiguration(ur_model, rng, 0.5);
    const Eigen::VectorXd q_ref = sampleJointConfiguration(ur_model, rng, 0.5);

    Eigen::VectorXd x0(2 * dof);
    x0.head(dof) = q0;
    x0.tail(dof).setZero();

    Data<double> ref_data(ur_model);
    forwardKinematics(ur_model, ref_data, q_ref);
    const Motor3D<double> M_ref = ref_data.M.col(ur_model.n - 1);

    std::vector<Eigen::VectorXd> init_xs(static_cast<std::size_t>(horizon) + 1, x0);
    std::vector<Eigen::VectorXd> init_us(static_cast<std::size_t>(horizon),
                                         Eigen::VectorXd::Zero(dof));

    std::cout << "========================================" << std::endl;
    std::cout << "Crocoddyl Collision Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "[Setup] dof=" << dof
              << ", capsules=" << ur_model.num_collision_ssl
              << ", obstacles=" << env.num_static_sphere
              << ", horizon=" << horizon
              << ", d_safe=" << d_safe
              << ", seed=" << kRandomSeed << std::endl;
    for (int i = 0; i < env.num_static_sphere; ++i) {
        std::cout << "  Obstacle " << i << ": radius=" << env.static_sphere[i].radius
                  << ", center=(" << env.static_sphere[i].center(0) << ", "
                  << env.static_sphere[i].center(1) << ", "
                  << env.static_sphere[i].center(2) << ")" << std::endl;
    }
    std::cout << "[Initial State] q0: " << q0.transpose() << std::endl;
    std::cout << "[Target State] q_ref: " << q_ref.transpose() << std::endl;

    const auto base_problem = buildProblem<ResidualModelTetraPGACollisionDistance<double>>(
        state, ur_model, env, x0, M_ref, dt, horizon, d_safe);
    const auto cache_problem = buildProblem<ResidualModelTetraPGACachedCollisionDistance<double>>(
        state, ur_model, env, x0, M_ref, dt, horizon, d_safe);

    const auto base_result =
        solveProblem(base_problem, init_xs, init_us, max_iter, 1e-6);
    const auto cache_result =
        solveProblem(cache_problem, init_xs, init_us, max_iter, 1e-8);

    const auto global_end = Clock::now();
    const auto total_ms =
        std::chrono::duration_cast<milliseconds>(global_end - global_start).count();

    std::cout << std::fixed << std::setprecision(12);
    std::cout << "\n[Baseline Solve]" << std::endl;
    std::cout << "  success=" << base_result.success
              << ", cost=" << base_result.cost
              << ", stop=" << base_result.stop
              << ", iter=" << base_result.iter
              << ", time_ms=" << base_result.solve_ms << std::endl;

    const Eigen::VectorXd x_terminal = base_result.xs.back();
    Data<double> final_data(ur_model);
    forwardKinematics(ur_model, final_data, x_terminal.head(dof));
    const Motor3D<double> M_actual = final_data.M.col(ur_model.n - 1);
    const Line3D<double> placement_error = ga_log(ga_mul(ga_rev(M_ref), M_actual));

    std::cout << "\n[End-Effector Verification]" << std::endl;
    std::cout << "  Final joint angles: " << x_terminal.head(dof).transpose() << std::endl;
    std::cout << "  Target Motor: " << M_ref.transpose() << std::endl;
    std::cout << "  Actual Motor: " << M_actual.transpose() << std::endl;
    std::cout << "  Placement error norm: " << placement_error.norm() << std::endl;

    const auto initial_collision = summarizeConfigurationCollision(ur_model, env, q0);
    const auto final_collision =
        summarizeConfigurationCollision(ur_model, env, x_terminal.head(dof));
    const auto trajectory_collision =
        summarizeTrajectoryCollision(ur_model, env, base_result.xs);

    std::cout << "\n[Collision Check]" << std::endl;
    std::cout << "  Initial min distance: " << initial_collision.min_distance << " m" << std::endl;
    std::cout << "  Final min distance: " << final_collision.min_distance << " m" << std::endl;
    std::cout << "  Final violations: " << final_collision.violation_count << std::endl;
    std::cout << "  Trajectory min distance: " << trajectory_collision.min_distance << " m"
              << std::endl;
    std::cout << "  Trajectory violations: " << trajectory_collision.violation_count
              << std::endl;
    std::cout << "  Trajectory is "
              << (trajectory_collision.violation_count == 0 ? "COLLISION-FREE"
                                                            : "COLLISION-PRONE")
              << std::endl;

    std::cout << "\n[Cache Comparison]" << std::endl;
    std::cout << "  cache success=" << cache_result.success
              << ", cost=" << cache_result.cost
              << ", stop=" << cache_result.stop
              << ", iter=" << cache_result.iter
              << ", time_ms=" << cache_result.solve_ms << std::endl;

    if (base_result.success != cache_result.success) {
        throw std::runtime_error("Solver success flags differ between base and cache residuals.");
    }

    const double cost_err = std::abs(base_result.cost - cache_result.cost);
    const double stop_err = std::abs(base_result.stop - cache_result.stop);
    const double max_x_err = maxTrajectoryDifference(base_result.xs, cache_result.xs);
    const double max_u_err = maxTrajectoryDifference(base_result.us, cache_result.us);

    std::cout << "  |cost_base-cost_cache| = " << cost_err << std::endl;
    std::cout << "  |stop_base-stop_cache| = " << stop_err << std::endl;
    std::cout << "  max|x_base-x_cache|   = " << max_x_err << std::endl;
    std::cout << "  max|u_base-u_cache|   = " << max_u_err << std::endl;

    if (cost_err > 1e-8 || stop_err > 1e-8 || max_x_err > 1e-7 || max_u_err > 1e-7) {
        throw std::runtime_error("Cache residual changes the solver trajectory beyond tolerance.");
    }

    std::cout << "\n[Summary]" << std::endl;
    std::cout << "  Total wall time: " << total_ms << " ms" << std::endl;
    std::cout << "  Baseline solve and cache consistency checks both passed." << std::endl;
    return 0;
}
