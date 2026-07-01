#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <crocoddyl/core/activations/quadratic-barrier.hpp>
#include <crocoddyl/core/costs/cost-sum.hpp>
#include <crocoddyl/core/costs/residual.hpp>
#include <crocoddyl/core/integrator/euler.hpp>
#include <crocoddyl/core/residuals/control.hpp>
#include <crocoddyl/core/solvers/box-fddp.hpp>
#include <crocoddyl/core/states/euclidean.hpp>
#include <crocoddyl/multibody/residuals/state.hpp>

#include "TetraPGA/Collision.hpp"
#include "TetraPGA/Kinematics.hpp"
#include "TetraPGA/ModelRepo.hpp"
#include "ga_ocp/CrocoddylActions.hpp"
#include "ga_ocp/CrocoddylResiduals.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using DurationSeconds = std::chrono::duration<double>;

struct CliConfig {
  std::vector<double> margins{0.03, 0.05, 0.08, 0.10, 0.15};
  std::vector<int> obstacle_counts{4, 8, 16};
  int samples = 5;
  std::uint32_t seed = 20260624u;
  int horizon = 50;
  int max_iterations = 50;
  double dt = 0.008;
  double position_amplitude = 0.45;
  double th_stop = 1e-6;
  double placement_weight = 10.0;
  double terminal_placement_weight = 100.0;
  double velocity_limit_weight = 100.0;
  double collision_weight = 100.0;
  double control_weight = 1e-4;
  double obstacle_radius_min = 0.04;
  double obstacle_radius_max = 0.10;
  double endpoint_clearance = 0.06;
  int obstacle_max_attempts = 2000;
  std::string output_csv;
};

struct SolverArtifacts {
  bool success = false;
  bool failed = false;
  std::string failure_message;
  double cost = std::numeric_limits<double>::quiet_NaN();
  double stop = std::numeric_limits<double>::quiet_NaN();
  std::size_t iter = 0;
  double solve_ms = 0.0;
  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
};

struct CollisionSummary {
  double min_distance = std::numeric_limits<double>::infinity();
  int safety_violation_count = 0;
  int collision_violation_count = 0;
};

struct TrajectoryMetrics {
  double path_length = 0.0;
  double jerk_rms = std::numeric_limits<double>::quiet_NaN();
  double torque_rate_rms = std::numeric_limits<double>::quiet_NaN();
};

std::string CsvEscape(std::string_view value) {
  std::string out;
  out.reserve(value.size() + 2);
  out.push_back('"');
  for (const char c : value) {
    if (c == '"') {
      out.push_back('"');
    }
    out.push_back(c);
  }
  out.push_back('"');
  return out;
}

std::string FormatCsvNumber(const double value) {
  if (!std::isfinite(value)) {
    return "nan";
  }
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(9) << value;
  std::string out = oss.str();
  while (!out.empty() && out.back() == '0') {
    out.pop_back();
  }
  if (!out.empty() && out.back() == '.') {
    out.pop_back();
  }
  return out.empty() ? "0" : out;
}

std::vector<double> ParseDoubleList(const std::string& raw) {
  std::vector<double> out;
  std::stringstream ss(raw);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      out.push_back(std::stod(item));
    }
  }
  if (out.empty()) {
    throw std::invalid_argument("empty double list");
  }
  return out;
}

std::vector<int> ParseIntList(const std::string& raw) {
  std::vector<int> out;
  std::stringstream ss(raw);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      out.push_back(std::stoi(item));
    }
  }
  if (out.empty()) {
    throw std::invalid_argument("empty integer list");
  }
  return out;
}

bool ConsumeOption(const std::string& arg, const std::string& name, std::string* value) {
  const std::string prefix = "--" + name + "=";
  if (arg.rfind(prefix, 0) != 0) {
    return false;
  }
  *value = arg.substr(prefix.size());
  return true;
}

CliConfig ParseCli(int argc, char** argv) {
  CliConfig config;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i] == nullptr ? "" : argv[i];
    if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: Crocoddyl_obstacle_margin_sweep [options]\n"
          << "  --output_csv=<path>\n"
          << "  --margins=0.03,0.05,0.08,0.10,0.15\n"
          << "  --obstacle_counts=4,8,16\n"
          << "  --samples=<int>\n"
          << "  --seed=<uint>\n"
          << "  --horizon=<int>\n"
          << "  --max_iterations=<int>\n"
          << "  --dt=<double>\n"
          << "  --position_amplitude=<double>\n"
          << "  --collision_weight=<double>\n"
          << "  --obstacle_radius_min=<double>\n"
          << "  --obstacle_radius_max=<double>\n"
          << "  --endpoint_clearance=<double>\n"
          << "  --obstacle_max_attempts=<int>\n";
      std::exit(0);
    }

    std::string value;
    if (ConsumeOption(arg, "output_csv", &value)) {
      config.output_csv = value;
    } else if (ConsumeOption(arg, "margins", &value)) {
      config.margins = ParseDoubleList(value);
    } else if (ConsumeOption(arg, "obstacle_counts", &value)) {
      config.obstacle_counts = ParseIntList(value);
    } else if (ConsumeOption(arg, "samples", &value)) {
      config.samples = std::stoi(value);
    } else if (ConsumeOption(arg, "seed", &value)) {
      config.seed = static_cast<std::uint32_t>(std::stoul(value));
    } else if (ConsumeOption(arg, "horizon", &value)) {
      config.horizon = std::stoi(value);
    } else if (ConsumeOption(arg, "max_iterations", &value)) {
      config.max_iterations = std::stoi(value);
    } else if (ConsumeOption(arg, "dt", &value)) {
      config.dt = std::stod(value);
    } else if (ConsumeOption(arg, "position_amplitude", &value)) {
      config.position_amplitude = std::stod(value);
    } else if (ConsumeOption(arg, "collision_weight", &value)) {
      config.collision_weight = std::stod(value);
    } else if (ConsumeOption(arg, "obstacle_radius_min", &value)) {
      config.obstacle_radius_min = std::stod(value);
    } else if (ConsumeOption(arg, "obstacle_radius_max", &value)) {
      config.obstacle_radius_max = std::stod(value);
    } else if (ConsumeOption(arg, "endpoint_clearance", &value)) {
      config.endpoint_clearance = std::stod(value);
    } else if (ConsumeOption(arg, "obstacle_max_attempts", &value)) {
      config.obstacle_max_attempts = std::stoi(value);
    } else {
      throw std::invalid_argument("unknown option: " + arg);
    }
  }
  if (config.output_csv.empty()) {
    config.output_csv = "Crocoddyl_obstacle_margin_sweep.csv";
  }
  if (config.obstacle_radius_min <= 0.0 ||
      config.obstacle_radius_max < config.obstacle_radius_min) {
    throw std::invalid_argument("invalid obstacle radius range");
  }
  if (config.endpoint_clearance < 0.0) {
    throw std::invalid_argument("endpoint_clearance must be non-negative");
  }
  if (config.obstacle_max_attempts <= 0) {
    throw std::invalid_argument("obstacle_max_attempts must be positive");
  }
  return config;
}

std::uint32_t MixSeed(std::uint32_t seed, std::uint32_t value) {
  return seed ^ (value + 0x9e3779b9u + (seed << 6) + (seed >> 2));
}

Eigen::VectorXd SampleJointConfiguration(const Model<double>& model, std::mt19937& rng,
                                         const double amplitude) {
  std::uniform_real_distribution<double> dist(-amplitude, amplitude);
  Eigen::VectorXd q(model.dof_a);
  for (int i = 0; i < model.dof_a; ++i) {
    q[i] = model.qa0[i] + dist(rng);
    const double lower = model.lowerPositionLimit[i];
    const double upper = model.upperPositionLimit[i];
    if (std::isfinite(lower)) {
      q[i] = std::max(q[i], lower);
    }
    if (std::isfinite(upper)) {
      q[i] = std::min(q[i], upper);
    }
  }
  return q;
}

CollisionSummary SummarizeConfigurationCollision(const Model<double>& model,
                                                 const Environment<double>& env,
                                                 const Eigen::VectorXd& q,
                                                 double d_safe);

Environment<double> MakeObstacleEnvironment(const Model<double>& model,
                                            const Eigen::VectorXd& q0,
                                            const Eigen::VectorXd& q_ref,
                                            const int obstacle_count,
                                            const std::uint32_t seed,
                                            const CliConfig& config) {
  std::vector<SSP<double>> obstacles;
  obstacles.reserve(static_cast<std::size_t>(obstacle_count));

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> xdist(-0.55, 0.35);
  std::uniform_real_distribution<double> ydist(-0.55, 0.55);
  std::uniform_real_distribution<double> zdist(0.35, 1.15);
  std::uniform_real_distribution<double> rdist(config.obstacle_radius_min,
                                               config.obstacle_radius_max);

  for (int i = 0; i < obstacle_count; ++i) {
    bool accepted = false;
    for (int attempt = 0; attempt < config.obstacle_max_attempts; ++attempt) {
      SSP<double> obs;
      obs.id = i;
      obs.radius = rdist(rng);
      obs.center = Point3D<double>(xdist(rng), ydist(rng), zdist(rng), 1.0);

      std::vector<SSP<double>> trial_obstacles = obstacles;
      trial_obstacles.push_back(obs);
      Environment<double> trial_env(trial_obstacles);
      const CollisionSummary initial_collision =
          SummarizeConfigurationCollision(model, trial_env, q0, 0.0);
      const CollisionSummary target_collision =
          SummarizeConfigurationCollision(model, trial_env, q_ref, 0.0);
      if (initial_collision.min_distance >= config.endpoint_clearance &&
          target_collision.min_distance >= config.endpoint_clearance) {
        obstacles.push_back(obs);
        accepted = true;
        break;
      }
    }
    if (!accepted) {
      throw std::runtime_error("failed to sample obstacle environment with endpoint clearance");
    }
  }
  return Environment<double>(obstacles);
}

std::shared_ptr<crocoddyl::ShootingProblem> BuildProblem(
    const std::shared_ptr<crocoddyl::StateVector>& state, const Model<double>& model,
    const Environment<double>& env, const Eigen::VectorXd& x0, const Motor3D<double>& target,
    const CliConfig& config, const double d_safe) {
  const int dof = model.dof_a;
  auto running_cost = std::make_shared<crocoddyl::CostModelSum>(state);
  auto terminal_cost = std::make_shared<crocoddyl::CostModelSum>(state);

  auto placement_residual =
      std::make_shared<ResidualModelTetraPGAFramePlacement<double>>(state, model, target);
  auto placement_cost = std::make_shared<crocoddyl::CostModelResidual>(state, placement_residual);

  const Eigen::VectorXd x_zero = Eigen::VectorXd::Zero(2 * dof);
  auto vel_residual = std::make_shared<crocoddyl::ResidualModelState>(state, x_zero, dof);
  crocoddyl::ActivationBounds vel_bounds;
  vel_bounds.lb = Eigen::VectorXd::Constant(2 * dof, -std::numeric_limits<double>::infinity());
  vel_bounds.ub = Eigen::VectorXd::Constant(2 * dof, std::numeric_limits<double>::infinity());
  for (int i = 0; i < dof; ++i) {
    const double vlim = model.velocityLimit[i];
    if (std::isfinite(vlim)) {
      vel_bounds.lb[dof + i] = -vlim;
      vel_bounds.ub[dof + i] = vlim;
    }
  }
  auto vel_activation = std::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(vel_bounds);
  auto vel_cost = std::make_shared<crocoddyl::CostModelResidual>(state, vel_activation, vel_residual);

  auto collision_residual =
      std::make_shared<ResidualModelTetraPGACollisionDistance<double>>(state, model, env, d_safe);
  const int num_collision_pairs = model.num_collision_ssl * env.num_static_sphere;
  crocoddyl::ActivationBounds collision_bounds;
  collision_bounds.lb = Eigen::VectorXd::Zero(num_collision_pairs);
  collision_bounds.ub =
      Eigen::VectorXd::Constant(num_collision_pairs, std::numeric_limits<double>::infinity());
  auto collision_activation =
      std::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(collision_bounds);
  auto collision_cost =
      std::make_shared<crocoddyl::CostModelResidual>(state, collision_activation, collision_residual);

  auto control_residual = std::make_shared<crocoddyl::ResidualModelControl>(state, dof);
  auto control_cost = std::make_shared<crocoddyl::CostModelResidual>(state, control_residual);

  running_cost->addCost("placement", placement_cost, config.placement_weight);
  running_cost->addCost("vel_limit", vel_cost, config.velocity_limit_weight);
  running_cost->addCost("collision", collision_cost, config.collision_weight);
  running_cost->addCost("control", control_cost, config.control_weight);
  terminal_cost->addCost("placement", placement_cost, config.terminal_placement_weight);
  terminal_cost->addCost("vel_limit", vel_cost, config.velocity_limit_weight);
  terminal_cost->addCost("collision", collision_cost, config.collision_weight);

  auto running_diff =
      std::make_shared<DifferentialActionModelTetraPGAForwardDynamics<double>>(state, model, running_cost);
  auto terminal_diff =
      std::make_shared<DifferentialActionModelTetraPGAForwardDynamics<double>>(state, model, terminal_cost);
  running_diff->set_u_lb(-model.effortLimit);
  running_diff->set_u_ub(model.effortLimit);
  terminal_diff->set_u_lb(-model.effortLimit);
  terminal_diff->set_u_ub(model.effortLimit);

  auto running_model =
      std::make_shared<crocoddyl::IntegratedActionModelEuler>(running_diff, config.dt);
  auto terminal_model =
      std::make_shared<crocoddyl::IntegratedActionModelEuler>(terminal_diff, config.dt);
  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> running_models(
      static_cast<std::size_t>(config.horizon), running_model);
  return std::make_shared<crocoddyl::ShootingProblem>(x0, running_models, terminal_model);
}

SolverArtifacts SolveProblem(const std::shared_ptr<crocoddyl::ShootingProblem>& problem,
                             const std::vector<Eigen::VectorXd>& init_xs,
                             const std::vector<Eigen::VectorXd>& init_us,
                             const CliConfig& config) {
  SolverArtifacts result;
  crocoddyl::SolverBoxFDDP solver(problem);
  solver.set_th_stop(config.th_stop);
  const Clock::time_point start = Clock::now();
  try {
    result.success = solver.solve(init_xs, init_us, static_cast<std::size_t>(config.max_iterations));
  } catch (const std::exception& e) {
    result.failed = true;
    result.failure_message = e.what();
  } catch (...) {
    result.failed = true;
    result.failure_message = "unknown exception";
  }
  result.solve_ms = DurationSeconds(Clock::now() - start).count() * 1e3;
  result.cost = solver.get_cost();
  result.stop = solver.get_stop();
  result.iter = solver.get_iter();
  result.xs = solver.get_xs();
  result.us = solver.get_us();
  if (result.xs.empty()) {
    result.xs = init_xs;
  }
  if (result.us.empty()) {
    result.us = init_us;
  }
  return result;
}

CollisionSummary SummarizeConfigurationCollision(const Model<double>& model,
                                                 const Environment<double>& env,
                                                 const Eigen::VectorXd& q,
                                                 const double d_safe) {
  Data<double> data(model);
  EnvironmentData<double> env_data(model, env);
  forwardKinematics(model, data, q);
  computeDistance(model, data, env, env_data);

  CollisionSummary summary;
  for (int i = 0; i < env_data.num_collision_pair; ++i) {
    const double distance = env_data.distance[i];
    summary.min_distance = std::min(summary.min_distance, distance);
    if (distance < d_safe) {
      ++summary.safety_violation_count;
    }
    if (distance < 0.0) {
      ++summary.collision_violation_count;
    }
  }
  return summary;
}

CollisionSummary SummarizeTrajectoryCollision(const Model<double>& model,
                                              const Environment<double>& env,
                                              const std::vector<Eigen::VectorXd>& xs,
                                              const double d_safe) {
  CollisionSummary out;
  for (const Eigen::VectorXd& x : xs) {
    const CollisionSummary step =
        SummarizeConfigurationCollision(model, env, x.head(model.dof_a), d_safe);
    out.min_distance = std::min(out.min_distance, step.min_distance);
    out.safety_violation_count += step.safety_violation_count;
    out.collision_violation_count += step.collision_violation_count;
  }
  return out;
}

TrajectoryMetrics ComputeTrajectoryMetrics(const std::vector<Eigen::VectorXd>& xs,
                                           const std::vector<Eigen::VectorXd>& us,
                                           const int dof, const double dt) {
  TrajectoryMetrics metrics;
  for (std::size_t k = 1; k < xs.size(); ++k) {
    metrics.path_length += (xs[k].head(dof) - xs[k - 1].head(dof)).norm();
  }

  if (xs.size() >= 4u && dt > 0.0) {
    double sum_sq = 0.0;
    std::size_t count = 0;
    const double inv_dt3 = 1.0 / (dt * dt * dt);
    for (std::size_t k = 3; k < xs.size(); ++k) {
      const Eigen::VectorXd jerk =
          (xs[k].head(dof) - 3.0 * xs[k - 1].head(dof) +
           3.0 * xs[k - 2].head(dof) - xs[k - 3].head(dof)) *
          inv_dt3;
      sum_sq += jerk.squaredNorm();
      ++count;
    }
    metrics.jerk_rms = std::sqrt(sum_sq / static_cast<double>(count));
  }

  if (us.size() >= 2u && dt > 0.0) {
    double sum_sq = 0.0;
    for (std::size_t k = 1; k < us.size(); ++k) {
      sum_sq += ((us[k] - us[k - 1]) / dt).squaredNorm();
    }
    metrics.torque_rate_rms = std::sqrt(sum_sq / static_cast<double>(us.size() - 1u));
  }
  return metrics;
}

double PlacementErrorNorm(const Model<double>& model, const Motor3D<double>& target,
                          const Eigen::VectorXd& q) {
  Data<double> data(model);
  forwardKinematics(model, data, q);
  const Motor3D<double> actual = data.M.col(model.n - 1);
  return ga_log(ga_mul(ga_rev(target), actual)).norm();
}

void WriteHeader(std::ofstream& out) {
  out << "model,dof,seed,sample_id,d_safe,obstacle_count,success,failed,cost,stop,iter,"
         "solve_ms,initial_min_distance,final_min_distance,trajectory_min_distance,"
         "safety_violation_count,collision_violation_count,placement_error_norm,"
         "path_length,jerk_rms,torque_rate_rms,failure_message\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CliConfig config = ParseCli(argc, argv);
    const std::filesystem::path output_path(config.output_csv);
    if (output_path.has_parent_path()) {
      std::filesystem::create_directories(output_path.parent_path());
    }
    std::ofstream out(config.output_csv);
    if (!out) {
      throw std::runtime_error("failed to open output CSV: " + config.output_csv);
    }
    WriteHeader(out);

    Model<double> model = ur();
    auto state = std::make_shared<crocoddyl::StateVector>(2 * model.dof_a);

    for (const double d_safe : config.margins) {
      for (const int obstacle_count : config.obstacle_counts) {
        for (int sample_id = 0; sample_id < config.samples; ++sample_id) {
          const std::uint32_t sample_seed =
              MixSeed(MixSeed(config.seed, static_cast<std::uint32_t>(obstacle_count)),
                      static_cast<std::uint32_t>(sample_id));
          std::mt19937 rng(sample_seed);
          const Eigen::VectorXd q0 =
              SampleJointConfiguration(model, rng, config.position_amplitude);
          const Eigen::VectorXd q_ref =
              SampleJointConfiguration(model, rng, config.position_amplitude);

          Eigen::VectorXd x0(2 * model.dof_a);
          x0.head(model.dof_a) = q0;
          x0.tail(model.dof_a).setZero();

          Data<double> ref_data(model);
          forwardKinematics(model, ref_data, q_ref);
          Data<double> initial_data(model);
          forwardKinematics(model, initial_data, q0);
          const Motor3D<double> target =
              align_motor_hemisphere(ref_data.M.col(model.n - 1),
                                     initial_data.M.col(model.n - 1));

          const Environment<double> env =
              MakeObstacleEnvironment(model, q0, q_ref, obstacle_count, sample_seed, config);
          const auto problem = BuildProblem(state, model, env, x0, target, config, d_safe);
          std::vector<Eigen::VectorXd> init_xs(static_cast<std::size_t>(config.horizon) + 1u,
                                               x0);
          std::vector<Eigen::VectorXd> init_us(static_cast<std::size_t>(config.horizon),
                                               Eigen::VectorXd::Zero(model.dof_a));

          const SolverArtifacts result = SolveProblem(problem, init_xs, init_us, config);
          const CollisionSummary initial_collision =
              SummarizeConfigurationCollision(model, env, q0, d_safe);
          const CollisionSummary trajectory_collision =
              SummarizeTrajectoryCollision(model, env, result.xs, d_safe);
          const Eigen::VectorXd final_q = result.xs.back().head(model.dof_a);
          const CollisionSummary final_collision =
              SummarizeConfigurationCollision(model, env, final_q, d_safe);
          const TrajectoryMetrics metrics =
              ComputeTrajectoryMetrics(result.xs, result.us, model.dof_a, config.dt);
          const double placement_error = PlacementErrorNorm(model, target, final_q);

          out << "ur10," << model.dof_a << ',' << sample_seed << ',' << sample_id << ','
              << FormatCsvNumber(d_safe) << ',' << obstacle_count << ','
              << (result.success ? 1 : 0) << ',' << (result.failed ? 1 : 0) << ','
              << FormatCsvNumber(result.cost) << ',' << FormatCsvNumber(result.stop) << ','
              << result.iter << ',' << FormatCsvNumber(result.solve_ms) << ','
              << FormatCsvNumber(initial_collision.min_distance) << ','
              << FormatCsvNumber(final_collision.min_distance) << ','
              << FormatCsvNumber(trajectory_collision.min_distance) << ','
              << trajectory_collision.safety_violation_count << ','
              << trajectory_collision.collision_violation_count << ','
              << FormatCsvNumber(placement_error) << ','
              << FormatCsvNumber(metrics.path_length) << ','
              << FormatCsvNumber(metrics.jerk_rms) << ','
              << FormatCsvNumber(metrics.torque_rate_rms) << ','
              << CsvEscape(result.failure_message) << '\n';
        }
      }
    }

    std::cout << "Wrote " << config.output_csv << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Crocoddyl_obstacle_margin_sweep failed: " << e.what() << std::endl;
    return 1;
  }
}
