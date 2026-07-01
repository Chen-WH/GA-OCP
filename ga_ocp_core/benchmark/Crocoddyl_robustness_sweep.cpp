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

#include <crocoddyl/core/solvers/fddp.hpp>

#include "ga_ocp/BenchUtils.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using DurationSeconds = std::chrono::duration<double>;

enum class PerturbationKind {
  kNominal,
  kMassInertiaScale,
  kComOffset,
  kExternalForce,
};

struct CliConfig {
  int samples = 20;
  std::uint32_t seed = 424242u;
  FDDPBenchConfig solver_config{};
  double success_rmse_tol = 0.05;
  std::size_t mpc_iterations = 10;
  std::vector<double> mass_scale_deltas{-0.30, -0.20, -0.10, 0.10, 0.20, 0.30};
  std::vector<double> com_offsets_m{0.01, 0.02, 0.05};
  std::vector<double> external_forces_n{5.0, 10.0};
  std::string output_csv = "Crocoddyl_robustness_sweep.csv";
};

struct PerturbationCase {
  PerturbationKind kind = PerturbationKind::kNominal;
  double level = 0.0;
};

struct SolveResult {
  bool failed = false;
  bool converged = false;
  std::string failure_message;
  double solve_ms = 0.0;
  double final_cost = std::numeric_limits<double>::quiet_NaN();
  double final_stop = std::numeric_limits<double>::quiet_NaN();
  std::size_t final_iter = 0;
  double planning_terminal_rmse = std::numeric_limits<double>::quiet_NaN();
  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
};

struct RolloutResult {
  bool finite = true;
  double terminal_error_norm = std::numeric_limits<double>::quiet_NaN();
  double terminal_rmse = std::numeric_limits<double>::quiet_NaN();
  double trajectory_rmse = std::numeric_limits<double>::quiet_NaN();
  double max_abs_q = 0.0;
  double max_abs_dq = 0.0;
  double rms_tau = 0.0;
  double max_tau_ratio = 0.0;
  int mpc_steps = 0;
  int mpc_failed_steps = 0;
  int mpc_converged_steps = 0;
  double mean_mpc_solve_ms = 0.0;
  double max_mpc_solve_ms = 0.0;
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
    throw std::invalid_argument("double list must not be empty");
  }
  return out;
}

CliConfig ParseCli(int argc, char** argv) {
  CliConfig config;
  config.solver_config.horizon = 50;
  config.solver_config.max_iterations = 100;
  config.solver_config.position_limit = 0.50;
  config.solver_config.terminal_weight = 1000.0;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i] == nullptr ? "" : argv[i];
    if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: Crocoddyl_robustness_sweep [options]\n"
          << "  --samples=<int>\n"
          << "  --seed=<uint>\n"
          << "  --output_csv=<path>\n"
          << "  --dt=<double>\n"
          << "  --horizon=<int>\n"
          << "  --max_iterations=<int>\n"
          << "  --mpc_iterations=<int>\n"
          << "  --position_limit=<double>\n"
          << "  --success_rmse_tol=<double>\n"
          << "  --mass_scale_deltas=-0.3,-0.2,...\n"
          << "  --com_offsets_m=0.01,0.02,0.05\n"
          << "  --external_forces_n=5,10      single-step EE wrench impulse levels\n";
      std::exit(0);
    }

    const std::size_t eq = arg.find('=');
    if (eq == std::string::npos) {
      throw std::invalid_argument("expected --key=value, got: " + arg);
    }
    const std::string key = arg.substr(0, eq);
    const std::string value = arg.substr(eq + 1);
    if (key == "--samples") {
      config.samples = std::stoi(value);
    } else if (key == "--seed") {
      config.seed = static_cast<std::uint32_t>(std::stoul(value));
    } else if (key == "--output_csv") {
      config.output_csv = value;
    } else if (key == "--dt") {
      config.solver_config.dt = std::stod(value);
    } else if (key == "--horizon") {
      config.solver_config.horizon = static_cast<std::size_t>(std::stoul(value));
    } else if (key == "--max_iterations") {
      config.solver_config.max_iterations = static_cast<std::size_t>(std::stoul(value));
    } else if (key == "--mpc_iterations") {
      config.mpc_iterations = static_cast<std::size_t>(std::stoul(value));
    } else if (key == "--position_limit") {
      config.solver_config.position_limit = std::stod(value);
    } else if (key == "--success_rmse_tol") {
      config.success_rmse_tol = std::stod(value);
    } else if (key == "--mass_scale_deltas") {
      config.mass_scale_deltas = ParseDoubleList(value);
    } else if (key == "--com_offsets_m") {
      config.com_offsets_m = ParseDoubleList(value);
    } else if (key == "--external_forces_n") {
      config.external_forces_n = ParseDoubleList(value);
    } else {
      throw std::invalid_argument("unknown option: " + key);
    }
  }

  if (config.samples <= 0) {
    throw std::invalid_argument("samples must be positive");
  }
  if (config.success_rmse_tol < 0.0) {
    throw std::invalid_argument("success_rmse_tol must be non-negative");
  }
  if (config.mpc_iterations == 0u) {
    throw std::invalid_argument("mpc_iterations must be positive");
  }
  if (config.output_csv.empty()) {
    throw std::invalid_argument("output_csv must not be empty");
  }
  return config;
}

std::string PerturbationName(const PerturbationKind kind) {
  switch (kind) {
    case PerturbationKind::kNominal:
      return "nominal";
    case PerturbationKind::kMassInertiaScale:
      return "mass_inertia_scale";
    case PerturbationKind::kComOffset:
      return "com_offset";
    case PerturbationKind::kExternalForce:
      return "external_force";
  }
  return "unknown";
}

std::vector<PerturbationCase> MakePerturbationCases(const CliConfig& config) {
  std::vector<PerturbationCase> cases;
  cases.push_back(PerturbationCase{PerturbationKind::kNominal, 0.0});
  for (const double level : config.mass_scale_deltas) {
    cases.push_back(PerturbationCase{PerturbationKind::kMassInertiaScale, level});
  }
  for (const double level : config.com_offsets_m) {
    cases.push_back(PerturbationCase{PerturbationKind::kComOffset, level});
  }
  for (const double level : config.external_forces_n) {
    cases.push_back(PerturbationCase{PerturbationKind::kExternalForce, level});
  }
  return cases;
}

Eigen::Matrix3d Skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d out;
  out << 0.0, -v.z(), v.y(),
         v.z(), 0.0, -v.x(),
         -v.y(), v.x(), 0.0;
  return out;
}

Eigen::Vector3d Unskew(const Eigen::Matrix3d& m) {
  return Eigen::Vector3d(
      0.5 * (m(2, 1) - m(1, 2)),
      0.5 * (m(0, 2) - m(2, 0)),
      0.5 * (m(1, 0) - m(0, 1)));
}

Eigen::Vector3d DeterministicUnitVector(const std::uint32_t seed,
                                        const std::uint32_t sample_id,
                                        const std::uint32_t stream_id,
                                        const std::uint32_t body_id) {
  std::mt19937 rng(MixBenchmarkSeed(seed, stream_id, sample_id, body_id));
  std::normal_distribution<double> normal(0.0, 1.0);
  Eigen::Vector3d direction(normal(rng), normal(rng), normal(rng));
  const double norm = direction.norm();
  if (norm <= 1e-12) {
    return Eigen::Vector3d::UnitZ();
  }
  return direction / norm;
}

void ApplyMassInertiaScale(Model<double>& model, const double delta) {
  const double scale = 1.0 + delta;
  if (scale <= 0.0) {
    throw std::invalid_argument("mass/inertia scale must remain positive");
  }
  for (int i = 1; i < model.n; ++i) {
    model.I[static_cast<std::size_t>(i)] *= scale;
  }
}

void ApplyComOffset(Model<double>& model, const double offset_m,
                    const std::uint32_t seed, const std::uint32_t sample_id) {
  if (offset_m == 0.0) {
    return;
  }
  for (int i = 1; i < model.n; ++i) {
    if (i != model.n - 1) {
      continue;
    }
    auto& inertia = model.I[static_cast<std::size_t>(i)];
    const double mass = inertia.block<3, 3>(0, 3).trace() / 3.0;
    if (!(mass > 1e-12)) {
      continue;
    }

    const Eigen::Matrix3d rc_old = inertia.block<3, 3>(3, 3) / mass;
    const Eigen::Vector3d com_old = Unskew(rc_old);
    const Eigen::Matrix3d central_inertia =
        inertia.block<3, 3>(3, 0) + mass * rc_old * rc_old;

    const Eigen::Vector3d direction = DeterministicUnitVector(
        seed, sample_id, 0xC0FFEEu, static_cast<std::uint32_t>(i));
    const Eigen::Vector3d com_new = com_old + offset_m * direction;
    const Eigen::Matrix3d rc_new = Skew(com_new);

    inertia.block<3, 3>(0, 0) = -mass * rc_new;
    inertia.block<3, 3>(0, 3) = mass * Eigen::Matrix3d::Identity();
    inertia.block<3, 3>(3, 0) = central_inertia - mass * rc_new * rc_new;
    inertia.block<3, 3>(3, 3) = mass * rc_new;
  }
}

Model<double> MakePerturbedModel(const Model<double>& nominal,
                                 const PerturbationCase& perturbation,
                                 const CliConfig& config,
                                 const int sample_id) {
  Model<double> model = nominal;
  switch (perturbation.kind) {
    case PerturbationKind::kNominal:
    case PerturbationKind::kExternalForce:
      break;
    case PerturbationKind::kMassInertiaScale:
      ApplyMassInertiaScale(model, perturbation.level);
      break;
    case PerturbationKind::kComOffset:
      ApplyComOffset(model, perturbation.level, config.seed, static_cast<std::uint32_t>(sample_id));
      break;
  }
  return model;
}

void ResetForwardDynamicsData(const Model<double>& model, Data<double>& data) {
  data.Mi.setZero();
  data.Mi.col(0) << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  data.M.setZero();
  data.M.col(0) << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  data.L.setZero();
  data.Lstar.setZero();
  data.V.setZero();
  data.dL.setZero();
  data.dV.setZero();
  data.dV.col(0) = -model.gravity;
  data.F.setZero();
  data.gamma.setZero();
  data.gammaT.setZero();
  data.d.setZero();
  data.u.setZero();
  for (auto& inertia : data.Ia) {
    inertia.setZero();
  }
  data.ddq.setZero();
}

Eigen::Matrix<double, 6, Eigen::Dynamic> MakeExternalForce(
    const Model<double>& model, const PerturbationCase& perturbation,
    const CliConfig& config, const int sample_id) {
  Eigen::Matrix<double, 6, Eigen::Dynamic> fext(6, model.n);
  fext.setZero();
  if (perturbation.kind != PerturbationKind::kExternalForce || perturbation.level == 0.0) {
    return fext;
  }

  const Eigen::Vector3d direction = DeterministicUnitVector(
      config.seed, static_cast<std::uint32_t>(sample_id), 0xBADC0DEu, 0u);
  fext.block<3, 1>(3, model.n - 1) =
      perturbation.level * config.solver_config.dt * direction;
  return fext;
}

double MaxTauRatio(const Model<double>& model, const Eigen::VectorXd& tau) {
  double out = 0.0;
  for (Eigen::Index i = 0; i < tau.size(); ++i) {
    const double limit = i < model.effortLimit.size() ? std::abs(model.effortLimit(i)) : 0.0;
    if (std::isfinite(limit) && limit > 1e-9) {
      out = std::max(out, std::abs(tau(i)) / limit);
    }
  }
  return out;
}

std::vector<Eigen::VectorXd> MakeInitialXs(const Eigen::VectorXd& x0,
                                           const std::size_t horizon) {
  return std::vector<Eigen::VectorXd>(horizon + 1, x0);
}

std::vector<Eigen::VectorXd> MakeInitialUs(const int dof, const std::size_t horizon) {
  return std::vector<Eigen::VectorXd>(horizon, Eigen::VectorXd::Zero(dof));
}

SolveResult SolveNominalProblem(const Model<double>& nominal,
                                const CliConfig& config,
                                const Eigen::VectorXd& x0,
                                const Eigen::VectorXd& x_target) {
  SolveResult result;
  auto problem = BuildGAFDDPProblem(nominal, x0, x_target, config.solver_config);
  const std::vector<Eigen::VectorXd> init_xs =
      MakeInitialXs(x0, config.solver_config.horizon);
  const std::vector<Eigen::VectorXd> init_us =
      MakeInitialUs(nominal.dof_a, config.solver_config.horizon);

  crocoddyl::SolverFDDP solver(problem);
  solver.set_th_stop(1e-4);

  const Clock::time_point start_time = Clock::now();
  try {
    result.converged = solver.solve(
        init_xs, init_us, config.solver_config.max_iterations, false);
  } catch (const std::exception& e) {
    result.failed = true;
    result.failure_message = e.what();
  } catch (...) {
    result.failed = true;
    result.failure_message = "unknown exception";
  }
  result.solve_ms = DurationSeconds(Clock::now() - start_time).count() * 1e3;

  result.xs = solver.get_xs();
  result.us = solver.get_us();
  if (!result.failed) {
    result.final_cost = solver.get_cost();
    result.final_stop = solver.get_stop();
    result.final_iter = solver.get_iter();
  }
  if (!result.xs.empty()) {
    const double terminal_norm =
        (result.xs.back().head(nominal.dof_a) - x_target.head(nominal.dof_a)).norm();
    result.planning_terminal_rmse =
        terminal_norm / std::sqrt(static_cast<double>(nominal.dof_a));
  }
  return result;
}

RolloutResult RolloutClosedLoopPlant(const Model<double>& nominal_model,
                                     const Model<double>& plant_model,
                                     const PerturbationCase& perturbation,
                                     const CliConfig& config,
                                     const int sample_id,
                                     const Eigen::VectorXd& x0,
                                     const Eigen::VectorXd& x_target) {
  RolloutResult result;
  const int dof = plant_model.dof_a;
  Eigen::VectorXd q = x0.head(dof);
  Eigen::VectorXd dq = x0.tail(dof);
  Data<double> data(plant_model);
  const Eigen::Matrix<double, 6, Eigen::Dynamic> disturbance_fext =
      MakeExternalForce(plant_model, perturbation, config, sample_id);
  Eigen::Matrix<double, 6, Eigen::Dynamic> zero_fext(6, plant_model.n);
  zero_fext.setZero();

  CliConfig mpc_config = config;
  mpc_config.solver_config.max_iterations = config.mpc_iterations;

  double trajectory_error_sq_sum = 0.0;
  double tau_sq_sum = 0.0;
  double mpc_solve_ms_sum = 0.0;
  std::size_t tau_count = 0;

  for (std::size_t k = 0; k < config.solver_config.horizon; ++k) {
    Eigen::VectorXd x_current(2 * dof);
    x_current << q, dq;
    const SolveResult step_solve =
        SolveNominalProblem(nominal_model, mpc_config, x_current, x_target);

    ++result.mpc_steps;
    result.mpc_failed_steps += step_solve.failed ? 1 : 0;
    result.mpc_converged_steps += step_solve.converged ? 1 : 0;
    mpc_solve_ms_sum += step_solve.solve_ms;
    result.max_mpc_solve_ms = std::max(result.max_mpc_solve_ms, step_solve.solve_ms);

    Eigen::VectorXd tau = Eigen::VectorXd::Zero(dof);
    if (!step_solve.us.empty() && step_solve.us.front().size() == dof) {
      tau = step_solve.us.front();
    }
    tau_sq_sum += tau.squaredNorm();
    tau_count += static_cast<std::size_t>(tau.size());
    result.max_tau_ratio = std::max(result.max_tau_ratio, MaxTauRatio(plant_model, tau));

    const bool apply_disturbance =
        perturbation.kind == PerturbationKind::kExternalForce &&
        k == config.solver_config.horizon / 2u;
    const Eigen::Matrix<double, 6, Eigen::Dynamic>& fext_step =
        apply_disturbance ? disturbance_fext : zero_fext;

    ResetForwardDynamicsData(plant_model, data);
    const Eigen::VectorXd ddq = forwardDynamics0(plant_model, data, q, dq, tau, fext_step);
    dq += config.solver_config.dt * ddq;
    q += config.solver_config.dt * dq;

    result.max_abs_q = std::max(result.max_abs_q, q.cwiseAbs().maxCoeff());
    result.max_abs_dq = std::max(result.max_abs_dq, dq.cwiseAbs().maxCoeff());
    trajectory_error_sq_sum += (q - x_target.head(dof)).squaredNorm();

    if (!q.allFinite() || !dq.allFinite() || !ddq.allFinite()) {
      result.finite = false;
      break;
    }
  }

  result.terminal_error_norm = (q - x_target.head(dof)).norm();
  result.terminal_rmse = result.terminal_error_norm / std::sqrt(static_cast<double>(dof));
  result.trajectory_rmse = std::sqrt(
      trajectory_error_sq_sum /
      std::max(1.0, static_cast<double>(std::max(1, result.mpc_steps)) * dof));
  result.rms_tau = std::sqrt(tau_sq_sum / std::max<std::size_t>(1, tau_count));
  result.mean_mpc_solve_ms =
      mpc_solve_ms_sum / std::max(1.0, static_cast<double>(result.mpc_steps));
  return result;
}

void WriteHeader(std::ofstream& out) {
  out << "scenario,method,sample_id,seed,dof,horizon,max_iterations,perturbation,level,"
         "solver_failed,solver_converged,solve_ms,solver_final_iter,solver_final_cost,"
         "solver_final_stop,planning_terminal_rmse,mpc_steps,mpc_failed_steps,"
         "mpc_converged_steps,mean_mpc_solve_ms,max_mpc_solve_ms,rollout_finite,rollout_terminal_error_norm,"
         "rollout_terminal_rmse,rollout_trajectory_rmse,max_abs_q,max_abs_dq,rms_tau,"
         "max_tau_ratio,success,failure_message\n";
}

void WriteRow(std::ofstream& out, const CliConfig& config, const int sample_id,
              const Model<double>& nominal, const PerturbationCase& perturbation,
              const SolveResult& solve, const RolloutResult& rollout) {
  const bool any_solve_failed = solve.failed || rollout.mpc_failed_steps > 0;
  const bool all_mpc_converged =
      rollout.mpc_steps > 0 && rollout.mpc_converged_steps == rollout.mpc_steps;
  const int success = (!any_solve_failed && rollout.finite &&
                       rollout.terminal_rmse <= config.success_rmse_tol)
                          ? 1
                          : 0;
  out << CsvEscape("ur10") << ','
      << CsvEscape("TetraPGA") << ','
      << sample_id << ','
      << config.seed << ','
      << nominal.dof_a << ','
      << config.solver_config.horizon << ','
      << config.mpc_iterations << ','
      << CsvEscape(PerturbationName(perturbation.kind)) << ','
      << FormatCsvNumber(perturbation.level) << ','
      << (any_solve_failed ? 1 : 0) << ','
      << (all_mpc_converged ? 1 : 0) << ','
      << FormatCsvNumber(rollout.mean_mpc_solve_ms) << ','
      << solve.final_iter << ','
      << FormatCsvNumber(solve.final_cost) << ','
      << FormatCsvNumber(solve.final_stop) << ','
      << FormatCsvNumber(solve.planning_terminal_rmse) << ','
      << rollout.mpc_steps << ','
      << rollout.mpc_failed_steps << ','
      << rollout.mpc_converged_steps << ','
      << FormatCsvNumber(rollout.mean_mpc_solve_ms) << ','
      << FormatCsvNumber(rollout.max_mpc_solve_ms) << ','
      << (rollout.finite ? 1 : 0) << ','
      << FormatCsvNumber(rollout.terminal_error_norm) << ','
      << FormatCsvNumber(rollout.terminal_rmse) << ','
      << FormatCsvNumber(rollout.trajectory_rmse) << ','
      << FormatCsvNumber(rollout.max_abs_q) << ','
      << FormatCsvNumber(rollout.max_abs_dq) << ','
      << FormatCsvNumber(rollout.rms_tau) << ','
      << FormatCsvNumber(rollout.max_tau_ratio) << ','
      << success << ','
      << CsvEscape(solve.failure_message) << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CliConfig config = ParseCli(argc, argv);
    const std::filesystem::path output_path(config.output_csv);
    if (output_path.has_parent_path()) {
      std::filesystem::create_directories(output_path.parent_path());
    }

    const Model<double> nominal = ur();
    const FDDPSampleBatch samples = MakeFDDPSamples(
        nominal.dof_a, config.samples, config.seed, config.solver_config.position_limit);
    const std::vector<PerturbationCase> perturbations = MakePerturbationCases(config);

    std::ofstream out(config.output_csv);
    if (!out) {
      throw std::runtime_error("failed to open output CSV: " + config.output_csv);
    }
    WriteHeader(out);

    for (int sample_id = 0; sample_id < config.samples; ++sample_id) {
      const Eigen::VectorXd& x0 = samples.x0[static_cast<std::size_t>(sample_id)];
      const Eigen::VectorXd& x_target =
          samples.x_target[static_cast<std::size_t>(sample_id)];

      std::cout << "[sample " << sample_id << "/" << config.samples
                << "] closed-loop nominal-controller rollout" << std::endl;
      const SolveResult solve = SolveNominalProblem(nominal, config, x0, x_target);

      for (const PerturbationCase& perturbation : perturbations) {
        const Model<double> plant =
            MakePerturbedModel(nominal, perturbation, config, sample_id);
        const RolloutResult rollout =
            RolloutClosedLoopPlant(nominal, plant, perturbation, config,
                                   sample_id, x0, x_target);
        WriteRow(out, config, sample_id, nominal, perturbation, solve, rollout);
      }
    }

    std::cout << "Wrote " << config.output_csv << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Crocoddyl_robustness_sweep failed: " << e.what() << std::endl;
    return 1;
  }
}
