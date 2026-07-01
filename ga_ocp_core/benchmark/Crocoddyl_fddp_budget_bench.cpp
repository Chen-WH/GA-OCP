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
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <crocoddyl/core/solver-base.hpp>
#include <crocoddyl/core/solvers/fddp.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include "ga_ocp/BenchUtils.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using DurationSeconds = std::chrono::duration<double>;

enum class ScenarioKind {
  kUR10,
  kLeapHand,
  kBinaryTree,
  kStanfordTidyBot,
  kBinaryTree31Dof,
};

enum class MethodKind {
  kTetraPGA,
  kPinocchio,
  kCasadi,
};

struct CliConfig {
  ScenarioKind scenario = ScenarioKind::kUR10;
  int branching_factor = 2;
  int level = 5;
  int samples = 24;
  std::uint32_t seed = BenchmarkRunSeed();
  std::vector<double> budgets_ms{1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0};
  FDDPBenchConfig solver_config{};
  double success_error_tol = 1e-2;
  double stop_tol = 1e-4;
  std::string output_prefix;
};

struct ScenarioContext {
  ScenarioKind scenario = ScenarioKind::kUR10;
  int dof = 0;
  std::string scenario_name;
  std::shared_ptr<Model<double>> ga_model;
  pinocchio::Model pin_model;
#ifdef GA_OCP_HAS_CASADI_BENCH
  std::shared_ptr<InlineAutoDiffABADerivatives> casadi_autodiff;
#endif
};

struct IterationRecord {
  std::size_t iter = 0;
  double elapsed_ms = 0.0;
  double iter_ms = 0.0;
  double cost = std::numeric_limits<double>::quiet_NaN();
  double best_cost = std::numeric_limits<double>::quiet_NaN();
  double terminal_q_error = std::numeric_limits<double>::quiet_NaN();
  double stop = std::numeric_limits<double>::quiet_NaN();
  int success = 0;
};

struct MethodRunResult {
  int sample_id = -1;
  MethodKind method = MethodKind::kTetraPGA;
  bool converged = false;
  bool failed = false;
  std::string failure_message;
  double solve_elapsed_ms = 0.0;
  double final_cost = std::numeric_limits<double>::quiet_NaN();
  double final_stop = std::numeric_limits<double>::quiet_NaN();
  std::size_t final_iter = 0;
  std::vector<IterationRecord> trace;
};

struct BudgetView {
  double best_cost = std::numeric_limits<double>::quiet_NaN();
  double terminal_q_error = std::numeric_limits<double>::quiet_NaN();
  int success = 0;
  std::size_t completed_iterations = 0;
  std::vector<double> iter_times_ms;
};

std::string ScenarioName(const ScenarioKind scenario) {
  switch (scenario) {
    case ScenarioKind::kUR10:
      return "ur10";
    case ScenarioKind::kLeapHand:
      return "leap_hand";
    case ScenarioKind::kBinaryTree:
      return "binary_tree";
    case ScenarioKind::kStanfordTidyBot:
      return "stanford_tidybot";
    case ScenarioKind::kBinaryTree31Dof:
      return "binary_tree_31dof";
  }
  return "unknown";
}

std::string MethodName(const MethodKind method) {
  switch (method) {
    case MethodKind::kTetraPGA:
      return "TetraPGA";
    case MethodKind::kPinocchio:
      return "Pinocchio";
    case MethodKind::kCasadi:
      return "CasADi";
  }
  return "Unknown";
}

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

std::string FormatVector(const Eigen::VectorXd& value) {
  std::ostringstream oss;
  oss << std::setprecision(17);
  for (Eigen::Index i = 0; i < value.size(); ++i) {
    if (i > 0) {
      oss << ' ';
    }
    oss << value(i);
  }
  return oss.str();
}

std::vector<double> ParseBudgets(const std::string& raw) {
  std::vector<double> out;
  std::stringstream ss(raw);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) {
      continue;
    }
    out.push_back(std::stod(item));
  }
  if (out.empty()) {
    throw std::invalid_argument("budgets_ms must contain at least one value");
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

ScenarioKind ParseScenario(const std::string& value) {
  if (value == "ur10") {
    return ScenarioKind::kUR10;
  }
  if (value == "leap_hand" || value == "leap") {
    return ScenarioKind::kLeapHand;
  }
  if (value == "binary_tree" || value == "tree") {
    return ScenarioKind::kBinaryTree;
  }
  if (value == "stanford_tidybot" || value == "tidybot") {
    return ScenarioKind::kStanfordTidyBot;
  }
  if (value == "binary_tree_31dof" || value == "bt31" || value == "tree31") {
    return ScenarioKind::kBinaryTree31Dof;
  }
  throw std::invalid_argument("unsupported scenario: " + value);
}

CliConfig ParseCli(int argc, char** argv) {
  CliConfig config;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i] == nullptr ? "" : argv[i];
    if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: Crocoddyl_fddp_budget_bench [options]\n"
          << "  --scenario=ur10|leap_hand|stanford_tidybot|binary_tree|binary_tree_31dof\n"
          << "  --level=<int>                  binary-tree level, DOF=(2^level-1)\n"
          << "  --branching_factor=<int>\n"
          << "  --samples=<int>\n"
          << "  --seed=<uint>\n"
          << "  --budgets_ms=1,2,5,10\n"
          << "  --output_prefix=<path>\n"
          << "  --dt=<double>\n"
          << "  --horizon=<int>\n"
          << "  --max_iterations=<int>\n"
          << "  --position_limit=<double>\n"
          << "  --state_weight=<double>\n"
          << "  --acc_weight=<double>\n"
          << "  --tau_weight=<double>\n"
          << "  --terminal_weight=<double>\n"
          << "  --success_error_tol=<double>\n"
          << "  --stop_tol=<double>\n";
      std::exit(0);
    }

    const std::size_t eq = arg.find('=');
    if (eq == std::string::npos) {
      throw std::invalid_argument("expected --key=value, got: " + arg);
    }

    const std::string key = arg.substr(0, eq);
    const std::string value = arg.substr(eq + 1);
    if (key == "--scenario" || key == "--robot") {
      config.scenario = ParseScenario(value);
    } else if (key == "--level" || key == "--tree_level") {
      config.level = std::stoi(value);
    } else if (key == "--branching_factor") {
      config.branching_factor = std::stoi(value);
    } else if (key == "--samples") {
      config.samples = std::stoi(value);
    } else if (key == "--seed") {
      config.seed = static_cast<std::uint32_t>(std::stoul(value));
    } else if (key == "--budgets_ms") {
      config.budgets_ms = ParseBudgets(value);
    } else if (key == "--output_prefix") {
      config.output_prefix = value;
    } else if (key == "--dt") {
      config.solver_config.dt = std::stod(value);
    } else if (key == "--horizon") {
      config.solver_config.horizon = static_cast<std::size_t>(std::stoul(value));
    } else if (key == "--max_iterations") {
      config.solver_config.max_iterations = static_cast<std::size_t>(std::stoul(value));
    } else if (key == "--position_limit") {
      config.solver_config.position_limit = std::stod(value);
    } else if (key == "--state_weight") {
      config.solver_config.state_weight = std::stod(value);
    } else if (key == "--acc_weight") {
      config.solver_config.acc_weight = std::stod(value);
    } else if (key == "--tau_weight") {
      config.solver_config.tau_weight = std::stod(value);
    } else if (key == "--terminal_weight") {
      config.solver_config.terminal_weight = std::stod(value);
    } else if (key == "--success_error_tol") {
      config.success_error_tol = std::stod(value);
    } else if (key == "--stop_tol") {
      config.stop_tol = std::stod(value);
    } else {
      throw std::invalid_argument("unknown option: " + key);
    }
  }

  if (config.samples <= 0) {
    throw std::invalid_argument("samples must be positive");
  }
  if (config.branching_factor <= 0) {
    throw std::invalid_argument("branching_factor must be positive");
  }
  if (config.level <= 0) {
    throw std::invalid_argument("level must be positive");
  }
  if (config.success_error_tol < 0.0) {
    throw std::invalid_argument("success_error_tol must be non-negative");
  }

  return config;
}

std::filesystem::path PackageRoot() {
  return std::filesystem::path(__FILE__).parent_path().parent_path();
}

std::filesystem::path RobotAssetsRoot() {
  return std::filesystem::path(GA_OCP_ROBOT_ASSETS_DIR);
}

std::string DefaultOutputPrefix(const CliConfig& config) {
  const std::filesystem::path log_dir = PackageRoot() / "log";
  std::filesystem::create_directories(log_dir);

  std::ostringstream stem;
  stem << "Crocoddyl_fddp_budget_bench_" << ScenarioName(config.scenario);
  if (config.scenario == ScenarioKind::kBinaryTree) {
    stem << "_bf" << config.branching_factor << "_level" << config.level;
  }
  stem << "_samples" << config.samples;
  return (log_dir / stem.str()).string();
}

std::vector<MethodKind> AvailableMethods() {
  std::vector<MethodKind> out{MethodKind::kTetraPGA, MethodKind::kPinocchio};
#ifdef GA_OCP_HAS_CASADI_BENCH
  out.push_back(MethodKind::kCasadi);
#endif
  return out;
}

pinocchio::Model BuildUr10PinModel() {
  const std::filesystem::path urdf_path = RobotAssetsRoot() / "ur10" / "urdf" / "ur10.urdf";
  pinocchio::Model pin_model;
  pinocchio::urdf::buildModel(urdf_path.string(), pin_model);
  return pin_model;
}

std::filesystem::path LeapHandUrdfPath() {
  return RobotAssetsRoot() / "leap_hand" / "urdf" / "leap_hand_left.urdf";
}

pinocchio::Model BuildLeapHandPinModel() {
  const std::filesystem::path urdf_path = LeapHandUrdfPath();
  pinocchio::Model pin_model;
  pinocchio::urdf::buildModel(urdf_path.string(), pin_model);
  return pin_model;
}

std::filesystem::path StanfordTidyBotUrdfPath() {
  return RobotAssetsRoot() / "stanford_tidybot" / "urdf" /
         "tidybot_gen3_10dof.urdf";
}

pinocchio::Model BuildStanfordTidyBotPinModel() {
  pinocchio::Model pin_model;
  pinocchio::urdf::buildModel(StanfordTidyBotUrdfPath().string(), pin_model);
  return pin_model;
}

ScenarioContext BuildScenarioContext(const CliConfig& config) {
  ScenarioContext context;
  context.scenario = config.scenario;
  context.scenario_name = ScenarioName(config.scenario);

  if (config.scenario == ScenarioKind::kUR10) {
    context.ga_model = std::make_shared<Model<double>>(ur());
    context.pin_model = BuildUr10PinModel();
    context.dof = context.ga_model->dof_a;
#ifdef GA_OCP_HAS_CASADI_BENCH
    context.casadi_autodiff =
        std::make_shared<InlineAutoDiffABADerivatives>(context.pin_model, "tetrapga_budget_ur10");
#endif
    return context;
  }

  if (config.scenario == ScenarioKind::kLeapHand) {
    const std::string urdf_path = LeapHandUrdfPath().string();
    context.ga_model = std::make_shared<Model<double>>(leap_hand(urdf_path));
    context.pin_model = BuildLeapHandPinModel();
    context.dof = context.ga_model->dof_a;
#ifdef GA_OCP_HAS_CASADI_BENCH
    context.casadi_autodiff =
        std::make_shared<InlineAutoDiffABADerivatives>(context.pin_model, "tetrapga_budget_leap_hand");
#endif
    return context;
  }

  if (config.scenario == ScenarioKind::kStanfordTidyBot) {
    const std::string urdf_path = StanfordTidyBotUrdfPath().string();
    context.ga_model = std::make_shared<Model<double>>(urdf_path);
    context.pin_model = BuildStanfordTidyBotPinModel();
    context.dof = context.ga_model->dof_a;
#ifdef GA_OCP_HAS_CASADI_BENCH
    context.casadi_autodiff = std::make_shared<InlineAutoDiffABADerivatives>(
        context.pin_model, "tetrapga_budget_stanford_tidybot");
#endif
    return context;
  }

  const int dof = config.scenario == ScenarioKind::kBinaryTree31Dof ? 31 : DofFromLevel(config.level);
  const int branching_factor =
      config.scenario == ScenarioKind::kBinaryTree31Dof ? 2 : config.branching_factor;
  const TreeTemplateParams params = MakeBenchTreeParams(branching_factor, dof);
  context.ga_model =
      std::make_shared<Model<double>>(MakeGABenchModel(params, "tetrapga_budget_binary_tree"));
  context.pin_model = BuildPinModel(params);
  context.dof = dof;
#ifdef GA_OCP_HAS_CASADI_BENCH
  context.casadi_autodiff = std::make_shared<InlineAutoDiffABADerivatives>(
      context.pin_model, "tetrapga_budget_tree_bf" + std::to_string(branching_factor) +
                             "_dof" + std::to_string(dof));
#endif
  return context;
}

std::shared_ptr<crocoddyl::ShootingProblem> BuildProblem(const ScenarioContext& context,
                                                         const MethodKind method,
                                                         const Eigen::VectorXd& x0,
                                                         const Eigen::VectorXd& x_target,
                                                         const FDDPBenchConfig& config) {
  switch (method) {
    case MethodKind::kTetraPGA:
      return BuildGAFDDPProblem(*context.ga_model, x0, x_target, config);
    case MethodKind::kPinocchio:
      return BuildPinFDDPProblem(context.pin_model, x0, x_target, config);
    case MethodKind::kCasadi:
#ifdef GA_OCP_HAS_CASADI_BENCH
      return BuildPinCasadiFDDPProblem(context.pin_model, x0, x_target, config,
                                       context.casadi_autodiff);
#else
      throw std::runtime_error("CasADi backend requested but benchmark was built without it");
#endif
  }
  throw std::runtime_error("unsupported method");
}

double TerminalQError(const std::vector<Eigen::VectorXd>& xs, const Eigen::VectorXd& x_target,
                      const int dof) {
  if (xs.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return (xs.back().head(dof) - x_target.head(dof)).norm();
}

class IterationTraceCallback final : public crocoddyl::CallbackAbstract {
 public:
  IterationTraceCallback(std::vector<IterationRecord>* trace, const Eigen::VectorXd* x_target,
                         const int dof, const double success_error_tol)
      : trace_(trace),
        x_target_(x_target),
        dof_(dof),
        success_error_tol_(success_error_tol) {}

  void Start(const Clock::time_point start_time, const double initial_best_cost) {
    start_time_ = start_time;
    last_elapsed_ms_ = 0.0;
    best_cost_ = initial_best_cost;
    accepted_iterations_ = 0;
  }

  void operator()(crocoddyl::SolverAbstract& solver) override {
    const double elapsed_ms = DurationSeconds(Clock::now() - start_time_).count() * 1e3;
    const double cost = solver.get_cost();
    if (std::isfinite(cost)) {
      best_cost_ = std::min(best_cost_, cost);
    }
    const double terminal_q_error = TerminalQError(solver.get_xs(), *x_target_, dof_);
    ++accepted_iterations_;

    trace_->push_back(IterationRecord{
        accepted_iterations_,
        elapsed_ms,
        elapsed_ms - last_elapsed_ms_,
        cost,
        best_cost_,
        terminal_q_error,
        solver.get_stop(),
        (std::isfinite(terminal_q_error) && terminal_q_error <= success_error_tol_) ? 1 : 0,
    });
    last_elapsed_ms_ = elapsed_ms;
  }

 private:
  std::vector<IterationRecord>* trace_;
  const Eigen::VectorXd* x_target_;
  int dof_;
  double success_error_tol_;
  Clock::time_point start_time_{};
  double last_elapsed_ms_ = 0.0;
  double best_cost_ = std::numeric_limits<double>::infinity();
  std::size_t accepted_iterations_ = 0;
};

std::vector<Eigen::VectorXd> MakeInitialXs(const Eigen::VectorXd& x0, const std::size_t horizon) {
  return std::vector<Eigen::VectorXd>(horizon + 1, x0);
}

std::vector<Eigen::VectorXd> MakeInitialUs(const int dof, const std::size_t horizon) {
  return std::vector<Eigen::VectorXd>(horizon, Eigen::VectorXd::Zero(dof));
}

MethodRunResult RunMethodOnSample(const ScenarioContext& context, const CliConfig& config,
                                  const int sample_id, const MethodKind method,
                                  const Eigen::VectorXd& x0, const Eigen::VectorXd& x_target) {
  MethodRunResult result;
  result.sample_id = sample_id;
  result.method = method;

  auto problem = BuildProblem(context, method, x0, x_target, config.solver_config);
  const std::vector<Eigen::VectorXd> init_xs =
      MakeInitialXs(x0, config.solver_config.horizon);
  const std::vector<Eigen::VectorXd> init_us =
      MakeInitialUs(context.dof, config.solver_config.horizon);

  const double initial_cost = problem->calc(init_xs, init_us);
  const double initial_error = TerminalQError(init_xs, x_target, context.dof);
  result.trace.push_back(IterationRecord{
      0,
      0.0,
      0.0,
      initial_cost,
      initial_cost,
      initial_error,
      std::numeric_limits<double>::quiet_NaN(),
      (std::isfinite(initial_error) && initial_error <= config.success_error_tol) ? 1 : 0,
  });

  crocoddyl::SolverFDDP solver(problem);
  solver.set_th_stop(config.stop_tol);

  auto callback = std::make_shared<IterationTraceCallback>(&result.trace, &x_target, context.dof,
                                                           config.success_error_tol);
  solver.setCallbacks({callback});

  const Clock::time_point start_time = Clock::now();
  callback->Start(start_time, initial_cost);
  try {
    result.converged = solver.solve(init_xs, init_us, config.solver_config.max_iterations, false);
  } catch (const std::exception& e) {
    result.failed = true;
    result.failure_message = e.what();
  } catch (...) {
    result.failed = true;
    result.failure_message = "unknown exception";
  }
  result.solve_elapsed_ms = DurationSeconds(Clock::now() - start_time).count() * 1e3;

  if (!result.failed) {
    result.final_cost = solver.get_cost();
    result.final_stop = solver.get_stop();
    result.final_iter = result.trace.empty() ? 0u : result.trace.back().iter;
  } else {
    result.final_cost = result.trace.back().cost;
    result.final_stop = std::numeric_limits<double>::quiet_NaN();
    result.final_iter = result.trace.back().iter;
  }

  return result;
}

BudgetView EvaluateBudget(const std::vector<IterationRecord>& trace, const double budget_ms) {
  BudgetView out;
  if (trace.empty()) {
    return out;
  }

  std::size_t last_index = 0;
  for (std::size_t i = 0; i < trace.size(); ++i) {
    if (trace[i].elapsed_ms <= budget_ms) {
      last_index = i;
    } else {
      break;
    }
  }

  const IterationRecord& last = trace[last_index];
  out.best_cost = last.best_cost;
  out.terminal_q_error = last.terminal_q_error;
  out.success = last.success;

  for (std::size_t i = 1; i <= last_index; ++i) {
    if (trace[i].iter_ms > 0.0 && std::isfinite(trace[i].iter_ms)) {
      out.iter_times_ms.push_back(trace[i].iter_ms);
    }
  }
  out.completed_iterations = out.iter_times_ms.size();
  return out;
}

double Mean(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  return sum / static_cast<double>(values.size());
}

double Median(std::vector<double> values) {
  if (values.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::sort(values.begin(), values.end());
  const std::size_t n = values.size();
  if (n % 2 == 1) {
    return values[n / 2];
  }
  return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

double Percentile95(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double rank = 0.95 * static_cast<double>(values.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(std::floor(rank));
  const std::size_t hi = static_cast<std::size_t>(std::ceil(rank));
  if (lo == hi) {
    return values[lo];
  }
  const double alpha = rank - static_cast<double>(lo);
  return (1.0 - alpha) * values[lo] + alpha * values[hi];
}

void WriteSamplesCsv(const std::string& path, const ScenarioContext& context,
                     const CliConfig& config, const FDDPSampleBatch& samples) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to open " + path);
  }

  out << "scenario,sample_id,seed,dof,horizon,max_iterations,x0,x_target\n";
  for (int sample_id = 0; sample_id < config.samples; ++sample_id) {
    out << CsvEscape(context.scenario_name) << ','
        << sample_id << ','
        << config.seed << ','
        << context.dof << ','
        << config.solver_config.horizon << ','
        << config.solver_config.max_iterations << ','
        << CsvEscape(FormatVector(samples.x0[static_cast<std::size_t>(sample_id)])) << ','
        << CsvEscape(FormatVector(samples.x_target[static_cast<std::size_t>(sample_id)])) << '\n';
  }
}

void WriteRunsCsv(const std::string& path, const std::vector<MethodRunResult>& runs) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to open " + path);
  }

  out << "sample_id,method,converged,failed,solve_elapsed_ms,final_iter,final_cost,final_stop,"
         "trace_points,failure_message\n";
  for (const MethodRunResult& run : runs) {
    out << run.sample_id << ','
        << CsvEscape(MethodName(run.method)) << ','
        << (run.converged ? 1 : 0) << ','
        << (run.failed ? 1 : 0) << ','
        << FormatCsvNumber(run.solve_elapsed_ms) << ','
        << run.final_iter << ','
        << FormatCsvNumber(run.final_cost) << ','
        << FormatCsvNumber(run.final_stop) << ','
        << run.trace.size() << ','
        << CsvEscape(run.failure_message) << '\n';
  }
}

void WriteTraceCsv(const std::string& path, const ScenarioContext& context,
                   const std::vector<MethodRunResult>& runs) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to open " + path);
  }

  out << "scenario,sample_id,method,iter,elapsed_ms,iter_ms,cost,best_cost,terminal_q_error,"
         "success,stop\n";
  for (const MethodRunResult& run : runs) {
    for (const IterationRecord& record : run.trace) {
      out << CsvEscape(context.scenario_name) << ','
          << run.sample_id << ','
          << CsvEscape(MethodName(run.method)) << ','
          << record.iter << ','
          << FormatCsvNumber(record.elapsed_ms) << ','
          << FormatCsvNumber(record.iter_ms) << ','
          << FormatCsvNumber(record.cost) << ','
          << FormatCsvNumber(record.best_cost) << ','
          << FormatCsvNumber(record.terminal_q_error) << ','
          << record.success << ','
          << FormatCsvNumber(record.stop) << '\n';
    }
  }
}

void WriteSummaryCsv(const std::string& path, const ScenarioContext& context, const CliConfig& config,
                     const std::vector<MethodRunResult>& runs) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to open " + path);
  }

  out << "scenario,method,budget_ms,num_samples,mean_best_cost,median_best_cost,"
         "mean_terminal_q_error,success_rate,mean_iterations,mean_iter_ms,p95_iter_ms\n";

  const std::vector<MethodKind> methods = AvailableMethods();
  for (const MethodKind method : methods) {
    std::vector<const MethodRunResult*> method_runs;
    for (const MethodRunResult& run : runs) {
      if (run.method == method) {
        method_runs.push_back(&run);
      }
    }

    for (const double budget_ms : config.budgets_ms) {
      std::vector<double> best_costs;
      std::vector<double> errors;
      std::vector<double> iter_counts;
      std::vector<double> all_iter_times_ms;
      std::size_t success_count = 0;

      best_costs.reserve(method_runs.size());
      errors.reserve(method_runs.size());
      iter_counts.reserve(method_runs.size());

      for (const MethodRunResult* run : method_runs) {
        const BudgetView budget_view = EvaluateBudget(run->trace, budget_ms);
        best_costs.push_back(budget_view.best_cost);
        errors.push_back(budget_view.terminal_q_error);
        iter_counts.push_back(static_cast<double>(budget_view.completed_iterations));
        success_count += static_cast<std::size_t>(budget_view.success);
        all_iter_times_ms.insert(all_iter_times_ms.end(), budget_view.iter_times_ms.begin(),
                                 budget_view.iter_times_ms.end());
      }

      out << CsvEscape(context.scenario_name) << ','
          << CsvEscape(MethodName(method)) << ','
          << FormatCsvNumber(budget_ms) << ','
          << method_runs.size() << ','
          << FormatCsvNumber(Mean(best_costs)) << ','
          << FormatCsvNumber(Median(best_costs)) << ','
          << FormatCsvNumber(Mean(errors)) << ','
          << FormatCsvNumber(method_runs.empty()
                                 ? 0.0
                                 : static_cast<double>(success_count) /
                                       static_cast<double>(method_runs.size())) << ','
          << FormatCsvNumber(Mean(iter_counts)) << ','
          << FormatCsvNumber(Mean(all_iter_times_ms)) << ','
          << FormatCsvNumber(Percentile95(all_iter_times_ms)) << '\n';
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    CliConfig config = ParseCli(argc, argv);
    if (config.output_prefix.empty()) {
      config.output_prefix = DefaultOutputPrefix(config);
    }

    const std::filesystem::path output_prefix(config.output_prefix);
    if (output_prefix.has_parent_path()) {
      std::filesystem::create_directories(output_prefix.parent_path());
    }

    const ScenarioContext context = BuildScenarioContext(config);
    const FDDPSampleBatch samples =
        MakeFDDPSamples(context.dof, config.samples, config.seed, config.solver_config.position_limit);
    const std::vector<MethodKind> methods = AvailableMethods();

    std::vector<MethodRunResult> runs;
    runs.reserve(static_cast<std::size_t>(config.samples) * methods.size());

    for (int sample_id = 0; sample_id < config.samples; ++sample_id) {
      const Eigen::VectorXd& x0 = samples.x0[static_cast<std::size_t>(sample_id)];
      const Eigen::VectorXd& x_target = samples.x_target[static_cast<std::size_t>(sample_id)];
      for (const MethodKind method : methods) {
        std::cout << "[sample " << sample_id << "/" << config.samples
                  << "] method=" << MethodName(method) << std::endl;
        runs.push_back(RunMethodOnSample(context, config, sample_id, method, x0, x_target));
      }
    }

    WriteSamplesCsv(config.output_prefix + "_samples.csv", context, config, samples);
    WriteRunsCsv(config.output_prefix + "_runs.csv", runs);
    WriteTraceCsv(config.output_prefix + "_trace.csv", context, runs);
    WriteSummaryCsv(config.output_prefix + "_summary.csv", context, config, runs);

    std::cout << "Wrote results to:" << std::endl;
    std::cout << "  " << config.output_prefix << "_samples.csv" << std::endl;
    std::cout << "  " << config.output_prefix << "_runs.csv" << std::endl;
    std::cout << "  " << config.output_prefix << "_trace.csv" << std::endl;
    std::cout << "  " << config.output_prefix << "_summary.csv" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Crocoddyl_fddp_budget_bench failed: " << e.what() << std::endl;
    return 1;
  }
}
