#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <crocoddyl/core/costs/cost-sum.hpp>
#include <crocoddyl/core/costs/residual.hpp>
#include <crocoddyl/core/integrator/euler.hpp>
#include <crocoddyl/core/residuals/control.hpp>
#include <crocoddyl/core/residuals/joint-acceleration.hpp>
#include <crocoddyl/core/solver-base.hpp>
#include <crocoddyl/core/solvers/fddp.hpp>
#include <crocoddyl/core/states/euclidean.hpp>
#include <crocoddyl/multibody/actions/free-fwddyn.hpp>
#include <crocoddyl/multibody/actuations/full.hpp>
#include <crocoddyl/multibody/residuals/state.hpp>
#include <crocoddyl/multibody/states/multibody.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include "ga_ocp/BenchUtils.hpp"
#include "ga_ocp/RuntimeProfiler.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using DurationSeconds = std::chrono::duration<double>;

enum class RobotKind {
  kUR10,
  kLeapHand,
  kUnitreeG1,
};

enum class BackendKind {
  kTetraPGA,
  kPinocchio,
  kCasadi,
};

struct CliConfig {
  std::vector<RobotKind> robots{RobotKind::kUR10, RobotKind::kLeapHand, RobotKind::kUnitreeG1};
  std::vector<BackendKind> backends{BackendKind::kTetraPGA, BackendKind::kPinocchio,
                                    BackendKind::kCasadi};
  int samples = 24;
  std::uint32_t seed = 0x20260627u;
  FDDPBenchConfig solver_config{};
  double stop_tol = 1e-4;
  bool warmup = true;
  std::string output_dir;
};

struct RobotContext {
  RobotKind kind = RobotKind::kUR10;
  std::string name;
  int dof = 0;
  std::shared_ptr<Model<double>> ga_model;
  pinocchio::Model pin_model;
#ifdef GA_OCP_HAS_CASADI_BENCH
  std::shared_ptr<InlineAutoDiffABADerivatives> casadi_autodiff;
#endif
};

struct ProfileMetrics {
  double dam_calc_ms = 0.0;
  double dam_calcdiff_ms = 0.0;
  double cost_state_calc_ms = 0.0;
  double cost_state_calcdiff_ms = 0.0;
  double cost_control_calc_ms = 0.0;
  double cost_control_calcdiff_ms = 0.0;
  double cost_acc_calc_ms = 0.0;
  double cost_acc_calcdiff_ms = 0.0;
  double cost_collision_calc_ms = 0.0;
  double cost_collision_calcdiff_ms = 0.0;
  double cost_other_calc_ms = 0.0;
  double cost_other_calcdiff_ms = 0.0;
  std::uint64_t dam_calc_calls = 0;
  std::uint64_t dam_calcdiff_calls = 0;
};

struct IterationProfileRecord {
  std::string robot;
  std::string backend;
  int sample_id = -1;
  std::size_t callback_index = 0;
  std::size_t solver_iter = 0;
  double elapsed_ms = 0.0;
  double iter_interval_ms = 0.0;
  double cost = std::numeric_limits<double>::quiet_NaN();
  double stop = std::numeric_limits<double>::quiet_NaN();
  ProfileMetrics metrics;
};

struct RunRecord {
  std::string robot;
  std::string backend;
  int dof = 0;
  int sample_id = -1;
  bool converged = false;
  bool failed = false;
  std::string failure_message;
  std::size_t solver_iterations = 0;
  double solve_total_ms = 0.0;
  double final_cost = std::numeric_limits<double>::quiet_NaN();
  double final_stop = std::numeric_limits<double>::quiet_NaN();
  ProfileMetrics metrics;
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

std::string RobotName(const RobotKind robot) {
  switch (robot) {
    case RobotKind::kUR10:
      return "ur10";
    case RobotKind::kLeapHand:
      return "leap_hand";
    case RobotKind::kUnitreeG1:
      return "unitree_g1";
  }
  return "unknown";
}

std::string BackendName(const BackendKind backend) {
  switch (backend) {
    case BackendKind::kTetraPGA:
      return "TetraPGA";
    case BackendKind::kPinocchio:
      return "Pinocchio";
    case BackendKind::kCasadi:
      return "CasADi";
  }
  return "Unknown";
}

std::filesystem::path RobotAssetsRoot() {
  return std::filesystem::path(GA_OCP_ROBOT_ASSETS_DIR);
}

std::filesystem::path DefaultOutputDir() {
  const std::filesystem::path package_root =
      std::filesystem::path(__FILE__).parent_path().parent_path();
  return package_root / "log" / "runtime_breakdown_offline";
}

std::vector<std::string> SplitCommaList(const std::string& raw) {
  std::vector<std::string> out;
  std::stringstream ss(raw);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      out.push_back(item);
    }
  }
  return out;
}

RobotKind ParseRobot(const std::string& value) {
  if (value == "ur10" || value == "ur") {
    return RobotKind::kUR10;
  }
  if (value == "leap_hand" || value == "leap" || value == "leap_left") {
    return RobotKind::kLeapHand;
  }
  if (value == "unitree_g1" || value == "g1") {
    return RobotKind::kUnitreeG1;
  }
  throw std::invalid_argument("unsupported robot: " + value);
}

BackendKind ParseBackend(const std::string& value) {
  if (value == "tetrapga" || value == "TetraPGA") {
    return BackendKind::kTetraPGA;
  }
  if (value == "pinocchio" || value == "Pinocchio") {
    return BackendKind::kPinocchio;
  }
  if (value == "casadi" || value == "CasADi") {
    return BackendKind::kCasadi;
  }
  throw std::invalid_argument("unsupported backend: " + value);
}

CliConfig ParseCli(int argc, char** argv) {
  CliConfig config;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i] == nullptr ? "" : argv[i];
    if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: Crocoddyl_runtime_breakdown [options]\n"
          << "  --robots=ur10,leap_hand,unitree_g1\n"
          << "  --backends=tetrapga,pinocchio,casadi\n"
          << "  --samples=<int>\n"
          << "  --seed=<uint>\n"
          << "  --output_dir=<path>\n"
          << "  --dt=<double>\n"
          << "  --horizon=<int>\n"
          << "  --max_iterations=<int>\n"
          << "  --position_limit=<double>\n"
          << "  --state_weight=<double>\n"
          << "  --acc_weight=<double>\n"
          << "  --tau_weight=<double>\n"
          << "  --terminal_weight=<double>\n"
          << "  --stop_tol=<double>\n"
          << "  --warmup=0|1\n";
      std::exit(0);
    }

    const std::size_t eq = arg.find('=');
    if (eq == std::string::npos) {
      throw std::invalid_argument("expected --key=value, got: " + arg);
    }
    const std::string key = arg.substr(0, eq);
    const std::string value = arg.substr(eq + 1);

    if (key == "--robots") {
      config.robots.clear();
      for (const std::string& item : SplitCommaList(value)) {
        config.robots.push_back(ParseRobot(item));
      }
    } else if (key == "--backends") {
      config.backends.clear();
      for (const std::string& item : SplitCommaList(value)) {
        config.backends.push_back(ParseBackend(item));
      }
    } else if (key == "--samples") {
      config.samples = std::stoi(value);
    } else if (key == "--seed") {
      config.seed = static_cast<std::uint32_t>(std::stoul(value));
    } else if (key == "--output_dir") {
      config.output_dir = value;
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
    } else if (key == "--stop_tol") {
      config.stop_tol = std::stod(value);
    } else if (key == "--warmup") {
      config.warmup = (value != "0" && value != "false" && value != "False");
    } else {
      throw std::invalid_argument("unknown option: " + key);
    }
  }

  if (config.robots.empty()) {
    throw std::invalid_argument("robots list must not be empty");
  }
  if (config.backends.empty()) {
    throw std::invalid_argument("backends list must not be empty");
  }
  if (config.samples <= 0) {
    throw std::invalid_argument("samples must be positive");
  }
  return config;
}

void SetGravity(RobotContext& context) {
  context.pin_model.gravity.linear() << 0.0, 0.0, -9.81;
  context.pin_model.gravity.angular().setZero();
  if (context.ga_model && context.ga_model->gravity.size() >= 6) {
    context.ga_model->gravity.setZero();
    context.ga_model->gravity(5) = -9.81;
  }
}

pinocchio::Model BuildPinModelFromUrdf(const std::filesystem::path& urdf_path) {
  pinocchio::Model model;
  pinocchio::urdf::buildModel(urdf_path.string(), model);
  model.gravity.linear() << 0.0, 0.0, -9.81;
  model.gravity.angular().setZero();
  return model;
}

RobotContext BuildRobotContext(const RobotKind robot) {
  RobotContext context;
  context.kind = robot;
  context.name = RobotName(robot);

  if (robot == RobotKind::kUR10) {
    const std::filesystem::path urdf_path = RobotAssetsRoot() / "ur10" / "urdf" / "ur10.urdf";
    context.ga_model = std::make_shared<Model<double>>(ur());
    context.pin_model = BuildPinModelFromUrdf(urdf_path);
  } else if (robot == RobotKind::kLeapHand) {
    const std::filesystem::path urdf_path =
        RobotAssetsRoot() / "leap_hand" / "urdf" / "leap_hand_left.urdf";
    context.ga_model = std::make_shared<Model<double>>(leap_hand(urdf_path.string()));
    context.pin_model = BuildPinModelFromUrdf(urdf_path);
  } else if (robot == RobotKind::kUnitreeG1) {
    const std::filesystem::path urdf_path =
        RobotAssetsRoot() / "unitree_g1" / "urdf" / "g1_29dof_rev_1_0.urdf";
    context.ga_model = std::make_shared<Model<double>>(urdf_path.string());
    context.pin_model = BuildPinModelFromUrdf(urdf_path);
  }

  SetGravity(context);
  context.dof = context.ga_model->dof_a;
  if (context.pin_model.nq != context.pin_model.nv) {
    throw std::runtime_error(context.name + " requires nq == nv for this Euclidean FDDP setup");
  }
  if (context.pin_model.nv != context.dof) {
    throw std::runtime_error(context.name + " DOF mismatch between TetraPGA and Pinocchio");
  }

#ifdef GA_OCP_HAS_CASADI_BENCH
  context.casadi_autodiff = std::make_shared<InlineAutoDiffABADerivatives>(
      context.pin_model, "runtime_breakdown_" + context.name);
#endif
  return context;
}

ProfileMetrics MetricsFromCounters(const ga_ocp::RuntimeProfilerCounters& c) {
  ProfileMetrics m;
  m.dam_calc_ms = c.dam_calc_ms;
  m.dam_calcdiff_ms = c.dam_calcdiff_ms;
  m.cost_state_calc_ms =
      c.cost_item_calc_ms[static_cast<std::size_t>(ga_ocp::RuntimeCostCategory::kState)];
  m.cost_state_calcdiff_ms =
      c.cost_item_calcdiff_ms[static_cast<std::size_t>(ga_ocp::RuntimeCostCategory::kState)];
  m.cost_control_calc_ms =
      c.cost_item_calc_ms[static_cast<std::size_t>(ga_ocp::RuntimeCostCategory::kControl)];
  m.cost_control_calcdiff_ms =
      c.cost_item_calcdiff_ms[static_cast<std::size_t>(ga_ocp::RuntimeCostCategory::kControl)];
  m.cost_acc_calc_ms =
      c.cost_item_calc_ms[static_cast<std::size_t>(ga_ocp::RuntimeCostCategory::kVelocity)];
  m.cost_acc_calcdiff_ms =
      c.cost_item_calcdiff_ms[static_cast<std::size_t>(ga_ocp::RuntimeCostCategory::kVelocity)];
  m.cost_collision_calc_ms =
      c.cost_item_calc_ms[static_cast<std::size_t>(ga_ocp::RuntimeCostCategory::kCollision)];
  m.cost_collision_calcdiff_ms =
      c.cost_item_calcdiff_ms[static_cast<std::size_t>(ga_ocp::RuntimeCostCategory::kCollision)];
  m.cost_other_calc_ms =
      c.cost_item_calc_ms[static_cast<std::size_t>(ga_ocp::RuntimeCostCategory::kOther)];
  m.cost_other_calcdiff_ms =
      c.cost_item_calcdiff_ms[static_cast<std::size_t>(ga_ocp::RuntimeCostCategory::kOther)];
  m.dam_calc_calls = c.dam_calc_calls;
  m.dam_calcdiff_calls = c.dam_calcdiff_calls;
  return m;
}

ProfileMetrics SubtractMetrics(const ProfileMetrics& a, const ProfileMetrics& b) {
  ProfileMetrics out;
  out.dam_calc_ms = a.dam_calc_ms - b.dam_calc_ms;
  out.dam_calcdiff_ms = a.dam_calcdiff_ms - b.dam_calcdiff_ms;
  out.cost_state_calc_ms = a.cost_state_calc_ms - b.cost_state_calc_ms;
  out.cost_state_calcdiff_ms = a.cost_state_calcdiff_ms - b.cost_state_calcdiff_ms;
  out.cost_control_calc_ms = a.cost_control_calc_ms - b.cost_control_calc_ms;
  out.cost_control_calcdiff_ms = a.cost_control_calcdiff_ms - b.cost_control_calcdiff_ms;
  out.cost_acc_calc_ms = a.cost_acc_calc_ms - b.cost_acc_calc_ms;
  out.cost_acc_calcdiff_ms = a.cost_acc_calcdiff_ms - b.cost_acc_calcdiff_ms;
  out.cost_collision_calc_ms = a.cost_collision_calc_ms - b.cost_collision_calc_ms;
  out.cost_collision_calcdiff_ms = a.cost_collision_calcdiff_ms - b.cost_collision_calcdiff_ms;
  out.cost_other_calc_ms = a.cost_other_calc_ms - b.cost_other_calc_ms;
  out.cost_other_calcdiff_ms = a.cost_other_calcdiff_ms - b.cost_other_calcdiff_ms;
  out.dam_calc_calls = a.dam_calc_calls - b.dam_calc_calls;
  out.dam_calcdiff_calls = a.dam_calcdiff_calls - b.dam_calcdiff_calls;
  return out;
}

double DamTotalMs(const ProfileMetrics& m) {
  return m.dam_calc_ms + m.dam_calcdiff_ms;
}

double CostStateTotalMs(const ProfileMetrics& m) {
  return m.cost_state_calc_ms + m.cost_state_calcdiff_ms;
}

double CostControlTotalMs(const ProfileMetrics& m) {
  return m.cost_control_calc_ms + m.cost_control_calcdiff_ms;
}

double CostAccTotalMs(const ProfileMetrics& m) {
  return m.cost_acc_calc_ms + m.cost_acc_calcdiff_ms;
}

double CostCollisionTotalMs(const ProfileMetrics& m) {
  return m.cost_collision_calc_ms + m.cost_collision_calcdiff_ms;
}

double CostOtherTotalMs(const ProfileMetrics& m) {
  return m.cost_other_calc_ms + m.cost_other_calcdiff_ms;
}

double CostTotalMs(const ProfileMetrics& m) {
  return CostStateTotalMs(m) + CostControlTotalMs(m) + CostAccTotalMs(m) +
         CostCollisionTotalMs(m) + CostOtherTotalMs(m);
}

double NonCostModelMs(const ProfileMetrics& m) {
  return std::max(0.0, DamTotalMs(m) - CostTotalMs(m));
}

double SolverOverheadMs(const double solve_total_ms, const ProfileMetrics& m) {
  return solve_total_ms - DamTotalMs(m);
}

class TimedDifferentialActionModel final
    : public crocoddyl::DifferentialActionModelAbstractTpl<double> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Base = crocoddyl::DifferentialActionModelAbstractTpl<double>;
  using DifferentialActionDataAbstract = crocoddyl::DifferentialActionDataAbstractTpl<double>;
  using VectorXs = typename crocoddyl::MathBaseTpl<double>::VectorXs;

  explicit TimedDifferentialActionModel(std::shared_ptr<Base> inner)
      : Base(inner->get_state(), inner->get_nu(), inner->get_nr(), inner->get_ng(),
             inner->get_nh(), inner->get_ng_T(), inner->get_nh_T()),
        inner_(std::move(inner)) {
    this->set_u_lb(inner_->get_u_lb());
    this->set_u_ub(inner_->get_u_ub());
    if (inner_->get_ng() > 0 || inner_->get_ng_T() > 0) {
      this->set_g_lb(inner_->get_g_lb());
      this->set_g_ub(inner_->get_g_ub());
    }
  }

  void calc(const std::shared_ptr<DifferentialActionDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u) override {
    if (!ga_ocp::RuntimeProfilerEnabled()) {
      inner_->calc(data, x, u);
      return;
    }
    const auto start = ga_ocp::RuntimeProfilerNow();
    inner_->calc(data, x, u);
    ga_ocp::RuntimeProfilerRecordDamCalc(ga_ocp::RuntimeProfilerElapsedMs(start));
  }

  void calc(const std::shared_ptr<DifferentialActionDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x) override {
    if (!ga_ocp::RuntimeProfilerEnabled()) {
      inner_->calc(data, x);
      return;
    }
    const auto start = ga_ocp::RuntimeProfilerNow();
    inner_->calc(data, x);
    ga_ocp::RuntimeProfilerRecordDamCalc(ga_ocp::RuntimeProfilerElapsedMs(start));
  }

  void calcDiff(const std::shared_ptr<DifferentialActionDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u) override {
    if (!ga_ocp::RuntimeProfilerEnabled()) {
      inner_->calcDiff(data, x, u);
      return;
    }
    const auto start = ga_ocp::RuntimeProfilerNow();
    inner_->calcDiff(data, x, u);
    ga_ocp::RuntimeProfilerRecordDamCalcDiff(ga_ocp::RuntimeProfilerElapsedMs(start));
  }

  void calcDiff(const std::shared_ptr<DifferentialActionDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& x) override {
    if (!ga_ocp::RuntimeProfilerEnabled()) {
      inner_->calcDiff(data, x);
      return;
    }
    const auto start = ga_ocp::RuntimeProfilerNow();
    inner_->calcDiff(data, x);
    ga_ocp::RuntimeProfilerRecordDamCalcDiff(ga_ocp::RuntimeProfilerElapsedMs(start));
  }

  std::shared_ptr<DifferentialActionDataAbstract> createData() override {
    return inner_->createData();
  }

  bool checkData(const std::shared_ptr<DifferentialActionDataAbstract>& data) override {
    return inner_->checkData(data);
  }

  std::shared_ptr<crocoddyl::DifferentialActionModelBase> cloneAsDouble() const override {
    throw std::runtime_error("cloneAsDouble not implemented for TimedDifferentialActionModel");
  }

  std::shared_ptr<crocoddyl::DifferentialActionModelBase> cloneAsFloat() const override {
    throw std::runtime_error("cloneAsFloat not implemented for TimedDifferentialActionModel");
  }

 private:
  std::shared_ptr<Base> inner_;
};

class IterationProfilerCallback final : public crocoddyl::CallbackAbstract {
 public:
  IterationProfilerCallback(std::vector<IterationProfileRecord>* records, std::string robot,
                            std::string backend, const int sample_id)
      : records_(records),
        robot_(std::move(robot)),
        backend_(std::move(backend)),
        sample_id_(sample_id) {}

  void Start(const Clock::time_point start_time) {
    start_time_ = start_time;
    last_elapsed_ms_ = 0.0;
    callback_index_ = 0;
    last_metrics_ = MetricsFromCounters(ga_ocp::SnapshotRuntimeProfilerCounters());
  }

  void operator()(crocoddyl::SolverAbstract& solver) override {
    const double elapsed_ms = DurationSeconds(Clock::now() - start_time_).count() * 1e3;
    const ProfileMetrics metrics =
        MetricsFromCounters(ga_ocp::SnapshotRuntimeProfilerCounters());
    const ProfileMetrics delta = SubtractMetrics(metrics, last_metrics_);

    ++callback_index_;
    records_->push_back(IterationProfileRecord{
        robot_,
        backend_,
        sample_id_,
        callback_index_,
        solver.get_iter(),
        elapsed_ms,
        elapsed_ms - last_elapsed_ms_,
        solver.get_cost(),
        solver.get_stop(),
        delta,
    });

    last_elapsed_ms_ = elapsed_ms;
    last_metrics_ = metrics;
  }

 private:
  std::vector<IterationProfileRecord>* records_;
  std::string robot_;
  std::string backend_;
  int sample_id_ = -1;
  Clock::time_point start_time_{};
  double last_elapsed_ms_ = 0.0;
  std::size_t callback_index_ = 0;
  ProfileMetrics last_metrics_;
};

std::shared_ptr<crocoddyl::CostModelAbstract> ProfiledResidualCost(
    const std::shared_ptr<crocoddyl::StateAbstract>& state,
    const std::shared_ptr<crocoddyl::ResidualModelAbstract>& residual,
    const ga_ocp::RuntimeCostCategory category) {
  return ga_ocp::ProfileCost<double>(std::make_shared<crocoddyl::CostModelResidual>(state, residual),
                                     category);
}

std::shared_ptr<crocoddyl::ShootingProblem> BuildTetraPGAProblem(
    const Model<double>& ga_model, const Eigen::VectorXd& x0, const Eigen::VectorXd& x_target,
    const FDDPBenchConfig& config) {
  auto state = std::static_pointer_cast<crocoddyl::StateAbstract>(
      std::make_shared<crocoddyl::StateVector>(2 * ga_model.dof_a));
  auto running_cost = std::make_shared<crocoddyl::CostModelSum>(state);
  auto terminal_cost = std::make_shared<crocoddyl::CostModelSum>(state);

  running_cost->addCost(
      "state_reg",
      ProfiledResidualCost(state, std::make_shared<crocoddyl::ResidualModelState>(state, x_target),
                           ga_ocp::RuntimeCostCategory::kState),
      config.state_weight);
  running_cost->addCost(
      "acc_reg",
      ProfiledResidualCost(
          state,
          std::make_shared<ResidualModelTetraPGAJointAcceleration<double>>(
              state, ga_model, Eigen::VectorXd::Zero(ga_model.dof_a)),
          ga_ocp::RuntimeCostCategory::kVelocity),
      config.acc_weight);
  running_cost->addCost(
      "tau_reg",
      ProfiledResidualCost(state,
                           std::make_shared<crocoddyl::ResidualModelControl>(state, ga_model.dof_a),
                           ga_ocp::RuntimeCostCategory::kControl),
      config.tau_weight);
  terminal_cost->addCost(
      "state_reg",
      ProfiledResidualCost(state, std::make_shared<crocoddyl::ResidualModelState>(state, x_target),
                           ga_ocp::RuntimeCostCategory::kState),
      config.terminal_weight);

  auto running_diff = std::make_shared<TimedDifferentialActionModel>(
      std::make_shared<DifferentialActionModelTetraPGAForwardDynamics<double>>(
          state, ga_model, running_cost));
  auto terminal_diff = std::make_shared<TimedDifferentialActionModel>(
      std::make_shared<DifferentialActionModelTetraPGAForwardDynamics<double>>(
          state, ga_model, terminal_cost));
  auto running_model = std::make_shared<crocoddyl::IntegratedActionModelEuler>(
      running_diff, config.dt);
  auto terminal_model = std::make_shared<crocoddyl::IntegratedActionModelEuler>(
      terminal_diff, config.dt);

  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> running_models(config.horizon,
                                                                              running_model);
  return std::make_shared<crocoddyl::ShootingProblem>(x0, running_models, terminal_model);
}

std::shared_ptr<crocoddyl::ShootingProblem> BuildPinocchioProblem(
    const pinocchio::Model& pin_model, const Eigen::VectorXd& x0,
    const Eigen::VectorXd& x_target, const FDDPBenchConfig& config) {
  auto state = std::static_pointer_cast<crocoddyl::StateAbstract>(
      std::make_shared<crocoddyl::StateMultibody>(
          std::make_shared<pinocchio::Model>(pin_model)));
  auto state_mb = std::static_pointer_cast<crocoddyl::StateMultibody>(state);
  auto running_cost = std::make_shared<crocoddyl::CostModelSum>(state);
  auto terminal_cost = std::make_shared<crocoddyl::CostModelSum>(state);

  running_cost->addCost(
      "state_reg",
      ProfiledResidualCost(state, std::make_shared<crocoddyl::ResidualModelState>(state, x_target),
                           ga_ocp::RuntimeCostCategory::kState),
      config.state_weight);
  running_cost->addCost(
      "acc_reg",
      ProfiledResidualCost(
          state,
          std::make_shared<crocoddyl::ResidualModelJointAcceleration>(
              state, Eigen::VectorXd::Zero(pin_model.nv)),
          ga_ocp::RuntimeCostCategory::kVelocity),
      config.acc_weight);
  running_cost->addCost(
      "tau_reg",
      ProfiledResidualCost(state,
                           std::make_shared<crocoddyl::ResidualModelControl>(state, pin_model.nv),
                           ga_ocp::RuntimeCostCategory::kControl),
      config.tau_weight);
  terminal_cost->addCost(
      "state_reg",
      ProfiledResidualCost(state, std::make_shared<crocoddyl::ResidualModelState>(state, x_target),
                           ga_ocp::RuntimeCostCategory::kState),
      config.terminal_weight);

  auto actuation = std::make_shared<crocoddyl::ActuationModelFull>(state_mb);
  auto running_diff = std::make_shared<TimedDifferentialActionModel>(
      std::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(
          state_mb, actuation, running_cost));
  auto terminal_diff = std::make_shared<TimedDifferentialActionModel>(
      std::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(
          state_mb, actuation, terminal_cost));
  auto running_model = std::make_shared<crocoddyl::IntegratedActionModelEuler>(
      running_diff, config.dt);
  auto terminal_model = std::make_shared<crocoddyl::IntegratedActionModelEuler>(
      terminal_diff, config.dt);

  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> running_models(config.horizon,
                                                                              running_model);
  return std::make_shared<crocoddyl::ShootingProblem>(x0, running_models, terminal_model);
}

#ifdef GA_OCP_HAS_CASADI_BENCH
std::shared_ptr<crocoddyl::ShootingProblem> BuildCasadiProblem(
    const pinocchio::Model& pin_model, const Eigen::VectorXd& x0,
    const Eigen::VectorXd& x_target, const FDDPBenchConfig& config,
    const std::shared_ptr<InlineAutoDiffABADerivatives>& autodiff) {
  auto state = std::static_pointer_cast<crocoddyl::StateAbstract>(
      std::make_shared<crocoddyl::StateMultibody>(
          std::make_shared<pinocchio::Model>(pin_model)));
  auto state_mb = std::static_pointer_cast<crocoddyl::StateMultibody>(state);
  auto running_cost = std::make_shared<crocoddyl::CostModelSum>(state);
  auto terminal_cost = std::make_shared<crocoddyl::CostModelSum>(state);

  running_cost->addCost(
      "state_reg",
      ProfiledResidualCost(state, std::make_shared<crocoddyl::ResidualModelState>(state, x_target),
                           ga_ocp::RuntimeCostCategory::kState),
      config.state_weight);
  running_cost->addCost(
      "acc_reg",
      ProfiledResidualCost(
          state,
          std::make_shared<ResidualModelAccelerationPinocchioCasadi>(
              state, Eigen::VectorXd::Zero(pin_model.nv)),
          ga_ocp::RuntimeCostCategory::kVelocity),
      config.acc_weight);
  running_cost->addCost(
      "tau_reg",
      ProfiledResidualCost(state,
                           std::make_shared<crocoddyl::ResidualModelControl>(state, pin_model.nv),
                           ga_ocp::RuntimeCostCategory::kControl),
      config.tau_weight);
  terminal_cost->addCost(
      "state_reg",
      ProfiledResidualCost(state, std::make_shared<crocoddyl::ResidualModelState>(state, x_target),
                           ga_ocp::RuntimeCostCategory::kState),
      config.terminal_weight);

  auto running_diff = std::make_shared<TimedDifferentialActionModel>(
      std::make_shared<DifferentialActionModelPinocchioCasadi>(
          state_mb, pin_model, running_cost, autodiff));
  auto terminal_diff = std::make_shared<TimedDifferentialActionModel>(
      std::make_shared<DifferentialActionModelPinocchioCasadi>(
          state_mb, pin_model, terminal_cost, autodiff));
  auto running_model = std::make_shared<crocoddyl::IntegratedActionModelEuler>(
      running_diff, config.dt);
  auto terminal_model = std::make_shared<crocoddyl::IntegratedActionModelEuler>(
      terminal_diff, config.dt);

  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> running_models(config.horizon,
                                                                              running_model);
  return std::make_shared<crocoddyl::ShootingProblem>(x0, running_models, terminal_model);
}
#endif

std::shared_ptr<crocoddyl::ShootingProblem> BuildProblem(
    const RobotContext& context, const BackendKind backend, const Eigen::VectorXd& x0,
    const Eigen::VectorXd& x_target, const FDDPBenchConfig& config) {
  switch (backend) {
    case BackendKind::kTetraPGA:
      return BuildTetraPGAProblem(*context.ga_model, x0, x_target, config);
    case BackendKind::kPinocchio:
      return BuildPinocchioProblem(context.pin_model, x0, x_target, config);
    case BackendKind::kCasadi:
#ifdef GA_OCP_HAS_CASADI_BENCH
      return BuildCasadiProblem(context.pin_model, x0, x_target, config,
                                context.casadi_autodiff);
#else
      throw std::runtime_error("CasADi backend requested but benchmark was built without it");
#endif
  }
  throw std::runtime_error("unsupported backend");
}

std::vector<Eigen::VectorXd> MakeInitialXs(const Eigen::VectorXd& x0,
                                           const std::size_t horizon) {
  return std::vector<Eigen::VectorXd>(horizon + 1, x0);
}

std::vector<Eigen::VectorXd> MakeInitialUs(const int dof, const std::size_t horizon) {
  return std::vector<Eigen::VectorXd>(horizon, Eigen::VectorXd::Zero(dof));
}

RunRecord RunOneSample(const RobotContext& context, const BackendKind backend,
                       const CliConfig& config, const int sample_id,
                       const Eigen::VectorXd& x0, const Eigen::VectorXd& x_target,
                       std::vector<IterationProfileRecord>* iteration_records) {
  RunRecord run;
  run.robot = context.name;
  run.backend = BackendName(backend);
  run.dof = context.dof;
  run.sample_id = sample_id;

  auto problem = BuildProblem(context, backend, x0, x_target, config.solver_config);
  const std::vector<Eigen::VectorXd> init_xs =
      MakeInitialXs(x0, config.solver_config.horizon);
  const std::vector<Eigen::VectorXd> init_us =
      MakeInitialUs(context.dof, config.solver_config.horizon);

  crocoddyl::SolverFDDP solver(problem);
  solver.set_th_stop(config.stop_tol);
  auto callback = std::make_shared<IterationProfilerCallback>(
      iteration_records, run.robot, run.backend, sample_id);
  solver.setCallbacks({callback});

  ga_ocp::ResetRuntimeProfilerCounters();
  ga_ocp::SetRuntimeProfilerEnabled(true);
  const Clock::time_point start_time = Clock::now();
  callback->Start(start_time);

  try {
    run.converged = solver.solve(init_xs, init_us,
                                 static_cast<unsigned int>(config.solver_config.max_iterations),
                                 false);
  } catch (const std::exception& e) {
    run.failed = true;
    run.failure_message = e.what();
  } catch (...) {
    run.failed = true;
    run.failure_message = "unknown exception";
  }

  run.solve_total_ms = DurationSeconds(Clock::now() - start_time).count() * 1e3;
  run.metrics = MetricsFromCounters(ga_ocp::SnapshotRuntimeProfilerCounters());
  ga_ocp::SetRuntimeProfilerEnabled(false);

  run.solver_iterations = static_cast<std::size_t>(solver.get_iter());
  run.final_cost = run.failed ? std::numeric_limits<double>::quiet_NaN() : solver.get_cost();
  run.final_stop = run.failed ? std::numeric_limits<double>::quiet_NaN() : solver.get_stop();
  return run;
}

void RunWarmup(const RobotContext& context, const BackendKind backend, const CliConfig& config,
               const FDDPSampleBatch& samples) {
  std::vector<IterationProfileRecord> ignored;
  const int sample_id = -1;
  const Eigen::VectorXd& x0 = samples.x0.front();
  const Eigen::VectorXd& x_target = samples.x_target.front();
  (void)RunOneSample(context, backend, config, sample_id, x0, x_target, &ignored);
}

double Mean(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

double Percentile(std::vector<double> values, const double p) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double rank = p * static_cast<double>(values.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(std::floor(rank));
  const std::size_t hi = static_cast<std::size_t>(std::ceil(rank));
  if (lo == hi) {
    return values[lo];
  }
  const double alpha = rank - static_cast<double>(lo);
  return (1.0 - alpha) * values[lo] + alpha * values[hi];
}

void WriteMetricColumns(std::ostream& out, const ProfileMetrics& m,
                        const double solve_total_ms, const double divisor) {
  const double safe_divisor = divisor > 0.0 ? divisor : 1.0;
  const double dam_total = DamTotalMs(m);
  const double cost_total = CostTotalMs(m);
  const double non_cost = NonCostModelMs(m);
  const double overhead = SolverOverheadMs(solve_total_ms, m);

  out << ','
      << FormatCsvNumber(m.dam_calc_ms / safe_divisor) << ','
      << FormatCsvNumber(m.dam_calcdiff_ms / safe_divisor) << ','
      << FormatCsvNumber(dam_total / safe_divisor) << ','
      << FormatCsvNumber(m.cost_state_calc_ms / safe_divisor) << ','
      << FormatCsvNumber(m.cost_state_calcdiff_ms / safe_divisor) << ','
      << FormatCsvNumber(CostStateTotalMs(m) / safe_divisor) << ','
      << FormatCsvNumber(m.cost_control_calc_ms / safe_divisor) << ','
      << FormatCsvNumber(m.cost_control_calcdiff_ms / safe_divisor) << ','
      << FormatCsvNumber(CostControlTotalMs(m) / safe_divisor) << ','
      << FormatCsvNumber(m.cost_acc_calc_ms / safe_divisor) << ','
      << FormatCsvNumber(m.cost_acc_calcdiff_ms / safe_divisor) << ','
      << FormatCsvNumber(CostAccTotalMs(m) / safe_divisor) << ','
      << FormatCsvNumber(m.cost_collision_calc_ms / safe_divisor) << ','
      << FormatCsvNumber(m.cost_collision_calcdiff_ms / safe_divisor) << ','
      << FormatCsvNumber(CostCollisionTotalMs(m) / safe_divisor) << ','
      << FormatCsvNumber(m.cost_other_calc_ms / safe_divisor) << ','
      << FormatCsvNumber(m.cost_other_calcdiff_ms / safe_divisor) << ','
      << FormatCsvNumber(CostOtherTotalMs(m) / safe_divisor) << ','
      << FormatCsvNumber(cost_total / safe_divisor) << ','
      << FormatCsvNumber(non_cost / safe_divisor) << ','
      << FormatCsvNumber(overhead / safe_divisor) << ','
      << FormatCsvNumber((cost_total + non_cost + overhead) / safe_divisor);
}

std::string MetricHeaderSuffix(const std::string& suffix) {
  return "dam_calc_" + suffix + ",dam_calcdiff_" + suffix + ",dam_total_" + suffix +
         ",cost_state_calc_" + suffix + ",cost_state_calcdiff_" + suffix +
         ",cost_state_total_" + suffix + ",cost_control_calc_" + suffix +
         ",cost_control_calcdiff_" + suffix + ",cost_control_total_" + suffix +
         ",cost_acc_calc_" + suffix + ",cost_acc_calcdiff_" + suffix +
         ",cost_acc_total_" + suffix + ",cost_collision_calc_" + suffix +
         ",cost_collision_calcdiff_" + suffix + ",cost_collision_total_" + suffix +
         ",cost_other_calc_" + suffix + ",cost_other_calcdiff_" + suffix +
         ",cost_other_total_" + suffix + ",cost_total_" + suffix +
         ",non_cost_model_" + suffix + ",solver_overhead_" + suffix +
         ",stack_total_" + suffix;
}

void WriteRawCsv(const std::filesystem::path& path, const std::vector<RunRecord>& runs) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to open " + path.string());
  }
  out << "robot,backend,dof,sample_id,converged,failed,solver_iterations,solve_total_ms,"
      << "final_cost,final_stop,failure_message," << MetricHeaderSuffix("ms")
      << ",dam_calc_calls,dam_calcdiff_calls," << MetricHeaderSuffix("per_iter_ms") << '\n';
  for (const RunRecord& run : runs) {
    const double iter_divisor = static_cast<double>(std::max<std::size_t>(run.solver_iterations, 1));
    out << CsvEscape(run.robot) << ','
        << CsvEscape(run.backend) << ','
        << run.dof << ','
        << run.sample_id << ','
        << (run.converged ? 1 : 0) << ','
        << (run.failed ? 1 : 0) << ','
        << run.solver_iterations << ','
        << FormatCsvNumber(run.solve_total_ms) << ','
        << FormatCsvNumber(run.final_cost) << ','
        << FormatCsvNumber(run.final_stop) << ','
        << CsvEscape(run.failure_message);
    WriteMetricColumns(out, run.metrics, run.solve_total_ms, 1.0);
    out << ',' << run.metrics.dam_calc_calls << ',' << run.metrics.dam_calcdiff_calls;
    WriteMetricColumns(out, run.metrics, run.solve_total_ms, iter_divisor);
    out << '\n';
  }
}

void WriteIterationsCsv(const std::filesystem::path& path,
                        const std::vector<IterationProfileRecord>& records) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to open " + path.string());
  }
  out << "robot,backend,sample_id,callback_index,solver_iter,elapsed_ms,iter_interval_ms,"
      << "cost,stop," << MetricHeaderSuffix("interval_ms")
      << ",dam_calc_calls,dam_calcdiff_calls\n";
  for (const IterationProfileRecord& record : records) {
    out << CsvEscape(record.robot) << ','
        << CsvEscape(record.backend) << ','
        << record.sample_id << ','
        << record.callback_index << ','
        << record.solver_iter << ','
        << FormatCsvNumber(record.elapsed_ms) << ','
        << FormatCsvNumber(record.iter_interval_ms) << ','
        << FormatCsvNumber(record.cost) << ','
        << FormatCsvNumber(record.stop);
    WriteMetricColumns(out, record.metrics, record.iter_interval_ms, 1.0);
    out << ',' << record.metrics.dam_calc_calls << ','
        << record.metrics.dam_calcdiff_calls << '\n';
  }
}

struct SummaryBucket {
  std::string robot;
  std::string backend;
  int dof = 0;
  std::vector<const RunRecord*> runs;
};

std::map<std::string, SummaryBucket> BuildSummaryBuckets(const std::vector<RunRecord>& runs) {
  std::map<std::string, SummaryBucket> buckets;
  for (const RunRecord& run : runs) {
    const std::string key = run.robot + "/" + run.backend;
    auto& bucket = buckets[key];
    bucket.robot = run.robot;
    bucket.backend = run.backend;
    bucket.dof = run.dof;
    bucket.runs.push_back(&run);
  }
  return buckets;
}

std::vector<double> CollectRunMetric(const SummaryBucket& bucket,
                                     const std::function<double(const RunRecord&)>& fn) {
  std::vector<double> values;
  values.reserve(bucket.runs.size());
  for (const RunRecord* run : bucket.runs) {
    if (!run->failed) {
      values.push_back(fn(*run));
    }
  }
  return values;
}

void WriteSummaryCsv(const std::filesystem::path& path, const std::vector<RunRecord>& runs,
                     const CliConfig& config) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to open " + path.string());
  }
  out << "robot,backend,dof,samples,valid_samples,horizon,dt,max_iterations,success_rate,"
      << "solver_iterations_mean,solve_total_mean_ms,solve_total_p95_ms,"
      << MetricHeaderSuffix("mean_ms") << ','
      << MetricHeaderSuffix("per_iter_mean_ms") << '\n';

  const auto buckets = BuildSummaryBuckets(runs);
  for (const auto& [_, bucket] : buckets) {
    const std::size_t valid = CollectRunMetric(bucket, [](const RunRecord& r) {
      return r.solve_total_ms;
    }).size();
    const std::size_t converged = static_cast<std::size_t>(std::count_if(
        bucket.runs.begin(), bucket.runs.end(), [](const RunRecord* run) {
          return !run->failed && run->converged;
        }));
    const std::vector<double> solve_times = CollectRunMetric(
        bucket, [](const RunRecord& r) { return r.solve_total_ms; });
    const std::vector<double> iterations = CollectRunMetric(bucket, [](const RunRecord& r) {
      return static_cast<double>(r.solver_iterations);
    });

    auto mean_metric = [&](const std::function<double(const RunRecord&)>& fn) {
      return Mean(CollectRunMetric(bucket, fn));
    };

    ProfileMetrics total_mean;
    total_mean.dam_calc_ms = mean_metric([](const RunRecord& r) { return r.metrics.dam_calc_ms; });
    total_mean.dam_calcdiff_ms =
        mean_metric([](const RunRecord& r) { return r.metrics.dam_calcdiff_ms; });
    total_mean.cost_state_calc_ms =
        mean_metric([](const RunRecord& r) { return r.metrics.cost_state_calc_ms; });
    total_mean.cost_state_calcdiff_ms =
        mean_metric([](const RunRecord& r) { return r.metrics.cost_state_calcdiff_ms; });
    total_mean.cost_control_calc_ms =
        mean_metric([](const RunRecord& r) { return r.metrics.cost_control_calc_ms; });
    total_mean.cost_control_calcdiff_ms =
        mean_metric([](const RunRecord& r) { return r.metrics.cost_control_calcdiff_ms; });
    total_mean.cost_acc_calc_ms =
        mean_metric([](const RunRecord& r) { return r.metrics.cost_acc_calc_ms; });
    total_mean.cost_acc_calcdiff_ms =
        mean_metric([](const RunRecord& r) { return r.metrics.cost_acc_calcdiff_ms; });
    total_mean.cost_collision_calc_ms =
        mean_metric([](const RunRecord& r) { return r.metrics.cost_collision_calc_ms; });
    total_mean.cost_collision_calcdiff_ms =
        mean_metric([](const RunRecord& r) { return r.metrics.cost_collision_calcdiff_ms; });
    total_mean.cost_other_calc_ms =
        mean_metric([](const RunRecord& r) { return r.metrics.cost_other_calc_ms; });
    total_mean.cost_other_calcdiff_ms =
        mean_metric([](const RunRecord& r) { return r.metrics.cost_other_calcdiff_ms; });

    ProfileMetrics per_iter_mean;
    per_iter_mean.dam_calc_ms = mean_metric([](const RunRecord& r) {
      return r.metrics.dam_calc_ms / static_cast<double>(std::max<std::size_t>(r.solver_iterations, 1));
    });
    per_iter_mean.dam_calcdiff_ms = mean_metric([](const RunRecord& r) {
      return r.metrics.dam_calcdiff_ms /
             static_cast<double>(std::max<std::size_t>(r.solver_iterations, 1));
    });
    per_iter_mean.cost_state_calc_ms = mean_metric([](const RunRecord& r) {
      return r.metrics.cost_state_calc_ms /
             static_cast<double>(std::max<std::size_t>(r.solver_iterations, 1));
    });
    per_iter_mean.cost_state_calcdiff_ms = mean_metric([](const RunRecord& r) {
      return r.metrics.cost_state_calcdiff_ms /
             static_cast<double>(std::max<std::size_t>(r.solver_iterations, 1));
    });
    per_iter_mean.cost_control_calc_ms = mean_metric([](const RunRecord& r) {
      return r.metrics.cost_control_calc_ms /
             static_cast<double>(std::max<std::size_t>(r.solver_iterations, 1));
    });
    per_iter_mean.cost_control_calcdiff_ms = mean_metric([](const RunRecord& r) {
      return r.metrics.cost_control_calcdiff_ms /
             static_cast<double>(std::max<std::size_t>(r.solver_iterations, 1));
    });
    per_iter_mean.cost_acc_calc_ms = mean_metric([](const RunRecord& r) {
      return r.metrics.cost_acc_calc_ms /
             static_cast<double>(std::max<std::size_t>(r.solver_iterations, 1));
    });
    per_iter_mean.cost_acc_calcdiff_ms = mean_metric([](const RunRecord& r) {
      return r.metrics.cost_acc_calcdiff_ms /
             static_cast<double>(std::max<std::size_t>(r.solver_iterations, 1));
    });
    per_iter_mean.cost_collision_calc_ms = mean_metric([](const RunRecord& r) {
      return r.metrics.cost_collision_calc_ms /
             static_cast<double>(std::max<std::size_t>(r.solver_iterations, 1));
    });
    per_iter_mean.cost_collision_calcdiff_ms = mean_metric([](const RunRecord& r) {
      return r.metrics.cost_collision_calcdiff_ms /
             static_cast<double>(std::max<std::size_t>(r.solver_iterations, 1));
    });
    per_iter_mean.cost_other_calc_ms = mean_metric([](const RunRecord& r) {
      return r.metrics.cost_other_calc_ms /
             static_cast<double>(std::max<std::size_t>(r.solver_iterations, 1));
    });
    per_iter_mean.cost_other_calcdiff_ms = mean_metric([](const RunRecord& r) {
      return r.metrics.cost_other_calcdiff_ms /
             static_cast<double>(std::max<std::size_t>(r.solver_iterations, 1));
    });

    const double solve_total_mean = Mean(solve_times);
    const double solve_per_iter_mean = mean_metric([](const RunRecord& r) {
      return r.solve_total_ms / static_cast<double>(std::max<std::size_t>(r.solver_iterations, 1));
    });

    out << CsvEscape(bucket.robot) << ','
        << CsvEscape(bucket.backend) << ','
        << bucket.dof << ','
        << bucket.runs.size() << ','
        << valid << ','
        << config.solver_config.horizon << ','
        << FormatCsvNumber(config.solver_config.dt) << ','
        << config.solver_config.max_iterations << ','
        << FormatCsvNumber(valid == 0 ? 0.0 : static_cast<double>(converged) / valid) << ','
        << FormatCsvNumber(Mean(iterations)) << ','
        << FormatCsvNumber(solve_total_mean) << ','
        << FormatCsvNumber(Percentile(solve_times, 0.95));
    WriteMetricColumns(out, total_mean, solve_total_mean, 1.0);
    WriteMetricColumns(out, per_iter_mean, solve_per_iter_mean, 1.0);
    out << '\n';
  }
}

void WriteStackSummaryCsv(const std::filesystem::path& path, const std::vector<RunRecord>& runs,
                          const CliConfig& config) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to open " + path.string());
  }
  out << "robot,backend,dof,samples,horizon,dt,max_iterations,success_rate,"
      << "solve_total_per_iter_mean_ms,non_cost_model_per_iter_mean_ms,"
      << "cost_state_per_iter_mean_ms,cost_control_per_iter_mean_ms,"
      << "cost_acc_per_iter_mean_ms,cost_collision_per_iter_mean_ms,"
      << "cost_other_per_iter_mean_ms,solver_overhead_per_iter_mean_ms,"
      << "stack_total_per_iter_mean_ms,dam_calc_per_iter_mean_ms,"
      << "dam_calcdiff_per_iter_mean_ms,dam_total_per_iter_mean_ms\n";

  const auto buckets = BuildSummaryBuckets(runs);
  for (const auto& [_, bucket] : buckets) {
    auto values = [&](const std::function<double(const RunRecord&)>& fn) {
      return CollectRunMetric(bucket, fn);
    };
    auto mean_value = [&](const std::function<double(const RunRecord&)>& fn) {
      return Mean(values(fn));
    };
    const std::size_t valid = values([](const RunRecord& r) { return r.solve_total_ms; }).size();
    const std::size_t converged = static_cast<std::size_t>(std::count_if(
        bucket.runs.begin(), bucket.runs.end(), [](const RunRecord* run) {
          return !run->failed && run->converged;
        }));
    auto per_iter = [](const RunRecord& r, const double total_ms) {
      return total_ms / static_cast<double>(std::max<std::size_t>(r.solver_iterations, 1));
    };
    const double solve = mean_value([&](const RunRecord& r) {
      return per_iter(r, r.solve_total_ms);
    });
    const double non_cost = mean_value([&](const RunRecord& r) {
      return per_iter(r, NonCostModelMs(r.metrics));
    });
    const double cost_state = mean_value([&](const RunRecord& r) {
      return per_iter(r, CostStateTotalMs(r.metrics));
    });
    const double cost_control = mean_value([&](const RunRecord& r) {
      return per_iter(r, CostControlTotalMs(r.metrics));
    });
    const double cost_acc = mean_value([&](const RunRecord& r) {
      return per_iter(r, CostAccTotalMs(r.metrics));
    });
    const double cost_collision = mean_value([&](const RunRecord& r) {
      return per_iter(r, CostCollisionTotalMs(r.metrics));
    });
    const double cost_other = mean_value([&](const RunRecord& r) {
      return per_iter(r, CostOtherTotalMs(r.metrics));
    });
    const double overhead = mean_value([&](const RunRecord& r) {
      return per_iter(r, SolverOverheadMs(r.solve_total_ms, r.metrics));
    });
    const double dam_calc = mean_value([&](const RunRecord& r) {
      return per_iter(r, r.metrics.dam_calc_ms);
    });
    const double dam_calcdiff = mean_value([&](const RunRecord& r) {
      return per_iter(r, r.metrics.dam_calcdiff_ms);
    });
    const double dam_total = dam_calc + dam_calcdiff;
    const double stack_total = non_cost + cost_state + cost_control + cost_acc +
                               cost_collision + cost_other + overhead;

    out << CsvEscape(bucket.robot) << ','
        << CsvEscape(bucket.backend) << ','
        << bucket.dof << ','
        << bucket.runs.size() << ','
        << config.solver_config.horizon << ','
        << FormatCsvNumber(config.solver_config.dt) << ','
        << config.solver_config.max_iterations << ','
        << FormatCsvNumber(valid == 0 ? 0.0 : static_cast<double>(converged) / valid) << ','
        << FormatCsvNumber(solve) << ','
        << FormatCsvNumber(non_cost) << ','
        << FormatCsvNumber(cost_state) << ','
        << FormatCsvNumber(cost_control) << ','
        << FormatCsvNumber(cost_acc) << ','
        << FormatCsvNumber(cost_collision) << ','
        << FormatCsvNumber(cost_other) << ','
        << FormatCsvNumber(overhead) << ','
        << FormatCsvNumber(stack_total) << ','
        << FormatCsvNumber(dam_calc) << ','
        << FormatCsvNumber(dam_calcdiff) << ','
        << FormatCsvNumber(dam_total) << '\n';
  }
}

void WriteMetadata(const std::filesystem::path& path, const CliConfig& config) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to open " + path.string());
  }
  out << "# Offline Runtime Breakdown\n\n"
      << "This run uses random point-to-point FDDP tasks, not closed-loop ROS/MuJoCo MPC.\n\n"
      << "- Robots: ";
  for (std::size_t i = 0; i < config.robots.size(); ++i) {
    if (i > 0) {
      out << ", ";
    }
    out << RobotName(config.robots[i]);
  }
  out << "\n- Backends: ";
  for (std::size_t i = 0; i < config.backends.size(); ++i) {
    if (i > 0) {
      out << ", ";
    }
    out << BackendName(config.backends[i]);
  }
  out << "\n- Samples: " << config.samples
      << "\n- Seed: " << config.seed
      << "\n- Horizon: " << config.solver_config.horizon
      << "\n- dt: " << config.solver_config.dt
      << "\n- Max iterations: " << config.solver_config.max_iterations
      << "\n- Position limit: " << config.solver_config.position_limit
      << "\n- Warmup excluded from CSV: " << (config.warmup ? "yes" : "no")
      << "\n\n"
      << "Cost terms are state regularization, acceleration regularization, and effort "
         "regularization. Collision terms are intentionally disabled for all robots.\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CliConfig config = ParseCli(argc, argv);
    const std::filesystem::path output_dir =
        config.output_dir.empty() ? DefaultOutputDir() : std::filesystem::path(config.output_dir);
    std::filesystem::create_directories(output_dir);

    std::vector<RunRecord> runs;
    std::vector<IterationProfileRecord> iteration_records;

    for (const RobotKind robot : config.robots) {
      RobotContext context = BuildRobotContext(robot);
      const std::uint32_t sample_seed =
          MixBenchmarkSeed(config.seed, 0xBDA7u, static_cast<std::uint32_t>(context.dof),
                           static_cast<std::uint32_t>(robot));
      const FDDPSampleBatch samples = MakeFDDPSamples(
          context.dof, config.samples, sample_seed, config.solver_config.position_limit);

      for (const BackendKind backend : config.backends) {
        std::cout << "[runtime-breakdown] robot=" << context.name
                  << " backend=" << BackendName(backend)
                  << " dof=" << context.dof << std::endl;
        if (config.warmup) {
          RunWarmup(context, backend, config, samples);
        }
        for (int sample_id = 0; sample_id < config.samples; ++sample_id) {
          runs.push_back(RunOneSample(
              context, backend, config, sample_id,
              samples.x0[static_cast<std::size_t>(sample_id)],
              samples.x_target[static_cast<std::size_t>(sample_id)],
              &iteration_records));
        }
      }
    }

    WriteRawCsv(output_dir / "runtime_breakdown_raw.csv", runs);
    WriteIterationsCsv(output_dir / "runtime_breakdown_iterations.csv", iteration_records);
    WriteSummaryCsv(output_dir / "runtime_breakdown_summary.csv", runs, config);
    WriteStackSummaryCsv(output_dir / "runtime_breakdown_stack_summary.csv", runs, config);
    WriteMetadata(output_dir / "README.md", config);

    std::cout << "[runtime-breakdown] wrote " << output_dir << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Crocoddyl_runtime_breakdown failed: " << e.what() << std::endl;
    return 1;
  }
}
