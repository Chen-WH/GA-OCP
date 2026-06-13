#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/string.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

#include <crocoddyl/core/activations/quadratic-barrier.hpp>
#include <crocoddyl/core/costs/cost-sum.hpp>
#include <crocoddyl/core/costs/residual.hpp>
#include <crocoddyl/core/integrator/euler.hpp>
#include <crocoddyl/core/residuals/control.hpp>
#include <crocoddyl/core/solvers/box-fddp.hpp>
#include <crocoddyl/core/states/euclidean.hpp>
#include <crocoddyl/multibody/actuations/full.hpp>
#include <crocoddyl/multibody/actions/free-fwddyn.hpp>
#include <crocoddyl/multibody/residuals/state.hpp>
#include <crocoddyl/multibody/states/multibody.hpp>

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/fwd.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/spatial/inertia.hpp>
#include <pinocchio/spatial/se3.hpp>

#include "ga_ocp/CrocoddylActions.hpp"
#include "ga_ocp/CrocoddylResiduals.hpp"
#include "TetraPGA/ModelRepo.hpp"

#ifdef GA_OCP_HAS_CASADI_BENCH
#include "ga_ocp/BenchUtils.hpp"
#endif

namespace {

using Clock = std::chrono::steady_clock;
using DurationSeconds = std::chrono::duration<double>;
constexpr double kTwoPi = 6.28318530717958647692;

enum class BackendKind {
  kTetraPGA,
  kPinocchio,
  kCasadi,
};

struct SolveCycleResult {
  std::vector<Eigen::VectorXd> best_xs;
  std::vector<Eigen::VectorXd> best_us;
  double best_cost = std::numeric_limits<double>::quiet_NaN();
  double final_stop = std::numeric_limits<double>::quiet_NaN();
  double solve_time_ms = 0.0;
  std::size_t iterations = 0;
  bool converged = false;
  bool failed = false;
  std::string failure_message;
};

struct CycleRecord {
  double t = 0.0;
  double tracking_error = 0.0;
  double velocity_error = 0.0;
  double torque_ratio = 0.0;
  double solve_time_ms = 0.0;
  double cycle_time_ms = 0.0;
  double realtime_ratio = 0.0;
  double best_cost = std::numeric_limits<double>::quiet_NaN();
  double final_stop = std::numeric_limits<double>::quiet_NaN();
  std::size_t iterations = 0;
  int converged = 0;
  int failed = 0;
  std::string failure_message;
  std::string q;
  std::string dq;
  std::string q_ref;
  std::string dq_ref;
  std::string q_cmd;
  std::string dq_cmd;
  std::string u_cmd;
  std::string effort;
};

struct RobotConfig {
  std::string robot;
  std::string urdf_path;
  std::string urdf_robot_name;
  std::vector<std::string> joint_names;
  Model<double> ga_model;
  Eigen::Vector3d payload_mount_translation = Eigen::Vector3d::Zero();
  Eigen::Quaterniond payload_mount_quaternion = Eigen::Quaterniond::Identity();
  std::string payload_parent_joint;
  Eigen::VectorXd default_amplitudes;
};

BackendKind ParseBackend(const std::string& value) {
  if (value == "tetrapga") {
    return BackendKind::kTetraPGA;
  }
  if (value == "pinocchio") {
    return BackendKind::kPinocchio;
  }
  if (value == "casadi") {
    return BackendKind::kCasadi;
  }
  throw std::invalid_argument("Unsupported backend: " + value);
}

std::string BackendName(const BackendKind backend) {
  switch (backend) {
    case BackendKind::kTetraPGA:
      return "tetrapga";
    case BackendKind::kPinocchio:
      return "pinocchio";
    case BackendKind::kCasadi:
      return "casadi";
  }
  return "unknown";
}

std::string CsvEscape(const std::string& value) {
  std::string out;
  out.reserve(value.size());
  for (const char c : value) {
    switch (c) {
      case '"':
        out.push_back('\'');
        break;
      case ',':
        out.push_back(';');
        break;
      case '\n':
      case '\r':
      case '\t':
        out.push_back(' ');
        break;
      default:
        out.push_back(c);
        break;
    }
  }
  return out;
}

std::string FormatCsvNumber(const double value) {
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

template <typename Derived>
std::string FormatVector(const Eigen::MatrixBase<Derived>& value) {
  std::ostringstream oss;
  oss << std::setprecision(17);
  for (Eigen::Index i = 0; i < value.size(); ++i) {
    if (i > 0) {
      oss << ' ';
    }
    oss << value[i];
  }
  return oss.str();
}

double Mean(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  return sum / static_cast<double>(values.size());
}

double Percentile(std::vector<double> values, const double p) {
  if (values.empty()) {
    return 0.0;
  }
  const double clamped = std::clamp(p, 0.0, 1.0);
  const std::size_t index =
      static_cast<std::size_t>(std::round(clamped * static_cast<double>(values.size() - 1)));
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(index), values.end());
  return values[index];
}

Eigen::Vector3d ParseVector3Param(const std::vector<double>& values, const std::string& name) {
  if (values.size() != 3) {
    throw std::invalid_argument(name + " must contain exactly 3 values");
  }
  return Eigen::Vector3d(values[0], values[1], values[2]);
}

Eigen::VectorXd ToEigenVector(const std::vector<double>& values) {
  Eigen::VectorXd out(static_cast<Eigen::Index>(values.size()));
  for (std::size_t i = 0; i < values.size(); ++i) {
    out[static_cast<Eigen::Index>(i)] = values[i];
  }
  return out;
}

Eigen::Matrix3d Skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d out;
  out << 0.0, -v.z(), v.y(),
         v.z(), 0.0, -v.x(),
         -v.y(), v.x(), 0.0;
  return out;
}

Eigen::Matrix<double, 6, 6> MakeGaSpatialInertia(const double mass,
                                                 const Eigen::Vector3d& com_in_body_frame) {
  Eigen::Matrix<double, 6, 6> inertia = Eigen::Matrix<double, 6, 6>::Zero();
  const Eigen::Matrix3d Rc = Skew(com_in_body_frame);
  const Eigen::Matrix3d J = -mass * Rc * Rc;
  inertia.block<3, 3>(0, 0) = -mass * Rc;
  inertia.block<3, 3>(0, 3) = mass * Eigen::Matrix3d::Identity();
  inertia.block<3, 3>(3, 0) = J;
  inertia.block<3, 3>(3, 3) = mass * Rc;
  return inertia;
}

void AddPayloadToGaModel(Model<double>& model, const double payload_mass,
                         const Eigen::Vector3d& payload_com_attachment,
                         const Eigen::Vector3d& mount_translation,
                         const Eigen::Quaterniond& mount_quaternion) {
  if (payload_mass <= 0.0 || model.n <= 1) {
    return;
  }
  const Eigen::Vector3d com_in_link =
      mount_translation + mount_quaternion.toRotationMatrix() * payload_com_attachment;
  model.I[model.n - 1] += MakeGaSpatialInertia(payload_mass, com_in_link);
}

void AddPayloadToPinModel(pinocchio::Model& model, const double payload_mass,
                          const Eigen::Vector3d& payload_com_attachment,
                          const Eigen::Vector3d& mount_translation,
                          const Eigen::Quaterniond& mount_quaternion,
                          const std::string& parent_joint_name) {
  if (payload_mass <= 0.0) {
    return;
  }
  const pinocchio::JointIndex joint_id = model.getJointId(parent_joint_name);
  if (joint_id == 0) {
    throw std::invalid_argument("Failed to find joint for payload modeling: " + parent_joint_name);
  }
  const pinocchio::SE3 body_placement(
      mount_quaternion.toRotationMatrix(), mount_translation);
  const pinocchio::Inertia payload_inertia(
      payload_mass, payload_com_attachment, pinocchio::Symmetric3::Zero());
  model.appendBodyToJoint(joint_id, payload_inertia, body_placement);
}

std::vector<std::string> MakeJointNames(std::initializer_list<const char*> names) {
  return std::vector<std::string>(names.begin(), names.end());
}

RobotConfig MakeRobotConfig(const std::string& robot) {
  const std::string share_dir = ament_index_cpp::get_package_share_directory("ga_ocp_core");
  if (robot == "ur") {
    RobotConfig config{
        robot,
        share_dir + "/robot-assets/ur10/urdf/ur10.urdf",
        "ur",
        MakeJointNames({
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        }),
        model_from_name("ur", share_dir + "/robot-assets/ur10/urdf/ur10.urdf"),
        Eigen::Vector3d(0.0, 0.0, 0.01),
        Eigen::Quaterniond(0.707107, 0.0, 0.0, -0.707107),
        "wrist_3_joint",
        Eigen::VectorXd::Constant(6, 0.35),
    };
    config.default_amplitudes <<
        0.35, 0.30, 0.32, 0.28, 0.24, 0.22;
    return config;
  }

  if (robot == "leap_left") {
    std::vector<std::string> joint_names;
    joint_names.reserve(16);
    for (int i = 0; i < 16; ++i) {
      joint_names.push_back(std::to_string(i));
    }
    RobotConfig config{
        robot,
        share_dir + "/robot-assets/leap_hand/urdf/leap_hand_left.urdf",
        "leap_left",
        joint_names,
        model_from_name("leap_left", share_dir + "/robot-assets/leap_hand/urdf/leap_hand_left.urdf"),
        Eigen::Vector3d::Zero(),
        Eigen::Quaterniond::Identity(),
        "",
        Eigen::VectorXd::Constant(16, 0.22),
    };
    config.default_amplitudes <<
        0.18, 0.18, 0.16, 0.16,
        0.18, 0.18, 0.16, 0.16,
        0.18, 0.18, 0.16, 0.16,
        0.14, 0.14, 0.18, 0.18;
    return config;
  }

  throw std::invalid_argument("Unsupported robot: " + robot);
}

bool ExtractJointVector(const sensor_msgs::msg::JointState& msg,
                        const std::vector<std::string>& joint_names,
                        const std::vector<double>& source,
                        Eigen::VectorXd& out) {
  if (static_cast<std::size_t>(out.size()) != joint_names.size()) {
    out.resize(static_cast<Eigen::Index>(joint_names.size()));
  }
  if (msg.name.empty()) {
    if (source.size() < joint_names.size()) {
      return false;
    }
    for (std::size_t i = 0; i < joint_names.size(); ++i) {
      out[static_cast<Eigen::Index>(i)] = source[i];
    }
    return true;
  }

  for (std::size_t joint_idx = 0; joint_idx < joint_names.size(); ++joint_idx) {
    const auto it = std::find(msg.name.begin(), msg.name.end(), joint_names[joint_idx]);
    if (it == msg.name.end()) {
      return false;
    }
    const std::size_t source_idx =
        static_cast<std::size_t>(std::distance(msg.name.begin(), it));
    if (source_idx >= source.size()) {
      return false;
    }
    out[static_cast<Eigen::Index>(joint_idx)] = source[source_idx];
  }
  return true;
}

double ComputeTorqueRatio(const Eigen::VectorXd& effort, const Eigen::VectorXd& effort_limit) {
  double ratio = 0.0;
  for (Eigen::Index i = 0; i < effort.size() && i < effort_limit.size(); ++i) {
    if (!std::isfinite(effort_limit[i]) || effort_limit[i] <= 1e-9) {
      continue;
    }
    ratio = std::max(ratio, std::abs(effort[i]) / effort_limit[i]);
  }
  return ratio;
}

std::filesystem::path DefaultOutputPrefix(const std::string& robot, const BackendKind backend) {
  const std::filesystem::path package_root =
      std::filesystem::path(__FILE__).parent_path().parent_path();
  const std::filesystem::path log_dir = package_root / "log";
  std::filesystem::create_directories(log_dir);
  return log_dir / ("closed_loop_mpc_" + robot + "_" + BackendName(backend));
}

}  // namespace

class ClosedLoopMpcNode : public rclcpp::Node {
 public:
  ClosedLoopMpcNode()
      : Node("closed_loop_mpc_node"),
        robot_config_(MakeRobotConfig(this->declare_parameter<std::string>("robot", "ur"))),
        backend_(ParseBackend(this->declare_parameter<std::string>("backend", "tetrapga"))),
        ga_model_(robot_config_.ga_model) {
    dt_ = this->declare_parameter<double>("dt", 0.02);
    horizon_ = this->declare_parameter<int>("horizon", 20);
    max_iterations_ = this->declare_parameter<int>("max_iterations", 25);
    solve_budget_ms_ = this->declare_parameter<double>("solve_budget_ms", 10.0);
    control_rate_hz_ = this->declare_parameter<double>("control_rate_hz", 50.0);
    experiment_duration_s_ = this->declare_parameter<double>("experiment_duration_s", 20.0);
    stop_tol_ = this->declare_parameter<double>("stop_tol", 1e-5);
    enforce_solve_budget_ = this->declare_parameter<bool>("enforce_solve_budget", true);
    use_warm_start_ = this->declare_parameter<bool>("use_warm_start", true);
    shutdown_on_finish_ = this->declare_parameter<bool>("shutdown_on_finish", true);
    auto_start_ = this->declare_parameter<bool>("auto_start", true);

    state_running_weight_ = this->declare_parameter<double>("state_running_weight", 6.0);
    state_terminal_weight_ = this->declare_parameter<double>("state_terminal_weight", 80.0);
    control_weight_ = this->declare_parameter<double>("control_weight", 1e-3);
    velocity_limit_weight_ = this->declare_parameter<double>("velocity_limit_weight", 20.0);
    velocity_limit_scale_ = this->declare_parameter<double>("velocity_limit_scale", 0.9);

    reference_frequency_hz_ = this->declare_parameter<double>("reference_frequency_hz", 0.12);
    reference_secondary_ratio_ = this->declare_parameter<double>("reference_secondary_ratio", 0.5);
    reference_amplitude_scale_ = this->declare_parameter<double>("reference_amplitude_scale", 1.0);
    reference_ramp_duration_s_ = this->declare_parameter<double>("reference_ramp_duration_s", 2.0);

    plant_mass_scale_ = this->declare_parameter<double>("plant_mass_scale", 1.0);
    plant_payload_mass_ = this->declare_parameter<double>("plant_payload_mass", 0.0);
    controller_payload_mass_ = this->declare_parameter<double>("controller_payload_mass", 0.0);
    model_payload_ = this->declare_parameter<bool>("model_payload", false);
    payload_com_attachment_ = ParseVector3Param(
        this->declare_parameter<std::vector<double>>("payload_com_attachment", {0.0, 0.0, 0.05}),
        "payload_com_attachment");

    std::vector<double> amplitude_override = this->declare_parameter<std::vector<double>>(
        "reference_amplitudes", std::vector<double>{});
    if (!amplitude_override.empty()) {
      if (static_cast<int>(amplitude_override.size()) != robot_config_.ga_model.dof_a) {
        throw std::invalid_argument("reference_amplitudes size must match robot dof");
      }
      reference_amplitudes_ = ToEigenVector(amplitude_override);
    } else {
      reference_amplitudes_ = robot_config_.default_amplitudes;
    }

    const std::string output_prefix =
        this->declare_parameter<std::string>("output_prefix", "");
    if (output_prefix.empty()) {
      output_prefix_ = DefaultOutputPrefix(robot_config_.robot, backend_);
    } else {
      output_prefix_ = output_prefix;
      if (output_prefix_.has_parent_path()) {
        std::filesystem::create_directories(output_prefix_.parent_path());
      }
    }

    effort_limit_ = ga_model_.effortLimit;
    velocity_soft_lb_ = -velocity_limit_scale_ * ga_model_.velocityLimit;
    velocity_soft_ub_ = velocity_limit_scale_ * ga_model_.velocityLimit;

    pinocchio::urdf::buildModel(robot_config_.urdf_path, pin_model_);

    if (model_payload_ && controller_payload_mass_ > 0.0 && robot_config_.robot == "ur") {
      AddPayloadToGaModel(ga_model_, controller_payload_mass_, payload_com_attachment_,
                          robot_config_.payload_mount_translation,
                          robot_config_.payload_mount_quaternion);
      AddPayloadToPinModel(pin_model_, controller_payload_mass_, payload_com_attachment_,
                           robot_config_.payload_mount_translation,
                           robot_config_.payload_mount_quaternion,
                           robot_config_.payload_parent_joint);
    }

#ifdef GA_OCP_HAS_CASADI_BENCH
    if (backend_ == BackendKind::kCasadi) {
      const std::string cache_tag =
          robot_config_.robot == "ur" ? "closed_loop_ur10" : "closed_loop_leap";
      casadi_autodiff_ = std::make_shared<InlineAutoDiffABADerivatives>(pin_model_, cache_tag);
    }
#else
    if (backend_ == BackendKind::kCasadi) {
      throw std::invalid_argument(
          "backend=casadi requested but GA_OCP_HAS_CASADI_BENCH is disabled at build time");
    }
#endif

    joint_pos_ = ga_model_.qa0;
    joint_vel_ = Eigen::VectorXd::Zero(ga_model_.dof_a);
    joint_effort_ = Eigen::VectorXd::Zero(ga_model_.dof_a);

    joint_cmd_pub_ =
        this->create_publisher<trajectory_msgs::msg::JointTrajectory>("/joint_commands", 20);
    status_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/planning_status", rclcpp::QoS(1).reliable().transient_local());
    joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", 20,
        std::bind(&ClosedLoopMpcNode::jointStateCallback, this, std::placeholders::_1));

    const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
        DurationSeconds(1.0 / std::max(control_rate_hz_, 1.0)));
    control_timer_ = this->create_wall_timer(
        period, std::bind(&ClosedLoopMpcNode::controlLoop, this));

    publishStatus("Status: waiting_for_joint_state");
    RCLCPP_INFO(this->get_logger(),
                "Closed-loop MPC node ready. robot=%s backend=%s budget=%.3f ms horizon=%d "
                "dt=%.3f control_rate=%.1fHz enforce_budget=%s output=%s",
                robot_config_.robot.c_str(), BackendName(backend_).c_str(), solve_budget_ms_,
                horizon_, dt_, control_rate_hz_, enforce_solve_budget_ ? "true" : "false",
                output_prefix_.string().c_str());
  }

 private:
  void publishStatus(const std::string& text) const {
    std_msgs::msg::String msg;
    msg.data = text;
    status_pub_->publish(msg);
  }

  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
    Eigen::VectorXd q(ga_model_.dof_a);
    Eigen::VectorXd dq(ga_model_.dof_a);
    Eigen::VectorXd effort(ga_model_.dof_a);

    if (!ExtractJointVector(*msg, robot_config_.joint_names, msg->position, q)) {
      return;
    }

    if (!ExtractJointVector(*msg, robot_config_.joint_names, msg->velocity, dq)) {
      dq.setZero();
    }
    if (!ExtractJointVector(*msg, robot_config_.joint_names, msg->effort, effort)) {
      effort.setZero();
    }

    joint_pos_ = q;
    joint_vel_ = dq;
    joint_effort_ = effort;
    has_joint_state_ = true;
  }

  Eigen::VectorXd referenceStateAt(const double t) const {
    const int dof = ga_model_.dof_a;
    Eigen::VectorXd x_ref = Eigen::VectorXd::Zero(2 * dof);
    const double ramp = reference_ramp_duration_s_ <= 1e-9
                            ? 1.0
                            : std::clamp(t / reference_ramp_duration_s_, 0.0, 1.0);

    for (int i = 0; i < dof; ++i) {
      const double base_amp = reference_amplitudes_[i] * reference_amplitude_scale_;
      const double lower_margin = std::max(0.0, ga_model_.qa0[i] - ga_model_.lowerPositionLimit[i]);
      const double upper_margin = std::max(0.0, ga_model_.upperPositionLimit[i] - ga_model_.qa0[i]);
      const double safe_amp = std::min(base_amp, 0.45 * std::min(lower_margin, upper_margin));
      const double omega_primary =
          kTwoPi * reference_frequency_hz_ * (1.0 + 0.04 * static_cast<double>(i % 5));
      const double omega_secondary =
          kTwoPi * reference_frequency_hz_ * reference_secondary_ratio_ *
          (1.0 + 0.03 * static_cast<double>((i + 2) % 7));
      const double phase_primary = 0.35 * static_cast<double>(i);
      const double phase_secondary = 0.21 * static_cast<double>(i + 1);
      const double q_delta =
          ramp * safe_amp *
          (0.7 * std::sin(omega_primary * t + phase_primary) +
           0.3 * std::sin(omega_secondary * t + phase_secondary));
      const double dq =
          ramp * safe_amp *
              (0.7 * omega_primary * std::cos(omega_primary * t + phase_primary) +
               0.3 * omega_secondary * std::cos(omega_secondary * t + phase_secondary)) +
          (ramp < 1.0 && reference_ramp_duration_s_ > 1e-9
               ? safe_amp *
                     (0.7 * std::sin(omega_primary * t + phase_primary) +
                      0.3 * std::sin(omega_secondary * t + phase_secondary)) /
                     reference_ramp_duration_s_
               : 0.0);
      x_ref[i] = std::clamp(ga_model_.qa0[i] + q_delta, ga_model_.lowerPositionLimit[i],
                            ga_model_.upperPositionLimit[i]);
      x_ref[dof + i] = dq;
    }

    return x_ref;
  }

  std::vector<Eigen::VectorXd> buildReferenceTrajectory(const double t0) const {
    std::vector<Eigen::VectorXd> refs(static_cast<std::size_t>(horizon_) + 1u);
    for (int k = 0; k <= horizon_; ++k) {
      refs[static_cast<std::size_t>(k)] = referenceStateAt(t0 + dt_ * static_cast<double>(k));
    }
    return refs;
  }

  void initializeWarmStart(const Eigen::VectorXd& x0,
                           const std::vector<Eigen::VectorXd>& x_refs,
                           std::vector<Eigen::VectorXd>& init_xs,
                           std::vector<Eigen::VectorXd>& init_us) const {
    const std::size_t expected_xs = static_cast<std::size_t>(horizon_) + 1u;
    const std::size_t expected_us = static_cast<std::size_t>(horizon_);
    init_xs.assign(expected_xs, x0);
    init_us.assign(expected_us, Eigen::VectorXd::Zero(ga_model_.dof_a));

    if (use_warm_start_ && last_best_xs_.size() == expected_xs && last_best_us_.size() == expected_us) {
      init_xs = last_best_xs_;
      init_us = last_best_us_;
      init_xs.front() = x0;
      for (std::size_t i = 1; i + 1 < init_xs.size(); ++i) {
        init_xs[i] = last_best_xs_[i + 1];
      }
      init_xs.back() = init_xs[init_xs.size() - 2];
      for (std::size_t i = 0; i + 1 < init_us.size(); ++i) {
        init_us[i] = last_best_us_[i + 1];
      }
      init_us.back().setZero();
      return;
    }

    for (std::size_t i = 1; i < init_xs.size(); ++i) {
      init_xs[i] = x_refs[i];
    }
  }

  std::shared_ptr<crocoddyl::ShootingProblem> buildGaProblem(
      const Eigen::VectorXd& x0, const std::vector<Eigen::VectorXd>& x_refs) const {
    auto state = std::make_shared<crocoddyl::StateVector>(2 * ga_model_.dof_a);
    std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> running_models;
    running_models.reserve(static_cast<std::size_t>(horizon_));

    const Eigen::VectorXd x_zero = Eigen::VectorXd::Zero(2 * ga_model_.dof_a);
    for (int t = 0; t < horizon_; ++t) {
      auto running_cost = std::make_shared<crocoddyl::CostModelSum>(state);
      auto state_residual =
          std::make_shared<crocoddyl::ResidualModelState>(state, x_refs[static_cast<std::size_t>(t)]);
      auto state_cost = std::make_shared<crocoddyl::CostModelResidual>(state, state_residual);
      auto control_residual =
          std::make_shared<crocoddyl::ResidualModelControl>(state, ga_model_.dof_a);
      auto control_cost =
          std::make_shared<crocoddyl::CostModelResidual>(state, control_residual);

      crocoddyl::ActivationBounds vel_bounds;
      vel_bounds.lb = Eigen::VectorXd::Constant(
          2 * ga_model_.dof_a, -std::numeric_limits<double>::infinity());
      vel_bounds.ub = Eigen::VectorXd::Constant(
          2 * ga_model_.dof_a, std::numeric_limits<double>::infinity());
      for (int i = 0; i < ga_model_.dof_a; ++i) {
        vel_bounds.lb[ga_model_.dof_a + i] = velocity_soft_lb_[i];
        vel_bounds.ub[ga_model_.dof_a + i] = velocity_soft_ub_[i];
      }
      auto vel_residual =
          std::make_shared<crocoddyl::ResidualModelState>(state, x_zero, ga_model_.dof_a);
      auto vel_activation =
          std::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(vel_bounds);
      auto vel_cost =
          std::make_shared<crocoddyl::CostModelResidual>(state, vel_activation, vel_residual);

      running_cost->addCost("state_reg", state_cost, state_running_weight_);
      running_cost->addCost("control_reg", control_cost, control_weight_);
      running_cost->addCost("vel_limit", vel_cost, velocity_limit_weight_);

      auto diff_model =
          std::make_shared<DifferentialActionModelTetraPGAForwardDynamics<double>>(state, ga_model_, running_cost);
      diff_model->set_u_lb(-effort_limit_);
      diff_model->set_u_ub(effort_limit_);
      running_models.push_back(
          std::make_shared<crocoddyl::IntegratedActionModelEuler>(diff_model, dt_));
    }

    auto terminal_cost = std::make_shared<crocoddyl::CostModelSum>(state);
    auto terminal_state_residual =
        std::make_shared<crocoddyl::ResidualModelState>(state, x_refs.back());
    auto terminal_state_cost =
        std::make_shared<crocoddyl::CostModelResidual>(state, terminal_state_residual);
    terminal_cost->addCost("state_reg", terminal_state_cost, state_terminal_weight_);
    auto terminal_diff =
        std::make_shared<DifferentialActionModelTetraPGAForwardDynamics<double>>(state, ga_model_, terminal_cost);
    terminal_diff->set_u_lb(-effort_limit_);
    terminal_diff->set_u_ub(effort_limit_);
    auto terminal_model =
        std::make_shared<crocoddyl::IntegratedActionModelEuler>(terminal_diff, dt_);
    return std::make_shared<crocoddyl::ShootingProblem>(x0, running_models, terminal_model);
  }

  std::shared_ptr<crocoddyl::ShootingProblem> buildPinProblem(
      const Eigen::VectorXd& x0, const std::vector<Eigen::VectorXd>& x_refs) const {
    auto state =
        std::make_shared<crocoddyl::StateMultibody>(std::make_shared<pinocchio::Model>(pin_model_));
    auto actuation = std::make_shared<crocoddyl::ActuationModelFull>(state);
    actuation->set_u_lb(-effort_limit_);
    actuation->set_u_ub(effort_limit_);

    std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> running_models;
    running_models.reserve(static_cast<std::size_t>(horizon_));

    const Eigen::VectorXd x_zero = Eigen::VectorXd::Zero(2 * ga_model_.dof_a);
    for (int t = 0; t < horizon_; ++t) {
      auto running_cost = std::make_shared<crocoddyl::CostModelSum>(state);
      auto state_residual =
          std::make_shared<crocoddyl::ResidualModelState>(state, x_refs[static_cast<std::size_t>(t)]);
      auto state_cost = std::make_shared<crocoddyl::CostModelResidual>(state, state_residual);
      auto control_residual =
          std::make_shared<crocoddyl::ResidualModelControl>(state, ga_model_.dof_a);
      auto control_cost =
          std::make_shared<crocoddyl::CostModelResidual>(state, control_residual);

      crocoddyl::ActivationBounds vel_bounds;
      vel_bounds.lb = Eigen::VectorXd::Constant(
          2 * ga_model_.dof_a, -std::numeric_limits<double>::infinity());
      vel_bounds.ub = Eigen::VectorXd::Constant(
          2 * ga_model_.dof_a, std::numeric_limits<double>::infinity());
      for (int i = 0; i < ga_model_.dof_a; ++i) {
        vel_bounds.lb[ga_model_.dof_a + i] = velocity_soft_lb_[i];
        vel_bounds.ub[ga_model_.dof_a + i] = velocity_soft_ub_[i];
      }
      auto vel_residual =
          std::make_shared<crocoddyl::ResidualModelState>(state, x_zero, ga_model_.dof_a);
      auto vel_activation =
          std::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(vel_bounds);
      auto vel_cost =
          std::make_shared<crocoddyl::CostModelResidual>(state, vel_activation, vel_residual);

      running_cost->addCost("state_reg", state_cost, state_running_weight_);
      running_cost->addCost("control_reg", control_cost, control_weight_);
      running_cost->addCost("vel_limit", vel_cost, velocity_limit_weight_);

      auto diff_model = std::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(
          state, actuation, running_cost);
      running_models.push_back(
          std::make_shared<crocoddyl::IntegratedActionModelEuler>(diff_model, dt_));
    }

    auto terminal_cost = std::make_shared<crocoddyl::CostModelSum>(state);
    auto terminal_state_residual =
        std::make_shared<crocoddyl::ResidualModelState>(state, x_refs.back());
    auto terminal_state_cost =
        std::make_shared<crocoddyl::CostModelResidual>(state, terminal_state_residual);
    terminal_cost->addCost("state_reg", terminal_state_cost, state_terminal_weight_);
    auto terminal_diff = std::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(
        state, actuation, terminal_cost);
    auto terminal_model =
        std::make_shared<crocoddyl::IntegratedActionModelEuler>(terminal_diff, dt_);
    return std::make_shared<crocoddyl::ShootingProblem>(x0, running_models, terminal_model);
  }

#ifdef GA_OCP_HAS_CASADI_BENCH
  std::shared_ptr<crocoddyl::ShootingProblem> buildCasadiProblem(
      const Eigen::VectorXd& x0, const std::vector<Eigen::VectorXd>& x_refs) const {
    auto state =
        std::make_shared<crocoddyl::StateMultibody>(std::make_shared<pinocchio::Model>(pin_model_));
    std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> running_models;
    running_models.reserve(static_cast<std::size_t>(horizon_));

    const Eigen::VectorXd x_zero = Eigen::VectorXd::Zero(2 * ga_model_.dof_a);
    for (int t = 0; t < horizon_; ++t) {
      auto running_cost = std::make_shared<crocoddyl::CostModelSum>(state);
      auto state_residual =
          std::make_shared<crocoddyl::ResidualModelState>(state, x_refs[static_cast<std::size_t>(t)]);
      auto state_cost = std::make_shared<crocoddyl::CostModelResidual>(state, state_residual);
      auto control_residual =
          std::make_shared<crocoddyl::ResidualModelControl>(state, ga_model_.dof_a);
      auto control_cost =
          std::make_shared<crocoddyl::CostModelResidual>(state, control_residual);

      crocoddyl::ActivationBounds vel_bounds;
      vel_bounds.lb = Eigen::VectorXd::Constant(
          2 * ga_model_.dof_a, -std::numeric_limits<double>::infinity());
      vel_bounds.ub = Eigen::VectorXd::Constant(
          2 * ga_model_.dof_a, std::numeric_limits<double>::infinity());
      for (int i = 0; i < ga_model_.dof_a; ++i) {
        vel_bounds.lb[ga_model_.dof_a + i] = velocity_soft_lb_[i];
        vel_bounds.ub[ga_model_.dof_a + i] = velocity_soft_ub_[i];
      }
      auto vel_residual =
          std::make_shared<crocoddyl::ResidualModelState>(state, x_zero, ga_model_.dof_a);
      auto vel_activation =
          std::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(vel_bounds);
      auto vel_cost =
          std::make_shared<crocoddyl::CostModelResidual>(state, vel_activation, vel_residual);

      running_cost->addCost("state_reg", state_cost, state_running_weight_);
      running_cost->addCost("control_reg", control_cost, control_weight_);
      running_cost->addCost("vel_limit", vel_cost, velocity_limit_weight_);

      auto diff_model = std::make_shared<DifferentialActionModelPinocchioCasadi>(
          state, pin_model_, running_cost, casadi_autodiff_);
      diff_model->set_u_lb(-effort_limit_);
      diff_model->set_u_ub(effort_limit_);
      running_models.push_back(
          std::make_shared<crocoddyl::IntegratedActionModelEuler>(diff_model, dt_));
    }

    auto terminal_cost = std::make_shared<crocoddyl::CostModelSum>(state);
    auto terminal_state_residual =
        std::make_shared<crocoddyl::ResidualModelState>(state, x_refs.back());
    auto terminal_state_cost =
        std::make_shared<crocoddyl::CostModelResidual>(state, terminal_state_residual);
    terminal_cost->addCost("state_reg", terminal_state_cost, state_terminal_weight_);
    auto terminal_diff = std::make_shared<DifferentialActionModelPinocchioCasadi>(
        state, pin_model_, terminal_cost, casadi_autodiff_);
    terminal_diff->set_u_lb(-effort_limit_);
    terminal_diff->set_u_ub(effort_limit_);
    auto terminal_model =
        std::make_shared<crocoddyl::IntegratedActionModelEuler>(terminal_diff, dt_);
    return std::make_shared<crocoddyl::ShootingProblem>(x0, running_models, terminal_model);
  }
#endif

  SolveCycleResult solveCycle(const Eigen::VectorXd& x0,
                              const std::vector<Eigen::VectorXd>& x_refs) {
    SolveCycleResult result;

    std::shared_ptr<crocoddyl::ShootingProblem> problem;
    switch (backend_) {
      case BackendKind::kTetraPGA:
        problem = buildGaProblem(x0, x_refs);
        break;
      case BackendKind::kPinocchio:
        problem = buildPinProblem(x0, x_refs);
        break;
      case BackendKind::kCasadi:
#ifdef GA_OCP_HAS_CASADI_BENCH
        problem = buildCasadiProblem(x0, x_refs);
        break;
#else
        throw std::runtime_error("CasADi backend not compiled");
#endif
    }

    std::vector<Eigen::VectorXd> init_xs;
    std::vector<Eigen::VectorXd> init_us;
    initializeWarmStart(x0, x_refs, init_xs, init_us);

    result.best_xs = init_xs;
    result.best_us = init_us;
    result.best_cost = problem->calc(init_xs, init_us);

    crocoddyl::SolverBoxFDDP solver(problem);
    solver.set_th_stop(enforce_solve_budget_ ? std::numeric_limits<double>::min() : stop_tol_);

    bool is_feasible = false;
    const Clock::time_point start_time = Clock::now();
    while (result.iterations < static_cast<std::size_t>(max_iterations_)) {
      const double elapsed_ms = DurationSeconds(Clock::now() - start_time).count() * 1e3;
      if (elapsed_ms >= solve_budget_ms_ && result.iterations > 0u) {
        break;
      }

      try {
        const bool iteration_converged = solver.solve(init_xs, init_us, 1, is_feasible);
        result.converged = result.converged || iteration_converged;
      } catch (const std::exception& e) {
        result.failed = true;
        result.failure_message = e.what();
        break;
      } catch (...) {
        result.failed = true;
        result.failure_message = "unknown exception";
        break;
      }

      ++result.iterations;
      result.final_stop = solver.get_stop();
      result.converged =
          result.converged ||
          (std::isfinite(result.final_stop) && result.final_stop <= stop_tol_);
      init_xs = solver.get_xs();
      init_us = solver.get_us();
      is_feasible = solver.get_is_feasible();

      if (init_xs.size() == static_cast<std::size_t>(horizon_) + 1u &&
          init_us.size() == static_cast<std::size_t>(horizon_)) {
        const double current_cost = solver.get_cost();
        if (std::isfinite(current_cost) && current_cost <= result.best_cost) {
          result.best_cost = current_cost;
          result.best_xs = init_xs;
          result.best_us = init_us;
        }
      }

      if (!enforce_solve_budget_ && result.converged) {
        break;
      }
    }

    result.solve_time_ms = DurationSeconds(Clock::now() - start_time).count() * 1e3;
    return result;
  }

  void publishCommand(const Eigen::VectorXd& q_cmd, const Eigen::VectorXd& dq_cmd,
                      const Eigen::VectorXd& u_cmd) {
    trajectory_msgs::msg::JointTrajectory traj;
    traj.header.stamp = this->now();
    traj.joint_names = robot_config_.joint_names;

    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions.resize(static_cast<std::size_t>(q_cmd.size()));
    point.velocities.resize(static_cast<std::size_t>(dq_cmd.size()));
    point.effort.resize(static_cast<std::size_t>(u_cmd.size()));
    for (Eigen::Index i = 0; i < q_cmd.size(); ++i) {
      point.positions[static_cast<std::size_t>(i)] = q_cmd[i];
      point.velocities[static_cast<std::size_t>(i)] = dq_cmd[i];
      point.effort[static_cast<std::size_t>(i)] = u_cmd[i];
    }
    point.time_from_start =
        rclcpp::Duration::from_seconds(1.0 / std::max(control_rate_hz_, 1.0));
    traj.points.push_back(std::move(point));
    joint_cmd_pub_->publish(traj);
  }

  void controlLoop() {
    if (finished_) {
      return;
    }
    if (!has_joint_state_) {
      return;
    }

    if (!started_) {
      if (!auto_start_) {
        return;
      }
      started_ = true;
      start_time_ = Clock::now();
      publishStatus("Status: running");
      RCLCPP_INFO(this->get_logger(), "Closed-loop experiment started.");
    }

    const Clock::time_point cycle_start = Clock::now();
    const double t = DurationSeconds(cycle_start - start_time_).count();
    if (t > experiment_duration_s_) {
      finishExperiment();
      return;
    }

    Eigen::VectorXd x0(2 * ga_model_.dof_a);
    x0.head(ga_model_.dof_a) = joint_pos_;
    x0.tail(ga_model_.dof_a) = joint_vel_;

    const Eigen::VectorXd x_ref_now = referenceStateAt(t);
    const std::vector<Eigen::VectorXd> x_refs = buildReferenceTrajectory(t);
    SolveCycleResult solve = solveCycle(x0, x_refs);

    Eigen::VectorXd q_cmd = x_refs[std::min<std::size_t>(1u, x_refs.size() - 1u)].head(ga_model_.dof_a);
    Eigen::VectorXd dq_cmd =
        x_refs[std::min<std::size_t>(1u, x_refs.size() - 1u)].tail(ga_model_.dof_a);
    Eigen::VectorXd u_cmd = Eigen::VectorXd::Zero(ga_model_.dof_a);
    if (solve.best_xs.size() >= 2u) {
      q_cmd = solve.best_xs[1].head(ga_model_.dof_a);
      dq_cmd = solve.best_xs[1].tail(ga_model_.dof_a);
    }
    if (!solve.best_us.empty()) {
      u_cmd = solve.best_us.front();
    }
    publishCommand(q_cmd, dq_cmd, u_cmd);
    last_best_xs_ = solve.best_xs;
    last_best_us_ = solve.best_us;

    CycleRecord record;
    record.t = t;
    record.tracking_error = (joint_pos_ - x_ref_now.head(ga_model_.dof_a)).norm();
    record.velocity_error = (joint_vel_ - x_ref_now.tail(ga_model_.dof_a)).norm();
    record.torque_ratio = ComputeTorqueRatio(joint_effort_, effort_limit_);
    record.solve_time_ms = solve.solve_time_ms;
    record.cycle_time_ms = DurationSeconds(Clock::now() - cycle_start).count() * 1e3;
    record.realtime_ratio =
        record.solve_time_ms / (1e3 / std::max(control_rate_hz_, 1.0));
    record.best_cost = solve.best_cost;
    record.final_stop = solve.final_stop;
    record.iterations = solve.iterations;
    record.converged = solve.converged ? 1 : 0;
    record.failed = solve.failed ? 1 : 0;
    record.failure_message = solve.failure_message;
    record.q = FormatVector(joint_pos_);
    record.dq = FormatVector(joint_vel_);
    record.q_ref = FormatVector(x_ref_now.head(ga_model_.dof_a));
    record.dq_ref = FormatVector(x_ref_now.tail(ga_model_.dof_a));
    record.q_cmd = FormatVector(q_cmd);
    record.dq_cmd = FormatVector(dq_cmd);
    record.u_cmd = FormatVector(u_cmd);
    record.effort = FormatVector(joint_effort_);
    cycle_records_.push_back(std::move(record));

    if (t + 1.0 / std::max(control_rate_hz_, 1.0) >= experiment_duration_s_) {
      finishExperiment();
    }
  }

  void writeCycleCsv() const {
    std::ofstream out(output_prefix_.string() + "_cycles.csv");
    out << "robot,backend,t,tracking_error,velocity_error,torque_ratio,solve_time_ms,"
           "cycle_time_ms,realtime_ratio,iterations,converged,failed,best_cost,final_stop,"
           "plant_mass_scale,plant_payload_mass,controller_payload_mass,model_payload,"
           "failure_message,q,dq,q_ref,dq_ref,q_cmd,dq_cmd,u_cmd,effort\n";
    for (const CycleRecord& record : cycle_records_) {
      out << CsvEscape(robot_config_.robot) << ','
          << CsvEscape(BackendName(backend_)) << ','
          << FormatCsvNumber(record.t) << ','
          << FormatCsvNumber(record.tracking_error) << ','
          << FormatCsvNumber(record.velocity_error) << ','
          << FormatCsvNumber(record.torque_ratio) << ','
          << FormatCsvNumber(record.solve_time_ms) << ','
          << FormatCsvNumber(record.cycle_time_ms) << ','
          << FormatCsvNumber(record.realtime_ratio) << ','
          << record.iterations << ','
          << record.converged << ','
          << record.failed << ','
          << FormatCsvNumber(record.best_cost) << ','
          << FormatCsvNumber(record.final_stop) << ','
          << FormatCsvNumber(plant_mass_scale_) << ','
          << FormatCsvNumber(plant_payload_mass_) << ','
          << FormatCsvNumber(controller_payload_mass_) << ','
          << (model_payload_ ? 1 : 0) << ','
          << CsvEscape(record.failure_message) << ','
          << CsvEscape(record.q) << ','
          << CsvEscape(record.dq) << ','
          << CsvEscape(record.q_ref) << ','
          << CsvEscape(record.dq_ref) << ','
          << CsvEscape(record.q_cmd) << ','
          << CsvEscape(record.dq_cmd) << ','
          << CsvEscape(record.u_cmd) << ','
          << CsvEscape(record.effort) << '\n';
    }
  }

  void writeSummaryCsv() const {
    std::vector<double> tracking_errors;
    std::vector<double> torque_ratios;
    std::vector<double> solve_times;
    std::vector<double> realtime_ratios;
    std::size_t deadline_miss_count = 0;
    std::size_t failure_count = 0;

    tracking_errors.reserve(cycle_records_.size());
    torque_ratios.reserve(cycle_records_.size());
    solve_times.reserve(cycle_records_.size());
    realtime_ratios.reserve(cycle_records_.size());

    for (const CycleRecord& record : cycle_records_) {
      tracking_errors.push_back(record.tracking_error);
      torque_ratios.push_back(record.torque_ratio);
      solve_times.push_back(record.solve_time_ms);
      realtime_ratios.push_back(record.realtime_ratio);
      deadline_miss_count += record.solve_time_ms > (1e3 / std::max(control_rate_hz_, 1.0)) ? 1u : 0u;
      failure_count += static_cast<std::size_t>(record.failed);
    }

    const double tracking_rmse =
        tracking_errors.empty()
            ? 0.0
            : std::sqrt(std::inner_product(tracking_errors.begin(), tracking_errors.end(),
                                           tracking_errors.begin(), 0.0) /
                        static_cast<double>(tracking_errors.size()));

    std::ofstream out(output_prefix_.string() + "_summary.csv");
    out << "robot,backend,num_cycles,tracking_rmse,tracking_mean,tracking_p95,torque_ratio_mean,"
           "torque_ratio_p95,torque_ratio_max,solve_time_mean_ms,solve_time_p95_ms,"
           "realtime_ratio_mean,deadline_miss_rate,failure_rate,dt,horizon,solve_budget_ms,"
           "control_rate_hz,experiment_duration_s,plant_mass_scale,plant_payload_mass,"
           "controller_payload_mass,model_payload,payload_com_attachment\n";
    out << CsvEscape(robot_config_.robot) << ','
        << CsvEscape(BackendName(backend_)) << ','
        << cycle_records_.size() << ','
        << FormatCsvNumber(tracking_rmse) << ','
        << FormatCsvNumber(Mean(tracking_errors)) << ','
        << FormatCsvNumber(Percentile(tracking_errors, 0.95)) << ','
        << FormatCsvNumber(Mean(torque_ratios)) << ','
        << FormatCsvNumber(Percentile(torque_ratios, 0.95)) << ','
        << FormatCsvNumber(torque_ratios.empty()
                               ? 0.0
                               : *std::max_element(torque_ratios.begin(), torque_ratios.end()))
        << ','
        << FormatCsvNumber(Mean(solve_times)) << ','
        << FormatCsvNumber(Percentile(solve_times, 0.95)) << ','
        << FormatCsvNumber(Mean(realtime_ratios)) << ','
        << FormatCsvNumber(cycle_records_.empty()
                               ? 0.0
                               : static_cast<double>(deadline_miss_count) /
                                     static_cast<double>(cycle_records_.size()))
        << ','
        << FormatCsvNumber(cycle_records_.empty()
                               ? 0.0
                               : static_cast<double>(failure_count) /
                                     static_cast<double>(cycle_records_.size()))
        << ','
        << FormatCsvNumber(dt_) << ','
        << horizon_ << ','
        << FormatCsvNumber(solve_budget_ms_) << ','
        << FormatCsvNumber(control_rate_hz_) << ','
        << FormatCsvNumber(experiment_duration_s_) << ','
        << FormatCsvNumber(plant_mass_scale_) << ','
        << FormatCsvNumber(plant_payload_mass_) << ','
        << FormatCsvNumber(controller_payload_mass_) << ','
        << (model_payload_ ? 1 : 0) << ','
        << CsvEscape(FormatVector(payload_com_attachment_)) << '\n';
  }

  void finishExperiment() {
    if (finished_) {
      return;
    }
    finished_ = true;
    control_timer_->cancel();
    publishStatus("Status: finished");
    writeCycleCsv();
    writeSummaryCsv();
    RCLCPP_INFO(this->get_logger(),
                "Closed-loop experiment finished. cycles=%zu summary=%s cycles=%s",
                cycle_records_.size(),
                (output_prefix_.string() + "_summary.csv").c_str(),
                (output_prefix_.string() + "_cycles.csv").c_str());
    if (shutdown_on_finish_) {
      rclcpp::shutdown();
    }
  }

  RobotConfig robot_config_;
  BackendKind backend_;
  Model<double> ga_model_;
  pinocchio::Model pin_model_;
#ifdef GA_OCP_HAS_CASADI_BENCH
  std::shared_ptr<InlineAutoDiffABADerivatives> casadi_autodiff_;
#endif

  double dt_{0.02};
  int horizon_{20};
  int max_iterations_{25};
  double solve_budget_ms_{10.0};
  double control_rate_hz_{50.0};
  double experiment_duration_s_{20.0};
  double stop_tol_{1e-5};
  bool enforce_solve_budget_{true};
  bool use_warm_start_{true};
  bool shutdown_on_finish_{true};
  bool auto_start_{true};

  double state_running_weight_{6.0};
  double state_terminal_weight_{80.0};
  double control_weight_{1e-3};
  double velocity_limit_weight_{20.0};
  double velocity_limit_scale_{0.9};

  double reference_frequency_hz_{0.12};
  double reference_secondary_ratio_{0.5};
  double reference_amplitude_scale_{1.0};
  double reference_ramp_duration_s_{2.0};

  double plant_mass_scale_{1.0};
  double plant_payload_mass_{0.0};
  double controller_payload_mass_{0.0};
  bool model_payload_{false};
  Eigen::Vector3d payload_com_attachment_{Eigen::Vector3d::Zero()};

  Eigen::VectorXd reference_amplitudes_;
  Eigen::VectorXd effort_limit_;
  Eigen::VectorXd velocity_soft_lb_;
  Eigen::VectorXd velocity_soft_ub_;

  Eigen::VectorXd joint_pos_;
  Eigen::VectorXd joint_vel_;
  Eigen::VectorXd joint_effort_;
  bool has_joint_state_{false};
  bool started_{false};
  bool finished_{false};
  Clock::time_point start_time_{};
  std::vector<Eigen::VectorXd> last_best_xs_;
  std::vector<Eigen::VectorXd> last_best_us_;
  std::vector<CycleRecord> cycle_records_;
  std::filesystem::path output_prefix_;

  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr joint_cmd_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::TimerBase::SharedPtr control_timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ClosedLoopMpcNode>());
  rclcpp::shutdown();
  return 0;
}
