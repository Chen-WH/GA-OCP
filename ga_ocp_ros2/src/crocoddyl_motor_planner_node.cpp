#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <initializer_list>
#include <memory>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <crocoddyl/core/activations/quadratic-barrier.hpp>
#include <crocoddyl/core/costs/cost-sum.hpp>
#include <crocoddyl/core/costs/residual.hpp>
#include <crocoddyl/core/integrator/euler.hpp>
#include <crocoddyl/core/solvers/fddp.hpp>
#include <crocoddyl/core/states/euclidean.hpp>
#include <crocoddyl/multibody/residuals/state.hpp>

#include "ga_ocp/CrocoddylActions.hpp"
#include "ga_ocp/CrocoddylResiduals.hpp"
#include "TetraPGA/Kinematics.hpp"
#include "TetraPGA/ModelRepo.hpp"

class CrocoddylMotorPlannerNode : public rclcpp::Node {
public:
  struct RobotConfig {
    std::string robot;
    std::string urdf_path;
    std::vector<std::string> joint_names;
    Model<double> model;

    RobotConfig(std::string robot_name, std::string urdf, std::vector<std::string> joints, Model<double> robot_model)
        : robot(std::move(robot_name)),
          urdf_path(std::move(urdf)),
          joint_names(std::move(joints)),
          model(std::move(robot_model)) {}
  };

  CrocoddylMotorPlannerNode()
      : Node("crocoddyl_motor_planner_node"),
        robot_config_(makeRobotConfig(this->declare_parameter<std::string>("robot", "ur"))),
        model_(robot_config_.model),
        state_(std::static_pointer_cast<crocoddyl::StateAbstract>(
            std::make_shared<crocoddyl::StateVector>(2 * model_.dof_a))),
        joint_names_(robot_config_.joint_names) {
    dt_ = this->declare_parameter<double>("dt", 0.008);
    horizon_ = this->declare_parameter<int>("horizon", 100);
    max_iterations_ = this->declare_parameter<int>("max_iterations", 100);
    use_warm_start_ = this->declare_parameter<bool>("use_warm_start", true);
    running_motor_weight_ = this->declare_parameter<double>("running_motor_weight", 10.0);
    terminal_motor_weight_ = this->declare_parameter<double>("terminal_motor_weight", 100.0);
    velocity_limit_weight_ = this->declare_parameter<double>("velocity_limit_weight", 100.0);
    velocity_limit_scale_ = this->declare_parameter<double>("velocity_limit_scale", 1.0);
    velocity_soft_lb_ = -velocity_limit_scale_ * model_.velocityLimit;
    velocity_soft_ub_ = velocity_limit_scale_ * model_.velocityLimit;

    joint_pos_.resize(model_.dof_a);
    joint_vel_.resize(model_.dof_a);
    joint_pos_ = model_.qa0;
    joint_vel_.setZero();

    joint_cmd_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>("/joint_commands", 10);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/ga_ocp/motor_plan_preview", rclcpp::QoS(1).reliable().transient_local());

    joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", 20,
        std::bind(&CrocoddylMotorPlannerNode::jointStateCallback, this, std::placeholders::_1));

    target_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/target_ee_pose", 10,
        std::bind(&CrocoddylMotorPlannerNode::targetPoseCallback, this, std::placeholders::_1));

    execute_sub_ = this->create_subscription<std_msgs::msg::Empty>(
        "/execute_planned_trajectory", 10,
        std::bind(&CrocoddylMotorPlannerNode::executePlannedTrajectoryTopic, this, std::placeholders::_1));
    plan_srv_ = this->create_service<std_srvs::srv::Trigger>(
        "/plan_trajectory_srv",
        std::bind(&CrocoddylMotorPlannerNode::handlePlanService, this,
                  std::placeholders::_1, std::placeholders::_2));
    execute_srv_ = this->create_service<std_srvs::srv::Trigger>(
        "/execute_planned_trajectory_srv",
        std::bind(&CrocoddylMotorPlannerNode::handleExecuteService, this,
                  std::placeholders::_1, std::placeholders::_2));
    clear_srv_ = this->create_service<std_srvs::srv::Trigger>(
        "/clear_planned_trajectory_srv",
        std::bind(&CrocoddylMotorPlannerNode::handleClearService, this,
                  std::placeholders::_1, std::placeholders::_2));
    status_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/planning_status", rclcpp::QoS(1).reliable().transient_local());

    visualization_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(1000),
        std::bind(&CrocoddylMotorPlannerNode::publishLatestPreview, this));

    publishPreview();
    updateStatus("Status: idle");
    RCLCPP_INFO(this->get_logger(),
                "Planner ready for robot='%s' with %d DoF. target topic sets goal pose, services handle "
                "plan/execute/clear.",
                robot_config_.robot.c_str(), model_.dof_a);
  }

private:
  static std::vector<std::string> makeJointNames(std::initializer_list<const char*> names) {
    return std::vector<std::string>(names.begin(), names.end());
  }

  static RobotConfig makeRobotConfig(const std::string& robot_name) {
    const std::string ga_ocp_core_share = ament_index_cpp::get_package_share_directory("ga_ocp_core");

    if (robot_name == "ur") {
      const std::string urdf_path = ga_ocp_core_share + "/robot-assets/ur10/urdf/ur10.urdf";
      std::vector<std::string> joint_names = makeJointNames({
          "shoulder_pan_joint",
          "shoulder_lift_joint",
          "elbow_joint",
          "wrist_1_joint",
          "wrist_2_joint",
          "wrist_3_joint",
      });
      return RobotConfig(robot_name, urdf_path, joint_names, model_from_name(robot_name, urdf_path));
    }

    if (robot_name == "franka") {
      const std::string urdf_path = ga_ocp_core_share + "/robot-assets/franka_panda/urdf/panda.urdf";
      std::vector<std::string> joint_names = makeJointNames({
          "panda_joint1",
          "panda_joint2",
          "panda_joint3",
          "panda_joint4",
          "panda_joint5",
          "panda_joint6",
          "panda_joint7",
      });
      return RobotConfig(robot_name, urdf_path, joint_names, model_from_name(robot_name, urdf_path));
    }

    throw std::invalid_argument(
        "Unsupported robot parameter for crocoddyl_motor_planner_node: " + robot_name +
        ". Supported robots are: ur, franka");
  }

  static std::string vectorToString(const Eigen::VectorXd& vec, const int precision = 3) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << "[";
    for (Eigen::Index i = 0; i < vec.size(); ++i) {
      if (i > 0) {
        oss << ", ";
      }
      oss << vec[i];
    }
    oss << "]";
    return oss.str();
  }

  static double maxAbsVelocity(const std::vector<Eigen::VectorXd>& xs, const int dof) {
    double max_abs_vel = 0.0;
    for (const auto& x : xs) {
      for (int i = 0; i < dof; ++i) {
        max_abs_vel = std::max(max_abs_vel, std::abs(x[dof + i]));
      }
    }
    return max_abs_vel;
  }

  static Motor3D<double> motorFromPose(const geometry_msgs::msg::Pose& pose) {
    const double tx = pose.position.x;
    const double ty = pose.position.y;
    const double tz = pose.position.z;

    const double qw = pose.orientation.w;
    const double qx = pose.orientation.x;
    const double qy = pose.orientation.y;
    const double qz = pose.orientation.z;

    Motor3D<double> M;
    M << qw,
        qx,
        qy,
        qz,
        0.5 * (tx * qw + ty * qz - tz * qy),
        0.5 * (ty * qw - tx * qz + tz * qx),
        0.5 * (tz * qw + tx * qy - ty * qx),
        0.5 * (tx * qx + ty * qy + tz * qz);
    return M;
  }

  geometry_msgs::msg::Point pointFromMotor(const Motor3D<double>& motor) const {
    const Point3D<double> origin(0.0, 0.0, 0.0, 1.0);
    const Point3D<double> point = pga_rbm3(motor, origin);
    geometry_msgs::msg::Point msg;
    msg.x = point.x();
    msg.y = point.y();
    msg.z = point.z();
    return msg;
  }

  void publishPreview(const std::vector<Eigen::VectorXd>* xs = nullptr,
                      const geometry_msgs::msg::PoseStamped* target_msg = nullptr) {
    visualization_msgs::msg::MarkerArray array;
    const auto stamp = this->now();

    visualization_msgs::msg::Marker path_marker;
    path_marker.header.frame_id = "base_link";
    path_marker.header.stamp = stamp;
    path_marker.ns = "planned_trajectory";
    path_marker.id = 0;
    path_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    path_marker.action = visualization_msgs::msg::Marker::ADD;
    path_marker.pose.orientation.w = 1.0;
    path_marker.scale.x = 0.01;
    path_marker.color.r = 1.0f;
    path_marker.color.g = 0.6f;
    path_marker.color.b = 0.1f;
    path_marker.color.a = 0.95f;

    visualization_msgs::msg::Marker waypoint_marker;
    waypoint_marker.header.frame_id = "base_link";
    waypoint_marker.header.stamp = stamp;
    waypoint_marker.ns = "planned_waypoints";
    waypoint_marker.id = 1;
    waypoint_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    waypoint_marker.action = visualization_msgs::msg::Marker::ADD;
    waypoint_marker.pose.orientation.w = 1.0;
    waypoint_marker.scale.x = 0.025;
    waypoint_marker.scale.y = 0.025;
    waypoint_marker.scale.z = 0.025;
    waypoint_marker.color.r = 1.0f;
    waypoint_marker.color.g = 0.7f;
    waypoint_marker.color.b = 0.1f;
    waypoint_marker.color.a = 0.9f;

    if (xs != nullptr) {
      for (const auto& x : *xs) {
        Data<double> data(model_);
        forwardKinematics(model_, data, x.head(model_.dof_a));
        const auto point = pointFromMotor(data.M.col(model_.n - 1));
        path_marker.points.push_back(point);
        waypoint_marker.points.push_back(point);
      }
    }
    array.markers.push_back(path_marker);
    array.markers.push_back(waypoint_marker);

    visualization_msgs::msg::Marker target_marker;
    target_marker.header.frame_id = target_msg != nullptr ? target_msg->header.frame_id : "base_link";
    target_marker.header.stamp = stamp;
    target_marker.ns = "planned_target";
    target_marker.id = 2;
    target_marker.type = visualization_msgs::msg::Marker::SPHERE;
    target_marker.action = target_msg != nullptr ? visualization_msgs::msg::Marker::ADD
                                                 : visualization_msgs::msg::Marker::DELETE;
    target_marker.pose.orientation.w = 1.0;
    target_marker.scale.x = 0.06;
    target_marker.scale.y = 0.06;
    target_marker.scale.z = 0.06;
    target_marker.color.r = 0.1f;
    target_marker.color.g = 0.9f;
    target_marker.color.b = 0.9f;
    target_marker.color.a = 0.85f;
    if (target_msg != nullptr) {
      target_marker.pose = target_msg->pose;
    }
    array.markers.push_back(target_marker);

    marker_pub_->publish(array);
  }

  void publishLatestPreview() {
    publishPreview(last_xs_.empty() ? nullptr : &last_xs_,
                   last_target_pose_.has_value() ? &last_target_pose_.value() : nullptr);
  }

  void updateStatus(const std::string& text) {
    std_msgs::msg::String msg;
    msg.data = text;
    status_pub_->publish(msg);
  }

  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
    if (msg->position.size() < static_cast<size_t>(model_.dof_a)) {
      return;
    }

    for (size_t joint_idx = 0; joint_idx < joint_names_.size(); ++joint_idx) {
      const auto it = std::find(msg->name.begin(), msg->name.end(), joint_names_[joint_idx]);
      if (it == msg->name.end()) {
        continue;
      }
      const auto src_idx = static_cast<size_t>(std::distance(msg->name.begin(), it));
      if (src_idx < msg->position.size()) {
        joint_pos_(static_cast<Eigen::Index>(joint_idx)) = msg->position[src_idx];
      }
      if (src_idx < msg->velocity.size()) {
        joint_vel_(static_cast<Eigen::Index>(joint_idx)) = msg->velocity[src_idx];
      }
    }
    has_joint_state_ = true;
  }

  void targetPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    last_target_pose_ = *msg;
    planned_traj_.points.clear();
    last_xs_.clear();
    last_us_.clear();
    has_pending_plan_ = false;
    publishPreview(nullptr, msg.get());
    updateStatus("Status: goal_set");
  }

  bool planTrajectory() {
    if (solving_) {
      RCLCPP_WARN(this->get_logger(), "Planner is already solving.");
      return false;
    }
    if (!last_target_pose_.has_value()) {
      RCLCPP_WARN(this->get_logger(), "No target pose available for planning.");
      updateStatus("Status: failed (no goal)");
      return false;
    }

    solving_ = true;
    updateStatus("Status: planning");
    const auto start = std::chrono::steady_clock::now();
    const auto& target_msg = last_target_pose_.value();
    const Motor3D<double> M_ref_raw = motorFromPose(target_msg.pose);

    Eigen::VectorXd x0(2 * model_.dof_a);
    x0.head(model_.dof_a) = joint_pos_;
    x0.tail(model_.dof_a) = has_joint_state_ ? joint_vel_ : Eigen::VectorXd::Zero(model_.dof_a);

    for (int i = 0; i < model_.dof_a; ++i) {
      const double q = x0[i];
      const double dq = x0[model_.dof_a + i];
      if (!std::isfinite(q)) {
        x0[i] = model_.qa0[i];
      }
      if (!std::isfinite(dq)) {
        x0[model_.dof_a + i] = 0.0;
      }
      x0[model_.dof_a + i] = std::clamp(x0[model_.dof_a + i], -1.0, 1.0);
      x0[i] = std::clamp(x0[i], model_.lowerPositionLimit[i], model_.upperPositionLimit[i]);
    }

    Data<double> initial_data(model_);
    forwardKinematics(model_, initial_data, x0.head(model_.dof_a));
    const Motor3D<double> M_ref = align_motor_hemisphere(
        M_ref_raw, initial_data.M.col(model_.n - 1));

    RCLCPP_INFO(this->get_logger(),
                "Planning request: target_p=(%.3f,%.3f,%.3f) target_q=(%.4f,%.4f,%.4f,%.4f) "
                "x0_q=%s x0_dq=%s target_motor=%s warm_start=%s",
                target_msg.pose.position.x, target_msg.pose.position.y, target_msg.pose.position.z,
                target_msg.pose.orientation.w, target_msg.pose.orientation.x,
                target_msg.pose.orientation.y, target_msg.pose.orientation.z,
                vectorToString(x0.head(model_.dof_a)).c_str(),
                vectorToString(x0.tail(model_.dof_a)).c_str(),
                vectorToString(M_ref).c_str(),
                (last_xs_.size() == static_cast<size_t>(horizon_) + 1 &&
                 last_us_.size() == static_cast<size_t>(horizon_))
                    ? "true"
                    : "false");

    auto running_cost_model = std::make_shared<crocoddyl::CostModelSum>(state_);
    auto terminal_cost_model = std::make_shared<crocoddyl::CostModelSum>(state_);

    auto motor_residual =
        std::make_shared<ResidualModelTetraPGAFramePlacement<double>>(state_, model_, M_ref);
    auto motor_cost = std::make_shared<crocoddyl::CostModelResidual>(state_, motor_residual);

    Eigen::VectorXd x_zero = Eigen::VectorXd::Zero(2 * model_.dof_a);
    auto vel_residual = std::make_shared<crocoddyl::ResidualModelState>(state_, x_zero, model_.dof_a);
    crocoddyl::ActivationBounds vel_bounds;
    vel_bounds.lb =
        Eigen::VectorXd::Constant(2 * model_.dof_a, -std::numeric_limits<double>::infinity());
    vel_bounds.ub =
        Eigen::VectorXd::Constant(2 * model_.dof_a, std::numeric_limits<double>::infinity());
    for (int i = 0; i < model_.dof_a; ++i) {
      vel_bounds.lb[model_.dof_a + i] = velocity_soft_lb_[i];
      vel_bounds.ub[model_.dof_a + i] = velocity_soft_ub_[i];
    }
    auto vel_activation =
        std::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(vel_bounds);
    auto vel_limit_cost =
        std::make_shared<crocoddyl::CostModelResidual>(state_, vel_activation, vel_residual);

    running_cost_model->addCost("motor_reg", motor_cost, running_motor_weight_);
    running_cost_model->addCost("vel_limit", vel_limit_cost, velocity_limit_weight_);
    terminal_cost_model->addCost("motor_reg", motor_cost, terminal_motor_weight_);
    terminal_cost_model->addCost("vel_limit", vel_limit_cost, velocity_limit_weight_);

    auto diff_model = std::make_shared<DifferentialActionModelTetraPGAForwardDynamics<double>>(state_, model_, running_cost_model);
    auto diff_model_term = std::make_shared<DifferentialActionModelTetraPGAForwardDynamics<double>>(state_, model_, terminal_cost_model);

    auto running_iam = std::make_shared<crocoddyl::IntegratedActionModelEuler>(diff_model, dt_);
    auto terminal_iam = std::make_shared<crocoddyl::IntegratedActionModelEuler>(diff_model_term, dt_);

    std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> running_models(
        static_cast<size_t>(horizon_), running_iam);

    auto problem = std::make_shared<crocoddyl::ShootingProblem>(x0, running_models, terminal_iam);
    crocoddyl::SolverFDDP solver(problem);
    solver.set_th_stop(1e-6);

    std::vector<Eigen::VectorXd> init_xs;
    std::vector<Eigen::VectorXd> init_us;

    const size_t expected_xs = static_cast<size_t>(horizon_) + 1;
    const size_t expected_us = static_cast<size_t>(horizon_);
    if (use_warm_start_ && last_xs_.size() == expected_xs && last_us_.size() == expected_us) {
      init_xs = last_xs_;
      init_us = last_us_;
      init_xs.front() = x0;
      for (size_t i = 1; i + 1 < init_xs.size(); ++i) {
        init_xs[i] = last_xs_[i + 1];
      }
      init_xs.back() = init_xs[init_xs.size() - 2];
      for (size_t i = 0; i + 1 < init_us.size(); ++i) {
        init_us[i] = last_us_[i + 1];
      }
      init_us.back().setZero();
    } else {
      init_xs.assign(expected_xs, x0);
      init_us.assign(expected_us, Eigen::VectorXd::Zero(model_.dof_a));
    }

    bool solved = solver.solve(init_xs, init_us, static_cast<std::size_t>(max_iterations_));

    if (!solved || solver.get_xs().empty()) {
      Eigen::VectorXd x0_retry = x0;
      x0_retry.tail(model_.dof_a).setZero();
      RCLCPP_WARN(this->get_logger(),
                  "First solve failed, retry with zero initial velocity. stop=%.3e cost=%.6f",
                  solver.get_stop(), solver.get_cost());
      problem->set_x0(x0_retry);
      solved = solver.solve({}, {}, static_cast<std::size_t>(max_iterations_ * 2));
    }

    if (!solved || solver.get_xs().empty()) {
      solving_ = false;
      RCLCPP_ERROR(this->get_logger(),
                   "FDDP failed. stop=%.3e cost=%.6f x0_q=%s x0_dq=%s target_motor=%s "
                   "target_p=(%.3f,%.3f,%.3f)",
                   solver.get_stop(), solver.get_cost(),
                   vectorToString(x0.head(model_.dof_a)).c_str(),
                   vectorToString(x0.tail(model_.dof_a)).c_str(),
                   vectorToString(M_ref).c_str(),
                   target_msg.pose.position.x, target_msg.pose.position.y, target_msg.pose.position.z);
      publishPreview(nullptr, &target_msg);
      updateStatus("Status: failed");
      return false;
    }
    last_xs_ = solver.get_xs();
    last_us_ = solver.get_us();
    has_pending_plan_ = true;
    planned_traj_ = buildTrajectoryFromXs(last_xs_);
    publishPreview(&last_xs_, &target_msg);

    const Eigen::VectorXd& x_final = solver.get_xs().back();
    Data<double> final_data(model_);
    forwardKinematics(model_, final_data, x_final.head(model_.dof_a));
    const Motor3D<double> M_final = final_data.M.col(model_.n - 1);
    const Motor3D<double> motor_error = M_final - M_ref;
    const Line3D<double> placement_error = ga_log(ga_mul(ga_rev(M_ref), M_final));
    const double max_abs_vel = maxAbsVelocity(solver.get_xs(), model_.dof_a);

    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - start)
                                .count();
    RCLCPP_INFO(this->get_logger(),
                "New plan ready for review: points=%zu cost=%.6f stop=%.3e solve_time=%ld ms "
                "max_abs_vel=%.3f final_q=%s final_dq=%s motor_error_norm=%.3e placement_error_norm=%.3e",
                planned_traj_.points.size(), solver.get_cost(), solver.get_stop(), elapsed_ms, max_abs_vel,
                vectorToString(x_final.head(model_.dof_a)).c_str(),
                vectorToString(x_final.tail(model_.dof_a)).c_str(),
                motor_error.norm(), placement_error.norm());

    updateStatus("Status: planned");
    solving_ = false;
    return true;
  }

  trajectory_msgs::msg::JointTrajectory buildTrajectoryFromXs(const std::vector<Eigen::VectorXd>& xs) const {
    trajectory_msgs::msg::JointTrajectory traj;
    traj.header.stamp = this->now();
    traj.joint_names.assign(joint_names_.begin(), joint_names_.end());
    traj.points.reserve(xs.size());

    for (size_t i = 0; i < xs.size(); ++i) {
      const Eigen::VectorXd& x = xs[i];
      trajectory_msgs::msg::JointTrajectoryPoint point;
      point.positions.resize(static_cast<size_t>(model_.dof_a));
      point.velocities.resize(static_cast<size_t>(model_.dof_a));
      for (int j = 0; j < model_.dof_a; ++j) {
        point.positions[static_cast<size_t>(j)] = x[j];
        point.velocities[static_cast<size_t>(j)] = x[model_.dof_a + j];
      }
      point.time_from_start = rclcpp::Duration::from_seconds(dt_ * static_cast<double>(i));
      traj.points.push_back(std::move(point));
    }
    return traj;
  }

  bool executePlannedTrajectory() {
    if (solving_) {
      RCLCPP_WARN(this->get_logger(), "Planner is still solving, execute request ignored.");
      return false;
    }
    if (!has_pending_plan_ || planned_traj_.points.empty()) {
      RCLCPP_WARN(this->get_logger(), "No planned trajectory waiting for execution.");
      return false;
    }

    planned_traj_.header.stamp = this->now();
    updateStatus("Status: executing");
    joint_cmd_pub_->publish(planned_traj_);
    has_pending_plan_ = false;
    RCLCPP_INFO(this->get_logger(), "Published cached trajectory to /joint_commands with %zu points.",
                planned_traj_.points.size());
    updateStatus("Status: idle");
    return true;
  }

  void executePlannedTrajectoryTopic(const std_msgs::msg::Empty::SharedPtr) {
    (void)executePlannedTrajectory();
  }

  void handlePlanService(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                         std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    response->success = planTrajectory();
    response->message = response->success ? "Plan ready for preview." : "Planning failed.";
  }

  void handleExecuteService(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                            std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    response->success = executePlannedTrajectory();
    response->message = response->success ? "Executed cached trajectory." : "No executable cached trajectory.";
  }

  void handleClearService(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                          std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    planned_traj_.points.clear();
    last_xs_.clear();
    last_us_.clear();
    has_pending_plan_ = false;
    publishPreview(nullptr, last_target_pose_.has_value() ? &last_target_pose_.value() : nullptr);
    updateStatus(last_target_pose_.has_value() ? "Status: goal_set" : "Status: idle");
    response->success = true;
    response->message = "Cleared cached trajectory preview.";
    RCLCPP_INFO(this->get_logger(), "Cleared cached trajectory preview.");
  }

private:
  RobotConfig robot_config_;
  Model<double> model_;
  std::shared_ptr<crocoddyl::StateAbstract> state_;

  double dt_{};
  int horizon_{};
  int max_iterations_{};
  bool use_warm_start_{true};
  double running_motor_weight_{};
  double terminal_motor_weight_{};
  double velocity_limit_weight_{};
  double velocity_limit_scale_{};
  Eigen::VectorXd velocity_soft_lb_;
  Eigen::VectorXd velocity_soft_ub_;

  Eigen::VectorXd joint_pos_;
  Eigen::VectorXd joint_vel_;
  bool has_joint_state_{false};
  bool solving_{false};
  bool has_pending_plan_{false};
  std::vector<Eigen::VectorXd> last_xs_;
  std::vector<Eigen::VectorXd> last_us_;
  trajectory_msgs::msg::JointTrajectory planned_traj_;
  std::optional<geometry_msgs::msg::PoseStamped> last_target_pose_;

  std::vector<std::string> joint_names_;

  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr joint_cmd_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_pose_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr execute_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr plan_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr execute_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr clear_srv_;
  rclcpp::TimerBase::SharedPtr visualization_timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CrocoddylMotorPlannerNode>());
  rclcpp::shutdown();
  return 0;
}
