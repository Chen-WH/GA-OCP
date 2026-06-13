#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Geometry>
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

#include "TetraPGA/Collision.hpp"
#include "ga_ocp/CrocoddylActions.hpp"
#include "ga_ocp/CrocoddylResiduals.hpp"
#include "TetraPGA/Kinematics.hpp"
#include "TetraPGA/ModelRepo.hpp"
#include "TetraPGA/PGA.hpp"

class CrocoddylCollisionPlannerNode : public rclcpp::Node {
public:
  CrocoddylCollisionPlannerNode()
      : Node("crocoddyl_collision_planner_node"),
        model_(ur()),
        state_(std::static_pointer_cast<crocoddyl::StateAbstract>(
            std::make_shared<crocoddyl::StateVector>(2 * model_.dof_a))),
        joint_names_({"shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
                      "wrist_2_joint", "wrist_3_joint"}) {
    dt_ = this->declare_parameter<double>("dt", 0.008);
    horizon_ = this->declare_parameter<int>("horizon", 100);
    max_iterations_ = this->declare_parameter<int>("max_iterations", 100);
    placement_running_weight_ = this->declare_parameter<double>("placement_running_weight", 10.0);
    placement_terminal_weight_ = this->declare_parameter<double>("placement_terminal_weight", 100.0);
    velocity_limit_weight_ = this->declare_parameter<double>("velocity_limit_weight", 100.0);
    collision_weight_ = this->declare_parameter<double>("collision_weight", 100.0);
    safety_distance_ = this->declare_parameter<double>("safety_distance", 0.1);
    publish_visualization_ = this->declare_parameter<bool>("publish_visualization", true);
    marker_frame_id_ = this->declare_parameter<std::string>("marker_frame_id", "base_link");
    velocity_limit_scale_ = this->declare_parameter<double>("velocity_limit_scale", 1.0);
    velocity_soft_lb_ = -velocity_limit_scale_ * model_.velocityLimit;
    velocity_soft_ub_ = velocity_limit_scale_ * model_.velocityLimit;

    const auto obstacle_radii =
        this->declare_parameter<std::vector<double>>("obstacle_radii", {0.15, 0.12, 0.18});
    const auto obstacle_centers = this->declare_parameter<std::vector<double>>(
        "obstacle_centers", {0.3, 0.5, 0.9, -0.2, 0.4, 1.0, 0.5, -0.3, 0.8});
    env_ = createEnvironment(obstacle_radii, obstacle_centers);

    joint_pos_.resize(model_.dof_a);
    joint_vel_.resize(model_.dof_a);
    joint_pos_ = model_.qa0;
    joint_vel_.setZero();

    joint_cmd_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>("/joint_commands", 10);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/ga_ocp/collision_scene", rclcpp::QoS(1).reliable().transient_local());

    joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", 20,
        std::bind(&CrocoddylCollisionPlannerNode::jointStateCallback, this, std::placeholders::_1));

    target_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/target_ee_pose", 10,
        std::bind(&CrocoddylCollisionPlannerNode::targetPoseCallback, this, std::placeholders::_1));
    execute_sub_ = this->create_subscription<std_msgs::msg::Empty>(
        "/execute_planned_trajectory", 10,
        std::bind(&CrocoddylCollisionPlannerNode::executePlannedTrajectoryTopic, this, std::placeholders::_1));
    plan_srv_ = this->create_service<std_srvs::srv::Trigger>(
        "/plan_trajectory_srv",
        std::bind(&CrocoddylCollisionPlannerNode::handlePlanService, this,
                  std::placeholders::_1, std::placeholders::_2));
    execute_srv_ = this->create_service<std_srvs::srv::Trigger>(
        "/execute_planned_trajectory_srv",
        std::bind(&CrocoddylCollisionPlannerNode::handleExecuteService, this,
                  std::placeholders::_1, std::placeholders::_2));
    clear_srv_ = this->create_service<std_srvs::srv::Trigger>(
        "/clear_planned_trajectory_srv",
        std::bind(&CrocoddylCollisionPlannerNode::handleClearService, this,
                  std::placeholders::_1, std::placeholders::_2));
    status_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/planning_status", rclcpp::QoS(1).reliable().transient_local());

    visualization_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(1000),
        std::bind(&CrocoddylCollisionPlannerNode::publishLatestVisualizations, this));

    publishVisualizations();
    updateStatus("Status: idle");
    RCLCPP_INFO(this->get_logger(),
                "Collision planner ready: target topic sets goal pose, services handle plan/execute/clear. "
                "obstacles=%d",
                env_.num_static_sphere);
  }

private:
  struct SolveResetGuard {
    explicit SolveResetGuard(bool& flag) : flag_(flag) { flag_ = true; }
    ~SolveResetGuard() { flag_ = false; }
    bool& flag_;
  };

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

  Environment<double> createEnvironment(const std::vector<double>& radii,
                                        const std::vector<double>& centers) const {
    if (radii.empty()) {
      RCLCPP_WARN(this->get_logger(), "Empty obstacle_radii received, collision planner will run without obstacles.");
      return Environment<double>({});
    }
    if (centers.size() != radii.size() * 3) {
      RCLCPP_WARN(this->get_logger(),
                  "Invalid obstacle_centers length (%zu). Expected %zu. Falling back to default collision scene.",
                  centers.size(), radii.size() * 3);
      std::vector<SSP<double>> fallback(3);
      fallback[0].id = 0;
      fallback[0].radius = 0.15;
      fallback[0].center = Point3D<double>(0.3, 0.5, 0.9, 1.0);
      fallback[1].id = 1;
      fallback[1].radius = 0.12;
      fallback[1].center = Point3D<double>(-0.2, 0.4, 1.0, 1.0);
      fallback[2].id = 2;
      fallback[2].radius = 0.18;
      fallback[2].center = Point3D<double>(0.5, -0.3, 0.8, 1.0);
      return Environment<double>(fallback);
    }

    std::vector<SSP<double>> obstacles;
    obstacles.reserve(radii.size());
    for (size_t i = 0; i < radii.size(); ++i) {
      SSP<double> obs;
      obs.id = static_cast<int>(i);
      obs.radius = radii[i];
      obs.center = Point3D<double>(centers[3 * i], centers[3 * i + 1], centers[3 * i + 2], 1.0);
      obstacles.push_back(obs);
    }
    return Environment<double>(obstacles);
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

  double computeMinDistance(const Eigen::VectorXd& q) const {
    if (env_.num_static_sphere == 0) {
      return std::numeric_limits<double>::infinity();
    }

    Data<double> data(model_);
    EnvironmentData<double> env_data(model_, env_);
    forwardKinematics(model_, data, q);
    computeDistance(model_, data, env_, env_data);

    double min_distance = std::numeric_limits<double>::infinity();
    for (int i = 0; i < env_data.num_collision_pair; ++i) {
      min_distance = std::min(min_distance, env_data.distance[i]);
    }
    return min_distance;
  }

  double computeTrajectoryMinDistance(const std::vector<Eigen::VectorXd>& xs) const {
    if (env_.num_static_sphere == 0 || xs.empty()) {
      return std::numeric_limits<double>::infinity();
    }

    double min_distance = std::numeric_limits<double>::infinity();
    for (const auto& x : xs) {
      min_distance = std::min(min_distance, computeMinDistance(x.head(model_.dof_a)));
    }
    return min_distance;
  }

  geometry_msgs::msg::Point pointFromEigen(const Point3D<double>& point) const {
    geometry_msgs::msg::Point msg;
    msg.x = point(0);
    msg.y = point(1);
    msg.z = point(2);
    return msg;
  }

  void publishVisualizations(const std::vector<Eigen::VectorXd>* xs = nullptr,
                             const geometry_msgs::msg::PoseStamped* target_msg = nullptr) {
    if (!publish_visualization_) {
      return;
    }

    visualization_msgs::msg::MarkerArray array;
    const auto stamp = this->now();

    for (int i = 0; i < env_.num_static_sphere; ++i) {
      visualization_msgs::msg::Marker marker;
      marker.header.frame_id = marker_frame_id_;
      marker.header.stamp = stamp;
      marker.ns = "obstacles";
      marker.id = i;
      marker.type = visualization_msgs::msg::Marker::SPHERE;
      marker.action = visualization_msgs::msg::Marker::ADD;
      marker.pose.position = pointFromEigen(env_.static_sphere[i].center);
      marker.pose.orientation.w = 1.0;
      marker.scale.x = 2.0 * env_.static_sphere[i].radius;
      marker.scale.y = 2.0 * env_.static_sphere[i].radius;
      marker.scale.z = 2.0 * env_.static_sphere[i].radius;
      marker.color.a = 0.45f;
      marker.color.r = (i == 0) ? 1.0f : 0.2f;
      marker.color.g = (i == 1) ? 1.0f : 0.2f;
      marker.color.b = (i == 2) ? 1.0f : 0.4f;
      array.markers.push_back(marker);
    }

    visualization_msgs::msg::Marker path_marker;
    path_marker.header.frame_id = marker_frame_id_;
    path_marker.header.stamp = stamp;
    path_marker.ns = "trajectory";
    path_marker.id = 100;
    path_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    path_marker.action = visualization_msgs::msg::Marker::ADD;
    path_marker.pose.orientation.w = 1.0;
    path_marker.scale.x = 0.01;
    path_marker.color.r = 1.0f;
    path_marker.color.g = 0.8f;
    path_marker.color.b = 0.1f;
    path_marker.color.a = 0.95f;

    visualization_msgs::msg::Marker waypoint_marker;
    waypoint_marker.header.frame_id = marker_frame_id_;
    waypoint_marker.header.stamp = stamp;
    waypoint_marker.ns = "trajectory_waypoints";
    waypoint_marker.id = 101;
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
      const Point3D<double> ee_origin(0.0, 0.0, 0.0, 1.0);
      for (const auto& x : *xs) {
        Data<double> data(model_);
        forwardKinematics(model_, data, x.head(model_.dof_a));
        const Point3D<double> ee_point = pga_rbm3(data.M.col(model_.n - 1), ee_origin);
        const auto ros_point = pointFromEigen(ee_point);
        path_marker.points.push_back(ros_point);
        waypoint_marker.points.push_back(ros_point);
      }
    }
    array.markers.push_back(path_marker);
    array.markers.push_back(waypoint_marker);

    visualization_msgs::msg::Marker target_marker;
    target_marker.header.frame_id = marker_frame_id_;
    target_marker.header.stamp = stamp;
    target_marker.ns = "target";
    target_marker.id = 200;
    target_marker.type = visualization_msgs::msg::Marker::SPHERE;
    target_marker.action = visualization_msgs::msg::Marker::ADD;
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
    } else {
      target_marker.action = visualization_msgs::msg::Marker::DELETE;
    }
    array.markers.push_back(target_marker);

    marker_pub_->publish(array);
  }

  void publishLatestVisualizations() {
    const std::vector<Eigen::VectorXd>* xs_ptr = last_xs_.empty() ? nullptr : &last_xs_;
    const geometry_msgs::msg::PoseStamped* target_ptr =
        last_target_pose_.has_value() ? &last_target_pose_.value() : nullptr;
    publishVisualizations(xs_ptr, target_ptr);
  }

  void updateStatus(const std::string& text) {
    std_msgs::msg::String msg;
    msg.data = text;
    status_pub_->publish(msg);
  }

  void targetPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    last_target_pose_ = *msg;
    planned_traj_.points.clear();
    last_xs_.clear();
    last_us_.clear();
    has_pending_plan_ = false;
    publishVisualizations(nullptr, msg.get());
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
    SolveResetGuard reset_guard(solving_);
    updateStatus("Status: planning");
    const auto& target_msg = last_target_pose_.value();

    const auto start = std::chrono::steady_clock::now();
    const Motor3D<double> M_ref = motorFromPose(target_msg.pose);

    Eigen::VectorXd x0(2 * model_.dof_a);
    x0.head(model_.dof_a) = joint_pos_;
    x0.tail(model_.dof_a) = has_joint_state_ ? joint_vel_ : Eigen::VectorXd::Zero(model_.dof_a);

    for (int i = 0; i < model_.dof_a; ++i) {
      if (!std::isfinite(x0[i])) {
        x0[i] = model_.qa0[i];
      }
      if (!std::isfinite(x0[model_.dof_a + i])) {
        x0[model_.dof_a + i] = 0.0;
      }
      x0[i] = std::clamp(x0[i], model_.lowerPositionLimit[i], model_.upperPositionLimit[i]);
      x0[model_.dof_a + i] = std::clamp(x0[model_.dof_a + i], -1.0, 1.0);
    }

    auto running_cost_model = std::make_shared<crocoddyl::CostModelSum>(state_);
    auto terminal_cost_model = std::make_shared<crocoddyl::CostModelSum>(state_);

    auto placement_residual = std::make_shared<ResidualModelTetraPGAFramePlacement<double>>(state_, model_, M_ref);
    auto placement_cost = std::make_shared<crocoddyl::CostModelResidual>(state_, placement_residual);

    const Eigen::VectorXd x_zero = Eigen::VectorXd::Zero(2 * model_.dof_a);
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
    auto vel_activation = std::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(vel_bounds);
    auto vel_limit_cost =
        std::make_shared<crocoddyl::CostModelResidual>(state_, vel_activation, vel_residual);

    if (env_.num_static_sphere > 0) {
      auto collision_residual =
          std::make_shared<ResidualModelTetraPGACollisionDistance<double>>(state_, model_, env_, safety_distance_);
      const int num_collision_pairs = model_.num_collision_ssl * env_.num_static_sphere;
      crocoddyl::ActivationBounds bounds;
      bounds.lb = Eigen::VectorXd::Zero(num_collision_pairs);
      bounds.ub = Eigen::VectorXd::Constant(num_collision_pairs, std::numeric_limits<double>::infinity());
      auto barrier_activation = std::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(bounds);
      auto collision_cost = std::make_shared<crocoddyl::CostModelResidual>(
          state_, barrier_activation, collision_residual);
      running_cost_model->addCost("collision", collision_cost, collision_weight_);
      terminal_cost_model->addCost("collision", collision_cost, collision_weight_);
    }

    running_cost_model->addCost("placement", placement_cost, placement_running_weight_);
    running_cost_model->addCost("vel_limit", vel_limit_cost, velocity_limit_weight_);
    terminal_cost_model->addCost("placement", placement_cost, placement_terminal_weight_);
    terminal_cost_model->addCost("vel_limit", vel_limit_cost, velocity_limit_weight_);

    auto diff_model =
        std::make_shared<DifferentialActionModelTetraPGAForwardDynamics<double>>(state_, model_, running_cost_model);
    auto diff_model_term =
        std::make_shared<DifferentialActionModelTetraPGAForwardDynamics<double>>(state_, model_, terminal_cost_model);

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
    if (last_xs_.size() == expected_xs && last_us_.size() == expected_us) {
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

    const bool solved = solver.solve(init_xs, init_us, static_cast<std::size_t>(max_iterations_));
    if (!solved || solver.get_xs().empty()) {
      RCLCPP_ERROR(this->get_logger(),
                   "Collision FDDP failed. stop=%.3e cost=%.6f x0_q=%s x0_dq=%s target_p=(%.3f,%.3f,%.3f)",
                   solver.get_stop(), solver.get_cost(),
                   vectorToString(x0.head(model_.dof_a)).c_str(),
                   vectorToString(x0.tail(model_.dof_a)).c_str(),
                   target_msg.pose.position.x, target_msg.pose.position.y, target_msg.pose.position.z);
      publishVisualizations(nullptr, &target_msg);
      updateStatus("Status: failed");
      return false;
    }

    last_xs_ = solver.get_xs();
    last_us_ = solver.get_us();
    planned_traj_ = buildTrajectoryFromXs(last_xs_);
    has_pending_plan_ = true;
    publishVisualizations(&last_xs_, &target_msg);

    Data<double> final_data(model_);
    forwardKinematics(model_, final_data, last_xs_.back().head(model_.dof_a));
    const auto ee_error = ga_log(ga_mul(ga_rev(M_ref), final_data.M.col(model_.n - 1)));
    const double min_initial_dist = computeMinDistance(x0.head(model_.dof_a));
    const double min_traj_dist = computeTrajectoryMinDistance(last_xs_);
    const double max_abs_vel = maxAbsVelocity(last_xs_, model_.dof_a);
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start)
            .count();

    RCLCPP_INFO(this->get_logger(),
                "Collision plan ready for review: points=%zu cost=%.6f stop=%.3e solve_time=%ld ms "
                "min_init_dist=%.4f min_traj_dist=%.4f ee_error=%.3e max_abs_vel=%.3f final_q=%s",
                planned_traj_.points.size(), solver.get_cost(), solver.get_stop(), elapsed_ms,
                min_initial_dist, min_traj_dist, ee_error.norm(), max_abs_vel,
                vectorToString(last_xs_.back().head(model_.dof_a)).c_str());
    updateStatus("Status: planned");
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
      RCLCPP_WARN(this->get_logger(), "No planned collision-free trajectory waiting for execution.");
      return false;
    }

    planned_traj_.header.stamp = this->now();
    updateStatus("Status: executing");
    joint_cmd_pub_->publish(planned_traj_);
    has_pending_plan_ = false;
    RCLCPP_INFO(this->get_logger(), "Published cached collision plan to /joint_commands with %zu points.",
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
    response->message = response->success ? "Collision plan ready for preview." : "Collision planning failed.";
  }

  void handleExecuteService(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                            std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    response->success = executePlannedTrajectory();
    response->message = response->success ? "Executed cached collision plan." : "No executable cached trajectory.";
  }

  void handleClearService(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                          std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    planned_traj_.points.clear();
    last_xs_.clear();
    last_us_.clear();
    has_pending_plan_ = false;
    publishVisualizations(nullptr, last_target_pose_.has_value() ? &last_target_pose_.value() : nullptr);
    updateStatus(last_target_pose_.has_value() ? "Status: goal_set" : "Status: idle");
    response->success = true;
    response->message = "Cleared cached collision trajectory preview.";
    RCLCPP_INFO(this->get_logger(), "Cleared cached collision trajectory preview.");
  }

private:
  Model<double> model_;
  Environment<double> env_{std::vector<SSP<double>>()};
  std::shared_ptr<crocoddyl::StateAbstract> state_;

  double dt_{};
  int horizon_{};
  int max_iterations_{};
  double placement_running_weight_{};
  double placement_terminal_weight_{};
  double velocity_limit_weight_{};
  double collision_weight_{};
  double safety_distance_{};
  bool publish_visualization_{true};
  std::string marker_frame_id_;
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

  std::array<std::string, 6> joint_names_;

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
  rclcpp::spin(std::make_shared<CrocoddylCollisionPlannerNode>());
  rclcpp::shutdown();
  return 0;
}
