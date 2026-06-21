#pragma once

#include <memory>
#include <stdexcept>

#include <Eigen/Dense>
#include <crocoddyl/core/residual-base.hpp>
#include <crocoddyl/core/state-base.hpp>

#include "TetraPGA/Collision.hpp"
#include "TetraPGA/Kinematics.hpp"
#include "ga_ocp/CrocoddylActions.hpp"

using namespace TetraPGA;

template <typename Scalar>
class ResidualModelTetraPGAJointAcceleration;

template <typename Scalar>
struct ResidualDataTetraPGAJointAcceleration
    : public crocoddyl::ResidualDataAbstractTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = crocoddyl::ResidualDataAbstractTpl<Scalar>;

  Data<Scalar>* ga_data;

  template <typename Model>
  explicit ResidualDataTetraPGAJointAcceleration(
      Model* const model,
      crocoddyl::DataCollectorAbstractTpl<Scalar>* const data)
      : Base(model, data),
        ga_data(nullptr) {
    auto* action_data = dynamic_cast<DifferentialActionDataTetraPGAForwardDynamics<Scalar>*>(data);
    if (action_data == nullptr) {
      throw std::invalid_argument(
          "ResidualDataTetraPGAJointAcceleration requires DifferentialActionDataTetraPGAForwardDynamics as data collector");
    }
    ga_data = &action_data->ga_data;
  }
};

template <typename Scalar>
class ResidualModelTetraPGAJointAcceleration
    : public crocoddyl::ResidualModelAbstractTpl<Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = crocoddyl::ResidualModelAbstractTpl<Scalar>;
  using VectorXs = typename crocoddyl::MathBaseTpl<Scalar>::VectorXs;

  ResidualModelTetraPGAJointAcceleration(
      const std::shared_ptr<crocoddyl::StateAbstractTpl<Scalar>>& state,
      const Model<Scalar>& ga_model,
      const VectorXs& a_ref)
      : Base(state,
             static_cast<std::size_t>(ga_model.dof_a),
             static_cast<std::size_t>(ga_model.dof_a),
             true,
             true,
             true),
        ga_model_(ga_model),
        a_ref_(a_ref) {}

  void calc(const std::shared_ptr<crocoddyl::ResidualDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u) override {
    auto* d = static_cast<ResidualDataTetraPGAJointAcceleration<Scalar>*>(data.get());
    data->r = d->ga_data->ddq - a_ref_;
  }

  void calcDiff(const std::shared_ptr<crocoddyl::ResidualDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u) override {
    auto* d = static_cast<ResidualDataTetraPGAJointAcceleration<Scalar>*>(data.get());
    const std::size_t nq = this->get_state()->get_nq();
    const std::size_t nv = this->get_state()->get_nv();

    data->Rx.setZero();
    data->Ru.setZero();
    data->Rx.leftCols(nq) = d->ga_data->pddq_pq;
    data->Rx.rightCols(nv) = d->ga_data->pddq_pdq;
    data->Ru = d->ga_data->pddq_ptau;
  }

  std::shared_ptr<crocoddyl::ResidualDataAbstract> createData(
      crocoddyl::DataCollectorAbstract* const data) override {
    return std::make_shared<ResidualDataTetraPGAJointAcceleration<Scalar>>(
        this, static_cast<crocoddyl::DataCollectorAbstractTpl<Scalar>*>(data));
  }

 private:
  Model<Scalar> ga_model_;
  VectorXs a_ref_;
};

template <typename Scalar>
class ResidualModelTetraPGAJointTorque;

template <typename Scalar>
struct ResidualDataTetraPGAJointTorque
    : public crocoddyl::ResidualDataAbstractTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = crocoddyl::ResidualDataAbstractTpl<Scalar>;

  Data<Scalar>* ga_data;

  template <typename Model>
  explicit ResidualDataTetraPGAJointTorque(
      Model* const model,
      crocoddyl::DataCollectorAbstractTpl<Scalar>* const data)
      : Base(model, data),
        ga_data(nullptr) {
    auto* action_data = dynamic_cast<DifferentialActionDataTetraPGAInverseDynamics<Scalar>*>(data);
    if (action_data == nullptr) {
      throw std::invalid_argument(
          "ResidualDataTetraPGAJointTorque requires DifferentialActionDataTetraPGAInverseDynamics as data collector");
    }
    ga_data = &action_data->ga_data;
  }
};

template <typename Scalar>
class ResidualModelTetraPGAJointTorque
    : public crocoddyl::ResidualModelAbstractTpl<Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = crocoddyl::ResidualModelAbstractTpl<Scalar>;
  using VectorXs = typename crocoddyl::MathBaseTpl<Scalar>::VectorXs;

  ResidualModelTetraPGAJointTorque(
      const std::shared_ptr<crocoddyl::StateAbstractTpl<Scalar>>& state,
      const Model<Scalar>& ga_model,
      const VectorXs& tau_ref)
      : Base(state,
             static_cast<std::size_t>(ga_model.dof_a),
             static_cast<std::size_t>(ga_model.dof_a),
             true,
             true,
             true),
        ga_model_(ga_model),
        tau_ref_(tau_ref) {}

  void calc(const std::shared_ptr<crocoddyl::ResidualDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u) override {
    auto* d = static_cast<ResidualDataTetraPGAJointTorque<Scalar>*>(data.get());
    data->r = d->ga_data->tau - tau_ref_;
  }

  void calcDiff(const std::shared_ptr<crocoddyl::ResidualDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u) override {
    auto* d = static_cast<ResidualDataTetraPGAJointTorque<Scalar>*>(data.get());
    const std::size_t nq = this->get_state()->get_nq();
    const std::size_t nv = this->get_state()->get_nv();

    data->Rx.setZero();
    data->Ru.setZero();
    data->Rx.leftCols(nq) = d->ga_data->ptau_pq;
    data->Rx.rightCols(nv) = d->ga_data->ptau_pdq;
    data->Ru = d->ga_data->ptau_pddq;
  }

  std::shared_ptr<crocoddyl::ResidualDataAbstract> createData(
      crocoddyl::DataCollectorAbstract* const data) override {
    return std::make_shared<ResidualDataTetraPGAJointTorque<Scalar>>(
        this, static_cast<crocoddyl::DataCollectorAbstractTpl<Scalar>*>(data));
  }

  const VectorXs& get_reference() const { return tau_ref_; }

 private:
  Model<Scalar> ga_model_;
  VectorXs tau_ref_;
};

/****** define the Placement Residual Model ******/

template <typename Scalar>
class ResidualModelTetraPGAFramePlacement;

// Align reference motors once before constructing residuals
template <typename DerivedReference, typename DerivedCandidate>
inline Motor3D<typename DerivedReference::Scalar> align_motor_hemisphere(
    const Eigen::MatrixBase<DerivedReference>& reference,
    const Eigen::MatrixBase<DerivedCandidate>& candidate) {
  using Scalar = typename DerivedReference::Scalar;
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(DerivedReference, 8);
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(DerivedCandidate, 8);
  Motor3D<Scalar> aligned = reference;
  if (aligned.template head<4>().dot(candidate.template head<4>()) < Scalar(0)) {
    aligned = -aligned;
  }
  return aligned;
}

template <typename Scalar>
struct ResidualDataTetraPGAFramePlacement
    : public crocoddyl::ResidualDataAbstractTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = crocoddyl::ResidualDataAbstractTpl<Scalar>;

  Data<Scalar>* ga_data;
  Line3D<Scalar> r;
  Eigen::Matrix<Scalar, 6, Eigen::Dynamic> J;

  template <typename Model>
  explicit ResidualDataTetraPGAFramePlacement(
      Model* const model,
      crocoddyl::DataCollectorAbstractTpl<Scalar>* const data)
      : Base(model, data),
        ga_data(nullptr) {
    auto* action_data = dynamic_cast<DifferentialActionDataTetraPGAForwardDynamics<Scalar>*>(data);
    if (action_data == nullptr) {
      throw std::invalid_argument(
          "ResidualDataTetraPGAFramePlacement requires DifferentialActionDataTetraPGAForwardDynamics as data collector");
    }
    ga_data = &action_data->ga_data;
    J.resize(6, ga_data->q.size());
  }
};

template <typename Scalar>
class ResidualModelTetraPGAFramePlacement
    : public crocoddyl::ResidualModelAbstractTpl<Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = crocoddyl::ResidualModelAbstractTpl<Scalar>;
  using VectorXs = typename crocoddyl::MathBaseTpl<Scalar>::VectorXs;

  ResidualModelTetraPGAFramePlacement(
      const std::shared_ptr<crocoddyl::StateAbstractTpl<Scalar>>& state,
      const Model<Scalar>& ga_model,
      const Motor3D<Scalar>& M_ref)
            : Base(state,
              static_cast<std::size_t>(6),
              static_cast<std::size_t>(state->get_nv()),
              true,
              false,
              false),
        ga_model_(ga_model),
        M_ref_(M_ref) {}

  void calc(const std::shared_ptr<crocoddyl::ResidualDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u) override {
    auto* d = static_cast<ResidualDataTetraPGAFramePlacement<Scalar>*>(data.get());

    const auto M_cur = d->ga_data->M.col(ga_model_.n - 1);
    d->r = ga_log(ga_mul(ga_rev(M_ref_), M_cur));
    data->r = d->r;
  }

  void calcDiff(const std::shared_ptr<crocoddyl::ResidualDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u) override {
    auto* d = static_cast<ResidualDataTetraPGAFramePlacement<Scalar>*>(data.get());
    const std::size_t nq = this->get_state()->get_nq();

    const auto M_cur = d->ga_data->M.col(ga_model_.n - 1);
    d->r = ga_log(ga_mul(ga_rev(M_ref_), M_cur));

    // Analytic Jacobian of log(M_ref^{-1} M(q))
    analyticJacobian(ga_model_, *(d->ga_data), x.head(nq), d->r);
    d->J = d->ga_data->jac;

    data->Rx.setZero();
    data->Ru.setZero();
    data->Rx.leftCols(nq) = d->J;
  }

  std::shared_ptr<crocoddyl::ResidualDataAbstract> createData(
      crocoddyl::DataCollectorAbstract* const data) override {
    return std::make_shared<ResidualDataTetraPGAFramePlacement<Scalar>>(
        this, static_cast<crocoddyl::DataCollectorAbstractTpl<Scalar>*>(data));
  }

  const Model<Scalar>& get_ga_model() const { return ga_model_; }

 private:
  Model<Scalar> ga_model_;
  Motor3D<Scalar> M_ref_;
};

/****** define the Motor Residual Model ******/

template <typename Scalar>
class ResidualModelTetraPGAFrameMotor;

template <typename Scalar>
struct ResidualDataTetraPGAFrameMotor
    : public crocoddyl::ResidualDataAbstractTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = crocoddyl::ResidualDataAbstractTpl<Scalar>;

  Data<Scalar>* ga_data;
  Motor3D<Scalar> r;
  Eigen::Matrix<Scalar, 8, Eigen::Dynamic> J;

  template <typename Model>
  explicit ResidualDataTetraPGAFrameMotor(
      Model* const model,
      crocoddyl::DataCollectorAbstractTpl<Scalar>* const data)
      : Base(model, data),
        ga_data(nullptr) {
    auto* action_data = dynamic_cast<DifferentialActionDataTetraPGAForwardDynamics<Scalar>*>(data);
    if (action_data == nullptr) {
      throw std::invalid_argument(
          "ResidualDataTetraPGAFrameMotor requires DifferentialActionDataTetraPGAForwardDynamics as data collector");
    }
    ga_data = &action_data->ga_data;
    J.resize(8, ga_data->q.size());
  }
};

template <typename Scalar>
class ResidualModelTetraPGAFrameMotor
    : public crocoddyl::ResidualModelAbstractTpl<Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = crocoddyl::ResidualModelAbstractTpl<Scalar>;
  using VectorXs = typename crocoddyl::MathBaseTpl<Scalar>::VectorXs;

  ResidualModelTetraPGAFrameMotor(
      const std::shared_ptr<crocoddyl::StateAbstractTpl<Scalar>>& state,
      const Model<Scalar>& ga_model,
      const Motor3D<Scalar>& M_ref)
      : Base(state,
             static_cast<std::size_t>(8),
             static_cast<std::size_t>(state->get_nv()),
             true,
             false,
             false),
        ga_model_(ga_model),
        M_ref_(M_ref) {}

  void calc(const std::shared_ptr<crocoddyl::ResidualDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u) override {
    auto* d = static_cast<ResidualDataTetraPGAFrameMotor<Scalar>*>(data.get());

    const auto M_cur = d->ga_data->M.col(ga_model_.n - 1);
    d->r = M_cur - M_ref_;
    data->r = d->r;
  }

  void calcDiff(const std::shared_ptr<crocoddyl::ResidualDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u) override {
    auto* d = static_cast<ResidualDataTetraPGAFrameMotor<Scalar>*>(data.get());
    const std::size_t nq = this->get_state()->get_nq();

    motorJacobian(ga_model_, *(d->ga_data), x.head(nq));
    d->J = d->ga_data->jacM;

    data->Rx.setZero();
    data->Ru.setZero();
    data->Rx.leftCols(nq) = d->J;
  }

  std::shared_ptr<crocoddyl::ResidualDataAbstract> createData(
      crocoddyl::DataCollectorAbstract* const data) override {
    return std::make_shared<ResidualDataTetraPGAFrameMotor<Scalar>>(
        this, static_cast<crocoddyl::DataCollectorAbstractTpl<Scalar>*>(data));
  }

  const Model<Scalar>& get_ga_model() const { return ga_model_; }

 private:
  Model<Scalar> ga_model_;
  Motor3D<Scalar> M_ref_;
};

/****** define the Collision Residual Model ******/

template <typename Scalar>
class ResidualModelTetraPGACollisionDistance;

template <typename Scalar>
struct ResidualDataTetraPGACollisionDistance
    : public crocoddyl::ResidualDataAbstractTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = crocoddyl::ResidualDataAbstractTpl<Scalar>;

  Data<Scalar>* ga_data;  // 指向 DifferentialActionDataTetraPGAForwardDynamics 中的 ga_data
  Environment<Scalar> env;
  EnvironmentData<Scalar> env_data;

  template <typename Model>
  explicit ResidualDataTetraPGACollisionDistance(
      Model* const model,
      crocoddyl::DataCollectorAbstractTpl<Scalar>* const data)
      : Base(model, data),
        env(static_cast<ResidualModelTetraPGACollisionDistance<Scalar>*>(model)->get_environment()),
        env_data(static_cast<ResidualModelTetraPGACollisionDistance<Scalar>*>(model)->get_ga_model(),
                 static_cast<ResidualModelTetraPGACollisionDistance<Scalar>*>(model)->get_environment()) {
    auto* action_data = dynamic_cast<DifferentialActionDataTetraPGAForwardDynamics<Scalar>*>(data);
    if (action_data == nullptr) {
      throw std::invalid_argument(
          "ResidualDataTetraPGACollisionDistance requires DifferentialActionDataTetraPGAForwardDynamics as data collector");
    }
    ga_data = &(action_data->ga_data);
  }
};

template <typename Scalar>
class ResidualModelTetraPGACollisionDistance
    : public crocoddyl::ResidualModelAbstractTpl<Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = crocoddyl::ResidualModelAbstractTpl<Scalar>;
  using VectorXs = typename crocoddyl::MathBaseTpl<Scalar>::VectorXs;

  ResidualModelTetraPGACollisionDistance(
      const std::shared_ptr<crocoddyl::StateAbstractTpl<Scalar>>& state,
      const Model<Scalar>& ga_model,
      const Environment<Scalar>& env,
      const Scalar d_safe = 0.1)
      : Base(state,
             static_cast<std::size_t>(ga_model.num_collision_ssl * env.num_static_sphere),
             static_cast<std::size_t>(state->get_nv()),
             true,
             false,
             false),
        ga_model_(ga_model),
        env_(env),
        d_safe_(d_safe) {}

  void calc(const std::shared_ptr<crocoddyl::ResidualDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u) override {
    auto* d = static_cast<ResidualDataTetraPGACollisionDistance<Scalar>*>(data.get());
    
    // Compute collision distances for all pairs
    computeDistance(ga_model_, *(d->ga_data), d->env, d->env_data);
    
    // Set residual as (d_safe - distance) for each collision pair
    // Activation model will handle max(0, r) to create barrier
    for (int i = 0; i < d->env_data.num_collision_pair; ++i) {
      data->r(i) = d_safe_ - d->env_data.distance[i];
    }
  }

  void calcDiff(const std::shared_ptr<crocoddyl::ResidualDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u) override {
    auto* d = static_cast<ResidualDataTetraPGACollisionDistance<Scalar>*>(data.get());
    const std::size_t nq = this->get_state()->get_nq();
    // Baseline path: recompute witness geometry inside calcDiff.
    computeDistanceJacobian(ga_model_, *(d->ga_data), d->env, d->env_data);
    
    data->Rx.setZero();
    data->Ru.setZero();
    
    // Set Jacobian for each collision pair
    // Negative sign because residual is (d_safe - distance)
    // dr/dq = -d(distance)/dq
    for (int i = 0; i < d->env_data.num_collision_pair; ++i) {
      data->Rx.row(i).head(nq) = -d->env_data.jac_dist[i];
    }
  }

  std::shared_ptr<crocoddyl::ResidualDataAbstract> createData(
      crocoddyl::DataCollectorAbstract* const data) override {
    return std::make_shared<ResidualDataTetraPGACollisionDistance<Scalar>>(
        this, static_cast<crocoddyl::DataCollectorAbstractTpl<Scalar>*>(data));
  }

  const Model<Scalar>& get_ga_model() const { return ga_model_; }
  const Environment<Scalar>& get_environment() const { return env_; }
  Scalar get_d_safe() const { return d_safe_; }

 private:
  Model<Scalar> ga_model_;
  Environment<Scalar> env_;
  Scalar d_safe_;  // Safety distance threshold
};
