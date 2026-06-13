#pragma once

#include <memory>
#include <stdexcept>

#include <Eigen/Dense>
#include <crocoddyl/core/costs/cost-sum.hpp>
#include <crocoddyl/core/data-collector-base.hpp>
#include <crocoddyl/core/diff-action-base.hpp>
#include <crocoddyl/core/state-base.hpp>

#include "TetraPGA/Dynamics.hpp"
#include "TetraPGA/Kinematics.hpp"

using namespace TetraPGA;

template <typename Scalar>
class DifferentialActionModelTetraPGAForwardDynamics;

template <typename Scalar>
struct DifferentialActionDataTetraPGAForwardDynamics : public crocoddyl::DifferentialActionDataAbstractTpl<Scalar>,
                                   public crocoddyl::DataCollectorAbstractTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef crocoddyl::DifferentialActionDataAbstractTpl<Scalar> Base;
  typedef crocoddyl::CostDataSumTpl<Scalar> CostDataSum;
  typedef crocoddyl::DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;

  Data<Scalar> ga_data;
  std::shared_ptr<CostDataSum> costs;

  template <typename Model>
  explicit DifferentialActionDataTetraPGAForwardDynamics(Model* const model);
};

template <typename Scalar>
class DifferentialActionModelTetraPGAInverseDynamics;

template <typename Scalar>
struct DifferentialActionDataTetraPGAInverseDynamics
    : public crocoddyl::DifferentialActionDataAbstractTpl<Scalar>,
      public crocoddyl::DataCollectorAbstractTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef crocoddyl::DifferentialActionDataAbstractTpl<Scalar> Base;
  typedef crocoddyl::CostDataSumTpl<Scalar> CostDataSum;
  typedef crocoddyl::DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;

  Data<Scalar> ga_data;
  std::shared_ptr<CostDataSum> costs;

  template <typename Model>
  explicit DifferentialActionDataTetraPGAInverseDynamics(Model* const model);
};

template <typename Scalar>
class DifferentialActionModelTetraPGAForwardDynamics : public crocoddyl::DifferentialActionModelAbstractTpl<Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  friend struct DifferentialActionDataTetraPGAForwardDynamics<Scalar>;

  DifferentialActionModelTetraPGAForwardDynamics(std::shared_ptr<crocoddyl::StateAbstractTpl<Scalar>> state,
                    const Model<Scalar>& ga_model,
                    std::shared_ptr<crocoddyl::CostModelSumTpl<Scalar>> cost_model = nullptr)
      : crocoddyl::DifferentialActionModelAbstractTpl<Scalar>(state, ga_model.dof_a), // nu = dof_a (假设全驱动)
        ga_model_(ga_model),
        costs_(cost_model) {
    // 检查 State 维度与 GA 模型是否匹配
    if (state->get_nq() != static_cast<std::size_t>(ga_model.dof_a) || 
        state->get_nv() != static_cast<std::size_t>(ga_model.dof_a)) {
    }
  }

  virtual ~DifferentialActionModelTetraPGAForwardDynamics() {}

  virtual void calc(const std::shared_ptr<crocoddyl::DifferentialActionDataAbstract>& data,
                    const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>& x,
                    const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>& u) {
    // 1. 转换 Data 指针
    DifferentialActionDataTetraPGAForwardDynamics<Scalar>* d = static_cast<DifferentialActionDataTetraPGAForwardDynamics<Scalar>*>(data.get());

    // 2. 拆分状态 x = [q; v]
    const std::size_t nq = this->get_state()->get_nq();
    const std::size_t nv = this->get_state()->get_nv();

    // 3. 调用正向动力学
    // forwardDynamics 计算出 acceleration 并存入 ga_data.ddq
    forwardDynamics(ga_model_, d->ga_data, x.head(nq), x.tail(nv), u);
    forwardKinematics(ga_model_, d->ga_data, x.head(nq));

    // 4. 将结果赋值给 Crocoddyl 需要的 xout (即 acceleration)
    d->xout = d->ga_data.ddq;

    // 5. 计算 Cost
    if (costs_) {
      costs_->calc(d->costs, x, u);
      d->cost = d->costs->cost;
    } else {
      d->cost = 0;
    }
  }

  virtual void calcDiff(const std::shared_ptr<crocoddyl::DifferentialActionDataAbstract>& data,
                        const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>& x,
                        const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>& u) {
	    DifferentialActionDataTetraPGAForwardDynamics<Scalar>* d = static_cast<DifferentialActionDataTetraPGAForwardDynamics<Scalar>*>(data.get());
	
	    const std::size_t nq = this->get_state()->get_nq();
	    const std::size_t nv = this->get_state()->get_nv();
	
	    // 1. 调用一阶导数算法
	    // 该函数会填充 ga_data.pddq_pq, ga_data.pddq_pdq, ga_data.pddq_ptau
	    forwardDynamics_fo(ga_model_, d->ga_data, x.head(nq), x.tail(nv), u);

    // 2. 填充 Fx (Dynamics Jacobian w.r.t State)
    // Fx = [ da/dq, da/dv ]
    // 注意：Data 中的矩阵是 resize 过的，直接赋值是安全的
    d->Fx.leftCols(nv) = d->ga_data.pddq_pq;
    d->Fx.rightCols(nv) = d->ga_data.pddq_pdq;

    // 3. 填充 Fu (Dynamics Jacobian w.r.t Control)
    d->Fu = d->ga_data.pddq_ptau;

    // 4. 计算 Cost Derivatives 并复制到 data
    if (costs_) {
      costs_->calcDiff(d->costs, x, u);
      // 关键：将 cost 梯度复制到 action data
      d->Lx = d->costs->Lx;
      d->Lu = d->costs->Lu;
      d->Lxx = d->costs->Lxx;
      d->Lxu = d->costs->Lxu;
      d->Luu = d->costs->Luu;
    }
  }

  // ===========================================================================
  // 创建数据结构
  // ===========================================================================
  virtual std::shared_ptr<crocoddyl::DifferentialActionDataAbstract> createData() {
    return std::make_shared<DifferentialActionDataTetraPGAForwardDynamics<Scalar>>(this);
  }

  virtual std::shared_ptr<crocoddyl::DifferentialActionModelBase> cloneAsDouble() const {
    // 简化实现：不支持跨标量类型克隆
    throw std::runtime_error("cloneAsDouble not implemented for DifferentialActionModelTetraPGAForwardDynamics");
  }

  virtual std::shared_ptr<crocoddyl::DifferentialActionModelBase> cloneAsFloat() const {
    // 简化实现：不支持跨标量类型克隆
    throw std::runtime_error("cloneAsFloat not implemented for DifferentialActionModelTetraPGAForwardDynamics");
  }

  // ===========================================================================
  // 访问接口
  // ===========================================================================
  const Model<Scalar>& get_ga_model() const { 
      return ga_model_; 
  }

 private:
  Model<Scalar> ga_model_;
  std::shared_ptr<crocoddyl::CostModelSumTpl<Scalar>> costs_;
};

template <typename Scalar>
template <typename Model>
DifferentialActionDataTetraPGAForwardDynamics<Scalar>::DifferentialActionDataTetraPGAForwardDynamics(Model* const model)
    : crocoddyl::DifferentialActionDataAbstractTpl<Scalar>(model),
      ga_data(static_cast<DifferentialActionModelTetraPGAForwardDynamics<Scalar>*>(model)->get_ga_model())
{
    // 初始化 costs 数据
    auto ga_model_ptr = static_cast<DifferentialActionModelTetraPGAForwardDynamics<Scalar>*>(model);
    if (ga_model_ptr->costs_) {
        costs = ga_model_ptr->costs_->createData(
            static_cast<DataCollectorAbstract*>(this));
    }
}
template <typename Scalar>
class DifferentialActionModelTetraPGAInverseDynamics
    : public crocoddyl::DifferentialActionModelAbstractTpl<Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  friend struct DifferentialActionDataTetraPGAInverseDynamics<Scalar>;

  DifferentialActionModelTetraPGAInverseDynamics(
      std::shared_ptr<crocoddyl::StateAbstractTpl<Scalar>> state,
      const Model<Scalar>& ga_model,
      std::shared_ptr<crocoddyl::CostModelSumTpl<Scalar>> cost_model = nullptr)
      : crocoddyl::DifferentialActionModelAbstractTpl<Scalar>(
            state, ga_model.dof_a),
        ga_model_(ga_model),
        costs_(cost_model) {}

  virtual ~DifferentialActionModelTetraPGAInverseDynamics() {}

  virtual void calc(
      const std::shared_ptr<crocoddyl::DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>& x,
      const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>& u) {
    DifferentialActionDataTetraPGAInverseDynamics<Scalar>* d =
        static_cast<DifferentialActionDataTetraPGAInverseDynamics<Scalar>*>(data.get());

    const std::size_t nq = this->get_state()->get_nq();
    const std::size_t nv = this->get_state()->get_nv();

    inverseDynamics(ga_model_, d->ga_data, x.head(nq), x.tail(nv), u);
    forwardKinematics(ga_model_, d->ga_data, x.head(nq));

    d->xout = u;

    if (costs_) {
      costs_->calc(d->costs, x, u);
      d->cost = d->costs->cost;
    } else {
      d->cost = 0;
    }
  }

  virtual void calcDiff(
      const std::shared_ptr<crocoddyl::DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>& x,
      const Eigen::Ref<const typename crocoddyl::MathBaseTpl<Scalar>::VectorXs>& u) {
    DifferentialActionDataTetraPGAInverseDynamics<Scalar>* d =
        static_cast<DifferentialActionDataTetraPGAInverseDynamics<Scalar>*>(data.get());

    const std::size_t nq = this->get_state()->get_nq();
    const std::size_t nv = this->get_state()->get_nv();

    inverseDynamics_fo(ga_model_, d->ga_data, x.head(nq), x.tail(nv), u);

    d->Fx.setZero();
    d->Fu.setZero();
    d->Fu.leftCols(nv).diagonal().setOnes();

    if (costs_) {
      costs_->calcDiff(d->costs, x, u);
      d->Lx = d->costs->Lx;
      d->Lu = d->costs->Lu;
      d->Lxx = d->costs->Lxx;
      d->Lxu = d->costs->Lxu;
      d->Luu = d->costs->Luu;
    }
  }

  virtual std::shared_ptr<crocoddyl::DifferentialActionDataAbstract>
  createData() {
    return std::make_shared<DifferentialActionDataTetraPGAInverseDynamics<Scalar>>(this);
  }

  virtual std::shared_ptr<crocoddyl::DifferentialActionModelBase>
  cloneAsDouble() const {
    throw std::runtime_error(
        "cloneAsDouble not implemented for DifferentialActionModelTetraPGAInverseDynamics");
  }

  virtual std::shared_ptr<crocoddyl::DifferentialActionModelBase>
  cloneAsFloat() const {
    throw std::runtime_error(
        "cloneAsFloat not implemented for DifferentialActionModelTetraPGAInverseDynamics");
  }

  const Model<Scalar>& get_ga_model() const { return ga_model_; }

 private:
  Model<Scalar> ga_model_;
  std::shared_ptr<crocoddyl::CostModelSumTpl<Scalar>> costs_;
};

template <typename Scalar>
template <typename Model>
DifferentialActionDataTetraPGAInverseDynamics<Scalar>::DifferentialActionDataTetraPGAInverseDynamics(Model* const model)
    : crocoddyl::DifferentialActionDataAbstractTpl<Scalar>(model),
      ga_data(static_cast<DifferentialActionModelTetraPGAInverseDynamics<Scalar>*>(model)->get_ga_model()) {
  auto ga_model_ptr = static_cast<DifferentialActionModelTetraPGAInverseDynamics<Scalar>*>(model);
  this->Fu.setZero();
  this->Fu.leftCols(model->get_state()->get_nv()).diagonal().setOnes();
  if (ga_model_ptr->costs_) {
    costs = ga_model_ptr->costs_->createData(
        static_cast<DataCollectorAbstract*>(this));
  }
}
