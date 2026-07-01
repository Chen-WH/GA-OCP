#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include <crocoddyl/core/cost-base.hpp>
#include <crocoddyl/core/data-collector-base.hpp>

namespace ga_ocp {

enum class RuntimeCostCategory : std::size_t {
  kState = 0,
  kControl = 1,
  kVelocity = 2,
  kCollision = 3,
  kOther = 4,
  kCount = 5,
};

constexpr std::size_t kRuntimeCostCategoryCount =
    static_cast<std::size_t>(RuntimeCostCategory::kCount);

struct RuntimeProfilerCounters {
  double dam_calc_ms = 0.0;
  double dam_calcdiff_ms = 0.0;
  double dynamics_calc_ms = 0.0;
  double dynamics_calcdiff_ms = 0.0;
  double cost_sum_calc_ms = 0.0;
  double cost_sum_calcdiff_ms = 0.0;
  double collision_residual_calc_ms = 0.0;
  double collision_residual_calcdiff_ms = 0.0;

  std::uint64_t dam_calc_calls = 0;
  std::uint64_t dam_calcdiff_calls = 0;
  std::uint64_t dynamics_calc_calls = 0;
  std::uint64_t dynamics_calcdiff_calls = 0;
  std::uint64_t cost_sum_calc_calls = 0;
  std::uint64_t cost_sum_calcdiff_calls = 0;
  std::uint64_t collision_residual_calc_calls = 0;
  std::uint64_t collision_residual_calcdiff_calls = 0;

  std::array<double, kRuntimeCostCategoryCount> cost_item_calc_ms{};
  std::array<double, kRuntimeCostCategoryCount> cost_item_calcdiff_ms{};
  std::array<std::uint64_t, kRuntimeCostCategoryCount> cost_item_calc_calls{};
  std::array<std::uint64_t, kRuntimeCostCategoryCount> cost_item_calcdiff_calls{};
};

inline RuntimeProfilerCounters& MutableRuntimeProfilerCounters() {
  static thread_local RuntimeProfilerCounters counters;
  return counters;
}

inline bool& MutableRuntimeProfilerEnabled() {
  static thread_local bool enabled = false;
  return enabled;
}

inline void SetRuntimeProfilerEnabled(const bool enabled) {
  MutableRuntimeProfilerEnabled() = enabled;
}

inline bool RuntimeProfilerEnabled() {
  return MutableRuntimeProfilerEnabled();
}

inline void ResetRuntimeProfilerCounters() {
  MutableRuntimeProfilerCounters() = RuntimeProfilerCounters{};
}

inline RuntimeProfilerCounters SnapshotRuntimeProfilerCounters() {
  return MutableRuntimeProfilerCounters();
}

inline std::chrono::steady_clock::time_point RuntimeProfilerNow() {
  return std::chrono::steady_clock::now();
}

inline double RuntimeProfilerElapsedMs(
    const std::chrono::steady_clock::time_point& start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() * 1e3;
}

inline void RuntimeProfilerRecordDamCalc(const double ms) {
  auto& c = MutableRuntimeProfilerCounters();
  c.dam_calc_ms += ms;
  ++c.dam_calc_calls;
}

inline void RuntimeProfilerRecordDamCalcDiff(const double ms) {
  auto& c = MutableRuntimeProfilerCounters();
  c.dam_calcdiff_ms += ms;
  ++c.dam_calcdiff_calls;
}

inline void RuntimeProfilerRecordDynamicsCalc(const double ms) {
  auto& c = MutableRuntimeProfilerCounters();
  c.dynamics_calc_ms += ms;
  ++c.dynamics_calc_calls;
}

inline void RuntimeProfilerRecordDynamicsCalcDiff(const double ms) {
  auto& c = MutableRuntimeProfilerCounters();
  c.dynamics_calcdiff_ms += ms;
  ++c.dynamics_calcdiff_calls;
}

inline void RuntimeProfilerRecordCostSumCalc(const double ms) {
  auto& c = MutableRuntimeProfilerCounters();
  c.cost_sum_calc_ms += ms;
  ++c.cost_sum_calc_calls;
}

inline void RuntimeProfilerRecordCostSumCalcDiff(const double ms) {
  auto& c = MutableRuntimeProfilerCounters();
  c.cost_sum_calcdiff_ms += ms;
  ++c.cost_sum_calcdiff_calls;
}

inline void RuntimeProfilerRecordCollisionResidualCalc(const double ms) {
  auto& c = MutableRuntimeProfilerCounters();
  c.collision_residual_calc_ms += ms;
  ++c.collision_residual_calc_calls;
}

inline void RuntimeProfilerRecordCollisionResidualCalcDiff(const double ms) {
  auto& c = MutableRuntimeProfilerCounters();
  c.collision_residual_calcdiff_ms += ms;
  ++c.collision_residual_calcdiff_calls;
}

inline void RuntimeProfilerRecordCostItemCalc(
    const RuntimeCostCategory category, const double ms) {
  const std::size_t index = static_cast<std::size_t>(category);
  auto& c = MutableRuntimeProfilerCounters();
  c.cost_item_calc_ms[index] += ms;
  ++c.cost_item_calc_calls[index];
}

inline void RuntimeProfilerRecordCostItemCalcDiff(
    const RuntimeCostCategory category, const double ms) {
  const std::size_t index = static_cast<std::size_t>(category);
  auto& c = MutableRuntimeProfilerCounters();
  c.cost_item_calcdiff_ms[index] += ms;
  ++c.cost_item_calcdiff_calls[index];
}

inline double RuntimeProfilerModelTimeMs(const RuntimeProfilerCounters& counters) {
  return counters.dam_calc_ms + counters.dam_calcdiff_ms;
}

inline double RuntimeProfilerCostItemCalcMs(
    const RuntimeProfilerCounters& counters, const RuntimeCostCategory category) {
  return counters.cost_item_calc_ms[static_cast<std::size_t>(category)];
}

inline double RuntimeProfilerCostItemCalcDiffMs(
    const RuntimeProfilerCounters& counters, const RuntimeCostCategory category) {
  return counters.cost_item_calcdiff_ms[static_cast<std::size_t>(category)];
}

inline double RuntimeProfilerCostItemTotalMs(
    const RuntimeProfilerCounters& counters, const RuntimeCostCategory category) {
  return RuntimeProfilerCostItemCalcMs(counters, category) +
         RuntimeProfilerCostItemCalcDiffMs(counters, category);
}

template <typename Scalar>
class ProfilingCostModelTpl : public crocoddyl::CostModelAbstractTpl<Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Base = crocoddyl::CostModelAbstractTpl<Scalar>;
  using CostDataAbstract = crocoddyl::CostDataAbstractTpl<Scalar>;
  using DataCollectorAbstract = crocoddyl::DataCollectorAbstractTpl<Scalar>;
  using VectorXs = typename crocoddyl::MathBaseTpl<Scalar>::VectorXs;

  ProfilingCostModelTpl(std::shared_ptr<Base> inner, const RuntimeCostCategory category)
      : Base(inner->get_state(), inner->get_activation(), inner->get_residual()),
        inner_(std::move(inner)),
        category_(category) {}

  void calc(const std::shared_ptr<CostDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u) override {
    if (!RuntimeProfilerEnabled()) {
      inner_->calc(data, x, u);
      return;
    }
    const auto start = RuntimeProfilerNow();
    inner_->calc(data, x, u);
    RuntimeProfilerRecordCostItemCalc(category_, RuntimeProfilerElapsedMs(start));
  }

  void calc(const std::shared_ptr<CostDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x) override {
    if (!RuntimeProfilerEnabled()) {
      inner_->calc(data, x);
      return;
    }
    const auto start = RuntimeProfilerNow();
    inner_->calc(data, x);
    RuntimeProfilerRecordCostItemCalc(category_, RuntimeProfilerElapsedMs(start));
  }

  void calcDiff(const std::shared_ptr<CostDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u) override {
    if (!RuntimeProfilerEnabled()) {
      inner_->calcDiff(data, x, u);
      return;
    }
    const auto start = RuntimeProfilerNow();
    inner_->calcDiff(data, x, u);
    RuntimeProfilerRecordCostItemCalcDiff(category_, RuntimeProfilerElapsedMs(start));
  }

  void calcDiff(const std::shared_ptr<CostDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& x) override {
    if (!RuntimeProfilerEnabled()) {
      inner_->calcDiff(data, x);
      return;
    }
    const auto start = RuntimeProfilerNow();
    inner_->calcDiff(data, x);
    RuntimeProfilerRecordCostItemCalcDiff(category_, RuntimeProfilerElapsedMs(start));
  }

  std::shared_ptr<CostDataAbstract> createData(
      DataCollectorAbstract* const data) override {
    return inner_->createData(data);
  }

  std::shared_ptr<crocoddyl::CostModelBase> cloneAsDouble() const override {
    return inner_->cloneAsDouble();
  }

  std::shared_ptr<crocoddyl::CostModelBase> cloneAsFloat() const override {
    return inner_->cloneAsFloat();
  }

 private:
  std::shared_ptr<Base> inner_;
  RuntimeCostCategory category_;
};

template <typename Scalar>
std::shared_ptr<crocoddyl::CostModelAbstractTpl<Scalar>> ProfileCost(
    const std::shared_ptr<crocoddyl::CostModelAbstractTpl<Scalar>>& inner,
    const RuntimeCostCategory category) {
  return std::make_shared<ProfilingCostModelTpl<Scalar>>(inner, category);
}

}  // namespace ga_ocp
