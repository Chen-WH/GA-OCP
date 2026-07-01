#include <benchmark/benchmark.h>

#include <chrono>
#include <crocoddyl/core/solvers/fddp.hpp>

#include "ga_ocp/BenchUtils.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using DurationSeconds = std::chrono::duration<double>;

constexpr int kMinLevel = 1;
constexpr int kMaxLevel = 7;
constexpr int kMaxLevel_AD = 7;
constexpr int kSampleBatchSize = 10;
constexpr int kBenchmarkIterations = 10;

void RunGAFDDPBenchmark(benchmark::State& state, int bf) {
  const int level = static_cast<int>(state.range(0));
  const int dof = DofFromLevel(level);

  const TreeTemplateParams params = MakeBenchTreeParams(bf, dof);
  const Model<double> model = MakeGABenchModel(params, "tetrapga_crocoddyl_fddp_benchmark");
  const FDDPBenchConfig config{};
  const std::uint32_t seed =
      MixBenchmarkSeed(BenchmarkRunSeed(), 0xFDD0u, static_cast<std::uint32_t>(bf),
                       static_cast<std::uint32_t>(dof));
  FDDPSampleBatch samples = MakeFDDPSamples(dof, kSampleBatchSize, seed, config.position_limit);

  state.counters["DOF"] = static_cast<double>(dof);
  state.counters["Horizon"] = static_cast<double>(config.horizon);
  state.counters["MaxIter"] = static_cast<double>(config.max_iterations);

  double total_solve_seconds = 0.0;
  double total_cost = 0.0;
  double total_stop = 0.0;
  double total_solver_iterations = 0.0;
  std::uint64_t converged_count = 0;

  for (auto _ : state) {
    const std::size_t i = samples.cursor;
    auto problem = BuildGAFDDPProblem(model, samples.x0[i], samples.x_target[i], config);
    crocoddyl::SolverFDDP solver(problem);
    solver.set_th_stop(1e-4);

    const auto start = Clock::now();
    const bool converged = solver.solve({}, {}, static_cast<unsigned int>(config.max_iterations));
    const double solve_seconds = DurationSeconds(Clock::now() - start).count();

    total_solve_seconds += solve_seconds;
    total_cost += solver.get_cost();
    total_stop += solver.get_stop();
    total_solver_iterations += static_cast<double>(solver.get_iter());
    converged_count += converged ? 1u : 0u;

    benchmark::DoNotOptimize(converged);
    benchmark::DoNotOptimize(solver.get_cost());
    benchmark::DoNotOptimize(solver.get_stop());

    samples.cursor = (samples.cursor + 1) % samples.x0.size();
  }

  const double samples_count = static_cast<double>(state.iterations());
  state.counters["SolvePerSolverIter_ms"] =
      total_solver_iterations == 0.0 ? 0.0 : total_solve_seconds * 1e3 / total_solver_iterations;
  state.counters["AvgIter"] = samples_count == 0.0 ? 0.0 : total_solver_iterations / samples_count;
  state.counters["Converged"] =
      benchmark::Counter(static_cast<double>(converged_count), benchmark::Counter::kAvgIterations);
  state.counters["Cost"] = samples_count == 0.0 ? 0.0 : total_cost / samples_count;
  state.counters["Stop"] = samples_count == 0.0 ? 0.0 : total_stop / samples_count;
}

void RunPinocchioFDDPBenchmark(benchmark::State& state, int bf) {
  const int level = static_cast<int>(state.range(0));
  const int dof = DofFromLevel(level);

  const TreeTemplateParams params = MakeBenchTreeParams(bf, dof);
  const pinocchio::Model model = BuildPinModel(params);
  const FDDPBenchConfig config{};
  const std::uint32_t seed =
      MixBenchmarkSeed(BenchmarkRunSeed(), 0xFDD0u, static_cast<std::uint32_t>(bf),
                       static_cast<std::uint32_t>(dof));
  FDDPSampleBatch samples = MakeFDDPSamples(dof, kSampleBatchSize, seed, config.position_limit);

  state.counters["DOF"] = static_cast<double>(dof);
  state.counters["Horizon"] = static_cast<double>(config.horizon);
  state.counters["MaxIter"] = static_cast<double>(config.max_iterations);

  double total_solve_seconds = 0.0;
  double total_cost = 0.0;
  double total_stop = 0.0;
  double total_solver_iterations = 0.0;
  std::uint64_t converged_count = 0;

  for (auto _ : state) {
    const std::size_t i = samples.cursor;
    auto problem = BuildPinFDDPProblem(model, samples.x0[i], samples.x_target[i], config);
    crocoddyl::SolverFDDP solver(problem);
    solver.set_th_stop(1e-4);

    const auto start = Clock::now();
    const bool converged = solver.solve({}, {}, static_cast<unsigned int>(config.max_iterations));
    const double solve_seconds = DurationSeconds(Clock::now() - start).count();

    total_solve_seconds += solve_seconds;
    total_cost += solver.get_cost();
    total_stop += solver.get_stop();
    total_solver_iterations += static_cast<double>(solver.get_iter());
    converged_count += converged ? 1u : 0u;

    benchmark::DoNotOptimize(converged);
    benchmark::DoNotOptimize(solver.get_cost());
    benchmark::DoNotOptimize(solver.get_stop());

    samples.cursor = (samples.cursor + 1) % samples.x0.size();
  }

  const double samples_count = static_cast<double>(state.iterations());
  state.counters["SolvePerSolverIter_ms"] =
      total_solver_iterations == 0.0 ? 0.0 : total_solve_seconds * 1e3 / total_solver_iterations;
  state.counters["AvgIter"] = samples_count == 0.0 ? 0.0 : total_solver_iterations / samples_count;
  state.counters["Converged"] =
      benchmark::Counter(static_cast<double>(converged_count), benchmark::Counter::kAvgIterations);
  state.counters["Cost"] = samples_count == 0.0 ? 0.0 : total_cost / samples_count;
  state.counters["Stop"] = samples_count == 0.0 ? 0.0 : total_stop / samples_count;
}

void RunGAIDMPCBenchmark(benchmark::State& state, int bf) {
  const int level = static_cast<int>(state.range(0));
  const int dof = DofFromLevel(level);

  const TreeTemplateParams params = MakeBenchTreeParams(bf, dof);
  const Model<double> model = MakeGABenchModel(params, "tetrapga_crocoddyl_idmpc_benchmark");
  const FDDPBenchConfig config{};
  const std::uint32_t seed =
      MixBenchmarkSeed(BenchmarkRunSeed(), 0x1D00u, static_cast<std::uint32_t>(bf),
                       static_cast<std::uint32_t>(dof));
  FDDPSampleBatch samples = MakeFDDPSamples(dof, kSampleBatchSize, seed, config.position_limit);

  state.counters["DOF"] = static_cast<double>(dof);
  state.counters["Horizon"] = static_cast<double>(config.horizon);
  state.counters["MaxIter"] = static_cast<double>(config.max_iterations);

  double total_solve_seconds = 0.0;
  double total_cost = 0.0;
  double total_stop = 0.0;
  double total_solver_iterations = 0.0;
  std::uint64_t converged_count = 0;

  for (auto _ : state) {
    const std::size_t i = samples.cursor;
    auto problem = BuildGAIDMPCProblem(model, samples.x0[i], samples.x_target[i], config);
    crocoddyl::SolverFDDP solver(problem);
    solver.set_th_stop(1e-4);

    const auto start = Clock::now();
    const bool converged = solver.solve({}, {}, static_cast<unsigned int>(config.max_iterations));
    const double solve_seconds = DurationSeconds(Clock::now() - start).count();

    total_solve_seconds += solve_seconds;
    total_cost += solver.get_cost();
    total_stop += solver.get_stop();
    total_solver_iterations += static_cast<double>(solver.get_iter());
    converged_count += converged ? 1u : 0u;

    benchmark::DoNotOptimize(converged);
    benchmark::DoNotOptimize(solver.get_cost());
    benchmark::DoNotOptimize(solver.get_stop());

    samples.cursor = (samples.cursor + 1) % samples.x0.size();
  }

  const double samples_count = static_cast<double>(state.iterations());
  state.counters["SolvePerSolverIter_ms"] =
      total_solver_iterations == 0.0 ? 0.0 : total_solve_seconds * 1e3 / total_solver_iterations;
  state.counters["AvgIter"] = samples_count == 0.0 ? 0.0 : total_solver_iterations / samples_count;
  state.counters["Converged"] =
      benchmark::Counter(static_cast<double>(converged_count), benchmark::Counter::kAvgIterations);
  state.counters["Cost"] = samples_count == 0.0 ? 0.0 : total_cost / samples_count;
  state.counters["Stop"] = samples_count == 0.0 ? 0.0 : total_stop / samples_count;
}

void RunPinocchioIDMPCBenchmark(benchmark::State& state, int bf) {
  const int level = static_cast<int>(state.range(0));
  const int dof = DofFromLevel(level);

  const TreeTemplateParams params = MakeBenchTreeParams(bf, dof);
  const pinocchio::Model model = BuildPinModel(params);
  const FDDPBenchConfig config{};
  const std::uint32_t seed =
      MixBenchmarkSeed(BenchmarkRunSeed(), 0x1D00u, static_cast<std::uint32_t>(bf),
                       static_cast<std::uint32_t>(dof));
  FDDPSampleBatch samples = MakeFDDPSamples(dof, kSampleBatchSize, seed, config.position_limit);

  state.counters["DOF"] = static_cast<double>(dof);
  state.counters["Horizon"] = static_cast<double>(config.horizon);
  state.counters["MaxIter"] = static_cast<double>(config.max_iterations);

  double total_solve_seconds = 0.0;
  double total_cost = 0.0;
  double total_stop = 0.0;
  double total_solver_iterations = 0.0;
  std::uint64_t converged_count = 0;

  for (auto _ : state) {
    const std::size_t i = samples.cursor;
    auto problem = BuildPinIDMPCProblem(model, samples.x0[i], samples.x_target[i], config);
    crocoddyl::SolverFDDP solver(problem);
    solver.set_th_stop(1e-4);

    const auto start = Clock::now();
    const bool converged = solver.solve({}, {}, static_cast<unsigned int>(config.max_iterations));
    const double solve_seconds = DurationSeconds(Clock::now() - start).count();

    total_solve_seconds += solve_seconds;
    total_cost += solver.get_cost();
    total_stop += solver.get_stop();
    total_solver_iterations += static_cast<double>(solver.get_iter());
    converged_count += converged ? 1u : 0u;

    benchmark::DoNotOptimize(converged);
    benchmark::DoNotOptimize(solver.get_cost());
    benchmark::DoNotOptimize(solver.get_stop());

    samples.cursor = (samples.cursor + 1) % samples.x0.size();
  }

  const double samples_count = static_cast<double>(state.iterations());
  state.counters["SolvePerSolverIter_ms"] =
      total_solver_iterations == 0.0 ? 0.0 : total_solve_seconds * 1e3 / total_solver_iterations;
  state.counters["AvgIter"] = samples_count == 0.0 ? 0.0 : total_solver_iterations / samples_count;
  state.counters["Converged"] =
      benchmark::Counter(static_cast<double>(converged_count), benchmark::Counter::kAvgIterations);
  state.counters["Cost"] = samples_count == 0.0 ? 0.0 : total_cost / samples_count;
  state.counters["Stop"] = samples_count == 0.0 ? 0.0 : total_stop / samples_count;
}

#ifdef GA_OCP_HAS_CASADI_BENCH
void RunPinocchioCasadiFDDPBenchmark(benchmark::State& state, int bf) {
  const int level = static_cast<int>(state.range(0));
  const int dof = DofFromLevel(level);

  const TreeTemplateParams params = MakeBenchTreeParams(bf, dof);
  const pinocchio::Model model = BuildPinModel(params);
  auto autodiff = std::make_shared<InlineAutoDiffABADerivatives>(
      model, "tetrapga_pinocchio_casadi_fddp_bf" + std::to_string(bf) + "_dof" +
                 std::to_string(dof));
  const FDDPBenchConfig config{};
  const std::uint32_t seed =
      MixBenchmarkSeed(BenchmarkRunSeed(), 0xFDD1u, static_cast<std::uint32_t>(bf),
                       static_cast<std::uint32_t>(dof));
  FDDPSampleBatch samples = MakeFDDPSamples(dof, kSampleBatchSize, seed, config.position_limit);

  state.counters["DOF"] = static_cast<double>(dof);
  state.counters["Horizon"] = static_cast<double>(config.horizon);
  state.counters["MaxIter"] = static_cast<double>(config.max_iterations);

  double total_solve_seconds = 0.0;
  double total_cost = 0.0;
  double total_stop = 0.0;
  double total_solver_iterations = 0.0;
  std::uint64_t converged_count = 0;

  for (auto _ : state) {
    const std::size_t i = samples.cursor;
    auto problem =
        BuildPinCasadiFDDPProblem(model, samples.x0[i], samples.x_target[i], config, autodiff);
    crocoddyl::SolverFDDP solver(problem);
    solver.set_th_stop(1e-4);

    const auto start = Clock::now();
    const bool converged = solver.solve({}, {}, static_cast<unsigned int>(config.max_iterations));
    const double solve_seconds = DurationSeconds(Clock::now() - start).count();

    total_solve_seconds += solve_seconds;
    total_cost += solver.get_cost();
    total_stop += solver.get_stop();
    total_solver_iterations += static_cast<double>(solver.get_iter());
    converged_count += converged ? 1u : 0u;

    benchmark::DoNotOptimize(converged);
    benchmark::DoNotOptimize(solver.get_cost());
    benchmark::DoNotOptimize(solver.get_stop());

    samples.cursor = (samples.cursor + 1) % samples.x0.size();
  }

  const double samples_count = static_cast<double>(state.iterations());
  state.counters["SolvePerSolverIter_ms"] =
      total_solver_iterations == 0.0 ? 0.0 : total_solve_seconds * 1e3 / total_solver_iterations;
  state.counters["AvgIter"] = samples_count == 0.0 ? 0.0 : total_solver_iterations / samples_count;
  state.counters["Converged"] =
      benchmark::Counter(static_cast<double>(converged_count), benchmark::Counter::kAvgIterations);
  state.counters["Cost"] = samples_count == 0.0 ? 0.0 : total_cost / samples_count;
  state.counters["Stop"] = samples_count == 0.0 ? 0.0 : total_stop / samples_count;
}

void RunPinocchioCasadiIDMPCBenchmark(benchmark::State& state, int bf) {
  const int level = static_cast<int>(state.range(0));
  const int dof = DofFromLevel(level);

  const TreeTemplateParams params = MakeBenchTreeParams(bf, dof);
  const pinocchio::Model model = BuildPinModel(params);
  auto autodiff = std::make_shared<InlineAutoDiffRNEADerivatives>(
      model, "tetrapga_pinocchio_casadi_idmpc_bf" + std::to_string(bf) + "_dof" +
                 std::to_string(dof));
  const FDDPBenchConfig config{};
  const std::uint32_t seed =
      MixBenchmarkSeed(BenchmarkRunSeed(), 0x1DD1u, static_cast<std::uint32_t>(bf),
                       static_cast<std::uint32_t>(dof));
  FDDPSampleBatch samples = MakeFDDPSamples(dof, kSampleBatchSize, seed, config.position_limit);

  state.counters["DOF"] = static_cast<double>(dof);
  state.counters["Horizon"] = static_cast<double>(config.horizon);
  state.counters["MaxIter"] = static_cast<double>(config.max_iterations);

  double total_solve_seconds = 0.0;
  double total_cost = 0.0;
  double total_stop = 0.0;
  double total_solver_iterations = 0.0;
  std::uint64_t converged_count = 0;

  for (auto _ : state) {
    const std::size_t i = samples.cursor;
    auto problem =
        BuildPinCasadiIDMPCProblem(model, samples.x0[i], samples.x_target[i], config, autodiff);
    crocoddyl::SolverFDDP solver(problem);
    solver.set_th_stop(1e-4);

    const auto start = Clock::now();
    const bool converged = solver.solve({}, {}, static_cast<unsigned int>(config.max_iterations));
    const double solve_seconds = DurationSeconds(Clock::now() - start).count();

    total_solve_seconds += solve_seconds;
    total_cost += solver.get_cost();
    total_stop += solver.get_stop();
    total_solver_iterations += static_cast<double>(solver.get_iter());
    converged_count += converged ? 1u : 0u;

    benchmark::DoNotOptimize(converged);
    benchmark::DoNotOptimize(solver.get_cost());
    benchmark::DoNotOptimize(solver.get_stop());

    samples.cursor = (samples.cursor + 1) % samples.x0.size();
  }

  const double samples_count = static_cast<double>(state.iterations());
  state.counters["SolvePerSolverIter_ms"] =
      total_solver_iterations == 0.0 ? 0.0 : total_solve_seconds * 1e3 / total_solver_iterations;
  state.counters["AvgIter"] = samples_count == 0.0 ? 0.0 : total_solver_iterations / samples_count;
  state.counters["Converged"] =
      benchmark::Counter(static_cast<double>(converged_count), benchmark::Counter::kAvgIterations);
  state.counters["Cost"] = samples_count == 0.0 ? 0.0 : total_cost / samples_count;
  state.counters["Stop"] = samples_count == 0.0 ? 0.0 : total_stop / samples_count;
}
#endif

void RegisterAll() {
  benchmark::RegisterBenchmark("serial_chain/TetraPGA/CrocoddylFDDP", [](benchmark::State& s) {
    RunGAFDDPBenchmark(s, 1);
  })->DenseRange(kMinLevel, kMaxLevel, 1)->Iterations(kBenchmarkIterations);

  benchmark::RegisterBenchmark("serial_chain/Pinocchio/CrocoddylFDDP", [](benchmark::State& s) {
    RunPinocchioFDDPBenchmark(s, 1);
  })->DenseRange(kMinLevel, kMaxLevel, 1)->Iterations(kBenchmarkIterations);

  #ifdef GA_OCP_HAS_CASADI_BENCH
  benchmark::RegisterBenchmark("serial_chain/CasADi/CrocoddylFDDP",
                               [](benchmark::State& s) {
    RunPinocchioCasadiFDDPBenchmark(s, 1);
  })->DenseRange(kMinLevel, kMaxLevel_AD, 1)->Iterations(kBenchmarkIterations);
  #endif

  benchmark::RegisterBenchmark("serial_chain/TetraPGA/CrocoddylIDMPC", [](benchmark::State& s) {
    RunGAIDMPCBenchmark(s, 1);
  })->DenseRange(kMinLevel, kMaxLevel, 1)->Iterations(kBenchmarkIterations);

  benchmark::RegisterBenchmark("serial_chain/Pinocchio/CrocoddylIDMPC", [](benchmark::State& s) {
    RunPinocchioIDMPCBenchmark(s, 1);
  })->DenseRange(kMinLevel, kMaxLevel, 1)->Iterations(kBenchmarkIterations);

  #ifdef GA_OCP_HAS_CASADI_BENCH
  benchmark::RegisterBenchmark("serial_chain/CasADi/CrocoddylIDMPC",
                               [](benchmark::State& s) {
    RunPinocchioCasadiIDMPCBenchmark(s, 1);
  })->DenseRange(kMinLevel, kMaxLevel_AD, 1)->Iterations(kBenchmarkIterations);
  #endif
}

}  // namespace

int main(int argc, char** argv) {
  RegisterAll();
  auto benchmark_args = PrepareBenchmarkCsvArgs(argc, argv, "Crocoddyl_fddp_bench");
  int benchmark_argc = benchmark_args.argc();
  char** benchmark_argv = benchmark_args.data();
  benchmark::Initialize(&benchmark_argc, benchmark_argv);
  benchmark::AddCustomContext("FixedIterations", std::to_string(kBenchmarkIterations));
  benchmark::AddCustomContext("SampleBatch", std::to_string(kSampleBatchSize));
  benchmark::AddCustomContext("SeedPolicy", "per-run seed, per-case mixed");
  benchmark::AddCustomContext("FDDPCostTerms", "state_reg,acc_reg,tau_reg");
  benchmark::AddCustomContext("IDMPCCostTerms", "state_reg,acc_reg,tau_reg");
  benchmark::AddCustomContext("CSVOutput", benchmark_args.csv_path);
#ifdef GA_OCP_HAS_CASADI_BENCH
  benchmark::AddCustomContext("CasADiCases", "compiled_in");
#else
  benchmark::AddCustomContext("CasADiCases", "not_built");
#endif
  benchmark::ConsoleReporter console_reporter;
  PivotCsvReporter csv_reporter(
      benchmark_args.csv_path,
      PivotCsvReporterConfig{PivotMetricSource::kCounter, "SolvePerSolverIter_ms", "case", "DOF"});
  CombinedReporter combined_reporter(&console_reporter, &csv_reporter);
  benchmark::RunSpecifiedBenchmarks(&combined_reporter);
  benchmark::Shutdown();
  return 0;
}
