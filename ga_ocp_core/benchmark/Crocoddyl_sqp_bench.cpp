#include <benchmark/benchmark.h>

#include <chrono>
#include <mim_solvers/sqp.hpp>

#include "ga_ocp/BenchUtils.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using DurationSeconds = std::chrono::duration<double>;

constexpr int kMinLevel = 1;
constexpr int kMaxLevel = 7;
constexpr int kMaxLevel_AD = 7;
constexpr int kSampleBatchSize = 10;
constexpr int kBenchmarkIterations = 10;

template <typename BuildProblemFn>
void RunSQPBenchmark(benchmark::State& state, int bf, std::uint32_t stream_id,
                     BuildProblemFn&& build_problem) {
  const int level = static_cast<int>(state.range(0));
  const int dof = DofFromLevel(level);
  const FDDPBenchConfig config{};
  const std::uint32_t seed =
      MixBenchmarkSeed(BenchmarkRunSeed(), stream_id, static_cast<std::uint32_t>(bf),
                       static_cast<std::uint32_t>(dof));
  FDDPSampleBatch samples = MakeFDDPSamples(dof, kSampleBatchSize, seed, config.position_limit);

  state.counters["DOF"] = static_cast<double>(dof);
  state.counters["Horizon"] = static_cast<double>(config.horizon);
  state.counters["MaxIter"] = static_cast<double>(config.max_iterations);
  double total_solve_seconds = 0.0;
  double total_cost = 0.0;
  double total_kkt = 0.0;
  double total_gap_norm = 0.0;
  double total_merit = 0.0;
  double total_solver_iterations = 0.0;
  std::uint64_t converged_count = 0;

  for (auto _ : state) {
    const std::size_t i = samples.cursor;
    auto problem = build_problem(dof, config, samples.x0[i], samples.x_target[i]);
    mim_solvers::SolverSQP solver(problem);
    solver.set_termination_tolerance(1e-4);

    const auto start = Clock::now();
    const bool converged =
        solver.solve({}, {}, static_cast<unsigned int>(config.max_iterations), false);
    const double solve_seconds = DurationSeconds(Clock::now() - start).count();

    total_solve_seconds += solve_seconds;
    total_cost += solver.get_cost();
    total_kkt += solver.get_KKT();
    total_gap_norm += solver.get_gap_norm();
    total_merit += solver.get_merit();
    total_solver_iterations += static_cast<double>(solver.get_iter());
    converged_count += converged ? 1u : 0u;

    benchmark::DoNotOptimize(converged);
    benchmark::DoNotOptimize(solver.get_cost());
    benchmark::DoNotOptimize(solver.get_KKT());
    benchmark::DoNotOptimize(solver.get_gap_norm());
    benchmark::DoNotOptimize(solver.get_merit());

    samples.cursor = (samples.cursor + 1) % samples.x0.size();
  }

  const double samples_count = static_cast<double>(state.iterations());
  state.counters["SolvePerSolverIter_ms"] =
      total_solver_iterations == 0.0 ? 0.0 : total_solve_seconds * 1e3 / total_solver_iterations;
  state.counters["AvgIter"] = samples_count == 0.0 ? 0.0 : total_solver_iterations / samples_count;
  state.counters["Converged"] =
      benchmark::Counter(static_cast<double>(converged_count), benchmark::Counter::kAvgIterations);
  state.counters["Cost"] = samples_count == 0.0 ? 0.0 : total_cost / samples_count;
  state.counters["KKT"] = samples_count == 0.0 ? 0.0 : total_kkt / samples_count;
  state.counters["GapNorm"] = samples_count == 0.0 ? 0.0 : total_gap_norm / samples_count;
  state.counters["Merit"] = samples_count == 0.0 ? 0.0 : total_merit / samples_count;
}

void RunGAFwdDynSQPBenchmark(benchmark::State& state, int bf) {
  RunSQPBenchmark(state, bf, 0x5A50u,
                  [bf](int dof, const FDDPBenchConfig& config, const Eigen::VectorXd& x0,
                       const Eigen::VectorXd& x_target) {
                    const TreeTemplateParams params = MakeBenchTreeParams(bf, dof);
                    const Model<double> model =
                        MakeGABenchModel(params, "tetrapga_mim_sqp_benchmark");
                    return BuildGAFDDPProblem(model, x0, x_target, config);
                  });
}

void RunPinocchioFwdDynSQPBenchmark(benchmark::State& state, int bf) {
  RunSQPBenchmark(state, bf, 0x5A50u,
                  [bf](int dof, const FDDPBenchConfig& config, const Eigen::VectorXd& x0,
                       const Eigen::VectorXd& x_target) {
                    const TreeTemplateParams params = MakeBenchTreeParams(bf, dof);
                    const pinocchio::Model model = BuildPinModel(params);
                    return BuildPinFDDPProblem(model, x0, x_target, config);
                  });
}

void RunGAIDMPCSQPBenchmark(benchmark::State& state, int bf) {
  RunSQPBenchmark(state, bf, 0x5A51u,
                  [bf](int dof, const FDDPBenchConfig& config, const Eigen::VectorXd& x0,
                       const Eigen::VectorXd& x_target) {
                    const TreeTemplateParams params = MakeBenchTreeParams(bf, dof);
                    const Model<double> model =
                        MakeGABenchModel(params, "tetrapga_mim_sqp_idmpc_benchmark");
                    return BuildGAIDMPCProblem(model, x0, x_target, config);
                  });
}

void RunPinocchioIDMPCSQPBenchmark(benchmark::State& state, int bf) {
  RunSQPBenchmark(state, bf, 0x5A51u,
                  [bf](int dof, const FDDPBenchConfig& config, const Eigen::VectorXd& x0,
                       const Eigen::VectorXd& x_target) {
                    const TreeTemplateParams params = MakeBenchTreeParams(bf, dof);
                    const pinocchio::Model model = BuildPinModel(params);
                    return BuildPinIDMPCProblem(model, x0, x_target, config);
                  });
}

#ifdef GA_OCP_HAS_CASADI_BENCH
void RunPinocchioCasadiFwdDynSQPBenchmark(benchmark::State& state, int bf) {
  RunSQPBenchmark(state, bf, 0x5A52u,
                  [bf](int dof, const FDDPBenchConfig& config, const Eigen::VectorXd& x0,
                       const Eigen::VectorXd& x_target) {
                    const TreeTemplateParams params = MakeBenchTreeParams(bf, dof);
                    const pinocchio::Model model = BuildPinModel(params);
                    auto autodiff = std::make_shared<InlineAutoDiffABADerivatives>(
                        model, "tetrapga_pinocchio_casadi_sqp_bf" + std::to_string(bf) + "_dof" +
                                   std::to_string(dof));
                    return BuildPinCasadiFDDPProblem(model, x0, x_target, config, autodiff);
                  });
}

void RunPinocchioCasadiIDMPCSQPBenchmark(benchmark::State& state, int bf) {
  RunSQPBenchmark(state, bf, 0x5A53u,
                  [bf](int dof, const FDDPBenchConfig& config, const Eigen::VectorXd& x0,
                       const Eigen::VectorXd& x_target) {
                    const TreeTemplateParams params = MakeBenchTreeParams(bf, dof);
                    const pinocchio::Model model = BuildPinModel(params);
                    auto autodiff = std::make_shared<InlineAutoDiffRNEADerivatives>(
                        model, "tetrapga_pinocchio_casadi_sqp_idmpc_bf" + std::to_string(bf) +
                                   "_dof" + std::to_string(dof));
                    return BuildPinCasadiIDMPCProblem(model, x0, x_target, config, autodiff);
                  });
}
#endif

void RegisterAll() {
  benchmark::RegisterBenchmark("serial_chain/TetraPGA/MimSQP/FwdDynamics",
                               [](benchmark::State& s) {
    RunGAFwdDynSQPBenchmark(s, 1);
  })->DenseRange(kMinLevel, kMaxLevel, 1)->Iterations(kBenchmarkIterations);

  benchmark::RegisterBenchmark("serial_chain/Pinocchio/MimSQP/FwdDynamics",
                               [](benchmark::State& s) {
    RunPinocchioFwdDynSQPBenchmark(s, 1);
  })->DenseRange(kMinLevel, kMaxLevel, 1)->Iterations(kBenchmarkIterations);

#ifdef GA_OCP_HAS_CASADI_BENCH
  benchmark::RegisterBenchmark("serial_chain/CasADi/MimSQP/FwdDynamics",
                               [](benchmark::State& s) {
    RunPinocchioCasadiFwdDynSQPBenchmark(s, 1);
  })->DenseRange(kMinLevel, kMaxLevel_AD, 1)->Iterations(kBenchmarkIterations);
#endif

  benchmark::RegisterBenchmark("serial_chain/TetraPGA/MimSQP/IDMPC", [](benchmark::State& s) {
    RunGAIDMPCSQPBenchmark(s, 1);
  })->DenseRange(kMinLevel, kMaxLevel, 1)->Iterations(kBenchmarkIterations);

  benchmark::RegisterBenchmark("serial_chain/Pinocchio/MimSQP/IDMPC",
                               [](benchmark::State& s) {
    RunPinocchioIDMPCSQPBenchmark(s, 1);
  })->DenseRange(kMinLevel, kMaxLevel, 1)->Iterations(kBenchmarkIterations);

#ifdef GA_OCP_HAS_CASADI_BENCH
  benchmark::RegisterBenchmark("serial_chain/CasADi/MimSQP/IDMPC",
                               [](benchmark::State& s) {
    RunPinocchioCasadiIDMPCSQPBenchmark(s, 1);
  })->DenseRange(kMinLevel, kMaxLevel_AD, 1)->Iterations(kBenchmarkIterations);
#endif
}

}  // namespace

int main(int argc, char** argv) {
  RegisterAll();
  auto benchmark_args = PrepareBenchmarkCsvArgs(argc, argv, "Crocoddyl_sqp_bench");
  int benchmark_argc = benchmark_args.argc();
  char** benchmark_argv = benchmark_args.data();
  benchmark::Initialize(&benchmark_argc, benchmark_argv);
  benchmark::AddCustomContext("FixedIterations", std::to_string(kBenchmarkIterations));
  benchmark::AddCustomContext("SampleBatch", std::to_string(kSampleBatchSize));
  benchmark::AddCustomContext("SeedPolicy", "per-run seed, per-case mixed");
  benchmark::AddCustomContext("SQPCostTerms", "state_reg,acc_reg,tau_reg");
  benchmark::AddCustomContext("IDMPCCostTerms", "state_reg,acc_reg,tau_reg");
  benchmark::AddCustomContext("CSVOutput", benchmark_args.csv_path);
#ifdef GA_OCP_HAS_CASADI_BENCH
  benchmark::AddCustomContext("CasADiCases", "fwd_dynamics,idmpc");
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
