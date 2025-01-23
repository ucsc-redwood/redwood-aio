#include <benchmark/benchmark.h>

#include <CLI/CLI.hpp>

#include "affinity.hpp"
#include "app.hpp"
#include "cifar-sparse/omp/sparse_kernel.hpp"

// ------------------------------------------------------------
// Global variables
// ------------------------------------------------------------

class OMP_CifarSparse : public benchmark::Fixture {
 protected:
  void SetUp(benchmark::State&) override {
    app_data = std::make_unique<cifar_sparse::AppData>(
        std::pmr::new_delete_resource());

#pragma omp parallel
    {
      cifar_sparse::omp::process_stage_1(*app_data);
      cifar_sparse::omp::process_stage_2(*app_data);
      cifar_sparse::omp::process_stage_3(*app_data);
      cifar_sparse::omp::process_stage_4(*app_data);
      cifar_sparse::omp::process_stage_5(*app_data);
      cifar_sparse::omp::process_stage_6(*app_data);
      cifar_sparse::omp::process_stage_7(*app_data);
      cifar_sparse::omp::process_stage_8(*app_data);
      cifar_sparse::omp::process_stage_9(*app_data);
    }
  }
  void TearDown(benchmark::State&) override { app_data.reset(); }

  std::unique_ptr<cifar_sparse::AppData> app_data;
};

#define DEFINE_STAGE_BENCHMARK(stage, core_type)                               \
  static void run_stage_##stage##_##core_type(cifar_sparse::AppData& app_data, \
                                              const std::vector<int>& cores,   \
                                              const int n_threads) {           \
    _Pragma("omp parallel num_threads(n_threads)") {                           \
      bind_thread_to_core(cores);                                              \
      cifar_sparse::omp::process_stage_##stage(app_data);                      \
    }                                                                          \
  }                                                                            \
                                                                               \
  BENCHMARK_DEFINE_F(OMP_CifarSparse, Stage##stage##core_type)                 \
  (benchmark::State & state) {                                                 \
    const auto n_threads = state.range(0);                                     \
    for (auto _ : state) {                                                     \
      run_stage_##stage##_##core_type(                                         \
          *app_data, g_##core_type##_cores, n_threads);                        \
    }                                                                          \
  }                                                                            \
                                                                               \
  void RegisterStage##stage##core_type##BenchmarkWithRange() {                 \
    for (size_t i = 1; i <= g_##core_type##_cores.size(); ++i) {               \
      ::benchmark::internal::RegisterBenchmarkInternal(                        \
          new OMP_CifarSparse_Stage##stage##core_type##_Benchmark())           \
          ->Arg(i)                                                             \
          ->Name("OMP_CifarSparse/Stage" #stage "_" #core_type)                \
          ->Iterations(100)                                                    \
          ->Unit(benchmark::kMillisecond);                                     \
    }                                                                          \
  }

DEFINE_STAGE_BENCHMARK(1, little);
DEFINE_STAGE_BENCHMARK(2, little);
DEFINE_STAGE_BENCHMARK(3, little);
DEFINE_STAGE_BENCHMARK(4, little);
DEFINE_STAGE_BENCHMARK(5, little);
DEFINE_STAGE_BENCHMARK(6, little);
DEFINE_STAGE_BENCHMARK(7, little);
DEFINE_STAGE_BENCHMARK(8, little);
DEFINE_STAGE_BENCHMARK(9, little);

DEFINE_STAGE_BENCHMARK(1, medium);
DEFINE_STAGE_BENCHMARK(2, medium);
DEFINE_STAGE_BENCHMARK(3, medium);
DEFINE_STAGE_BENCHMARK(4, medium);
DEFINE_STAGE_BENCHMARK(5, medium);
DEFINE_STAGE_BENCHMARK(6, medium);
DEFINE_STAGE_BENCHMARK(7, medium);
DEFINE_STAGE_BENCHMARK(8, medium);
DEFINE_STAGE_BENCHMARK(9, medium);

DEFINE_STAGE_BENCHMARK(1, big);
DEFINE_STAGE_BENCHMARK(2, big);
DEFINE_STAGE_BENCHMARK(3, big);
DEFINE_STAGE_BENCHMARK(4, big);
DEFINE_STAGE_BENCHMARK(5, big);
DEFINE_STAGE_BENCHMARK(6, big);
DEFINE_STAGE_BENCHMARK(7, big);
DEFINE_STAGE_BENCHMARK(8, big);
DEFINE_STAGE_BENCHMARK(9, big);

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------

int main(int argc, char** argv) {
  parse_args(argc, argv);

  RegisterStage1littleBenchmarkWithRange();
  RegisterStage2littleBenchmarkWithRange();
  RegisterStage3littleBenchmarkWithRange();
  RegisterStage4littleBenchmarkWithRange();
  RegisterStage5littleBenchmarkWithRange();
  RegisterStage6littleBenchmarkWithRange();
  RegisterStage7littleBenchmarkWithRange();
  RegisterStage8littleBenchmarkWithRange();
  RegisterStage9littleBenchmarkWithRange();

  RegisterStage1mediumBenchmarkWithRange();
  RegisterStage2mediumBenchmarkWithRange();
  RegisterStage3mediumBenchmarkWithRange();
  RegisterStage4mediumBenchmarkWithRange();
  RegisterStage5mediumBenchmarkWithRange();
  RegisterStage6mediumBenchmarkWithRange();
  RegisterStage7mediumBenchmarkWithRange();
  RegisterStage8mediumBenchmarkWithRange();
  RegisterStage9mediumBenchmarkWithRange();

  RegisterStage1bigBenchmarkWithRange();
  RegisterStage2bigBenchmarkWithRange();
  RegisterStage3bigBenchmarkWithRange();
  RegisterStage4bigBenchmarkWithRange();
  RegisterStage5bigBenchmarkWithRange();
  RegisterStage6bigBenchmarkWithRange();
  RegisterStage7bigBenchmarkWithRange();
  RegisterStage8bigBenchmarkWithRange();
  RegisterStage9bigBenchmarkWithRange();

  // Initialize and run benchmarks
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  return 0;
}
