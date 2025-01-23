#include <benchmark/benchmark.h>

#include <CLI/CLI.hpp>

#include "affinity.hpp"
#include "app.hpp"
#include "cifar-dense/omp/dense_kernel.hpp"

// ------------------------------------------------------------
// Global variables
// ------------------------------------------------------------

class OMP_CifarDense : public benchmark::Fixture {
 protected:
  void SetUp(benchmark::State&) override {
    app_data =
        std::make_unique<cifar_dense::AppData>(std::pmr::new_delete_resource());

#pragma omp parallel
    {
      cifar_dense::omp::process_stage_1(*app_data);
      cifar_dense::omp::process_stage_2(*app_data);
      cifar_dense::omp::process_stage_3(*app_data);
      cifar_dense::omp::process_stage_4(*app_data);
      cifar_dense::omp::process_stage_5(*app_data);
      cifar_dense::omp::process_stage_6(*app_data);
      cifar_dense::omp::process_stage_7(*app_data);
      cifar_dense::omp::process_stage_8(*app_data);
      cifar_dense::omp::process_stage_9(*app_data);
    }
  }
  void TearDown(benchmark::State&) override { app_data.reset(); }

  std::unique_ptr<cifar_dense::AppData> app_data;
};

#define DEFINE_STAGE_BENCHMARK(stage, core_type)                              \
  static void run_stage_##stage##_##core_type(cifar_dense::AppData& app_data, \
                                              const std::vector<int>& cores,  \
                                              const int n_threads) {          \
    _Pragma("omp parallel num_threads(n_threads)") {                          \
      bind_thread_to_core(cores);                                             \
      cifar_dense::omp::process_stage_##stage(app_data);                      \
    }                                                                         \
  }                                                                           \
                                                                              \
  BENCHMARK_DEFINE_F(OMP_CifarDense, Stage##stage##core_type)                 \
  (benchmark::State & state) {                                                \
    const auto n_threads = state.range(0);                                    \
    for (auto _ : state) {                                                    \
      run_stage_##stage##_##core_type(                                        \
          *app_data, g_##core_type##_cores, n_threads);                       \
    }                                                                         \
  }                                                                           \
                                                                              \
  void RegisterStage##stage##core_type##BenchmarkWithRange() {                \
    for (size_t i = 1; i <= g_##core_type##_cores.size(); ++i) {              \
      ::benchmark::internal::RegisterBenchmarkInternal(                       \
          new OMP_CifarDense_Stage##stage##core_type##_Benchmark())           \
          ->Arg(i)                                                            \
          ->Name("OMP_CifarDense/Stage" #stage "_" #core_type)                \
          ->Unit(benchmark::kMillisecond);                                    \
    }                                                                         \
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
  RegisterStage1mediumBenchmarkWithRange();
  RegisterStage1bigBenchmarkWithRange();

  RegisterStage2littleBenchmarkWithRange();
  RegisterStage2mediumBenchmarkWithRange();
  RegisterStage2bigBenchmarkWithRange();

  RegisterStage3littleBenchmarkWithRange();
  RegisterStage3mediumBenchmarkWithRange();
  RegisterStage3bigBenchmarkWithRange();

  RegisterStage4littleBenchmarkWithRange();
  RegisterStage4mediumBenchmarkWithRange();
  RegisterStage4bigBenchmarkWithRange();

  RegisterStage5littleBenchmarkWithRange();
  RegisterStage5mediumBenchmarkWithRange();
  RegisterStage5bigBenchmarkWithRange();

  RegisterStage6littleBenchmarkWithRange();
  RegisterStage6mediumBenchmarkWithRange();
  RegisterStage6bigBenchmarkWithRange();

  RegisterStage7littleBenchmarkWithRange();
  RegisterStage7mediumBenchmarkWithRange();
  RegisterStage7bigBenchmarkWithRange();

  RegisterStage8littleBenchmarkWithRange();
  RegisterStage8mediumBenchmarkWithRange();
  RegisterStage8bigBenchmarkWithRange();

  RegisterStage9littleBenchmarkWithRange();
  RegisterStage9mediumBenchmarkWithRange();
  RegisterStage9bigBenchmarkWithRange();

  // Initialize and run benchmarks
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  return 0;
}
