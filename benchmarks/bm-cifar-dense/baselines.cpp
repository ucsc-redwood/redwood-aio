#include <benchmark/benchmark.h>

#include <CLI/CLI.hpp>
#include <memory>
#include <thread>
// #include <vector>

// #include "affinity.hpp"
#include "app.hpp"
#include "cifar-dense/omp/dense_kernel.hpp"

// // ----------------------------------------------------------------
// // 1) The underlying run functions for pinned/unrestricted
// // ----------------------------------------------------------------
// static void run_baseline_pinned(cifar_dense::AppData& app_data,
//                                 const std::vector<int>& cores,
//                                 const int n_threads) {
// #pragma omp parallel num_threads(n_threads)
//   {
//     bind_thread_to_core(cores);

//     cifar_dense::omp::process_stage_1(app_data);
//     cifar_dense::omp::process_stage_2(app_data);
//     cifar_dense::omp::process_stage_3(app_data);
//     cifar_dense::omp::process_stage_4(app_data);
//     cifar_dense::omp::process_stage_5(app_data);
//     cifar_dense::omp::process_stage_6(app_data);
//     cifar_dense::omp::process_stage_7(app_data);
//     cifar_dense::omp::process_stage_8(app_data);
//     cifar_dense::omp::process_stage_9(app_data);
//   }
// }

static void run_baseline_unrestricted(cifar_dense::AppData& app_data,
                                      const int n_threads) {
#pragma omp parallel num_threads(n_threads)
  {
    cifar_dense::omp::process_stage_1(app_data);
    cifar_dense::omp::process_stage_2(app_data);
    cifar_dense::omp::process_stage_3(app_data);
    cifar_dense::omp::process_stage_4(app_data);
    cifar_dense::omp::process_stage_5(app_data);
    cifar_dense::omp::process_stage_6(app_data);
    cifar_dense::omp::process_stage_7(app_data);
    cifar_dense::omp::process_stage_8(app_data);
    cifar_dense::omp::process_stage_9(app_data);
  }
}

// // ----------------------------------------------------------------
// // 2) Free-function benchmark for pinned. We'll dynamically register it.
// // ----------------------------------------------------------------
// static void OMP_BaselinePinned_Benchmark(benchmark::State& state,
//                                          const std::vector<int>& pinnedCores)
//                                          {
//   // Equivalent to "SetUp"
//   cifar_dense::AppData app_data{std::pmr::new_delete_resource()};

//   const auto n_threads = state.range(0);
//   for (auto _ : state) {
//     run_baseline_pinned(app_data, pinnedCores, n_threads);
//   }
// }

// // ----------------------------------------------------------------
// // 3) Helper function to register pinned benchmarks for 1..N threads
// // ----------------------------------------------------------------
// static void RegisterPinnedBenchmark(const std::vector<int>& cores,
//                                     const std::string& coreType) {
//   const auto size = static_cast<int>(cores.size());
//   for (int i = 1; i <= size; ++i) {
//     benchmark::RegisterBenchmark(
//         ("OMP_CifarDense/Baseline_Pinned_" + coreType).c_str(),
//         [=](benchmark::State& st) { OMP_BaselinePinned_Benchmark(st, cores);
//         })
//         ->Arg(i)
//         ->Unit(benchmark::kMillisecond);
//   }
// }

// ----------------------------------------------------------------
// 4) For the unrestricted case, we *can* either do a fixture, or
//    simply do another free function. If we do want a fixture:
// ----------------------------------------------------------------
class OMP_CifarDense : public benchmark::Fixture {
 protected:
  void SetUp(benchmark::State&) override {
    app_data =
        std::make_unique<cifar_dense::AppData>(std::pmr::new_delete_resource());
  }
  void TearDown(benchmark::State&) override { app_data.reset(); }

  std::unique_ptr<cifar_dense::AppData> app_data;
};

BENCHMARK_DEFINE_F(OMP_CifarDense, Baseline)
(benchmark::State& state) {
  const auto n_threads = state.range(0);
  for (auto _ : state) {
    run_baseline_unrestricted(*app_data, n_threads);
  }
}

BENCHMARK_REGISTER_F(OMP_CifarDense, Baseline)
    ->DenseRange(1, std::thread::hardware_concurrency())
    ->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// 5) The main
// ----------------------------------------------------------------
int main(int argc, char** argv) {
  // parse_args(...) presumably loads g_little_cores, g_medium_cores,
  // g_big_cores
  parse_args(argc, argv);

  // RegisterPinnedBenchmark(g_little_cores, "Little");
  // RegisterPinnedBenchmark(g_medium_cores, "Medium");
  // RegisterPinnedBenchmark(g_big_cores, "Big");

  // Then run the normal Google Benchmark suite
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
