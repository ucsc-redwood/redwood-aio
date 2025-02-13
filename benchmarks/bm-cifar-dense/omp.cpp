// #include <benchmark/benchmark.h>

// #include <thread>

// #include "affinity.hpp"
// #include "app.hpp"
// #include "cifar-dense/omp/dense_kernel.hpp"
// #include "third-party/CLI11.hpp"

// // ------------------------------------------------------------
// // Global variables
// // ------------------------------------------------------------

// static void run_baseline_unrestricted(cifar_dense::AppData& app_data, const int n_threads) {
// #pragma omp parallel num_threads(n_threads)
//   {
//     cifar_dense::omp::run_stage<1>(app_data);
//     cifar_dense::omp::run_stage<2>(app_data);
//     cifar_dense::omp::run_stage<3>(app_data);
//     cifar_dense::omp::run_stage<4>(app_data);
//     cifar_dense::omp::run_stage<5>(app_data);
//     cifar_dense::omp::run_stage<6>(app_data);
//     cifar_dense::omp::run_stage<7>(app_data);
//     cifar_dense::omp::run_stage<8>(app_data);
//     cifar_dense::omp::run_stage<9>(app_data);
//   }
// }

// class OMP_CifarDense : public benchmark::Fixture {
//  protected:
//   void SetUp(benchmark::State&) override {
//     app_data = std::make_unique<cifar_dense::AppData>(std::pmr::new_delete_resource());

//     run_baseline_unrestricted(*app_data, std::thread::hardware_concurrency());
//   }
//   void TearDown(benchmark::State&) override { app_data.reset(); }

//   std::unique_ptr<cifar_dense::AppData> app_data;
// };

// // ------------------------------------------------------------
// // Baseline benchmarks
// // ------------------------------------------------------------

// BENCHMARK_DEFINE_F(OMP_CifarDense, Baseline)
// (benchmark::State& state) {
//   const auto n_threads = state.range(0);
//   for (auto _ : state) {
//     run_baseline_unrestricted(*app_data, n_threads);
//   }
// }

// BENCHMARK_REGISTER_F(OMP_CifarDense, Baseline)
//     ->DenseRange(1, std::thread::hardware_concurrency())
//     ->Unit(benchmark::kMillisecond);

// // ------------------------------------------------------------
// // Stage benchmarks
// // ------------------------------------------------------------

// template <int stage, CoreType core_type>
// static void run_bm_stage(cifar_dense::AppData& app_data, const int n_threads) {
// #pragma omp parallel num_threads(n_threads)
//   { cifar_dense::omp::run_stage<stage>(app_data); }
// }

// BENCHMARK_DEFINE_F(OMP_CifarDense, Stage1little)
// (benchmark::State& state) {
//   const auto n_threads = state.range(0);
//   for (auto _ : state) {
//     bind_thread_to_cores(g_little_cores);
//     run_bm_stage<1, CoreType::kLittle>(*app_data, n_threads);
//   }
// }

// // void RegisterStage1littleBenchmarkWithRange() {
// //   for (size_t i = 1; i <= g_little_cores.size(); ++i) {
// //     ::benchmark::internal::RegisterBenchmarkInternal(new
// OMP_CifarDense_Stage1little_Benchmark())
// //         ->Arg(i)
// //         ->Name("OMP_CifarDense/Stage1/little")
// //         ->Unit(benchmark::kMillisecond);
// //   }
// // }

// // #define DEFINE_STAGE_BENCHMARK(stage, core_type)                                            \
// //   static void run_stage_##stage##_##core_type(                                              \
// //       cifar_dense::AppData& app_data, const std::vector<int>& cores, const int n_threads) { \
// //     _Pragma("omp parallel num_threads(n_threads)") {                                        \
// //       bind_thread_to_cores(cores);                                                          \
// //       cifar_dense::omp::process_stage_##stage(app_data);                                    \
// //     }                                                                                       \
// //   }                                                                                         \
// //                                                                                             \
// //   BENCHMARK_DEFINE_F(OMP_CifarDense, Stage##stage##core_type)                               \
// //   (benchmark::State & state) {                                                              \
// //     const auto n_threads = state.range(0);                                                  \
// //     for (auto _ : state) {                                                                  \
// //       run_stage_##stage##_##core_type(*app_data, g_##core_type##_cores, n_threads);         \
// //     }                                                                                       \
// //   }                                                                                         \
// //                                                                                             \
// //   void RegisterStage##stage##core_type##BenchmarkWithRange() {                              \
// //     for (size_t i = 1; i <= g_##core_type##_cores.size(); ++i) {                            \
// //       ::benchmark::internal::RegisterBenchmarkInternal(                                     \
// //           new OMP_CifarDense_Stage##stage##core_type##_Benchmark())                         \
// //           ->Arg(i)                                                                          \
// //           ->Name("OMP_CifarDense/Stage" #stage "_" #core_type)                              \
// //           ->Unit(benchmark::kMillisecond);                                                  \
// //     }                                                                                       \
// //   }

// // DEFINE_STAGE_BENCHMARK(1, little);
// // DEFINE_STAGE_BENCHMARK(2, little);
// // DEFINE_STAGE_BENCHMARK(3, little);
// // DEFINE_STAGE_BENCHMARK(4, little);
// // DEFINE_STAGE_BENCHMARK(5, little);
// // DEFINE_STAGE_BENCHMARK(6, little);
// // DEFINE_STAGE_BENCHMARK(7, little);
// // DEFINE_STAGE_BENCHMARK(8, little);
// // DEFINE_STAGE_BENCHMARK(9, little);

// // DEFINE_STAGE_BENCHMARK(1, medium);
// // DEFINE_STAGE_BENCHMARK(2, medium);
// // DEFINE_STAGE_BENCHMARK(3, medium);
// // DEFINE_STAGE_BENCHMARK(4, medium);
// // DEFINE_STAGE_BENCHMARK(5, medium);
// // DEFINE_STAGE_BENCHMARK(6, medium);
// // DEFINE_STAGE_BENCHMARK(7, medium);
// // DEFINE_STAGE_BENCHMARK(8, medium);
// // DEFINE_STAGE_BENCHMARK(9, medium);

// // DEFINE_STAGE_BENCHMARK(1, big);
// // DEFINE_STAGE_BENCHMARK(2, big);
// // DEFINE_STAGE_BENCHMARK(3, big);
// // DEFINE_STAGE_BENCHMARK(4, big);
// // DEFINE_STAGE_BENCHMARK(5, big);
// // DEFINE_STAGE_BENCHMARK(6, big);
// // DEFINE_STAGE_BENCHMARK(7, big);
// // DEFINE_STAGE_BENCHMARK(8, big);
// // DEFINE_STAGE_BENCHMARK(9, big);

// // ------------------------------------------------------------
// // Main
// // ------------------------------------------------------------

// int main(int argc, char** argv) {
//   parse_args(argc, argv);

//   // RegisterStage1littleBenchmarkWithRange();
//   // RegisterStage1mediumBenchmarkWithRange();
//   // RegisterStage1bigBenchmarkWithRange();

//   // RegisterStage2littleBenchmarkWithRange();
//   // RegisterStage2mediumBenchmarkWithRange();
//   // RegisterStage2bigBenchmarkWithRange();

//   // RegisterStage3littleBenchmarkWithRange();
//   // RegisterStage3mediumBenchmarkWithRange();
//   // RegisterStage3bigBenchmarkWithRange();

//   // RegisterStage4littleBenchmarkWithRange();
//   // RegisterStage4mediumBenchmarkWithRange();
//   // RegisterStage4bigBenchmarkWithRange();

//   // RegisterStage5littleBenchmarkWithRange();
//   // RegisterStage5mediumBenchmarkWithRange();
//   // RegisterStage5bigBenchmarkWithRange();

//   // RegisterStage6littleBenchmarkWithRange();
//   // RegisterStage6mediumBenchmarkWithRange();
//   // RegisterStage6bigBenchmarkWithRange();

//   // RegisterStage7littleBenchmarkWithRange();
//   // RegisterStage7mediumBenchmarkWithRange();
//   // RegisterStage7bigBenchmarkWithRange();

//   // RegisterStage8littleBenchmarkWithRange();
//   // RegisterStage8mediumBenchmarkWithRange();
//   // RegisterStage8bigBenchmarkWithRange();

//   // RegisterStage9littleBenchmarkWithRange();
//   // RegisterStage9mediumBenchmarkWithRange();
//   // RegisterStage9bigBenchmarkWithRange();

//   // Initialize and run benchmarks
//   benchmark::Initialize(&argc, argv);
//   benchmark::RunSpecifiedBenchmarks();
//   benchmark::Shutdown();

//   return 0;
// }

#include <benchmark/benchmark.h>

#include <string>
#include <thread>
#include <vector>

#include "affinity.hpp"
#include "app.hpp"
#include "cifar-dense/omp/dense_kernel.hpp"

static void run_baseline_unrestricted(cifar_dense::AppData& app_data, const int n_threads) {
#pragma omp parallel num_threads(n_threads)
  {
    cifar_dense::omp::run_stage<1>(app_data);
    cifar_dense::omp::run_stage<2>(app_data);
    cifar_dense::omp::run_stage<3>(app_data);
    cifar_dense::omp::run_stage<4>(app_data);
    cifar_dense::omp::run_stage<5>(app_data);
    cifar_dense::omp::run_stage<6>(app_data);
    cifar_dense::omp::run_stage<7>(app_data);
    cifar_dense::omp::run_stage<8>(app_data);
    cifar_dense::omp::run_stage<9>(app_data);
  }
}

class OMP_CifarDense : public benchmark::Fixture {
 protected:
  void SetUp(benchmark::State&) override {
    app_data = std::make_unique<cifar_dense::AppData>(std::pmr::new_delete_resource());

    run_baseline_unrestricted(*app_data, std::thread::hardware_concurrency());
  }
  void TearDown(benchmark::State&) override { app_data.reset(); }

  std::unique_ptr<cifar_dense::AppData> app_data;
};

// ------------------------------------------------------------
// Baseline benchmarks
// ------------------------------------------------------------

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

// ------------------------------------------------------------
// Stage benchmarks
// ------------------------------------------------------------

template <int Stage, typename CoreType>
void run_bm_stage(cifar_dense::AppData& app_data, const std::vector<int>& cores, int n_threads) {
#pragma omp parallel num_threads(n_threads)
  {
    bind_thread_to_cores(cores);
    cifar_dense::omp::run_stage<Stage>(app_data);
  }
}

template <int Stage, typename CoreType>
void register_stage_benchmark(const std::vector<int>& cores) {
  for (size_t i = 1; i <= cores.size(); ++i) {
    std::string benchmark_name =
        "OMP_CifarDense/Stage" + std::to_string(Stage) + "_" + CoreType::name();

    benchmark::RegisterBenchmark(benchmark_name.c_str(),
                                 [i, cores](benchmark::State& state) {
                                   auto mr = std::pmr::new_delete_resource();
                                   cifar_dense::AppData app_data(mr);

                                   for (auto _ : state) {
                                     run_bm_stage<Stage, CoreType>(
                                         app_data, cores, static_cast<int>(i));
                                   }
                                 })
        ->Arg(i)
        ->Unit(benchmark::kMillisecond);
  }
}

struct LittleCore {
  static constexpr const char* name() { return "little"; }
};
struct MediumCore {
  static constexpr const char* name() { return "medium"; }
};
struct BigCore {
  static constexpr const char* name() { return "big"; }
};

int main(int argc, char** argv) {
  parse_args(argc, argv);

#define REGISTER_STAGE(STAGE)                                  \
  register_stage_benchmark<STAGE, LittleCore>(g_little_cores); \
  register_stage_benchmark<STAGE, MediumCore>(g_medium_cores); \
  register_stage_benchmark<STAGE, BigCore>(g_big_cores)

  REGISTER_STAGE(1);
  REGISTER_STAGE(2);
  REGISTER_STAGE(3);
  REGISTER_STAGE(4);
  REGISTER_STAGE(5);
  REGISTER_STAGE(6);
  REGISTER_STAGE(7);
  REGISTER_STAGE(8);
  REGISTER_STAGE(9);

#undef REGISTER_STAGE

  benchmark::Initialize(nullptr, nullptr);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
