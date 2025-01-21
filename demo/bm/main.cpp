#include <benchmark/benchmark.h>

#include <CLI/CLI.hpp>
#include <affinity.hpp>
#include <cifar_dense_kernel.hpp>
#include <cifar_sparse_kernel.hpp>
#include <memory>

#include "conf.hpp"

// ------------------------------------------------------------
// Global variables
// ------------------------------------------------------------

Device g_device;

class OMP_CifarDense : public benchmark::Fixture {
 protected:
  void SetUp(benchmark::State&) override {
    app_data =
        std::make_unique<cifar_dense::AppData>(std::pmr::new_delete_resource());
  }

  void TearDown(benchmark::State&) override { app_data.reset(); }

  std::unique_ptr<cifar_dense::AppData> app_data;
};

// ------------------------------------------------------------
// Helper macros for stage benchmarks
// ------------------------------------------------------------

#define DEFINE_STAGE_RUNNER(stage_num)                              \
  static void run_stage_##stage_num(cifar_dense::AppData& app_data, \
                                    int n_threads) {                \
    _Pragma("omp parallel num_threads(n_threads)") {                \
      cifar_dense::omp::process_stage_##stage_num(app_data);        \
    }                                                               \
  }

#define DEFINE_STAGE_BENCHMARK(stage_num)                \
  BENCHMARK_DEFINE_F(OMP_CifarDense, Stage##stage_num)   \
  (benchmark::State & state) {                           \
    auto n_threads = state.range(0);                     \
    for (auto _ : state) {                               \
      run_stage_##stage_num(*app_data, n_threads);       \
    }                                                    \
  }                                                      \
                                                         \
  BENCHMARK_REGISTER_F(OMP_CifarDense, Stage##stage_num) \
      ->DenseRange(1, 6)                                 \
      ->Unit(benchmark::kMillisecond);

// // ------------------------------------------------------------
// // Baseline 1: Unpinned using all cores default
// // ------------------------------------------------------------

// static void run_baseline(cifar_dense::AppData& app_data, int n_threads) {
// #pragma omp parallel num_threads(n_threads)
//   {
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

// BENCHMARK_DEFINE_F(OMP_CifarDense, Baseline)(benchmark::State& state) {
//   auto n_threads = state.range(0);

//   for (auto _ : state) {
//     run_baseline(*app_data, n_threads);
//   }
// }

// ------------------------------------------------------------
// Baseline 2: Pinned using all cores default
// ------------------------------------------------------------

static void run_baseline_pinned(cifar_dense::AppData& app_data,
                                int n_threads,
                                const std::vector<int>& all_cores) {
#pragma omp parallel num_threads(n_threads)
  {
    bind_thread_to_core(all_cores);
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

BENCHMARK_DEFINE_F(OMP_CifarDense, BaselinePinned)(benchmark::State& state) {
  auto n_threads = state.range(0);

  for (auto _ : state) {
    run_baseline_pinned(*app_data, n_threads, g_device.get_pinable_cores());
  }
}

void RegisterPinnedBenchmarkWithRange(const std::vector<int>& pinable_cores) {
  for (size_t i = 1; i <= pinable_cores.size(); ++i) {
    ::benchmark::internal::RegisterBenchmarkInternal(
        new OMP_CifarDense_BaselinePinned_Benchmark())
        ->Arg(i)
        ->Name("OMP_CifarDense/BaselinePinned")
        ->Unit(benchmark::kMillisecond);
  }
}

// ------------------------------------------------------------
// Baseline 3: Little cores only
// ------------------------------------------------------------

static void run_baseline_little(cifar_dense::AppData& app_data,
                                const std::vector<int>& cores,
                                const int n_threads) {
#pragma omp parallel num_threads(n_threads)
  {
    bind_thread_to_core(cores);
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

BENCHMARK_DEFINE_F(OMP_CifarDense, BaselineLittle)
(benchmark::State& state) {
  auto cores = g_device.get_pinable_cores(kLittleCoreType);

  const auto n_threads = state.range(0);
  for (auto _ : state) {
    run_baseline_little(*app_data, cores, n_threads);
  }
}

void RegisterLittleBenchmarkWithRange(
    const std::vector<int>& pinable_little_cores) {
  for (size_t i = 1; i <= pinable_little_cores.size(); ++i) {
    ::benchmark::internal::RegisterBenchmarkInternal(
        new OMP_CifarDense_BaselineLittle_Benchmark())
        ->Arg(i)
        ->Name("OMP_CifarDense/BaselineLittleOnly")
        ->Unit(benchmark::kMillisecond);
  }
}

// ------------------------------------------------------------
// Baseline 4: Medium cores only
// ------------------------------------------------------------

static void run_baseline_medium(cifar_dense::AppData& app_data,
                                const std::vector<int>& cores,
                                const int n_threads) {
#pragma omp parallel num_threads(n_threads)
  {
    bind_thread_to_core(cores);
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

BENCHMARK_DEFINE_F(OMP_CifarDense, BaselineMedium)
(benchmark::State& state) {
  auto cores = g_device.get_pinable_cores(kMediumCoreType);

  const auto n_threads = state.range(0);
  for (auto _ : state) {
    run_baseline_medium(*app_data, cores, n_threads);
  }
}

void RegisterMediumBenchmarkWithRange(
    const std::vector<int>& pinable_medium_cores) {
  for (size_t i = 1; i <= pinable_medium_cores.size(); ++i) {
    ::benchmark::internal::RegisterBenchmarkInternal(
        new OMP_CifarDense_BaselineMedium_Benchmark())
        ->Arg(i)
        ->Name("OMP_CifarDense/BaselineMediumOnly")
        ->Unit(benchmark::kMillisecond);
  }
}

// ------------------------------------------------------------
// Baseline 5: Big cores only
// ------------------------------------------------------------

static void run_baseline_big(cifar_dense::AppData& app_data,
                             const std::vector<int>& cores,
                             const int n_threads) {
#pragma omp parallel num_threads(n_threads)
  {
    bind_thread_to_core(cores);
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

BENCHMARK_DEFINE_F(OMP_CifarDense, BaselineBig)
(benchmark::State& state) {
  auto cores = g_device.get_pinable_cores(kBigCoreType);

  const auto n_threads = state.range(0);
  for (auto _ : state) {
    run_baseline_big(*app_data, cores, n_threads);
  }
}

void RegisterBigBenchmarkWithRange(const std::vector<int>& pinable_big_cores) {
  for (size_t i = 1; i <= pinable_big_cores.size(); ++i) {
    ::benchmark::internal::RegisterBenchmarkInternal(
        new OMP_CifarDense_BaselineBig_Benchmark())
        ->Arg(i)
        ->Name("OMP_CifarDense/BaselineBigOnly")
        ->Unit(benchmark::kMillisecond);
  }
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------

int main(int argc, char** argv) {
  std::string device_id;

  CLI::App app{"Cifar Dense Benchmark"};
  app.add_option("-d,--device", device_id, "Device ID")->required();
  app.allow_extras();

  CLI11_PARSE(app, argc, argv);

  std::cout << "Device ID: " << device_id << std::endl;

  g_device = get_device(device_id);

  for (auto core_type = 0u; core_type < g_device.core_type_count; ++core_type) {
    auto pinable_cores = g_device.get_pinable_cores(core_type);
    std::cout << "Core type " << core_type << " pinable cores: ";
    for (auto core : pinable_cores) {
      std::cout << core << " ";
    }
    std::cout << std::endl;
  }

  auto little_cores = g_device.get_pinable_cores(kLittleCoreType);
  auto medium_cores = g_device.get_pinable_cores(kMediumCoreType);
  auto big_cores = g_device.get_pinable_cores(kBigCoreType);
  auto all_cores = g_device.get_pinable_cores();

  // RegisterPinnedBenchmarkWithRange(little_cores + medium_cores + big_cores);
  RegisterPinnedBenchmarkWithRange(all_cores);
  RegisterLittleBenchmarkWithRange(little_cores);
  RegisterMediumBenchmarkWithRange(medium_cores);
  RegisterBigBenchmarkWithRange(big_cores);

  // Initialize and run benchmarks
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  return 0;
}
