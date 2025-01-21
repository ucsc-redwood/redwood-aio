#include <benchmark/benchmark.h>

#include <CLI/CLI.hpp>
#include <affinity.hpp>
#include <cifar_dense_kernel.hpp>
#include <cifar_sparse_kernel.hpp>
#include <memory>

#include "conf.hpp"

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

// Define baseline runner and benchmark
static void run_baseline(cifar_dense::AppData& app_data, int n_threads) {
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

BENCHMARK_DEFINE_F(OMP_CifarDense, Baseline)(benchmark::State& state) {
  auto n_threads = state.range(0);
  for (auto _ : state) {
    run_baseline(*app_data, n_threads);
  }
}

BENCHMARK_REGISTER_F(OMP_CifarDense, Baseline)
    ->DenseRange(1, 6)
    ->Unit(benchmark::kMillisecond);

// Define baseline runner and benchmark
static void run_baseline_pinned(cifar_dense::AppData& app_data, int n_threads) {
#pragma omp parallel num_threads(n_threads)
  {
    auto tid = omp_get_thread_num();
    bind_thread_to_core(tid);
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
    run_baseline_pinned(*app_data, n_threads);
  }
}

BENCHMARK_REGISTER_F(OMP_CifarDense, BaselinePinned)
    ->DenseRange(1, 6)
    ->Unit(benchmark::kMillisecond);

int main(int argc, char** argv) {
  std::string device_id;

  CLI::App app{"Cifar Dense Benchmark"};
  app.add_option("-d,--device", device_id, "Device ID")->required();
  app.allow_extras();

  CLI11_PARSE(app, argc, argv);

  std::cout << "Device ID: " << device_id << std::endl;

  Device device = get_device(device_id);

  for (auto core_type = 0u; core_type < device.core_type_count; ++core_type) {
    auto pinable_cores = device.get_pinable_cores(core_type);
    std::cout << "Core type " << core_type << " pinable cores: ";
    for (auto core : pinable_cores) {
      std::cout << core << " ";
    }
    std::cout << std::endl;
  }

  // Initialize and run benchmarks
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
