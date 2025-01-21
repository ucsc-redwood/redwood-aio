#include <benchmark/benchmark.h>

#include <CLI/CLI.hpp>
#include <affinity.hpp>
#include <cifar_dense_kernel.hpp>
#include <cifar_sparse_kernel.hpp>
#include <memory>

#include "../conf.hpp"

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

#define DEFINE_STAGE_BENCHMARK(stage_num, core_type)                    \
  static void run_stage_##stage_num##_##core_type(                      \
      cifar_dense::AppData& app_data,                                   \
      const std::vector<int>& cores,                                    \
      const int n_threads) {                                            \
    _Pragma("omp parallel num_threads(n_threads)") {                    \
      bind_thread_to_core(cores);                                       \
      cifar_dense::omp::process_stage_##stage_num(app_data);            \
    }                                                                   \
  }                                                                     \
                                                                        \
  BENCHMARK_DEFINE_F(OMP_CifarDense, Stage##stage_num##core_type)       \
  (benchmark::State & state) {                                          \
    auto cores = g_device.get_pinable_cores(k##core_type##CoreType);    \
                                                                        \
    const auto n_threads = state.range(0);                              \
    for (auto _ : state) {                                              \
      run_stage_##stage_num##_##core_type(*app_data, cores, n_threads); \
    }                                                                   \
  }

#define REGISTER_STAGE_BENCHMARK(stage_num, core_type)                  \
  void RegisterStage##stage_num##core_type##BenchmarkWithRange(         \
      const std::vector<int>& pinable_##core_type##_cores) {            \
    for (size_t i = 1; i <= pinable_##core_type##_cores.size(); ++i) {  \
      ::benchmark::internal::RegisterBenchmarkInternal(                 \
          new OMP_CifarDense_Stage##stage_num##core_type##_Benchmark()) \
          ->Arg(i)                                                      \
          ->Name("OMP_CifarDense/Stage" #stage_num "_" #core_type)      \
          ->Unit(benchmark::kMillisecond);                              \
    }                                                                   \
  }

// Define benchmarks for all 9 stages with little cores
DEFINE_STAGE_BENCHMARK(1, Little)
DEFINE_STAGE_BENCHMARK(2, Little)
DEFINE_STAGE_BENCHMARK(3, Little)
DEFINE_STAGE_BENCHMARK(4, Little)
DEFINE_STAGE_BENCHMARK(5, Little)
DEFINE_STAGE_BENCHMARK(6, Little)
DEFINE_STAGE_BENCHMARK(7, Little)
DEFINE_STAGE_BENCHMARK(8, Little)
DEFINE_STAGE_BENCHMARK(9, Little)

// Define benchmarks for all 9 stages with medium cores
DEFINE_STAGE_BENCHMARK(1, Medium)
DEFINE_STAGE_BENCHMARK(2, Medium)
DEFINE_STAGE_BENCHMARK(3, Medium)
DEFINE_STAGE_BENCHMARK(4, Medium)
DEFINE_STAGE_BENCHMARK(5, Medium)
DEFINE_STAGE_BENCHMARK(6, Medium)
DEFINE_STAGE_BENCHMARK(7, Medium)
DEFINE_STAGE_BENCHMARK(8, Medium)
DEFINE_STAGE_BENCHMARK(9, Medium)

// Define benchmarks for all 9 stages with big cores
DEFINE_STAGE_BENCHMARK(1, Big)
DEFINE_STAGE_BENCHMARK(2, Big)
DEFINE_STAGE_BENCHMARK(3, Big)
DEFINE_STAGE_BENCHMARK(4, Big)
DEFINE_STAGE_BENCHMARK(5, Big)
DEFINE_STAGE_BENCHMARK(6, Big)
DEFINE_STAGE_BENCHMARK(7, Big)
DEFINE_STAGE_BENCHMARK(8, Big)
DEFINE_STAGE_BENCHMARK(9, Big)

// Register all stage benchmarks for little cores
REGISTER_STAGE_BENCHMARK(1, Little)
REGISTER_STAGE_BENCHMARK(2, Little)
REGISTER_STAGE_BENCHMARK(3, Little)
REGISTER_STAGE_BENCHMARK(4, Little)
REGISTER_STAGE_BENCHMARK(5, Little)
REGISTER_STAGE_BENCHMARK(6, Little)
REGISTER_STAGE_BENCHMARK(7, Little)
REGISTER_STAGE_BENCHMARK(8, Little)
REGISTER_STAGE_BENCHMARK(9, Little)

// Register all stage benchmarks for medium cores
REGISTER_STAGE_BENCHMARK(1, Medium)
REGISTER_STAGE_BENCHMARK(2, Medium)
REGISTER_STAGE_BENCHMARK(3, Medium)
REGISTER_STAGE_BENCHMARK(4, Medium)
REGISTER_STAGE_BENCHMARK(5, Medium)
REGISTER_STAGE_BENCHMARK(6, Medium)
REGISTER_STAGE_BENCHMARK(7, Medium)
REGISTER_STAGE_BENCHMARK(8, Medium)
REGISTER_STAGE_BENCHMARK(9, Medium)

// Register all stage benchmarks for big cores
REGISTER_STAGE_BENCHMARK(1, Big)
REGISTER_STAGE_BENCHMARK(2, Big)
REGISTER_STAGE_BENCHMARK(3, Big)
REGISTER_STAGE_BENCHMARK(4, Big)
REGISTER_STAGE_BENCHMARK(5, Big)
REGISTER_STAGE_BENCHMARK(6, Big)
REGISTER_STAGE_BENCHMARK(7, Big)
REGISTER_STAGE_BENCHMARK(8, Big)
REGISTER_STAGE_BENCHMARK(9, Big)

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

  // Register all stage benchmarks for little cores
  RegisterStage1LittleBenchmarkWithRange(little_cores);
  RegisterStage2LittleBenchmarkWithRange(little_cores);
  RegisterStage3LittleBenchmarkWithRange(little_cores);
  RegisterStage4LittleBenchmarkWithRange(little_cores);
  RegisterStage5LittleBenchmarkWithRange(little_cores);
  RegisterStage6LittleBenchmarkWithRange(little_cores);
  RegisterStage7LittleBenchmarkWithRange(little_cores);
  RegisterStage8LittleBenchmarkWithRange(little_cores);
  RegisterStage9LittleBenchmarkWithRange(little_cores);

  // Register all stage benchmarks for medium cores
  RegisterStage1MediumBenchmarkWithRange(medium_cores);
  RegisterStage2MediumBenchmarkWithRange(medium_cores);
  RegisterStage3MediumBenchmarkWithRange(medium_cores);
  RegisterStage4MediumBenchmarkWithRange(medium_cores);
  RegisterStage5MediumBenchmarkWithRange(medium_cores);
  RegisterStage6MediumBenchmarkWithRange(medium_cores);
  RegisterStage7MediumBenchmarkWithRange(medium_cores);
  RegisterStage8MediumBenchmarkWithRange(medium_cores);
  RegisterStage9MediumBenchmarkWithRange(medium_cores);

  // Register all stage benchmarks for big cores
  RegisterStage1BigBenchmarkWithRange(big_cores);
  RegisterStage2BigBenchmarkWithRange(big_cores);
  RegisterStage3BigBenchmarkWithRange(big_cores);
  RegisterStage4BigBenchmarkWithRange(big_cores);
  RegisterStage5BigBenchmarkWithRange(big_cores);
  RegisterStage6BigBenchmarkWithRange(big_cores);
  RegisterStage7BigBenchmarkWithRange(big_cores);
  RegisterStage8BigBenchmarkWithRange(big_cores);
  RegisterStage9BigBenchmarkWithRange(big_cores);

  // Initialize and run benchmarks
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  return 0;
}
