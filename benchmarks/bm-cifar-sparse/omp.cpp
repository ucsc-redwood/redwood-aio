#include <benchmark/benchmark.h>

#include <string>
#include <thread>
#include <vector>

#include "affinity.hpp"
#include "app.hpp"
#include "cifar-sparse/omp/sparse_kernel.hpp"

static void run_baseline_unrestricted(cifar_sparse::AppData& app_data, const int n_threads) {
#pragma omp parallel num_threads(n_threads)
  {
    cifar_sparse::omp::run_stage<1>(app_data);
    cifar_sparse::omp::run_stage<2>(app_data);
    cifar_sparse::omp::run_stage<3>(app_data);
    cifar_sparse::omp::run_stage<4>(app_data);
    cifar_sparse::omp::run_stage<5>(app_data);
    cifar_sparse::omp::run_stage<6>(app_data);
    cifar_sparse::omp::run_stage<7>(app_data);
    cifar_sparse::omp::run_stage<8>(app_data);
    cifar_sparse::omp::run_stage<9>(app_data);
  }
}

[[nodiscard]] cifar_sparse::AppData make_appdata() {
  auto mr = std::pmr::new_delete_resource();
  cifar_sparse::AppData app_data(mr);
  run_baseline_unrestricted(app_data, std::thread::hardware_concurrency());
  return app_data;
}

// ------------------------------------------------------------
// Baseline benchmarks
// ------------------------------------------------------------

void register_baseline_benchmark() {
  std::string benchmark_name = "OMP_CifarSparse/Baseline";

  benchmark::RegisterBenchmark(benchmark_name.c_str(), [](benchmark::State& state) {
    auto app_data = make_appdata();
    for (auto _ : state) {
      run_baseline_unrestricted(app_data, std::thread::hardware_concurrency());
    }
  })->Unit(benchmark::kMillisecond);
}

// ------------------------------------------------------------
// Stage benchmarks
// ------------------------------------------------------------

template <int Stage, typename CoreType>
void run_bm_stage(cifar_sparse::AppData& app_data, const std::vector<int>& cores, int n_threads) {
#pragma omp parallel num_threads(n_threads)
  {
    bind_thread_to_cores(cores);
    cifar_sparse::omp::run_stage<Stage>(app_data);
  }
}

template <int Stage, typename CoreType>
void register_stage_benchmark(const std::vector<int>& cores) {
  for (size_t i = 1; i <= cores.size(); ++i) {
    std::string benchmark_name =
        "OMP_CifarSparse/Stage" + std::to_string(Stage) + "_" + CoreType::name();

    benchmark::RegisterBenchmark(benchmark_name.c_str(),
                                 [i, cores](benchmark::State& state) {
                                   auto app_data = make_appdata();
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

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------

int main(int argc, char** argv) {
  parse_args(argc, argv);

  register_baseline_benchmark();

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
