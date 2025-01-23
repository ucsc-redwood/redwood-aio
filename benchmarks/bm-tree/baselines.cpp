#include <benchmark/benchmark.h>

#include <CLI/CLI.hpp>
#include <thread>

#include "affinity.hpp"
#include "app.hpp"
#include "tree/omp/func_sort.hpp"
#include "tree/omp/tree_kernel.hpp"
#include "tree/tree_appdata.hpp"

// Add this fixture class definition before the benchmark
class OMP_Tree : public benchmark::Fixture {
 protected:
  void SetUp(benchmark::State&) override {
    app_data = std::make_unique<tree::AppData>(std::pmr::new_delete_resource());
  }

  void TearDown(benchmark::State&) override { app_data.reset(); }

  std::unique_ptr<tree::AppData> app_data;
};

static void run_baseline_pinned(tree::AppData& app_data,
                                const std::vector<int>& cores,
                                const int n_threads,
                                tree::omp::v2::TempStorage& temp_storage) {
#pragma omp parallel num_threads(n_threads)
  {
    bind_thread_to_core(cores);

    tree::omp::process_stage_1(app_data);
    tree::omp::v2::process_stage_2(app_data, temp_storage);
    tree::omp::process_stage_3(app_data);
    tree::omp::process_stage_4(app_data);
    tree::omp::process_stage_5(app_data);
    tree::omp::process_stage_6(app_data);
    tree::omp::process_stage_7(app_data);
  }
}

BENCHMARK_DEFINE_F(OMP_Tree, Baseline_Pinned_Little)
(benchmark::State& state) {
  const auto n_threads = state.range(0);

  for (auto _ : state) {
    tree::omp::v2::TempStorage temp_storage(n_threads, n_threads);
    run_baseline_pinned(*app_data, g_little_cores, n_threads, temp_storage);
  }

  assert(std::ranges::is_sorted(app_data->u_morton_keys));
}

BENCHMARK_DEFINE_F(OMP_Tree, Baseline_Pinned_Medium)
(benchmark::State& state) {
  const auto n_threads = state.range(0);

  for (auto _ : state) {
    tree::omp::v2::TempStorage temp_storage(n_threads, n_threads);
    run_baseline_pinned(*app_data, g_medium_cores, n_threads, temp_storage);
  }

  assert(std::ranges::is_sorted(app_data->u_morton_keys));
}

BENCHMARK_DEFINE_F(OMP_Tree, Baseline_Pinned_Big)
(benchmark::State& state) {
  const auto n_threads = state.range(0);

  for (auto _ : state) {
    tree::omp::v2::TempStorage temp_storage(n_threads, n_threads);
    run_baseline_pinned(*app_data, g_big_cores, n_threads, temp_storage);
  }

  assert(std::ranges::is_sorted(app_data->u_morton_keys));
}

void RegisterBaselinePinnedLittleBenchmarkWithRange(
    const std::vector<int>& pinable_cores) {
  for (size_t i = 1; i <= pinable_cores.size(); ++i) {
    ::benchmark::internal::RegisterBenchmarkInternal(
        new OMP_Tree_Baseline_Pinned_Little_Benchmark())
        ->Arg(i)
        ->Name("OMP_Tree/Baseline_Pinned_Little")
        ->Unit(benchmark::kMillisecond);
  }
}

void RegisterBaselinePinnedMediumBenchmarkWithRange(
    const std::vector<int>& pinable_cores) {
  for (size_t i = 1; i <= pinable_cores.size(); ++i) {
    ::benchmark::internal::RegisterBenchmarkInternal(
        new OMP_Tree_Baseline_Pinned_Medium_Benchmark())
        ->Arg(i)
        ->Name("OMP_Tree/Baseline_Pinned_Medium")
        ->Unit(benchmark::kMillisecond);
  }
}

void RegisterBaselinePinnedBigBenchmarkWithRange(
    const std::vector<int>& pinable_cores) {
  for (size_t i = 1; i <= pinable_cores.size(); ++i) {
    ::benchmark::internal::RegisterBenchmarkInternal(
        new OMP_Tree_Baseline_Pinned_Big_Benchmark())
        ->Arg(i)
        ->Name("OMP_Tree/Baseline_Pinned_Big")
        ->Unit(benchmark::kMillisecond);
  }
}

// ------------------------------------------------------------
// Baseline unrestricted
// ------------------------------------------------------------

static void run_baseline_unrestricted(
    tree::AppData& app_data,
    const int n_threads,
    tree::omp::v2::TempStorage& temp_storage) {
#pragma omp parallel num_threads(n_threads)
  {
    tree::omp::process_stage_1(app_data);
    tree::omp::v2::process_stage_2(app_data, temp_storage);
    tree::omp::process_stage_3(app_data);
    tree::omp::process_stage_4(app_data);
    tree::omp::process_stage_5(app_data);
    tree::omp::process_stage_6(app_data);
    tree::omp::process_stage_7(app_data);
  }
}

BENCHMARK_DEFINE_F(OMP_Tree, Baseline_Unrestricted)
(benchmark::State& state) {
  const auto n_threads = state.range(0);
  for (auto _ : state) {
    tree::omp::v2::TempStorage temp_storage(n_threads, n_threads);
    run_baseline_unrestricted(*app_data, n_threads, temp_storage);
  }
}

BENCHMARK_REGISTER_F(OMP_Tree, Baseline_Unrestricted)
    ->DenseRange(1, std::thread::hardware_concurrency())
    ->Unit(benchmark::kMillisecond);

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------

int main(int argc, char** argv) {
  parse_args(argc, argv);

  RegisterBaselinePinnedLittleBenchmarkWithRange(g_little_cores);
  RegisterBaselinePinnedMediumBenchmarkWithRange(g_medium_cores);
  RegisterBaselinePinnedBigBenchmarkWithRange(g_big_cores);

  // Initialize and run benchmarks
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  return 0;
}
