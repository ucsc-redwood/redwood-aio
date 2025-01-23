#include <benchmark/benchmark.h>

#include <CLI/CLI.hpp>
#include <thread>

#include "affinity.hpp"
#include "app.hpp"
#include "tree/omp/func_sort.hpp"
#include "tree/omp/tree_kernel.hpp"
#include "tree/tree_appdata.hpp"

// Add this fixture class definition before the benchmark
class OMP_Tree : public benchmark::Fixture {};

static void run_stage_2_little(tree::AppData& app_data,
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

BENCHMARK_DEFINE_F(OMP_Tree, Stage2little)
(benchmark::State& state) {
  const auto n_threads = state.range(0);

  auto app_data =
      std::make_unique<tree::AppData>(std::pmr::new_delete_resource());

  for (auto _ : state) {
    tree::omp::v2::TempStorage temp_storage(n_threads, n_threads);
    run_stage_2_little(*app_data, g_little_cores, n_threads, temp_storage);
  }

  assert(std::ranges::is_sorted(app_data->u_morton_keys));
}

void RegisterBaselineLittleBenchmarkWithRange(
    const std::vector<int>& pinable_little_cores) {
  for (size_t i = 1; i <= pinable_little_cores.size(); ++i) {
    ::benchmark::internal::RegisterBenchmarkInternal(
        new OMP_Tree_Stage2little_Benchmark())
        ->Arg(i)
        ->Name("OMP_Tree/Stage2_little")
        ->Unit(benchmark::kMillisecond);
  }
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------

int main(int argc, char** argv) {
  parse_args(argc, argv);

  RegisterBaselineLittleBenchmarkWithRange(g_little_cores);
  // RegisterBaselineMediumBenchmarkWithRange(g_medium_cores);
  // RegisterBaselineBigBenchmarkWithRange(g_big_cores);

  // Initialize and run benchmarks
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  return 0;
}
