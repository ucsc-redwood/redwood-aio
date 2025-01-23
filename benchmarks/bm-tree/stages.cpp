#include <benchmark/benchmark.h>

#include <CLI/CLI.hpp>
#include <cassert>
#include <memory>
#include <ranges>
#include <thread>
#include <vector>

// Your includes
#include "affinity.hpp"
#include "app.hpp"
#include "tree/omp/func_sort.hpp"
#include "tree/omp/tree_kernel.hpp"
#include "tree/tree_appdata.hpp"

//
// ----------------------------------------------------------------------
// 1) "Fixture-like" class that replicates your old SetUp/TearDown logic
// ----------------------------------------------------------------------
//
// We'll allocate AppData in the constructor, then run the "SetUp" steps
// you had before. Then the destructor does "TearDown" if needed.
//
class OMP_Tree_Fixture {
 public:
  OMP_Tree_Fixture() {
    app_data = std::make_unique<tree::AppData>(std::pmr::new_delete_resource());

    // Simulate your old "SetUp" code:
    //   - 1) Stage1 in parallel
    //   - 2) Single-thread sort
    //   - 3..7 in parallel
#pragma omp parallel
    {
      tree::omp::process_stage_1(*app_data);
#pragma omp single
      { std::ranges::sort(app_data->u_morton_keys); }
      tree::omp::process_stage_3(*app_data);
      tree::omp::process_stage_4(*app_data);
      tree::omp::process_stage_5(*app_data);
      tree::omp::process_stage_6(*app_data);
      tree::omp::process_stage_7(*app_data);
    }
  }

  ~OMP_Tree_Fixture() {
    // If you had a TearDown, put that logic here.
    // For now, we just reset the pointer:
    app_data.reset();
  }

  std::unique_ptr<tree::AppData> app_data;
};

//
// ----------------------------------------------------------------------
// 2) Shared "runner" functions for each stage
// ----------------------------------------------------------------------
//
// We define one pinned function for each stage except Stage2 (which is
// special). Each function runs the stage in an OMP region, binding threads to
// 'cores'.
//

// For stages that do NOT require TempStorage:
using StageFuncNoTemp = void (*)(tree::AppData&,
                                 int n_threads,
                                 const std::vector<int>& cores);

// Example: Stage1
static void run_stage_1(tree::AppData& app_data,
                        int n_threads,
                        const std::vector<int>& cores) {
#pragma omp parallel num_threads(n_threads)
  {
    bind_thread_to_core(cores);
    tree::omp::process_stage_1(app_data);
  }
}

static void run_stage_3(tree::AppData& app_data,
                        int n_threads,
                        const std::vector<int>& cores) {
#pragma omp parallel num_threads(n_threads)
  {
    bind_thread_to_core(cores);
    tree::omp::process_stage_3(app_data);
  }
}

static void run_stage_4(tree::AppData& app_data,
                        int n_threads,
                        const std::vector<int>& cores) {
#pragma omp parallel num_threads(n_threads)
  {
    bind_thread_to_core(cores);
    tree::omp::process_stage_4(app_data);
  }
}

static void run_stage_5(tree::AppData& app_data,
                        int n_threads,
                        const std::vector<int>& cores) {
#pragma omp parallel num_threads(n_threads)
  {
    bind_thread_to_core(cores);
    tree::omp::process_stage_5(app_data);
  }
}

static void run_stage_6(tree::AppData& app_data,
                        int n_threads,
                        const std::vector<int>& cores) {
#pragma omp parallel num_threads(n_threads)
  {
    bind_thread_to_core(cores);
    tree::omp::process_stage_6(app_data);
  }
}

static void run_stage_7(tree::AppData& app_data,
                        int n_threads,
                        const std::vector<int>& cores) {
#pragma omp parallel num_threads(n_threads)
  {
    bind_thread_to_core(cores);
    tree::omp::process_stage_7(app_data);
  }
}

// For Stage2, we need a different signature that includes TempStorage.
using StageFuncWithTemp = void (*)(tree::AppData&,
                                   int n_threads,
                                   const std::vector<int>& cores);

static void run_stage_2_likewise(tree::AppData& app_data,
                                 int n_threads,
                                 const std::vector<int>& cores) {
  // We typically want to create a fresh TempStorage *inside each iteration*
  // of the benchmark. So do that at the last moment:
  tree::omp::v2::TempStorage temp_storage(n_threads, n_threads);

#pragma omp parallel num_threads(n_threads)
  {
    bind_thread_to_core(cores);
    tree::omp::v2::process_stage_2(app_data, temp_storage);
  }

  // If you want to assert sorted after stage2:
  assert(std::ranges::is_sorted(app_data.u_morton_keys));
}

// If you also want to benchmark "Stage2_std" (just sorting):
static void run_stage_2_std(tree::AppData& app_data) {
  std::ranges::sort(app_data.u_morton_keys);
  assert(std::ranges::is_sorted(app_data.u_morton_keys));
}

//
// ----------------------------------------------------------------------
// 3) Single function to register a "pinned stage" benchmark
// ----------------------------------------------------------------------
//
// This function creates a Google Benchmark that, for each iteration,
// constructs (and destructs) an OMP_Tree_Fixture and calls the desired stage.
//
static void RegisterPinnedStageBenchmark(const std::string& benchmarkName,
                                         const std::vector<int>& cores,
                                         StageFuncNoTemp stageFunc) {
  // We'll do one Benchmark for each #threads from 1.. cores.size()
  const int maxThreads = static_cast<int>(cores.size());
  for (int t = 1; t <= maxThreads; t++) {
    // We capture [=] so that 't' and 'cores' are copied into the lambda
    benchmark::RegisterBenchmark(
        (benchmarkName + "/" + std::to_string(t)).c_str(),
        [=](benchmark::State& st) {
          for (auto _ : st) {
            // Construct the fixture for each iteration (like old SetUp)
            OMP_Tree_Fixture fixture;
            // Run the pinned stage
            stageFunc(*fixture.app_data, t, cores);
            // fixture destructor => old TearDown
          }
        })
        ->Unit(benchmark::kMillisecond);
  }
}

//
// For Stage2, we do the same idea but pass a StageFuncWithTemp if we like
// (In practice, we can unify with StageFuncNoTemp if we allocate TempStorage
// inside the function. It's your choice.)
//
static void RegisterPinnedStage2Benchmark(const std::string& benchmarkName,
                                          const std::vector<int>& cores,
                                          StageFuncWithTemp stageFunc) {
  const int maxThreads = static_cast<int>(cores.size());
  for (int t = 1; t <= maxThreads; t++) {
    benchmark::RegisterBenchmark(
        (benchmarkName + "/" + std::to_string(t)).c_str(),
        [=](benchmark::State& st) {
          for (auto _ : st) {
            OMP_Tree_Fixture fixture;
            stageFunc(*fixture.app_data, t, cores);
          }
        })
        ->Unit(benchmark::kMillisecond);
  }
}

// If you want a "Stage2_std" that doesn't do pinned threads at all:
static void RegisterStage2StdBenchmark(const std::string& benchmarkName) {
  benchmark::RegisterBenchmark(benchmarkName.c_str(),
                               [=](benchmark::State& st) {
                                 for (auto _ : st) {
                                   OMP_Tree_Fixture fixture;
                                   run_stage_2_std(*fixture.app_data);
                                 }
                               })
      ->Unit(benchmark::kMillisecond);
}

//
// ----------------------------------------------------------------------
// 4) "Main": parse args, then dynamically register all the combos
// ----------------------------------------------------------------------
int main(int argc, char** argv) {
  parse_args(argc, argv);  // presumably populates g_little_cores, etc.

  //
  // Stage1
  //
  RegisterPinnedStageBenchmark("Stage1_little", g_little_cores, run_stage_1);
  RegisterPinnedStageBenchmark("Stage1_medium", g_medium_cores, run_stage_1);
  RegisterPinnedStageBenchmark("Stage1_big", g_big_cores, run_stage_1);

  //
  // Stage2
  //
  RegisterPinnedStage2Benchmark(
      "Stage2_little", g_little_cores, run_stage_2_likewise);
  RegisterPinnedStage2Benchmark(
      "Stage2_medium", g_medium_cores, run_stage_2_likewise);
  RegisterPinnedStage2Benchmark(
      "Stage2_big", g_big_cores, run_stage_2_likewise);

  // Also the std::sort version
  RegisterStage2StdBenchmark("Stage2_std");

  //
  // Stage3
  //
  RegisterPinnedStageBenchmark("Stage3_little", g_little_cores, run_stage_3);
  RegisterPinnedStageBenchmark("Stage3_medium", g_medium_cores, run_stage_3);
  RegisterPinnedStageBenchmark("Stage3_big", g_big_cores, run_stage_3);

  // Repeat for stages 4..7
  RegisterPinnedStageBenchmark("Stage4_little", g_little_cores, run_stage_4);
  RegisterPinnedStageBenchmark("Stage4_medium", g_medium_cores, run_stage_4);
  RegisterPinnedStageBenchmark("Stage4_big", g_big_cores, run_stage_4);

  RegisterPinnedStageBenchmark("Stage5_little", g_little_cores, run_stage_5);
  RegisterPinnedStageBenchmark("Stage5_medium", g_medium_cores, run_stage_5);
  RegisterPinnedStageBenchmark("Stage5_big", g_big_cores, run_stage_5);

  RegisterPinnedStageBenchmark("Stage6_little", g_little_cores, run_stage_6);
  RegisterPinnedStageBenchmark("Stage6_medium", g_medium_cores, run_stage_6);
  RegisterPinnedStageBenchmark("Stage6_big", g_big_cores, run_stage_6);

  RegisterPinnedStageBenchmark("Stage7_little", g_little_cores, run_stage_7);
  RegisterPinnedStageBenchmark("Stage7_medium", g_medium_cores, run_stage_7);
  RegisterPinnedStageBenchmark("Stage7_big", g_big_cores, run_stage_7);

  //
  // Finally, run all benchmarks
  //
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
