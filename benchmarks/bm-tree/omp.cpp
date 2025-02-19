#include <benchmark/benchmark.h>

#include <CLI/CLI.hpp>
#include <algorithm>
#include <thread>

#include "../argc_argv_sanitizer.hpp"
#include "builtin-apps/affinity.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/resources_path.hpp"
#include "builtin-apps/tree/omp/tree_kernel.hpp"
#include "builtin-apps/tree/tree_appdata.hpp"

// ------------------------------------------------------------
// Global variables
// ------------------------------------------------------------

class OMP_Tree : public benchmark::Fixture {
 protected:
  std::unique_ptr<tree::AppData> appdata_ptr;

  void SetUp(const ::benchmark::State&) override {
    appdata_ptr = std::make_unique<tree::AppData>(std::pmr::new_delete_resource());
  }

  void TearDown(const ::benchmark::State&) override { appdata_ptr.reset(); }
};

// ----------------------------------------------------------------
// Baseline
// ----------------------------------------------------------------

static void run_baseline_unrestricted(tree::AppData& appdata,
                                      tree::omp::TempStorage& temp_storage,
                                      const int n_threads) {
#pragma omp parallel num_threads(n_threads)
  {
    tree::omp::process_stage_1(appdata, temp_storage);
    tree::omp::process_stage_2(appdata, temp_storage);
    tree::omp::process_stage_3(appdata, temp_storage);
    tree::omp::process_stage_4(appdata, temp_storage);
    tree::omp::process_stage_5(appdata, temp_storage);
    tree::omp::process_stage_6(appdata, temp_storage);
    tree::omp::process_stage_7(appdata, temp_storage);
  }
}

BENCHMARK_DEFINE_F(OMP_Tree, Baseline)
(benchmark::State& state) {
  const auto n_threads = state.range(0);
  for (auto _ : state) {
    tree::omp::TempStorage temp_storage(n_threads, n_threads);
    run_baseline_unrestricted(*appdata_ptr, temp_storage, n_threads);
  }
}

BENCHMARK_REGISTER_F(OMP_Tree, Baseline)
    ->DenseRange(1, std::thread::hardware_concurrency())
    ->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stages
// ----------------------------------------------------------------

template <int stage, ProcessorType processor_type>
  requires(stage >= 1 && stage <= 9)
void run_stage(tree::AppData& appdata, tree::omp::TempStorage& temp_storage, const int n_threads) {
#pragma omp parallel num_threads(n_threads)
  {
    // Bind to core if needed:
    if constexpr (processor_type == ProcessorType::kLittleCore) {
      bind_thread_to_cores(g_little_cores);
    } else if constexpr (processor_type == ProcessorType::kMediumCore) {
      bind_thread_to_cores(g_medium_cores);
    } else if constexpr (processor_type == ProcessorType::kBigCore) {
      bind_thread_to_cores(g_big_cores);
    } else {
      assert(false);
    }

    tree::omp::run_stage<stage>(appdata, temp_storage);
  }
}

BENCHMARK_DEFINE_F(OMP_Tree, Stage1little)
(benchmark::State& state) {
  const auto n_threads = state.range(0);
  for (auto _ : state) {
    tree::omp::TempStorage temp_storage(n_threads, n_threads);
    run_stage<1, ProcessorType::kLittleCore>(*appdata_ptr, temp_storage, n_threads);
  }
}

BENCHMARK_DEFINE_F(OMP_Tree, Stage2little)
(benchmark::State& state) {
  const auto n_threads = state.range(0);
  for (auto _ : state) {
    tree::omp::TempStorage temp_storage(n_threads, n_threads);
    run_stage<2, ProcessorType::kLittleCore>(*appdata_ptr, temp_storage, n_threads);
  }
}

BENCHMARK_DEFINE_F(OMP_Tree, Stage3little)
(benchmark::State& state) {
  const auto n_threads = state.range(0);
  for (auto _ : state) {
    tree::omp::TempStorage temp_storage(n_threads, n_threads);
    run_stage<3, ProcessorType::kLittleCore>(*appdata_ptr, temp_storage, n_threads);
  }
}

BENCHMARK_DEFINE_F(OMP_Tree, Stage4little)
(benchmark::State& state) {
  const auto n_threads = state.range(0);
  for (auto _ : state) {
    tree::omp::TempStorage temp_storage(n_threads, n_threads);
    run_stage<4, ProcessorType::kLittleCore>(*appdata_ptr, temp_storage, n_threads);
  }
}

BENCHMARK_DEFINE_F(OMP_Tree, Stage5little)
(benchmark::State& state) {
  const auto n_threads = state.range(0);
  for (auto _ : state) {
    tree::omp::TempStorage temp_storage(n_threads, n_threads);
    run_stage<5, ProcessorType::kLittleCore>(*appdata_ptr, temp_storage, n_threads);
  }
}

BENCHMARK_DEFINE_F(OMP_Tree, Stage6little)
(benchmark::State& state) {
  const auto n_threads = state.range(0);
  for (auto _ : state) {
    tree::omp::TempStorage temp_storage(n_threads, n_threads);
    run_stage<6, ProcessorType::kLittleCore>(*appdata_ptr, temp_storage, n_threads);
  }
}

BENCHMARK_DEFINE_F(OMP_Tree, Stage7little)
(benchmark::State& state) {
  const auto n_threads = state.range(0);
  for (auto _ : state) {
    tree::omp::TempStorage temp_storage(n_threads, n_threads);
    run_stage<7, ProcessorType::kLittleCore>(*appdata_ptr, temp_storage, n_threads);
  }
}

// Then define the registration functions for each stage
void RegisterStage1LittleBenchmarkWithRange(const std::vector<int>& pinable_little_cores) {
  for (size_t i = 1; i <= pinable_little_cores.size(); ++i) {
    ::benchmark::internal::RegisterBenchmarkInternal(new OMP_Tree_Stage1little_Benchmark())
        ->Arg(i)
        ->Name("OMP_Tree/Stage1_little")
        ->Unit(benchmark::kMillisecond);
  }
}

void RegisterStage2LittleBenchmarkWithRange(const std::vector<int>& pinable_little_cores) {
  for (size_t i = 1; i <= pinable_little_cores.size(); ++i) {
    ::benchmark::internal::RegisterBenchmarkInternal(new OMP_Tree_Stage2little_Benchmark())
        ->Arg(i)
        ->Name("OMP_Tree/Stage2_little")
        ->Unit(benchmark::kMillisecond);
  }
}

void RegisterStage3LittleBenchmarkWithRange(const std::vector<int>& pinable_little_cores) {
  for (size_t i = 1; i <= pinable_little_cores.size(); ++i) {
    ::benchmark::internal::RegisterBenchmarkInternal(new OMP_Tree_Stage3little_Benchmark())
        ->Arg(i)
        ->Name("OMP_Tree/Stage3_little")
        ->Unit(benchmark::kMillisecond);
  }
}

void RegisterStage4LittleBenchmarkWithRange(const std::vector<int>& pinable_little_cores) {
  for (size_t i = 1; i <= pinable_little_cores.size(); ++i) {
    ::benchmark::internal::RegisterBenchmarkInternal(new OMP_Tree_Stage4little_Benchmark())
        ->Arg(i)
        ->Name("OMP_Tree/Stage4_little")
        ->Unit(benchmark::kMillisecond);
  }
}

void RegisterStage5LittleBenchmarkWithRange(const std::vector<int>& pinable_little_cores) {
  for (size_t i = 1; i <= pinable_little_cores.size(); ++i) {
    ::benchmark::internal::RegisterBenchmarkInternal(new OMP_Tree_Stage5little_Benchmark())
        ->Arg(i)
        ->Name("OMP_Tree/Stage5_little")
        ->Unit(benchmark::kMillisecond);
  }
}

void RegisterStage6LittleBenchmarkWithRange(const std::vector<int>& pinable_little_cores) {
  for (size_t i = 1; i <= pinable_little_cores.size(); ++i) {
    ::benchmark::internal::RegisterBenchmarkInternal(new OMP_Tree_Stage6little_Benchmark())
        ->Arg(i)
        ->Name("OMP_Tree/Stage6_little")
        ->Unit(benchmark::kMillisecond);
  }
}

void RegisterStage7LittleBenchmarkWithRange(const std::vector<int>& pinable_little_cores) {
  for (size_t i = 1; i <= pinable_little_cores.size(); ++i) {
    ::benchmark::internal::RegisterBenchmarkInternal(new OMP_Tree_Stage7little_Benchmark())
        ->Arg(i)
        ->Name("OMP_Tree/Stage7_little")
        ->Unit(benchmark::kMillisecond);
  }
}

// BENCHMARK_DEFINE_F(OMP_Tree, Stage1little)
// (benchmark::State& state) {
//   const auto n_threads = state.range(0);
//   for (auto _ : state) {
//     run_stage_1_little(*appdata, *temp_storage, g_little_cores, n_threads);
//   }
// }

// static void run_stage_3_little(tree::AppData& appdata,
//                                tree::omp::TempStorage& temp_storage,
//                                const std::vector<int>& cores,
//                                const int n_threads) {
// #pragma omp parallel num_threads(n_threads)
//   {
//     bind_thread_to_cores(cores);
//     tree::omp::process_stage_3(appdata, temp_storage);
//     tree::omp::process_stage_3(appdata, temp_storage);
//   }
// }

// BENCHMARK_DEFINE_F(OMP_Tree, Stage3little)
// (benchmark::State& state) {
//   const auto n_threads = state.range(0);
//   for (auto _ : state) {
//     run_stage_3_little(*appdata, *temp_storage, g_little_cores, n_threads);
//   }
// }

// static void run_stage_4_little(tree::AppData& appdata,
//                                tree::omp::TempStorage& temp_storage,
//                                const std::vector<int>& cores,
//                                const int n_threads) {
// #pragma omp parallel num_threads(n_threads)
//   {
//     bind_thread_to_cores(cores);
//     tree::omp::process_stage_4(appdata, temp_storage);
//   }
// }

// BENCHMARK_DEFINE_F(OMP_Tree, Stage4little)
// (benchmark::State& state) {
//   const auto n_threads = state.range(0);
//   for (auto _ : state) {
//     run_stage_4_little(*appdata, *temp_storage, g_little_cores, n_threads);
//   }
// }

// static void run_stage_5_little(tree::AppData& appdata,
//                                tree::omp::TempStorage& temp_storage,
//                                const std::vector<int>& cores,
//                                const int n_threads) {
// #pragma omp parallel num_threads(n_threads)
//   {
//     bind_thread_to_cores(cores);
//     tree::omp::process_stage_5(appdata, temp_storage);
//     tree::omp::process_stage_5(appdata, temp_storage);
//   }
// }

// BENCHMARK_DEFINE_F(OMP_Tree, Stage5little)
// (benchmark::State& state) {
//   const auto n_threads = state.range(0);
//   for (auto _ : state) {
//     run_stage_5_little(*appdata, *temp_storage, g_little_cores, n_threads);
//   }
// }

// static void run_stage_6_little(tree::AppData& appdata,
//                                tree::omp::TempStorage& temp_storage,
//                                const std::vector<int>& cores,
//                                const int n_threads) {
// #pragma omp parallel num_threads(n_threads)
//   {
//     bind_thread_to_cores(cores);
//     tree::omp::process_stage_6(appdata, temp_storage);
//   }
// }

// BENCHMARK_DEFINE_F(OMP_Tree, Stage6little)
// (benchmark::State& state) {
//   const auto n_threads = state.range(0);
//   for (auto _ : state) {
//     run_stage_6_little(*appdata, *temp_storage, g_little_cores, n_threads);
//   }
// }

// static void run_stage_7_little(tree::AppData& appdata,
//                                tree::omp::TempStorage& temp_storage,
//                                const std::vector<int>& cores,
//                                const int n_threads) {
// #pragma omp parallel num_threads(n_threads)
//   {
//     bind_thread_to_cores(cores);
//     tree::omp::process_stage_7(appdata, temp_storage);
//   }
// }

// BENCHMARK_DEFINE_F(OMP_Tree, Stage7little)
// (benchmark::State& state) {
//   const auto n_threads = state.range(0);
//   for (auto _ : state) {
//     run_stage_7_little(*appdata, *temp_storage, g_little_cores, n_threads);
//   }
// }

// // Then define the registration functions for each stage
// void RegisterStage1LittleBenchmarkWithRange(const std::vector<int>& pinable_little_cores) {
//   for (size_t i = 1; i <= pinable_little_cores.size(); ++i) {
//     ::benchmark::internal::RegisterBenchmarkInternal(new OMP_Tree_Stage1little_Benchmark())
//         ->Arg(i)
//         ->Name("OMP_Tree/Stage1_little")
//         ->Unit(benchmark::kMillisecond);
//   }
// }

// // void RegisterStage2LittleBenchmarkWithRange(
// //     const std::vector<int>& pinable_little_cores) {
// //   for (size_t i = 1; i <= pinable_little_cores.size(); ++i) {
// //     ::benchmark::internal::RegisterBenchmarkInternal(
// //         new OMP_Tree_Stage2little_Benchmark())
// //         ->Arg(i)
// //         ->Name("OMP_Tree/Stage2_little")
// //         ->Unit(benchmark::kMillisecond);
// //   }
// // }

// void RegisterStage3LittleBenchmarkWithRange(const std::vector<int>& pinable_little_cores) {
//   for (size_t i = 1; i <= pinable_little_cores.size(); ++i) {
//     ::benchmark::internal::RegisterBenchmarkInternal(new OMP_Tree_Stage3little_Benchmark())
//         ->Arg(i)
//         ->Name("OMP_Tree/Stage3_little")
//         ->Unit(benchmark::kMillisecond);
//   }
// }

// void RegisterStage4LittleBenchmarkWithRange(const std::vector<int>& pinable_little_cores) {
//   for (size_t i = 1; i <= pinable_little_cores.size(); ++i) {
//     ::benchmark::internal::RegisterBenchmarkInternal(new OMP_Tree_Stage4little_Benchmark())
//         ->Arg(i)
//         ->Name("OMP_Tree/Stage4_little")
//         ->Unit(benchmark::kMillisecond);
//   }
// }

// void RegisterStage5LittleBenchmarkWithRange(const std::vector<int>& pinable_little_cores) {
//   for (size_t i = 1; i <= pinable_little_cores.size(); ++i) {
//     ::benchmark::internal::RegisterBenchmarkInternal(new OMP_Tree_Stage5little_Benchmark())
//         ->Arg(i)
//         ->Name("OMP_Tree/Stage5_little")
//         ->Unit(benchmark::kMillisecond);
//   }
// }

// void RegisterStage6LittleBenchmarkWithRange(const std::vector<int>& pinable_little_cores) {
//   for (size_t i = 1; i <= pinable_little_cores.size(); ++i) {
//     ::benchmark::internal::RegisterBenchmarkInternal(new OMP_Tree_Stage6little_Benchmark())
//         ->Arg(i)
//         ->Name("OMP_Tree/Stage6_little")
//         ->Unit(benchmark::kMillisecond);
//   }
// }

// void RegisterStage7LittleBenchmarkWithRange(const std::vector<int>& pinable_little_cores) {
//   for (size_t i = 1; i <= pinable_little_cores.size(); ++i) {
//     ::benchmark::internal::RegisterBenchmarkInternal(new OMP_Tree_Stage7little_Benchmark())
//         ->Arg(i)
//         ->Name("OMP_Tree/Stage7_little")
//         ->Unit(benchmark::kMillisecond);
//   }
// }

// static void run_stage_1_medium(tree::AppData& appdata,
//                                tree::omp::TempStorage& temp_storage,
//                                const std::vector<int>& cores,
//                                const int n_threads) {
// #pragma omp parallel num_threads(n_threads)
//   {
//     bind_thread_to_cores(cores);
//     tree::omp::process_stage_1(appdata, temp_storage);
//   }
// }

// BENCHMARK_DEFINE_F(OMP_Tree, Stage1medium)
// (benchmark::State& state) {
//   const auto n_threads = state.range(0);
//   for (auto _ : state) {
//     run_stage_1_medium(*appdata, *temp_storage, g_medium_cores, n_threads);
//   }
// }

// static void run_stage_3_medium(tree::AppData& appdata,
//                                tree::omp::TempStorage& temp_storage,
//                                const std::vector<int>& cores,
//                                const int n_threads) {
// #pragma omp parallel num_threads(n_threads)
//   {
//     bind_thread_to_cores(cores);
//     tree::omp::process_stage_3(appdata, temp_storage);
//   }
// }

// BENCHMARK_DEFINE_F(OMP_Tree, Stage3medium)
// (benchmark::State& state) {
//   const auto n_threads = state.range(0);
//   for (auto _ : state) {
//     run_stage_3_medium(*appdata, *temp_storage, g_medium_cores, n_threads);
//   }
// }

// static void run_stage_4_medium(tree::AppData& appdata,
//                                tree::omp::TempStorage& temp_storage,
//                                const std::vector<int>& cores,
//                                const int n_threads) {
// #pragma omp parallel num_threads(n_threads)
//   {
//     bind_thread_to_cores(cores);
//     tree::omp::process_stage_4(appdata, temp_storage);
//   }
// }

// BENCHMARK_DEFINE_F(OMP_Tree, Stage4medium)
// (benchmark::State& state) {
//   const auto n_threads = state.range(0);
//   for (auto _ : state) {
//     run_stage_4_medium(*appdata, *temp_storage, g_medium_cores, n_threads);
//   }
// }

// static void run_stage_5_medium(tree::AppData& appdata,
//                                tree::omp::TempStorage& temp_storage,
//                                const std::vector<int>& cores,
//                                const int n_threads) {
// #pragma omp parallel num_threads(n_threads)
//   {
//     bind_thread_to_cores(cores);
//     tree::omp::process_stage_5(appdata, temp_storage);
//   }
// }

// BENCHMARK_DEFINE_F(OMP_Tree, Stage5medium)
// (benchmark::State& state) {
//   const auto n_threads = state.range(0);
//   for (auto _ : state) {
//     run_stage_5_medium(*appdata, *temp_storage, g_medium_cores, n_threads);
//   }
// }

// static void run_stage_6_medium(tree::AppData& appdata,
//                                tree::omp::TempStorage& temp_storage,
//                                const std::vector<int>& cores,
//                                const int n_threads) {
// #pragma omp parallel num_threads(n_threads)
//   {
//     bind_thread_to_cores(cores);
//     tree::omp::process_stage_6(appdata, temp_storage);
//   }
// }

// BENCHMARK_DEFINE_F(OMP_Tree, Stage6medium)
// (benchmark::State& state) {
//   const auto n_threads = state.range(0);
//   for (auto _ : state) {
//     run_stage_6_medium(*appdata, *temp_storage, g_medium_cores, n_threads);
//   }
// }

// static void run_stage_7_medium(tree::AppData& appdata,
//                                tree::omp::TempStorage& temp_storage,
//                                const std::vector<int>& cores,
//                                const int n_threads) {
// #pragma omp parallel num_threads(n_threads)
//   {
//     bind_thread_to_cores(cores);
//     tree::omp::process_stage_7(appdata, temp_storage);
//   }
// }

// BENCHMARK_DEFINE_F(OMP_Tree, Stage7medium)
// (benchmark::State& state) {
//   const auto n_threads = state.range(0);
//   for (auto _ : state) {
//     run_stage_7_medium(*appdata, *temp_storage, g_medium_cores, n_threads);
//   }
// }

// static void run_stage_1_big(tree::AppData& appdata,
//                             tree::omp::TempStorage& temp_storage,
//                             const std::vector<int>& cores,
//                             const int n_threads) {
// #pragma omp parallel num_threads(n_threads)
//   {
//     bind_thread_to_cores(cores);
//     tree::omp::process_stage_1(appdata, temp_storage);
//   }
// }

// BENCHMARK_DEFINE_F(OMP_Tree, Stage1big)
// (benchmark::State& state) {
//   const auto n_threads = state.range(0);
//   for (auto _ : state) {
//     run_stage_1_big(*appdata, *temp_storage, g_big_cores, n_threads);
//   }
// }

// static void run_stage_3_big(tree::AppData& appdata,
//                             tree::omp::TempStorage& temp_storage,
//                             const std::vector<int>& cores,
//                             const int n_threads) {
// #pragma omp parallel num_threads(n_threads)
//   {
//     bind_thread_to_cores(cores);
//     tree::omp::process_stage_3(appdata, temp_storage);
//   }
// }

// BENCHMARK_DEFINE_F(OMP_Tree, Stage3big)
// (benchmark::State& state) {
//   const auto n_threads = state.range(0);
//   for (auto _ : state) {
//     run_stage_3_big(*appdata, *temp_storage, g_big_cores, n_threads);
//   }
// }

// static void run_stage_4_big(tree::AppData& appdata,
//                             tree::omp::TempStorage& temp_storage,
//                             const std::vector<int>& cores,
//                             const int n_threads) {
// #pragma omp parallel num_threads(n_threads)
//   {
//     bind_thread_to_cores(cores);
//     tree::omp::process_stage_4(appdata, temp_storage);
//   }
// }

// BENCHMARK_DEFINE_F(OMP_Tree, Stage4big)
// (benchmark::State& state) {
//   const auto n_threads = state.range(0);
//   for (auto _ : state) {
//     run_stage_4_big(*appdata, *temp_storage, g_big_cores, n_threads);
//   }
// }

// static void run_stage_5_big(tree::AppData& appdata,
//                             tree::omp::TempStorage& temp_storage,
//                             const std::vector<int>& cores,
//                             const int n_threads) {
// #pragma omp parallel num_threads(n_threads)
//   {
//     bind_thread_to_cores(cores);
//     tree::omp::process_stage_5(appdata, temp_storage);
//   }
// }

// BENCHMARK_DEFINE_F(OMP_Tree, Stage5big)
// (benchmark::State& state) {
//   const auto n_threads = state.range(0);
//   for (auto _ : state) {
//     run_stage_5_big(*appdata, *temp_storage, g_big_cores, n_threads);
//   }
// }

// static void run_stage_6_big(tree::AppData& appdata,
//                             tree::omp::TempStorage& temp_storage,
//                             const std::vector<int>& cores,
//                             const int n_threads) {
// #pragma omp parallel num_threads(n_threads)
//   {
//     bind_thread_to_cores(cores);
//     tree::omp::process_stage_6(appdata, temp_storage);
//   }
// }

// BENCHMARK_DEFINE_F(OMP_Tree, Stage6big)
// (benchmark::State& state) {
//   const auto n_threads = state.range(0);
//   for (auto _ : state) {
//     run_stage_6_big(*appdata, *temp_storage, g_big_cores, n_threads);
//   }
// }

// static void run_stage_7_big(tree::AppData& appdata,
//                             tree::omp::TempStorage& temp_storage,
//                             const std::vector<int>& cores,
//                             const int n_threads) {
// #pragma omp parallel num_threads(n_threads)
//   {
//     bind_thread_to_cores(cores);
//     tree::omp::process_stage_7(appdata, temp_storage);
//   }
// }

// BENCHMARK_DEFINE_F(OMP_Tree, Stage7big)
// (benchmark::State& state) {
//   const auto n_threads = state.range(0);
//   for (auto _ : state) {
//     run_stage_7_big(*appdata, *temp_storage, g_big_cores, n_threads);
//   }
// }

// void RegisterStage1MediumBenchmarkWithRange(const std::vector<int>& pinable_medium_cores) {
//   for (size_t i = 1; i <= pinable_medium_cores.size(); ++i) {
//     ::benchmark::internal::RegisterBenchmarkInternal(new OMP_Tree_Stage1medium_Benchmark())
//         ->Arg(i)
//         ->Name("OMP_Tree/Stage1_medium")
//         ->Unit(benchmark::kMillisecond);
//   }
// }

// void RegisterStage3MediumBenchmarkWithRange(const std::vector<int>& pinable_medium_cores) {
//   for (size_t i = 1; i <= pinable_medium_cores.size(); ++i) {
//     ::benchmark::internal::RegisterBenchmarkInternal(new OMP_Tree_Stage3medium_Benchmark())
//         ->Arg(i)
//         ->Name("OMP_Tree/Stage3_medium")
//         ->Unit(benchmark::kMillisecond);
//   }
// }

// void RegisterStage4MediumBenchmarkWithRange(const std::vector<int>& pinable_medium_cores) {
//   for (size_t i = 1; i <= pinable_medium_cores.size(); ++i) {
//     ::benchmark::internal::RegisterBenchmarkInternal(new OMP_Tree_Stage4medium_Benchmark())
//         ->Arg(i)
//         ->Name("OMP_Tree/Stage4_medium")
//         ->Unit(benchmark::kMillisecond);
//   }
// }

// void RegisterStage5MediumBenchmarkWithRange(const std::vector<int>& pinable_medium_cores) {
//   for (size_t i = 1; i <= pinable_medium_cores.size(); ++i) {
//     ::benchmark::internal::RegisterBenchmarkInternal(new OMP_Tree_Stage5medium_Benchmark())
//         ->Arg(i)
//         ->Name("OMP_Tree/Stage5_medium")
//         ->Unit(benchmark::kMillisecond);
//   }
// }

// void RegisterStage6MediumBenchmarkWithRange(const std::vector<int>& pinable_medium_cores) {
//   for (size_t i = 1; i <= pinable_medium_cores.size(); ++i) {
//     ::benchmark::internal::RegisterBenchmarkInternal(new OMP_Tree_Stage6medium_Benchmark())
//         ->Arg(i)
//         ->Name("OMP_Tree/Stage6_medium")
//         ->Unit(benchmark::kMillisecond);
//   }
// }

// void RegisterStage7MediumBenchmarkWithRange(const std::vector<int>& pinable_medium_cores) {
//   for (size_t i = 1; i <= pinable_medium_cores.size(); ++i) {
//     ::benchmark::internal::RegisterBenchmarkInternal(new OMP_Tree_Stage7medium_Benchmark())
//         ->Arg(i)
//         ->Name("OMP_Tree/Stage7_medium")
//         ->Unit(benchmark::kMillisecond);
//   }
// }

// void RegisterStage1BigBenchmarkWithRange(const std::vector<int>& pinable_big_cores) {
//   for (size_t i = 1; i <= pinable_big_cores.size(); ++i) {
//     ::benchmark::internal::RegisterBenchmarkInternal(new OMP_Tree_Stage1big_Benchmark())
//         ->Arg(i)
//         ->Name("OMP_Tree/Stage1_big")
//         ->Unit(benchmark::kMillisecond);
//   }
// }

// void RegisterStage3BigBenchmarkWithRange(const std::vector<int>& pinable_big_cores) {
//   for (size_t i = 1; i <= pinable_big_cores.size(); ++i) {
//     ::benchmark::internal::RegisterBenchmarkInternal(new OMP_Tree_Stage3big_Benchmark())
//         ->Arg(i)
//         ->Name("OMP_Tree/Stage3_big")
//         ->Unit(benchmark::kMillisecond);
//   }
// }

// void RegisterStage4BigBenchmarkWithRange(const std::vector<int>& pinable_big_cores) {
//   for (size_t i = 1; i <= pinable_big_cores.size(); ++i) {
//     ::benchmark::internal::RegisterBenchmarkInternal(new OMP_Tree_Stage4big_Benchmark())
//         ->Arg(i)
//         ->Name("OMP_Tree/Stage4_big")
//         ->Unit(benchmark::kMillisecond);
//   }
// }

// void RegisterStage5BigBenchmarkWithRange(const std::vector<int>& pinable_big_cores) {
//   for (size_t i = 1; i <= pinable_big_cores.size(); ++i) {
//     ::benchmark::internal::RegisterBenchmarkInternal(new OMP_Tree_Stage5big_Benchmark())
//         ->Arg(i)
//         ->Name("OMP_Tree/Stage5_big")
//         ->Unit(benchmark::kMillisecond);
//   }
// }

// void RegisterStage6BigBenchmarkWithRange(const std::vector<int>& pinable_big_cores) {
//   for (size_t i = 1; i <= pinable_big_cores.size(); ++i) {
//     ::benchmark::internal::RegisterBenchmarkInternal(new OMP_Tree_Stage6big_Benchmark())
//         ->Arg(i)
//         ->Name("OMP_Tree/Stage6_big")
//         ->Unit(benchmark::kMillisecond);
//   }
// }

// void RegisterStage7BigBenchmarkWithRange(const std::vector<int>& pinable_big_cores) {
//   for (size_t i = 1; i <= pinable_big_cores.size(); ++i) {
//     ::benchmark::internal::RegisterBenchmarkInternal(new OMP_Tree_Stage7big_Benchmark())
//         ->Arg(i)
//         ->Name("OMP_Tree/Stage7_big")
//         ->Unit(benchmark::kMillisecond);
//   }
// }

// // ------------------------------------------------------------
// // Special case for stage 2
// // ------------------------------------------------------------

// static void run_stage_2_little(tree::AppData& appdata,
//                                const std::vector<int>& cores,
//                                const int n_threads,
//                                tree::omp::v2::TempStorage& temp_storage) {
// #pragma omp parallel num_threads(n_threads)
//   {
//     bind_thread_to_cores(cores);
//     tree::omp::v2::process_stage_2(appdata, temp_storage);
//   }
// }

// BENCHMARK_DEFINE_F(OMP_Tree, Stage2little)
// (benchmark::State& state) {
//   const auto n_threads = state.range(0);

//   for (auto _ : state) {
//     tree::omp::v2::TempStorage temp_storage(n_threads, n_threads);
//     run_stage_2_little(*appdata, g_little_cores, n_threads, temp_storage);
//   }

//   assert(std::ranges::is_sorted(appdata->u_morton_keys_sorted_s2));
// }

// void RegisterStage2LittleBenchmarkWithRange(const std::vector<int>& pinable_little_cores) {
//   for (size_t i = 1; i <= pinable_little_cores.size(); ++i) {
//     ::benchmark::internal::RegisterBenchmarkInternal(new OMP_Tree_Stage2little_Benchmark())
//         ->Arg(i)
//         ->Name("OMP_Tree/Stage2_little")
//         ->Unit(benchmark::kMillisecond);
//   }
// }

// static void run_stage_2_medium(tree::AppData& appdata,
//                                const std::vector<int>& cores,
//                                const int n_threads,
//                                tree::omp::v2::TempStorage& temp_storage) {
// #pragma omp parallel num_threads(n_threads)
//   {
//     bind_thread_to_cores(cores);
//     tree::omp::v2::process_stage_2(appdata, temp_storage);
//   }
// }

// BENCHMARK_DEFINE_F(OMP_Tree, Stage2medium)
// (benchmark::State& state) {
//   const auto n_threads = state.range(0);

//   for (auto _ : state) {
//     tree::omp::v2::TempStorage temp_storage(n_threads, n_threads);
//     run_stage_2_medium(*appdata, g_medium_cores, n_threads, temp_storage);
//   }

//   assert(std::ranges::is_sorted(appdata->u_morton_keys_sorted_s2));
// }

// void RegisterStage2MediumBenchmarkWithRange(const std::vector<int>& pinable_medium_cores) {
//   for (size_t i = 1; i <= pinable_medium_cores.size(); ++i) {
//     ::benchmark::internal::RegisterBenchmarkInternal(new OMP_Tree_Stage2medium_Benchmark())
//         ->Arg(i)
//         ->Name("OMP_Tree/Stage2_medium")
//         ->Unit(benchmark::kMillisecond);
//   }
// }

// static void run_stage_2_big(tree::AppData& appdata,
//                             const std::vector<int>& cores,
//                             const int n_threads,
//                             tree::omp::v2::TempStorage& temp_storage) {
// #pragma omp parallel num_threads(n_threads)
//   {
//     bind_thread_to_cores(cores);
//     tree::omp::v2::process_stage_2(appdata, temp_storage);
//   }
// }

// BENCHMARK_DEFINE_F(OMP_Tree, Stage2big)
// (benchmark::State& state) {
//   const auto n_threads = state.range(0);

//   for (auto _ : state) {
//     tree::omp::v2::TempStorage temp_storage(n_threads, n_threads);
//     run_stage_2_big(*appdata, g_big_cores, n_threads, temp_storage);
//   }

//   assert(std::ranges::is_sorted(appdata->u_morton_keys_sorted_s2));
// }

// void RegisterStage2BigBenchmarkWithRange(const std::vector<int>& pinable_big_cores) {
//   for (size_t i = 1; i <= pinable_big_cores.size(); ++i) {
//     ::benchmark::internal::RegisterBenchmarkInternal(new OMP_Tree_Stage2big_Benchmark())
//         ->Arg(i)
//         ->Name("OMP_Tree/Stage2_big")
//         ->Unit(benchmark::kMillisecond);
//   }
// }

// ------------------------------------------------------------
// std::sort
// ------------------------------------------------------------

// BENCHMARK_DEFINE_F(OMP_Tree, Stage2std)
// (benchmark::State& state) {
//   for (auto _ : state) {
//     std::ranges::sort(appdata->u_morton_keys_s1);
//   }

//   assert(std::ranges::is_sorted(appdata->u_morton_keys_s1));
// }

// void RegisterStage2stdBenchmark() {
//   ::benchmark::internal::RegisterBenchmarkInternal(new OMP_Tree_Stage2std_Benchmark())
//       ->Name("OMP_Tree/Stage2_std")
//       ->Unit(benchmark::kMillisecond);
// }

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------

int main(int argc, char** argv) {
  parse_args(argc, argv);

  RegisterStage1LittleBenchmarkWithRange(g_little_cores);
  RegisterStage2LittleBenchmarkWithRange(g_little_cores);
  RegisterStage3LittleBenchmarkWithRange(g_little_cores);
  RegisterStage4LittleBenchmarkWithRange(g_little_cores);
  RegisterStage5LittleBenchmarkWithRange(g_little_cores);
  RegisterStage6LittleBenchmarkWithRange(g_little_cores);
  RegisterStage7LittleBenchmarkWithRange(g_little_cores);

  // RegisterStage1LittleBenchmarkWithRange(g_little_cores);
  // RegisterStage1MediumBenchmarkWithRange(g_medium_cores);
  // RegisterStage1BigBenchmarkWithRange(g_big_cores);

  // // RegisterStage2LittleBenchmarkWithRange(g_little_cores);
  // // RegisterStage2MediumBenchmarkWithRange(g_medium_cores);
  // // RegisterStage2BigBenchmarkWithRange(g_big_cores);

  // // RegisterStage2stdBenchmark();

  // RegisterStage3LittleBenchmarkWithRange(g_little_cores);
  // RegisterStage3MediumBenchmarkWithRange(g_medium_cores);
  // RegisterStage3BigBenchmarkWithRange(g_big_cores);

  // RegisterStage4LittleBenchmarkWithRange(g_little_cores);
  // RegisterStage4MediumBenchmarkWithRange(g_medium_cores);
  // RegisterStage4BigBenchmarkWithRange(g_big_cores);

  // RegisterStage5LittleBenchmarkWithRange(g_little_cores);
  // RegisterStage5MediumBenchmarkWithRange(g_medium_cores);
  // RegisterStage5BigBenchmarkWithRange(g_big_cores);

  // RegisterStage6LittleBenchmarkWithRange(g_little_cores);
  // RegisterStage6MediumBenchmarkWithRange(g_medium_cores);
  // RegisterStage6BigBenchmarkWithRange(g_big_cores);

  // RegisterStage7LittleBenchmarkWithRange(g_little_cores);
  // RegisterStage7MediumBenchmarkWithRange(g_medium_cores);
  // RegisterStage7BigBenchmarkWithRange(g_big_cores);

  // // Initialize and run benchmarks
  const auto storage_location = helpers::get_benchmark_storage_location();
  const auto out_name = storage_location.string() + "/BM_Tree_OMP_" + g_device_id + ".json";

  auto [new_argc, new_argv] = sanitize_argc_argv_for_benchmark(argc, argv, out_name);

  benchmark::Initialize(&new_argc, new_argv.data());
  if (benchmark::ReportUnrecognizedArguments(new_argc, new_argv.data())) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
