#include <benchmark/benchmark.h>

#include <CLI/CLI.hpp>

#include "affinity.hpp"
#include "app.hpp"
#include "tree/omp/tree_kernel.hpp"
#include "tree/tree_appdata.hpp"

// ------------------------------------------------------------
// Global variables
// ------------------------------------------------------------

class OMP_Tree : public benchmark::Fixture {
 protected:
  void SetUp(benchmark::State&) override {
    app_data = std::make_unique<tree::AppData>(std::pmr::new_delete_resource());

    // need to run the first

    tree::omp::process_stage_1(*app_data);
    std::sort(app_data->u_morton_keys.begin(), app_data->u_morton_keys.end());
    tree::omp::process_stage_3(*app_data);
    tree::omp::process_stage_4(*app_data);
    tree::omp::process_stage_5(*app_data);
    tree::omp::process_stage_6(*app_data);
    tree::omp::process_stage_7(*app_data);
  }

  void TearDown(benchmark::State&) override { app_data.reset(); }

  std::unique_ptr<tree::AppData> app_data;
};

// ------------------------------------------------------------
// Helper macros for stage benchmarks
// ------------------------------------------------------------

// static void run_stage_1_Little(tree::AppData& app_data,
//                                const std::vector<int>& cores,
//                                const int n_threads) {
//   _Pragma("omp parallel num_threads(n_threads)") {
//     bind_thread_to_core(cores);
//     tree::omp::process_stage_1(app_data);
//   }
// }

// BENCHMARK_DEFINE_F(OMP_Tree, Stage1Little)
// (benchmark::State& state) {
//   auto n_threads = state.range(0);

//   for (auto _ : state) {
//     run_stage_1_Little(*app_data, g_little_cores, n_threads);
//   }
// }

// #define DEFINE_STAGE_BENCHMARK(stage_num, core_type)                         \
//   static void run_stage_##stage_num##_##core_type(                           \
//       tree::AppData& app_data,                                               \
//       const std::vector<int>& cores,                                         \
//       const int n_threads) {                                                 \
//     _Pragma("omp parallel num_threads(n_threads)") {                         \
//       bind_thread_to_core(cores);                                            \
//       tree::omp::process_stage_##stage_num(app_data);                        \
//     }                                                                        \
//   }                                                                          \
//                                                                              \
//   class OMP_Tree_Stage##stage_num##core_type##_Benchmark : public OMP_Tree { \
//    public:                                                                   \
//     void BenchmarkCase(benchmark::State& state) {                            \
//       const auto n_threads = state.range(0);                                 \
//       for (auto _ : state) {                                                 \
//         run_stage_##stage_num##_##core_type(                                 \
//             *app_data, g_##core_type##_cores, n_threads);                    \
//       }                                                                      \
//     }                                                                        \
//   };

// #define REGISTER_STAGE_BENCHMARK(stage_num, core_type)                 \
//   void RegisterStage##stage_num##core_type##BenchmarkWithRange(        \
//       const std::vector<int>& pinable_##core_type##_cores) {           \
//     for (size_t i = 1; i <= pinable_##core_type##_cores.size(); ++i) { \
//       ::benchmark::internal::RegisterBenchmarkInternal(                \
//           new OMP_Tree_Stage##stage_num##core_type##_Benchmark())      \
//           ->Arg(i)                                                     \
//           ->Name("OMP_Tree/Stage" #stage_num "_" #core_type)           \
//           ->Unit(benchmark::kMillisecond);                             \
//     }                                                                  \
//   }

// #define DEFINE_STAGE_BENCHMARK(stage_num, core_type)                    \
//   static void run_stage_##stage_num##_##core_type(                      \
//       tree::AppData& app_data,                                          \
//       const std::vector<int>& cores,                                    \
//       const int n_threads) {                                            \
//     _Pragma("omp parallel num_threads(n_threads)") {                    \
//       bind_thread_to_core(cores);                                       \
//       tree::omp::process_stage_##stage_num(app_data);                   \
//     }                                                                   \
//   }                                                                     \
//                                                                         \
//   BENCHMARK_DEFINE_F(OMP_CifarDense, Stage##stage_num##core_type)       \
//   (benchmark::State & state) {                                          \
//     auto cores = g_device.get_pinable_cores(k##core_type##CoreType);    \
//                                                                         \
//     const auto n_threads = state.range(0);                              \
//     for (auto _ : state) {                                              \
//       run_stage_##stage_num##_##core_type(*app_data, cores, n_threads); \
//     }                                                                   \
//   }

// DEFINE_STAGE_BENCHMARK(1, little)
// DEFINE_STAGE_BENCHMARK(3, little)
// DEFINE_STAGE_BENCHMARK(4, little)
// DEFINE_STAGE_BENCHMARK(5, little)
// DEFINE_STAGE_BENCHMARK(6, little)
// DEFINE_STAGE_BENCHMARK(7, little)

static void run_stage_1_little(tree::AppData& app_data,
                               const std::vector<int>& cores,
                               const int n_threads) {
  _Pragma("omp parallel num_threads(n_threads)") {
    bind_thread_to_core(cores);
    tree::omp::process_stage_1(app_data);
  }
}

BENCHMARK_DEFINE_F(OMP_Tree, Stage1little)
(benchmark::State& state) {
  const auto n_threads = state.range(0);
  for (auto _ : state) {
    run_stage_1_little(*app_data, g_little_cores, n_threads);
  }
}

// Add stages 2-7
// static void run_stage_2_little(tree::AppData& app_data,
//                                const std::vector<int>& cores,
//                                const int n_threads) {
//   _Pragma("omp parallel num_threads(n_threads)") {
//     bind_thread_to_core(cores);
//     tree::omp::v2::process_stage_2(app_data);
//   }
// }

// BENCHMARK_DEFINE_F(OMP_Tree, Stage2little)
// (benchmark::State& state) {

//   const auto n_threads = state.range(0);

//   tree::omp::v2::TempStorage temp_storage(n_threads, n_threads);

//   for (auto _ : state) {
//     run_stage_2_little(*app_data, g_little_cores, n_threads);
//   }
// }

static void run_stage_3_little(tree::AppData& app_data,
                               const std::vector<int>& cores,
                               const int n_threads) {
  _Pragma("omp parallel num_threads(n_threads)") {
    bind_thread_to_core(cores);
    tree::omp::process_stage_3(app_data);
  }
}

BENCHMARK_DEFINE_F(OMP_Tree, Stage3little)
(benchmark::State& state) {
  const auto n_threads = state.range(0);
  for (auto _ : state) {
    run_stage_3_little(*app_data, g_little_cores, n_threads);
  }
}

static void run_stage_4_little(tree::AppData& app_data,
                               const std::vector<int>& cores,
                               const int n_threads) {
  _Pragma("omp parallel num_threads(n_threads)") {
    bind_thread_to_core(cores);
    tree::omp::process_stage_4(app_data);
  }
}

BENCHMARK_DEFINE_F(OMP_Tree, Stage4little)
(benchmark::State& state) {
  const auto n_threads = state.range(0);
  for (auto _ : state) {
    run_stage_4_little(*app_data, g_little_cores, n_threads);
  }
}

static void run_stage_5_little(tree::AppData& app_data,
                               const std::vector<int>& cores,
                               const int n_threads) {
  _Pragma("omp parallel num_threads(n_threads)") {
    bind_thread_to_core(cores);
    tree::omp::process_stage_5(app_data);
  }
}

BENCHMARK_DEFINE_F(OMP_Tree, Stage5little)
(benchmark::State& state) {
  const auto n_threads = state.range(0);
  for (auto _ : state) {
    run_stage_5_little(*app_data, g_little_cores, n_threads);
  }
}

static void run_stage_6_little(tree::AppData& app_data,
                               const std::vector<int>& cores,
                               const int n_threads) {
  _Pragma("omp parallel num_threads(n_threads)") {
    bind_thread_to_core(cores);
    tree::omp::process_stage_6(app_data);
  }
}

BENCHMARK_DEFINE_F(OMP_Tree, Stage6little)
(benchmark::State& state) {
  const auto n_threads = state.range(0);
  for (auto _ : state) {
    run_stage_6_little(*app_data, g_little_cores, n_threads);
  }
}

static void run_stage_7_little(tree::AppData& app_data,
                               const std::vector<int>& cores,
                               const int n_threads) {
  _Pragma("omp parallel num_threads(n_threads)") {
    bind_thread_to_core(cores);
    tree::omp::process_stage_7(app_data);
  }
}

BENCHMARK_DEFINE_F(OMP_Tree, Stage7little)
(benchmark::State& state) {
  const auto n_threads = state.range(0);
  for (auto _ : state) {
    run_stage_7_little(*app_data, g_little_cores, n_threads);
  }
}

// Then define the registration functions for each stage
void RegisterStage1LittleBenchmarkWithRange(
    const std::vector<int>& pinable_little_cores) {
  for (size_t i = 1; i <= pinable_little_cores.size(); ++i) {
    ::benchmark::internal::RegisterBenchmarkInternal(
        new OMP_Tree_Stage1little_Benchmark())
        ->Arg(i)
        ->Name("OMP_Tree/Stage1_little")
        ->Unit(benchmark::kMillisecond);
  }
}

// void RegisterStage2LittleBenchmarkWithRange(
//     const std::vector<int>& pinable_little_cores) {
//   for (size_t i = 1; i <= pinable_little_cores.size(); ++i) {
//     ::benchmark::internal::RegisterBenchmarkInternal(
//         new OMP_Tree_Stage2little_Benchmark())
//         ->Arg(i)
//         ->Name("OMP_Tree/Stage2_little")
//         ->Unit(benchmark::kMillisecond);
//   }
// }

void RegisterStage3LittleBenchmarkWithRange(
    const std::vector<int>& pinable_little_cores) {
  for (size_t i = 1; i <= pinable_little_cores.size(); ++i) {
    ::benchmark::internal::RegisterBenchmarkInternal(
        new OMP_Tree_Stage3little_Benchmark())
        ->Arg(i)
        ->Name("OMP_Tree/Stage3_little")
        ->Unit(benchmark::kMillisecond);
  }
}

void RegisterStage4LittleBenchmarkWithRange(
    const std::vector<int>& pinable_little_cores) {
  for (size_t i = 1; i <= pinable_little_cores.size(); ++i) {
    ::benchmark::internal::RegisterBenchmarkInternal(
        new OMP_Tree_Stage4little_Benchmark())
        ->Arg(i)
        ->Name("OMP_Tree/Stage4_little")
        ->Unit(benchmark::kMillisecond);
  }
}

void RegisterStage5LittleBenchmarkWithRange(
    const std::vector<int>& pinable_little_cores) {
  for (size_t i = 1; i <= pinable_little_cores.size(); ++i) {
    ::benchmark::internal::RegisterBenchmarkInternal(
        new OMP_Tree_Stage5little_Benchmark())
        ->Arg(i)
        ->Name("OMP_Tree/Stage5_little")
        ->Unit(benchmark::kMillisecond);
  }
}

void RegisterStage6LittleBenchmarkWithRange(
    const std::vector<int>& pinable_little_cores) {
  for (size_t i = 1; i <= pinable_little_cores.size(); ++i) {
    ::benchmark::internal::RegisterBenchmarkInternal(
        new OMP_Tree_Stage6little_Benchmark())
        ->Arg(i)
        ->Name("OMP_Tree/Stage6_little")
        ->Unit(benchmark::kMillisecond);
  }
}

void RegisterStage7LittleBenchmarkWithRange(
    const std::vector<int>& pinable_little_cores) {
  for (size_t i = 1; i <= pinable_little_cores.size(); ++i) {
    ::benchmark::internal::RegisterBenchmarkInternal(
        new OMP_Tree_Stage7little_Benchmark())
        ->Arg(i)
        ->Name("OMP_Tree/Stage7_little")
        ->Unit(benchmark::kMillisecond);
  }
}

static void run_stage_1_medium(tree::AppData& app_data,
                               const std::vector<int>& cores,
                               const int n_threads) {
  _Pragma("omp parallel num_threads(n_threads)") {
    bind_thread_to_core(cores);
    tree::omp::process_stage_1(app_data);
  }
}

BENCHMARK_DEFINE_F(OMP_Tree, Stage1medium)
(benchmark::State& state) {
  const auto n_threads = state.range(0);
  for (auto _ : state) {
    run_stage_1_medium(*app_data, g_medium_cores, n_threads);
  }
}

static void run_stage_3_medium(tree::AppData& app_data,
                               const std::vector<int>& cores,
                               const int n_threads) {
  _Pragma("omp parallel num_threads(n_threads)") {
    bind_thread_to_core(cores);
    tree::omp::process_stage_3(app_data);
  }
}

BENCHMARK_DEFINE_F(OMP_Tree, Stage3medium)
(benchmark::State& state) {
  const auto n_threads = state.range(0);
  for (auto _ : state) {
    run_stage_3_medium(*app_data, g_medium_cores, n_threads);
  }
}

static void run_stage_4_medium(tree::AppData& app_data,
                               const std::vector<int>& cores,
                               const int n_threads) {
  _Pragma("omp parallel num_threads(n_threads)") {
    bind_thread_to_core(cores);
    tree::omp::process_stage_4(app_data);
  }
}

BENCHMARK_DEFINE_F(OMP_Tree, Stage4medium)
(benchmark::State& state) {
  const auto n_threads = state.range(0);
  for (auto _ : state) {
    run_stage_4_medium(*app_data, g_medium_cores, n_threads);
  }
}

static void run_stage_5_medium(tree::AppData& app_data,
                               const std::vector<int>& cores,
                               const int n_threads) {
  _Pragma("omp parallel num_threads(n_threads)") {
    bind_thread_to_core(cores);
    tree::omp::process_stage_5(app_data);
  }
}

BENCHMARK_DEFINE_F(OMP_Tree, Stage5medium)
(benchmark::State& state) {
  const auto n_threads = state.range(0);
  for (auto _ : state) {
    run_stage_5_medium(*app_data, g_medium_cores, n_threads);
  }
}

static void run_stage_6_medium(tree::AppData& app_data,
                               const std::vector<int>& cores,
                               const int n_threads) {
  _Pragma("omp parallel num_threads(n_threads)") {
    bind_thread_to_core(cores);
    tree::omp::process_stage_6(app_data);
  }
}

BENCHMARK_DEFINE_F(OMP_Tree, Stage6medium)
(benchmark::State& state) {
  const auto n_threads = state.range(0);
  for (auto _ : state) {
    run_stage_6_medium(*app_data, g_medium_cores, n_threads);
  }
}

static void run_stage_7_medium(tree::AppData& app_data,
                               const std::vector<int>& cores,
                               const int n_threads) {
  _Pragma("omp parallel num_threads(n_threads)") {
    bind_thread_to_core(cores);
    tree::omp::process_stage_7(app_data);
  }
}

BENCHMARK_DEFINE_F(OMP_Tree, Stage7medium)
(benchmark::State& state) {
  const auto n_threads = state.range(0);
  for (auto _ : state) {
    run_stage_7_medium(*app_data, g_medium_cores, n_threads);
  }
}

void RegisterStage1MediumBenchmarkWithRange(
    const std::vector<int>& pinable_medium_cores) {
  for (size_t i = 1; i <= pinable_medium_cores.size(); ++i) {
    ::benchmark::internal::RegisterBenchmarkInternal(
        new OMP_Tree_Stage1medium_Benchmark())
        ->Arg(i)
        ->Name("OMP_Tree/Stage1_medium")
        ->Unit(benchmark::kMillisecond);
  }
}

void RegisterStage3MediumBenchmarkWithRange(
    const std::vector<int>& pinable_medium_cores) {
  for (size_t i = 1; i <= pinable_medium_cores.size(); ++i) {
    ::benchmark::internal::RegisterBenchmarkInternal(
        new OMP_Tree_Stage3medium_Benchmark())
        ->Arg(i)
        ->Name("OMP_Tree/Stage3_medium")
        ->Unit(benchmark::kMillisecond);
  }
}

void RegisterStage4MediumBenchmarkWithRange(
    const std::vector<int>& pinable_medium_cores) {
  for (size_t i = 1; i <= pinable_medium_cores.size(); ++i) {
    ::benchmark::internal::RegisterBenchmarkInternal(
        new OMP_Tree_Stage4medium_Benchmark())
        ->Arg(i)
        ->Name("OMP_Tree/Stage4_medium")
        ->Unit(benchmark::kMillisecond);
  }
}

void RegisterStage5MediumBenchmarkWithRange(
    const std::vector<int>& pinable_medium_cores) {
  for (size_t i = 1; i <= pinable_medium_cores.size(); ++i) {
    ::benchmark::internal::RegisterBenchmarkInternal(
        new OMP_Tree_Stage5medium_Benchmark())
        ->Arg(i)
        ->Name("OMP_Tree/Stage5_medium")
        ->Unit(benchmark::kMillisecond);
  }
}

void RegisterStage6MediumBenchmarkWithRange(
    const std::vector<int>& pinable_medium_cores) {
  for (size_t i = 1; i <= pinable_medium_cores.size(); ++i) {
    ::benchmark::internal::RegisterBenchmarkInternal(
        new OMP_Tree_Stage6medium_Benchmark())
        ->Arg(i)
        ->Name("OMP_Tree/Stage6_medium")
        ->Unit(benchmark::kMillisecond);
  }
}

void RegisterStage7MediumBenchmarkWithRange(
    const std::vector<int>& pinable_medium_cores) {
  for (size_t i = 1; i <= pinable_medium_cores.size(); ++i) {
    ::benchmark::internal::RegisterBenchmarkInternal(
        new OMP_Tree_Stage7medium_Benchmark())
        ->Arg(i)
        ->Name("OMP_Tree/Stage7_medium")
        ->Unit(benchmark::kMillisecond);
  }
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------

int main(int argc, char** argv) {
  parse_args(argc, argv);

  RegisterStage1LittleBenchmarkWithRange(g_little_cores);
  // RegisterStage2LittleBenchmarkWithRange(g_little_cores);
  RegisterStage3LittleBenchmarkWithRange(g_little_cores);
  RegisterStage4LittleBenchmarkWithRange(g_little_cores);
  RegisterStage5LittleBenchmarkWithRange(g_little_cores);
  RegisterStage6LittleBenchmarkWithRange(g_little_cores);
  RegisterStage7LittleBenchmarkWithRange(g_little_cores);

  RegisterStage1MediumBenchmarkWithRange(g_medium_cores);
  RegisterStage3MediumBenchmarkWithRange(g_medium_cores);
  RegisterStage4MediumBenchmarkWithRange(g_medium_cores);
  RegisterStage5MediumBenchmarkWithRange(g_medium_cores);
  RegisterStage6MediumBenchmarkWithRange(g_medium_cores);
  RegisterStage7MediumBenchmarkWithRange(g_medium_cores);

  // Initialize and run benchmarks
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  return 0;
}
