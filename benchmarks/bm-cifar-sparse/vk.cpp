#include <benchmark/benchmark.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>

#include "app.hpp"
#include "cifar-sparse/sparse_appdata.hpp"
#include "cifar-sparse/vulkan/vk_dispatcher.hpp"

// ----------------------------------------------------------------
// Baseline
// ----------------------------------------------------------------

static void run_baseline_pinned(cifar_sparse::AppData& app_data) {
  cifar_sparse::vulkan::Singleton::getInstance().process_stage_1(app_data);
  cifar_sparse::vulkan::Singleton::getInstance().process_stage_2(app_data);
  cifar_sparse::vulkan::Singleton::getInstance().process_stage_3(app_data);
  cifar_sparse::vulkan::Singleton::getInstance().process_stage_4(app_data);
  cifar_sparse::vulkan::Singleton::getInstance().process_stage_5(app_data);
  cifar_sparse::vulkan::Singleton::getInstance().process_stage_6(app_data);
  cifar_sparse::vulkan::Singleton::getInstance().process_stage_7(app_data);
  cifar_sparse::vulkan::Singleton::getInstance().process_stage_8(app_data);
  cifar_sparse::vulkan::Singleton::getInstance().process_stage_9(app_data);
}

static void VK_CifarSparse_Baseline(benchmark::State& state) {
  auto mr = cifar_sparse::vulkan::Singleton::getInstance().get_mr();
  cifar_sparse::AppData app_data(mr);

  for (auto _ : state) {
    run_baseline_pinned(app_data);
  }
}

BENCHMARK(VK_CifarSparse_Baseline)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

class VK_CifarSparse : public benchmark::Fixture {
 protected:
  void SetUp(benchmark::State&) override {
    auto mr = cifar_sparse::vulkan::Singleton::getInstance().get_mr();

    app_data = std::make_unique<cifar_sparse::AppData>(mr);

    // process the data once
    run_baseline_pinned(*app_data);
  }

  void TearDown(benchmark::State&) override { app_data.reset(); }

  std::unique_ptr<cifar_sparse::AppData> app_data;
};

// ----------------------------------------------------------------
// Stage 1
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_CifarSparse, Stage1)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_sparse::vulkan::Singleton::getInstance().process_stage_1(*app_data);
  }
}

BENCHMARK_REGISTER_F(VK_CifarSparse, Stage1)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

// ----------------------------------------------------------------
// Stage 2
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_CifarSparse, Stage2)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_sparse::vulkan::Singleton::getInstance().process_stage_2(*app_data);
  }
}

BENCHMARK_REGISTER_F(VK_CifarSparse, Stage2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

// ----------------------------------------------------------------
// Stage 3
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_CifarSparse, Stage3)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_sparse::vulkan::Singleton::getInstance().process_stage_3(*app_data);
  }
}

BENCHMARK_REGISTER_F(VK_CifarSparse, Stage3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

// ----------------------------------------------------------------
// Stage 4
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_CifarSparse, Stage4)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_sparse::vulkan::Singleton::getInstance().process_stage_4(*app_data);
  }
}

BENCHMARK_REGISTER_F(VK_CifarSparse, Stage4)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

// ----------------------------------------------------------------
// Stage 5
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_CifarSparse, Stage5)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_sparse::vulkan::Singleton::getInstance().process_stage_5(*app_data);
  }
}

BENCHMARK_REGISTER_F(VK_CifarSparse, Stage5)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

// ----------------------------------------------------------------
// Stage 6
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_CifarSparse, Stage6)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_sparse::vulkan::Singleton::getInstance().process_stage_6(*app_data);
  }
}

BENCHMARK_REGISTER_F(VK_CifarSparse, Stage6)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

// ----------------------------------------------------------------
// Stage 7
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_CifarSparse, Stage7)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_sparse::vulkan::Singleton::getInstance().process_stage_7(*app_data);
  }
}

BENCHMARK_REGISTER_F(VK_CifarSparse, Stage7)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

// ----------------------------------------------------------------
// Stage 8
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_CifarSparse, Stage8)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_sparse::vulkan::Singleton::getInstance().process_stage_8(*app_data);
  }
}

BENCHMARK_REGISTER_F(VK_CifarSparse, Stage8)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

// ----------------------------------------------------------------
// Stage 9
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_CifarSparse, Stage9)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_sparse::vulkan::Singleton::getInstance().process_stage_9(*app_data);
  }
}

BENCHMARK_REGISTER_F(VK_CifarSparse, Stage9)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

// ----------------------------------------------------------------
// Main
// ----------------------------------------------------------------
int main(int argc, char** argv) {
  parse_args(argc, argv);
  spdlog::set_level(spdlog::level::off);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  return 0;
}