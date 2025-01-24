#include <benchmark/benchmark.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>

#include "app.hpp"
#include "spdlog/common.h"
#include "tree/tree_appdata.hpp"
#include "tree/vulkan/vk_dispatcher.hpp"

// ----------------------------------------------------------------
// Baseline
// ----------------------------------------------------------------

static void run_baseline(tree::AppData& app_data,
                         vulkan::TmpStorage& stage_3_tmp_storage) {
  tree::vulkan::Singleton::getInstance().process_stage_1(app_data);
  tree::vulkan::Singleton::getInstance().process_stage_2(app_data);
  tree::vulkan::Singleton::getInstance().process_stage_3(app_data,
                                                         stage_3_tmp_storage);
  tree::vulkan::Singleton::getInstance().process_stage_4(app_data);
  tree::vulkan::Singleton::getInstance().process_stage_5(app_data);
  tree::vulkan::Singleton::getInstance().process_stage_6(app_data);
  tree::vulkan::Singleton::getInstance().process_stage_7(app_data);
}

static void VK_Baseline_Benchmark(benchmark::State& state) {
  auto mr = tree::vulkan::Singleton::getInstance().get_mr();
  tree::AppData app_data(mr);
  vulkan::TmpStorage stage_3_tmp_storage(mr, app_data.get_n_input());

  for (auto _ : state) {
    run_baseline(app_data, stage_3_tmp_storage);
  }
}

BENCHMARK(VK_Baseline_Benchmark)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Individual stages
// ----------------------------------------------------------------

class VK_Tree : public benchmark::Fixture {
 protected:
  void SetUp(benchmark::State&) override {
    auto mr = tree::vulkan::Singleton::getInstance().get_mr();

    app_data = std::make_unique<tree::AppData>(mr);
    stage_3_tmp_storage =
        std::make_unique<vulkan::TmpStorage>(mr, app_data->get_n_input());

    // process the data once
    run_baseline(*app_data, *stage_3_tmp_storage);
  }

  void TearDown(benchmark::State&) override {
    app_data.reset();
    stage_3_tmp_storage.reset();
  }

  std::unique_ptr<tree::AppData> app_data;
  std::unique_ptr<vulkan::TmpStorage> stage_3_tmp_storage;
};

// ----------------------------------------------------------------
// Stage 1
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_Tree, Stage1)
(benchmark::State& state) {
  for (auto _ : state) {
    tree::vulkan::Singleton::getInstance().process_stage_1(*app_data);
  }
}

BENCHMARK_REGISTER_F(VK_Tree, Stage1)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

// ----------------------------------------------------------------
// Stage 2
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_Tree, Stage2)
(benchmark::State& state) {
  for (auto _ : state) {
    tree::vulkan::Singleton::getInstance().process_stage_2(*app_data);
  }
}

BENCHMARK_REGISTER_F(VK_Tree, Stage2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

// ----------------------------------------------------------------
// Stage 3
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_Tree, Stage3)
(benchmark::State& state) {
  for (auto _ : state) {
    tree::vulkan::Singleton::getInstance().process_stage_3(
        *app_data, *stage_3_tmp_storage);
  }
}

BENCHMARK_REGISTER_F(VK_Tree, Stage3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

// ----------------------------------------------------------------
// Stage 4
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_Tree, Stage4)
(benchmark::State& state) {
  for (auto _ : state) {
    tree::vulkan::Singleton::getInstance().process_stage_4(*app_data);
  }
}

BENCHMARK_REGISTER_F(VK_Tree, Stage4)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

// ----------------------------------------------------------------
// Stage 5
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_Tree, Stage5)
(benchmark::State& state) {
  for (auto _ : state) {
    tree::vulkan::Singleton::getInstance().process_stage_5(*app_data);
  }
}

BENCHMARK_REGISTER_F(VK_Tree, Stage5)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

// ----------------------------------------------------------------
// Stage 6
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_Tree, Stage6)
(benchmark::State& state) {
  for (auto _ : state) {
    tree::vulkan::Singleton::getInstance().process_stage_6(*app_data);
  }
}

BENCHMARK_REGISTER_F(VK_Tree, Stage6)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

// ----------------------------------------------------------------
// Stage 7
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(VK_Tree, Stage7)
(benchmark::State& state) {
  for (auto _ : state) {
    tree::vulkan::Singleton::getInstance().process_stage_7(*app_data);
  }
}

BENCHMARK_REGISTER_F(VK_Tree, Stage7)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);

int main(int argc, char** argv) {
  //   parse_args(argc, argv);
  //   spdlog::set_level(spdlog::level::off);

  //   benchmark::Initialize(&argc, argv);
  //   benchmark::RunSpecifiedBenchmarks();
  //   benchmark::Shutdown();

  spdlog::set_level(spdlog::level::trace);
  auto mr = tree::vulkan::Singleton::getInstance().get_mr();
  tree::AppData app_data(mr);
  vulkan::TmpStorage tmp_storage(mr, app_data.get_n_input());

  for (int i = 0; i < 10; i++) {

  tree::vulkan::Singleton::getInstance().process_stage_1(app_data);
  tree::vulkan::Singleton::getInstance().process_stage_2(app_data);
  tree::vulkan::Singleton::getInstance().process_stage_3(app_data, tmp_storage);
  tree::vulkan::Singleton::getInstance().process_stage_4(app_data);
  tree::vulkan::Singleton::getInstance().process_stage_5(app_data);
  tree::vulkan::Singleton::getInstance().process_stage_6(app_data);
  tree::vulkan::Singleton::getInstance().process_stage_7(app_data);

    std::cout << "===========" << std::endl;

  }

  return 0;
}