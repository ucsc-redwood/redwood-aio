#include <benchmark/benchmark.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>

#include "app.hpp"
#include "cifar-dense/dense_appdata.hpp"
#include "cifar-dense/vulkan/vk_dispatcher.hpp"

// ----------------------------------------------------------------
// Baseline
// ----------------------------------------------------------------

static void run_baseline_pinned(cifar_dense::AppData& app_data) {
  cifar_dense::vulkan::Singleton::getInstance().process_stage_1(app_data);
  cifar_dense::vulkan::Singleton::getInstance().process_stage_2(app_data);
  cifar_dense::vulkan::Singleton::getInstance().process_stage_3(app_data);
  cifar_dense::vulkan::Singleton::getInstance().process_stage_4(app_data);
  cifar_dense::vulkan::Singleton::getInstance().process_stage_5(app_data);
  cifar_dense::vulkan::Singleton::getInstance().process_stage_6(app_data);
  cifar_dense::vulkan::Singleton::getInstance().process_stage_7(app_data);
  cifar_dense::vulkan::Singleton::getInstance().process_stage_8(app_data);
  cifar_dense::vulkan::Singleton::getInstance().process_stage_9(app_data);
}

static void OMP_BaselinePinned_Benchmark(benchmark::State& state) {
  auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();
  cifar_dense::AppData app_data(mr);

  for (auto _ : state) {
    run_baseline_pinned(app_data);
  }
}

BENCHMARK(OMP_BaselinePinned_Benchmark)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Individual stages
// ----------------------------------------------------------------

class OMP_CifarDense : public benchmark::Fixture {
 protected:
  void SetUp(benchmark::State&) override {
    auto mr = cifar_dense::vulkan::Singleton::getInstance().get_mr();

    app_data = std::make_unique<cifar_dense::AppData>(mr);

    // process the data once
    run_baseline_pinned(*app_data);
  }

  void TearDown(benchmark::State&) override { app_data.reset(); }

  std::unique_ptr<cifar_dense::AppData> app_data;
};

// ----------------------------------------------------------------
// Stage 1
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(OMP_CifarDense, Stage1)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_dense::vulkan::Singleton::getInstance().process_stage_1(*app_data);
  }
}

BENCHMARK_REGISTER_F(OMP_CifarDense, Stage1)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 2
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(OMP_CifarDense, Stage2)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_dense::vulkan::Singleton::getInstance().process_stage_2(*app_data);
  }
}

BENCHMARK_REGISTER_F(OMP_CifarDense, Stage2)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 3
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(OMP_CifarDense, Stage3)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_dense::vulkan::Singleton::getInstance().process_stage_3(*app_data);
  }
}

BENCHMARK_REGISTER_F(OMP_CifarDense, Stage3)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 4
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(OMP_CifarDense, Stage4)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_dense::vulkan::Singleton::getInstance().process_stage_4(*app_data);
  }
}

BENCHMARK_REGISTER_F(OMP_CifarDense, Stage4)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 5
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(OMP_CifarDense, Stage5)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_dense::vulkan::Singleton::getInstance().process_stage_5(*app_data);
  }
}

BENCHMARK_REGISTER_F(OMP_CifarDense, Stage5)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 6
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(OMP_CifarDense, Stage6)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_dense::vulkan::Singleton::getInstance().process_stage_6(*app_data);
  }
}

BENCHMARK_REGISTER_F(OMP_CifarDense, Stage6)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 7
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(OMP_CifarDense, Stage7)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_dense::vulkan::Singleton::getInstance().process_stage_7(*app_data);
  }
}

BENCHMARK_REGISTER_F(OMP_CifarDense, Stage7)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 8
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(OMP_CifarDense, Stage8)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_dense::vulkan::Singleton::getInstance().process_stage_8(*app_data);
  }
}

BENCHMARK_REGISTER_F(OMP_CifarDense, Stage8)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 9
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(OMP_CifarDense, Stage9)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_dense::vulkan::Singleton::getInstance().process_stage_9(*app_data);
  }
}

BENCHMARK_REGISTER_F(OMP_CifarDense, Stage9)->Unit(benchmark::kMillisecond);

int main(int argc, char** argv) {
  parse_args(argc, argv);
  spdlog::set_level(spdlog::level::off);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  return 0;
}