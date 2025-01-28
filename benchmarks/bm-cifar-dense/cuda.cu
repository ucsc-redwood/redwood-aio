#include <benchmark/benchmark.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>

#include "app.hpp"
#include "cifar-dense/cuda/cu_dense_kernel.cuh"
#include "cifar-dense/dense_appdata.hpp"
#include "common/cuda/cu_mem_resource.cuh"
#include "common/cuda/helpers.cuh"

// ----------------------------------------------------------------
// Baseline
// ----------------------------------------------------------------

static void run_baseline(cifar_dense::AppData& app_data) {
  cifar_dense::cuda::process_stage_1(app_data);
  cifar_dense::cuda::process_stage_2(app_data);
  cifar_dense::cuda::process_stage_3(app_data);
  cifar_dense::cuda::process_stage_4(app_data);
  cifar_dense::cuda::process_stage_5(app_data);
  cifar_dense::cuda::process_stage_6(app_data);
  cifar_dense::cuda::process_stage_7(app_data);
  cifar_dense::cuda::process_stage_8(app_data);
  cifar_dense::cuda::process_stage_9(app_data);
  CUDA_CHECK(cudaDeviceSynchronize());
}

static void CUDA_Baseline_Benchmark(benchmark::State& state) {
  auto mr = cuda::CudaMemoryResource();
  cifar_dense::AppData app_data(&mr);

  for (auto _ : state) {
    run_baseline(app_data);
  }
}

BENCHMARK(CUDA_Baseline_Benchmark)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

// ----------------------------------------------------------------
// Individual stages
// ----------------------------------------------------------------

class CUDA_CifarDense : public benchmark::Fixture {
 protected:
  void SetUp(benchmark::State&) override {
    auto mr = cuda::CudaMemoryResource();

    app_data = std::make_unique<cifar_dense::AppData>(&mr);

    // process the data once
    run_baseline(*app_data);
  }

  void TearDown(benchmark::State&) override { app_data.reset(); }

  std::unique_ptr<cifar_dense::AppData> app_data;
};

// ----------------------------------------------------------------
// Stage 1
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarDense, Stage1)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_dense::cuda::process_stage_1(*app_data);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarDense, Stage1)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

// ----------------------------------------------------------------
// Stage 2
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarDense, Stage2)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_dense::cuda::process_stage_2(*app_data);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarDense, Stage2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

// ----------------------------------------------------------------
// Stage 3
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarDense, Stage3)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_dense::cuda::process_stage_3(*app_data);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarDense, Stage3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

// ----------------------------------------------------------------
// Stage 4
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarDense, Stage4)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_dense::cuda::process_stage_4(*app_data);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarDense, Stage4)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

// ----------------------------------------------------------------
// Stage 5
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarDense, Stage5)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_dense::cuda::process_stage_5(*app_data);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarDense, Stage5)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

// ----------------------------------------------------------------
// Stage 6
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarDense, Stage6)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_dense::cuda::process_stage_6(*app_data);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarDense, Stage6)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

// ----------------------------------------------------------------
// Stage 7
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarDense, Stage7)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_dense::cuda::process_stage_7(*app_data);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarDense, Stage7)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

// ----------------------------------------------------------------
// Stage 8
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarDense, Stage8)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_dense::cuda::process_stage_8(*app_data);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarDense, Stage8)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

// ----------------------------------------------------------------
// Stage 9
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarDense, Stage9)
(benchmark::State& state) {
  for (auto _ : state) {
    cifar_dense::cuda::process_stage_9(*app_data);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarDense, Stage9)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

int main(int argc, char** argv) {
  parse_args(argc, argv);
  spdlog::set_level(spdlog::level::off);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  return 0;
}