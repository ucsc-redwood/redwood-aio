#include <benchmark/benchmark.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>

#include "common/cuda/cu_mem_resource.cuh"
#include "common/cuda/helpers.cuh"
#include "tree/cuda/kernel.cuh"
#include "tree/tree_appdata.hpp"

// ----------------------------------------------------------------
// Baseline
// ----------------------------------------------------------------

class CUDA_Tree : public benchmark::Fixture {};

BENCHMARK_DEFINE_F(CUDA_Tree, Baseline)
(benchmark::State& state) {
  auto mr = cuda::CudaMemoryResource();
  tree::AppData app_data(&mr);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    tree::cuda::process_stage_1(app_data);
    tree::cuda::process_stage_2(app_data);
    tree::cuda::process_stage_3(app_data);
    tree::cuda::process_stage_4(app_data);
    tree::cuda::process_stage_5(app_data);
    tree::cuda::process_stage_6(app_data);
    tree::cuda::process_stage_7(app_data);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_Tree, Baseline)->Unit(benchmark::kMillisecond)->Iterations(10);

// ----------------------------------------------------------------
// Stage 1
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_Tree, Stage1)
(benchmark::State& state) {
  auto mr = cuda::CudaMemoryResource();
  tree::AppData app_data(&mr);

  // warmup
  tree::cuda::process_stage_1(app_data);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    tree::cuda::process_stage_1(app_data);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_Tree, Stage1)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 2
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_Tree, Stage2)
(benchmark::State& state) {
  auto mr = cuda::CudaMemoryResource();
  tree::AppData app_data(&mr);

  tree::cuda::process_stage_1(app_data);
  CUDA_CHECK(cudaDeviceSynchronize());

  // warmup
  tree::cuda::process_stage_2(app_data);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    tree::cuda::process_stage_2(app_data);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_Tree, Stage2)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 3
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_Tree, Stage3)
(benchmark::State& state) {
  auto mr = cuda::CudaMemoryResource();
  tree::AppData app_data(&mr);

  tree::cuda::process_stage_1(app_data);
  tree::cuda::process_stage_2(app_data);
  CUDA_CHECK(cudaDeviceSynchronize());

  // warmup
  tree::cuda::process_stage_3(app_data);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    tree::cuda::process_stage_3(app_data);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_Tree, Stage3)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 4
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_Tree, Stage4)
(benchmark::State& state) {
  auto mr = cuda::CudaMemoryResource();
  tree::AppData app_data(&mr);

  tree::cuda::process_stage_1(app_data);
  tree::cuda::process_stage_2(app_data);
  tree::cuda::process_stage_3(app_data);
  CUDA_CHECK(cudaDeviceSynchronize());

  // warmup
  tree::cuda::process_stage_4(app_data);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    tree::cuda::process_stage_4(app_data);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_Tree, Stage4)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 5
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_Tree, Stage5)
(benchmark::State& state) {
  auto mr = cuda::CudaMemoryResource();
  tree::AppData app_data(&mr);

  tree::cuda::process_stage_1(app_data);
  tree::cuda::process_stage_2(app_data);
  tree::cuda::process_stage_3(app_data);
  tree::cuda::process_stage_4(app_data);
  CUDA_CHECK(cudaDeviceSynchronize());

  // warmup
  tree::cuda::process_stage_5(app_data);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    tree::cuda::process_stage_5(app_data);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_Tree, Stage5)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 6
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_Tree, Stage6)
(benchmark::State& state) {
  auto mr = cuda::CudaMemoryResource();
  tree::AppData app_data(&mr);

  tree::cuda::process_stage_1(app_data);
  tree::cuda::process_stage_2(app_data);
  tree::cuda::process_stage_3(app_data);
  tree::cuda::process_stage_4(app_data);
  tree::cuda::process_stage_5(app_data);
  CUDA_CHECK(cudaDeviceSynchronize());

  // warmup
  tree::cuda::process_stage_6(app_data);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    tree::cuda::process_stage_6(app_data);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_Tree, Stage6)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 7
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_Tree, Stage7)
(benchmark::State& state) {
  auto mr = cuda::CudaMemoryResource();
  tree::AppData app_data(&mr);

  tree::cuda::process_stage_1(app_data);
  tree::cuda::process_stage_2(app_data);
  tree::cuda::process_stage_3(app_data);
  tree::cuda::process_stage_4(app_data);
  tree::cuda::process_stage_5(app_data);
  tree::cuda::process_stage_6(app_data);
  CUDA_CHECK(cudaDeviceSynchronize());

  // warmup
  tree::cuda::process_stage_7(app_data);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    tree::cuda::process_stage_7(app_data);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_Tree, Stage7)->Unit(benchmark::kMillisecond)->Iterations(10);

int main(int argc, char** argv) {
  spdlog::set_level(spdlog::level::off);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  tree::cuda::cleanup();

  return 0;
}