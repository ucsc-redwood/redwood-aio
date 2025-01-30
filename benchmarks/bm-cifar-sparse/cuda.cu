#include <benchmark/benchmark.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>

#include "cifar-sparse/cuda/cu_dispatcher.cuh"
#include "cifar-sparse/sparse_appdata.hpp"
#include "common/cuda/cu_mem_resource.cuh"
#include "common/cuda/helpers.cuh"

// ----------------------------------------------------------------
// Baseline
// ----------------------------------------------------------------

class CUDA_CifarSparse : public benchmark::Fixture {};

static void run_baseline(cifar_sparse::AppData& app_data) {
  cifar_sparse::cuda::process_stage_1(app_data);
  cifar_sparse::cuda::process_stage_2(app_data);
  cifar_sparse::cuda::process_stage_3(app_data);
  cifar_sparse::cuda::process_stage_4(app_data);
  cifar_sparse::cuda::process_stage_5(app_data);
  cifar_sparse::cuda::process_stage_6(app_data);
  cifar_sparse::cuda::process_stage_7(app_data);
  cifar_sparse::cuda::process_stage_8(app_data);
  cifar_sparse::cuda::process_stage_9(app_data);
  CUDA_CHECK(cudaDeviceSynchronize());
}

// static void CUDA_Baseline_Benchmark(benchmark::State& state) {
//   auto mr = cuda::CudaMemoryResource();
//   cifar_sparse::AppData app_data(&mr);

//   for (auto _ : state) {
//     run_baseline(app_data);
//   }
// }

// BENCHMARK(CUDA_Baseline_Benchmark)
//     ->Unit(benchmark::kMillisecond)
//     ->Iterations(100);

BENCHMARK_DEFINE_F(CUDA_CifarSparse, Baseline)
(benchmark::State& state) {
  auto mr = cuda::CudaMemoryResource();
  cifar_sparse::AppData app_data(&mr);
  for (auto _ : state) {
    run_baseline(app_data);
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarSparse, Baseline)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// ----------------------------------------------------------------
// Individual stages
// ----------------------------------------------------------------

// ----------------------------------------------------------------
// Stage 1
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarSparse, Stage1)
(benchmark::State& state) {
  auto mr = cuda::CudaMemoryResource();
  cifar_sparse::AppData app_data(&mr);

  for (auto _ : state) {
    cifar_sparse::cuda::process_stage_1(app_data);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarSparse, Stage1)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

// ----------------------------------------------------------------
// Stage 2
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarSparse, Stage2)
(benchmark::State& state) {
  auto mr = cuda::CudaMemoryResource();
  cifar_sparse::AppData app_data(&mr);

  cifar_sparse::cuda::process_stage_1(app_data);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    cifar_sparse::cuda::process_stage_2(app_data);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarSparse, Stage2)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

// ----------------------------------------------------------------
// Stage 3
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarSparse, Stage3)
(benchmark::State& state) {
  auto mr = cuda::CudaMemoryResource();
  cifar_sparse::AppData app_data(&mr);

  cifar_sparse::cuda::process_stage_1(app_data);
  cifar_sparse::cuda::process_stage_2(app_data);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    cifar_sparse::cuda::process_stage_3(app_data);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarSparse, Stage3)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

// ----------------------------------------------------------------
// Stage 4
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarSparse, Stage4)
(benchmark::State& state) {
  auto mr = cuda::CudaMemoryResource();
  cifar_sparse::AppData app_data(&mr);

  cifar_sparse::cuda::process_stage_1(app_data);
  cifar_sparse::cuda::process_stage_2(app_data);
  cifar_sparse::cuda::process_stage_3(app_data);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    cifar_sparse::cuda::process_stage_4(app_data);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarSparse, Stage4)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

// ----------------------------------------------------------------
// Stage 5
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarSparse, Stage5)
(benchmark::State& state) {
  auto mr = cuda::CudaMemoryResource();
  cifar_sparse::AppData app_data(&mr);

  cifar_sparse::cuda::process_stage_1(app_data);
  cifar_sparse::cuda::process_stage_2(app_data);
  cifar_sparse::cuda::process_stage_3(app_data);
  cifar_sparse::cuda::process_stage_4(app_data);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    cifar_sparse::cuda::process_stage_5(app_data);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarSparse, Stage5)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

// ----------------------------------------------------------------
// Stage 6
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarSparse, Stage6)
(benchmark::State& state) {
  auto mr = cuda::CudaMemoryResource();
  cifar_sparse::AppData app_data(&mr);

  cifar_sparse::cuda::process_stage_1(app_data);
  cifar_sparse::cuda::process_stage_2(app_data);
  cifar_sparse::cuda::process_stage_3(app_data);
  cifar_sparse::cuda::process_stage_4(app_data);
  cifar_sparse::cuda::process_stage_5(app_data);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    cifar_sparse::cuda::process_stage_6(app_data);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarSparse, Stage6)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

// ----------------------------------------------------------------
// Stage 7
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarSparse, Stage7)
(benchmark::State& state) {
  auto mr = cuda::CudaMemoryResource();
  cifar_sparse::AppData app_data(&mr);

  cifar_sparse::cuda::process_stage_1(app_data);
  cifar_sparse::cuda::process_stage_2(app_data);
  cifar_sparse::cuda::process_stage_3(app_data);
  cifar_sparse::cuda::process_stage_4(app_data);
  cifar_sparse::cuda::process_stage_5(app_data);
  cifar_sparse::cuda::process_stage_6(app_data);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    cifar_sparse::cuda::process_stage_7(app_data);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarSparse, Stage7)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

// ----------------------------------------------------------------
// Stage 8
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarSparse, Stage8)
(benchmark::State& state) {
  auto mr = cuda::CudaMemoryResource();
  cifar_sparse::AppData app_data(&mr);

  cifar_sparse::cuda::process_stage_1(app_data);
  cifar_sparse::cuda::process_stage_2(app_data);
  cifar_sparse::cuda::process_stage_3(app_data);
  cifar_sparse::cuda::process_stage_4(app_data);
  cifar_sparse::cuda::process_stage_5(app_data);
  cifar_sparse::cuda::process_stage_6(app_data);
  cifar_sparse::cuda::process_stage_7(app_data);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    cifar_sparse::cuda::process_stage_8(app_data);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarSparse, Stage8)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

// ----------------------------------------------------------------
// Stage 9
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarSparse, Stage9)
(benchmark::State& state) {
  auto mr = cuda::CudaMemoryResource();
  cifar_sparse::AppData app_data(&mr);

  cifar_sparse::cuda::process_stage_1(app_data);
  cifar_sparse::cuda::process_stage_2(app_data);
  cifar_sparse::cuda::process_stage_3(app_data);
  cifar_sparse::cuda::process_stage_4(app_data);
  cifar_sparse::cuda::process_stage_5(app_data);
  cifar_sparse::cuda::process_stage_6(app_data);
  cifar_sparse::cuda::process_stage_7(app_data);
  cifar_sparse::cuda::process_stage_8(app_data);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    cifar_sparse::cuda::process_stage_9(app_data);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarSparse, Stage9)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

int main(int argc, char** argv) {
  spdlog::set_level(spdlog::level::off);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  return 0;
}