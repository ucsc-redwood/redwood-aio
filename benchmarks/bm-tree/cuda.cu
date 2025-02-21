#include <benchmark/benchmark.h>

#include "../argc_argv_sanitizer.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/common/cuda/cu_mem_resource.cuh"
#include "builtin-apps/common/cuda/helpers.cuh"
#include "builtin-apps/resources_path.hpp"
#include "builtin-apps/tree/cuda/dispatchers.cuh"

#define PREPARE_DATA                    \
  auto mr = cuda::CudaMemoryResource(); \
  tree::AppData appdata(&mr);           \
  tree::cuda::TempStorage tmp_storage;  \
  CUDA_CHECK(cudaDeviceSynchronize());

// ----------------------------------------------------------------
// Baseline
// ----------------------------------------------------------------

class CUDA_Tree : public benchmark::Fixture {};

BENCHMARK_DEFINE_F(CUDA_Tree, Baseline)
(benchmark::State& state) {
  PREPARE_DATA;

  for (auto _ : state) {
    tree::cuda::run_stage<1>(appdata, tmp_storage);
    tree::cuda::run_stage<2>(appdata, tmp_storage);
    tree::cuda::run_stage<3>(appdata, tmp_storage);
    tree::cuda::run_stage<4>(appdata, tmp_storage);
    tree::cuda::run_stage<5>(appdata, tmp_storage);
    tree::cuda::run_stage<6>(appdata, tmp_storage);
    tree::cuda::run_stage<7>(appdata, tmp_storage);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_Tree, Baseline)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 1
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_Tree, Stage1)
(benchmark::State& state) {
  PREPARE_DATA;

  // warmup
  tree::cuda::run_stage<1>(appdata, tmp_storage);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    tree::cuda::run_stage<1>(appdata, tmp_storage);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_Tree, Stage1)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 2
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_Tree, Stage2)
(benchmark::State& state) {
  PREPARE_DATA;

  tree::cuda::run_stage<1>(appdata, tmp_storage);
  CUDA_CHECK(cudaDeviceSynchronize());

  // warmup
  tree::cuda::run_stage<2>(appdata, tmp_storage);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    tree::cuda::run_stage<2>(appdata, tmp_storage);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_Tree, Stage2)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 3
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_Tree, Stage3)
(benchmark::State& state) {
  PREPARE_DATA;

  tree::cuda::run_stage<1>(appdata, tmp_storage);
  tree::cuda::run_stage<2>(appdata, tmp_storage);
  CUDA_CHECK(cudaDeviceSynchronize());

  // warmup
  tree::cuda::run_stage<3>(appdata, tmp_storage);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    tree::cuda::run_stage<3>(appdata, tmp_storage);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_Tree, Stage3)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 4
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_Tree, Stage4)
(benchmark::State& state) {
  PREPARE_DATA;

  tree::cuda::run_stage<1>(appdata, tmp_storage);
  tree::cuda::run_stage<2>(appdata, tmp_storage);
  tree::cuda::run_stage<3>(appdata, tmp_storage);
  CUDA_CHECK(cudaDeviceSynchronize());

  // warmup
  tree::cuda::run_stage<4>(appdata, tmp_storage);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    tree::cuda::run_stage<4>(appdata, tmp_storage);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_Tree, Stage4)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 5
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_Tree, Stage5)
(benchmark::State& state) {
  PREPARE_DATA;

  tree::cuda::run_stage<1>(appdata, tmp_storage);
  tree::cuda::run_stage<2>(appdata, tmp_storage);
  tree::cuda::run_stage<3>(appdata, tmp_storage);
  tree::cuda::run_stage<4>(appdata, tmp_storage);
  CUDA_CHECK(cudaDeviceSynchronize());

  // warmup
  tree::cuda::run_stage<5>(appdata, tmp_storage);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    tree::cuda::run_stage<5>(appdata, tmp_storage);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_Tree, Stage5)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 6
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_Tree, Stage6)
(benchmark::State& state) {
  PREPARE_DATA;

  tree::cuda::run_stage<1>(appdata, tmp_storage);
  tree::cuda::run_stage<2>(appdata, tmp_storage);
  tree::cuda::run_stage<3>(appdata, tmp_storage);
  tree::cuda::run_stage<4>(appdata, tmp_storage);
  tree::cuda::run_stage<5>(appdata, tmp_storage);
  CUDA_CHECK(cudaDeviceSynchronize());

  // warmup
  tree::cuda::run_stage<6>(appdata, tmp_storage);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    tree::cuda::run_stage<6>(appdata, tmp_storage);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_Tree, Stage6)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 7
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_Tree, Stage7)
(benchmark::State& state) {
  PREPARE_DATA;

  tree::cuda::run_stage<1>(appdata, tmp_storage);
  tree::cuda::run_stage<2>(appdata, tmp_storage);
  tree::cuda::run_stage<3>(appdata, tmp_storage);
  tree::cuda::run_stage<4>(appdata, tmp_storage);
  tree::cuda::run_stage<5>(appdata, tmp_storage);
  tree::cuda::run_stage<6>(appdata, tmp_storage);
  CUDA_CHECK(cudaDeviceSynchronize());

  // warmup
  tree::cuda::run_stage<7>(appdata, tmp_storage);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    tree::cuda::run_stage<7>(appdata, tmp_storage);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_Tree, Stage7)->Unit(benchmark::kMillisecond);

int main(int argc, char** argv) {
  parse_args(argc, argv);
  spdlog::set_level(spdlog::level::off);

  // Where to save the results json file?
  const auto storage_location = helpers::get_benchmark_storage_location();
  const auto out_name = storage_location.string() + "/BM_Tree_CUDA_" + g_device_id + ".json";

  // Sanitize the arguments to pass to Google Benchmark
  auto [new_argc, new_argv] = sanitize_argc_argv_for_benchmark(argc, argv, out_name);

  benchmark::Initialize(&new_argc, new_argv.data());
  if (benchmark::ReportUnrecognizedArguments(new_argc, new_argv.data())) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}