#include <benchmark/benchmark.h>

#include "../argc_argv_sanitizer.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/cifar-dense/cuda/cu_kernels.cuh"
#include "builtin-apps/cifar-dense/dense_appdata.hpp"
#include "builtin-apps/common/cuda/cu_mem_resource.cuh"
#include "builtin-apps/common/cuda/helpers.cuh"
#include "builtin-apps/resources_path.hpp"

#define PREPARE_DATA                    \
  auto mr = cuda::CudaMemoryResource(); \
  cifar_dense::AppData appdata(&mr);    \
  CUDA_CHECK(cudaDeviceSynchronize());

// ----------------------------------------------------------------
// Baseline
// ----------------------------------------------------------------

class CUDA_CifarDense : public benchmark::Fixture {};

BENCHMARK_DEFINE_F(CUDA_CifarDense, Baseline)
(benchmark::State& state) {
  PREPARE_DATA;

  for (auto _ : state) {
    cifar_dense::cuda::run_stage<1>(appdata);
    cifar_dense::cuda::run_stage<2>(appdata);
    cifar_dense::cuda::run_stage<3>(appdata);
    cifar_dense::cuda::run_stage<4>(appdata);
    cifar_dense::cuda::run_stage<5>(appdata);
    cifar_dense::cuda::run_stage<6>(appdata);
    cifar_dense::cuda::run_stage<7>(appdata);
    cifar_dense::cuda::run_stage<8>(appdata);
    cifar_dense::cuda::run_stage<9>(appdata);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarDense, Baseline)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 1
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarDense, Stage1)
(benchmark::State& state) {
  PREPARE_DATA;

  // warmup
  cifar_dense::cuda::run_stage<1>(appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    cifar_dense::cuda::run_stage<1>(appdata);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarDense, Stage1)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 2
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarDense, Stage2)
(benchmark::State& state) {
  PREPARE_DATA;

  cifar_dense::cuda::run_stage<1>(appdata);
  // warmup
  cifar_dense::cuda::run_stage<2>(appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    cifar_dense::cuda::run_stage<2>(appdata);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarDense, Stage2)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 3
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarDense, Stage3)
(benchmark::State& state) {
  PREPARE_DATA;

  cifar_dense::cuda::run_stage<1>(appdata);
  cifar_dense::cuda::run_stage<2>(appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

  // warmup
  cifar_dense::cuda::run_stage<3>(appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    cifar_dense::cuda::run_stage<3>(appdata);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarDense, Stage3)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 4
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarDense, Stage4)
(benchmark::State& state) {
  PREPARE_DATA;

  cifar_dense::cuda::run_stage<1>(appdata);
  cifar_dense::cuda::run_stage<2>(appdata);
  cifar_dense::cuda::run_stage<3>(appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

  cifar_dense::cuda::run_stage<4>(appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    cifar_dense::cuda::run_stage<4>(appdata);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarDense, Stage4)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 5
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarDense, Stage5)
(benchmark::State& state) {
  PREPARE_DATA;

  cifar_dense::cuda::run_stage<1>(appdata);
  cifar_dense::cuda::run_stage<2>(appdata);
  cifar_dense::cuda::run_stage<3>(appdata);
  cifar_dense::cuda::run_stage<4>(appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

  // warmup
  cifar_dense::cuda::run_stage<5>(appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    cifar_dense::cuda::run_stage<5>(appdata);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarDense, Stage5)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 6
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarDense, Stage6)
(benchmark::State& state) {
  PREPARE_DATA;

  cifar_dense::cuda::run_stage<1>(appdata);
  cifar_dense::cuda::run_stage<2>(appdata);
  cifar_dense::cuda::run_stage<3>(appdata);
  cifar_dense::cuda::run_stage<4>(appdata);
  cifar_dense::cuda::run_stage<5>(appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

  // warmup
  cifar_dense::cuda::run_stage<6>(appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    cifar_dense::cuda::run_stage<6>(appdata);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarDense, Stage6)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 7
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarDense, Stage7)
(benchmark::State& state) {
  PREPARE_DATA;

  cifar_dense::cuda::run_stage<1>(appdata);
  cifar_dense::cuda::run_stage<2>(appdata);
  cifar_dense::cuda::run_stage<3>(appdata);
  cifar_dense::cuda::run_stage<4>(appdata);
  cifar_dense::cuda::run_stage<5>(appdata);
  cifar_dense::cuda::run_stage<6>(appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

  // warmup
  cifar_dense::cuda::run_stage<7>(appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    cifar_dense::cuda::run_stage<7>(appdata);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarDense, Stage7)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 8
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarDense, Stage8)
(benchmark::State& state) {
  PREPARE_DATA;

  cifar_dense::cuda::run_stage<1>(appdata);
  cifar_dense::cuda::run_stage<2>(appdata);
  cifar_dense::cuda::run_stage<3>(appdata);
  cifar_dense::cuda::run_stage<4>(appdata);
  cifar_dense::cuda::run_stage<5>(appdata);
  cifar_dense::cuda::run_stage<6>(appdata);
  cifar_dense::cuda::run_stage<7>(appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

  // warmup
  cifar_dense::cuda::run_stage<8>(appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    cifar_dense::cuda::run_stage<8>(appdata);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarDense, Stage8)->Unit(benchmark::kMillisecond);

// ----------------------------------------------------------------
// Stage 9
// ----------------------------------------------------------------

BENCHMARK_DEFINE_F(CUDA_CifarDense, Stage9)
(benchmark::State& state) {
  PREPARE_DATA;

  cifar_dense::cuda::run_stage<1>(appdata);
  cifar_dense::cuda::run_stage<2>(appdata);
  cifar_dense::cuda::run_stage<3>(appdata);
  cifar_dense::cuda::run_stage<4>(appdata);
  cifar_dense::cuda::run_stage<5>(appdata);
  cifar_dense::cuda::run_stage<6>(appdata);
  cifar_dense::cuda::run_stage<7>(appdata);
  cifar_dense::cuda::run_stage<8>(appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

  // warmup
  cifar_dense::cuda::run_stage<9>(appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : state) {
    cifar_dense::cuda::run_stage<9>(appdata);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

BENCHMARK_REGISTER_F(CUDA_CifarDense, Stage9)->Unit(benchmark::kMillisecond);

int main(int argc, char** argv) {
  parse_args(argc, argv);
  spdlog::set_level(spdlog::level::off);

  // Where to save the results json file?
  const auto storage_location = helpers::get_benchmark_storage_location();
  const auto out_name = storage_location.string() + "/BM_CifarDense_CUDA_" + g_device_id + ".json";

  // Sanitize the arguments to pass to Google Benchmark
  auto [new_argc, new_argv] = sanitize_argc_argv_for_benchmark(argc, argv, out_name);

  benchmark::Initialize(&new_argc, new_argv.data());
  if (benchmark::ReportUnrecognizedArguments(new_argc, new_argv.data())) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();

  return 0;
}