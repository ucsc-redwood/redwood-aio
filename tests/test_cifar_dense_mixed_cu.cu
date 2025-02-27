#include <gtest/gtest.h>

#include "builtin-apps/affinity.hpp"
#include "builtin-apps/app.hpp"
#include "builtin-apps/cifar-dense/cuda/dispatchers.cuh"
#include "builtin-apps/cifar-dense/omp/dispatchers.hpp"
#include "builtin-apps/common/cuda/cu_mem_resource.cuh"
#include "builtin-apps/common/cuda/helpers.cuh"
#include "spdlog/common.h"

#define PREPARE_DATA                    \
  auto mr = cuda::CudaMemoryResource(); \
  cifar_dense::AppData appdata(&mr);    \
  CUDA_CHECK(cudaDeviceSynchronize());

// ----------------------------------------------------------------------------
// Stages (OMP then CUDA)
// ----------------------------------------------------------------------------

TEST(CUDA_CIFAR_DENSE, Stage1_OMP_Then_CUDA) {
  PREPARE_DATA;

#pragma omp parallel num_threads(g_little_cores.size())
  {
    bind_thread_to_cores(g_little_cores);
    cifar_dense::omp::run_stage<1>(appdata);
  }

  cifar_dense::cuda::run_stage<2>(appdata);
  cifar_dense::cuda::run_stage<3>(appdata);

  CUDA_CHECK(cudaDeviceSynchronize());

  SUCCEED();
}

TEST(CUDA_CIFAR_DENSE, Stage12_OMP_Then_CUDA) {
  PREPARE_DATA;

#pragma omp parallel num_threads(g_little_cores.size())
  {
    bind_thread_to_cores(g_little_cores);
    cifar_dense::omp::run_stage<1>(appdata);
    cifar_dense::omp::run_stage<2>(appdata);
  }

  cifar_dense::cuda::run_stage<3>(appdata);
  cifar_dense::cuda::run_stage<4>(appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

  SUCCEED();
}

TEST(CUDA_CIFAR_DENSE, Stage123_OMP_Then_CUDA) {
  PREPARE_DATA;

#pragma omp parallel num_threads(g_little_cores.size())
  {
    bind_thread_to_cores(g_little_cores);
    cifar_dense::omp::run_stage<1>(appdata);
    cifar_dense::omp::run_stage<2>(appdata);
    cifar_dense::omp::run_stage<3>(appdata);
  }

  cifar_dense::cuda::run_stage<4>(appdata);
  cifar_dense::cuda::run_stage<5>(appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

  SUCCEED();
}

TEST(CUDA_CIFAR_DENSE, Stage1234_OMP_Then_CUDA) {
  PREPARE_DATA;

#pragma omp parallel num_threads(g_little_cores.size())
  {
    bind_thread_to_cores(g_little_cores);
    cifar_dense::omp::run_stage<1>(appdata);
    cifar_dense::omp::run_stage<2>(appdata);
    cifar_dense::omp::run_stage<3>(appdata);
    cifar_dense::omp::run_stage<4>(appdata);
  }

  cifar_dense::cuda::run_stage<5>(appdata);
  cifar_dense::cuda::run_stage<6>(appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

  SUCCEED();
}

// ----------------------------------------------------------------------------
// Stages (CUDA then OMP)
// ----------------------------------------------------------------------------

TEST(CUDA_CIFAR_DENSE, Stage12_CUDA_Then_OMP) {
  PREPARE_DATA;

  cifar_dense::cuda::run_stage<1>(appdata);
  cifar_dense::cuda::run_stage<2>(appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

#pragma omp parallel num_threads(g_little_cores.size())
  {
    bind_thread_to_cores(g_little_cores);
    cifar_dense::omp::run_stage<3>(appdata);
    cifar_dense::omp::run_stage<4>(appdata);
  }

  SUCCEED();
}

TEST(CUDA_CIFAR_DENSE, Stage123_CUDA_Then_OMP) {
  PREPARE_DATA;

  cifar_dense::cuda::run_stage<1>(appdata);
  cifar_dense::cuda::run_stage<2>(appdata);
  cifar_dense::cuda::run_stage<3>(appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

#pragma omp parallel num_threads(g_little_cores.size())
  {
    bind_thread_to_cores(g_little_cores);
    cifar_dense::omp::run_stage<4>(appdata);
    cifar_dense::omp::run_stage<5>(appdata);
  }

  SUCCEED();
}

TEST(CUDA_CIFAR_DENSE, Stage1234_CUDA_Then_OMP) {
  PREPARE_DATA;

  cifar_dense::cuda::run_stage<1>(appdata);
  cifar_dense::cuda::run_stage<2>(appdata);
  cifar_dense::cuda::run_stage<3>(appdata);
  cifar_dense::cuda::run_stage<4>(appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

#pragma omp parallel num_threads(g_little_cores.size())
  {
    bind_thread_to_cores(g_little_cores);
    cifar_dense::omp::run_stage<5>(appdata);
    cifar_dense::omp::run_stage<6>(appdata);
  }

  SUCCEED();
}

int main(int argc, char **argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::debug);

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
