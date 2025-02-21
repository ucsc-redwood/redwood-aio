#include <gtest/gtest.h>

#include "builtin-apps/app.hpp"
#include "builtin-apps/common/cuda/cu_mem_resource.cuh"
#include "builtin-apps/common/cuda/helpers.cuh"
#include "builtin-apps/tree/cuda/dispatchers.cuh"
#include "verify_tree.hpp"

#define PREPARE_DATA                    \
  auto mr = cuda::CudaMemoryResource(); \
  tree::AppData appdata(&mr);           \
  tree::cuda::TempStorage tmp_storage;  \
  CUDA_CHECK(cudaDeviceSynchronize());

// ----------------------------------------------------------------------------
// Stage 1 Basic Correctness
// ----------------------------------------------------------------------------

TEST(CUDA_Tree, Stage1) {
  PREPARE_DATA;

  tree::cuda::run_stage<1>(appdata, tmp_storage);
  CUDA_CHECK(cudaDeviceSynchronize());

  test_tree::verify_stage_1(appdata);
}

// ----------------------------------------------------------------------------
// Stage 1 Basic Correctness
// ----------------------------------------------------------------------------

TEST(CUDA_Tree, Stage2) {
  PREPARE_DATA;

  tree::cuda::run_stage<1>(appdata, tmp_storage);

  tree::cuda::run_stage<2>(appdata, tmp_storage);
  CUDA_CHECK(cudaDeviceSynchronize());

  test_tree::verify_stage_2(appdata);
}

// ----------------------------------------------------------------------------
// Stage 3 Basic Correctness
// ----------------------------------------------------------------------------

TEST(CUDA_Tree, Stage3) {
  PREPARE_DATA;

  tree::cuda::run_stage<1>(appdata, tmp_storage);
  tree::cuda::run_stage<2>(appdata, tmp_storage);

  tree::cuda::run_stage<3>(appdata, tmp_storage);
  CUDA_CHECK(cudaDeviceSynchronize());

  test_tree::verify_stage_3(appdata);
}

// ----------------------------------------------------------------------------
// Stage 4 Basic Correctness
// ----------------------------------------------------------------------------

TEST(CUDA_Tree, Stage4) {
  PREPARE_DATA;

  tree::cuda::run_stage<1>(appdata, tmp_storage);
  tree::cuda::run_stage<2>(appdata, tmp_storage);
  tree::cuda::run_stage<3>(appdata, tmp_storage);

  tree::cuda::run_stage<4>(appdata, tmp_storage);
  CUDA_CHECK(cudaDeviceSynchronize());

  test_tree::verify_stage_4(appdata);
}

// ----------------------------------------------------------------------------
// Stage 5 Basic Correctness
// ----------------------------------------------------------------------------

TEST(CUDA_Tree, Stage5) {
  PREPARE_DATA;

  tree::cuda::run_stage<1>(appdata, tmp_storage);
  tree::cuda::run_stage<2>(appdata, tmp_storage);
  tree::cuda::run_stage<3>(appdata, tmp_storage);
  tree::cuda::run_stage<4>(appdata, tmp_storage);

  tree::cuda::run_stage<5>(appdata, tmp_storage);
  CUDA_CHECK(cudaDeviceSynchronize());

  test_tree::verify_stage_5(appdata);
}

// ----------------------------------------------------------------------------
// Stage 6 Basic Correctness
// ----------------------------------------------------------------------------

TEST(CUDA_Tree, Stage6) {
  PREPARE_DATA;

  tree::cuda::run_stage<1>(appdata, tmp_storage);
  tree::cuda::run_stage<2>(appdata, tmp_storage);
  tree::cuda::run_stage<3>(appdata, tmp_storage);
  tree::cuda::run_stage<4>(appdata, tmp_storage);
  tree::cuda::run_stage<5>(appdata, tmp_storage);

  tree::cuda::run_stage<6>(appdata, tmp_storage);
  CUDA_CHECK(cudaDeviceSynchronize());

  test_tree::verify_stage_6(appdata);
}

// ----------------------------------------------------------------------------
// Stage 7 Basic Correctness
// ----------------------------------------------------------------------------

TEST(CUDA_Tree, Stage7) {
  PREPARE_DATA;

  tree::cuda::run_stage<1>(appdata, tmp_storage);
  tree::cuda::run_stage<2>(appdata, tmp_storage);
  tree::cuda::run_stage<3>(appdata, tmp_storage);
  tree::cuda::run_stage<4>(appdata, tmp_storage);
  tree::cuda::run_stage<5>(appdata, tmp_storage);
  tree::cuda::run_stage<6>(appdata, tmp_storage);

  tree::cuda::run_stage<7>(appdata, tmp_storage);
  CUDA_CHECK(cudaDeviceSynchronize());

  test_tree::verify_stage_7(appdata);
}

// ----------------------------------------------------------------------------
// All Stages Multi-threaded / Multi-iteration
// ----------------------------------------------------------------------------

TEST(CUDA_Tree, AllStages_MultiIteration) {
  for (int i = 0; i < 10; ++i) {
    PREPARE_DATA;

    tree::cuda::run_stage<1>(appdata, tmp_storage);
    tree::cuda::run_stage<2>(appdata, tmp_storage);
    tree::cuda::run_stage<3>(appdata, tmp_storage);
    tree::cuda::run_stage<4>(appdata, tmp_storage);
    tree::cuda::run_stage<5>(appdata, tmp_storage);
    tree::cuda::run_stage<6>(appdata, tmp_storage);
    tree::cuda::run_stage<7>(appdata, tmp_storage);
    CUDA_CHECK(cudaDeviceSynchronize());

    test_tree::verify_stage_1(appdata);
    test_tree::verify_stage_2(appdata);
    test_tree::verify_stage_3(appdata);
    test_tree::verify_stage_4(appdata);
    test_tree::verify_stage_5(appdata);
    test_tree::verify_stage_6(appdata);
    test_tree::verify_stage_7(appdata);
  }
}

// ----------------------------------------------------------------------------
// Main function for running tests
// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::off);

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
