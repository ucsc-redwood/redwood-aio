#include <gtest/gtest.h>
#include <omp.h>

#include <memory_resource>
#include <thread>

#include "builtin-apps/tree/omp/dispatchers.hpp"
#include "verify_tree.hpp"

#define PREPARE_APPDATA                                       \
  const auto n_threads = std::thread::hardware_concurrency(); \
  auto mr = std::pmr::new_delete_resource();                  \
  tree::AppData appdata(mr);                                  \
  tree::omp::TmpStorage tmp_storage;                          \
  tmp_storage.allocate(n_threads, n_threads);

// ----------------------------------------------------------------------------
// Stage 1 Basic Correctness
// ----------------------------------------------------------------------------

TEST(OMP_Tree, Stage1) {
  PREPARE_APPDATA;

#pragma omp parallel
  { tree::omp::run_stage<1>(appdata, tmp_storage); }

  test_tree::verify_stage_1(appdata);
}

// ----------------------------------------------------------------------------
// Stage 2 Basic Correctness
// ----------------------------------------------------------------------------

TEST(OMP_Tree, Stage2) {
  PREPARE_APPDATA;

  tree::omp::run_stage<1>(appdata, tmp_storage);

#pragma omp parallel
  { tree::omp::run_stage<2>(appdata, tmp_storage); }

  test_tree::verify_stage_2(appdata);
}

// ----------------------------------------------------------------------------
// Stage 3 Basic Correctness
// ----------------------------------------------------------------------------

TEST(OMP_Tree, Stage3) {
  PREPARE_APPDATA;

  tree::omp::run_stage<1>(appdata, tmp_storage);
  tree::omp::run_stage<2>(appdata, tmp_storage);

#pragma omp parallel
  { tree::omp::run_stage<3>(appdata, tmp_storage); }

  test_tree::verify_stage_3(appdata);
}

// ----------------------------------------------------------------------------
// Stage 4 Basic Correctness
// ----------------------------------------------------------------------------

TEST(OMP_Tree, Stage4) {
  PREPARE_APPDATA;

  tree::omp::run_stage<1>(appdata, tmp_storage);
  tree::omp::run_stage<2>(appdata, tmp_storage);
  tree::omp::run_stage<3>(appdata, tmp_storage);

#pragma omp parallel
  { tree::omp::run_stage<4>(appdata, tmp_storage); }

  test_tree::verify_stage_4(appdata);
}

// ----------------------------------------------------------------------------
// Stage 5 Basic Correctness
// ----------------------------------------------------------------------------

TEST(OMP_Tree, Stage5) {
  PREPARE_APPDATA;

  tree::omp::run_stage<1>(appdata, tmp_storage);
  tree::omp::run_stage<2>(appdata, tmp_storage);
  tree::omp::run_stage<3>(appdata, tmp_storage);
  tree::omp::run_stage<4>(appdata, tmp_storage);

#pragma omp parallel
  { tree::omp::run_stage<5>(appdata, tmp_storage); }

  test_tree::verify_stage_5(appdata);
}

// ----------------------------------------------------------------------------
// Stage 6 Basic Correctness
// ----------------------------------------------------------------------------

TEST(OMP_Tree, Stage6) {
  PREPARE_APPDATA;

  tree::omp::run_stage<1>(appdata, tmp_storage);
  tree::omp::run_stage<2>(appdata, tmp_storage);
  tree::omp::run_stage<3>(appdata, tmp_storage);
  tree::omp::run_stage<4>(appdata, tmp_storage);
  tree::omp::run_stage<5>(appdata, tmp_storage);

#pragma omp parallel
  { tree::omp::run_stage<6>(appdata, tmp_storage); }

  test_tree::verify_stage_6(appdata);
}

// ----------------------------------------------------------------------------
// Stage 7 Basic Correctness
// ----------------------------------------------------------------------------

TEST(OMP_Tree, Stage7) {
  PREPARE_APPDATA;

  tree::omp::run_stage<1>(appdata, tmp_storage);
  tree::omp::run_stage<2>(appdata, tmp_storage);
  tree::omp::run_stage<3>(appdata, tmp_storage);
  tree::omp::run_stage<4>(appdata, tmp_storage);
  tree::omp::run_stage<5>(appdata, tmp_storage);
  tree::omp::run_stage<6>(appdata, tmp_storage);

#pragma omp parallel
  { tree::omp::run_stage<7>(appdata, tmp_storage); }

  test_tree::verify_stage_7(appdata);
}

// ----------------------------------------------------------------------------
// Stage 1 Multi-threaded / Multi-iteration
// ----------------------------------------------------------------------------

TEST(OMP_Tree, Stage1_MultiIteration) {
  for (int i = 0; i < 10; ++i) {
    PREPARE_APPDATA;

#pragma omp parallel num_threads(n_threads / 2)
    {
      tree::omp::run_stage<1>(appdata, tmp_storage);
      tree::omp::run_stage<2>(appdata, tmp_storage);
      tree::omp::run_stage<3>(appdata, tmp_storage);
      tree::omp::run_stage<4>(appdata, tmp_storage);
      tree::omp::run_stage<5>(appdata, tmp_storage);
      tree::omp::run_stage<6>(appdata, tmp_storage);
      tree::omp::run_stage<7>(appdata, tmp_storage);
    }

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
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
