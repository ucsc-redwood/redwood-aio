#include <gtest/gtest.h>

#include <memory_resource>

#include "builtin-apps/tree/omp/tree_kernel.hpp"
#include "tests/verify_tree.hpp"

// ----------------------------------------------------------------------------
// Stage 1
// ----------------------------------------------------------------------------

TEST(OMP_Tree, Stage1) {
  auto mr = std::pmr::new_delete_resource();
  tree::AppData appdata(mr);

#pragma omp parallel
  { tree::omp::run_stage<1>(appdata); }

  test_tree::verify_stage_1(appdata);
}

// ----------------------------------------------------------------------------
// Stage 2
// ----------------------------------------------------------------------------

TEST(OMP_Tree, Stage2) {
  auto mr = std::pmr::new_delete_resource();
  tree::AppData appdata(mr);

  tree::omp::run_stage<1>(appdata);

#pragma omp parallel
  { tree::omp::run_stage<2>(appdata); }

  test_tree::verify_stage_2(appdata);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
