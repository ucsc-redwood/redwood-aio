#include <gtest/gtest.h>
#include <omp.h>

#include "builtin-apps/app.hpp"
// #include "builtin-apps/tree/omp/tree_kernel.hpp"
#include "builtin-apps/tree/vulkan/vk_dispatcher.hpp"
#include "tests/verify_tree.hpp"

#define PREPARE_APPDATA                                      \
  auto mr = tree::vulkan::Singleton::getInstance().get_mr(); \
  tree::AppData appdata(mr);                                 \
  tree::vulkan::TmpStorage tmp_storage(mr, appdata.get_n_input());

// ----------------------------------------------------------------------------
// Stage 1 Basic Correctness
// ----------------------------------------------------------------------------

TEST(Vulkan_Tree, Stage1) {
  PREPARE_APPDATA;

  tree::vulkan::Singleton::getInstance().process_stage_1(appdata, tmp_storage);

  // print first 10 morton codes
  for (int i = 0; i < 10; ++i) {
    std::cout << "morton[" << i << "] = " << appdata.u_morton_keys_s1[i] << std::endl;
  }

  // check if all morton codes are non-zero
  for (int i = 0; i < appdata.get_n_input(); ++i) {
    std::cout << "morton[" << i << "] = " << appdata.u_morton_keys_s1[i] << std::endl;
    EXPECT_NE(appdata.u_morton_keys_s1[i], 0);
  }

  test_tree::verify_stage_1(appdata);
}

// ----------------------------------------------------------------------------
// Stage 2 Basic Correctness
// ----------------------------------------------------------------------------

TEST(Vulkan_Tree, Stage2) {
  PREPARE_APPDATA;

  tree::vulkan::Singleton::getInstance().process_stage_1(appdata, tmp_storage);
  tree::vulkan::Singleton::getInstance().process_stage_2(appdata, tmp_storage);

  test_tree::verify_stage_2(appdata);
}

// ----------------------------------------------------------------------------
// Stage 3 Basic Correctness
// ----------------------------------------------------------------------------

TEST(Vulkan_Tree, Stage3) {
  PREPARE_APPDATA;

  tree::vulkan::Singleton::getInstance().process_stage_1(appdata, tmp_storage);
  tree::vulkan::Singleton::getInstance().process_stage_2(appdata, tmp_storage);
  tree::vulkan::Singleton::getInstance().process_stage_3(appdata, tmp_storage);

  test_tree::verify_stage_3(appdata);
}

// ----------------------------------------------------------------------------
// Stage 4 Basic Correctness
// ----------------------------------------------------------------------------

TEST(Vulkan_Tree, Stage4) {
  PREPARE_APPDATA;

  tree::vulkan::Singleton::getInstance().process_stage_1(appdata, tmp_storage);
  tree::vulkan::Singleton::getInstance().process_stage_2(appdata, tmp_storage);
  tree::vulkan::Singleton::getInstance().process_stage_3(appdata, tmp_storage);
  tree::vulkan::Singleton::getInstance().process_stage_4(appdata, tmp_storage);

  test_tree::verify_stage_4(appdata);
}

// ----------------------------------------------------------------------------
// Stage 5 Basic Correctness
// ----------------------------------------------------------------------------

TEST(Vulkan_Tree, Stage5) {
  PREPARE_APPDATA;

  tree::vulkan::Singleton::getInstance().process_stage_1(appdata, tmp_storage);
  tree::vulkan::Singleton::getInstance().process_stage_2(appdata, tmp_storage);
  tree::vulkan::Singleton::getInstance().process_stage_3(appdata, tmp_storage);
  tree::vulkan::Singleton::getInstance().process_stage_4(appdata, tmp_storage);
  tree::vulkan::Singleton::getInstance().process_stage_5(appdata, tmp_storage);

  test_tree::verify_stage_5(appdata);
}

// ----------------------------------------------------------------------------
// Stage 6 Basic Correctness
// ----------------------------------------------------------------------------

TEST(Vulkan_Tree, Stage6) {
  PREPARE_APPDATA;

  tree::vulkan::Singleton::getInstance().process_stage_1(appdata, tmp_storage);
  tree::vulkan::Singleton::getInstance().process_stage_2(appdata, tmp_storage);
  tree::vulkan::Singleton::getInstance().process_stage_3(appdata, tmp_storage);
  tree::vulkan::Singleton::getInstance().process_stage_4(appdata, tmp_storage);
  tree::vulkan::Singleton::getInstance().process_stage_5(appdata, tmp_storage);
  tree::vulkan::Singleton::getInstance().process_stage_6(appdata, tmp_storage);

  test_tree::verify_stage_6(appdata);
}
// ----------------------------------------------------------------------------
// Stage 7 Basic Correctness
// ----------------------------------------------------------------------------

TEST(Vulkan_Tree, Stage7) {
  PREPARE_APPDATA;

  tree::vulkan::Singleton::getInstance().process_stage_1(appdata, tmp_storage);
  tree::vulkan::Singleton::getInstance().process_stage_2(appdata, tmp_storage);
  tree::vulkan::Singleton::getInstance().process_stage_3(appdata, tmp_storage);
  tree::vulkan::Singleton::getInstance().process_stage_4(appdata, tmp_storage);
  tree::vulkan::Singleton::getInstance().process_stage_5(appdata, tmp_storage);
  tree::vulkan::Singleton::getInstance().process_stage_6(appdata, tmp_storage);
  tree::vulkan::Singleton::getInstance().process_stage_7(appdata, tmp_storage);

  test_tree::verify_stage_7(appdata);
}

// ----------------------------------------------------------------------------
// Stage 1 Multi-threaded / Multi-iteration
// ----------------------------------------------------------------------------

TEST(Vulkan_Tree, Stage1_MultiIteration) {
  for (int i = 0; i < 10; ++i) {
    PREPARE_APPDATA;

    tree::vulkan::Singleton::getInstance().process_stage_1(appdata, tmp_storage);
    tree::vulkan::Singleton::getInstance().process_stage_2(appdata, tmp_storage);
    tree::vulkan::Singleton::getInstance().process_stage_3(appdata, tmp_storage);
    tree::vulkan::Singleton::getInstance().process_stage_4(appdata, tmp_storage);
    tree::vulkan::Singleton::getInstance().process_stage_5(appdata, tmp_storage);
    tree::vulkan::Singleton::getInstance().process_stage_6(appdata, tmp_storage);
    tree::vulkan::Singleton::getInstance().process_stage_7(appdata, tmp_storage);

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
