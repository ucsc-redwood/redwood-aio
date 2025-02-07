#include <gtest/gtest.h>
#include <omp.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>

#include "tree/omp/tree_kernel.hpp"

std::string device_id;

// ----------------------------------------------------------------------------
// Manual Test
// ----------------------------------------------------------------------------

static void manual_visualized_test() {
  spdlog::set_level(spdlog::level::debug);

  const auto n_threads = omp_get_max_threads();

  auto mr = std::pmr::new_delete_resource();
  tree::AppData app_data(mr, 640 * 480);
  tree::omp::v2::TempStorage temp_storage(n_threads, n_threads);

  spdlog::info("Number of OpenMP threads: {}", n_threads);

#pragma omp parallel
  {
    tree::omp::process_stage_1(app_data);

#pragma omp single
    {
      spdlog::debug("First 10 morton codes:");
      for (auto i = 0u; i < std::min(10u, app_data.get_n_input()); i++) {
        spdlog::debug("\tmorton[{}] = {}", i, app_data.u_morton_keys_s1[i]);
      }
    }

    tree::omp::v2::process_stage_2(app_data, temp_storage);

#pragma omp single
    {
      spdlog::debug("First 10 sorted morton codes:");
      for (auto i = 0u; i < std::min(10u, app_data.get_n_input()); i++) {
        spdlog::debug(
            "\tsorted_morton[{}] = {}", i, app_data.u_morton_keys_sorted_s2[i]);
      }
    }

    tree::omp::process_stage_3(app_data);

#pragma omp single
    {
      spdlog::debug("First 10 unique morton codes:");
      for (auto i = 0u; i < std::min(10u, app_data.get_n_unique()); i++) {
        spdlog::debug(
            "\tunique_morton[{}] = {}", i, app_data.u_morton_keys_unique_s3[i]);
      }
    }

    tree::omp::process_stage_4(app_data);

#pragma omp single
    {
      spdlog::debug("First 10 BRT node parents:");
      for (auto i = 0u; i < std::min(10u, app_data.get_n_brt_nodes()); i++) {
        spdlog::debug("\tbrt_parent[{}] = {}", i, app_data.u_brt_parents_s4[i]);
      }
    }

    tree::omp::process_stage_5(app_data);

#pragma omp single
    {
      spdlog::debug("First 10 edge counts:");
      for (auto i = 0u; i < std::min(10u, app_data.get_n_brt_nodes()); i++) {
        spdlog::debug("\tedge_count[{}] = {}", i, app_data.u_edge_count_s5[i]);
      }
    }

    tree::omp::process_stage_6(app_data);

#pragma omp single
    {
      spdlog::debug("First 10 edge offsets:");
      for (auto i = 0u; i < std::min(10u, app_data.get_n_brt_nodes()); i++) {
        spdlog::debug(
            "\tedge_offset[{}] = {}", i, app_data.u_edge_offset_s6[i]);
      }
    }

    tree::omp::process_stage_7(app_data);

#pragma omp single
    {
      spdlog::debug("First 10 octree nodes:");
      for (auto i = 0u; i < std::min(10u, app_data.get_n_octree_nodes()); i++) {
        spdlog::debug("\tchild_node_mask[{}] = 0b{:8b}",
                      i,
                      app_data.u_oct_child_node_mask_s7[i]);
      }
    }
  }

  spdlog::info("n_input: {}", app_data.get_n_input());
  spdlog::info("n_unique: {} ({:.2f}%)",
               app_data.get_n_unique(),
               100.0f * app_data.get_n_unique() / app_data.get_n_input());
  spdlog::info("n_brt_nodes: {} ({:.2f}%)",
               app_data.get_n_brt_nodes(),
               100.0f * app_data.get_n_brt_nodes() / app_data.get_n_input());
  spdlog::info("n_octree_nodes: {} ({:.2f}%)",
               app_data.get_n_octree_nodes(),
               100.0f * app_data.get_n_octree_nodes() / app_data.get_n_input());
}

// ----------------------------------------------------------------------------
// Google Test
// ----------------------------------------------------------------------------

class OmpTreeTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    auto mr = std::pmr::new_delete_resource();
    // Create a small test dataset
    appdata = std::make_unique<tree::AppData>(mr, 100);
    n_threads = omp_get_max_threads();
    temp_storage =
        std::make_unique<tree::omp::v2::TempStorage>(n_threads, n_threads);
  }

  void TearDown() override {
    appdata.reset();
    temp_storage.reset();
  }

 protected:
  std::unique_ptr<tree::AppData> appdata;
  std::unique_ptr<tree::omp::v2::TempStorage> temp_storage;
  int n_threads;
};

TEST_F(OmpTreeTestFixture, Stage1_MortonCodeGeneration) {
#pragma omp parallel
  {
    tree::omp::process_stage_1(*appdata);
  }

  // Verify that morton codes are generated and non-zero
  ASSERT_GT(appdata->get_n_input(), 0);
  for (size_t i = 0; i < appdata->get_n_input(); ++i) {
    EXPECT_GT(appdata->u_morton_keys_s1[i], 0);
  }
}

TEST_F(OmpTreeTestFixture, Stage2_MortonCodeSorting) {
#pragma omp parallel
  {
    tree::omp::process_stage_1(*appdata);
    tree::omp::v2::process_stage_2(*appdata, *temp_storage);
  }

  // Verify that morton codes are sorted
  for (size_t i = 1; i < appdata->get_n_input(); ++i) {
    EXPECT_LE(appdata->u_morton_keys_sorted_s2[i - 1],
              appdata->u_morton_keys_sorted_s2[i]);
  }
}

TEST_F(OmpTreeTestFixture, Stage3_UniqueMortonCodes) {
#pragma omp parallel
  {
    tree::omp::process_stage_1(*appdata);
    tree::omp::v2::process_stage_2(*appdata, *temp_storage);
    tree::omp::process_stage_3(*appdata);
  }

  // Verify unique morton codes
  ASSERT_GT(appdata->get_n_unique(), 0);
  EXPECT_LE(appdata->get_n_unique(), appdata->get_n_input());
  for (size_t i = 1; i < appdata->get_n_unique(); ++i) {
    EXPECT_LT(appdata->u_morton_keys_unique_s3[i - 1],
              appdata->u_morton_keys_unique_s3[i]);
  }
}

TEST_F(OmpTreeTestFixture, Stage4_BRTParents) {
#pragma omp parallel
  {
    tree::omp::process_stage_1(*appdata);
    tree::omp::v2::process_stage_2(*appdata, *temp_storage);
    tree::omp::process_stage_3(*appdata);
    tree::omp::process_stage_4(*appdata);
  }

  // Verify BRT parents are non-zero
  ASSERT_GT(appdata->get_n_brt_nodes(), 0);

  ASSERT_EQ(appdata->u_brt_parents_s4[0], 0);
  for (size_t i = 1; i < appdata->get_n_brt_nodes(); ++i) {
    EXPECT_GE(appdata->u_brt_parents_s4[i], 0);
  }
}

TEST_F(OmpTreeTestFixture, Stage5_EdgeCounts) {
#pragma omp parallel
  {
    tree::omp::process_stage_1(*appdata);
    tree::omp::v2::process_stage_2(*appdata, *temp_storage);
    tree::omp::process_stage_3(*appdata);
    tree::omp::process_stage_4(*appdata);
    tree::omp::process_stage_5(*appdata);
  }

  // Verify not all are zeros
  bool all_zeros = std::ranges::all_of(appdata->u_edge_count_s5,
                                       [](int x) { return x == 0; });
  EXPECT_FALSE(all_zeros);
}

TEST_F(OmpTreeTestFixture, Stage6_EdgeOffsets) {
#pragma omp parallel
  {
    tree::omp::process_stage_1(*appdata);
    tree::omp::v2::process_stage_2(*appdata, *temp_storage);
    tree::omp::process_stage_3(*appdata);
    tree::omp::process_stage_4(*appdata);
    tree::omp::process_stage_5(*appdata);
    tree::omp::process_stage_6(*appdata);
  }

  // Verify not all are zeros
  bool all_zeros = std::ranges::all_of(appdata->u_edge_offset_s6,
                                       [](int x) { return x == 0; });
  EXPECT_FALSE(all_zeros);

  // Verify edge offsets are monotonically increasing
  for (size_t i = 1; i < appdata->get_n_brt_nodes(); ++i) {
    EXPECT_LE(appdata->u_edge_offset_s6[i - 1], appdata->u_edge_offset_s6[i]);
  }
}

TEST_F(OmpTreeTestFixture, Stage7_OctreeNodes) {
#pragma omp parallel
  {
    tree::omp::process_stage_1(*appdata);
    tree::omp::v2::process_stage_2(*appdata, *temp_storage);
    tree::omp::process_stage_3(*appdata);
    tree::omp::process_stage_4(*appdata);
    tree::omp::process_stage_5(*appdata);
    tree::omp::process_stage_6(*appdata);
    tree::omp::process_stage_7(*appdata);
  }

  // Verify not all are zeros
  bool all_zeros = std::ranges::all_of(appdata->u_oct_child_node_mask_s7,
                                       [](int x) { return x == 0; });
  EXPECT_FALSE(all_zeros);

  // Verify octree nodes
  ASSERT_GT(appdata->get_n_octree_nodes(), 0);
  for (size_t i = 0; i < appdata->get_n_octree_nodes(); ++i) {
    EXPECT_GE(appdata->u_oct_child_node_mask_s7[i], 0);
    EXPECT_LE(appdata->u_oct_child_node_mask_s7[i], 255);  // 8-bit mask
  }
}

// ----------------------------------------------------------------------------
// Main function for running tests
// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
  CLI::App app{"default"};
  app.add_option("-d,--device", device_id, "Device ID")->required();
  app.allow_extras();

  CLI11_PARSE(app, argc, argv);

  if (device_id.empty()) {
    spdlog::error("Device ID is required");
    return 1;
  }

  manual_visualized_test();

  spdlog::set_level(spdlog::level::off);

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
