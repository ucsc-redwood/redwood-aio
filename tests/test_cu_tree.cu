#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>

#include "common/cuda/cu_mem_resource.cuh"
#include "common/cuda/helpers.cuh"
#include "tree/cuda/kernel.cuh"

std::string device_id;

// ----------------------------------------------------------------------------
// Manual Test
// ----------------------------------------------------------------------------

static void manual_visualized_test(size_t n_input) {
  spdlog::set_level(spdlog::level::debug);

  auto mr = cuda::CudaMemoryResource();
  tree::AppData appdata(&mr, n_input);

  constexpr auto n_iter = 1;
  for (auto i = 0u; i < n_iter; i++) {
    spdlog::info("Iteration {}...", i);

    tree::cuda::process_stage_1(appdata);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (i == 0) {
      spdlog::debug("First 10 morton codes:");
      for (auto i = 0u; i < std::min(10u, appdata.get_n_input()); i++) {
        spdlog::debug("\tmorton[{}] = {}", i, appdata.u_morton_keys_s1[i]);
      }
    }

    tree::cuda::process_stage_2(appdata);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (i == 0) {
      spdlog::debug("First 10 sorted morton codes:");
      for (auto i = 0u; i < std::min(10u, appdata.get_n_input()); i++) {
        spdlog::debug(
            "\tsorted_morton[{}] = {}", i, appdata.u_morton_keys_sorted_s2[i]);
      }
    }

    tree::cuda::process_stage_3(appdata);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (i == 0) {
      spdlog::debug("First 10 unique morton codes:");
      for (auto i = 0u; i < std::min(10u, appdata.get_n_unique()); i++) {
        spdlog::debug(
            "\tunique_morton[{}] = {}", i, appdata.u_morton_keys_unique_s3[i]);
      }
    }

    tree::cuda::process_stage_4(appdata);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (i == 0) {
      spdlog::debug("First 10 BRT node parents:");
      for (auto i = 0u; i < std::min(10u, appdata.get_n_brt_nodes()); i++) {
        spdlog::debug("\tbrt_parent[{}] = {}", i, appdata.u_brt_parents_s4[i]);
      }
    }

    tree::cuda::process_stage_5(appdata);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (i == 0) {
      spdlog::debug("First 10 edge counts:");
      for (auto i = 0u; i < std::min(10u, appdata.get_n_brt_nodes()); i++) {
        spdlog::debug("\tedge_count[{}] = {}", i, appdata.u_edge_count_s5[i]);
      }
    }

    tree::cuda::process_stage_6(appdata);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (i == 0) {
      spdlog::debug("First 10 edge offsets:");
      for (auto i = 0u; i < std::min(10u, appdata.get_n_brt_nodes()); i++) {
        spdlog::debug("\tedge_offset[{}] = {}", i, appdata.u_edge_offset_s6[i]);
      }
    }

    tree::cuda::process_stage_7(appdata);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (i == 0) {
      spdlog::debug("First 10 octree nodes:");
      for (auto i = 0u; i < std::min(10u, appdata.get_n_octree_nodes()); i++) {
        spdlog::debug("\tchild_node_mask[{}] = 0b{:8b}",
                      i,
                      appdata.u_oct_child_node_mask_s7[i]);
      }
    }
  }

  spdlog::info("n_input: {}", appdata.get_n_input());
  spdlog::info("n_unique: {} ({:.2f}%)",
               appdata.get_n_unique(),
               100.0f * appdata.get_n_unique() / appdata.get_n_input());
  spdlog::info("n_brt_nodes: {} ({:.2f}%)",
               appdata.get_n_brt_nodes(),
               100.0f * appdata.get_n_brt_nodes() / appdata.get_n_input());
  spdlog::info("n_octree_nodes: {} ({:.2f}%)",
               appdata.get_n_octree_nodes(),
               100.0f * appdata.get_n_octree_nodes() / appdata.get_n_input());
}

// ----------------------------------------------------------------------------
// Google Test
// ----------------------------------------------------------------------------

class CudaTreeTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    auto mr = cuda::CudaMemoryResource();
    appdata = std::make_unique<tree::AppData>(&mr);

    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void TearDown() override { appdata.reset(); }

 protected:
  std::unique_ptr<tree::AppData> appdata;
};

TEST_F(CudaTreeTestFixture, Stage1_MortonCodeGeneration) {
  tree::cuda::process_stage_1(*appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Verify that morton codes are generated and non-zero
  ASSERT_GT(appdata->get_n_input(), 0);
  for (size_t i = 0; i < appdata->get_n_input(); ++i) {
    EXPECT_GT(appdata->u_morton_keys_s1[i], 0);
  }
}

TEST_F(CudaTreeTestFixture, Stage2_MortonCodeSorting) {
  tree::cuda::process_stage_2(*appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Verify that morton codes are sorted
  for (size_t i = 1; i < appdata->get_n_input(); ++i) {
    EXPECT_LE(appdata->u_morton_keys_sorted_s2[i - 1],
              appdata->u_morton_keys_sorted_s2[i]);
  }
}

TEST_F(CudaTreeTestFixture, Stage3_UniqueMortonCodes) {
  tree::cuda::process_stage_3(*appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Verify unique morton codes
  ASSERT_GT(appdata->get_n_unique(), 0);
  EXPECT_LE(appdata->get_n_unique(), appdata->get_n_input());
  for (size_t i = 1; i < appdata->get_n_unique(); ++i) {
    EXPECT_LT(appdata->u_morton_keys_unique_s3[i - 1],
              appdata->u_morton_keys_unique_s3[i]);
  }
}

TEST_F(CudaTreeTestFixture, Stage4_BRTParents) {
  tree::cuda::process_stage_4(*appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Verify BRT parents are non-zero
  ASSERT_GT(appdata->get_n_brt_nodes(), 0);

  ASSERT_EQ(appdata->u_brt_parents_s4[0], 0);
  for (size_t i = 1; i < appdata->get_n_brt_nodes(); ++i) {
    EXPECT_GE(appdata->u_brt_parents_s4[i], 0);
  }
}

TEST_F(CudaTreeTestFixture, Stage5_EdgeCounts) {
  tree::cuda::process_stage_5(*appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Verify not all are zeros
  bool all_zeros = std::ranges::all_of(appdata->u_edge_count_s5,
                                       [](int x) { return x == 0; });
  EXPECT_FALSE(all_zeros);
}

TEST_F(CudaTreeTestFixture, Stage6_EdgeOffsets) {
  tree::cuda::process_stage_6(*appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Verify not all are zeros
  bool all_zeros = std::ranges::all_of(appdata->u_edge_offset_s6,
                                       [](int x) { return x == 0; });
  EXPECT_FALSE(all_zeros);

  // Verify edge offsets are monotonically increasing
  for (size_t i = 1; i < appdata->get_n_brt_nodes(); ++i) {
    EXPECT_LE(appdata->u_edge_offset_s6[i - 1], appdata->u_edge_offset_s6[i]);
  }
}

TEST_F(CudaTreeTestFixture, Stage7_OctreeNodes) {
  tree::cuda::process_stage_7(*appdata);
  CUDA_CHECK(cudaDeviceSynchronize());

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

  manual_visualized_test(640 * 480);   // 306k points
  manual_visualized_test(1440 * 900);  // 1.3M points

  spdlog::set_level(spdlog::level::off);

  // ::testing::InitGoogleTest(&argc, argv);
  // return RUN_ALL_TESTS();

  return 0;
}
