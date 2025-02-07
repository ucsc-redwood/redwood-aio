#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>

#include "app.hpp"
#include "tree/vulkan/tmp_storage.hpp"
#include "tree/vulkan/vk_dispatcher.hpp"

// ----------------------------------------------------------------------------
// Manual Test
// ----------------------------------------------------------------------------

static void manual_visualized_test(size_t n_input) {
  spdlog::set_level(spdlog::level::debug);

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();
  tree::AppData appdata(mr, n_input);
  tree::vulkan::TmpStorage tmp_storage(mr, n_input);

  constexpr auto n_iter = 1;
  for (auto i = 0u; i < n_iter; i++) {
    tree::vulkan::Singleton::getInstance().process_stage_1(appdata);

    if (i == n_iter - 1) {
      spdlog::debug("First 10 morton codes:");
      for (auto i = 0u; i < std::min(10u, appdata.get_n_input()); i++) {
        spdlog::debug("\tmorton[{}] = {}", i, appdata.u_morton_keys_s1[i]);
      }
    }

    tree::vulkan::Singleton::getInstance().process_stage_2(appdata);

    if (i == n_iter - 1) {
      spdlog::debug("First 10 sorted morton codes:");
      for (auto i = 0u; i < std::min(10u, appdata.get_n_input()); i++) {
        spdlog::debug(
            "\tmorton[{}] = {}", i, appdata.u_morton_keys_sorted_s2[i]);
      }
    }

    tree::vulkan::Singleton::getInstance().process_stage_3(appdata,
                                                           tmp_storage);

    if (i == n_iter - 1) {
      spdlog::debug("First 10 unique morton codes:");
      for (auto i = 0u; i < std::min(10u, appdata.get_n_unique()); i++) {
        spdlog::debug(
            "\tunique_morton[{}] = {}", i, appdata.u_morton_keys_unique_s3[i]);
      }
    }

    tree::vulkan::Singleton::getInstance().process_stage_4(appdata);

    if (i == n_iter - 1) {
      spdlog::debug("First 10 BRT node parents:");
      for (auto i = 0u; i < std::min(10u, appdata.get_n_brt_nodes()); i++) {
        spdlog::debug("\tbrt_parent[{}] = {}", i, appdata.u_brt_parents_s4[i]);
      }
    }

    tree::vulkan::Singleton::getInstance().process_stage_5(appdata);

    if (i == n_iter - 1) {
      spdlog::debug("First 10 edge counts:");
      for (auto i = 0u; i < std::min(10u, appdata.get_n_brt_nodes()); i++) {
        spdlog::debug("\tedge_count[{}] = {}", i, appdata.u_edge_count_s5[i]);
      }
    }

    tree::vulkan::Singleton::getInstance().process_stage_6(appdata);

    if (i == n_iter - 1) {
      spdlog::debug("First 10 edge offsets:");
      for (auto i = 0u; i < std::min(10u, appdata.get_n_brt_nodes()); i++) {
        spdlog::debug("\tedge_offset[{}] = {}", i, appdata.u_edge_offset_s6[i]);
      }
    }

    tree::vulkan::Singleton::getInstance().process_stage_7(appdata);

    if (i == n_iter - 1) {
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

// ----------------------------------------------------------------------------
// Main function for running tests
// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
  parse_args(argc, argv);

  manual_visualized_test(640 * 480);  // 306k points
  // manual_visualized_test(1440 * 900);  // 1.3M points

  spdlog::set_level(spdlog::level::off);

  // ::testing::InitGoogleTest(&argc, argv);
  // return RUN_ALL_TESTS();
  return 0;
}
