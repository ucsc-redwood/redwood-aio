#include <gtest/gtest.h>
#include <omp.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>

#include "tree/omp/tree_kernel.hpp"

std::string device_id;

// Main function for running tests
int main(int argc, char **argv) {
  CLI::App app{"default"};
  app.add_option("-d,--device", device_id, "Device ID")->required();
  app.allow_extras();

  spdlog::set_level(spdlog::level::debug);

  CLI11_PARSE(app, argc, argv);

  if (device_id.empty()) {
    spdlog::error("Device ID is required");
    return 1;
  }

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
        spdlog::debug("\tchild_node_mask[{}] = {:b}",
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

  //   ::testing::InitGoogleTest(&argc, argv);
  //   return RUN_ALL_TESTS();
}
