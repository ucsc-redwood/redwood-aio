#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>

#include "app.hpp"
#include "spdlog/common.h"
#include "tree/tree_appdata.hpp"
#include "tree/vulkan/vk_dispatcher.hpp"

int main(int argc, char** argv) {
  parse_args(argc, argv);

  spdlog::set_level(spdlog::level::info);

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();
  auto app_data = std::make_unique<tree::AppData>(mr);
  ::vulkan::TmpStorage tmp_storage(mr, app_data->get_n_input());

  auto& vk = tree::vulkan::Singleton::getInstance();

  for (int i = 0; i < 20; i++) {
    spdlog::info(" ============================== Iteration {}", i);
    vk.process_stage_1(*app_data);

    // Peek first 10 morton keys
    spdlog::info("First 10 morton keys:");
    for (auto j = 0u; j < std::min(10u, app_data->get_n_input()); j++) {
      spdlog::info("\tu_morton_keys[{}] = {}",
                   j,
                   app_data->get_unsorted_morton_keys()[j]);
    }

    vk.process_stage_2(*app_data);

    // Peek first 10 sorted morton keys
    spdlog::info("First 10 sorted morton keys:");
    for (auto j = 0u; j < std::min(10u, app_data->get_n_input()); j++) {
      spdlog::info("\tu_sorted_morton_keys[{}] = {}",
                   j,
                   app_data->get_sorted_morton_keys()[j]);
    }

    vk.process_stage_3(*app_data, tmp_storage);

    // Peek first 10 sorted unique morton keys
    spdlog::info("First 10 sorted unique morton keys:");
    for (auto j = 0u; j < std::min(10u, app_data->get_n_input()); j++) {
      spdlog::info("\tu_sorted_unique_morton_keys[{}] = {}",
                   j,
                   app_data->get_sorted_unique_morton_keys()[j]);
    }

    vk.process_stage_4(*app_data);

    // Peek first 10 radix tree nodes
    spdlog::info("First 10 radix tree nodes:");
    for (auto j = 0u; j < std::min(10u, app_data->get_n_input()); j++) {
      spdlog::info("\tu_brt_parents[{}] = {}", j, app_data->u_brt_parents[j]);
    }

    vk.process_stage_5(*app_data);

    // Peek first 10 edge counts
    spdlog::info("First 10 edge counts:");
    for (auto j = 0u; j < std::min(10u, app_data->get_n_input()); j++) {
      spdlog::info("\tu_edge_count[{}] = {}", j, app_data->u_edge_count[j]);
    }

    vk.process_stage_6(*app_data);

    // Peek first 10 edge offsets
    spdlog::info("First 10 edge offsets:");
    for (auto j = 0u; j < std::min(10u, app_data->get_n_input()); j++) {
      spdlog::info("\tu_edge_offset[{}] = {}", j, app_data->u_edge_offset[j]);
    }

    vk.process_stage_7(*app_data);
  }
  return 0;
}