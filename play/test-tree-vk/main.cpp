#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>

#include "app.hpp"
#include "spdlog/common.h"
#include "tree/tree_appdata.hpp"
#include "tree/vulkan/vk_dispatcher.hpp"

int main(int argc, char** argv) {
  parse_args(argc, argv);

  constexpr auto n_iterations = 20;

  spdlog::set_level(spdlog::level::info);

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();
  // auto app_data = std::make_unique<tree::AppData>(mr);

  std::vector<std::unique_ptr<tree::AppData>> app_datas(n_iterations);
  for (int i = 0; i < n_iterations; i++) {
    app_datas[i] = std::make_unique<tree::AppData>(mr);
  }

  ::vulkan::TmpStorage tmp_storage(mr, app_datas[0]->get_n_input());

  auto& vk = tree::vulkan::Singleton::getInstance();

  for (int i = 0; i < n_iterations; i++) {
    spdlog::info(" ============================== Iteration {}", i);
    vk.process_stage_1(*app_datas[i]);

    // Peek first 10 morton keys
    spdlog::info("First 10 morton keys:");
    for (auto j = 0u; j < std::min(10u, app_datas[i]->get_n_input()); j++) {
      spdlog::info("\tu_morton_keys[{}] = {}",
                   j,
                   app_datas[i]->get_unsorted_morton_keys()[j]);
    }

    vk.process_stage_2(*app_datas[i]);

    // Peek first 10 sorted morton keys
    spdlog::info("First 10 sorted morton keys:");
    for (auto j = 0u; j < std::min(10u, app_datas[i]->get_n_input()); j++) {
      spdlog::info("\tu_sorted_morton_keys[{}] = {}",
                   j,
                   app_datas[i]->get_sorted_morton_keys()[j]);
    }

    vk.process_stage_3(*app_datas[i], tmp_storage);

    // Peek first 10 sorted unique morton keys
    spdlog::info("First 10 sorted unique morton keys:");
    for (auto j = 0u; j < std::min(10u, app_datas[i]->get_n_input()); j++) {
      spdlog::info("\tu_sorted_unique_morton_keys[{}] = {}",
                   j,
                   app_datas[i]->get_sorted_unique_morton_keys()[j]);
    }

    vk.process_stage_4(*app_datas[i]);

    // Peek first 10 radix tree nodes
    spdlog::info("First 10 radix tree nodes:");
    for (auto j = 0u; j < std::min(10u, app_datas[i]->get_n_input()); j++) {
      spdlog::info(
          "\tu_brt_parents[{}] = {}", j, app_datas[i]->u_brt_parents[j]);
    }

    vk.process_stage_5(*app_datas[i]);

    // Peek first 10 edge counts
    spdlog::info("First 10 edge counts:");
    for (auto j = 0u; j < std::min(10u, app_datas[i]->get_n_input()); j++) {
      spdlog::info("\tu_edge_count[{}] = {}", j, app_datas[i]->u_edge_count[j]);
    }

    vk.process_stage_6(*app_datas[i]);

    // Peek first 10 edge offsets
    spdlog::info("First 10 edge offsets:");
    for (auto j = 0u; j < std::min(10u, app_datas[i]->get_n_input()); j++) {
      spdlog::info(
          "\tu_edge_offset[{}] = {}", j, app_datas[i]->u_edge_offset[j]);
    }

    vk.process_stage_7(*app_datas[i]);
  }
  return 0;
}