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
    vk.process_stage_2(*app_data);
    vk.process_stage_3(*app_data, tmp_storage);
    vk.process_stage_4(*app_data);
    vk.process_stage_5(*app_data);
    vk.process_stage_6(*app_data);
    vk.process_stage_7(*app_data);
  }
  return 0;
}