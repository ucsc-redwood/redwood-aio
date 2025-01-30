#include <iostream>
#include <memory>

#include "spdlog/spdlog.h"
#include "tree/tree_appdata.hpp"
#include "tree/vulkan/vk_dispatcher.hpp"

int main() {
  spdlog::set_level(spdlog::level::trace);

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  auto app_data = std::make_unique<tree::AppData>(mr);
  ::vulkan::TmpStorage tmp_storage(mr, app_data->get_n_input());

  auto& vk = tree::vulkan::Singleton::getInstance();

  vk.process_stage_1(*app_data);
  vk.process_stage_2(*app_data);
  vk.process_stage_3(*app_data, tmp_storage);
  vk.process_stage_4(*app_data);
  vk.process_stage_5(*app_data);
  vk.process_stage_6(*app_data);
  vk.process_stage_7(*app_data);

  spdlog::info("Done");
  return 0;
}
