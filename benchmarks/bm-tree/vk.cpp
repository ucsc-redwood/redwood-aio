#include <benchmark/benchmark.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>

#include "app.hpp"
#include "tree/tree_appdata.hpp"
#include "tree/vulkan/vk_dispatcher.hpp"

int main(int argc, char** argv) {
  parse_args(argc, argv);
  spdlog::set_level(spdlog::level::trace);

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();
  tree::AppData app_data(mr);
  vulkan::TmpStorage tmp_storage(mr, app_data.get_n_input());

  tree::vulkan::Singleton::getInstance().process_stage_1(app_data);
  tree::vulkan::Singleton::getInstance().process_stage_2(app_data);
  tree::vulkan::Singleton::getInstance().process_stage_3(app_data, tmp_storage);
  tree::vulkan::Singleton::getInstance().process_stage_4(app_data);
  tree::vulkan::Singleton::getInstance().process_stage_5(app_data);
  tree::vulkan::Singleton::getInstance().process_stage_6(app_data);
  tree::vulkan::Singleton::getInstance().process_stage_7(app_data);

  return 0;
}