#include <spdlog/spdlog.h>

#include "builtin-apps/app.hpp"
#include "builtin-apps/tree/safe_tree_appdata.hpp"
#include "builtin-apps/tree/vulkan/dispatchers.hpp"

int main(int argc, char** argv) {
  parse_args(argc, argv);
  spdlog::set_level(spdlog::level::from_str(g_spdlog_log_level));

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  tree::SafeAppData safe_appdata(mr);
  tree::vulkan::TmpStorage vulkan_tmp_storage(mr, safe_appdata.get_n_input());

  tree::vulkan::Singleton::getInstance().process_stage_1(safe_appdata, vulkan_tmp_storage);
  tree::vulkan::Singleton::getInstance().process_stage_2(safe_appdata, vulkan_tmp_storage);
  tree::vulkan::Singleton::getInstance().process_stage_3(safe_appdata, vulkan_tmp_storage);
  tree::vulkan::Singleton::getInstance().process_stage_4(safe_appdata, vulkan_tmp_storage);
  tree::vulkan::Singleton::getInstance().process_stage_5(safe_appdata, vulkan_tmp_storage);
  tree::vulkan::Singleton::getInstance().process_stage_6(safe_appdata, vulkan_tmp_storage);
  tree::vulkan::Singleton::getInstance().process_stage_7(safe_appdata, vulkan_tmp_storage);

  return 0;
}