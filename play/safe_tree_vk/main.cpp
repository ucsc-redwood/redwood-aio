#include <spdlog/spdlog.h>

#include "builtin-apps/app.hpp"
#include "builtin-apps/tree/safe_tree_appdata.hpp"
#include "builtin-apps/tree/vulkan/dispatchers.hpp"

int main(int argc, char** argv) {
  parse_args(argc, argv);

  tree::HostTreeManager::getInstance().initialize();

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  tree::SafeAppData safe_appdata(mr);
  tree::vulkan::TmpStorage vulkan_tmp_storage(mr, safe_appdata.get_n_input());

  tree::vulkan::Singleton::getInstance().process_safe_stage_1(safe_appdata, vulkan_tmp_storage);

  return 0;
}