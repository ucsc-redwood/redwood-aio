#include <spdlog/spdlog.h>

#include "builtin-apps/app.hpp"
#include "builtin-apps/tree/safe_tree_appdata.hpp"
#include "builtin-apps/tree/vulkan/dispatchers.hpp"

int main(int argc, char** argv) {
  parse_args(argc, argv);

  tree::HostTreeManager::getInstance().initialize();

  auto mr = tree::vulkan::Singleton::getInstance().get_mr();

  tree::SafeAppData safe_appdata(mr);

  //   print the num_octree_nodes
  spdlog::info("num_octree_nodes = {}", safe_appdata.get_n_octree_nodes());

  safe_appdata.set_n_octree_nodes(100);

  spdlog::info("num_octree_nodes = {}", safe_appdata.get_n_octree_nodes());

  return 0;
}