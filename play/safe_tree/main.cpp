#include <spdlog/spdlog.h>

#include "builtin-apps/tree/safe_tree_appdata.hpp"

int main() {
  tree::HostTreeManager::getInstance().initialize();

  auto mr = std::pmr::new_delete_resource();

  tree::SafeAppData safe_appdata(mr);

  //   print the num_octree_nodes
  spdlog::info("num_octree_nodes = {}", safe_appdata.get_n_octree_nodes());

  safe_appdata.set_n_octree_nodes(100);

  spdlog::info("num_octree_nodes = {}", safe_appdata.get_n_octree_nodes());

  return 0;
}