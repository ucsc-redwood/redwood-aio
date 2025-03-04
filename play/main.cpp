#include "builtin-apps/tree/safe_tree_appdata.hpp"

int main() {
  tree::HostTreeManager::getInstance().initialize();
  return 0;
}