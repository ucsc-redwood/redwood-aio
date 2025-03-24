#pragma once

#include <concurrentqueue.h>

#include <vector>

#include "builtin-apps/tree/tree_appdata.hpp"
#include "builtin-apps/tree/vulkan/tmp_storage.hpp"

// ---------------------------------------------------------------------
// Task structure (new)
// ---------------------------------------------------------------------

struct Task {
  tree::AppData *data;
  tree::vulkan::TmpStorage *vulkan_tmp_storage = nullptr;

  explicit Task(tree::AppData *data, tree::vulkan::TmpStorage *vulkan_tmp_storage)
      : data(data), vulkan_tmp_storage(vulkan_tmp_storage) {}
};

[[nodiscard]] moodycamel::ConcurrentQueue<Task *> init_tasks(
    std::vector<tree::AppData> &data,
    std::vector<tree::vulkan::TmpStorage> &vulkan_tmp_storages,
    size_t initial_capacity = 32);