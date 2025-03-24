#pragma once

#include <concurrentqueue.h>

#include <vector>

#include "builtin-apps/tree/vulkan/vk_appdata.hpp"

// ---------------------------------------------------------------------
// Task structure (new)
// ---------------------------------------------------------------------

struct Task {
  tree::vulkan::VkAppData *data;

  explicit Task(tree::vulkan::VkAppData *data) : data(data) {}
};

[[nodiscard]] moodycamel::ConcurrentQueue<Task *> init_tasks(
    std::vector<tree::vulkan::VkAppData> &data, size_t initial_capacity = 32);