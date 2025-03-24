#pragma once

#include <queue>

#include "builtin-apps/tree/omp/func_sort.hpp"
#include "builtin-apps/tree/vulkan/dispatchers.hpp"

// ---------------------------------------------------------------------
// Task structure
// ---------------------------------------------------------------------
struct Task {
  tree::AppData* app_data = nullptr;
  tree::omp::TmpStorage* omp_tmp_storage = nullptr;
  tree::vulkan::TmpStorage* vulkan_tmp_storage = nullptr;
  bool done = false;

  [[nodiscard]] bool is_sentinel() const { return app_data == nullptr; }
};

[[nodiscard]] std::queue<Task> init_tasks(const size_t num_tasks);

void cleanup(std::queue<Task>& tasks);