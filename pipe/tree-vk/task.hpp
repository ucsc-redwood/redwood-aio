#pragma once

#include "builtin-apps/tree/omp/func_sort.hpp"
#include "builtin-apps/tree/tree_appdata.hpp"
#include "builtin-apps/tree/vulkan/tmp_storage.hpp"

// ---------------------------------------------------------------------
// Task structure
// ---------------------------------------------------------------------

struct Task {
  tree::AppData* app_data;  // basically just a pointer
  tree::omp::TmpStorage* temp_storage;
  tree::vulkan::TmpStorage* vulkan_temp_storage;
};

[[nodiscard]] std::vector<Task> init_tasks(const size_t num_tasks);

void cleanup(std::vector<Task>& tasks);