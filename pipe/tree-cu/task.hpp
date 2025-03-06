#pragma once

#include <queue>

#include "builtin-apps/tree/cuda/temp_storage.cuh"
#include "builtin-apps/tree/omp/func_sort.hpp"
#include "builtin-apps/tree/tree_appdata.hpp"

// ---------------------------------------------------------------------
// Task structure
// ---------------------------------------------------------------------

struct Task {
  tree::AppData* app_data;  // basically just a pointer
  tree::omp::TmpStorage* omp_tmp_storage = nullptr;
  tree::cuda::TempStorage* cuda_tmp_storage = nullptr;

  bool done = false;

  [[nodiscard]] bool is_sentinel() const { return app_data == nullptr; }
};

[[nodiscard]] std::queue<Task> init_tasks_queue(const size_t num_tasks);
void cleanup(std::queue<Task>& tasks);
