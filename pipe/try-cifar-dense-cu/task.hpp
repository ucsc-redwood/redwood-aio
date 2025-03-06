#pragma once

#include <memory>
#include <queue>
#include <vector>

#include "builtin-apps/cifar-dense/dense_appdata.hpp"

// ---------------------------------------------------------------------
// Task structure
// ---------------------------------------------------------------------

struct Task {
  // std::unique_ptr<cifar_dense::AppData> app_data;  // Back to unique_ptr
  cifar_dense::AppData* app_data;  // Back to unique_ptr
  bool done = false;

  [[nodiscard]] bool is_sentinel() const { return done; }
};

[[deprecated("Use init_tasks_queue instead")]] [[nodiscard]] std::vector<Task> init_tasks(
    const size_t num_tasks);

[[nodiscard]] std::queue<Task> init_tasks_queue(const size_t num_tasks);
void cleanup(std::queue<Task>& tasks);
