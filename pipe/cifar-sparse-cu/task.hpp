#pragma once

#include <queue>

#include "builtin-apps/cifar-sparse/sparse_appdata.hpp"

// ---------------------------------------------------------------------
// Task structure
// ---------------------------------------------------------------------

struct Task {
  cifar_sparse::AppData* app_data;  // Just a pointer
  bool done = false;

  [[nodiscard]] bool is_sentinel() const { return done; }
};

[[nodiscard]] std::queue<Task> init_tasks_queue(const size_t num_tasks);
void cleanup(std::queue<Task>& tasks);
