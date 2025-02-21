#pragma once

#include <queue>

#include "builtin-apps/cifar-dense/dense_appdata.hpp"

// ---------------------------------------------------------------------
// Task structure
// ---------------------------------------------------------------------

struct Task {
  cifar_dense::AppData* app_data;  // basically just a pointer
  bool done = false;

  [[nodiscard]] bool is_sentinel() const { return app_data == nullptr; }
};

[[nodiscard]] std::queue<Task> init_tasks(const size_t num_tasks);

void cleanup(std::queue<Task>& tasks);