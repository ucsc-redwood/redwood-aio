#pragma once

#include "builtin-apps/cifar-sparse/sparse_appdata.hpp"

// ---------------------------------------------------------------------
// Task structure
// ---------------------------------------------------------------------

struct Task {
  cifar_sparse::AppData* app_data;  // basically just a pointer
  bool done = false;

  [[nodiscard]] bool is_sentinel() const { return app_data == nullptr; }
};

[[nodiscard]] std::vector<Task> init_tasks(const size_t num_tasks);

void cleanup(std::vector<Task>& tasks);