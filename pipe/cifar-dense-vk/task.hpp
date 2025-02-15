#pragma once

#include "builtin-apps/cifar-dense/dense_appdata.hpp"

// ---------------------------------------------------------------------
// Task structure
// ---------------------------------------------------------------------

struct Task {
  cifar_dense::AppData* app_data;  // basically just a pointer
};

[[nodiscard]] std::vector<Task> init_tasks(const size_t num_tasks);

void cleanup(std::vector<Task>& tasks);