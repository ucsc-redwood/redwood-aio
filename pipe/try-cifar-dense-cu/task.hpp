#pragma once

#include <memory>
#include <vector>

#include "builtin-apps/cifar-dense/dense_appdata.hpp"

// ---------------------------------------------------------------------
// Task structure
// ---------------------------------------------------------------------

struct Task {
  std::unique_ptr<cifar_dense::AppData> app_data;  // Back to unique_ptr
  bool done = false;

  [[nodiscard]] bool is_sentinel() const { return done; }
};

[[nodiscard]] std::vector<Task> init_tasks(const size_t num_tasks);

void cleanup(std::vector<Task>& tasks);