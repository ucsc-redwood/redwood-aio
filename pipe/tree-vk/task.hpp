#pragma once

#include "builtin-apps/tree/tree_appdata.hpp"

// ---------------------------------------------------------------------
// Task structure
// ---------------------------------------------------------------------

struct Task {
  tree::AppData* app_data;  // basically just a pointer
};

[[nodiscard]] std::vector<Task> init_tasks(const size_t num_tasks);

void cleanup(std::vector<Task>& tasks);