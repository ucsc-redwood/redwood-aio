#pragma once

#include <concurrentqueue.h>

#include <vector>

#include "builtin-apps/tree/tree_appdata.hpp"

// ---------------------------------------------------------------------
// Task structure (new)
// ---------------------------------------------------------------------

struct Task {
  tree::AppData *data;
  explicit Task(tree::AppData *data) : data(data) {}
};

[[nodiscard]] moodycamel::ConcurrentQueue<Task *> init_tasks(std::vector<tree::AppData> &data,
                                                             size_t initial_capacity = 32);