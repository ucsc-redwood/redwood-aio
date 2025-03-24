#pragma once

#include <concurrentqueue.h>

#include <vector>

#include "builtin-apps/common/cuda/manager.cuh"
#include "builtin-apps/tree/tree_appdata.hpp"

// ---------------------------------------------------------------------
// Task structure (new)
// ---------------------------------------------------------------------

struct Task {
  tree::AppData *data;
  cuda::CudaManager *mgr;

  explicit Task(tree::AppData *data, cuda::CudaManager *mgr) : data(data), mgr(mgr) {}
};

[[nodiscard]] moodycamel::ConcurrentQueue<Task *> init_tasks(std::vector<tree::AppData> &data,
                                                             cuda::CudaManager *mgr,
                                                             size_t initial_capacity = 32);