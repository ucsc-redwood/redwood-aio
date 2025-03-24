#pragma once

#include <concurrentqueue.h>

#include <vector>

#include "builtin-apps/cifar-dense/dense_appdata.hpp"
#include "builtin-apps/common/cuda/manager.cuh"

// ---------------------------------------------------------------------
// Task structure (new)
// ---------------------------------------------------------------------

struct Task {
  cifar_dense::AppData *data;
  cuda::CudaManager *mgr;

  explicit Task(cifar_dense::AppData *data, cuda::CudaManager *mgr) : data(data), mgr(mgr) {}
};

[[nodiscard]] moodycamel::ConcurrentQueue<Task *> init_tasks(
    std::vector<cifar_dense::AppData> &data, cuda::CudaManager *mgr, size_t initial_capacity = 32);