#pragma once

#include <concurrentqueue.h>

#include <vector>

#include "builtin-apps/cifar-dense/dense_appdata.hpp"
#include "builtin-apps/common/cuda/manager.cuh"

// ---------------------------------------------------------------------
// Task structure
// ---------------------------------------------------------------------

// struct Task {
//   cifar_dense::AppData* app_data;
//   bool done = false;

//   [[nodiscard]] bool is_sentinel() const { return done; }
// };

// [[deprecated("Use init_tasks_queue instead")]] [[nodiscard]] std::vector<Task> init_tasks(
//     const size_t num_tasks);

// [[nodiscard]] std::queue<Task> init_tasks_queue(const size_t num_tasks);
// void cleanup(std::queue<Task>& tasks);

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