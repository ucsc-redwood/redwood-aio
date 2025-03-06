#include "task.hpp"

#include <spdlog/spdlog.h>

#include "builtin-apps/common/cuda/cu_mem_resource.cuh"

// ---------------------------------------------------------------------
// Global Variables
// ---------------------------------------------------------------------

cuda::CudaMemoryResource_PinnedHost g_mr;

// ---------------------------------------------------------------------
// Queue version
// ---------------------------------------------------------------------

[[nodiscard]] std::queue<Task> init_tasks_queue(const size_t num_tasks) {
  std::queue<Task> tasks;

  for (uint32_t i = 0; i < num_tasks; ++i) {
    Task task{
        .app_data = new tree::AppData(&g_mr),
        .omp_tmp_storage = new tree::omp::TmpStorage(),
        .done = false,
    };

    tasks.push(task);
  }

  // create a sentinel task
  tasks.push(Task{
      .app_data = nullptr,
      .omp_tmp_storage = nullptr,
      .done = true,
  });

  return tasks;
}

void cleanup(std::queue<Task>& tasks) {
  spdlog::debug("cleanup, tasks.size() = {}", tasks.size());

  while (!tasks.empty()) {
    auto& task = tasks.front();
    if (!task.is_sentinel()) {
      delete task.app_data;
      delete task.omp_tmp_storage;
    }
    tasks.pop();
  }
}