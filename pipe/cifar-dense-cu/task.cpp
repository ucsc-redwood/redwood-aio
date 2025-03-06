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
    Task t{
        .app_data = new cifar_dense::AppData(&g_mr),
        .done = false,
    };

    tasks.push(std::move(t));
  }

  tasks.push(Task{
      .app_data = nullptr,
      .done = true,
  });

  return tasks;
}

void cleanup(std::queue<Task>& tasks) {
  spdlog::debug("cleanup, tasks.size() = {}", tasks.size());

  while (!tasks.empty()) {
    auto& task = tasks.front();
    if (task.is_sentinel()) {
      tasks.pop();
      continue;
    }

    delete task.app_data;
    task.app_data = nullptr;

    tasks.pop();
  }
}
