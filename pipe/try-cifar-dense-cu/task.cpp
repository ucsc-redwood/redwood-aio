#include "task.hpp"

#include <spdlog/spdlog.h>

#include "builtin-apps/common/cuda/cu_mem_resource.cuh"

// ---------------------------------------------------------------------
// Task structure
// ---------------------------------------------------------------------

/**
 * Initializes a queue of tasks and adds a sentinel task at the end.
 * The sentinel task (with null pointers) is used to signal the end of
 * the task stream to the pipeline stages.
 */
[[nodiscard]] std::vector<Task> init_tasks(const size_t num_tasks) {
  auto mr = cuda::CudaMemoryResource();
  std::vector<Task> tasks;
  tasks.reserve(num_tasks + 1);  // Reserve space including sentinel

  for (uint32_t i = 0; i < num_tasks; ++i) {
    tasks.push_back(Task{
        .app_data = std::make_unique<cifar_dense::AppData>(&mr),
        .done = false,
    });
  }

  // Add sentinel task
  tasks.push_back(Task{
      .app_data = nullptr,
      .done = true,
  });

  return tasks;
}

void cleanup(std::vector<Task>& tasks) {
  spdlog::trace("cleanup, tasks.size() = {}", tasks.size());

  // for (auto& task : tasks) {
  //   const void* task_ptr = static_cast<const void*>(&task);
  //   const void* app_data_ptr = static_cast<const void*>(task.app_data.get());

  //   spdlog::trace("cleaning up task, task = {}, task.app_data = {}", task_ptr, app_data_ptr);

  //   // if (!task.is_sentinel()) {
  //   //   task.app_data.reset();
  //   // }
  // }

  tasks.clear();
}
