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
[[nodiscard]] std::queue<Task> init_tasks(const size_t num_tasks) {
  auto mr = cuda::CudaMemoryResource();
  std::queue<Task> tasks;

  for (uint32_t i = 0; i < num_tasks; ++i) {
    Task task{
        .app_data = std::make_unique<cifar_dense::AppData>(&mr),
        .done = false,
    };

    tasks.push(std::move(task));
  }

  // create a sentinel task
  tasks.push(Task{
      .app_data = nullptr,
      .done = true,
  });

  return tasks;
}

void cleanup(std::queue<Task>& tasks) {
  spdlog::trace("cleanup, tasks.size() = {}", tasks.size());

  while (!tasks.empty()) {
    // // Create a copy of the pointer values before popping
    const void* task_ptr = static_cast<const void*>(&tasks.front());
    const void* app_data_ptr = static_cast<const void*>(tasks.front().app_data.get());
    
    // // Log before destroying the task
    spdlog::trace("cleaning up task, task = {}, task.app_data = {}", 
                  task_ptr, app_data_ptr);
    

    tasks.pop();
  }
}
